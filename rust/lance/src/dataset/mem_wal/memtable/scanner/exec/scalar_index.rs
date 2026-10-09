// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ScalarIndexExec`: a filter answered from a memtable's indexes, with MVCC
//! visibility.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow::compute::{and, filter_record_batch, prep_null_mask_filter};
use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::Result as DataFusionResult;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExprRef};
use futures::stream::{self, StreamExt};
use lance_core::Result;

use lance_index::scalar::expression::ScalarIndexExpr;

use crate::dataset::mem_wal::index::{SearchContext, evaluate_index_filter};
use crate::dataset::mem_wal::memtable::scanner::exec::{scan_record_batch, take_projected_columns};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// An index may decline a search matching more than `1 / MATCH_BUDGET_SHARE`
/// of the visible rows, which are then all read, since listing that many costs
/// more than reading them.
const MATCH_BUDGET_SHARE: u64 = 16;

/// Matches always worth listing, whatever the share.
const MIN_MATCH_BUDGET: u64 = 4096;

/// The share for a read that keeps only each key's newest version, where each
/// match also costs a seek in the primary-key index.
const NEWEST_ONLY_MATCH_BUDGET_SHARE: u64 = 8;

/// Metric counting the matches checked for being their key's newest version.
const NEWEST_CHECKS_METRIC: &str = "newest_checks";

/// Execution-plan node answering a filter from the memtable's indexes,
/// filtered by visibility.
pub struct ScalarIndexExec {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    /// The index searches the filter was split into: a tree of `AND`/`OR` over
    /// per-index queries, built by the same pass the base table's scan uses.
    index_expr: ScalarIndexExpr,
    /// The whole filter, compiled, applied to the rows the indexes narrowed to;
    /// `None` when they answered exactly.
    recheck: Option<PhysicalExprRef>,
    /// Whether the index tree is the whole filter.
    is_filter_covered: bool,
    readable_count: usize,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    /// Whether to include _rowid column (row position) in output.
    with_row_id: bool,
    /// Whether to include _rowaddr column (same as row position) in output.
    with_row_address: bool,
    /// Keep only the newest visible version of each primary key, given the key
    /// columns' positions in the stored batches. An older version can match a
    /// filter its key's newest version fails, and must not be returned.
    newest_of: Option<Vec<usize>>,
    /// Run instead when the indexes match too many rows to be worth listing.
    /// Required with `newest_of`.
    broad_fallback: Option<Arc<dyn ExecutionPlan>>,
}

impl Debug for ScalarIndexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarIndexExec")
            .field("index_expr", &self.index_expr)
            .field("rechecked", &self.recheck.is_some())
            .field("readable_count", &self.readable_count)
            .field("with_row_id", &self.with_row_id)
            .field("with_row_address", &self.with_row_address)
            .finish()
    }
}

impl ScalarIndexExec {
    /// Create a new ScalarIndexExec.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - Lock-free batch store containing data
    /// * `indexes` - The memtable's indexes, which answer `index_expr`
    /// * `index_expr` - The index searches the filter was split into
    /// * `recheck` - The whole filter, compiled, for rows an index only narrowed
    /// * `is_filter_covered` - Whether `index_expr` is the whole filter
    /// * `readable_count` - Exclusive count of batch positions this scan may read
    /// * `projection` - Optional column indices to project
    /// * `output_schema` - Schema after projection (should include _rowid/_rowaddr if requested)
    /// * `with_row_id` - Whether to include _rowid column (row position)
    /// * `with_row_address` - Whether to include _rowaddr column (same as row position)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        index_expr: ScalarIndexExpr,
        recheck: Option<PhysicalExprRef>,
        is_filter_covered: bool,
        readable_count: usize,
        projection: Option<Vec<usize>>,
        output_schema: SchemaRef,
        with_row_id: bool,
        with_row_address: bool,
    ) -> Result<Self> {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            batch_store,
            indexes,
            index_expr,
            recheck,
            is_filter_covered,
            readable_count,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            with_row_id,
            with_row_address,
            newest_of: None,
            broad_fallback: None,
        })
    }

    /// Return only the newest visible version of each primary key, reading
    /// with `broad_fallback` when the indexes match too many rows to list.
    pub fn with_newest_check(
        mut self,
        pk_indices: Vec<usize>,
        broad_fallback: Arc<dyn ExecutionPlan>,
    ) -> Self {
        self.newest_of = Some(pk_indices);
        self.broad_fallback = Some(broad_fallback);
        self
    }

    /// How many matches are worth listing out of `visible_rows`.
    fn match_budget(&self, visible_rows: u64) -> u64 {
        if self.newest_of.is_some() {
            visible_rows / NEWEST_ONLY_MATCH_BUDGET_SHARE
        } else {
            (visible_rows / MATCH_BUDGET_SHARE).max(MIN_MATCH_BUDGET)
        }
    }

    /// Last row position within `readable_count`, or None if nothing is
    /// readable.
    fn compute_max_readable_row(&self) -> Option<u64> {
        let mut max_readable_row_exclusive: u64 = 0;
        let mut current_row: u64 = 0;

        for (batch_position, stored_batch) in self.batch_store.iter().enumerate() {
            let batch_end = current_row + stored_batch.num_rows as u64;
            if batch_position < self.readable_count {
                max_readable_row_exclusive = batch_end;
            }
            current_row = batch_end;
        }

        if max_readable_row_exclusive > 0 {
            Some(max_readable_row_exclusive - 1)
        } else {
            None
        }
    }

    /// Evaluate the index searches: the candidate positions, or `None` when
    /// every visible row is one, and whether the answer still needs the filter.
    fn query_index(&self) -> Result<(Option<Vec<u64>>, bool)> {
        let Some(max_readable_row) = self.compute_max_readable_row() else {
            return Ok((Some(Vec::new()), true));
        };
        let visible_rows = max_readable_row + 1;
        let budget = self.match_budget(visible_rows);
        let mut ctx = SearchContext::new(max_readable_row);
        // Offer an index the choice to decline only when a filter or fallback
        // can take its rows.
        if self.recheck.is_some() || self.broad_fallback.is_some() {
            ctx = ctx.with_match_budget(budget);
        }
        let result = evaluate_index_filter(&self.index_expr, &self.indexes, &ctx)?;
        let exact = result.is_exact();
        // An index may list past its budget; the fallback is then cheaper.
        let past_budget = self.broad_fallback.is_some() && result.at_most.len() > budget;
        Ok(if past_budget || result.at_most.len() == visible_rows {
            (None, exact)
        } else {
            (Some(result.at_most.into()), exact)
        })
    }

    /// Read the candidate rows, one stored batch at a time: those at
    /// `candidates` (ascending), or every visible row when it is `None`.
    ///
    /// Each touched batch is filtered with one mask, and a batch without a
    /// candidate is not read.
    fn read_rows(
        &self,
        candidates: Option<&[u64]>,
        recheck: Option<&PhysicalExprRef>,
        max_readable_row: Option<u64>,
        newest_checks: &Count,
    ) -> DataFusionResult<Vec<RecordBatch>> {
        let mut results = Vec::new();
        let mut next = 0;
        for stored in self.batch_store.iter().take(self.readable_count) {
            let start = stored.row_offset;
            // `None` keeps every row of the batch.
            let mut mask = match candidates {
                None => None,
                Some(positions) => {
                    if next == positions.len() {
                        break;
                    }
                    let end = start + stored.num_rows as u64;
                    let first = next;
                    while next < positions.len() && positions[next] < end {
                        next += 1;
                    }
                    let in_batch = &positions[first..next];
                    if in_batch.is_empty() {
                        continue;
                    }
                    if in_batch.len() == stored.num_rows {
                        None
                    } else {
                        let mut selected = BooleanBufferBuilder::new(stored.num_rows);
                        selected.append_n(stored.num_rows, false);
                        for &position in in_batch {
                            selected.set_bit((position - start) as usize, true);
                        }
                        Some(BooleanArray::new(selected.finish(), None))
                    }
                }
            };
            if let Some(recheck) = recheck {
                let evaluated = recheck
                    .evaluate(&stored.data)?
                    .into_array(stored.num_rows)?;
                let evaluated = evaluated.as_boolean_opt().ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "a filter must evaluate to a boolean".to_string(),
                    )
                })?;
                let mut combined = match &mask {
                    Some(selected) => and(selected, evaluated)?,
                    None => evaluated.clone(),
                };
                // A null result is not a match, as it is not in a full scan.
                if combined.null_count() > 0 {
                    combined = prep_null_mask_filter(&combined);
                }
                mask = Some(combined);
            }
            if let (Some(pk_indices), Some(max_readable_row)) = (&self.newest_of, max_readable_row)
            {
                let checked = mask
                    .as_ref()
                    .map_or(stored.num_rows, |mask| mask.true_count());
                newest_checks.add(checked);
                mask = Some(self.keep_newest(
                    &stored.data,
                    start,
                    mask,
                    pk_indices,
                    max_readable_row,
                )?);
            }
            let kept = mask
                .as_ref()
                .map_or(stored.num_rows, |mask| mask.true_count());
            if kept == 0 {
                continue;
            }

            let data = scan_record_batch(&stored.data)?;
            let data = match &mask {
                Some(mask) if kept < stored.num_rows => filter_record_batch(&data, mask)?,
                _ => data,
            };
            let mut columns: Vec<Arc<dyn Array>> = match &self.projection {
                Some(projection) => take_projected_columns(
                    data.columns(),
                    data.schema().fields(),
                    projection,
                    self.output_schema.as_ref(),
                    kept,
                )?,
                None => data.columns().to_vec(),
            };
            if self.with_row_id || self.with_row_address {
                let row_positions: Arc<dyn Array> = Arc::new(match &mask {
                    Some(mask) if kept < stored.num_rows => UInt64Array::from_iter_values(
                        mask.values().set_indices().map(|row| start + row as u64),
                    ),
                    _ => UInt64Array::from_iter_values(start..start + stored.num_rows as u64),
                });
                if self.with_row_id {
                    columns.push(row_positions.clone());
                }
                // A memtable row's address is its position.
                if self.with_row_address {
                    columns.push(row_positions);
                }
            }
            results.push(RecordBatch::try_new(self.output_schema.clone(), columns)?);
        }
        Ok(results)
    }

    /// Narrow `mask` (every row when `None`) to the rows that are their key's
    /// newest visible version: one seek in the primary-key index per row.
    fn keep_newest(
        &self,
        data: &RecordBatch,
        start: u64,
        mask: Option<BooleanArray>,
        pk_indices: &[usize],
        max_readable_row: u64,
    ) -> DataFusionResult<BooleanArray> {
        let rows = data.num_rows();
        let mut keep = BooleanBufferBuilder::new(rows);
        keep.append_n(rows, false);
        let mut check = |row: usize| -> DataFusionResult<()> {
            let values = pk_indices
                .iter()
                .map(|&column| ScalarValue::try_from_array(data.column(column), row))
                .collect::<DataFusionResult<Vec<_>>>()?;
            if self
                .indexes
                .pk_is_newest(&values, start + row as u64, max_readable_row)
            {
                keep.set_bit(row, true);
            }
            Ok(())
        };
        match &mask {
            Some(mask) => mask.values().set_indices().try_for_each(&mut check)?,
            None => (0..rows).try_for_each(&mut check)?,
        }
        Ok(BooleanArray::new(keep.finish(), None))
    }
}

impl DisplayAs for ScalarIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ScalarIndexExec: query={}, rechecked={}{}, with_row_id={}, with_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    if self.newest_of.is_some() {
                        ", newest_only=true"
                    } else {
                        ""
                    },
                    self.with_row_id,
                    self.with_row_address
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ScalarIndexExec\nquery={}\nrechecked={}\nnewest_only={}\nwith_row_id={}\nwith_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    self.newest_of.is_some(),
                    self.with_row_id,
                    self.with_row_address
                )
            }
        }
    }
}

impl ExecutionPlan for ScalarIndexExec {
    fn name(&self) -> &str {
        "ScalarIndexExec"
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let newest_checks =
            MetricBuilder::new(&self.metrics).counter(NEWEST_CHECKS_METRIC, partition);
        let (positions, exact) = self.query_index()?;
        if positions.is_none()
            && let Some(fallback) = &self.broad_fallback
        {
            return fallback.execute(partition, context);
        }

        // Candidates from a narrowing or partial answer still need the filter.
        let recheck = if !exact || !self.is_filter_covered {
            let Some(recheck) = &self.recheck else {
                return Err(datafusion::error::DataFusionError::Internal(
                    "the indexes did not decide the filter, but no filter was given to re-check with"
                        .to_string(),
                ));
            };
            Some(recheck)
        } else {
            None
        };
        let batches = self.read_rows(
            positions.as_deref(),
            recheck,
            self.compute_max_readable_row(),
            &newest_checks,
        )?;

        let stream = stream::iter(batches.into_iter().map(Ok)).boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        // We can't know the exact count without querying the index
        Ok(Arc::new(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: Statistics::unknown_column(&self.schema()),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use futures::TryStreamExt;
    use lance_index::scalar::SargableQuery;
    use lance_index::scalar::expression::ScalarIndexSearch;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &Schema, start_id: i32, count: usize) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|id| format!("name_{}", id)).collect();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    /// Past the match budget every visible row is read with the filter, and a
    /// batch past the readable count stays unread.
    #[tokio::test]
    async fn a_broad_answer_reads_every_visible_row_with_the_filter() {
        const ROWS_PER_BATCH: usize = 1_000;
        const READABLE_BATCHES: usize = 20;
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());
        for n in 0..=READABLE_BATCHES {
            let start = n * ROWS_PER_BATCH;
            let batch = create_test_batch(&schema, start as i32, ROWS_PER_BATCH);
            batch_store.append(batch.clone()).unwrap();
            indexes
                .insert_with_batch_position(&batch, start as u64, Some(n))
                .unwrap();
        }
        let visible_rows = (READABLE_BATCHES * ROWS_PER_BATCH) as u64;

        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Int32(Some(100))),
            std::ops::Bound::Unbounded,
        );
        let budget = (visible_rows / MATCH_BUDGET_SHARE).max(MIN_MATCH_BUDGET);
        let ctx = SearchContext::new(visible_rows - 1).with_match_budget(budget);
        assert!(
            indexes
                .get_index("id_idx")
                .unwrap()
                .search(&query, &ctx)
                .unwrap()
                .is_none(),
            "the B-tree must decline this many matches"
        );

        let planner = lance_datafusion::planner::Planner::new(schema.clone());
        let recheck = planner
            .create_physical_expr(&planner.parse_filter("id >= 100").unwrap())
            .unwrap();
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));
        let exec = ScalarIndexExec::new(
            batch_store,
            Arc::new(indexes),
            search("id_idx", "id", query),
            Some(recheck),
            true,
            READABLE_BATCHES,
            None,
            schema_with_rowid,
            true,
            false,
        )
        .unwrap();
        let batches: Vec<RecordBatch> = exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch["id"]
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let row_ids: Vec<u64> = batches
            .iter()
            .flat_map(|batch| {
                batch["_rowid"]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let expected: Vec<i32> = (100..visible_rows as i32).collect();
        assert_eq!(ids, expected);
        assert_eq!(
            row_ids,
            expected.iter().map(|id| *id as u64).collect::<Vec<_>>()
        );
    }

    /// One index search, the way the expression pass produces it.
    fn search(index_name: &str, column: &str, query: SargableQuery) -> ScalarIndexExpr {
        ScalarIndexExpr::Query(ScalarIndexSearch {
            column: column.to_string(),
            index_name: index_name.to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(query),
            needs_recheck: false,
            fragment_bitmap: None,
        })
    }

    #[tokio::test]
    async fn test_btree_index_eq_query() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Create index registry with btree index on "id" (field_id = 0)
        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert test data and update index
        let batch = create_test_batch(&schema, 0, 10);
        registry.insert(&batch, 0).unwrap();
        batch_store.append(batch).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1, // readable_count (batch at position 0)
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find one row with id=5
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_btree_index_in_query() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        let batch = create_test_batch(&schema, 0, 10);
        registry.insert(&batch, 0).unwrap();
        batch_store.append(batch).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::IsIn(vec![
                ScalarValue::Int32(Some(2)),
                ScalarValue::Int32(Some(5)),
                ScalarValue::Int32(Some(8)),
            ]),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1,
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find 3 rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn test_btree_index_visibility() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert two batches at positions 0 and 1
        let batch1 = create_test_batch(&schema, 0, 10);
        let batch2 = create_test_batch(&schema, 10, 10);
        registry.insert(&batch1, 0).unwrap();
        registry.insert(&batch2, 10).unwrap();
        batch_store.append(batch1).unwrap();
        batch_store.append(batch2).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(15))),
        );

        // Query with max_readable=0 should not see batch at position 1
        let exec = ScalarIndexExec::new(
            batch_store.clone(),
            indexes.clone(),
            index_expr.clone(),
            None,
            true,
            1,
            None,
            schema.clone(),
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);

        // Query with max_readable=1 should see both batches
        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            2,
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_btree_index_with_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert batch with 10 rows at position 0
        let batch = create_test_batch(&schema, 0, 10);
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();

        let indexes = Arc::new(indexes);

        // Add _rowid to schema
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1,
            None,
            schema_with_rowid.clone(),
            true,
            false,
        )
        .unwrap();

        // Verify the plan output
        let debug_str = format!("{:?}", exec);
        assert!(debug_str.contains("with_row_id: true"));
        assert!(debug_str.contains("with_row_address: false"));

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find one row with id=5
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);

        // Verify _rowid column is present and has correct value
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(2).name(), "_rowid");

        let row_ids = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 5); // Row position for id=5 is 5
    }

    #[tokio::test]
    async fn test_btree_plan_display() {
        use crate::utils::test::assert_plan_node_equals;
        use datafusion::physical_plan::ExecutionPlan;

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());

        let batch = create_test_batch(&schema, 0, 10);
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();

        let indexes = Arc::new(indexes);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        // Test plan display without _rowid
        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            ScalarIndexExec::new(
                batch_store.clone(),
                indexes.clone(),
                index_expr.clone(),
                None,
                true,
                1,
                None,
                schema.clone(),
                false,
                false,
            )
            .unwrap(),
        );

        assert_plan_node_equals(
            exec,
            "ScalarIndexExec: query=id = Int32(5), rechecked=false, with_row_id=false, with_row_address=false",
        )
        .await
        .unwrap();

        // Test plan display with _rowid
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));

        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            ScalarIndexExec::new(
                batch_store,
                indexes,
                index_expr,
                None,
                true,
                1,
                None,
                schema_with_rowid,
                true,
                false,
            )
            .unwrap(),
        );

        assert_plan_node_equals(
            exec,
            "ScalarIndexExec: query=id = Int32(5), rechecked=false, with_row_id=true, with_row_address=false",
        )
        .await
        .unwrap();
    }
}
