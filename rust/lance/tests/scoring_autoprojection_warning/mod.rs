// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The scoring-column autoprojection deprecation warning is logged once per
//! process, however many searches reach it. It used to be logged on every
//! planned scanner, which for an LSM read is one line per generation searched.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

/// Counts the deprecation warnings for each scoring column. The binary's only
/// logger, so every warning any test emits passes through it.
struct CountingLogger;

static DISTANCE_WARNINGS: AtomicUsize = AtomicUsize::new(0);
static SCORE_WARNINGS: AtomicUsize = AtomicUsize::new(0);

impl log::Log for CountingLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        let message = record.args().to_string();
        if !message.contains("disable_scoring_autoprojection") {
            return;
        }
        if message.contains("`_distance`") {
            DISTANCE_WARNINGS.fetch_add(1, Ordering::Relaxed);
        } else if message.contains("`_score`") {
            SCORE_WARNINGS.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {}
}

static LOGGER: CountingLogger = CountingLogger;

#[tokio::test]
async fn scoring_autoprojection_warning_is_logged_once_per_process() {
    log::set_logger(&LOGGER).expect("no other logger in this test binary");
    log::set_max_level(log::LevelFilter::Warn);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
    ]));
    let ids: Vec<i32> = (0..64).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids.clone())),
            Arc::new(StringArray::from(
                ids.iter()
                    .map(|i| format!("lorem r{i}"))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                2,
                Arc::new(Float32Array::from(
                    ids.iter()
                        .flat_map(|i| [*i as f32, 0.0])
                        .collect::<Vec<_>>(),
                )),
                None,
            )),
        ],
    )
    .unwrap();
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    const SEARCHES: usize = 20;
    for i in 0..SEARCHES {
        // Each search projects `id` alone, so the scoring column is appended
        // by the deprecated autoprojection.
        let mut vector = dataset.scan();
        vector
            .project(&["id"])
            .unwrap()
            .nearest("vec", &Float32Array::from(vec![i as f32, 0.0]), 3)
            .unwrap();
        let rows: Vec<RecordBatch> = vector
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(
            rows[0].column_by_name("_distance").is_some(),
            "the column is still appended"
        );

        let mut fts = dataset.scan();
        fts.project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("lorem".to_string()))
            .unwrap();
        let rows: Vec<RecordBatch> = fts
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(
            rows[0].column_by_name("_score").is_some(),
            "the column is still appended"
        );
    }

    assert_eq!(
        DISTANCE_WARNINGS.load(Ordering::Relaxed),
        1,
        "{SEARCHES} vector searches must log the `_distance` warning once"
    );
    assert_eq!(
        SCORE_WARNINGS.load(Ordering::Relaxed),
        1,
        "{SEARCHES} full-text searches must log the `_score` warning once"
    );
}
