//! Rust-native Parquet export for IMM-classifier training samples (task 5.5).
//!
//! Gated behind the `training-export` feature so the `arrow`/`parquet` deps
//! stay out of default and CI builds. The schema is intentionally simple and
//! self-describing so the Phase 6 PyTorch loader can read it directly with
//! `pyarrow`:
//!
//! | column              | type            | notes                              |
//! |---------------------|-----------------|------------------------------------|
//! | `trajectory_id`     | uint32          | source trajectory                  |
//! | `track_id`          | uint64          | tracker track the feature is from  |
//! | `time_s`            | float64         | tick time (s)                      |
//! | `feature`           | list<float64>   | 12-dim filter-state feature        |
//! | `label`             | utf8            | `cv`/`ca`/`ctrv`/`coord_turn`      |
//! | `label_index`       | int32           | IMM mode index `0..4`              |
//! | `imm_dominant_mode` | int32 nullable  | analytic IMM's mode (reference)    |

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, Float64Array, Float64Builder, Int32Array,
    Int64Builder, ListBuilder, StringArray, UInt32Array, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::ImmTrainingSample;
use super::detector::DetectorSample;

/// Assemble the training samples into a single Arrow [`RecordBatch`] matching
/// the documented schema.
pub fn samples_to_record_batch(
    samples: &[ImmTrainingSample],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let trajectory_id =
        UInt32Array::from(samples.iter().map(|s| s.trajectory_id).collect::<Vec<_>>());
    let track_id = UInt64Array::from(samples.iter().map(|s| s.track_id).collect::<Vec<_>>());
    let time_s = Float64Array::from(samples.iter().map(|s| s.time_s).collect::<Vec<_>>());
    let label = StringArray::from(samples.iter().map(|s| s.label.as_str()).collect::<Vec<_>>());
    let label_index = Int32Array::from(
        samples
            .iter()
            .map(|s| s.label.as_index() as i32)
            .collect::<Vec<_>>(),
    );
    let imm_dominant_mode = Int32Array::from(
        samples
            .iter()
            .map(|s| s.imm_dominant_mode.map(|m| m as i32))
            .collect::<Vec<Option<i32>>>(),
    );

    let mut feature_builder = ListBuilder::new(Float64Builder::new());
    for s in samples {
        feature_builder.values().append_slice(&s.feature);
        feature_builder.append(true);
    }
    let feature = feature_builder.finish();

    // `try_from_iter` infers a schema (all fields nullable) directly from the
    // arrays, avoiding inner-field-name mismatches on the list column.
    let batch = RecordBatch::try_from_iter(vec![
        ("trajectory_id", Arc::new(trajectory_id) as ArrayRef),
        ("track_id", Arc::new(track_id) as ArrayRef),
        ("time_s", Arc::new(time_s) as ArrayRef),
        ("feature", Arc::new(feature) as ArrayRef),
        ("label", Arc::new(label) as ArrayRef),
        ("label_index", Arc::new(label_index) as ArrayRef),
        ("imm_dominant_mode", Arc::new(imm_dominant_mode) as ArrayRef),
    ])?;
    Ok(batch)
}

/// Write IMM training samples to a Parquet file at `path`.
pub fn write_imm_samples_parquet(
    samples: &[ImmTrainingSample],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let batch = samples_to_record_batch(samples)?;
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Track A detector samples (Phase 7 part 2)
// ---------------------------------------------------------------------------

/// Assemble detector samples into an Arrow [`RecordBatch`].
///
/// Schema (read torch-free by `python/training/detector_dataset.py`):
///
/// | column          | type           | notes                              |
/// |-----------------|----------------|------------------------------------|
/// | `trajectory_id` | uint32         | source scene                       |
/// | `snapshot_index`| uint64         | snapshot within the scene          |
/// | `time_s`        | float64        | snapshot time (s)                  |
/// | `point_cloud`   | list<float32>  | 1000×4 flattened `[x,y,z,intensity]` |
/// | `gt_boxes`      | list<float32>  | 100×7 flattened `[x,y,z,L,W,H,yaw]`  |
/// | `gt_valid`      | list<bool>     | 100 validity flags                 |
/// | `gt_classes`    | list<int64>    | 100 class indices                  |
pub fn detector_samples_to_record_batch(
    samples: &[DetectorSample],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    // Validate shapes up front: ragged `append_slice` would otherwise silently
    // write variable-length lists that the Python reader's reshape rejects.
    const PC_LEN: usize = 1000 * 4;
    const BOX_LEN: usize = 100 * 7;
    const MASK_LEN: usize = 100;
    for (i, s) in samples.iter().enumerate() {
        if s.point_cloud.len() != PC_LEN
            || s.gt_boxes.len() != BOX_LEN
            || s.gt_valid.len() != MASK_LEN
            || s.gt_classes.len() != MASK_LEN
        {
            return Err(format!(
                "detector sample {i} has malformed shapes \
                 (point_cloud={}, gt_boxes={}, gt_valid={}, gt_classes={}); \
                 expected ({PC_LEN}, {BOX_LEN}, {MASK_LEN}, {MASK_LEN})",
                s.point_cloud.len(),
                s.gt_boxes.len(),
                s.gt_valid.len(),
                s.gt_classes.len(),
            )
            .into());
        }
    }

    let trajectory_id =
        UInt32Array::from(samples.iter().map(|s| s.trajectory_id).collect::<Vec<_>>());
    let snapshot_index =
        UInt64Array::from(samples.iter().map(|s| s.snapshot_index).collect::<Vec<_>>());
    let time_s = Float64Array::from(samples.iter().map(|s| s.time_s).collect::<Vec<_>>());

    let mut pc = ListBuilder::new(Float32Builder::new());
    for s in samples {
        pc.values().append_slice(&s.point_cloud);
        pc.append(true);
    }
    let point_cloud = pc.finish();

    let mut gb = ListBuilder::new(Float32Builder::new());
    for s in samples {
        gb.values().append_slice(&s.gt_boxes);
        gb.append(true);
    }
    let gt_boxes = gb.finish();

    let mut gv = ListBuilder::new(BooleanBuilder::new());
    for s in samples {
        for &b in &s.gt_valid {
            gv.values().append_value(b);
        }
        gv.append(true);
    }
    let gt_valid = gv.finish();

    let mut gc = ListBuilder::new(Int64Builder::new());
    for s in samples {
        gc.values().append_slice(&s.gt_classes);
        gc.append(true);
    }
    let gt_classes = gc.finish();

    let batch = RecordBatch::try_from_iter(vec![
        ("trajectory_id", Arc::new(trajectory_id) as ArrayRef),
        ("snapshot_index", Arc::new(snapshot_index) as ArrayRef),
        ("time_s", Arc::new(time_s) as ArrayRef),
        ("point_cloud", Arc::new(point_cloud) as ArrayRef),
        ("gt_boxes", Arc::new(gt_boxes) as ArrayRef),
        ("gt_valid", Arc::new(gt_valid) as ArrayRef),
        ("gt_classes", Arc::new(gt_classes) as ArrayRef),
    ])?;
    Ok(batch)
}

/// Write detector training samples to a Parquet file at `path`.
///
/// Uses Snappy compression — the point clouds are mostly zero-padding and
/// clutter, so they compress well, keeping the checked-in sample small.
pub fn write_detector_samples_parquet(
    samples: &[DetectorSample],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let batch = detector_samples_to_record_batch(samples)?;
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::ImmTrainingSample;
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use thresh_core::motion_mode::MotionModeLabel;

    fn sample(label: MotionModeLabel, dom: Option<usize>) -> ImmTrainingSample {
        ImmTrainingSample {
            trajectory_id: 3,
            track_id: 42,
            time_s: 1.5,
            feature: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            label,
            imm_dominant_mode: dom,
        }
    }

    #[test]
    fn parquet_round_trip_preserves_rows_and_feature_width() {
        let samples = vec![
            sample(MotionModeLabel::Cv, Some(0)),
            sample(MotionModeLabel::CoordTurn, None),
        ];
        let path = std::env::temp_dir().join(format!(
            "thresh_imm_phase5_roundtrip_{}.parquet",
            std::process::id()
        ));

        write_imm_samples_parquet(&samples, &path).expect("write parquet");

        let file = std::fs::File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 7);

        // The `feature` list column should hold 12 floats per row.
        let features = batch
            .column_by_name("feature")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(features.value(0).len(), 12);

        let labels = batch
            .column_by_name("label")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(labels.value(0), "cv");
        assert_eq!(labels.value(1), "coord_turn");

        // imm_dominant_mode is nullable and the second row was None.
        let dom = batch
            .column_by_name("imm_dominant_mode")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(dom.value(0), 0);
        assert!(dom.is_null(1));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn detector_parquet_round_trip_preserves_shapes() {
        use crate::training::detector::DetectorSample;
        let s = DetectorSample {
            trajectory_id: 1,
            snapshot_index: 0,
            time_s: 0.5,
            point_cloud: vec![0.25_f32; 1000 * 4],
            gt_boxes: vec![1.0_f32; 100 * 7],
            gt_valid: {
                let mut v = vec![false; 100];
                v[0] = true;
                v
            },
            gt_classes: {
                let mut c = vec![0_i64; 100];
                c[0] = 2;
                c
            },
        };
        let path = std::env::temp_dir().join(format!(
            "thresh_detector_phase7_roundtrip_{}.parquet",
            std::process::id()
        ));

        write_detector_samples_parquet(std::slice::from_ref(&s), &path).expect("write parquet");

        let file = std::fs::File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 7);

        let pc = batch
            .column_by_name("point_cloud")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(pc.value(0).len(), 1000 * 4);
        let boxes = batch
            .column_by_name("gt_boxes")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(boxes.value(0).len(), 100 * 7);
        let classes = batch
            .column_by_name("gt_classes")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(classes.value(0).len(), 100);

        let valid = batch
            .column_by_name("gt_valid")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(valid.value(0).len(), 100);
        let valid_bools = valid
            .value(0)
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap()
            .iter()
            .filter(|b| b == &Some(true))
            .count();
        assert_eq!(valid_bools, 1, "exactly one valid box was written");

        std::fs::remove_file(&path).ok();
    }
}
