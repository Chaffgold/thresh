//! Python bindings for MOT metric computation.

use pyo3::prelude::*;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::compute_mot_metrics;

/// Convert Python-side frame tuples into `FrameData`.
fn to_frame_data(
    gt_frames: &[Vec<(u64, Vec<f64>)>],
    track_frames: &[Vec<(u64, Vec<f64>)>],
) -> Vec<FrameData> {
    let n = gt_frames.len().max(track_frames.len());
    (0..n)
        .map(|i| {
            let gt = gt_frames
                .get(i)
                .map(|g| {
                    g.iter()
                        .map(|(id, pos)| {
                            let p = vec_to_pos3(pos);
                            (*id, p)
                        })
                        .collect()
                })
                .unwrap_or_default();

            let tracks = track_frames
                .get(i)
                .map(|t| {
                    t.iter()
                        .map(|(id, pos)| {
                            let p = vec_to_pos3(pos);
                            (*id, p)
                        })
                        .collect()
                })
                .unwrap_or_default();

            FrameData { gt, tracks }
        })
        .collect()
}

/// Convert a `Vec<f64>` (possibly shorter/longer than 3) to `[f64; 3]`.
fn vec_to_pos3(v: &[f64]) -> [f64; 3] {
    [
        v.first().copied().unwrap_or(0.0),
        v.get(1).copied().unwrap_or(0.0),
        v.get(2).copied().unwrap_or(0.0),
    ]
}

/// Compute MOT metrics from ground truth and track lists.
///
/// # Arguments
/// * `gt_frames` - List of frames, each a list of `(id, [x, y, z])` tuples.
/// * `track_frames` - List of frames, each a list of `(id, [x, y, z])` tuples.
/// * `distance_threshold` - Maximum match distance.
///
/// # Returns
/// Tuple of `(mota, motp, id_switches)`.
#[pyfunction]
#[pyo3(signature = (gt_frames, track_frames, distance_threshold))]
pub fn compute_mot_metrics_py(
    gt_frames: Vec<Vec<(u64, Vec<f64>)>>,
    track_frames: Vec<Vec<(u64, Vec<f64>)>>,
    distance_threshold: f64,
) -> PyResult<(f64, f64, usize)> {
    let frames = to_frame_data(&gt_frames, &track_frames);
    let (mota, motp, idsw) = compute_mot_metrics(&frames, distance_threshold);
    Ok((mota, motp, idsw))
}

// ── Rust-only unit tests (no Python runtime needed) ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_pos3_full() {
        let p = vec_to_pos3(&[1.0, 2.0, 3.0]);
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec_to_pos3_short() {
        let p = vec_to_pos3(&[5.0]);
        assert_eq!(p, [5.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vec_to_pos3_empty() {
        let p = vec_to_pos3(&[]);
        assert_eq!(p, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_to_frame_data_basic() {
        let gt = vec![vec![(1u64, vec![0.0, 0.0, 0.0])]];
        let tr = vec![vec![(10u64, vec![0.0, 0.0, 0.0])]];
        let frames = to_frame_data(&gt, &tr);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].gt.len(), 1);
        assert_eq!(frames[0].tracks.len(), 1);
    }

    #[test]
    fn test_compute_metrics_perfect_match() {
        let gt = vec![
            vec![(1u64, vec![0.0, 0.0, 0.0]), (2, vec![100.0, 0.0, 0.0])],
            vec![(1u64, vec![10.0, 0.0, 0.0]), (2, vec![110.0, 0.0, 0.0])],
        ];
        let tr = vec![
            vec![(101u64, vec![0.0, 0.0, 0.0]), (102, vec![100.0, 0.0, 0.0])],
            vec![(101u64, vec![10.0, 0.0, 0.0]), (102, vec![110.0, 0.0, 0.0])],
        ];

        let frames = to_frame_data(&gt, &tr);
        let (mota, motp, idsw) = compute_mot_metrics(&frames, 1.0);
        assert!((mota - 1.0).abs() < 1e-10, "MOTA should be 1.0, got {mota}");
        assert!(motp < 1e-10, "MOTP should be ~0, got {motp}");
        assert_eq!(idsw, 0);
    }

    #[test]
    fn test_compute_metrics_no_tracks() {
        let gt = vec![
            vec![(1u64, vec![0.0, 0.0, 0.0])],
            vec![(1u64, vec![10.0, 0.0, 0.0])],
        ];
        let tr: Vec<Vec<(u64, Vec<f64>)>> = vec![vec![], vec![]];

        let frames = to_frame_data(&gt, &tr);
        let (mota, _motp, _idsw) = compute_mot_metrics(&frames, 10.0);
        assert!(
            (mota - 0.0).abs() < 1e-10,
            "MOTA should be 0.0 with no tracks, got {mota}"
        );
    }

    // PyO3 integration tests require a Python interpreter.
    #[test]
    #[ignore = "requires maturin develop + Python interpreter"]
    fn test_py_function_callable() {
        // Would test via pyo3::Python::with_gil.
    }
}
