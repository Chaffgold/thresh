//! Python wrapper around `MultiObjectTracker`.

use nalgebra::DVector;
use pyo3::prelude::*;
use thresh_core::track::TrackState;
use thresh_tracker::tracker::MultiObjectTracker;

/// Python-visible track state snapshot.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTrackState {
    /// Track ID.
    #[pyo3(get)]
    pub id: u64,
    /// Position [x, y, z].
    #[pyo3(get)]
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz].
    #[pyo3(get)]
    pub velocity: [f64; 3],
    /// Whether the track is confirmed.
    #[pyo3(get)]
    pub is_confirmed: bool,
}

/// Multi-object tracker exposed to Python.
///
/// Marked `unsendable` because `MultiObjectTracker` contains trait objects
/// (`dyn Fn`, `dyn MotionModel`) that are not `Send + Sync`. Access is
/// restricted to the thread holding the GIL, which is safe for typical usage.
#[pyclass(unsendable)]
pub struct PyMultiObjectTracker {
    inner: MultiObjectTracker,
}

#[pymethods]
impl PyMultiObjectTracker {
    /// Create a new tracker with constant-velocity motion model.
    ///
    /// # Arguments
    /// * `measurement_noise_sigma` - Standard deviation of measurement noise.
    /// * `gate_threshold` - Mahalanobis distance gating threshold.
    #[new]
    #[pyo3(signature = (measurement_noise_sigma, gate_threshold))]
    fn new(measurement_noise_sigma: f64, gate_threshold: f64) -> Self {
        Self {
            inner: MultiObjectTracker::new_cv_position(measurement_noise_sigma, gate_threshold),
        }
    }

    /// Step the tracker with a list of 3D detections.
    ///
    /// Each detection is a list `[x, y, z]`.
    fn step(&mut self, detections: Vec<Vec<f64>>, dt: f64) {
        let dets: Vec<DVector<f64>> = detections.into_iter().map(DVector::from_vec).collect();
        self.inner.step(&dets, dt);
    }

    /// Get current tracks as a list of `PyTrackState`.
    fn get_tracks(&self) -> Vec<PyTrackState> {
        tracks_to_py(&self.inner)
    }

    /// Number of alive tracks (tentative + confirmed + coasting).
    fn alive_count(&self) -> usize {
        self.inner.alive_count()
    }

    /// Number of confirmed tracks.
    fn confirmed_count(&self) -> usize {
        self.inner.confirmed_count()
    }
}

/// Convert the tracker's internal tracks to Python-visible snapshots.
fn tracks_to_py(tracker: &MultiObjectTracker) -> Vec<PyTrackState> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.is_alive())
        .map(|t| {
            // State layout: [x, vx, y, vy, z, vz]
            let s = &t.state;
            let (px, py, pz) = extract_position(s);
            let (vx, vy, vz) = extract_velocity(s);
            PyTrackState {
                id: t.id.0,
                position: [px, py, pz],
                velocity: [vx, vy, vz],
                is_confirmed: t.lifecycle == TrackState::Confirmed,
            }
        })
        .collect()
}

/// Extract position [x, y, z] from state vector [x, vx, y, vy, z, vz].
fn extract_position(s: &DVector<f64>) -> (f64, f64, f64) {
    (
        if !s.is_empty() { s[0] } else { 0.0 },
        if s.len() > 2 { s[2] } else { 0.0 },
        if s.len() > 4 { s[4] } else { 0.0 },
    )
}

/// Extract velocity [vx, vy, vz] from state vector [x, vx, y, vy, z, vz].
fn extract_velocity(s: &DVector<f64>) -> (f64, f64, f64) {
    (
        if s.len() > 1 { s[1] } else { 0.0 },
        if s.len() > 3 { s[3] } else { 0.0 },
        if s.len() > 5 { s[5] } else { 0.0 },
    )
}

// ── Rust-only unit tests (no Python runtime needed) ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_position_and_velocity() {
        let s = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (px, py, pz) = extract_position(&s);
        assert!((px - 1.0).abs() < 1e-12);
        assert!((py - 3.0).abs() < 1e-12);
        assert!((pz - 5.0).abs() < 1e-12);

        let (vx, vy, vz) = extract_velocity(&s);
        assert!((vx - 2.0).abs() < 1e-12);
        assert!((vy - 4.0).abs() < 1e-12);
        assert!((vz - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_extract_short_state() {
        let s = DVector::from_column_slice(&[10.0]);
        let (px, py, pz) = extract_position(&s);
        assert!((px - 10.0).abs() < 1e-12);
        assert!(py.abs() < 1e-12);
        assert!(pz.abs() < 1e-12);
    }

    #[test]
    fn test_tracks_to_py_empty_tracker() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let py_tracks = tracks_to_py(&tracker);
        assert!(py_tracks.is_empty());
    }

    /// Test that stepping the tracker with detections produces alive tracks.
    /// Uses the raw Rust API (no Python interpreter).
    #[test]
    fn test_tracker_step_and_get_tracks() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);

        // Step 5 times with the same detection to get past M-of-N confirmation.
        for _ in 0..5 {
            tracker.step(std::slice::from_ref(&det), 1.0);
        }

        let py_tracks = tracks_to_py(&tracker);
        assert!(!py_tracks.is_empty(), "should have at least one track");
        assert!(
            py_tracks.iter().any(|t| t.is_confirmed),
            "at least one track should be confirmed after 5 steps"
        );

        // Position should be near the detection.
        let confirmed = py_tracks.iter().find(|t| t.is_confirmed).unwrap();
        assert!((confirmed.position[0] - 100.0).abs() < 20.0);
        assert!((confirmed.position[1] - 200.0).abs() < 20.0);
        assert!((confirmed.position[2] - 50.0).abs() < 20.0);
    }

    #[test]
    fn test_tracker_alive_count() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        assert_eq!(tracker.alive_count(), 0);

        let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
        tracker.step(std::slice::from_ref(&det), 1.0);
        assert!(tracker.alive_count() >= 1);

        // Add a second, well-separated detection.
        let dets = vec![
            DVector::from_column_slice(&[100.0, 200.0, 50.0]),
            DVector::from_column_slice(&[9000.0, 9000.0, 9000.0]),
        ];
        tracker.step(&dets, 1.0);
        assert!(tracker.alive_count() >= 2);
    }

    // PyO3 integration tests require a Python interpreter (maturin develop + pytest).
    #[test]
    #[ignore = "requires maturin develop + Python interpreter"]
    fn test_py_module_loads() {
        // Would use pyo3::Python::with_gil to test module registration.
    }
}
