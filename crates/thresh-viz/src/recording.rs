//! Recording types for track visualization data.
//!
//! These types capture tracker state at each timestep in a serializable form
//! suitable for playback, analysis, and JSON export/import.

use serde::{Deserialize, Serialize};

use thresh_core::track::TrackState as CoreTrackState;
use thresh_tracker::tracker::MultiObjectTracker;

/// A recorded track visualization frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizFrame {
    /// Wall-clock or simulation timestamp in seconds.
    pub timestamp: f64,
    /// Track states at this timestep.
    pub tracks: Vec<VizTrack>,
    /// Raw detections at this timestep.
    pub detections: Vec<VizDetection>,
    /// Ground-truth positions at this timestep.
    pub ground_truth: Vec<VizGroundTruth>,
}

/// Serializable snapshot of a single track.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizTrack {
    /// Track identifier.
    pub id: u64,
    /// Position `[x, y, z]`.
    pub position: [f64; 3],
    /// Velocity `[vx, vy, vz]`.
    pub velocity: [f64; 3],
    /// Diagonal of the 6x6 covariance (pos+vel).
    pub covariance_diag: [f64; 6],
    /// Whether the track is confirmed.
    pub is_confirmed: bool,
    /// Optional classification label.
    pub class_label: Option<String>,
}

/// A raw detection (sensor measurement).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizDetection {
    /// Detection position `[x, y, z]`.
    pub position: [f64; 3],
    /// Sensor that produced this detection.
    pub sensor_id: u32,
}

/// A ground-truth object position for metric overlay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizGroundTruth {
    /// Ground-truth object identifier.
    pub id: u64,
    /// True position `[x, y, z]`.
    pub position: [f64; 3],
}

/// A full recording: sequence of frames that can be saved/loaded as JSON.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recording {
    /// Human-readable name for this recording.
    pub name: String,
    /// Ordered sequence of visualization frames.
    pub frames: Vec<VizFrame>,
}

impl Recording {
    /// Create a new empty recording with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            frames: Vec::new(),
        }
    }

    /// Append a frame to the recording.
    pub fn push_frame(&mut self, frame: VizFrame) {
        self.frames.push(frame);
    }

    /// Save the recording as pretty-printed JSON.
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a recording from a JSON file.
    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Duration in seconds (last timestamp minus first).
    ///
    /// Returns `0.0` if fewer than two frames exist.
    pub fn duration(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        self.frames.last().unwrap().timestamp - self.frames.first().unwrap().timestamp
    }

    /// Number of frames in the recording.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl VizFrame {
    /// Build a [`VizFrame`] from the current tracker state.
    ///
    /// `detections` are the raw measurement vectors (3D position each).
    /// `ground_truth` is optional truth data as `(id, [x, y, z])` pairs.
    pub fn from_tracker(
        tracker: &MultiObjectTracker,
        detections: &[nalgebra::DVector<f64>],
        ground_truth: &[(u64, [f64; 3])],
        timestamp: f64,
    ) -> Self {
        let tracks = tracker
            .tracks
            .iter()
            .filter(|t| t.is_alive())
            .map(|t| {
                let s = &t.state;
                let state_dim = s.len();
                // State layout: [x, vx, y, vy, z, vz]
                let position = [
                    if state_dim > 0 { s[0] } else { 0.0 },
                    if state_dim > 2 { s[2] } else { 0.0 },
                    if state_dim > 4 { s[4] } else { 0.0 },
                ];
                let velocity = [
                    if state_dim > 1 { s[1] } else { 0.0 },
                    if state_dim > 3 { s[3] } else { 0.0 },
                    if state_dim > 5 { s[5] } else { 0.0 },
                ];

                let cov = &t.covariance;
                let cov_rows = cov.nrows();
                let mut covariance_diag = [0.0; 6];
                for i in 0..6.min(cov_rows) {
                    covariance_diag[i] = cov[(i, i)];
                }

                let class_label = match t.class {
                    thresh_core::track::TargetClass::Unknown => None,
                    other => Some(format!("{other:?}")),
                };

                VizTrack {
                    id: t.id.0,
                    position,
                    velocity,
                    covariance_diag,
                    is_confirmed: t.lifecycle == CoreTrackState::Confirmed,
                    class_label,
                }
            })
            .collect();

        let viz_detections = detections
            .iter()
            .map(|d| {
                let pos = [
                    if !d.is_empty() { d[0] } else { 0.0 },
                    if d.len() > 1 { d[1] } else { 0.0 },
                    if d.len() > 2 { d[2] } else { 0.0 },
                ];
                VizDetection {
                    position: pos,
                    sensor_id: 0,
                }
            })
            .collect();

        let viz_gt = ground_truth
            .iter()
            .map(|(id, pos)| VizGroundTruth {
                id: *id,
                position: *pos,
            })
            .collect();

        VizFrame {
            timestamp,
            tracks,
            detections: viz_detections,
            ground_truth: viz_gt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(timestamp: f64) -> VizFrame {
        VizFrame {
            timestamp,
            tracks: vec![VizTrack {
                id: 1,
                position: [1.0, 2.0, 3.0],
                velocity: [0.1, 0.2, 0.3],
                covariance_diag: [1.0; 6],
                is_confirmed: true,
                class_label: None,
            }],
            detections: vec![VizDetection {
                position: [1.0, 2.0, 3.0],
                sensor_id: 0,
            }],
            ground_truth: vec![VizGroundTruth {
                id: 100,
                position: [1.0, 2.0, 3.0],
            }],
        }
    }

    #[test]
    fn test_recording_push_and_count() {
        let mut rec = Recording::new("test");
        for i in 0..5 {
            rec.push_frame(make_frame(i as f64));
        }
        assert_eq!(rec.frame_count(), 5);
    }

    #[test]
    fn test_recording_save_load_json() {
        let mut rec = Recording::new("roundtrip");
        for i in 0..3 {
            rec.push_frame(make_frame(i as f64));
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        let path_str = path.to_str().unwrap();

        rec.save_json(path_str).unwrap();
        let loaded = Recording::load_json(path_str).unwrap();

        assert_eq!(rec, loaded);
    }

    #[test]
    fn test_recording_duration() {
        let mut rec = Recording::new("duration-test");
        for i in 0..10 {
            rec.push_frame(make_frame(i as f64));
        }
        assert!((rec.duration() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_recording_duration_empty() {
        let rec = Recording::new("empty");
        assert_eq!(rec.duration(), 0.0);
    }

    #[test]
    fn test_recording_duration_single_frame() {
        let mut rec = Recording::new("single");
        rec.push_frame(make_frame(5.0));
        assert_eq!(rec.duration(), 0.0);
    }

    #[test]
    fn test_viz_frame_from_tracker() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let det = nalgebra::DVector::from_column_slice(&[100.0, 200.0, 50.0]);

        // Run a few steps to birth and start updating a track
        for _ in 0..4 {
            tracker.step(std::slice::from_ref(&det), 1.0);
        }

        let gt = vec![(1u64, [100.0, 200.0, 50.0])];
        let frame = VizFrame::from_tracker(&tracker, std::slice::from_ref(&det), &gt, 4.0);

        assert_eq!(frame.timestamp, 4.0);
        assert!(!frame.tracks.is_empty(), "should have at least one track");
        assert_eq!(frame.detections.len(), 1);
        assert_eq!(frame.ground_truth.len(), 1);
        assert_eq!(frame.ground_truth[0].id, 1);

        // The track position should be near the detection
        let t = &frame.tracks[0];
        assert!((t.position[0] - 100.0).abs() < 50.0);
        assert!((t.position[1] - 200.0).abs() < 50.0);
    }

    #[test]
    fn test_viz_track_serialization() {
        let track = VizTrack {
            id: 42,
            position: [1.0, 2.0, 3.0],
            velocity: [0.1, 0.2, 0.3],
            covariance_diag: [0.5; 6],
            is_confirmed: true,
            class_label: Some("Aircraft".to_string()),
        };

        let json = serde_json::to_string(&track).unwrap();
        let deserialized: VizTrack = serde_json::from_str(&json).unwrap();

        assert_eq!(track, deserialized);
    }
}
