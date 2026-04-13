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

/// Summary metrics for a recording, useful for analysis without replaying.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordingSummary {
    /// Total number of frames.
    pub frame_count: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// Maximum number of simultaneous tracks.
    pub max_track_count: usize,
    /// Total unique track IDs seen across all frames.
    pub unique_track_ids: usize,
    /// Total detections across all frames.
    pub total_detections: usize,
}

impl Recording {
    /// Compute a summary of this recording.
    pub fn summary(&self) -> RecordingSummary {
        let mut unique_ids = std::collections::HashSet::new();
        let mut max_tracks = 0usize;
        let mut total_dets = 0usize;

        for frame in &self.frames {
            max_tracks = max_tracks.max(frame.tracks.len());
            total_dets += frame.detections.len();
            for t in &frame.tracks {
                unique_ids.insert(t.id);
            }
        }

        RecordingSummary {
            frame_count: self.frames.len(),
            duration: self.duration(),
            max_track_count: max_tracks,
            unique_track_ids: unique_ids.len(),
            total_detections: total_dets,
        }
    }

    /// Get a sub-range of frames by index (clamped to bounds).
    pub fn slice(&self, start: usize, end: usize) -> Vec<&VizFrame> {
        let start = start.min(self.frames.len());
        let end = end.min(self.frames.len());
        self.frames[start..end].iter().collect()
    }

    /// Find the frame closest to the given timestamp.
    pub fn frame_at_time(&self, timestamp: f64) -> Option<&VizFrame> {
        self.frames.iter().min_by(|a, b| {
            (a.timestamp - timestamp)
                .abs()
                .partial_cmp(&(b.timestamp - timestamp).abs())
                .unwrap()
        })
    }
}

impl VizFrame {
    /// Create a frame from raw data without a tracker reference.
    ///
    /// Useful for building frames from external data sources or for testing.
    pub fn from_raw(
        timestamp: f64,
        tracks: Vec<VizTrack>,
        detections: Vec<VizDetection>,
        ground_truth: Vec<VizGroundTruth>,
    ) -> Self {
        Self {
            timestamp,
            tracks,
            detections,
            ground_truth,
        }
    }

    /// Number of confirmed tracks in this frame.
    pub fn confirmed_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_confirmed).count()
    }

    /// Number of tentative (unconfirmed) tracks in this frame.
    pub fn tentative_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| !t.is_confirmed).count()
    }

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
            .map(convert_track_to_viz)
            .collect();

        let viz_detections = detections.iter().map(convert_detection_to_viz).collect();

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

/// Extract position from a 6D state vector `[x, vx, y, vy, z, vz]`.
fn extract_position(state: &nalgebra::DVector<f64>) -> [f64; 3] {
    let n = state.len();
    [
        if n > 0 { state[0] } else { 0.0 },
        if n > 2 { state[2] } else { 0.0 },
        if n > 4 { state[4] } else { 0.0 },
    ]
}

/// Extract velocity from a 6D state vector `[x, vx, y, vy, z, vz]`.
fn extract_velocity(state: &nalgebra::DVector<f64>) -> [f64; 3] {
    let n = state.len();
    [
        if n > 1 { state[1] } else { 0.0 },
        if n > 3 { state[3] } else { 0.0 },
        if n > 5 { state[5] } else { 0.0 },
    ]
}

/// Extract the leading diagonal entries from a covariance matrix.
fn extract_covariance_diag(cov: &nalgebra::DMatrix<f64>) -> [f64; 6] {
    let cov_rows = cov.nrows();
    let mut diag = [0.0; 6];
    for i in 0..6.min(cov_rows) {
        diag[i] = cov[(i, i)];
    }
    diag
}

/// Convert a tracker track to a visualization track snapshot.
fn convert_track_to_viz(t: &thresh_tracker::track::Track) -> VizTrack {
    let class_label = match t.class {
        thresh_core::track::TargetClass::Unknown => None,
        other => Some(format!("{other:?}")),
    };

    VizTrack {
        id: t.id.0,
        position: extract_position(&t.state),
        velocity: extract_velocity(&t.state),
        covariance_diag: extract_covariance_diag(&t.covariance),
        is_confirmed: t.lifecycle == CoreTrackState::Confirmed,
        class_label,
    }
}

/// Convert a detection measurement vector to a visualization detection.
fn convert_detection_to_viz(d: &nalgebra::DVector<f64>) -> VizDetection {
    let pos = [
        if !d.is_empty() { d[0] } else { 0.0 },
        if d.len() > 1 { d[1] } else { 0.0 },
        if d.len() > 2 { d[2] } else { 0.0 },
    ];
    VizDetection {
        position: pos,
        sensor_id: 0,
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
    fn test_viz_frame_from_raw() {
        let frame = VizFrame::from_raw(
            1.5,
            vec![VizTrack {
                id: 1,
                position: [1.0, 2.0, 3.0],
                velocity: [0.0; 3],
                covariance_diag: [1.0; 6],
                is_confirmed: true,
                class_label: None,
            }],
            vec![VizDetection {
                position: [1.0, 2.0, 3.0],
                sensor_id: 0,
            }],
            Vec::new(),
        );
        assert_eq!(frame.timestamp, 1.5);
        assert_eq!(frame.tracks.len(), 1);
        assert_eq!(frame.detections.len(), 1);
        assert!(frame.ground_truth.is_empty());
    }

    #[test]
    fn test_viz_frame_track_counts() {
        let frame = VizFrame::from_raw(
            0.0,
            vec![
                VizTrack {
                    id: 1,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
                VizTrack {
                    id: 2,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: false,
                    class_label: None,
                },
                VizTrack {
                    id: 3,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(frame.confirmed_track_count(), 2);
        assert_eq!(frame.tentative_track_count(), 1);
    }

    #[test]
    fn test_recording_summary() {
        let mut rec = Recording::new("summary-test");
        // Frame 0: 2 tracks, 3 detections
        rec.push_frame(VizFrame::from_raw(
            0.0,
            vec![
                VizTrack {
                    id: 1,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
                VizTrack {
                    id: 2,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
            ],
            vec![
                VizDetection {
                    position: [0.0; 3],
                    sensor_id: 0,
                },
                VizDetection {
                    position: [1.0; 3],
                    sensor_id: 0,
                },
                VizDetection {
                    position: [2.0; 3],
                    sensor_id: 0,
                },
            ],
            Vec::new(),
        ));
        // Frame 1: 3 tracks (one new), 1 detection
        rec.push_frame(VizFrame::from_raw(
            1.0,
            vec![
                VizTrack {
                    id: 1,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
                VizTrack {
                    id: 2,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
                VizTrack {
                    id: 3,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                },
            ],
            vec![VizDetection {
                position: [0.0; 3],
                sensor_id: 0,
            }],
            Vec::new(),
        ));

        let summary = rec.summary();
        assert_eq!(summary.frame_count, 2);
        assert!((summary.duration - 1.0).abs() < 1e-10);
        assert_eq!(summary.max_track_count, 3);
        assert_eq!(summary.unique_track_ids, 3);
        assert_eq!(summary.total_detections, 4);
    }

    #[test]
    fn test_recording_slice() {
        let mut rec = Recording::new("slice-test");
        for i in 0..10 {
            rec.push_frame(make_frame(i as f64));
        }
        let slice = rec.slice(2, 5);
        assert_eq!(slice.len(), 3);
        assert!((slice[0].timestamp - 2.0).abs() < 1e-10);
        assert!((slice[2].timestamp - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_recording_slice_clamped() {
        let mut rec = Recording::new("slice-clamp");
        for i in 0..5 {
            rec.push_frame(make_frame(i as f64));
        }
        let slice = rec.slice(3, 100);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_recording_frame_at_time() {
        let mut rec = Recording::new("frame-at-time");
        for i in 0..5 {
            rec.push_frame(make_frame(i as f64 * 0.5));
        }
        let frame = rec.frame_at_time(0.7).unwrap();
        // Closest to 0.7 is 0.5 (distance 0.2) vs 1.0 (distance 0.3)
        assert!((frame.timestamp - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_recording_summary_serialization() {
        let summary = RecordingSummary {
            frame_count: 100,
            duration: 10.0,
            max_track_count: 5,
            unique_track_ids: 8,
            total_detections: 300,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let loaded: RecordingSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(summary, loaded);
    }

    #[test]
    fn generate_sample_recording() {
        let mut rec = Recording::new("sample");
        for i in 0..50 {
            let t = i as f64 * 0.5;
            rec.push_frame(VizFrame {
                timestamp: t,
                tracks: vec![
                    VizTrack {
                        id: 1,
                        position: [100.0 + t * 20.0, 200.0 + t * 5.0, 0.0],
                        velocity: [20.0, 5.0, 0.0],
                        covariance_diag: [1.0; 6],
                        is_confirmed: true,
                        class_label: Some("aircraft".into()),
                    },
                    VizTrack {
                        id: 2,
                        position: [500.0 - t * 15.0, 300.0 + t * 10.0, 0.0],
                        velocity: [-15.0, 10.0, 0.0],
                        covariance_diag: [1.0; 6],
                        is_confirmed: true,
                        class_label: None,
                    },
                ],
                detections: vec![
                    VizDetection {
                        position: [100.0 + t * 20.0 + 5.0, 200.0 + t * 5.0 - 3.0, 0.0],
                        sensor_id: 0,
                    },
                    VizDetection {
                        position: [500.0 - t * 15.0 + 8.0, 300.0 + t * 10.0 + 2.0, 0.0],
                        sensor_id: 0,
                    },
                ],
                ground_truth: vec![
                    VizGroundTruth {
                        id: 1,
                        position: [100.0 + t * 20.0, 200.0 + t * 5.0, 0.0],
                    },
                    VizGroundTruth {
                        id: 2,
                        position: [500.0 - t * 15.0, 300.0 + t * 10.0, 0.0],
                    },
                ],
            });
        }

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample_recording.json");
        rec.save_json(path.to_str().unwrap()).unwrap();

        // Verify round-trip.
        let loaded = Recording::load_json(path.to_str().unwrap()).unwrap();
        assert_eq!(rec, loaded);
        assert_eq!(loaded.frame_count(), 50);
        assert_eq!(loaded.frames[0].tracks.len(), 2);
        assert_eq!(loaded.frames[0].detections.len(), 2);
        assert_eq!(loaded.frames[0].ground_truth.len(), 2);
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
