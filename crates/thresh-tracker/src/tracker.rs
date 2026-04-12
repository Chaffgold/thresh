//! Main tracker loop: predict -> associate -> update -> lifecycle.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use thresh_association::hungarian::hungarian_assignment;
use thresh_core::track::{TargetClass, TrackState};
use thresh_filter::imm::{ImmConfig, ImmFilter};
use thresh_filter::kf::KalmanFilter;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::{LinearModel, MotionModel};

use crate::cost_matrix::{alive_indices, build_track_cost_matrix, predict_all};

use crate::heads::HeadRegistry;
use crate::lifecycle::update_lifecycle;
use crate::track::Track;

/// Main multi-object tracker.
pub struct MultiObjectTracker {
    /// Active tracks.
    pub tracks: Vec<Track>,
    /// Class-specific head registry.
    pub heads: HeadRegistry,
    /// Observation matrix H (maps state to measurement).
    pub observation_matrix: DMatrix<f64>,
    /// Measurement noise R.
    pub measurement_noise: DMatrix<f64>,
    /// Gating threshold (chi-squared).
    pub gate_threshold: f64,
    /// Factory function that produces a fresh `ImmConfig` for new tracks.
    /// `None` means single-model KF mode.
    imm_config_factory: Option<Box<dyn Fn() -> ImmConfig>>,
    /// Per-track IMM filters, keyed by a stable track key assigned at birth.
    imm_filters: HashMap<usize, ImmFilter>,
    /// Next track key for the IMM filter map.
    next_imm_key: usize,
}

impl MultiObjectTracker {
    /// Create a tracker for position-only observations of CV-model tracks.
    ///
    /// Observes [x, y, z] from state [x, vx, y, vy, z, vz].
    pub fn new_cv_position(measurement_noise_sigma: f64, gate_threshold: f64) -> Self {
        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * (measurement_noise_sigma * measurement_noise_sigma);

        Self {
            tracks: Vec::new(),
            heads: HeadRegistry::default(),
            observation_matrix: h,
            measurement_noise: r,
            gate_threshold,
            imm_config_factory: None,
            imm_filters: HashMap::new(),
            next_imm_key: 0,
        }
    }

    /// Create a tracker that uses an Interacting Multiple Model filter
    /// for position-only observations of 6D state `[x, vx, y, vy, z, vz]`.
    ///
    /// `config_factory` is called once per new track to produce a fresh
    /// `ImmConfig`. Each track maintains its own `ImmFilter` bank.
    /// Association and lifecycle logic are shared with the single-model path.
    pub fn new_imm_position(
        config_factory: impl Fn() -> ImmConfig + 'static,
        measurement_noise_sigma: f64,
        gate_threshold: f64,
    ) -> Self {
        // Validate once to fail early on bad config.
        config_factory()
            .validate()
            .expect("Invalid ImmConfig from factory");

        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * (measurement_noise_sigma * measurement_noise_sigma);

        Self {
            tracks: Vec::new(),
            heads: HeadRegistry::default(),
            observation_matrix: h,
            measurement_noise: r,
            gate_threshold,
            imm_config_factory: Some(Box::new(config_factory)),
            imm_filters: HashMap::new(),
            next_imm_key: 0,
        }
    }

    /// Run one tracking cycle: predict all tracks, associate with detections, update.
    pub fn step(&mut self, detections: &[DVector<f64>], dt: f64) {
        let is_imm = self.imm_config_factory.is_some();

        // 1. Predict all tracks
        if is_imm {
            // IMM predict: interaction + predict + combine for each track
            for track in &mut self.tracks {
                if !track.is_alive() {
                    continue;
                }
                if let Some(key) = track.imm_key
                    && let Some(imm) = self.imm_filters.get_mut(&key)
                {
                    let (state, cov) = imm.predict(dt);
                    track.state = state;
                    track.covariance = cov;
                }
            }
        } else {
            let model = ConstantVelocity::new(5.0);
            let f = model.transition_matrix(dt);
            let q = model.process_noise(dt);
            predict_all(&mut self.tracks, &f, &q);
        }

        // 2. Build Mahalanobis cost matrix
        let alive = alive_indices(&self.tracks);
        let h = &self.observation_matrix;
        let r = &self.measurement_noise;
        let cost_matrix =
            build_track_cost_matrix(&self.tracks, &alive, h, r, detections, self.gate_threshold);

        // 3. Hungarian assignment
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        // 4. Update matched tracks
        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; detections.len()];

        for &(ai, dj) in &result.matches {
            associated_tracks[ai] = true;
            associated_dets[dj] = true;

            let ti = alive[ai];

            if is_imm {
                // IMM update
                if let Some(key) = self.tracks[ti].imm_key
                    && let Some(imm) = self.imm_filters.get_mut(&key)
                {
                    let result = imm.update_with_measurement(&detections[dj], h, r);
                    self.tracks[ti].state = result.state;
                    self.tracks[ti].covariance = result.covariance;
                    self.tracks[ti].dominant_mode = Some(result.dominant_mode);
                    self.tracks[ti].mode_probabilities = Some(result.mode_probabilities);
                }
            } else {
                // KF update
                let track = &self.tracks[ti];
                let mut kf = KalmanFilter::new(track.state.clone(), track.covariance.clone());
                kf.update(&detections[dj], h, r);
                self.tracks[ti].state = kf.x;
                self.tracks[ti].covariance = kf.p;
            }
        }

        // 5. Lifecycle updates
        let heads = self.heads.clone();
        for (ai, &ti) in alive.iter().enumerate() {
            let was_associated = associated_tracks[ai];
            let head = heads.get(self.tracks[ti].class);
            update_lifecycle(
                &mut self.tracks[ti],
                was_associated,
                &head.confirmation,
                &head.deletion,
            );
        }

        // 6. Birth new tracks from unassigned detections
        for (dj, det) in detections.iter().enumerate() {
            if !associated_dets[dj] {
                self.birth_track(det, TargetClass::Unknown);
            }
        }

        // 7. Remove deleted tracks — also clean up IMM filters for deleted tracks
        let imm_filters = &mut self.imm_filters;
        self.tracks.retain(|t| {
            if t.lifecycle == TrackState::Deleted {
                if let Some(key) = t.imm_key {
                    imm_filters.remove(&key);
                }
                false
            } else {
                true
            }
        });
    }

    /// Create a new track from a detection.
    fn birth_track(&mut self, detection: &DVector<f64>, class: TargetClass) {
        let head = self.heads.get(class);
        let mut state = DVector::zeros(head.state_dim);
        // Initialize position from detection, velocity from zero
        let h = &self.observation_matrix;
        let m_dim = detection.len();
        for i in 0..m_dim.min(head.state_dim) {
            // Map measurement indices to state indices via H
            for j in 0..head.state_dim {
                if h[(i, j)].abs() > 0.5 {
                    state[j] = detection[i];
                }
            }
        }

        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&head.initial_covariance));
        let mut track = Track::new(state.clone(), cov.clone(), class);

        // If in IMM mode, create an ImmFilter for this track.
        if let Some(factory) = &self.imm_config_factory {
            let config = factory();
            let imm = ImmFilter::new(config, &state, &cov);

            let key = self.next_imm_key;
            self.next_imm_key += 1;
            self.imm_filters.insert(key, imm);
            track.imm_key = Some(key);
        }

        self.tracks.push(track);
    }

    /// Number of confirmed tracks.
    pub fn confirmed_count(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .count()
    }

    /// Number of alive tracks (tentative + confirmed + coasting).
    pub fn alive_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_alive()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_filter::imm::ImmConfig;

    #[test]
    fn track_birth_and_confirmation() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);

        // Frame 1: detection at (100, 200, 50)
        let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
        tracker.step(std::slice::from_ref(&det), 1.0);
        assert_eq!(tracker.alive_count(), 1);
        assert_eq!(tracker.confirmed_count(), 0); // still tentative

        // Frames 2-4: same detection (close enough)
        for _ in 0..3 {
            tracker.step(std::slice::from_ref(&det), 1.0);
        }
        assert_eq!(tracker.confirmed_count(), 1);
    }

    #[test]
    fn track_coast_and_delete() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);

        // Create and confirm a track
        let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
        for _ in 0..5 {
            tracker.step(std::slice::from_ref(&det), 1.0);
        }
        assert_eq!(tracker.confirmed_count(), 1);

        // No detections for many frames -> coast then delete
        for _ in 0..10 {
            tracker.step(&[], 1.0);
        }
        assert_eq!(tracker.alive_count(), 0);
    }

    #[test]
    fn track_identity_preserved() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);

        let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
        tracker.step(std::slice::from_ref(&det), 1.0);
        let id = tracker.tracks[0].id;

        // Re-associate for several frames
        for _ in 0..5 {
            tracker.step(std::slice::from_ref(&det), 1.0);
        }
        assert_eq!(tracker.tracks[0].id, id);
    }

    #[test]
    fn no_id_collisions_many_tracks() {
        use std::collections::HashSet;
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);

        for cycle in 0..1000 {
            // Create detections far apart so each spawns a new track
            let det = DVector::from_column_slice(&[cycle as f64 * 1000.0, 0.0, 0.0]);
            tracker.step(&[det], 1.0);

            // Check no duplicate IDs within the current live set
            let mut ids = HashSet::new();
            for t in &tracker.tracks {
                assert!(ids.insert(t.id), "Duplicate TrackId in live set: {}", t.id);
            }
        }
    }

    /// 6.7 Integration test: 4-model IMM through `new_imm_position` --
    /// birth, confirm, and maintain a maneuvering track over 50 steps.
    /// Verify `dominant_mode` changes when the target maneuvers.
    #[test]
    fn imm_tracker_maneuvering_target() {
        let mut tracker = MultiObjectTracker::new_imm_position(
            || ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            10.0,
            100.0,
        );

        let dt = 1.0;
        let speed = 100.0; // m/s

        // Phase 1: straight flight for 25 steps (positive x direction)
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let mut early_dominant_modes = Vec::new();

        for step in 0..25 {
            x += speed * dt;
            let det = DVector::from_column_slice(&[x, y, 0.0]);
            tracker.step(std::slice::from_ref(&det), dt);

            if step >= 10 {
                // After sufficient convergence, record dominant mode
                if let Some(dm) = tracker.tracks[0].dominant_mode {
                    early_dominant_modes.push(dm);
                }
            }
        }

        // Track should be confirmed by now
        assert_eq!(tracker.confirmed_count(), 1, "track should be confirmed");
        assert_eq!(tracker.alive_count(), 1, "should have exactly one track");

        // Phase 2: coordinated turn for 25 steps (turning left)
        let turn_rate = 0.1; // rad/s
        let mut heading = 0.0_f64; // initially moving along +x
        let mut late_dominant_modes = Vec::new();

        for _ in 0..25 {
            heading += turn_rate * dt;
            x += speed * heading.cos() * dt;
            y += speed * heading.sin() * dt;
            let det = DVector::from_column_slice(&[x, y, 0.0]);
            tracker.step(std::slice::from_ref(&det), dt);

            if let Some(dm) = tracker.tracks[0].dominant_mode {
                late_dominant_modes.push(dm);
            }
        }

        // The track should still be alive
        assert!(
            tracker.alive_count() >= 1,
            "track should survive the maneuver"
        );

        // Dominant mode should have changed between straight and turning phases.
        // During straight flight, CV (mode 0) or CA (mode 1) should dominate.
        // During the turn, CTRV (mode 2) or CT (mode 3) should become more
        // likely (or at least the mode distribution should shift).
        let early_last = *early_dominant_modes.last().unwrap();
        let late_last = *late_dominant_modes.last().unwrap();
        // At minimum, verify that the dominant mode is populated and that
        // it has changed at some point during the turn.
        let mode_changed =
            late_dominant_modes.iter().any(|&m| m != early_last) || late_last != early_last;
        assert!(
            mode_changed,
            "dominant mode should shift during maneuver: early={early_last}, late modes={late_dominant_modes:?}"
        );
    }

    /// 6.8 Integration test: IMM covariance stays positive semi-definite
    /// over 1000 steps with mode switching (eigenvalue check).
    #[test]
    fn imm_covariance_stays_psd() {
        let mut tracker = MultiObjectTracker::new_imm_position(
            || ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            10.0,
            200.0,
        );

        let dt = 1.0;
        let speed = 50.0;
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let mut heading = 0.0_f64;

        for step in 0..1000 {
            // Switch between straight and turning every 100 steps
            let turn_rate = if (step / 100) % 2 == 0 {
                0.0 // straight
            } else {
                0.15 // turning
            };

            heading += turn_rate * dt;
            x += speed * heading.cos() * dt;
            y += speed * heading.sin() * dt;

            let det = DVector::from_column_slice(&[x, y, 0.0]);
            tracker.step(std::slice::from_ref(&det), dt);

            // Check covariance PSD for all alive tracks
            for track in &tracker.tracks {
                if !track.is_alive() {
                    continue;
                }
                let eigenvalues = track.covariance.clone().symmetric_eigen().eigenvalues;
                for (i, &ev) in eigenvalues.iter().enumerate() {
                    assert!(
                        ev > -1e-10,
                        "step {step}: eigenvalue[{i}] = {ev} is not PSD"
                    );
                }
            }
        }
    }
}
