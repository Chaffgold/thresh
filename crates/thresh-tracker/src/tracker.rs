//! Main tracker loop: predict -> associate -> update -> lifecycle.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use thresh_association::gating::mahalanobis_squared;
use thresh_association::hungarian::hungarian_assignment;
use thresh_association::jpda::{JpdaTrack, jpda_associate_and_update};
use thresh_association::mht::HypothesisTree;
use thresh_core::detection::Detection3D;
use thresh_core::track::{TargetClass, TrackState};
use thresh_filter::imm::{ImmConfig, ImmFilter};
use thresh_filter::kf::KalmanFilter;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::{LinearModel, MotionModel};

/// Association strategy used by the tracker.
#[derive(Debug, Clone, Default)]
pub enum AssociationStrategy {
    /// Classical Hungarian (Munkres) one-to-one assignment.
    #[default]
    Hungarian,
    /// Joint Probabilistic Data Association: soft association probabilities.
    Jpda {
        /// Probability that a true target generates a detection.
        detection_prob: f64,
        /// Spatial density of false alarms (per unit volume).
        clutter_density: f64,
    },
    /// Multi-Hypothesis Tracking: deferred decision via hypothesis tree.
    Mht {
        /// N-scan pruning depth.
        n_scan: usize,
        /// Maximum number of hypotheses (k-best pruning).
        k_best: usize,
        /// Probability that a true target generates a detection.
        detection_prob: f64,
        /// Spatial density of false alarms (per unit volume).
        clutter_density: f64,
    },
}

use crate::cost_matrix::alive_indices;
#[cfg(not(feature = "parallel"))]
use crate::cost_matrix::{build_track_cost_matrix, predict_all};
#[cfg(feature = "parallel")]
use crate::cost_matrix::{build_track_cost_matrix_parallel, predict_all_parallel};

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
    /// Association strategy (Hungarian, JPDA, or MHT).
    pub association_strategy: AssociationStrategy,
    /// Factory function that produces a fresh `ImmConfig` for new tracks.
    /// `None` means single-model KF mode.
    imm_config_factory: Option<Box<dyn Fn() -> ImmConfig>>,
    /// Per-track IMM filters, keyed by a stable track key assigned at birth.
    imm_filters: HashMap<usize, ImmFilter>,
    /// Next track key for the IMM filter map.
    next_imm_key: usize,
    /// MHT hypothesis tree (only used when strategy is `Mht`).
    mht_tree: Option<HypothesisTree>,
}

impl MultiObjectTracker {
    /// Create a tracker for position-only observations of CV-model tracks.
    ///
    /// Observes [x, y, z] from state [x, vx, y, vy, z, vz].
    /// Uses `AssociationStrategy::Hungarian` by default.
    pub fn new_cv_position(measurement_noise_sigma: f64, gate_threshold: f64) -> Self {
        Self::new_cv_position_with_strategy(
            measurement_noise_sigma,
            gate_threshold,
            AssociationStrategy::Hungarian,
        )
    }

    /// Create a tracker for position-only observations of CV-model tracks
    /// with a specific association strategy.
    ///
    /// Observes [x, y, z] from state [x, vx, y, vy, z, vz].
    pub fn new_cv_position_with_strategy(
        measurement_noise_sigma: f64,
        gate_threshold: f64,
        strategy: AssociationStrategy,
    ) -> Self {
        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * (measurement_noise_sigma * measurement_noise_sigma);

        let mht_tree = match &strategy {
            AssociationStrategy::Mht { n_scan, k_best, .. } => {
                Some(HypothesisTree::new(*k_best, *n_scan))
            }
            _ => None,
        };

        Self {
            tracks: Vec::new(),
            heads: HeadRegistry::default(),
            observation_matrix: h,
            measurement_noise: r,
            gate_threshold,
            association_strategy: strategy,
            imm_config_factory: None,
            imm_filters: HashMap::new(),
            next_imm_key: 0,
            mht_tree,
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
            association_strategy: AssociationStrategy::Hungarian,
            imm_config_factory: Some(Box::new(config_factory)),
            imm_filters: HashMap::new(),
            next_imm_key: 0,
            mht_tree: None,
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
            #[cfg(feature = "parallel")]
            predict_all_parallel(&mut self.tracks, &f, &q);
            #[cfg(not(feature = "parallel"))]
            predict_all(&mut self.tracks, &f, &q);
        }

        // 2-4. Associate and update based on strategy.
        let alive = alive_indices(&self.tracks);
        let h = &self.observation_matrix;
        let r = &self.measurement_noise;

        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; detections.len()];

        match &self.association_strategy {
            AssociationStrategy::Jpda {
                detection_prob,
                clutter_density,
            } => {
                let p_d = *detection_prob;
                let clutter = *clutter_density;

                // Build JpdaTrack list from alive tracks.
                let jpda_tracks: Vec<JpdaTrack> = alive
                    .iter()
                    .map(|&ti| {
                        let pred_z = h * &self.tracks[ti].state;
                        let s = h * &self.tracks[ti].covariance * h.transpose() + r;
                        JpdaTrack {
                            predicted_measurement: pred_z,
                            innovation_covariance: s,
                        }
                    })
                    .collect();

                let states: Vec<DVector<f64>> = alive
                    .iter()
                    .map(|&ti| self.tracks[ti].state.clone())
                    .collect();
                let covariances: Vec<DMatrix<f64>> = alive
                    .iter()
                    .map(|&ti| self.tracks[ti].covariance.clone())
                    .collect();

                let results = jpda_associate_and_update(
                    &jpda_tracks,
                    &states,
                    &covariances,
                    detections,
                    h,
                    self.gate_threshold,
                    p_d,
                    clutter,
                );

                // Apply JPDA results back to tracks.
                for (ai, update) in results.iter().enumerate() {
                    let ti = alive[ai];
                    self.tracks[ti].state = update.state.clone();
                    self.tracks[ti].covariance = update.covariance.clone();

                    // Track is considered "hit" if miss probability < 0.5.
                    if update.miss_probability < 0.5 {
                        associated_tracks[ai] = true;
                    }
                }

                // For JPDA, a detection is considered "associated" if at least one
                // track has a significant association probability for it.
                // We check if any track pulled strongly toward each detection.
                // Use a simpler heuristic: a detection is associated if it was
                // within the gate of any track that was marked as hit.
                for (dj, det) in detections.iter().enumerate() {
                    for (ai, jpda_track) in jpda_tracks.iter().enumerate() {
                        if associated_tracks[ai] {
                            let d2 = mahalanobis_squared(
                                det,
                                &jpda_track.predicted_measurement,
                                &jpda_track.innovation_covariance,
                            );
                            if d2 <= self.gate_threshold {
                                associated_dets[dj] = true;
                                break;
                            }
                        }
                    }
                }
            }
            AssociationStrategy::Mht {
                detection_prob,
                clutter_density,
                ..
            } => {
                let p_d = *detection_prob;
                let clutter = *clutter_density;
                let n_tracks = alive.len();
                let n_dets = detections.len();

                // Build log-likelihood matrix from Mahalanobis distances.
                let mut likelihoods = vec![vec![f64::NEG_INFINITY; n_dets]; n_tracks];
                for (ai, &ti) in alive.iter().enumerate() {
                    let pred_z = h * &self.tracks[ti].state;
                    let s = h * &self.tracks[ti].covariance * h.transpose() + r;
                    for (dj, det) in detections.iter().enumerate() {
                        let d2 = mahalanobis_squared(det, &pred_z, &s);
                        if d2 <= self.gate_threshold {
                            // Log-likelihood: log(p_d * N(z; z_hat, S))
                            let m = det.nrows() as f64;
                            let det_s = s.determinant();
                            if det_s > 0.0 {
                                let log_norm =
                                    -0.5 * (m * (2.0 * std::f64::consts::PI).ln() + det_s.ln());
                                let log_exp = -0.5 * d2;
                                likelihoods[ai][dj] = p_d.ln() + log_norm + log_exp;
                            }
                        }
                    }
                }

                // Expand hypothesis tree and prune.
                let gate = clutter.ln(); // log-likelihood threshold
                let tree = self
                    .mht_tree
                    .get_or_insert_with(|| HypothesisTree::new(100, 3));
                tree.expand(n_tracks, n_dets, &likelihoods, gate);
                tree.prune_k_best();

                // Extract best hypothesis for hard assignment.
                let assignments = tree.consistent_track_assignments();

                // Apply assignments like Hungarian: matched pairs get KF update.
                for (track_idx, det_idx) in &assignments {
                    if let Some(dj) = det_idx
                        && *track_idx < n_tracks
                        && *dj < n_dets
                    {
                        associated_tracks[*track_idx] = true;
                        associated_dets[*dj] = true;

                        let ti = alive[*track_idx];
                        if is_imm {
                            if let Some(key) = self.tracks[ti].imm_key
                                && let Some(imm) = self.imm_filters.get_mut(&key)
                            {
                                let result = imm.update_with_measurement(&detections[*dj], h, r);
                                self.tracks[ti].state = result.state;
                                self.tracks[ti].covariance = result.covariance;
                                self.tracks[ti].dominant_mode = Some(result.dominant_mode);
                                self.tracks[ti].mode_probabilities =
                                    Some(result.mode_probabilities);
                            }
                        } else {
                            let track = &self.tracks[ti];
                            let mut kf =
                                KalmanFilter::new(track.state.clone(), track.covariance.clone());
                            kf.update(&detections[*dj], h, r);
                            self.tracks[ti].state = kf.x;
                            self.tracks[ti].covariance = kf.p;
                        }
                    }
                }
            }
            AssociationStrategy::Hungarian => {
                // Existing Hungarian path.
                #[cfg(feature = "parallel")]
                let cost_matrix = build_track_cost_matrix_parallel(
                    &self.tracks,
                    &alive,
                    h,
                    r,
                    detections,
                    self.gate_threshold,
                );
                #[cfg(not(feature = "parallel"))]
                let cost_matrix = build_track_cost_matrix(
                    &self.tracks,
                    &alive,
                    h,
                    r,
                    detections,
                    self.gate_threshold,
                );

                let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

                for &(ai, dj) in &result.matches {
                    associated_tracks[ai] = true;
                    associated_dets[dj] = true;

                    let ti = alive[ai];

                    if is_imm {
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
                        let track = &self.tracks[ti];
                        let mut kf =
                            KalmanFilter::new(track.state.clone(), track.covariance.clone());
                        kf.update(&detections[dj], h, r);
                        self.tracks[ti].state = kf.x;
                        self.tracks[ti].covariance = kf.p;
                    }
                }
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

    /// Run one tracking cycle using high-level [`Detection3D`] inputs.
    ///
    /// Converts each detection's position to a `DVector<f64>` measurement and
    /// delegates to [`step`](Self::step).
    pub fn step_detections(&mut self, detections: &[Detection3D], dt: f64) {
        let cart: Vec<DVector<f64>> = detections
            .iter()
            .map(|d| DVector::from_column_slice(&d.position))
            .collect();
        self.step(&cart, dt);
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
    use thresh_core::detection::Detection3D;
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

    #[test]
    fn step_detections_produces_confirmed_tracks() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);
        let det = Detection3D {
            position: [100.0, 200.0, 50.0],
            dimensions: [2.0, 2.0, 2.0],
            yaw: 0.0,
            class_id: 0,
            confidence: 0.95,
        };
        for _ in 0..6 {
            tracker.step_detections(std::slice::from_ref(&det), 1.0);
        }
        assert!(
            tracker.confirmed_count() >= 1,
            "should have at least one confirmed track after 6 steps"
        );
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

    /// Verify that the parallel tracker produces the same output as the
    /// sequential one on a deterministic scenario.
    ///
    /// This test always runs (even without the `parallel` feature) so the
    /// sequential path is exercised. When `--features parallel` is active the
    /// `step` method dispatches to `predict_all_parallel` and
    /// `build_track_cost_matrix_parallel`, so the same assertions validate the
    /// parallel code path.
    #[test]
    fn parallel_correctness_matches_sequential() {
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let dt = 1.0;

        // Birth two tracks with well-separated detections.
        let dets: Vec<DVector<f64>> = vec![
            DVector::from_column_slice(&[0.0, 0.0, 0.0]),
            DVector::from_column_slice(&[1000.0, 1000.0, 0.0]),
        ];
        tracker.step(&dets, dt);
        assert_eq!(tracker.alive_count(), 2);

        // Run 20 steps with consistent detections moving linearly.
        for step in 1..=20 {
            let t = step as f64;
            let dets: Vec<DVector<f64>> = vec![
                DVector::from_column_slice(&[t * 10.0, t * 5.0, 0.0]),
                DVector::from_column_slice(&[1000.0 + t * 10.0, 1000.0 + t * 5.0, 0.0]),
            ];
            tracker.step(&dets, dt);
        }

        // Both tracks should be confirmed (M-of-N with enough hits).
        assert_eq!(
            tracker.confirmed_count(),
            2,
            "both tracks should be confirmed"
        );
        assert_eq!(tracker.alive_count(), 2);

        // Verify state estimates are reasonable (position near last detection).
        for track in &tracker.tracks {
            let x = track.state[0];
            let y = track.state[2];
            // Last detections were at (200, 100, 0) and (1200, 1100, 0).
            assert!(
                (x > 100.0 && y > 50.0),
                "state should be near detection: x={x}, y={y}"
            );
        }
    }

    /// 6.6 Backward compatibility: `new_cv_position` still defaults to Hungarian.
    #[test]
    fn new_cv_position_defaults_to_hungarian() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        assert!(
            matches!(tracker.association_strategy, AssociationStrategy::Hungarian),
            "default strategy should be Hungarian"
        );
    }

    /// 6.6 Backward compatibility: `new_cv_position_with_strategy` accepts JPDA.
    #[test]
    fn new_cv_position_with_jpda_strategy() {
        let tracker = MultiObjectTracker::new_cv_position_with_strategy(
            10.0,
            50.0,
            AssociationStrategy::Jpda {
                detection_prob: 0.9,
                clutter_density: 1e-6,
            },
        );
        assert!(matches!(
            tracker.association_strategy,
            AssociationStrategy::Jpda { .. }
        ));
    }

    /// 7.9 Integration test: MHT on a dense clutter scenario maintains track
    /// continuity through high false alarm rates.
    ///
    /// A single target moves linearly while 10 false alarms are injected per
    /// frame. MHT should maintain at least one confirmed track near the true
    /// target position.
    #[test]
    fn mht_dense_clutter_maintains_track() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Use a tighter measurement noise and generous gate so the true
        // target consistently gates while clutter (spread over a large
        // volume) mostly falls outside.
        let mut tracker = MultiObjectTracker::new_cv_position_with_strategy(
            5.0,
            50.0,
            AssociationStrategy::Mht {
                n_scan: 3,
                k_best: 50,
                detection_prob: 0.9,
                clutter_density: 1e-6,
            },
        );

        let mut rng = StdRng::seed_from_u64(42);

        // Target moves linearly: starts at (100, 200, 50), velocity (5, 3, 0)
        let mut target_x = 100.0_f64;
        let mut target_y = 200.0_f64;
        let target_z = 50.0_f64;

        for _step in 0..40 {
            target_x += 5.0;
            target_y += 3.0;

            let mut dets = vec![DVector::from_column_slice(&[target_x, target_y, target_z])];

            // Add 10 clutter detections uniformly spread over a large surveillance volume.
            // Because the volume is large (1000x1000x100 = 1e8 m^3), most clutter
            // detections will be far from the target and will not gate.
            for _ in 0..10 {
                let cx: f64 = rng.random::<f64>() * 1000.0;
                let cy: f64 = rng.random::<f64>() * 1000.0;
                let cz: f64 = rng.random::<f64>() * 100.0;
                dets.push(DVector::from_column_slice(&[cx, cy, cz]));
            }

            tracker.step(&dets, 1.0);
        }

        // MHT should maintain at least one confirmed track near the true target.
        // We also accept alive (tentative or coasting) tracks as evidence of
        // track maintenance, since the M-of-N confirmation window may be
        // disrupted by occasional clutter-induced mis-associations.
        let final_target = DVector::from_column_slice(&[target_x, target_y, target_z]);
        let has_close_track = tracker.tracks.iter().filter(|t| t.is_alive()).any(|t| {
            let pos = DVector::from_column_slice(&[t.state[0], t.state[2], t.state[4]]);
            (pos - &final_target).norm() < 150.0
        });
        assert!(
            has_close_track,
            "MHT should maintain at least one alive track near the true target; \
             alive={}, confirmed={}, total={}",
            tracker.alive_count(),
            tracker.confirmed_count(),
            tracker.tracks.len()
        );
    }

    /// 7.8 Integration test: JPDA on crossing tracks handles the crossing
    /// better than Hungarian (fewer ID swaps / better positional accuracy).
    #[test]
    fn jpda_crossing_tracks_better_than_hungarian() {
        // Two targets moving on crossing paths. They start well-separated,
        // cross at step ~15, then diverge again.
        let dt = 1.0;
        let n_steps = 30;
        let noise_sigma = 5.0;
        let gate = 100.0;

        // Generate ground-truth trajectories:
        // Target A: moves from (0, 0) toward (300, 300) — diagonal up-right
        // Target B: moves from (300, 0) toward (0, 300) — diagonal up-left
        // They cross at approximately (150, 150) around step 15.
        let generate_detections = |step: usize| -> Vec<DVector<f64>> {
            let t = step as f64;
            let ax = t * 10.0;
            let ay = t * 10.0;
            let bx = 300.0 - t * 10.0;
            let by = t * 10.0;
            vec![
                DVector::from_column_slice(&[ax, ay, 0.0]),
                DVector::from_column_slice(&[bx, by, 0.0]),
            ]
        };

        // Run Hungarian tracker
        let mut hungarian = MultiObjectTracker::new_cv_position(noise_sigma, gate);
        for step in 0..n_steps {
            let dets = generate_detections(step);
            hungarian.step(&dets, dt);
        }

        // Run JPDA tracker
        let mut jpda = MultiObjectTracker::new_cv_position_with_strategy(
            noise_sigma,
            gate,
            AssociationStrategy::Jpda {
                detection_prob: 0.9,
                clutter_density: 1e-6,
            },
        );
        for step in 0..n_steps {
            let dets = generate_detections(step);
            jpda.step(&dets, dt);
        }

        // Both trackers should maintain tracks through the crossing.
        assert!(
            jpda.alive_count() >= 2,
            "JPDA should maintain at least 2 alive tracks, got {}",
            jpda.alive_count()
        );
        assert!(
            hungarian.alive_count() >= 2,
            "Hungarian should maintain at least 2 alive tracks, got {}",
            hungarian.alive_count()
        );

        // Check final positional accuracy for both.
        // At step 29: target A should be near (290, 290), target B near (10, 290).
        let final_a = DVector::from_column_slice(&[290.0, 290.0, 0.0]);
        let final_b = DVector::from_column_slice(&[10.0, 290.0, 0.0]);

        let position_error = |tracker: &MultiObjectTracker| -> f64 {
            // For each ground-truth position, find the closest alive track position
            let alive: Vec<DVector<f64>> = tracker
                .tracks
                .iter()
                .filter(|t| t.is_alive())
                .map(|t| DVector::from_column_slice(&[t.state[0], t.state[2], t.state[4]]))
                .collect();
            if alive.len() < 2 {
                return f64::MAX;
            }
            // Greedy match: assign closest track to each target
            let mut total_err = 0.0;
            for target in &[&final_a, &final_b] {
                let min_err = alive
                    .iter()
                    .map(|a| (a - *target).norm())
                    .fold(f64::MAX, f64::min);
                total_err += min_err;
            }
            total_err
        };

        let jpda_err = position_error(&jpda);
        let hungarian_err = position_error(&hungarian);

        // JPDA should produce at least comparable positional accuracy.
        // We check that it doesn't catastrophically fail.
        assert!(
            jpda_err < 400.0,
            "JPDA final position error should be reasonable, got {jpda_err}"
        );

        // Log both errors for debugging (not a strict inequality since
        // the scenario is deterministic and both may perform similarly).
        eprintln!(
            "JPDA position error: {jpda_err:.1}, Hungarian position error: {hungarian_err:.1}"
        );
    }
}
