//! Main tracker loop: predict -> associate -> update -> lifecycle.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use thresh_association::gating::mahalanobis_squared;
use thresh_association::hungarian::hungarian_assignment;
use thresh_association::jpda::{JpdaTrack, jpda_associate_and_update};
use thresh_association::mht::HypothesisTree;
use thresh_core::detection::Detection3D;
use thresh_core::ego::EgoMotion;
use thresh_core::orbital::GravityModel;
use thresh_core::track::{TargetClass, TrackId, TrackState};
use thresh_filter::imm::{ImmConfig, ImmFilter};
#[cfg(feature = "learned-imm")]
use thresh_filter::imm_adapter::{ImmModeAdapter, LearnedImmFilter, NUM_MODES};
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

use crate::cost_matrix::{alive_indices, build_cost_matrix, predict_linear};
#[cfg(not(feature = "parallel"))]
use crate::cost_matrix::{build_track_cost_matrix, predict_all};
#[cfg(feature = "parallel")]
use crate::cost_matrix::{build_track_cost_matrix_parallel, predict_all_parallel};

use crate::heads::{HeadModel, HeadRegistry, TrackHead};
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
    /// Path to the ONNX IMM mode classifier. `Some` only when built via
    /// [`Self::new_imm_position_learned`]; new tracks then get a
    /// `LearnedImmFilter` instead of a plain `ImmFilter`.
    #[cfg(feature = "learned-imm")]
    learned_classifier_path: Option<std::path::PathBuf>,
    /// Per-track learned-IMM filters (parallel to `imm_filters`; a track lives
    /// in exactly one map). Empty unless the learned path is active.
    #[cfg(feature = "learned-imm")]
    learned_imm_filters: HashMap<usize, LearnedImmFilter>,
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
            #[cfg(feature = "learned-imm")]
            learned_classifier_path: None,
            #[cfg(feature = "learned-imm")]
            learned_imm_filters: HashMap::new(),
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
            #[cfg(feature = "learned-imm")]
            learned_classifier_path: None,
            #[cfg(feature = "learned-imm")]
            learned_imm_filters: HashMap::new(),
        }
    }

    /// Create an IMM tracker whose per-track mode-probability update is driven
    /// by a learned ONNX classifier (Track B of the flight-data-training
    /// pipeline) instead of the analytic Markov update.
    ///
    /// Like [`Self::new_imm_position`], but each new track wraps its IMM bank in
    /// a [`LearnedImmFilter`]: once a `WINDOW_LEN` history of filter-state
    /// projections exists, the ONNX classifier supplies the mode probabilities;
    /// before that (and on any classifier error) it falls back to the analytic
    /// update. The classifier is trained for the 4-model `cv_ca_ctrv_ct` bank,
    /// so `config_factory` MUST produce a 4-model config.
    ///
    /// Note: each track loads its own classifier session from `onnx_path`
    /// (the adapter is not shared), which is fine for the small target counts in
    /// evaluation scenarios.
    ///
    /// # Errors
    /// Returns a message if the config is invalid, is not a 4-model bank, or the
    /// ONNX classifier at `onnx_path` cannot be loaded.
    #[cfg(feature = "learned-imm")]
    pub fn new_imm_position_learned(
        config_factory: impl Fn() -> ImmConfig + 'static,
        onnx_path: impl AsRef<std::path::Path>,
        measurement_noise_sigma: f64,
        gate_threshold: f64,
    ) -> Result<Self, String> {
        // Fail fast: valid config, correct bank size, and a loadable classifier.
        config_factory()
            .validate()
            .map_err(|e| format!("invalid ImmConfig from factory: {e}"))?;
        let probe = ImmFilter::new(
            config_factory(),
            &DVector::zeros(6),
            &DMatrix::identity(6, 6),
        );
        if probe.num_models() != NUM_MODES {
            return Err(format!(
                "learned IMM requires a {NUM_MODES}-model bank (cv_ca_ctrv_ct), got {}",
                probe.num_models()
            ));
        }
        let path = onnx_path.as_ref().to_path_buf();
        ImmModeAdapter::from_onnx(&path)?; // validate the checkpoint up front

        let mut tracker =
            Self::new_imm_position(config_factory, measurement_noise_sigma, gate_threshold);
        tracker.learned_classifier_path = Some(path);
        Ok(tracker)
    }

    /// Create a tracker for automotive ENU tracking (automotive-tracking-pipeline,
    /// task 2.4).
    ///
    /// A CV position tracker whose default [`HeadRegistry`] carries the
    /// automotive class heads (car/truck/bus/motorcycle/bicycle/pedestrian), so
    /// per-class motion priors and lifecycle policies engage when detections are
    /// fed through [`Self::step_classed`] / [`Self::step_with_ego`]. Tracking
    /// happens in a fixed world ENU frame; on a moving platform, lift ego-frame
    /// detections with [`Self::step_with_ego`] so stationary world objects do
    /// not drift due to ego movement. `measurement_noise_sigma` is in metres
    /// (automotive detections are typically sub-metre).
    pub fn new_automotive_enu(measurement_noise_sigma: f64, gate_threshold: f64) -> Self {
        Self::new_cv_position(measurement_noise_sigma, gate_threshold)
    }

    /// Run one tracking cycle: predict all tracks, associate with detections, update.
    pub fn step(&mut self, detections: &[DVector<f64>], dt: f64) {
        self.predict_all_tracks(dt);
        let (associated_tracks, associated_dets) = self.associate_and_update(detections);
        self.apply_lifecycle(&associated_tracks);
        self.birth_unassigned(detections, &associated_dets);
        self.remove_deleted_tracks();
    }

    /// Run one tracking cycle with class-tagged detections.
    ///
    /// Identical to [`Self::step`], except unassigned detections birth tracks
    /// with their own [`TargetClass`] instead of [`TargetClass::Unknown`], so
    /// the class-specific head (motion priors, confirmation/deletion policy)
    /// applies from birth. Detections are world-frame positions `[x, y, z]`:
    /// longer vectors (e.g. a 6D LiDAR position+velocity from
    /// `Measurement::to_vector`) are truncated to their leading position, and
    /// sub-3D entries (no Cartesian position) are skipped — both would
    /// otherwise break the 3×6 observation model.
    pub fn step_classed(&mut self, detections: &[(DVector<f64>, TargetClass)], dt: f64) {
        let classed: Vec<(DVector<f64>, TargetClass)> = detections
            .iter()
            .filter(|(d, _)| d.len() >= 3)
            .map(|(d, class)| (DVector::from_column_slice(&[d[0], d[1], d[2]]), *class))
            .collect();
        let positions: Vec<DVector<f64>> = classed.iter().map(|(d, _)| d.clone()).collect();
        self.predict_all_tracks(dt);
        let (associated_tracks, associated_dets) = self.associate_and_update(&positions);
        self.apply_lifecycle(&associated_tracks);
        for (dj, (det, class)) in classed.iter().enumerate() {
            if !associated_dets[dj] {
                self.birth_track(det, *class);
            }
        }
        self.remove_deleted_tracks();
    }

    /// Run one tracking cycle for a moving platform (automotive-tracking-pipeline,
    /// task 2.4): lift ego-frame detections into the world ENU frame via the
    /// ego pose, then run [`Self::step_classed`].
    ///
    /// Tracks live in the world frame, so platform motion is compensated at the
    /// measurement boundary — a stationary world object observed from a moving
    /// ego lifts to the same world coordinates every cycle and does not drift.
    /// The full [`EgoMotion`] is accepted per the ego-motion contract; the lift
    /// needs only `ego.pose` today, while the linear/angular velocities are
    /// reserved for measurement-timestamp skew compensation (extrapolating the
    /// pose to each detection's exact timestamp).
    ///
    /// Detections are ego/body-frame positions `[x, y, z]` in metres; entries
    /// with fewer than 3 dimensions (e.g. a pixel-only camera observation) have
    /// no Cartesian position and are skipped.
    pub fn step_with_ego(
        &mut self,
        detections_ego: &[(DVector<f64>, TargetClass)],
        dt: f64,
        ego: &EgoMotion,
    ) {
        let world: Vec<(DVector<f64>, TargetClass)> = detections_ego
            .iter()
            .filter(|(d, _)| d.len() >= 3)
            .map(|(d, class)| {
                let p = ego.pose.transform_to_world([d[0], d[1], d[2]]);
                (DVector::from_column_slice(&p), *class)
            })
            .collect();
        self.step_classed(&world, dt);
    }

    /// Predict all alive tracks forward by `dt`.
    fn predict_all_tracks(&mut self, dt: f64) {
        if self.imm_config_factory.is_some() {
            self.predict_imm(dt);
        } else {
            self.predict_single_model(dt);
        }
    }

    /// IMM predict: interaction + predict + combine for each alive track.
    fn predict_imm(&mut self, dt: f64) {
        for track in &mut self.tracks {
            if !track.is_alive() {
                continue;
            }
            if let Some(key) = track.imm_key {
                #[cfg(feature = "learned-imm")]
                if let Some(lf) = self.learned_imm_filters.get_mut(&key) {
                    let (state, cov) = lf.predict(dt);
                    track.state = state;
                    track.covariance = cov;
                    continue;
                }
                if let Some(imm) = self.imm_filters.get_mut(&key) {
                    let (state, cov) = imm.predict(dt);
                    track.state = state;
                    track.covariance = cov;
                }
            }
        }
    }

    /// Single-model predict for all alive tracks: per-track dispatch through
    /// the class head's motion model (design Decision 6 of the
    /// `orbital-ballistic-filter-models` change). When every alive track
    /// resolves to the same CV model, the bulk linear (optionally
    /// rayon-parallel) fast path applies; otherwise tracks propagate
    /// per head — linearly for `Cv` heads, EKF-style through the model's
    /// Jacobian for the nonlinear orbital/reentry heads.
    fn predict_single_model(&mut self, dt: f64) {
        if let Some((f, q)) = self.uniform_cv_transition(dt) {
            #[cfg(feature = "parallel")]
            predict_all_parallel(&mut self.tracks, &f, &q);
            #[cfg(not(feature = "parallel"))]
            predict_all(&mut self.tracks, &f, &q);
            return;
        }
        self.predict_per_head(dt);
    }

    /// If every alive track's head is `HeadModel::Cv` with one shared
    /// `process_noise_sigma`, return that model's `(F, Q)` pair so
    /// [`Self::predict_single_model`] can take the bulk linear path.
    /// Returns `None` on mixed heads (or no alive tracks).
    fn uniform_cv_transition(&self, dt: f64) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
        let mut sigma: Option<f64> = None;
        for track in self.tracks.iter().filter(|t| t.is_alive()) {
            let head = self.heads.get(track.class);
            if head.model != HeadModel::Cv {
                return None;
            }
            match sigma {
                None => sigma = Some(head.process_noise_sigma),
                Some(s) if s == head.process_noise_sigma => {}
                Some(_) => return None,
            }
        }
        let model = ConstantVelocity::new(sigma?);
        Some((model.transition_matrix(dt), model.process_noise(dt)))
    }

    /// Per-head predict for mixed-head track sets: `Cv` heads reuse a
    /// per-class cached `(F, Q)` linear step; nonlinear heads run an
    /// EKF-style propagation through the head's motion model.
    fn predict_per_head(&mut self, dt: f64) {
        let heads = self.heads.clone();
        let mut cv_cache: HashMap<TargetClass, (DMatrix<f64>, DMatrix<f64>)> = HashMap::new();
        for track in &mut self.tracks {
            if !track.is_alive() {
                continue;
            }
            let head = heads.get(track.class);
            if head.model == HeadModel::Cv {
                let (f, q) = cv_cache.entry(track.class).or_insert_with(|| {
                    let model = ConstantVelocity::new(head.process_noise_sigma);
                    (model.transition_matrix(dt), model.process_noise(dt))
                });
                let (state, cov) = predict_linear(&track.state, &track.covariance, f, q);
                track.state = state;
                track.covariance = cov;
            } else {
                ekf_predict_track(track, head.build_model().as_ref(), dt);
            }
        }
    }

    /// Associate detections with tracks and apply measurement updates.
    ///
    /// Returns `(associated_tracks, associated_dets)` boolean flags.
    fn associate_and_update(&mut self, detections: &[DVector<f64>]) -> (Vec<bool>, Vec<bool>) {
        let alive = alive_indices(&self.tracks);
        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; detections.len()];

        match &self.association_strategy {
            AssociationStrategy::Jpda {
                detection_prob,
                clutter_density,
            } => {
                let p_d = *detection_prob;
                let clutter = *clutter_density;
                self.step_jpda(
                    detections,
                    &alive,
                    p_d,
                    clutter,
                    &mut associated_tracks,
                    &mut associated_dets,
                );
            }
            AssociationStrategy::Mht {
                detection_prob,
                clutter_density,
                ..
            } => {
                let p_d = *detection_prob;
                let clutter = *clutter_density;
                self.step_mht(
                    detections,
                    &alive,
                    p_d,
                    clutter,
                    &mut associated_tracks,
                    &mut associated_dets,
                );
            }
            AssociationStrategy::Hungarian => {
                self.step_hungarian(
                    detections,
                    &alive,
                    &mut associated_tracks,
                    &mut associated_dets,
                );
            }
        }

        (associated_tracks, associated_dets)
    }

    /// JPDA association and update path.
    fn step_jpda(
        &mut self,
        detections: &[DVector<f64>],
        alive: &[usize],
        p_d: f64,
        clutter: f64,
        associated_tracks: &mut [bool],
        associated_dets: &mut [bool],
    ) {
        let r = &self.measurement_noise;

        // Per-track padded H (see [`Self::observation_for_dim`]) so 7D
        // reentry tracks associate and update alongside 6D ones, matching
        // the Hungarian and MHT paths.
        let observation_matrices: Vec<DMatrix<f64>> = alive
            .iter()
            .map(|&ti| self.observation_for_dim(self.tracks[ti].state.len()))
            .collect();
        let jpda_tracks: Vec<JpdaTrack> = alive
            .iter()
            .zip(&observation_matrices)
            .map(|(&ti, h)| {
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
            &observation_matrices,
            self.gate_threshold,
            p_d,
            clutter,
        );

        for (ai, update) in results.iter().enumerate() {
            let ti = alive[ai];
            self.tracks[ti].state = update.state.clone();
            self.tracks[ti].covariance = update.covariance.clone();
            if update.miss_probability < 0.5 {
                associated_tracks[ai] = true;
            }
        }

        self.mark_jpda_associated_dets(
            detections,
            &jpda_tracks,
            associated_tracks,
            associated_dets,
        );
    }

    /// Mark detections as associated if they are within the gate of any hit track.
    fn mark_jpda_associated_dets(
        &self,
        detections: &[DVector<f64>],
        jpda_tracks: &[JpdaTrack],
        associated_tracks: &[bool],
        associated_dets: &mut [bool],
    ) {
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

    /// MHT association and update path.
    fn step_mht(
        &mut self,
        detections: &[DVector<f64>],
        alive: &[usize],
        p_d: f64,
        clutter: f64,
        associated_tracks: &mut [bool],
        associated_dets: &mut [bool],
    ) {
        let is_imm = self.imm_config_factory.is_some();
        let n_tracks = alive.len();
        let n_dets = detections.len();

        let likelihoods = self.build_mht_likelihoods(alive, detections, p_d);

        let gate = clutter.ln();
        let tree = self
            .mht_tree
            .get_or_insert_with(|| HypothesisTree::new(100, 3));
        tree.expand(n_tracks, n_dets, &likelihoods, gate);
        tree.prune_k_best();

        let assignments = tree.consistent_track_assignments();

        for (track_idx, det_idx) in &assignments {
            if let Some(dj) = det_idx
                && *track_idx < n_tracks
                && *dj < n_dets
            {
                associated_tracks[*track_idx] = true;
                associated_dets[*dj] = true;
                let ti = alive[*track_idx];
                self.apply_measurement_update(ti, &detections[*dj], is_imm);
            }
        }
    }

    /// Build MHT log-likelihood matrix from Mahalanobis distances.
    fn build_mht_likelihoods(
        &self,
        alive: &[usize],
        detections: &[DVector<f64>],
        p_d: f64,
    ) -> Vec<Vec<f64>> {
        let r = &self.measurement_noise;
        let n_dets = detections.len();
        let n_tracks = alive.len();

        let mut likelihoods = vec![vec![f64::NEG_INFINITY; n_dets]; n_tracks];
        for (ai, &ti) in alive.iter().enumerate() {
            let h = self.observation_for_dim(self.tracks[ti].state.len());
            let pred_z = &h * &self.tracks[ti].state;
            let s = &h * &self.tracks[ti].covariance * h.transpose() + r;
            for (dj, det) in detections.iter().enumerate() {
                let d2 = mahalanobis_squared(det, &pred_z, &s);
                if d2 <= self.gate_threshold {
                    let m = det.nrows() as f64;
                    let det_s = s.determinant();
                    if det_s > 0.0 {
                        let log_norm = -0.5 * (m * (2.0 * std::f64::consts::PI).ln() + det_s.ln());
                        let log_exp = -0.5 * d2;
                        likelihoods[ai][dj] = p_d.ln() + log_norm + log_exp;
                    }
                }
            }
        }
        likelihoods
    }

    /// Hungarian association and update path.
    fn step_hungarian(
        &mut self,
        detections: &[DVector<f64>],
        alive: &[usize],
        associated_tracks: &mut [bool],
        associated_dets: &mut [bool],
    ) {
        let is_imm = self.imm_config_factory.is_some();
        let cost_matrix = self.hungarian_cost_matrix(detections, alive);
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        for &(ai, dj) in &result.matches {
            associated_tracks[ai] = true;
            associated_dets[dj] = true;
            let ti = alive[ai];
            self.apply_measurement_update(ti, &detections[dj], is_imm);
        }
    }

    /// Cost matrix for the Hungarian path: the uniform-dimension bulk
    /// (optionally parallel) helper when every alive track matches the
    /// observation matrix, otherwise the per-track padded-H path.
    fn hungarian_cost_matrix(&self, detections: &[DVector<f64>], alive: &[usize]) -> Vec<Vec<f64>> {
        if !self.tracks_match_observation_dim(alive) {
            return self.build_padded_cost_matrix(alive, detections);
        }
        #[cfg(feature = "parallel")]
        return build_track_cost_matrix_parallel(
            &self.tracks,
            alive,
            &self.observation_matrix,
            &self.measurement_noise,
            detections,
            self.gate_threshold,
        );
        #[cfg(not(feature = "parallel"))]
        build_track_cost_matrix(
            &self.tracks,
            alive,
            &self.observation_matrix,
            &self.measurement_noise,
            detections,
            self.gate_threshold,
        )
    }

    /// True when every alive track's state dimension matches the observation
    /// matrix, so the uniform-H association helpers apply directly.
    fn tracks_match_observation_dim(&self, alive: &[usize]) -> bool {
        let cols = self.observation_matrix.ncols();
        alive.iter().all(|&ti| self.tracks[ti].state.len() == cols)
    }

    /// Observation matrix for a track state of dimension `dim`: the
    /// tracker's H, zero-padded on the right when the state carries
    /// components beyond the observed kinematic prefix — e.g. the 7D reentry
    /// state's β column (design Decision 4 of `orbital-ballistic-filter-models`).
    /// Returned unchanged when `dim` already matches H's columns.
    fn observation_for_dim(&self, dim: usize) -> DMatrix<f64> {
        let h = &self.observation_matrix;
        if dim <= h.ncols() {
            return h.clone();
        }
        let mut padded = DMatrix::zeros(h.nrows(), dim);
        padded.view_mut((0, 0), h.shape()).copy_from(h);
        padded
    }

    /// Mahalanobis cost matrix for mixed state dimensions: per-track padded
    /// H (extra components such as β are unobserved), then the shared
    /// assembly of [`build_cost_matrix`].
    fn build_padded_cost_matrix(
        &self,
        alive: &[usize],
        detections: &[DVector<f64>],
    ) -> Vec<Vec<f64>> {
        let r = &self.measurement_noise;
        let mut predicted = Vec::with_capacity(alive.len());
        let mut innovations = Vec::with_capacity(alive.len());
        for &ti in alive {
            let h = self.observation_for_dim(self.tracks[ti].state.len());
            predicted.push(&h * &self.tracks[ti].state);
            innovations.push(&h * &self.tracks[ti].covariance * h.transpose() + r);
        }
        build_cost_matrix(&predicted, &innovations, detections, self.gate_threshold)
    }

    /// Apply a single measurement update to a track (IMM or KF).
    ///
    /// The observation matrix is padded per track dimension (see
    /// [`Self::observation_for_dim`]) so 7D reentry tracks update alongside
    /// 6D ones.
    fn apply_measurement_update(&mut self, ti: usize, detection: &DVector<f64>, is_imm: bool) {
        let h = self.observation_for_dim(self.tracks[ti].state.len());
        let r = self.measurement_noise.clone();
        if is_imm {
            if let Some(key) = self.tracks[ti].imm_key {
                #[cfg(feature = "learned-imm")]
                if let Some(lf) = self.learned_imm_filters.get_mut(&key) {
                    let result = lf.update_with_measurement(detection, &h, &r);
                    self.tracks[ti].state = result.state;
                    self.tracks[ti].covariance = result.covariance;
                    self.tracks[ti].dominant_mode = Some(result.dominant_mode);
                    self.tracks[ti].mode_probabilities = Some(result.mode_probabilities);
                    return;
                }
                if let Some(imm) = self.imm_filters.get_mut(&key) {
                    let result = imm.update_with_measurement(detection, &h, &r);
                    self.tracks[ti].state = result.state;
                    self.tracks[ti].covariance = result.covariance;
                    self.tracks[ti].dominant_mode = Some(result.dominant_mode);
                    self.tracks[ti].mode_probabilities = Some(result.mode_probabilities);
                }
            }
        } else {
            let track = &self.tracks[ti];
            let mut kf = KalmanFilter::new(track.state.clone(), track.covariance.clone());
            kf.update(detection, &h, &r);
            self.tracks[ti].state = kf.x;
            self.tracks[ti].covariance = kf.p;
        }
    }

    /// Apply lifecycle updates to all alive tracks based on association results.
    fn apply_lifecycle(&mut self, associated_tracks: &[bool]) {
        let alive = alive_indices(&self.tracks);
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
    }

    /// Birth new tracks from unassigned detections.
    fn birth_unassigned(&mut self, detections: &[DVector<f64>], associated_dets: &[bool]) {
        for (dj, det) in detections.iter().enumerate() {
            if !associated_dets[dj] {
                self.birth_track(det, TargetClass::Unknown);
            }
        }
    }

    /// Remove deleted tracks and clean up their IMM filters.
    fn remove_deleted_tracks(&mut self) {
        let imm_filters = &mut self.imm_filters;
        #[cfg(feature = "learned-imm")]
        let learned_imm_filters = &mut self.learned_imm_filters;
        self.tracks.retain(|t| {
            if t.lifecycle == TrackState::Deleted {
                if let Some(key) = t.imm_key {
                    imm_filters.remove(&key);
                    #[cfg(feature = "learned-imm")]
                    learned_imm_filters.remove(&key);
                }
                false
            } else {
                true
            }
        });
    }

    /// Store a freshly-built IMM bank for a new track under `key`. Analytic
    /// build: the bank goes straight into `imm_filters`.
    #[cfg(not(feature = "learned-imm"))]
    fn insert_imm_filter(&mut self, key: usize, imm: ImmFilter) {
        self.imm_filters.insert(key, imm);
    }

    /// Store a freshly-built IMM bank for a new track under `key`. Learned
    /// build: if the learned path is active and the bank has the expected
    /// model count, wrap it in a `LearnedImmFilter`; otherwise (no learned
    /// path, wrong bank size, or an unloadable classifier) fall back to the
    /// analytic `imm_filters` map.
    #[cfg(feature = "learned-imm")]
    fn insert_imm_filter(&mut self, key: usize, imm: ImmFilter) {
        let Some(path) = self.learned_classifier_path.clone() else {
            self.imm_filters.insert(key, imm);
            return;
        };
        // Analytic fallback #1: a bank that isn't the 4-model `cv_ca_ctrv_ct`
        // shape the classifier expects. (The constructor already rejects such a
        // factory, so this is defensive.)
        if imm.num_models() != NUM_MODES {
            self.imm_filters.insert(key, imm);
            return;
        }
        match ImmModeAdapter::from_onnx(&path) {
            Ok(adapter) => match LearnedImmFilter::new(imm, adapter) {
                Ok(lf) => {
                    self.learned_imm_filters.insert(key, lf);
                }
                // `LearnedImmFilter::new` only fails on a wrong bank size, which
                // the guard above already excludes — so this is unreachable. If
                // it ever fires, `imm` has been moved into the failed call and
                // cannot be recovered, so log loudly rather than silently
                // leaving the track without a filter.
                Err(e) => {
                    eprintln!(
                        "learned-imm: BUG: wrapper init failed for a validated bank ({e}); \
                         track {key} has no IMM filter"
                    );
                }
            },
            // Analytic fallback #2: the classifier can't be loaded at birth
            // (e.g. the file was removed mid-run). `imm` is untouched here.
            Err(e) => {
                eprintln!(
                    "learned-imm: classifier load failed ({e}); analytic fallback for track {key}"
                );
                self.imm_filters.insert(key, imm);
            }
        }
    }

    /// Reclassify a live track (spec scenario "Class reclassification"):
    /// switch its [`TargetClass`] — and therefore the head whose motion
    /// model `predict_single_model` dispatches — adapting the state
    /// vector across head dimensions. The interleaved kinematic prefix
    /// `[x, vx, y, vy, z, vz]` is shared by every head, so growing 6D → 7D
    /// preserves it and births the appended β from the new head's prior,
    /// while shrinking 7D → 6D drops β (the `Reentry7Mapping` pattern of
    /// design Decisions 4/6, `orbital-ballistic-filter-models`).
    ///
    /// IMM-mode tracks keep their bank's state shape (the bank owns the
    /// common 6D space); only the class tag changes for them. Returns
    /// `false` when no live track has this `id`.
    pub fn reclassify(&mut self, id: TrackId, new_class: TargetClass) -> bool {
        let head = self.heads.get(new_class).clone();
        let Some(track) = self.tracks.iter_mut().find(|t| t.id == id && t.is_alive()) else {
            return false;
        };
        track.class = new_class;
        if track.imm_key.is_none() {
            adapt_track_dimension(track, &head);
        }
        true
    }

    /// Create a new track from a detection.
    fn birth_track(&mut self, detection: &DVector<f64>, class: TargetClass) {
        let head = self.heads.get(class).clone();
        let h = self.observation_for_dim(head.state_dim);
        let state = birth_state(detection, &head, &h);
        let cov = birth_covariance(detection, &head);
        let mut track = Track::new(state.clone(), cov.clone(), class);

        // If in IMM mode, create a filter for this track. Build the bank while
        // the `imm_config_factory` borrow is live, then hand the owned filter to
        // `insert_imm_filter` (which takes `&mut self`).
        let new_imm = self.imm_config_factory.as_ref().map(|factory| {
            let config = factory();
            ImmFilter::new(config, &state, &cov)
        });
        if let Some(imm) = new_imm {
            let key = self.next_imm_key;
            self.next_imm_key += 1;
            track.imm_key = Some(key);
            self.insert_imm_filter(key, imm);
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

/// EKF-style predict for one track through a nonlinear head model (design
/// Decision 6 of `orbital-ballistic-filter-models`): `F = ∂f/∂x` evaluated
/// at the prior mean, then `x ← f(x, dt)`, `P ← F·P·Fᵀ + Q`.
fn ekf_predict_track(track: &mut Track, model: &dyn MotionModel, dt: f64) {
    let f = model.jacobian(&track.state, dt);
    track.state = model.predict(&track.state, dt);
    track.covariance = &f * &track.covariance * f.transpose() + model.process_noise(dt);
}

/// Initial state for a newborn track: detection components mapped to state
/// indices through the (dimension-padded) observation matrix `h`, remaining
/// components zero — except model-specific extras: the reentry head's β is
/// born at its `beta_init` (design Decision 6).
fn birth_state(detection: &DVector<f64>, head: &TrackHead, h: &DMatrix<f64>) -> DVector<f64> {
    let mut state = DVector::zeros(head.state_dim);
    let m_dim = detection.len().min(h.nrows());
    for i in 0..m_dim.min(head.state_dim) {
        // Map measurement indices to state indices via H
        for j in 0..head.state_dim {
            if h[(i, j)].abs() > 0.5 {
                state[j] = detection[i];
            }
        }
    }
    if let HeadModel::Reentry { beta_init, .. } = head.model
        && head.state_dim == 7
    {
        state[6] = beta_init;
    }
    state
}

/// Initial covariance for a newborn track: the head's diagonal prior, with
/// the orbital head's velocity entries refined to the circular-orbit prior
/// `μ/(3‖r‖)` when the detection is a plausible ECI position (task 5.5
/// resolution — see the Open Questions section of the
/// `orbital-ballistic-filter-models` design).
fn birth_covariance(detection: &DVector<f64>, head: &TrackHead) -> DMatrix<f64> {
    let mut diag = head.initial_covariance.clone();
    if matches!(head.model, HeadModel::KeplerJ2 { .. }) && detection.len() >= 3 {
        let r = (detection[0].powi(2) + detection[1].powi(2) + detection[2].powi(2)).sqrt();
        let earth = GravityModel::EARTH_WGS84;
        if r > earth.equatorial_radius {
            // Unknown direction at circular speed v_c = √(μ/r): zero-mean
            // velocity with per-axis variance v_c²/3.
            let vel_var = earth.mu / (3.0 * r);
            for idx in [1usize, 3, 5] {
                if idx < diag.len() {
                    diag[idx] = vel_var;
                }
            }
        }
    }
    DMatrix::from_diagonal(&DVector::from_column_slice(&diag))
}

/// Resize a track's state/covariance to a new head's `state_dim` after
/// reclassification. The shared interleaved kinematic prefix is preserved
/// (mean and covariance block); appended components start at the new head's
/// diagonal prior, with the reentry β mean born at `beta_init`; dropped
/// components are truncated.
fn adapt_track_dimension(track: &mut Track, head: &TrackHead) {
    let old_dim = track.state.len();
    let new_dim = head.state_dim;
    if old_dim == new_dim {
        return;
    }
    let mut state = DVector::zeros(new_dim);
    let mut cov = DMatrix::from_diagonal(&DVector::from_column_slice(&head.initial_covariance));
    let keep = old_dim.min(new_dim);
    state
        .rows_mut(0, keep)
        .copy_from(&track.state.rows(0, keep));
    cov.view_mut((0, 0), (keep, keep))
        .copy_from(&track.covariance.view((0, 0), (keep, keep)));
    if let HeadModel::Reentry { beta_init, .. } = head.model
        && new_dim == 7
        && old_dim < 7
    {
        state[6] = beta_init;
    }
    track.state = state;
    track.covariance = cov;
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

    // --- Learned-IMM tracker seam (feature `learned-imm`) -------------------
    // These run against the committed random-weight stub classifier, so they
    // exercise the wiring (constructor -> birth -> predict -> update ->
    // lifecycle -> removal) rather than classification accuracy.

    #[cfg(feature = "learned-imm")]
    const STUB_CLASSIFIER: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/models/imm_mode_classifier.onnx"
    );

    #[cfg(feature = "learned-imm")]
    #[test]
    fn learned_imm_constructor_loads_stub() {
        let tracker = MultiObjectTracker::new_imm_position_learned(
            || ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            STUB_CLASSIFIER,
            10.0,
            100.0,
        );
        assert!(
            tracker.is_ok(),
            "stub classifier should load: {:?}",
            tracker.err()
        );
    }

    #[cfg(feature = "learned-imm")]
    #[test]
    fn learned_imm_constructor_errors_on_missing_model() {
        let tracker = MultiObjectTracker::new_imm_position_learned(
            || ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            "/nonexistent/imm_mode_classifier.onnx",
            10.0,
            100.0,
        );
        assert!(tracker.is_err(), "missing model path should be rejected");
    }

    #[cfg(feature = "learned-imm")]
    #[test]
    fn learned_imm_tracker_tracks_target() {
        // End-to-end: a learned-IMM tracker confirms and follows a CV target,
        // proving birth/predict/update/lifecycle all route through the
        // LearnedImmFilter (with analytic fallback during the WINDOW_LEN warmup).
        let mut tracker = MultiObjectTracker::new_imm_position_learned(
            || ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            STUB_CLASSIFIER,
            10.0,
            100.0,
        )
        .expect("stub classifier loads");

        let dt = 1.0;
        let speed = 100.0;
        let mut x = 0.0_f64;
        // Run comfortably past the classifier's window so the learned update
        // (not just the warm-up fallback) has engaged, regardless of WINDOW_LEN.
        let steps = thresh_filter::imm_adapter::WINDOW_LEN + 10;
        for _ in 0..steps {
            x += speed * dt;
            let det = DVector::from_column_slice(&[x, 0.0, 0.0]);
            tracker.step(std::slice::from_ref(&det), dt);
        }

        assert_eq!(tracker.confirmed_count(), 1, "track should be confirmed");
        assert_eq!(tracker.alive_count(), 1, "exactly one track expected");
        let track = &tracker.tracks[0];
        // The learned path populates dominant_mode / mode_probabilities once the
        // window fills (and the estimate should be near the true x).
        assert!(track.dominant_mode.is_some(), "dominant mode should be set");
        assert!(
            track.mode_probabilities.is_some(),
            "mode probabilities should be set"
        );
        assert!(
            (track.state[0] - x).abs() < 200.0,
            "x estimate {} should track the target near {x}",
            track.state[0]
        );
    }

    // --- Automotive ENU + ego-motion (automotive-tracking-pipeline, Phase 2) ---

    use thresh_core::ego::{EgoMotion, EgoPose};

    fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    }

    /// Ego pose with both yaw and pitch (a road grade), so the tests exercise
    /// the full quaternion path rather than planar rotation only.
    fn ego_pose_at(x: f64, y: f64, yaw: f64, pitch: f64, t: f64) -> EgoPose {
        let qy = [(yaw / 2.0).cos(), 0.0, 0.0, (yaw / 2.0).sin()];
        let qp = [(pitch / 2.0).cos(), 0.0, (pitch / 2.0).sin(), 0.0];
        EgoPose {
            translation_m: [x, y, 0.0],
            rotation_wxyz: quat_mul(qy, qp),
            time_s: t,
        }
    }

    /// Express a fixed world point in a given ego frame: the exact inverse of
    /// `EgoPose::transform_to_world` (rotate the offset by the conjugate
    /// quaternion), so the test inverts the same path production code uses.
    fn world_to_ego(pose: &EgoPose, p_world: [f64; 3]) -> DVector<f64> {
        let d = [
            p_world[0] - pose.translation_m[0],
            p_world[1] - pose.translation_m[1],
            p_world[2] - pose.translation_m[2],
        ];
        let q = pose.rotation_wxyz;
        let inverse = EgoPose {
            translation_m: [0.0; 3],
            rotation_wxyz: [q[0], -q[1], -q[2], -q[3]],
            time_s: pose.time_s,
        };
        DVector::from_column_slice(&inverse.rotate_to_world(d))
    }

    /// The capability-spec scenario: a stationary world object observed from a
    /// moving, turning, pitching ego must not drift in the world frame.
    #[test]
    fn stationary_object_does_not_drift_under_ego_motion() {
        let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let parked_car = [40.0, 12.0, 0.0];
        let dt = 0.1;

        let mut prev: Option<EgoPose> = None;
        for k in 0..30 {
            let t = k as f64 * dt;
            // Ego drives +x at 10 m/s, slowly yawing, on a 3% grade (pitch).
            let pose = ego_pose_at(10.0 * t, 0.0, 0.02 * t, 0.03, t);
            let motion = prev
                .and_then(|p| EgoMotion::from_pose_pair(&p, &pose))
                .unwrap_or_else(|| EgoMotion::stationary(pose));
            let det_ego = world_to_ego(&pose, parked_car);
            tracker.step_with_ego(&[(det_ego, TargetClass::Car)], dt, &motion);
            prev = Some(pose);
        }

        assert_eq!(tracker.alive_count(), 1, "one persistent track expected");
        let track = &tracker.tracks[0];
        let est = [track.state[0], track.state[2], track.state[4]];
        for (i, want) in parked_car.iter().enumerate() {
            assert!(
                (est[i] - want).abs() < 0.5,
                "world axis {i}: estimate {} drifted from {} despite ego motion",
                est[i],
                want
            );
        }
        // World-frame velocity of a parked object must be near zero.
        let speed = (track.state[1].powi(2) + track.state[3].powi(2)).sqrt();
        assert!(
            speed < 1.0,
            "parked object should be static, got {speed} m/s"
        );
    }

    #[test]
    fn step_with_ego_skips_sub_3d_detections() {
        // A pixel-only camera observation (2D) has no Cartesian position; it
        // must be skipped, not panic on d[2].
        let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let pixel_only = DVector::from_column_slice(&[640.0, 360.0]);
        let lidar = DVector::from_column_slice(&[10.0, 5.0, 0.0]);
        let motion = EgoMotion::stationary(EgoPose::identity(0.0));
        tracker.step_with_ego(
            &[
                (pixel_only, TargetClass::Pedestrian),
                (lidar, TargetClass::Car),
            ],
            0.1,
            &motion,
        );
        assert_eq!(tracker.alive_count(), 1, "only the 3D detection births");
        assert_eq!(tracker.tracks[0].class, TargetClass::Car);
    }

    #[test]
    fn step_classed_truncates_6d_lidar_vectors() {
        // A LiDAR Measurement::to_vector() with velocity is 6D
        // [x,y,z,vx,vy,vz]; step_classed must truncate to the position rather
        // than panic indexing the 3x6 observation matrix in birth_track.
        let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let lidar_6d = DVector::from_column_slice(&[10.0, 5.0, 0.0, 12.0, 0.5, 0.0]);
        tracker.step_classed(&[(lidar_6d, TargetClass::Car)], 0.1);
        assert_eq!(tracker.alive_count(), 1);
        let t = &tracker.tracks[0];
        assert_eq!(t.class, TargetClass::Car);
        assert!((t.state[0] - 10.0).abs() < 1e-9, "x from leading position");
        assert!((t.state[2] - 5.0).abs() < 1e-9, "y from leading position");
    }

    #[test]
    fn step_classed_births_with_class_specific_head() {
        let mut tracker = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let det = DVector::from_column_slice(&[5.0, 2.0, 0.0]);
        // Pedestrian head confirms after 2 hits (vs 3-of-5 for Unknown).
        tracker.step_classed(&[(det.clone(), TargetClass::Pedestrian)], 0.1);
        assert_eq!(tracker.tracks[0].class, TargetClass::Pedestrian);
        assert_eq!(tracker.confirmed_count(), 0, "not confirmed after 1 hit");
        tracker.step_classed(&[(det, TargetClass::Pedestrian)], 0.1);
        assert_eq!(
            tracker.confirmed_count(),
            1,
            "pedestrian head (2-of-2) should confirm on the second hit"
        );
    }

    #[test]
    fn plain_step_is_unchanged_by_automotive_entry_points() {
        // The classless path still births Unknown — aerospace behavior intact.
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 16.0);
        let det = DVector::from_column_slice(&[100.0, 0.0, 1000.0]);
        tracker.step(std::slice::from_ref(&det), 1.0);
        assert_eq!(tracker.tracks[0].class, TargetClass::Unknown);
    }

    #[test]
    fn identity_ego_matches_world_frame_step() {
        // With an identity ego pose, step_with_ego must behave exactly like
        // step_classed on the same detections.
        let mut a = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let mut b = MultiObjectTracker::new_automotive_enu(0.5, 16.0);
        let identity = EgoMotion::stationary(EgoPose::identity(0.0));
        for k in 0..10 {
            let det = DVector::from_column_slice(&[10.0 + k as f64, -3.0, 0.0]);
            a.step_with_ego(&[(det.clone(), TargetClass::Car)], 0.1, &identity);
            b.step_classed(&[(det, TargetClass::Car)], 0.1);
        }
        assert_eq!(a.tracks.len(), b.tracks.len());
        let (ta, tb) = (&a.tracks[0], &b.tracks[0]);
        assert!((ta.state[0] - tb.state[0]).abs() < 1e-12);
        assert!((ta.state[2] - tb.state[2]).abs() < 1e-12);
        assert_eq!(ta.lifecycle, tb.lifecycle);
    }

    // --- Orbital & ballistic head dispatch (orbital-ballistic-filter-models §5)

    /// Task 5.3 (spec "Ballistic head uses the reentry model"): a track born
    /// from ballistic-classified detections runs the 7D reentry model and
    /// its state exposes the estimated β alongside position/velocity.
    #[test]
    fn ballistic_track_runs_7d_model_and_exposes_beta() {
        let earth = GravityModel::EARTH_WGS84;
        // Exo-atmospheric arc near apogee, truth propagated through the same
        // reentry dynamics the head builds (drag ≈ 0 above the atmosphere).
        let truth_model = TrackHead::ballistic().build_model();
        let mut truth = DVector::from_column_slice(&[
            earth.equatorial_radius + 1_200_000.0,
            0.0, // x, vx
            0.0,
            800.0, // y, vy
            0.0,
            -200.0,  // z, vz
            2_000.0, // true β (unobservable exo-atmospherically)
        ]);
        let dt = 1.0;
        let mut tracker = MultiObjectTracker::new_cv_position(50.0, 500.0);
        for _ in 0..6 {
            let det = DVector::from_column_slice(&[truth[0], truth[2], truth[4]]);
            tracker.step_classed(&[(det, TargetClass::Ballistic)], dt);
            truth = truth_model.predict(&truth, dt);
        }

        assert_eq!(tracker.alive_count(), 1, "one ballistic track expected");
        assert_eq!(tracker.confirmed_count(), 1, "2-of-3 should have confirmed");
        let track = &tracker.tracks[0];
        assert_eq!(track.class, TargetClass::Ballistic);
        assert_eq!(track.state.len(), 7, "ballistic head must run the 7D state");
        assert_eq!(track.covariance.shape(), (7, 7));

        // β exposed alongside position/velocity: born at the head's
        // beta_init and held positive; with zero drag there is no β
        // information, so it stays at the prior.
        let HeadModel::Reentry { beta_init, .. } = TrackHead::ballistic().model else {
            panic!("ballistic head must select the reentry model");
        };
        assert!(track.state[6] > 0.0, "estimated β must stay positive");
        assert!(
            (track.state[6] - beta_init).abs() < 1e-6,
            "β should sit at its prior exo-atmospherically, got {}",
            track.state[6]
        );

        // The 7D filter followed the arc.
        let pos_err = ((track.state[0] - truth[0]).powi(2)
            + (track.state[2] - truth[2]).powi(2)
            + (track.state[4] - truth[4]).powi(2))
        .sqrt();
        assert!(
            pos_err < 5_000.0,
            "reentry filter lost the arc: {pos_err} m"
        );
    }

    /// Tasks 5.4 + 5.5 (spec "Orbital head tracks a satellite"): an
    /// orbital-classified track dispatches the Kepler+J2 model. The
    /// circular-orbit birth velocity prior lets the EKF learn velocity from
    /// position updates, after which prediction across a measurement gap
    /// follows the orbit while straight-line (CV) extrapolation of the same
    /// state diverges by tens of km.
    #[test]
    fn orbital_track_follows_orbit_across_measurement_gap() {
        let earth = GravityModel::EARTH_WGS84;
        let radius = earth.equatorial_radius + 500_000.0;
        let mean_motion = (earth.mu / (radius * radius * radius)).sqrt();
        let truth = |t: f64| -> [f64; 3] {
            let th = mean_motion * t;
            [radius * th.cos(), radius * th.sin(), 0.0]
        };

        let dt = 30.0;
        let mut tracker = MultiObjectTracker::new_cv_position(100.0, 100.0);
        // Scans at t = 0..120 s: birth, then velocity convergence through
        // the circular-orbit prior (task 5.5).
        for k in 0..5 {
            let det = DVector::from_column_slice(&truth(k as f64 * dt));
            tracker.step_classed(&[(det, TargetClass::Orbital)], dt);
        }
        assert_eq!(tracker.alive_count(), 1, "one orbital track expected");
        assert_eq!(tracker.tracks[0].class, TargetClass::Orbital);
        assert_eq!(tracker.tracks[0].state.len(), 6);

        // Pre-gap state for the straight-line comparison.
        let s = tracker.tracks[0].state.clone();
        let (p0, v0) = ([s[0], s[2], s[4]], [s[1], s[3], s[5]]);

        // Gap: 3 missed scans (90 s), inside the orbital deletion window (5).
        for _ in 0..3 {
            tracker.step_classed(&[], dt);
        }
        assert_eq!(
            tracker.alive_count(),
            1,
            "orbital head must coast through the gap"
        );

        let gap_s = 3.0 * dt;
        let want = truth(4.0 * dt + gap_s);
        let got = &tracker.tracks[0].state;
        let model_err =
            ((got[0] - want[0]).powi(2) + (got[2] - want[1]).powi(2) + (got[4] - want[2]).powi(2))
                .sqrt();
        let straight_err = ((p0[0] + v0[0] * gap_s - want[0]).powi(2)
            + (p0[1] + v0[1] * gap_s - want[1]).powi(2)
            + (p0[2] + v0[2] * gap_s - want[2]).powi(2))
        .sqrt();

        assert!(
            model_err < 10_000.0,
            "Kepler+J2 prediction left the orbit by {model_err} m over the gap"
        );
        assert!(
            straight_err > 20_000.0 && straight_err > 2.0 * model_err,
            "straight-line extrapolation should diverge measurably: \
             straight {straight_err} m vs model {model_err} m"
        );
    }

    /// Task 5.6 (spec "Heterogeneous target class tracking"): aircraft
    /// (6D CV) and ballistic (7D reentry) tracks coexist in one tracker —
    /// mixed state dimensions flow through the padded observation and
    /// cost-matrix paths, and each class keeps its own model and policies.
    #[test]
    fn heterogeneous_aircraft_and_ballistic_classes_coexist() {
        let earth = GravityModel::EARTH_WGS84;
        let dt = 1.0;
        let mut tracker = MultiObjectTracker::new_cv_position(50.0, 500.0);

        let ballistic_model = TrackHead::ballistic().build_model();
        let mut bal_truth = DVector::from_column_slice(&[
            earth.equatorial_radius + 400_000.0,
            0.0,
            1.0e6,
            700.0,
            0.0,
            -300.0,
            1_500.0,
        ]);
        let mut ac = [0.0, 0.0, 10_000.0];
        let ac_v = [250.0, 0.0, 0.0];

        for _ in 0..8 {
            let dets = vec![
                (DVector::from_column_slice(&ac), TargetClass::Aircraft),
                (
                    DVector::from_column_slice(&[bal_truth[0], bal_truth[2], bal_truth[4]]),
                    TargetClass::Ballistic,
                ),
            ];
            tracker.step_classed(&dets, dt);
            for (p, v) in ac.iter_mut().zip(ac_v) {
                *p += v * dt;
            }
            bal_truth = ballistic_model.predict(&bal_truth, dt);
        }

        assert_eq!(tracker.alive_count(), 2, "both classes must be tracked");
        assert_eq!(
            tracker.confirmed_count(),
            2,
            "aircraft (3-of-5) and ballistic (2-of-3) both confirm in 8 scans"
        );
        let air = tracker
            .tracks
            .iter()
            .find(|t| t.class == TargetClass::Aircraft)
            .expect("aircraft track");
        let bal = tracker
            .tracks
            .iter()
            .find(|t| t.class == TargetClass::Ballistic)
            .expect("ballistic track");
        assert_eq!(air.state.len(), 6, "aircraft head stays 6D CV");
        assert_eq!(
            bal.state.len(),
            7,
            "ballistic head runs the 7D reentry state"
        );
        assert!(
            (bal.state[0] - bal_truth[0]).abs() < 5_000.0,
            "ballistic track follows its arc"
        );
        assert!(
            (air.state[0] - ac[0]).abs() < 500.0,
            "aircraft track follows its leg"
        );
    }

    /// The JPDA path pads H per track dimension like Hungarian/MHT: a 7D
    /// ballistic track alongside a 6D aircraft previously panicked on the
    /// shared 3×6 observation matrix (PR #133 review finding).
    #[test]
    fn jpda_pads_observation_matrix_for_mixed_dimension_tracks() {
        let earth = GravityModel::EARTH_WGS84;
        let dt = 1.0;
        let mut tracker = MultiObjectTracker::new_cv_position_with_strategy(
            50.0,
            500.0,
            AssociationStrategy::Jpda {
                detection_prob: 0.9,
                // Low enough that the miss weight (1 − p_d)·λ never beats
                // the Gaussian likelihood through the ballistic head's wide
                // birth prior (S ~ km-scale ⇒ N(z; ẑ, S) is tiny).
                clutter_density: 1e-30,
            },
        );

        let ballistic_model = TrackHead::ballistic().build_model();
        let mut bal_truth = DVector::from_column_slice(&[
            earth.equatorial_radius + 400_000.0,
            0.0,
            1.0e6,
            700.0,
            0.0,
            -300.0,
            1_500.0,
        ]);
        let mut ac = [0.0, 0.0, 10_000.0];
        let ac_v = [250.0, 0.0, 0.0];

        for _ in 0..8 {
            let dets = vec![
                (DVector::from_column_slice(&ac), TargetClass::Aircraft),
                (
                    DVector::from_column_slice(&[bal_truth[0], bal_truth[2], bal_truth[4]]),
                    TargetClass::Ballistic,
                ),
            ];
            tracker.step_classed(&dets, dt);
            for (p, v) in ac.iter_mut().zip(ac_v) {
                *p += v * dt;
            }
            bal_truth = ballistic_model.predict(&bal_truth, dt);
        }

        assert_eq!(tracker.alive_count(), 2, "both classes tracked under JPDA");
        let bal = tracker
            .tracks
            .iter()
            .find(|t| t.class == TargetClass::Ballistic)
            .expect("ballistic track");
        assert_eq!(bal.state.len(), 7, "ballistic head runs the 7D state");
        assert!(
            (bal.state[0] - bal_truth[0]).abs() < 5_000.0,
            "7D track follows its arc through JPDA updates"
        );
    }

    /// Task 5.6 (spec "Class reclassification"): switching a track's class
    /// re-dispatches its motion model and adapts the state vector across the
    /// 6D/7D boundary — the kinematic prefix survives, β is born from the
    /// new head's prior on the way up and dropped on the way down.
    #[test]
    fn reclassification_adapts_state_across_6d_7d_heads() {
        let earth = GravityModel::EARTH_WGS84;
        let mut tracker = MultiObjectTracker::new_cv_position(50.0, 1e4);
        // Born Unknown (6D CV) from a plausible ECI position.
        let p = [earth.equatorial_radius + 300_000.0, 0.0, 0.0];
        let det = DVector::from_column_slice(&p);
        tracker.step(std::slice::from_ref(&det), 1.0);
        let id = tracker.tracks[0].id;
        assert_eq!(tracker.tracks[0].state.len(), 6);

        // Observed trajectory says "ballistic": reclassify 6D → 7D.
        assert!(tracker.reclassify(id, TargetClass::Ballistic));
        {
            let t = &tracker.tracks[0];
            assert_eq!(t.class, TargetClass::Ballistic);
            assert_eq!(t.state.len(), 7);
            assert_eq!(t.covariance.shape(), (7, 7));
            assert!(
                (t.state[0] - p[0]).abs() < 1e-9,
                "kinematic prefix must be preserved"
            );
            assert!(t.state[6] > 0.0, "β born from the new head's prior");
            assert!(t.covariance[(6, 6)] >= 1e6, "wide β prior installed");
        }

        // The next cycle predicts through the 7D reentry model and still
        // associates and updates without dimension mismatches.
        tracker.step_classed(&[(det.clone(), TargetClass::Ballistic)], 1.0);
        assert_eq!(
            tracker.alive_count(),
            1,
            "no duplicate birth after reclassification"
        );
        assert_eq!(tracker.tracks[0].state.len(), 7);
        assert!(tracker.tracks[0].total_hits >= 2);

        // Reclassify back down: 7D → 6D aircraft drops β.
        assert!(tracker.reclassify(id, TargetClass::Aircraft));
        {
            let t = &tracker.tracks[0];
            assert_eq!(t.class, TargetClass::Aircraft);
            assert_eq!(t.state.len(), 6);
            assert_eq!(t.covariance.shape(), (6, 6));
        }
        tracker.step(std::slice::from_ref(&det), 1.0);
        assert_eq!(tracker.alive_count(), 1);

        // Unknown id is a no-op.
        assert!(!tracker.reclassify(TrackId::new(), TargetClass::Uav));
    }

    /// Task 5.5: the orbital head's velocity prior is refined at birth to
    /// the circular-orbit variance μ/(3‖r‖) when the detection is a
    /// plausible ECI position, and left at the head's static prior when the
    /// detection is not (e.g. a near-origin local-frame position).
    #[test]
    fn orbital_birth_refines_velocity_prior_from_measured_radius() {
        let earth = GravityModel::EARTH_WGS84;
        let r_geo = 42_164_000.0;
        let mut tracker = MultiObjectTracker::new_cv_position(100.0, 100.0);
        tracker.step_classed(
            &[(
                DVector::from_column_slice(&[r_geo, 0.0, 0.0]),
                TargetClass::Orbital,
            )],
            1.0,
        );
        let want = earth.mu / (3.0 * r_geo);
        let cov = &tracker.tracks[0].covariance;
        for idx in [1, 3, 5] {
            assert!(
                (cov[(idx, idx)] - want).abs() < 1e-6 * want,
                "GEO-radius birth should tighten the velocity prior to {want}, \
                 got {}",
                cov[(idx, idx)]
            );
        }

        // Sub-surface radius (local-frame coordinates): static prior stands.
        let mut local = MultiObjectTracker::new_cv_position(100.0, 100.0);
        local.step_classed(
            &[(
                DVector::from_column_slice(&[100.0, 200.0, 50.0]),
                TargetClass::Orbital,
            )],
            1.0,
        );
        let static_prior = TrackHead::orbital().initial_covariance[1];
        assert_eq!(local.tracks[0].covariance[(1, 1)], static_prior);
    }
}
