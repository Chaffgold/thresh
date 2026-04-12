//! Interacting Multiple Model (IMM) filter.
//!
//! Maintains a bank of model-conditioned EKFs with different motion models
//! and blends their outputs using Bayesian mode probabilities. Handles
//! heterogeneous state dimensions via explicit state mapping to a common
//! 6D representation `[x, vx, y, vy, z, vz]`.

use nalgebra::{DMatrix, DVector};

use crate::ekf::ExtendedKalmanFilter;
use crate::models::ca::ConstantAcceleration;
use crate::models::coordinated_turn::CoordinatedTurn;
use crate::models::ctrv::Ctrv;
use crate::models::cv::ConstantVelocity;
use crate::traits::MotionModel;

// ---------------------------------------------------------------------------
// State mapping trait and implementations
// ---------------------------------------------------------------------------

/// Maps between a model's native state representation and the common 6D
/// representation `[x, vx, y, vy, z, vz]`.
pub trait StateMapping: Send + Sync {
    /// Project model state/covariance into common 6D space.
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>);

    /// Project common 6D state/covariance into model-native space.
    #[allow(clippy::wrong_self_convention)]
    fn from_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>)
    -> (DVector<f64>, DMatrix<f64>);

    /// Dimension of the common representation (always 6).
    fn common_dim(&self) -> usize {
        6
    }

    /// Dimension of the model-native state.
    fn model_dim(&self) -> usize;
}

/// Identity mapping for ConstantVelocity (6D <-> 6D).
pub struct CvMapping;

impl StateMapping for CvMapping {
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
        (state.clone(), cov.clone())
    }

    fn from_common(
        &self,
        state: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        (state.clone(), cov.clone())
    }

    fn model_dim(&self) -> usize {
        6
    }
}

/// Mapping for ConstantAcceleration (9D <-> 6D).
///
/// CA state: `[x, vx, ax, y, vy, ay, z, vz, az]`
/// Common:   `[x, vx, y, vy, z, vz]`
///
/// `to_common` drops acceleration components.
/// `from_common` zero-initializes acceleration components.
pub struct CaMapping;

impl StateMapping for CaMapping {
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
        // Index mapping: CA[0,1,3,4,6,7] -> Common[0,1,2,3,4,5]
        let ca_to_common = [0, 1, 3, 4, 6, 7];
        let mut cs = DVector::zeros(6);
        let mut cc = DMatrix::zeros(6, 6);
        for (ci, &si) in ca_to_common.iter().enumerate() {
            cs[ci] = state[si];
            for (cj, &sj) in ca_to_common.iter().enumerate() {
                cc[(ci, cj)] = cov[(si, sj)];
            }
        }
        (cs, cc)
    }

    fn from_common(
        &self,
        state: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let ca_to_common = [0, 1, 3, 4, 6, 7];
        let mut ms = DVector::zeros(9);
        let mut mc = DMatrix::zeros(9, 9);
        for (ci, &si) in ca_to_common.iter().enumerate() {
            ms[si] = state[ci];
            for (cj, &sj) in ca_to_common.iter().enumerate() {
                mc[(si, sj)] = cov[(ci, cj)];
            }
        }
        // Acceleration components stay zero; give them some initial covariance
        // so the filter isn't degenerate.
        mc[(2, 2)] = 100.0;
        mc[(5, 5)] = 100.0;
        mc[(8, 8)] = 100.0;
        (ms, mc)
    }

    fn model_dim(&self) -> usize {
        9
    }
}

/// Mapping for CTRV (5D <-> 6D).
///
/// CTRV state: `[x, y, theta, v, omega]`
/// Common:     `[x, vx, y, vy, z, vz]`
///
/// `to_common`: `vx = v*cos(theta)`, `vy = v*sin(theta)`, `z = vz = 0`.
/// `from_common`: `theta = atan2(vy, vx)`, `v = sqrt(vx^2 + vy^2)`, `omega = 0`.
///
/// Covariance transformation uses the Jacobian of the nonlinear mapping.
pub struct CtrvMapping;

impl StateMapping for CtrvMapping {
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
        let x = state[0];
        let y = state[1];
        let theta = state[2];
        let v = state[3];
        // omega (state[4]) is dropped

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let cs = DVector::from_column_slice(&[x, v * cos_t, y, v * sin_t, 0.0, 0.0]);

        // Jacobian J of [x, v*cos(theta), y, v*sin(theta), 0, 0]
        // w.r.t. [x, y, theta, v, omega]
        // Shape: 6 x 5
        let mut j = DMatrix::zeros(6, 5);
        j[(0, 0)] = 1.0; // dx/dx
        j[(1, 2)] = -v * sin_t; // d(vx)/d(theta)
        j[(1, 3)] = cos_t; // d(vx)/dv
        j[(2, 1)] = 1.0; // dy/dy
        j[(3, 2)] = v * cos_t; // d(vy)/d(theta)
        j[(3, 3)] = sin_t; // d(vy)/dv

        let cc = &j * cov * j.transpose();
        // Ensure z-axis entries have some variance
        let mut cc = cc;
        cc[(4, 4)] = 100.0;
        cc[(5, 5)] = 100.0;

        (cs, cc)
    }

    fn from_common(
        &self,
        state: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let x = state[0];
        let vx = state[1];
        let y = state[2];
        let vy = state[3];

        let v = (vx * vx + vy * vy).sqrt().max(1e-10);
        let theta = vy.atan2(vx);

        let ms = DVector::from_column_slice(&[x, y, theta, v, 0.0]);

        // Jacobian J of [x, y, atan2(vy,vx), sqrt(vx^2+vy^2), 0]
        // w.r.t. [x, vx, y, vy, z, vz]
        // Shape: 5 x 6
        let mut j = DMatrix::zeros(5, 6);
        j[(0, 0)] = 1.0; // dx/dx
        j[(1, 2)] = 1.0; // dy/dy
        // d(theta)/d(vx) = -vy / (vx^2 + vy^2)
        let v2 = vx * vx + vy * vy;
        let v2_safe = v2.max(1e-20);
        j[(2, 1)] = -vy / v2_safe;
        j[(2, 3)] = vx / v2_safe;
        // d(v)/d(vx) = vx / v, d(v)/d(vy) = vy / v
        j[(3, 1)] = vx / v;
        j[(3, 3)] = vy / v;
        // omega row stays zero (no info from common state)

        let mc = &j * cov * j.transpose();
        // Give omega some initial covariance
        let mut mc = mc;
        mc[(4, 4)] = 1.0;

        (ms, mc)
    }

    fn model_dim(&self) -> usize {
        5
    }
}

/// Mapping for CoordinatedTurn (5D <-> 6D).
///
/// CT state: `[x, vx, y, vy, omega]`
/// Common:   `[x, vx, y, vy, z, vz]`
///
/// `to_common`: copies x,vx,y,vy, sets z=vz=0.
/// `from_common`: copies x,vx,y,vy, sets omega=0.
pub struct CtMapping;

impl StateMapping for CtMapping {
    fn to_common(&self, state: &DVector<f64>, cov: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
        // CT[0,1,2,3] -> Common[0,1,2,3], z/vz = 0
        let ct_to_common = [0, 1, 2, 3];
        let mut cs = DVector::zeros(6);
        let mut cc = DMatrix::zeros(6, 6);
        for (ci, &si) in ct_to_common.iter().enumerate() {
            cs[ci] = state[si];
            for (cj, &sj) in ct_to_common.iter().enumerate() {
                cc[(ci, cj)] = cov[(si, sj)];
            }
        }
        // z-axis covariance
        cc[(4, 4)] = 100.0;
        cc[(5, 5)] = 100.0;
        (cs, cc)
    }

    fn from_common(
        &self,
        state: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let ct_to_common = [0, 1, 2, 3];
        let mut ms = DVector::zeros(5);
        let mut mc = DMatrix::zeros(5, 5);
        for (ci, &si) in ct_to_common.iter().enumerate() {
            ms[si] = state[ci];
            for (cj, &sj) in ct_to_common.iter().enumerate() {
                mc[(si, sj)] = cov[(ci, cj)];
            }
        }
        // omega gets some initial covariance
        mc[(4, 4)] = 1.0;
        (ms, mc)
    }

    fn model_dim(&self) -> usize {
        5
    }
}

// ---------------------------------------------------------------------------
// IMM Configuration
// ---------------------------------------------------------------------------

/// Configuration for an Interacting Multiple Model filter.
pub struct ImmConfig {
    /// Motion models (one per mode).
    pub models: Vec<Box<dyn MotionModel>>,
    /// State mappings (one per mode, model <-> common space).
    pub mappings: Vec<Box<dyn StateMapping>>,
    /// Markov transition probability matrix (N x N, rows sum to 1).
    pub transition_matrix: DMatrix<f64>,
    /// Initial mode probabilities (N-vector, sums to 1).
    pub initial_mode_probabilities: DVector<f64>,
}

impl ImmConfig {
    /// Validate the configuration. Returns an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.models.len();
        if n == 0 {
            return Err("No models provided".into());
        }
        if self.mappings.len() != n {
            return Err(format!(
                "Number of mappings ({}) must match number of models ({n})",
                self.mappings.len()
            ));
        }
        if self.initial_mode_probabilities.len() != n {
            return Err(format!(
                "Mode probability vector length ({}) must match number of models ({n})",
                self.initial_mode_probabilities.len()
            ));
        }
        if self.transition_matrix.nrows() != n || self.transition_matrix.ncols() != n {
            return Err(format!(
                "TPM dimensions ({}x{}) must be {n}x{n}",
                self.transition_matrix.nrows(),
                self.transition_matrix.ncols()
            ));
        }

        // Check TPM rows sum to 1
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| self.transition_matrix[(i, j)]).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(format!("TPM row {i} sums to {row_sum}, expected 1.0"));
            }
        }

        // Check mode probabilities sum to 1
        let prob_sum: f64 = self.initial_mode_probabilities.iter().copied().sum();
        if (prob_sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Initial mode probabilities sum to {prob_sum}, expected 1.0"
            ));
        }

        // Check model dims match mapping dims
        for (i, (model, mapping)) in self.models.iter().zip(self.mappings.iter()).enumerate() {
            if model.state_dim() != mapping.model_dim() {
                return Err(format!(
                    "Model {i} state_dim ({}) doesn't match mapping model_dim ({})",
                    model.state_dim(),
                    mapping.model_dim()
                ));
            }
        }

        Ok(())
    }

    /// 2-model CV + CA configuration.
    ///
    /// TPM: `[[0.95, 0.05], [0.05, 0.95]]`, uniform initial mode probabilities.
    pub fn cv_ca(sigma_a: f64, sigma_j: f64) -> Self {
        Self {
            models: vec![
                Box::new(ConstantVelocity::new(sigma_a)),
                Box::new(ConstantAcceleration::new(sigma_j)),
            ],
            mappings: vec![Box::new(CvMapping), Box::new(CaMapping)],
            transition_matrix: DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]),
            initial_mode_probabilities: DVector::from_column_slice(&[0.5, 0.5]),
        }
    }

    /// 2-model CV + CTRV configuration.
    ///
    /// TPM: `[[0.95, 0.05], [0.05, 0.95]]`, uniform initial mode probabilities.
    pub fn cv_ctrv(sigma_a: f64, sigma_v: f64, sigma_omega: f64) -> Self {
        Self {
            models: vec![
                Box::new(ConstantVelocity::new(sigma_a)),
                Box::new(Ctrv::new(sigma_v, sigma_omega)),
            ],
            mappings: vec![Box::new(CvMapping), Box::new(CtrvMapping)],
            transition_matrix: DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]),
            initial_mode_probabilities: DVector::from_column_slice(&[0.5, 0.5]),
        }
    }

    /// 4-model CV + CA + CTRV + CT configuration.
    ///
    /// TPM: 0.90 self-transition, ~0.033 cross-transition.
    pub fn cv_ca_ctrv_ct(sigma_a: f64, sigma_j: f64, sigma_v: f64, sigma_omega: f64) -> Self {
        let cross = (1.0 - 0.90) / 3.0; // ~0.0333
        #[rustfmt::skip]
        let tpm = DMatrix::from_row_slice(4, 4, &[
            0.90, cross, cross, cross,
            cross, 0.90, cross, cross,
            cross, cross, 0.90, cross,
            cross, cross, cross, 0.90,
        ]);
        Self {
            models: vec![
                Box::new(ConstantVelocity::new(sigma_a)),
                Box::new(ConstantAcceleration::new(sigma_j)),
                Box::new(Ctrv::new(sigma_v, sigma_omega)),
                Box::new(CoordinatedTurn::new(sigma_v, sigma_omega)),
            ],
            mappings: vec![
                Box::new(CvMapping),
                Box::new(CaMapping),
                Box::new(CtrvMapping),
                Box::new(CtMapping),
            ],
            transition_matrix: tpm,
            initial_mode_probabilities: DVector::from_column_slice(&[0.25, 0.25, 0.25, 0.25]),
        }
    }
}

// ---------------------------------------------------------------------------
// Model-conditioned filter
// ---------------------------------------------------------------------------

/// A single model-conditioned filter within the IMM bank.
pub struct ModelConditionedFilter {
    /// The EKF instance (handles both linear and nonlinear models).
    pub ekf: ExtendedKalmanFilter,
    /// The motion model for this mode.
    pub model: Box<dyn MotionModel>,
    /// State mapping between model-native and common space.
    pub mapping: Box<dyn StateMapping>,
}

// ---------------------------------------------------------------------------
// IMM Filter
// ---------------------------------------------------------------------------

/// Interacting Multiple Model filter.
///
/// Maintains a bank of model-conditioned EKFs and blends their outputs
/// using Bayesian mode probabilities updated at every measurement step.
pub struct ImmFilter {
    /// Bank of model-conditioned filters.
    pub filters: Vec<ModelConditionedFilter>,
    /// Current mode probabilities.
    pub mode_probabilities: DVector<f64>,
    /// Markov transition probability matrix.
    pub transition_matrix: DMatrix<f64>,
    /// Minimum mode probability to prevent mode starvation.
    pub mode_probability_floor: f64,
}

/// Result of a single IMM step.
pub struct ImmStepResult {
    /// Combined state estimate in common 6D space.
    pub state: DVector<f64>,
    /// Combined covariance in common 6D space.
    pub covariance: DMatrix<f64>,
    /// Updated mode probabilities.
    pub mode_probabilities: DVector<f64>,
    /// Index of the dominant (highest probability) mode.
    pub dominant_mode: usize,
}

impl ImmFilter {
    /// Create a new IMM filter from a configuration and initial state/covariance
    /// in common 6D space `[x, vx, y, vy, z, vz]`.
    pub fn new(
        config: ImmConfig,
        initial_state: &DVector<f64>,
        initial_covariance: &DMatrix<f64>,
    ) -> Self {
        config.validate().expect("Invalid ImmConfig");

        let n = config.models.len();
        let mode_probabilities = config.initial_mode_probabilities.clone();
        let transition_matrix = config.transition_matrix.clone();

        let mut models: Vec<_> = config.models.into_iter().collect();
        let mut mappings: Vec<_> = config.mappings.into_iter().collect();

        let mut filters = Vec::with_capacity(n);
        // Drain in reverse so we can pop efficiently, then reverse back
        models.reverse();
        mappings.reverse();
        for _ in 0..n {
            let model = models.pop().unwrap();
            let mapping = mappings.pop().unwrap();

            let (ms, mc) = mapping.from_common(initial_state, initial_covariance);
            let ekf = ExtendedKalmanFilter::new(ms, mc);
            filters.push(ModelConditionedFilter {
                ekf,
                model,
                mapping,
            });
        }

        Self {
            filters,
            mode_probabilities,
            transition_matrix,
            mode_probability_floor: 0.01,
        }
    }

    /// Number of models in the filter bank.
    pub fn num_models(&self) -> usize {
        self.filters.len()
    }

    /// Run the full IMM cycle: interaction -> predict -> update -> mode update -> combine.
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds.
    /// * `z` - Measurement vector.
    /// * `h` - Observation matrix (maps common 6D state to measurement).
    /// * `r` - Measurement noise covariance.
    ///
    /// Returns the combined state, covariance, mode probabilities, and dominant mode index.
    pub fn step(
        &mut self,
        dt: f64,
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> ImmStepResult {
        // 1. Interaction
        self.interaction_step();

        // 2. Model-conditioned prediction
        self.predict_step(dt);

        // 3. Model-conditioned update + innovation likelihoods
        let log_likelihoods = self.update_step(z, h, r);

        // 4. Mode probability update
        self.update_mode_probabilities(&log_likelihoods);

        // 5. Combine
        let (state, covariance) = self.combine();

        let dominant_mode = self
            .mode_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        ImmStepResult {
            state,
            covariance,
            mode_probabilities: self.mode_probabilities.clone(),
            dominant_mode,
        }
    }

    /// Interaction step: compute mixing probabilities, mix states/covariances
    /// in common space, then back-project to each model's native representation.
    pub fn interaction_step(&mut self) {
        let n = self.num_models();
        let mu = &self.mode_probabilities;
        let pi = &self.transition_matrix;

        // Compute predicted mode probabilities c_j = sum_i(pi_{ij} * mu_i)
        let mut c_bar: DVector<f64> = DVector::zeros(n);
        for j in 0..n {
            for i in 0..n {
                c_bar[j] += pi[(i, j)] * mu[i];
            }
        }

        // Compute mixing probabilities mu_{i|j} = pi_{ij} * mu_i / c_j
        let mut mixing_probs = DMatrix::<f64>::zeros(n, n);
        for j in 0..n {
            if c_bar[j] > 1e-30 {
                for i in 0..n {
                    mixing_probs[(i, j)] = pi[(i, j)] * mu[i] / c_bar[j];
                }
            }
        }

        // Project all filter states into common space
        let common_states: Vec<(DVector<f64>, DMatrix<f64>)> = self
            .filters
            .iter()
            .map(|f| f.mapping.to_common(&f.ekf.x, &f.ekf.p))
            .collect();

        // For each target model j, compute mixed state/covariance in common space
        for j in 0..n {
            // Mixed state: x_j^0 = sum_i(mu_{i|j} * x_i_common)
            let mut mixed_state: DVector<f64> = DVector::zeros(6);
            for i in 0..n {
                let term: DVector<f64> = mixing_probs[(i, j)] * &common_states[i].0;
                mixed_state += term;
            }

            // Mixed covariance: P_j^0 = sum_i(mu_{i|j} * (P_i + delta_i * delta_i'))
            let mut mixed_cov: DMatrix<f64> = DMatrix::zeros(6, 6);
            for i in 0..n {
                let delta = &common_states[i].0 - &mixed_state;
                let p_i = &common_states[i].1;
                let term: DMatrix<f64> = mixing_probs[(i, j)] * (p_i + &delta * delta.transpose());
                mixed_cov += term;
            }

            // Back-project to model j's native space
            let (ms, mc) = self.filters[j]
                .mapping
                .from_common(&mixed_state, &mixed_cov);
            self.filters[j].ekf.x = ms;
            self.filters[j].ekf.p = mc;
        }
    }

    /// Run the interaction and prediction steps, then return the combined
    /// state and covariance. Call this before association so the tracker has
    /// the predicted combined estimate for gating.
    ///
    /// After calling this, the internal model-conditioned filters are in
    /// predicted (prior) state, ready for [`update_with_measurement`].
    pub fn predict(&mut self, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        self.interaction_step();
        self.predict_step(dt);
        self.combine()
    }

    /// Run the measurement update, mode probability update, and combination
    /// steps. Must be called after [`predict`].
    ///
    /// Returns the combined state, covariance, mode probabilities, and
    /// dominant mode index.
    pub fn update_with_measurement(
        &mut self,
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> ImmStepResult {
        let log_likelihoods = self.update_step(z, h, r);
        self.update_mode_probabilities(&log_likelihoods);
        let (state, covariance) = self.combine();

        let dominant_mode = self
            .mode_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        ImmStepResult {
            state,
            covariance,
            mode_probabilities: self.mode_probabilities.clone(),
            dominant_mode,
        }
    }

    /// Model-conditioned prediction step.
    fn predict_step(&mut self, dt: f64) {
        for f in &mut self.filters {
            f.ekf.predict(f.model.as_ref(), dt);
        }
    }

    /// Model-conditioned update step. Returns log-likelihoods for each model.
    fn update_step(
        &mut self,
        z: &DVector<f64>,
        h_common: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Vec<f64> {
        let n = self.num_models();
        let mut log_likelihoods = Vec::with_capacity(n);
        let m = z.len();

        for f in &mut self.filters {
            // Map predicted state to common space for observation
            let (x_common, p_common) = f.mapping.to_common(&f.ekf.x, &f.ekf.p);

            // Innovation in common space
            let y = z - h_common * &x_common;
            let s = h_common * &p_common * h_common.transpose() + r;

            // Log-likelihood: -0.5 * (m*ln(2*pi) + ln(|S|) + y'*S^{-1}*y)
            let s_inv = s.clone().try_inverse().unwrap_or_else(|| {
                // Fallback: add regularization
                let reg = &s + DMatrix::identity(m, m) * 1e-6;
                reg.try_inverse()
                    .expect("Innovation covariance S is singular even with regularization")
            });
            let log_det = s.determinant().abs().max(1e-300).ln();
            let maha = (y.transpose() * &s_inv * &y)[(0, 0)];
            let log_lik = -0.5 * (m as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + maha);
            log_likelihoods.push(log_lik);

            // Now perform the actual EKF update using the observation model
            // We need to construct an H matrix in the model's native space
            // H_model = H_common * J_to_common, but for linear mappings
            // (CV, CA, CT) this is a simple index remapping.
            // Instead, we update in common space and back-project.
            //
            // Approach: update a temporary common-space EKF, then back-project.
            let mut common_ekf = ExtendedKalmanFilter::new(x_common, p_common);
            common_ekf.update_linear(z, h_common, r);
            let (ms, mc) = f.mapping.from_common(&common_ekf.x, &common_ekf.p);
            f.ekf.x = ms;
            f.ekf.p = mc;
        }

        log_likelihoods
    }

    /// Update mode probabilities using innovation likelihoods.
    fn update_mode_probabilities(&mut self, log_likelihoods: &[f64]) {
        let n = self.num_models();
        let mu = &self.mode_probabilities;
        let pi = &self.transition_matrix;

        // Predicted mode probabilities c_j = sum_i(pi_{ij} * mu_i)
        let mut c_bar: DVector<f64> = DVector::zeros(n);
        for j in 0..n {
            for i in 0..n {
                c_bar[j] += pi[(i, j)] * mu[i];
            }
        }

        // Log-space: log(c_j * Lambda_j)
        let mut log_weights: Vec<f64> = (0..n)
            .map(|j| c_bar[j].max(1e-300).ln() + log_likelihoods[j])
            .collect();

        // Log-sum-exp normalization
        let max_log = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum: f64 = log_weights
            .iter()
            .map(|&lw| (lw - max_log).exp())
            .sum::<f64>()
            .ln()
            + max_log;

        for lw in &mut log_weights {
            *lw -= log_sum;
        }

        // Exponentiate and clamp
        let mut new_mu = DVector::zeros(n);
        for j in 0..n {
            new_mu[j] = log_weights[j].exp().max(self.mode_probability_floor);
        }

        // Re-normalize after clamping
        let sum: f64 = new_mu.iter().copied().sum();
        new_mu /= sum;

        self.mode_probabilities = new_mu;
    }

    /// Combine model-conditioned estimates into a single output in common 6D space.
    pub fn combine(&self) -> (DVector<f64>, DMatrix<f64>) {
        // Project all states to common space
        let commons: Vec<(DVector<f64>, DMatrix<f64>)> = self
            .filters
            .iter()
            .map(|f| f.mapping.to_common(&f.ekf.x, &f.ekf.p))
            .collect();

        // Combined state
        let mut combined_state: DVector<f64> = DVector::zeros(6);
        for (j, (xc, _)) in commons.iter().enumerate() {
            let term: DVector<f64> = self.mode_probabilities[j] * xc;
            combined_state += term;
        }

        // Combined covariance
        let mut combined_cov: DMatrix<f64> = DMatrix::zeros(6, 6);
        for (j, (xc, pc)) in commons.iter().enumerate() {
            let delta = xc - &combined_state;
            let term: DMatrix<f64> = self.mode_probabilities[j] * (pc + &delta * delta.transpose());
            combined_cov += term;
        }

        (combined_state, combined_cov)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 6.1: State mapping round-trips
    #[test]
    fn cv_mapping_round_trip() {
        let mapping = CvMapping;
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = DMatrix::identity(6, 6) * 10.0;
        let (xc, pc) = mapping.to_common(&x, &p);
        let (xr, pr) = mapping.from_common(&xc, &pc);
        assert!((xr - &x).norm() < 1e-12);
        assert!((pr - &p).norm() < 1e-12);
    }

    #[test]
    fn ca_mapping_round_trip_preserves_shared() {
        let mapping = CaMapping;
        // CA state: [x, vx, ax, y, vy, ay, z, vz, az]
        let x = DVector::from_column_slice(&[1.0, 2.0, 0.5, 3.0, 4.0, 0.3, 5.0, 6.0, 0.1]);
        let p = DMatrix::identity(9, 9) * 10.0;
        let (xc, _pc) = mapping.to_common(&x, &p);
        // Common should have [x=1, vx=2, y=3, vy=4, z=5, vz=6]
        assert!((xc[0] - 1.0).abs() < 1e-12);
        assert!((xc[1] - 2.0).abs() < 1e-12);
        assert!((xc[2] - 3.0).abs() < 1e-12);
        assert!((xc[3] - 4.0).abs() < 1e-12);
        assert!((xc[4] - 5.0).abs() < 1e-12);
        assert!((xc[5] - 6.0).abs() < 1e-12);

        // Round-trip: shared components preserved
        let (xr, _pr) = mapping.from_common(&xc, &_pc);
        assert!((xr[0] - 1.0).abs() < 1e-12); // x
        assert!((xr[1] - 2.0).abs() < 1e-12); // vx
        assert!((xr[3] - 3.0).abs() < 1e-12); // y
        assert!((xr[4] - 4.0).abs() < 1e-12); // vy
        assert!((xr[6] - 5.0).abs() < 1e-12); // z
        assert!((xr[7] - 6.0).abs() < 1e-12); // vz
    }

    #[test]
    fn ctrv_mapping_round_trip_preserves_shared() {
        let mapping = CtrvMapping;
        // CTRV state: [x, y, theta, v, omega]
        let theta = 0.5_f64;
        let v = 100.0;
        let x = DVector::from_column_slice(&[10.0, 20.0, theta, v, 0.1]);
        let p = DMatrix::identity(5, 5) * 10.0;
        let (xc, _pc) = mapping.to_common(&x, &p);
        // Common: [x, vx=v*cos(theta), y, vy=v*sin(theta), z=0, vz=0]
        assert!((xc[0] - 10.0).abs() < 1e-12);
        assert!((xc[1] - v * theta.cos()).abs() < 1e-10);
        assert!((xc[2] - 20.0).abs() < 1e-12);
        assert!((xc[3] - v * theta.sin()).abs() < 1e-10);

        // Round-trip: position and velocity magnitude preserved
        let (xr, _pr) = mapping.from_common(&xc, &_pc);
        assert!((xr[0] - 10.0).abs() < 1e-10); // x
        assert!((xr[1] - 20.0).abs() < 1e-10); // y
        assert!((xr[2] - theta).abs() < 1e-10); // theta
        assert!((xr[3] - v).abs() < 1e-10); // v
    }

    #[test]
    fn ct_mapping_round_trip_preserves_shared() {
        let mapping = CtMapping;
        // CT state: [x, vx, y, vy, omega]
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 0.5]);
        let p = DMatrix::identity(5, 5) * 10.0;
        let (xc, _pc) = mapping.to_common(&x, &p);
        // Common: [x=1, vx=2, y=3, vy=4, z=0, vz=0]
        assert!((xc[0] - 1.0).abs() < 1e-12);
        assert!((xc[1] - 2.0).abs() < 1e-12);
        assert!((xc[2] - 3.0).abs() < 1e-12);
        assert!((xc[3] - 4.0).abs() < 1e-12);

        let (xr, _pr) = mapping.from_common(&xc, &_pc);
        assert!((xr[0] - 1.0).abs() < 1e-12);
        assert!((xr[1] - 2.0).abs() < 1e-12);
        assert!((xr[2] - 3.0).abs() < 1e-12);
        assert!((xr[3] - 4.0).abs() < 1e-12);
    }

    // 6.2: Config validation
    #[test]
    fn config_validation_rejects_mismatched_dims() {
        let config = ImmConfig {
            models: vec![
                Box::new(ConstantVelocity::new(1.0)),
                Box::new(ConstantAcceleration::new(1.0)),
            ],
            mappings: vec![Box::new(CvMapping)], // only 1 mapping for 2 models
            transition_matrix: DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]),
            initial_mode_probabilities: DVector::from_column_slice(&[0.5, 0.5]),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validation_rejects_non_stochastic_tpm() {
        let config = ImmConfig {
            models: vec![
                Box::new(ConstantVelocity::new(1.0)),
                Box::new(ConstantAcceleration::new(1.0)),
            ],
            mappings: vec![Box::new(CvMapping), Box::new(CaMapping)],
            transition_matrix: DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.05, 0.95]), // row 0 sums to 0.9
            initial_mode_probabilities: DVector::from_column_slice(&[0.5, 0.5]),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validation_rejects_bad_probabilities() {
        let config = ImmConfig {
            models: vec![
                Box::new(ConstantVelocity::new(1.0)),
                Box::new(ConstantAcceleration::new(1.0)),
            ],
            mappings: vec![Box::new(CvMapping), Box::new(CaMapping)],
            transition_matrix: DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]),
            initial_mode_probabilities: DVector::from_column_slice(&[0.3, 0.3]), // sums to 0.6
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validation_accepts_valid() {
        let config = ImmConfig::cv_ca(1.0, 1.0);
        assert!(config.validate().is_ok());

        let config = ImmConfig::cv_ctrv(1.0, 1.0, 0.1);
        assert!(config.validate().is_ok());

        let config = ImmConfig::cv_ca_ctrv_ct(1.0, 1.0, 1.0, 0.1);
        assert!(config.validate().is_ok());
    }

    // 6.3: Interaction step invariant
    #[test]
    fn interaction_uniform_identical_states_unchanged() {
        let config = ImmConfig::cv_ca(1.0, 1.0);
        let x0 = DVector::from_column_slice(&[100.0, 10.0, 200.0, 5.0, 0.0, 0.0]);
        let p0 = DMatrix::identity(6, 6) * 50.0;
        let mut imm = ImmFilter::new(config, &x0, &p0);

        // Store pre-interaction states in common space
        let pre: Vec<(DVector<f64>, DMatrix<f64>)> = imm
            .filters
            .iter()
            .map(|f| f.mapping.to_common(&f.ekf.x, &f.ekf.p))
            .collect();

        imm.interaction_step();

        // After interaction with uniform probs and identical common states,
        // the common-space states should be approximately unchanged.
        for (j, f) in imm.filters.iter().enumerate() {
            let (xc, _) = f.mapping.to_common(&f.ekf.x, &f.ekf.p);
            // The shared components should match
            for k in 0..4 {
                // x, vx, y, vy
                assert!(
                    (xc[k] - pre[j].0[k]).abs() < 1.0,
                    "Model {j}, component {k}: pre={}, post={}",
                    pre[j].0[k],
                    xc[k]
                );
            }
        }
    }

    // 6.4: Mode probability shift
    #[test]
    fn mode_probability_shifts_to_better_model() {
        let config = ImmConfig::cv_ca(1.0, 1.0);
        let x0 = DVector::from_column_slice(&[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let p0 = DMatrix::identity(6, 6) * 100.0;
        let mut imm = ImmFilter::new(config, &x0, &p0);

        // Position-only observation of x, y, z
        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 25.0;

        // Straight-line trajectory: CV should have smaller innovations than CA
        let dt = 1.0;
        for step in 0..10 {
            let t = (step + 1) as f64 * dt;
            let z = DVector::from_column_slice(&[100.0 * t, 0.0, 0.0]);
            imm.step(dt, &z, &h, &r);
        }

        // CV (mode 0) should have higher probability than CA (mode 1) for straight line
        assert!(
            imm.mode_probabilities[0] > imm.mode_probabilities[1],
            "Expected CV dominant for straight line: mu = {:?}",
            imm.mode_probabilities
        );
    }

    // 6.5: CV+CA on straight line -> CV dominant
    #[test]
    fn cv_ca_straight_line_cv_dominant() {
        let config = ImmConfig::cv_ca(5.0, 1.0);
        let x0 = DVector::from_column_slice(&[0.0, 50.0, 0.0, 0.0, 0.0, 0.0]);
        let p0 = DMatrix::identity(6, 6) * 100.0;
        let mut imm = ImmFilter::new(config, &x0, &p0);

        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 25.0;

        // Constant velocity: 50 m/s along x
        let dt = 1.0;
        for step in 0..20 {
            let t = (step + 1) as f64 * dt;
            let z = DVector::from_column_slice(&[50.0 * t, 0.0, 0.0]);
            imm.step(dt, &z, &h, &r);
        }

        assert!(
            imm.mode_probabilities[0] > 0.8,
            "CV mode probability should be > 0.8 after 20 steps on straight line, got {}",
            imm.mode_probabilities[0]
        );
    }

    // 6.6: CV+CTRV mode switching on maneuver
    #[test]
    fn cv_ctrv_mode_switching_on_maneuver() {
        // Use lower process noise for CV so the turn mismatch is more visible,
        // and a higher turn rate + more steps so CTRV has time to dominate.
        let config = ImmConfig::cv_ctrv(1.0, 5.0, 0.5);
        let x0 = DVector::from_column_slice(&[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let p0 = DMatrix::identity(6, 6) * 100.0;
        let mut imm = ImmFilter::new(config, &x0, &p0);

        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
            ],
        );
        let r = DMatrix::identity(3, 3) * 10.0;
        let dt = 1.0;

        // Phase 1: straight flight (15 steps at dt=1)
        let ctrv_true = Ctrv::new(0.0, 0.0);
        let mut true_state = DVector::from_column_slice(&[0.0, 0.0, 0.0, 100.0, 0.0]);

        // Single-model filters for RMSE comparison
        let config_cv_only = ImmConfig {
            models: vec![Box::new(ConstantVelocity::new(1.0))],
            mappings: vec![Box::new(CvMapping)],
            transition_matrix: DMatrix::from_row_slice(1, 1, &[1.0]),
            initial_mode_probabilities: DVector::from_column_slice(&[1.0]),
        };
        let config_ctrv_only = ImmConfig {
            models: vec![Box::new(Ctrv::new(5.0, 0.5))],
            mappings: vec![Box::new(CtrvMapping)],
            transition_matrix: DMatrix::from_row_slice(1, 1, &[1.0]),
            initial_mode_probabilities: DVector::from_column_slice(&[1.0]),
        };
        let mut cv_only = ImmFilter::new(config_cv_only, &x0, &p0);
        let mut ctrv_only = ImmFilter::new(config_ctrv_only, &x0, &p0);

        let mut imm_sse = 0.0;
        let mut cv_sse = 0.0;
        let mut ctrv_sse = 0.0;
        let mut count = 0.0;

        for _ in 0..15 {
            true_state = ctrv_true.predict(&true_state, dt);
            let z = DVector::from_column_slice(&[true_state[0], true_state[1], 0.0]);

            let res = imm.step(dt, &z, &h, &r);
            let res_cv = cv_only.step(dt, &z, &h, &r);
            let res_ctrv = ctrv_only.step(dt, &z, &h, &r);

            imm_sse +=
                (res.state[0] - true_state[0]).powi(2) + (res.state[2] - true_state[1]).powi(2);
            cv_sse += (res_cv.state[0] - true_state[0]).powi(2)
                + (res_cv.state[2] - true_state[1]).powi(2);
            ctrv_sse += (res_ctrv.state[0] - true_state[0]).powi(2)
                + (res_ctrv.state[2] - true_state[1]).powi(2);
            count += 1.0;
        }

        let cv_prob_straight = imm.mode_probabilities[0];

        // Phase 2: coordinated turn (40 steps) - omega = 0.15 rad/s
        // At v=100 m/s this is a ~570m radius turn, producing significant lateral acceleration.
        true_state[4] = 0.15;
        for _ in 0..40 {
            true_state = ctrv_true.predict(&true_state, dt);
            let z = DVector::from_column_slice(&[true_state[0], true_state[1], 0.0]);

            let res = imm.step(dt, &z, &h, &r);
            let res_cv = cv_only.step(dt, &z, &h, &r);
            let res_ctrv = ctrv_only.step(dt, &z, &h, &r);

            imm_sse +=
                (res.state[0] - true_state[0]).powi(2) + (res.state[2] - true_state[1]).powi(2);
            cv_sse += (res_cv.state[0] - true_state[0]).powi(2)
                + (res_cv.state[2] - true_state[1]).powi(2);
            ctrv_sse += (res_ctrv.state[0] - true_state[0]).powi(2)
                + (res_ctrv.state[2] - true_state[1]).powi(2);
            count += 1.0;
        }

        let ctrv_prob_turn = imm.mode_probabilities[1];

        // Mode probabilities should have shifted: CV dominant during straight,
        // CTRV should gain significant weight during the turn.
        assert!(
            cv_prob_straight > 0.5,
            "CV should be dominant during straight flight: {cv_prob_straight}"
        );
        assert!(
            ctrv_prob_turn > cv_prob_straight.min(0.3),
            "CTRV should gain weight during turn: CTRV={ctrv_prob_turn}, CV_straight={cv_prob_straight}"
        );

        // IMM RMSE should be competitive (not worse than the worse single model)
        let imm_rmse = (imm_sse / count).sqrt();
        let cv_rmse = (cv_sse / count).sqrt();
        let ctrv_rmse = (ctrv_sse / count).sqrt();
        let worst_single = cv_rmse.max(ctrv_rmse);
        assert!(
            imm_rmse < worst_single * 1.5,
            "IMM RMSE ({imm_rmse:.2}) should not be much worse than worst single model ({worst_single:.2})"
        );
    }
}
