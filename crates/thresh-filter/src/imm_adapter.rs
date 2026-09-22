//! Learned IMM mode-probability adapter (requires the `learned-imm` feature).
//!
//! Track B of the flight-data-training-pipeline: an ONNX classifier predicts
//! IMM mode probabilities `(p_CV, p_CA, p_CTRV, p_coord_turn)` from a sliding
//! window of **filter-state** projections (the 13-dim
//! [`project_filter_state`](crate::imm::project_filter_state) output — state +
//! covariance diagonal + elapsed seconds), and those probabilities replace the analytic
//! mode-transition update in the IMM blend.
//!
//! Two pieces:
//! - [`ImmModeAdapter`] — loads the ONNX checkpoint and runs one window.
//! - [`LearnedImmFilter`] — wraps an [`ImmFilter`], feeding it the learned
//!   probabilities once a full window of history exists, and **falling back to
//!   the analytic update** (with a one-time warning) on any classifier error.
//!
//! The wrapper deliberately leaves the core [`ImmFilter`] untouched: it sets
//! the public `mode_probabilities` field and re-runs the public `combine()`
//! step, so every existing analytic-IMM test holds with the feature enabled.

use std::collections::VecDeque;
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use thresh_inference::session::OnnxModel;

use crate::imm::{CLASSIFIER_FEATURE_DIM, ImmFilter, ImmStepResult, project_filter_state};

/// Sliding-window length (timesteps) the classifier consumes.
pub const WINDOW_LEN: usize = 10;
/// Number of IMM modes scored: CV, CA, CTRV, CoordTurn (matches `cv_ca_ctrv_ct`).
pub const NUM_MODES: usize = 4;

/// ONNX-backed IMM mode classifier.
///
/// Expects an ONNX model with input shape `(1, WINDOW_LEN, CLASSIFIER_FEATURE_DIM)`
/// = `(1, 10, 13)` and output shape `(1, NUM_MODES)` = `(1, 4)` whose values are
/// (approximately) a probability distribution. Features MUST come from
/// [`project_filter_state`] — never from raw measurements.
pub struct ImmModeAdapter {
    model: OnnxModel,
}

impl ImmModeAdapter {
    /// Load the classifier from an ONNX checkpoint.
    ///
    /// # Errors
    /// Returns a message if the model cannot be loaded.
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Self, String> {
        let model = OnnxModel::load(path)
            .map_err(|e| format!("failed to load IMM classifier ONNX: {e}"))?;
        validate_model_contract(&model)?;
        let mut adapter = Self { model };
        adapter.predict(&vec![vec![0.0; CLASSIFIER_FEATURE_DIM]; WINDOW_LEN])?;
        Ok(adapter)
    }

    /// Predict mode probabilities from a `WINDOW_LEN` × `CLASSIFIER_FEATURE_DIM`
    /// window of filter-state projections.
    ///
    /// # Errors
    /// Returns a message (so the caller can fall back to the analytic update) if
    /// the window shape disagrees with the contract or inference fails.
    pub fn predict(&mut self, window: &[Vec<f64>]) -> Result<[f64; NUM_MODES], String> {
        if window.len() != WINDOW_LEN {
            return Err(format!("window length {} != {WINDOW_LEN}", window.len()));
        }
        let mut flat = Vec::with_capacity(WINDOW_LEN * CLASSIFIER_FEATURE_DIM);
        for (i, step) in window.iter().enumerate() {
            if step.len() != CLASSIFIER_FEATURE_DIM {
                return Err(format!(
                    "window[{i}] has dim {} != {CLASSIFIER_FEATURE_DIM}",
                    step.len()
                ));
            }
            if step.iter().any(|&v| !(v as f32).is_finite()) || step[12] < 0.0 {
                return Err(format!(
                    "window[{i}] must be finite float32 values with nonnegative elapsed_seconds"
                ));
            }
            flat.extend(step.iter().map(|&v| v as f32));
        }
        let shape = [1, WINDOW_LEN as i64, CLASSIFIER_FEATURE_DIM as i64];
        let out = self
            .model
            .run_f32(&flat, &shape)
            .map_err(|e| format!("IMM classifier inference failed: {e}"))?;
        if out.len() != NUM_MODES {
            return Err(format!(
                "classifier output length {} != {NUM_MODES}",
                out.len()
            ));
        }
        Ok(normalize(&out))
    }
}

/// Reject legacy shapes at load time, before a track silently falls back.
fn validate_model_contract(model: &OnnxModel) -> Result<(), String> {
    let inputs = model.session().inputs();
    let outputs = model.session().outputs();
    let valid_input = inputs.len() == 1
        && inputs[0].name() == "features"
        && inputs[0].dtype().tensor_shape().is_some_and(|shape| {
            shape.len() == 3
                && matches!(shape[0], -1 | 1)
                && shape[1] == WINDOW_LEN as i64
                && shape[2] == CLASSIFIER_FEATURE_DIM as i64
        });
    let valid_output = outputs.len() == 1
        && outputs[0].name() == "mode_probs"
        && outputs[0].dtype().tensor_shape().is_some_and(|shape| {
            shape.len() == 2 && matches!(shape[0], -1 | 1) && shape[1] == NUM_MODES as i64
        });
    if !valid_input || !valid_output {
        return Err(format!(
            "IMM classifier requires features [batch, {WINDOW_LEN}, {CLASSIFIER_FEATURE_DIM}] \
             -> mode_probs [batch, {NUM_MODES}]; re-export legacy models with elapsed_seconds"
        ));
    }
    Ok(())
}

/// Normalize 4 model outputs into a probability distribution. The exported
/// graph already applies softmax, so this just guards against numerical drift.
fn normalize(values: &[f32]) -> [f64; NUM_MODES] {
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    let denom = if sum.abs() < 1e-12 { 1.0 } else { sum };
    let mut out = [0.0_f64; NUM_MODES];
    for (o, v) in out.iter_mut().zip(values.iter()) {
        *o = (*v as f64) / denom;
    }
    out
}

/// Index of the largest probability.
fn argmax(probs: &[f64; NUM_MODES]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// An [`ImmFilter`] whose mode-probability update is supplied by a learned
/// classifier once enough history has accumulated.
///
/// Drive it like an `ImmFilter`: [`predict`](Self::predict) then
/// [`update_with_measurement`](Self::update_with_measurement). The wrapper keeps
/// a rolling `WINDOW_LEN` window of the filter's own state projections; until it
/// is full (and on any classifier error) the analytic IMM result is returned
/// unchanged.
pub struct LearnedImmFilter {
    imm: ImmFilter,
    adapter: ImmModeAdapter,
    history: VecDeque<Vec<f64>>,
    fallback_logged: bool,
    elapsed_since_update: f64,
}

impl LearnedImmFilter {
    /// Wrap an analytic IMM filter with a learned mode classifier.
    ///
    /// # Errors
    /// Returns a message if the IMM bank does not have exactly [`NUM_MODES`]
    /// models (the classifier is trained for the 4-model `cv_ca_ctrv_ct` bank).
    pub fn new(imm: ImmFilter, adapter: ImmModeAdapter) -> Result<Self, String> {
        if imm.num_models() != NUM_MODES {
            return Err(format!(
                "LearnedImmFilter requires a {NUM_MODES}-model IMM bank, got {}",
                imm.num_models()
            ));
        }
        Ok(Self {
            imm,
            adapter,
            history: VecDeque::with_capacity(WINDOW_LEN),
            fallback_logged: false,
            elapsed_since_update: 0.0,
        })
    }

    /// The wrapped analytic IMM filter.
    pub fn inner(&self) -> &ImmFilter {
        &self.imm
    }

    /// IMM predict step, accumulating all intervals through missed observations.
    ///
    /// # Panics
    /// Rejects negative/nonfinite intervals or float32 overflow before mutation.
    pub fn predict(&mut self, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let elapsed = self.elapsed_since_update + dt;
        assert!(
            dt >= 0.0 && (elapsed as f32).is_finite(),
            "learned IMM dt and accumulated elapsed_seconds must be finite, nonnegative and fit float32"
        );
        self.elapsed_since_update = elapsed;
        self.imm.predict(dt)
    }

    /// IMM measurement update. Once a full `WINDOW_LEN` history exists and the
    /// classifier succeeds, the learned mode probabilities replace the analytic
    /// ones and the estimate is re-blended; otherwise the analytic result is
    /// returned unchanged.
    pub fn update_with_measurement(
        &mut self,
        z: &DVector<f64>,
        h: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> ImmStepResult {
        let analytic = self.imm.update_with_measurement(z, h, r);
        let elapsed = if self.history.is_empty() {
            0.0
        } else {
            self.elapsed_since_update
        };
        self.history.push_back(project_filter_state(
            &analytic.state,
            &analytic.covariance,
            elapsed,
        ));
        self.elapsed_since_update = 0.0;
        while self.history.len() > WINDOW_LEN {
            self.history.pop_front();
        }
        if self.history.len() < WINDOW_LEN {
            return analytic; // warming up: not enough history yet
        }
        let window: Vec<Vec<f64>> = self.history.iter().cloned().collect();
        match self.adapter.predict(&window) {
            Ok(probs) => self.apply_learned(probs),
            Err(e) => {
                if !self.fallback_logged {
                    eprintln!("learned-imm: falling back to analytic mode update ({e})");
                    self.fallback_logged = true;
                }
                analytic
            }
        }
    }

    /// Override the IMM mode probabilities with the learned distribution and
    /// re-blend the model-conditioned estimates.
    fn apply_learned(&mut self, probs: [f64; NUM_MODES]) -> ImmStepResult {
        self.imm.mode_probabilities = DVector::from_row_slice(&probs);
        let (state, covariance) = self.imm.combine();
        ImmStepResult {
            state,
            covariance,
            mode_probabilities: DVector::from_row_slice(&probs),
            dominant_mode: argmax(&probs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_makes_a_distribution() {
        let p = normalize(&[1.0, 1.0, 1.0, 1.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|&v| (v - 0.25).abs() < 1e-12));
    }

    #[test]
    fn normalize_handles_all_zero() {
        let p = normalize(&[0.0, 0.0, 0.0, 0.0]);
        assert!(p.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.2, 0.6, 0.1]), 2);
        assert_eq!(argmax(&[0.7, 0.1, 0.1, 0.1]), 0);
    }

    fn learned_filter() -> LearnedImmFilter {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-data/models/imm_mode_classifier.onnx");
        let adapter = ImmModeAdapter::from_onnx(path).unwrap();
        let imm = ImmFilter::new(
            crate::imm::ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1),
            &DVector::zeros(6),
            &DMatrix::identity(6, 6),
        );
        LearnedImmFilter::new(imm, adapter).unwrap()
    }

    fn update(filter: &mut LearnedImmFilter) {
        let mut h = DMatrix::zeros(3, 6);
        for i in 0..3 {
            h[(i, i * 2)] = 1.0;
        }
        filter.update_with_measurement(&DVector::zeros(3), &h, &DMatrix::identity(3, 3));
    }

    #[test]
    fn history_timing_distinguishes_rates_and_preserves_sliding_intervals() {
        for dt in [0.1, 1.0] {
            let mut filter = learned_filter();
            assert!(filter.history.is_empty(), "birth is not an update");
            filter.predict(dt);
            update(&mut filter);
            assert_eq!(filter.history[0][12], 0.0);
            for _ in 0..WINDOW_LEN {
                filter.predict(dt);
                update(&mut filter);
            }
            assert_eq!(filter.history.len(), WINDOW_LEN);
            assert!(filter.history.iter().all(|row| row[12] == dt));
        }
    }

    #[test]
    fn missed_updates_accumulate_all_prediction_intervals() {
        let mut filter = learned_filter();
        filter.predict(0.1);
        update(&mut filter);
        for _ in 0..3 {
            filter.predict(0.1);
        }
        assert_eq!(filter.history.len(), 1, "coasting must not append history");
        update(&mut filter);
        assert!((filter.history[1][12] - 0.3).abs() < 1e-12);
        update(&mut filter);
        assert_eq!(
            filter.history[2][12], 0.0,
            "no prediction means no elapsed time"
        );
    }

    #[test]
    fn invalid_prediction_intervals_are_rejected_before_mutation() {
        for dt in [-0.1, f64::NAN, f64::INFINITY, f64::MAX] {
            let mut filter = learned_filter();
            let before = filter.inner().combine();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                filter.predict(dt);
            }));
            assert!(result.is_err());
            assert_eq!(filter.elapsed_since_update, 0.0);
            assert_eq!(filter.inner().combine(), before);
        }
    }
}
