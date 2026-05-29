//! Integration tests for the learned IMM mode classifier (task 6.7).
//!
//! Run with: `cargo test -p thresh-filter --features learned-imm`.
//! The whole file is empty without the feature so default builds skip it.
#![cfg(feature = "learned-imm")]

use std::path::PathBuf;

use nalgebra::{DMatrix, DVector};
use thresh_filter::imm::{CLASSIFIER_FEATURE_DIM, ImmConfig, ImmFilter};
use thresh_filter::imm_adapter::{ImmModeAdapter, LearnedImmFilter, NUM_MODES, WINDOW_LEN};

/// Path to the committed stub classifier (relative to this crate).
fn stub_onnx() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/models/imm_mode_classifier.onnx")
}

/// Position-only 3x6 observation matrix for common state `[x, vx, y, vy, z, vz]`.
fn position_h() -> DMatrix<f64> {
    let mut h = DMatrix::zeros(3, 6);
    h[(0, 0)] = 1.0;
    h[(1, 2)] = 1.0;
    h[(2, 4)] = 1.0;
    h
}

#[test]
fn adapter_loads_stub_and_returns_distribution() {
    let mut adapter = ImmModeAdapter::from_onnx(stub_onnx()).expect("load stub onnx");
    let window: Vec<Vec<f64>> = (0..WINDOW_LEN)
        .map(|i| {
            (0..CLASSIFIER_FEATURE_DIM)
                .map(|j| (i + j) as f64 * 0.01)
                .collect()
        })
        .collect();

    let probs = adapter.predict(&window).expect("predict");
    assert_eq!(probs.len(), NUM_MODES);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probs must sum to 1, got {sum} ({probs:?})"
    );
    assert!(
        probs.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "probs in [0,1]: {probs:?}"
    );
}

#[test]
fn adapter_rejects_malformed_windows() {
    let mut adapter = ImmModeAdapter::from_onnx(stub_onnx()).expect("load stub onnx");

    // Too few timesteps.
    let short = vec![vec![0.0; CLASSIFIER_FEATURE_DIM]; WINDOW_LEN - 1];
    assert!(adapter.predict(&short).is_err());

    // Wrong feature width.
    let wide = vec![vec![0.0; CLASSIFIER_FEATURE_DIM + 1]; WINDOW_LEN];
    assert!(adapter.predict(&wide).is_err());
}

#[test]
fn learned_filter_requires_four_model_bank() {
    let x0 = DVector::from_column_slice(&[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]);
    let p0 = DMatrix::identity(6, 6) * 100.0;
    // 2-model bank must be rejected.
    let imm = ImmFilter::new(ImmConfig::cv_ca(5.0, 1.0), &x0, &p0);
    let adapter = ImmModeAdapter::from_onnx(stub_onnx()).expect("load stub onnx");
    assert!(LearnedImmFilter::new(imm, adapter).is_err());
}

#[test]
fn learned_filter_drives_a_track_and_blends_with_learned_probs() {
    let dt = 1.0;
    let x0 = DVector::from_column_slice(&[0.0, 50.0, 0.0, 0.0, 1000.0, 0.0]);
    let p0 = DMatrix::identity(6, 6) * 100.0;
    let imm = ImmFilter::new(ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1), &x0, &p0);
    let adapter = ImmModeAdapter::from_onnx(stub_onnx()).expect("load stub onnx");
    let mut filter = LearnedImmFilter::new(imm, adapter).expect("4-model bank");

    let h = position_h();
    let r = DMatrix::identity(3, 3) * 100.0;

    // Constant-velocity target along +x at 50 m/s.
    let mut last = None;
    for step in 0..(WINDOW_LEN + 5) {
        filter.predict(dt);
        let x = 50.0 * (step as f64 + 1.0) * dt;
        let z = DVector::from_column_slice(&[x, 0.0, 1000.0]);
        let result = filter.update_with_measurement(&z, &h, &r);

        // Mode probabilities are always a valid 4-way distribution.
        assert_eq!(result.mode_probabilities.len(), NUM_MODES);
        let sum: f64 = result.mode_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "step {step}: probs sum {sum}");
        assert!(result.state.iter().all(|v| v.is_finite()));
        assert!(result.dominant_mode < NUM_MODES);
        last = Some(result);
    }

    // After a full window the state estimate should track the target's x.
    let final_x = last.unwrap().state[0];
    let truth_x = 50.0 * (WINDOW_LEN + 5) as f64 * dt;
    assert!(
        (final_x - truth_x).abs() < 200.0,
        "tracked x={final_x} should be near truth {truth_x}"
    );
}
