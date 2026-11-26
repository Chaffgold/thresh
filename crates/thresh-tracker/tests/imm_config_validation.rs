//! IMM configuration validation through the tracker constructors
//! (`imm-estimation`, "IMM configuration validation").
//!
//! The infallible `MultiObjectTracker::new_imm_position` refuses a rejected
//! configuration by panicking, so it never builds a filter bank from one.
//! The fallible `new_imm_position_learned` (feature `learned-imm`) returns
//! an error instead, before it loads any model. Run the gated test with
//! `cargo test -p thresh-tracker --features learned-imm`.

use nalgebra::{DMatrix, DVector};
use thresh_filter::imm::ImmConfig;
use thresh_tracker::tracker::MultiObjectTracker;

/// The 4-model `cv_ca_ctrv_ct` preset (the bank the learned path requires)
/// with transition-matrix entry `(0, 1)` replaced by `entry`. Row 0 keeps
/// its sum when `entry` is finite.
fn cv_ca_ctrv_ct_with_entry_0_1(entry: f64) -> ImmConfig {
    let mut config = ImmConfig::cv_ca_ctrv_ct(5.0, 1.0, 2.0, 0.1);
    let old = config.transition_matrix[(0, 1)];
    config.transition_matrix[(0, 1)] = entry;
    if entry.is_finite() {
        config.transition_matrix[(0, 0)] += old - entry;
    }
    config
}

/// `cv_ca(5, 1)` with the given initial mode probabilities.
fn cv_ca_with_initial_probabilities(mu: [f64; 2]) -> ImmConfig {
    let mut config = ImmConfig::cv_ca(5.0, 1.0);
    config.initial_mode_probabilities = DVector::from_column_slice(&mu);
    config
}

#[test]
#[should_panic(expected = "transition matrix entry (0, 1)")]
fn new_imm_position_panics_on_nan_transition_entry() {
    let _ = MultiObjectTracker::new_imm_position(
        || cv_ca_ctrv_ct_with_entry_0_1(f64::NAN),
        10.0,
        100.0,
    );
}

#[test]
#[should_panic(expected = "transition matrix entry (0, 1)")]
fn new_imm_position_panics_on_negative_transition_entry() {
    let _ = MultiObjectTracker::new_imm_position(
        || {
            let mut config = ImmConfig::cv_ca(5.0, 1.0);
            config.transition_matrix = DMatrix::from_row_slice(2, 2, &[1.2, -0.2, 0.05, 0.95]);
            config
        },
        10.0,
        100.0,
    );
}

#[test]
#[should_panic(expected = "initial mode probability 1")]
fn new_imm_position_panics_on_negative_initial_probability() {
    let _ = MultiObjectTracker::new_imm_position(
        || cv_ca_with_initial_probabilities([1.5, -0.5]),
        10.0,
        100.0,
    );
}

#[cfg(feature = "learned-imm")]
#[test]
fn new_imm_position_learned_returns_error_for_nan_transition_entry() {
    // The committed stub classifier loads, so the configuration is the only
    // reason to fail.
    let stub = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/models/imm_mode_classifier.onnx"
    );
    let result = MultiObjectTracker::new_imm_position_learned(
        || cv_ca_ctrv_ct_with_entry_0_1(f64::NAN),
        stub,
        10.0,
        100.0,
    );
    let err = result
        .err()
        .expect("an invalid configuration must be rejected");
    assert!(err.contains("invalid ImmConfig"), "unexpected error: {err}");
    assert!(
        err.contains("transition matrix entry (0, 1)"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "learned-imm")]
#[test]
fn new_imm_position_learned_validates_before_loading_the_classifier() {
    // The model path does not exist, so this returns a configuration error
    // only if validation runs before the classifier is loaded.
    let missing = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/models/does-not-exist.onnx"
    );
    let result = MultiObjectTracker::new_imm_position_learned(
        || cv_ca_ctrv_ct_with_entry_0_1(f64::NAN),
        missing,
        10.0,
        100.0,
    );
    let err = result
        .err()
        .expect("an invalid configuration must be rejected");
    assert!(err.contains("invalid ImmConfig"), "unexpected error: {err}");
}
