//! Integration tests for the radar-scene PyO3 bridge.
//!
//! These tests are only compiled when the `radar-scene` feature is enabled
//! and are all `#[ignore]` by default because they require the Python
//! `radarsimpy` package to be installed. Run with:
//!
//! ```text
//! cargo test -p thresh-synth --features radar-scene -- --ignored
//! ```

#![cfg(feature = "radar-scene")]

use thresh_synth::radar_scene::*;

#[test]
#[ignore] // requires radarsimpy Python package
fn simulate_single_target_scene() {
    let mut scene = RadarScene::x_band_monostatic([0.0, 0.0, 0.0]);
    scene.add_target(SceneTarget {
        position: [10_000.0, 0.0, 1_000.0],
        velocity: [0.0, 0.0, 0.0],
        rcs_m2: 1.0,
    });
    let _detections = RadarScenePyBridge::simulate(&scene).expect("simulate");
}

#[test]
#[ignore] // requires radarsimpy Python package
fn simulate_target_below_noise_floor_yields_no_detection() {
    let mut scene = RadarScene::x_band_monostatic([0.0, 0.0, 0.0]);
    // Extremely weak target at very long range.
    scene.add_target(SceneTarget {
        position: [500_000.0, 0.0, 100.0],
        velocity: [0.0, 0.0, 0.0],
        rcs_m2: 1e-6,
    });
    let detections = RadarScenePyBridge::simulate(&scene).expect("simulate");
    assert!(
        detections.is_empty(),
        "weak target should not produce a detection"
    );
}
