//! Headless egui-app tests using `egui_kittest`.
//!
//! These exercise the dashboard end-to-end without opening a window:
//! run a few frames, simulate keyboard input, then assert the app
//! state changed as expected. No GPU is required — `egui_kittest`'s
//! default backend is CPU-only.
//!
//! Gated to the `gui` feature.

#![cfg(feature = "gui")]

use eframe::egui;
use egui_kittest::Harness;

use thresh_viz::app::ThreshVizApp;
use thresh_viz::recording::{Recording, VizDetection, VizFrame, VizGroundTruth, VizTrack};

fn make_recording() -> Recording {
    let mut rec = Recording::new("kittest");
    for i in 0..5 {
        let t = i as f64;
        rec.push_frame(VizFrame {
            timestamp: t,
            tracks: vec![VizTrack {
                id: 1,
                position: [t * 10.0, 0.0, 0.0],
                velocity: [10.0, 0.0, 0.0],
                covariance_diag: [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
                is_confirmed: true,
                class_label: None,
            }],
            detections: vec![VizDetection {
                position: [t * 10.0 + 1.0, 0.0, 0.0],
                sensor_id: 0,
            }],
            ground_truth: vec![VizGroundTruth {
                id: 1,
                position: [t * 10.0, 0.0, 0.0],
            }],
            associations: vec![(0, 1)],
            events: Vec::new(),
        });
    }
    rec
}

fn build_harness() -> Harness<'static, ThreshVizApp> {
    Harness::new_eframe(|_cc| ThreshVizApp::new(Some(make_recording())))
}

#[test]
fn app_renders_one_frame_without_panicking() {
    let mut harness = build_harness();
    harness.run();
    // If we got here, the central panel + plot + sidebar + metrics all
    // rendered without panic. The recording has GT, so MotMetricsBuilder
    // ran on the first frame too.
    let app = harness.state();
    // Help overlay starts hidden.
    assert!(!app.help_overlay_open());
    // Ellipses start hidden, associations start visible.
    assert!(!app.ellipses_shown());
    assert!(app.associations_shown());
}

#[test]
fn pressing_help_key_opens_then_closes_overlay() {
    let mut harness = build_harness();
    harness.run();
    assert!(!harness.state().help_overlay_open());

    harness.press_key(egui::Key::Questionmark);
    harness.run();
    assert!(harness.state().help_overlay_open(), "? should open overlay");

    // Pressing Escape closes it.
    harness.press_key(egui::Key::Escape);
    harness.run();
    assert!(
        !harness.state().help_overlay_open(),
        "Esc should close overlay"
    );
}

#[test]
fn pressing_e_toggles_covariance_ellipses() {
    let mut harness = build_harness();
    harness.run();
    let initial = harness.state().ellipses_shown();
    harness.press_key(egui::Key::E);
    harness.run();
    assert_ne!(harness.state().ellipses_shown(), initial);
    harness.press_key(egui::Key::E);
    harness.run();
    assert_eq!(harness.state().ellipses_shown(), initial);
}

#[test]
fn pressing_a_toggles_association_lines() {
    let mut harness = build_harness();
    harness.run();
    let initial = harness.state().associations_shown();
    harness.press_key(egui::Key::A);
    harness.run();
    assert_ne!(harness.state().associations_shown(), initial);
}

#[test]
fn pressing_l_toggles_event_log_panel() {
    let mut harness = build_harness();
    harness.run();
    let initial = harness.state().event_log_visible();
    harness.press_key(egui::Key::L);
    harness.run();
    assert_ne!(harness.state().event_log_visible(), initial);
}
