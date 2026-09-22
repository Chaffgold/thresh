//! Learned-mode requests must never report analytic fallback metrics.
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_eval-tracker"))
        .args(args)
        .output()
        .expect("run eval-tracker")
}

fn assert_rejected(args: &[&str], message: &str) {
    let output = run(args);
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "rejected run must not print metrics"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(message), "{stderr}");
}

#[test]
fn rejects_unknown_flags_and_orphan_model_options() {
    assert_rejected(&["--json", "--typo"], "unknown argument");
    assert_rejected(
        &["--json", "--imm-model", "unused.onnx"],
        "requires --learned-imm",
    );
    assert_rejected(
        &["--json", "--detector-model", "unused.onnx"],
        "requires --learned-detector",
    );
    assert_rejected(
        &["--json", "--detector-baseline-model", "unused.onnx"],
        "requires --learned-detector",
    );
    assert_rejected(&["--json", "--detector-model"], "requires a model path");
}

#[test]
fn learned_imm_requires_feature_and_model_even_in_json_mode() {
    let expected = if cfg!(feature = "learned-imm") {
        "requires --imm-model"
    } else {
        "--features learned-imm"
    };
    assert_rejected(&["--json", "--learned-imm"], expected);
}

#[test]
fn detector_requires_feature_and_model_even_in_json_mode() {
    let expected = if cfg!(feature = "onnx") {
        "requires --detector-model"
    } else {
        "--features onnx"
    };
    assert_rejected(&["--json", "--learned-detector"], expected);
}

#[test]
fn rejects_invalid_durations() {
    for duration in ["0", "-1", "NaN", "inf", "invalid"] {
        assert_rejected(
            &["--json", "--duration-seconds", duration],
            "positive finite number",
        );
    }
    assert_rejected(&["--duration-seconds"], "positive finite number");
}

#[cfg(feature = "onnx")]
fn model(name: &str) -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/models")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

#[cfg(feature = "onnx")]
#[test]
fn missing_detector_model_is_an_error_not_a_baseline() {
    assert_rejected(
        &[
            "--json",
            "--learned-detector",
            "--detector-model",
            "nonexistent-model.onnx",
        ],
        "model file does not exist",
    );
}

#[cfg(feature = "onnx")]
#[test]
fn unloadable_detector_model_is_an_error_not_a_baseline() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = run(&[
        "--json",
        "--learned-detector",
        "--detector-model",
        path.to_str().unwrap(),
        "--duration-seconds",
        "1.5",
    ]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn identical_detector_models_get_identical_point_cloud_metrics() {
    let path = model("test_detector.onnx");
    let output = run(&[
        "--json",
        "--duration-seconds",
        "1.5",
        "--learned-detector",
        "--detector-model",
        &path,
        "--detector-baseline-model",
        &path,
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<_> = stdout.lines().collect();
    assert!(!String::from_utf8_lossy(&output.stderr).contains("failed"));
    assert_eq!(lines.len(), 6);
    for pair in lines.chunks_exact(2) {
        assert!(pair[0].contains("detector-candidate/analytic-imm"));
        assert_eq!(
            pair[0].replace("detector-candidate", "detector-baseline"),
            pair[1]
        );
    }
}

#[cfg(all(feature = "onnx", feature = "learned-imm"))]
#[test]
fn combined_detector_and_learned_imm_runs() {
    let output = run(&[
        "--json",
        "--duration-seconds",
        "1.5",
        "--learned-detector",
        "--detector-model",
        &model("eval_single_detection.onnx"),
        "--learned-imm",
        "--imm-model",
        &model("imm_mode_classifier.onnx"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 3);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("failed"));
    assert!(
        stdout
            .lines()
            .all(|line| line.contains("detector-candidate/learned-imm"))
    );
}
