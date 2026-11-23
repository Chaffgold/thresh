//! Bounded exact JPDA checks with invented data and an independent Rust oracle.

#![cfg(feature = "stonesoup")]

use std::ffi::CString;
use std::process::Command;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use thresh_bridge::error::BridgeError;
use thresh_bridge::jpda::{JpdaAssociator, JpdaConfig};

// Independent tabulated 1-dof chi-squared 0.99 quantile. Ordinary fixture gates
// use the exact <= predicate; no tolerance expands gate membership.
const GATE_99: f64 = 6.634_896_601_021_211;
const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-10;

struct Oracle {
    marginals: Vec<Vec<f64>>,
    events: usize,
}

fn scalar_gaussian(residual: f64, variance: f64) -> f64 {
    (-residual * residual / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

fn raw_weights(
    states: &[[f64; 2]],
    observations: &[f64],
    seconds: &[f64],
    config: &JpdaConfig,
) -> Vec<Vec<Option<f64>>> {
    states
        .iter()
        .map(|state| track_weights(*state, observations, seconds, config))
        .collect()
}

fn track_weights(
    state: [f64; 2],
    observations: &[f64],
    seconds: &[f64],
    config: &JpdaConfig,
) -> Vec<Option<f64>> {
    assert_eq!(config.gate_probability, 0.99);
    let mut weights = vec![Some(
        1.0 - config.detection_probability * config.gate_probability,
    )];
    for (&observation, &elapsed) in observations.iter().zip(seconds) {
        let mean = state[0] + state[1] * elapsed;
        let variance = 2.0 + elapsed * elapsed + 0.1 * elapsed.powi(3) / 3.0;
        let residual = observation - mean;
        let weight = (residual * residual / variance <= GATE_99).then(|| {
            config.detection_probability * scalar_gaussian(residual, variance)
                / config.clutter_spatial_density
        });
        weights.push(weight);
    }
    weights
}

fn enumerate(weights: &[Vec<Option<f64>>], detections: usize) -> Oracle {
    let mut events = Vec::new();
    enumerate_events(
        weights,
        &mut vec![false; detections + 1],
        &mut Vec::new(),
        1.0,
        &mut events,
    );
    let total: f64 = events.iter().map(|(_, weight)| weight).sum();
    assert!(total.is_finite() && total > 0.0);
    let mut marginals = vec![vec![0.0; detections + 1]; weights.len()];
    for (assignment, weight) in &events {
        for (track, &column) in assignment.iter().enumerate() {
            marginals[track][column] += weight / total;
        }
    }
    Oracle {
        marginals,
        events: events.len(),
    }
}

fn enumerate_events(
    weights: &[Vec<Option<f64>>],
    occupied: &mut [bool],
    assignment: &mut Vec<usize>,
    product: f64,
    events: &mut Vec<(Vec<usize>, f64)>,
) {
    if assignment.len() == weights.len() {
        events.push((assignment.clone(), product));
        return;
    }
    for (column, weight) in weights[assignment.len()].iter().enumerate() {
        if let Some(weight) = weight
            && (column == 0 || !occupied[column])
        {
            occupied[column] = true;
            assignment.push(column);
            enumerate_events(weights, occupied, assignment, product * weight, events);
            assignment.pop();
            occupied[column] = false;
        }
    }
}

fn with_fixtures(test: impl for<'py> FnOnce(Python<'py>, Bound<'py, PyModule>)) {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            &CString::new(include_str!("support/jpda_fixture.py")).unwrap(),
            c"jpda_fixture.py",
            c"jpda_fixture",
        )
        .unwrap();
        test(py, module);
    });
}

fn fixture<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
    states: &[[f64; 2]],
    observations: &[f64],
    seconds: &[f64],
    call_seconds: f64,
) -> Bound<'py, PyAny> {
    let states: Vec<Vec<f64>> = states.iter().map(|state| state.to_vec()).collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("states", states).unwrap();
    kwargs.set_item("observations", observations).unwrap();
    kwargs.set_item("observation_seconds", seconds).unwrap();
    kwargs.set_item("call_seconds", call_seconds).unwrap();
    module
        .getattr("fixture")
        .unwrap()
        .call((), Some(&kwargs))
        .unwrap()
}

fn associator(data: &Bound<'_, PyAny>, config: &JpdaConfig) -> JpdaAssociator {
    JpdaAssociator::new(
        config,
        &data.get_item("predictor").unwrap().unbind(),
        &data.get_item("updater").unwrap().unbind(),
    )
    .unwrap()
}

fn associate(data: &Bound<'_, PyAny>, config: &JpdaConfig) -> Py<PyAny> {
    associator(data, config)
        .associate(
            &data.get_item("tracks").unwrap().unbind(),
            &data.get_item("detections").unwrap().unbind(),
            &data.get_item("timestamp").unwrap().unbind(),
        )
        .unwrap()
}

fn marginals(
    module: &Bound<'_, PyModule>,
    result: &Py<PyAny>,
    data: &Bound<'_, PyAny>,
) -> Vec<Vec<f64>> {
    module
        .getattr("marginals")
        .unwrap()
        .call1((result, data))
        .unwrap()
        .extract()
        .unwrap()
}

fn compare_marginals(actual: &[Vec<f64>], expected: &[Vec<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut max_residual = 0.0_f64;
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
            assert!(actual_value.is_finite() && actual_value >= 0.0);
            let residual = (actual_value - expected_value).abs();
            assert!(
                residual <= ATOL + RTOL * expected_value.abs(),
                "marginal {actual_value} != independent expectation {expected_value}"
            );
            max_residual = max_residual.max(residual);
        }
        assert!((actual_row.iter().sum::<f64>() - 1.0).abs() <= ATOL);
    }
    assert_occupancy(actual);
    max_residual
}

fn assert_occupancy(rows: &[Vec<f64>]) {
    if let Some(first) = rows.first() {
        for column in 1..first.len() {
            let occupancy: f64 = rows.iter().map(|row| row[column]).sum();
            assert!(occupancy <= 1.0 + ATOL, "occupancy {occupancy}");
        }
    }
}

fn check_fixture(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    states: &[[f64; 2]],
    observations: &[f64],
    config: &JpdaConfig,
) -> f64 {
    let seconds = vec![1.0; observations.len()];
    let data = fixture(py, module, states, observations, &seconds, 1.0);
    let expected = enumerate(
        &raw_weights(states, observations, &seconds, config),
        observations.len(),
    );
    let result = associate(&data, config);
    compare_marginals(&marginals(module, &result, &data), &expected.marginals)
}

#[test]
fn independent_oracle_counts_events_and_matches_hand_calculation() {
    let two = enumerate(&vec![vec![Some(1.0); 3]; 2], 2);
    assert_eq!(two.events, 7);
    compare_marginals(
        &two.marginals,
        &vec![vec![3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0]; 2],
    );
    let three = enumerate(&vec![vec![Some(1.0); 4]; 3], 3);
    assert_eq!(three.events, 34);
    let config = JpdaConfig {
        detection_probability: 0.5,
        clutter_spatial_density: scalar_gaussian(0.0, 3.0 + 0.1 / 3.0),
        ..JpdaConfig::default()
    };
    // Detected raw weight = .5; missed = 1 - .5*.99 = .505.
    let one = enumerate(&raw_weights(&[[0.0, 0.0]], &[0.0], &[1.0], &config), 1);
    assert_eq!(one.events, 2);
    compare_marginals(&one.marginals, &[vec![101.0 / 201.0, 100.0 / 201.0]]);
}

#[test]
fn zero_to_three_tracks_and_detections_match_independent_events() {
    with_fixtures(|py, module| {
        let states = [[-1.0, 0.2], [0.3, -0.1], [1.4, 0.4]];
        let observations = [-0.4, 0.7, 1.9];
        let config = JpdaConfig {
            clutter_spatial_density: 0.2,
            ..JpdaConfig::default()
        };
        let mut maximum = 0.0_f64;
        for track_count in 0..=3 {
            for detection_count in 0..=3 {
                maximum = maximum.max(check_fixture(
                    py,
                    &module,
                    &states[..track_count],
                    &observations[..detection_count],
                    &config,
                ));
            }
        }
        eprintln!("JPDA 0-3 dimensions maximum marginal residual: {maximum:.17e}");
    });
}

#[test]
fn shared_tied_and_asymmetric_observations_match_independent_events() {
    with_fixtures(|py, module| {
        let config = JpdaConfig::default();
        let cases = [
            (vec![[0.0, 0.0]; 3], vec![0.0]),
            (vec![[0.0, 0.0]; 2], vec![-0.5, 0.5]),
            (
                vec![[-1.0, 0.5], [2.0, -0.2], [0.1, 0.0]],
                vec![-0.4, 0.9, 2.3],
            ),
        ];
        let mut maximum = 0.0_f64;
        for (states, observations) in cases {
            maximum = maximum.max(check_fixture(py, &module, &states, &observations, &config));
        }
        eprintln!("JPDA ambiguous fixtures maximum marginal residual: {maximum:.17e}");
    });
}

#[test]
fn empty_gated_and_detection_probability_endpoints_preserve_missed_entries() {
    with_fixtures(|py, module| {
        let states = [[0.0, 0.0], [1.0, 0.0]];
        for observations in [vec![], vec![100.0, -100.0]] {
            check_fixture(py, &module, &states, &observations, &JpdaConfig::default());
        }
        for detection_probability in [0.0, 1.0] {
            let config = JpdaConfig {
                detection_probability,
                clutter_spatial_density: 0.2,
                ..JpdaConfig::default()
            };
            check_fixture(py, &module, &states, &[0.1, 0.8], &config);
        }
    });
}

#[test]
fn probability_gate_uses_independent_one_dimensional_squared_threshold() {
    with_fixtures(|py, module| {
        let variance = 3.0 + 0.1 / 3.0;
        let observations = [
            (variance * (GATE_99 - 0.01)).sqrt(),
            (variance * (GATE_99 + 0.01)).sqrt(),
        ];
        let data = fixture(py, &module, &[[0.0, 0.0]], &observations, &[1.0, 1.0], 1.0);
        let (distance, threshold): (f64, f64) = module
            .getattr("gate_geometry")
            .unwrap()
            .call1((&data,))
            .unwrap()
            .extract()
            .unwrap();
        assert!((threshold - GATE_99).abs() <= ATOL);
        assert!((distance - (GATE_99 - 0.01)).abs() <= ATOL);
        let result = associate(&data, &JpdaConfig::default());
        let actual = marginals(&module, &result, &data);
        assert!(actual[0][1] > 0.0);
        assert_eq!(actual[0][2], 0.0);
        let oracle = enumerate(
            &raw_weights(
                &[[0.0, 0.0]],
                &observations,
                &[1.0, 1.0],
                &JpdaConfig::default(),
            ),
            2,
        );
        compare_marginals(&actual, &oracle.marginals);
    });
}

#[test]
fn controlled_gate_equality_is_inclusive_without_rounding_tolerance() {
    with_fixtures(|py, module| {
        // The pinned chi2 implementation differs from the independent tabulated
        // quantile by 4e-15. For this comparator-only fixture, S=4 yields an
        // exactly representable computed d² equal to the pinned threshold.
        // Assert bitwise equality before relying on this deliberate boundary.
        let pinned_threshold: f64 = 6.634_896_601_021_215;
        assert!((pinned_threshold - GATE_99).abs() <= ATOL);
        let boundary = 2.0 * pinned_threshold.sqrt();
        let kwargs = PyDict::new(py);
        kwargs.set_item("diffusion", 0.0).unwrap();
        kwargs.set_item("measurement_variance", 2.0).unwrap();
        let data = module
            .getattr("fixture")
            .unwrap()
            .call(
                (
                    vec![vec![0.0, 0.0]],
                    vec![boundary, boundary.next_up()],
                    vec![1.0, 1.0],
                ),
                Some(&kwargs),
            )
            .unwrap();
        let (distance, threshold): (f64, f64) = module
            .getattr("gate_geometry")
            .unwrap()
            .call1((&data,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(distance, threshold);
        assert_eq!(threshold, pinned_threshold);
        let result = associate(&data, &JpdaConfig::default());
        let actual = marginals(&module, &result, &data);
        assert!(actual[0][1] > 0.0);
        assert_eq!(actual[0][2], 0.0);
        assert!((actual[0].iter().sum::<f64>() - 1.0).abs() <= ATOL);
    });
}

#[test]
fn observation_and_association_times_retain_identity_and_distinct_predictions() {
    with_fixtures(|py, module| {
        let states = [[0.0, 1.0]];
        let observations = [1.0];
        let data = fixture(py, &module, &states, &observations, &[1.0], 2.0);
        let config = JpdaConfig::default();
        let result = associate(&data, &config);
        let context: (f64, f64, bool) = module
            .getattr("prediction_context")
            .unwrap()
            .call1((&result, &data))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(context, (1.0, 2.0, true));
        let expected = enumerate(&raw_weights(&states, &observations, &[1.0], &config), 1);
        compare_marginals(&marginals(&module, &result, &data), &expected.marginals);
    });
}

fn fresh_process(name: &str, assertion: impl FnOnce()) {
    if std::env::var("THRESH_JPDA_REFERENCE_CASE").as_deref() == Ok(name) {
        assertion();
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("THRESH_JPDA_REFERENCE_CASE", name)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child {name} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

fn invalid_configs() -> Vec<JpdaConfig> {
    let mut configs = Vec::new();
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.1,
        0.0,
        1.0,
        1.1,
    ] {
        configs.push(JpdaConfig {
            gate_probability: value,
            ..JpdaConfig::default()
        });
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        configs.push(JpdaConfig {
            detection_probability: value,
            ..JpdaConfig::default()
        });
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 0.0] {
        configs.push(JpdaConfig {
            clutter_spatial_density: value,
            ..JpdaConfig::default()
        });
    }
    configs
}

#[test]
fn invalid_configuration_is_rejected_before_any_package_import() {
    fresh_process(
        "invalid_configuration_is_rejected_before_any_package_import",
        || {
            Python::initialize();
            Python::attach(|py| {
                let globals = PyDict::new(py);
                py.run(c"import sys\nattempts = []\nclass RejectImports:\n    def find_spec(self, fullname, path=None, target=None):\n        attempts.append(fullname)\n        raise AssertionError('configuration attempted an import')\nsys.meta_path.insert(0, RejectImports())\n",
                   Some(&globals), None).unwrap();
                for config in invalid_configs() {
                    assert!(matches!(
                        JpdaAssociator::new(&config, &py.None(), &py.None()),
                        Err(BridgeError::AlgorithmError(_))
                    ));
                }
                let attempts: Vec<String> = globals
                    .get_item("attempts")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                assert!(attempts.is_empty());
            });
        },
    );
}

#[test]
fn missing_hypothesiser_api_retains_structured_lookup_error() {
    fresh_process(
        "missing_hypothesiser_api_retains_structured_lookup_error",
        || {
            with_fixtures(|py, module| {
                let data = fixture(py, &module, &[], &[], &[], 1.0);
                py.import("stonesoup.dataassociator.probability").unwrap();
                py.import("stonesoup.hypothesiser.probability")
                    .unwrap()
                    .delattr("PDAHypothesiser")
                    .unwrap();
                let error = JpdaAssociator::new(
                    &JpdaConfig::default(),
                    &data.get_item("predictor").unwrap().unbind(),
                    &data.get_item("updater").unwrap().unbind(),
                )
                .err()
                .unwrap();
                match error {
                    BridgeError::UnsupportedApi { api, source } => {
                        assert_eq!(api, "PDAHypothesiser");
                        assert_eq!(source.operation, "JPDA hypothesiser lookup");
                        assert_eq!(source.exception_type, "AttributeError");
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            });
        },
    );
}

#[test]
fn constructor_failure_retains_original_exception_and_operation() {
    fresh_process(
        "constructor_failure_retains_original_exception_and_operation",
        || {
            with_fixtures(|py, module| {
                let data = fixture(py, &module, &[], &[], &[], 1.0);
                py.import("stonesoup.dataassociator.probability").unwrap();
                py.run(
                    c"import stonesoup.hypothesiser.probability as probability\ndef fail_constructor(**kwargs):\n    raise ValueError('synthetic JPDA constructor failure')\nprobability.PDAHypothesiser = fail_constructor\n",
                    None,
                    None,
                )
                .unwrap();
                let error = JpdaAssociator::new(
                    &JpdaConfig::default(),
                    &data.get_item("predictor").unwrap().unbind(),
                    &data.get_item("updater").unwrap().unbind(),
                )
                .err()
                .unwrap();
                match error {
                    BridgeError::PythonError(source) => {
                        assert_eq!(source.operation, "JPDA construction");
                        assert_eq!(source.exception_type, "ValueError");
                        assert_eq!(source.message, "synthetic JPDA constructor failure");
                        assert!(
                            source
                                .traceback
                                .is_some_and(|traceback| traceback.contains("fail_constructor"))
                        );
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            });
        },
    );
}

#[test]
fn invalid_python_association_input_retains_original_exception() {
    with_fixtures(|py, module| {
        let data = fixture(py, &module, &[[0.0, 0.0]], &[0.0], &[1.0], 1.0);
        let wrapper = associator(&data, &JpdaConfig::default());
        let error = wrapper
            .associate(
                &py.None(),
                &data.get_item("detections").unwrap().unbind(),
                &data.get_item("timestamp").unwrap().unbind(),
            )
            .err()
            .unwrap();
        match error {
            BridgeError::PythonError(source) => {
                assert_eq!(source.operation, "JPDA association");
                assert_eq!(source.exception_type, "TypeError");
                assert!(
                    source
                        .traceback
                        .is_some_and(|traceback| traceback.contains("associate"))
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    });
}
