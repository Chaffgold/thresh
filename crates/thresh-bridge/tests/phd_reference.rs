//! Bounded, invented linear-Gaussian PHD fixtures, not sequence accuracy or native parity.
//!
//! All update expectations use ordinary Rust arithmetic. The Python helper only
//! constructs inputs and inspects results; it never runs a second reference graph.

#![cfg(feature = "stonesoup")]

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use thresh_bridge::phd::{PhdConfig, PhdFilter};

const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-10;
static MAX_RESIDUAL: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct Component {
    weight: f64,
    mean: Vec<f64>,
    covariance: Vec<f64>,
}

struct Fixture<'py> {
    module: Bound<'py, PyModule>,
    scan: Bound<'py, PyDict>,
}

impl<'py> Fixture<'py> {
    fn new(py: Python<'py>, kind: &str, prior: &[f64], births: &[f64], detections: &[f64]) -> Self {
        let source = CString::new(include_str!("support/phd_fixtures.py")).unwrap();
        let module = PyModule::from_code(py, &source, c"phd_fixtures.py", c"phd_fixtures")
            .expect("the pinned real Stone Soup environment is required");
        let scan = module
            .getattr("make_scan")
            .unwrap()
            .call1((kind, prior.to_vec(), births.to_vec(), detections.to_vec()))
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();
        Self { module, scan }
    }

    fn item(&self, name: &str) -> Bound<'py, PyAny> {
        self.scan.get_item(name).unwrap().unwrap()
    }

    fn filter(&self, config: PhdConfig) -> Bound<'py, PyAny> {
        let py = self.scan.py();
        let filter = PhdFilter::new(py, config, &self.item("predictor"), &self.item("updater"))
            .expect("construct the real composed PHD reference");
        filter
            .filter(
                &self.item("prior").unbind(),
                &self.item("births").unbind(),
                &self.item("detections").unbind(),
                &self.item("timestamp").unbind(),
            )
            .expect("execute one real PHD scan")
            .into_bound(py)
    }

    fn components(&self, result: &Bound<'py, PyAny>) -> Vec<Component> {
        type Row = (f64, Vec<f64>, Vec<f64>, String, bool);
        let rows: Vec<Row> = self
            .module
            .getattr("numeric_components")
            .unwrap()
            .call1((result, self.item("timestamp")))
            .unwrap()
            .extract()
            .unwrap();
        rows.into_iter()
            .map(|(weight, mean, covariance, tag, at_scan_time)| {
                assert!(at_scan_time, "every output component uses the scan time");
                assert!(!tag.is_empty());
                assert_ne!(tag, "birth", "undetected births have no output component");
                assert!(weight.is_finite() && weight >= 0.0);
                Component {
                    weight,
                    mean,
                    covariance,
                }
            })
            .collect()
    }

    fn snapshot(&self) -> Bound<'py, PyAny> {
        self.module
            .getattr("snapshot")
            .unwrap()
            .call1((&self.scan,))
            .unwrap()
    }
}

fn with_python(test: impl for<'py> FnOnce(Python<'py>)) {
    Python::initialize();
    Python::attach(test);
    eprintln!(
        "GM-PHD maximum absolute numeric residual: {:.17e}",
        f64::from_bits(MAX_RESIDUAL.load(AtomicOrdering::Relaxed))
    );
}

fn scalar_likelihood(observation: f64) -> f64 {
    (-observation * observation / 4.0).exp() / (4.0 * PI).sqrt()
}

fn neutral_config() -> PhdConfig {
    PhdConfig {
        survival_probability: 0.8,
        detection_probability: 0.5,
        clutter_intensity: 0.2 * scalar_likelihood(0.0),
        prune_threshold: 0.0,
        merge_threshold: 0.0,
        max_components: 100,
        gating_threshold_squared: None,
    }
}

fn scalar(weight: f64, mean: f64, variance: f64) -> Component {
    Component {
        weight,
        mean: vec![mean],
        covariance: vec![variance],
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite result: {actual}");
    let residual = (actual - expected).abs();
    MAX_RESIDUAL.fetch_max(residual.to_bits(), AtomicOrdering::Relaxed);
    assert!(
        residual <= ATOL + RTOL * expected.abs(),
        "actual={actual:.17e}, expected={expected:.17e}, absolute residual={residual:.17e}"
    );
}

fn numeric_order(left: &Component, right: &Component) -> Ordering {
    left.covariance
        .iter()
        .chain(&left.mean)
        .chain(std::iter::once(&left.weight))
        .zip(
            right
                .covariance
                .iter()
                .chain(&right.mean)
                .chain(std::iter::once(&right.weight)),
        )
        .map(|(left, right)| left.total_cmp(right))
        .find(|order| *order != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn assert_component(actual: &Component, expected: &Component) {
    assert_close(actual.weight, expected.weight);
    assert_eq!(actual.mean.len(), expected.mean.len());
    assert_eq!(actual.covariance.len(), expected.covariance.len());
    for (value, reference) in actual.mean.iter().zip(&expected.mean) {
        assert_close(*value, *reference);
    }
    for (value, reference) in actual.covariance.iter().zip(&expected.covariance) {
        assert_close(*value, *reference);
    }
}

fn assert_components(mut actual: Vec<Component>, mut expected: Vec<Component>, intensity: f64) {
    // Generated tags and Python set order are deliberately absent from matching.
    actual.sort_by(numeric_order);
    expected.sort_by(numeric_order);
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (component, reference) in actual.iter().zip(&expected) {
        assert_component(component, reference);
    }
    assert_close(
        actual.iter().map(|component| component.weight).sum(),
        intensity,
    );
}

fn assert_empty(fixture: &Fixture<'_>, result: &Bound<'_, PyAny>) {
    assert_components(fixture.components(result), vec![], 0.0);
    assert!(result.getattr("timestamp").unwrap().is_none());
    assert!(!result.getattr("hypothesis").unwrap().is_none());
}

#[test]
fn scalar_update_matches_independent_weights_moments_and_intensity() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[0.5], &[], &[0.0]);
        let result = fixture.filter(neutral_config());
        assert_components(
            fixture.components(&result),
            vec![scalar(0.5, 0.0, 0.5), scalar(0.2, 0.0, 1.0)],
            0.7,
        );
        assert_eq!(result.getattr("hypothesis").unwrap().len().unwrap(), 2);
        assert!(
            result
                .getattr("timestamp")
                .unwrap()
                .eq(fixture.item("timestamp"))
                .unwrap()
        );
    });
}

#[test]
fn nonzero_interval_cv_prediction_and_update_match_independent_moments() {
    with_python(|py| {
        let fixture = Fixture::new(py, "cv", &[0.5], &[], &[2.0]);
        let likelihood = (-1.0_f64 / 6.0).exp() / (6.0 * PI).sqrt();
        let config = PhdConfig {
            clutter_intensity: 0.2 * likelihood,
            ..neutral_config()
        };
        // F=[[1,1],[0,1]], Q=0: Fm=[1,1], FPF'=[[2,1],[1,1]].
        // S=3, K=[2/3,1/3]; z-HFm=1 gives these independent moments.
        let detected = Component {
            weight: 0.5,
            mean: vec![5.0 / 3.0, 4.0 / 3.0],
            covariance: vec![2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
        };
        let missed = Component {
            weight: 0.2,
            mean: vec![1.0, 1.0],
            covariance: vec![2.0, 1.0, 1.0, 1.0],
        };
        let result = fixture.filter(config);
        assert_components(fixture.components(&result), vec![detected, missed], 0.7);
    });
}

#[test]
fn detected_birth_receives_survival_and_no_missed_birth() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[], &[0.5], &[0.0]);
        let result = fixture.filter(neutral_config());
        // Pinned 1.9.1 retags the detected birth before updating, so Ps applies.
        // Pd*Ps*w*q / (kappa + Pd*Ps*w*q) = .2q / .4q = .5.
        assert_components(
            fixture.components(&result),
            vec![scalar(0.5, 0.0, 0.5)],
            0.5,
        );
    });
}

#[test]
fn existing_and_birth_intensities_share_detection_denominator() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[0.5], &[0.25], &[0.0]);
        let result = fixture.filter(neutral_config());
        // The denominator is .2q clutter + .2q existing + .1q birth.
        assert_components(
            fixture.components(&result),
            vec![
                scalar(0.4, 0.0, 0.5),
                scalar(0.2, 0.0, 0.5),
                scalar(0.2, 0.0, 1.0),
            ],
            0.8,
        );
    });
}

#[test]
fn independently_empty_scan_inputs_preserve_birth_and_missed_semantics() {
    with_python(|py| {
        for (prior, births) in [(vec![], vec![]), (vec![], vec![0.5])] {
            let fixture = Fixture::new(py, "scalar", &prior, &births, &[]);
            assert_empty(&fixture, &fixture.filter(neutral_config()));
        }
        for births in [vec![], vec![0.25]] {
            let fixture = Fixture::new(py, "scalar", &[0.5], &births, &[]);
            let result = fixture.filter(neutral_config());
            assert_components(
                fixture.components(&result),
                vec![scalar(0.2, 0.0, 1.0)],
                0.2,
            );
        }
        let fixture = Fixture::new(py, "scalar", &[], &[], &[0.0, 1.0]);
        assert_empty(&fixture, &fixture.filter(neutral_config()));
    });
}

fn nonzero_or_epsilon(weight: f64) -> f64 {
    if weight == 0.0 { f64::EPSILON } else { weight }
}

fn assert_endpoint(py: Python<'_>, survival: f64, detection: f64, is_birth: bool) {
    let (prior, births) = if is_birth {
        (vec![], vec![0.5])
    } else {
        (vec![0.5], vec![])
    };
    let fixture = Fixture::new(py, "scalar", &prior, &births, &[0.0]);
    let config = PhdConfig {
        survival_probability: survival,
        detection_probability: detection,
        ..neutral_config()
    };
    let raw = survival * detection * 0.5 * scalar_likelihood(0.0);
    let detected = nonzero_or_epsilon(raw / (config.clutter_intensity + raw));
    let mut expected = vec![scalar(detected, 0.0, 0.5)];
    if !is_birth {
        expected.push(scalar(
            nonzero_or_epsilon((1.0 - detection) * survival * 0.5),
            0.0,
            1.0,
        ));
    }
    let result = fixture.filter(config);
    let actual = fixture.components(&result);
    let expected_zeros = expected
        .iter()
        .filter(|value| value.weight == f64::EPSILON)
        .count();
    assert_eq!(
        actual
            .iter()
            .filter(|value| value.weight == f64::EPSILON)
            .count(),
        expected_zeros
    );
    let intensity = expected.iter().map(|value| value.weight).sum();
    assert_components(actual, expected, intensity);
}

#[test]
fn probability_endpoints_explicitly_preserve_updater_machine_epsilon() {
    with_python(|py| {
        // This epsilon is a pinned updater compatibility rule, not positive target
        // evidence. Neutral reduction retains it; pruning may remove/redistribute it.
        for (survival, detection) in [
            (0.0, 0.5),
            (1.0, 0.5),
            (0.8, 0.0),
            (0.8, 1.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ] {
            assert_endpoint(py, survival, detection, false);
            assert_endpoint(py, survival, detection, true);
        }
    });
}

#[test]
fn intensity_weights_may_exceed_one_without_normalization() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[4.0], &[], &[]);
        let result = fixture.filter(neutral_config());
        assert_components(
            fixture.components(&result),
            vec![scalar(1.6, 0.0, 1.0)],
            1.6,
        );
    });
}

fn assert_measurement_gate(py: Python<'_>, threshold: Option<f64>, detected: bool) {
    let fixture = Fixture::new(py, "scalar", &[0.5], &[], &[2.0]);
    let config = PhdConfig {
        gating_threshold_squared: threshold,
        ..neutral_config()
    };
    let mut expected = vec![scalar(0.2, 0.0, 1.0)];
    if detected {
        let raw = 0.5 * 0.8 * 0.5 * scalar_likelihood(2.0);
        expected.push(scalar(raw / (config.clutter_intensity + raw), 1.0, 0.5));
    }
    let intensity = expected.iter().map(|component| component.weight).sum();
    let result = fixture.filter(config);
    assert_components(fixture.components(&result), expected, intensity);
}

#[test]
fn measurement_gate_uses_squared_distance_and_strict_boundary() {
    with_python(|py| {
        // z=2, predicted measurement mean=0 and variance=2 gives d²=2 exactly.
        // With merge=0 throughout, only the measurement gate can remove detection.
        assert_measurement_gate(py, Some(1.9), false);
        assert_measurement_gate(py, Some(2.0), false);
        assert_measurement_gate(py, Some(2.1), true);
        assert_measurement_gate(py, None, true);
    });
}

#[test]
fn repeated_scans_match_numeric_contents_without_using_generated_tags() {
    with_python(|py| {
        let first = Fixture::new(py, "scalar", &[0.5], &[0.25], &[-1.0, 1.0]);
        let second = Fixture::new(py, "scalar", &[0.5], &[0.25], &[1.0, -1.0]);
        let denominator = 0.2 * scalar_likelihood(0.0) + 0.3 * scalar_likelihood(1.0);
        let existing_weight = 0.2 * scalar_likelihood(1.0) / denominator;
        let birth_weight = 0.1 * scalar_likelihood(1.0) / denominator;
        let expected = vec![
            scalar(existing_weight, -0.5, 0.5),
            scalar(birth_weight, -0.5, 0.5),
            scalar(existing_weight, 0.5, 0.5),
            scalar(birth_weight, 0.5, 0.5),
            scalar(0.2, 0.0, 1.0),
        ];
        let intensity = 2.0 * (existing_weight + birth_weight) + 0.2;
        assert_components(
            first.components(&first.filter(neutral_config())),
            expected.clone(),
            intensity,
        );
        assert_components(
            second.components(&second.filter(neutral_config())),
            expected,
            intensity,
        );
    });
}

#[test]
fn successful_scan_preserves_inputs_arrays_and_reachable_ancestry() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[0.5], &[0.25], &[0.0]);
        fixture
            .module
            .getattr("add_ancestry")
            .unwrap()
            .call1((&fixture.scan,))
            .unwrap();
        let before = fixture.snapshot();
        let result = fixture.filter(neutral_config());
        assert!(before.eq(fixture.snapshot()).unwrap());
        assert!(
            fixture
                .module
                .getattr("provenance_uses_detached_components")
                .unwrap()
                .call1((&result, &fixture.scan))
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        assert_components(
            fixture.components(&result),
            vec![
                scalar(0.4, 0.0, 0.5),
                scalar(0.2, 0.0, 0.5),
                scalar(0.2, 0.0, 1.0),
            ],
            0.8,
        );
    });
}

#[test]
fn distinct_components_with_shared_arrays_and_ancestry_preserve_total_intensity() {
    with_python(|py| {
        let fixture = Fixture::new(py, "scalar", &[0.5, 0.5], &[], &[]);
        fixture
            .module
            .getattr("share_prior_arrays_and_ancestry")
            .unwrap()
            .call1((&fixture.scan,))
            .unwrap();
        let before = fixture.snapshot();
        let config = PhdConfig {
            survival_probability: 1.0,
            detection_probability: 0.0,
            ..neutral_config()
        };
        let result = fixture.filter(config);
        // Each distinct prior contributes (1-Pd)*Ps*w = (1-0)*1*.5 = .5.
        // Shared arrays, ancestry and equal tags must not collapse multiplicity.
        assert_components(
            fixture.components(&result),
            vec![scalar(0.5, 0.0, 1.0), scalar(0.5, 0.0, 1.0)],
            1.0,
        );
        // The snapshot records component/container IDs, shared array IDs and the
        // complete shared prior graph as well as values, weights, tags and times.
        assert!(before.eq(fixture.snapshot()).unwrap());
        let components = result.getattr("components").unwrap();
        let first_tag = components.get_item(0).unwrap().getattr("tag").unwrap();
        let second_tag = components.get_item(1).unwrap().getattr("tag").unwrap();
        // Neutral numeric merging still permits the pin's duplicate-tag repair.
        assert!(!first_tag.eq(second_tag).unwrap());
    });
}
