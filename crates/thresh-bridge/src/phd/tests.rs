use super::*;
use std::ffi::CString;

use pyo3::types::PyModule;

fn fixture_module(py: Python<'_>) -> Bound<'_, PyModule> {
    let source = CString::new(include_str!("../../tests/support/phd_contract_fixture.py")).unwrap();
    PyModule::from_code(
        py,
        &source,
        c"phd_contract_fixture.py",
        c"phd_contract_fixture",
    )
    .unwrap()
}

struct Fixture {
    module: Py<PyModule>,
    object: Py<PyAny>,
    filter: PhdFilter,
}

impl Fixture {
    fn new(source: Option<&str>) -> Self {
        Python::initialize();
        Python::attach(|py| {
            let module = fixture_module(py);
            let object = match source {
                Some(source) => module.call_method1("malformed_fixture", (source,)).unwrap(),
                None => module.call_method0("fixture").unwrap(),
            };
            let filter = PhdFilter {
                py_hypothesiser: object.getattr("hypothesiser").unwrap().unbind(),
                py_updater: object.getattr("phd_updater").unwrap().unbind(),
                py_reducer: object.getattr("reducer").unwrap().unbind(),
                config: PhdConfig::default(),
            };
            Self {
                module: module.unbind(),
                object: object.unbind(),
                filter,
            }
        })
    }

    fn invoke(&self) -> BridgeResult<Py<PyAny>> {
        let args = Python::attach(|py| {
            ["prior", "births", "detections", "time"]
                .map(|name| self.object.bind(py).getattr(name).unwrap().unbind())
        });
        // Exercise host attachment without a live token held by the test.
        self.filter.filter(&args[0], &args[1], &args[2], &args[3])
    }

    fn preserved(&self) {
        Python::attach(|py| {
            self.module
                .bind(py)
                .call_method1("assert_preserved", (self.object.bind(py),))
                .unwrap();
        });
    }

    fn calls(&self) -> Vec<String> {
        Python::attach(|py| {
            self.object
                .bind(py)
                .getattr("calls")
                .unwrap()
                .extract()
                .unwrap()
        })
    }
}

fn isolated(name: &str, check: impl FnOnce()) {
    if std::env::var("THRESH_PHD_UNIT_CASE").as_deref() == Ok(name) {
        check();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("THRESH_PHD_UNIT_CASE", name)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

#[test]
fn construction_maps_config_and_preserves_component_identity() {
    Python::initialize();
    Python::attach(|py| {
        let module = fixture_module(py);
        let models = module.call_method0("models").unwrap();
        let predictor = models.get_item(0).unwrap();
        let updater = models.get_item(1).unwrap();
        let config = PhdConfig {
            survival_probability: 0.8,
            detection_probability: 0.5,
            clutter_intensity: 0.2,
            prune_threshold: 0.01,
            merge_threshold: 3.5,
            max_components: 12,
            gating_threshold_squared: Some(6.0),
        };
        let filter = PhdFilter::new(py, config, &predictor, &updater).unwrap();
        let mixture = filter.py_hypothesiser.bind(py);
        assert!(
            mixture
                .getattr("order_by_detection")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        let distance = mixture.getattr("hypothesiser").unwrap();
        assert!(distance.getattr("predictor").unwrap().is(&predictor));
        assert!(distance.getattr("updater").unwrap().is(&updater));
        assert_eq!(
            distance
                .getattr("measure")
                .unwrap()
                .get_type()
                .name()
                .unwrap(),
            "SquaredMahalanobis"
        );
        assert_float_attr(&distance, "missed_distance", 6.0);
        let phd = filter.py_updater.bind(py);
        assert!(phd.getattr("updater").unwrap().is(&updater));
        assert_float_attr(phd, "prob_survival", 0.8);
        assert_float_attr(phd, "prob_detection", 0.5);
        assert_float_attr(phd, "clutter_spatial_density", 0.2);
        assert!(
            phd.getattr("normalisation")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        assert_float_attr(filter.py_reducer.bind(py), "prune_threshold", 0.01);
        assert_float_attr(filter.py_reducer.bind(py), "merge_threshold", 3.5);
        assert_eq!(
            filter
                .py_reducer
                .bind(py)
                .getattr("max_number_components")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            12
        );
        assert_eq!(filter.config().gating_threshold_squared, Some(6.0));
        let default_filter =
            PhdFilter::new(py, PhdConfig::default(), &predictor, &updater).unwrap();
        let default_distance = default_filter
            .py_hypothesiser
            .bind(py)
            .getattr("hypothesiser")
            .unwrap();
        assert_float_attr(&default_distance, "missed_distance", f64::INFINITY);
    });
}

fn assert_float_attr(object: &Bound<'_, PyAny>, attribute: &str, expected: f64) {
    assert_eq!(
        object.getattr(attribute).unwrap().extract::<f64>().unwrap(),
        expected,
        "{attribute}"
    );
}

fn invalid_configs() -> Vec<PhdConfig> {
    let mut configurations = Vec::new();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        configurations.push(PhdConfig {
            survival_probability: value,
            ..PhdConfig::default()
        });
        configurations.push(PhdConfig {
            detection_probability: value,
            ..PhdConfig::default()
        });
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 0.0] {
        configurations.push(PhdConfig {
            clutter_intensity: value,
            ..PhdConfig::default()
        });
        configurations.push(PhdConfig {
            gating_threshold_squared: Some(value),
            ..PhdConfig::default()
        });
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        configurations.push(PhdConfig {
            prune_threshold: value,
            ..PhdConfig::default()
        });
        configurations.push(PhdConfig {
            merge_threshold: value,
            ..PhdConfig::default()
        });
    }
    configurations.push(PhdConfig {
        max_components: 0,
        ..PhdConfig::default()
    });
    configurations
}

#[test]
fn invalid_config_fails_before_imports_and_endpoints_are_valid() {
    isolated(
        "phd::tests::invalid_config_fails_before_imports_and_endpoints_are_valid",
        || {
            Python::initialize();
            Python::attach(|py| {
                py.run(c"import sys\nclass ImportTrap:\n    calls = 0\n    def find_spec(self, fullname, path=None, target=None):\n        self.calls += 1\n        raise AssertionError('configuration must fail before imports')\ntrap = ImportTrap()\nsys.meta_path.insert(0, trap)\n", None, None).unwrap();
                let input = py.None();
                for config in invalid_configs() {
                    let error = PhdFilter::new(py, config, input.bind(py), input.bind(py))
                        .err()
                        .unwrap();
                    assert!(matches!(error, BridgeError::AlgorithmError(_)), "{error:?}");
                }
                assert_eq!(
                    py.eval(c"trap.calls", None, None)
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    0
                );
            });
        },
    );
    for probability in [0.0, 1.0] {
        PhdConfig {
            survival_probability: probability,
            detection_probability: probability,
            prune_threshold: 0.0,
            merge_threshold: 0.0,
            ..PhdConfig::default()
        }
        .validate()
        .unwrap();
    }
}

#[test]
fn constructor_import_and_api_failures_retain_structured_causes() {
    isolated(
        "phd::tests::constructor_import_and_api_failures_retain_structured_causes",
        || {
            Python::initialize();
            Python::attach(check_constructor_failures);
        },
    );
}

fn check_constructor_failures(py: Python<'_>) {
    py.run(c"import sys\nclass ImportTrap:\n    missing = 'stonesoup'\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname == 'stonesoup':\n            raise ModuleNotFoundError('synthetic constructor import', name=self.missing)\ntrap = ImportTrap()\nsys.meta_path.insert(0, trap)\n", None, None).unwrap();
    let input = py.None();
    for missing in ["stonesoup", "numpy", "stonesoup.measures"] {
        py.import("__main__")
            .unwrap()
            .getattr("trap")
            .unwrap()
            .setattr("missing", missing)
            .unwrap();
        let error = PhdFilter::new(py, PhdConfig::default(), input.bind(py), input.bind(py))
            .err()
            .unwrap();
        assert_import_category(&error, missing);
    }
    py.run(c"sys.meta_path.remove(trap)", None, None).unwrap();
    let measures = py.import("stonesoup.measures").unwrap();
    measures.delattr("SquaredMahalanobis").unwrap();
    let error = PhdFilter::new(py, PhdConfig::default(), input.bind(py), input.bind(py))
        .err()
        .unwrap();
    let BridgeError::UnsupportedApi { api, source } = error else {
        panic!("{error:?}")
    };
    assert_eq!(api, "SquaredMahalanobis");
    assert_eq!(source.operation, "PHD hypothesiser construction lookup");
}

fn assert_import_category(error: &BridgeError, missing: &str) {
    let source = match error {
        BridgeError::StoneSoupNotInstalled(source) => {
            assert_eq!(missing, "stonesoup");
            source
        }
        BridgeError::MissingDependency { name, source } => {
            assert_eq!(name, missing);
            source
        }
        BridgeError::UnsupportedApi { api, source } => {
            assert_eq!(api, missing);
            source
        }
        _ => panic!("{error:?}"),
    };
    assert_eq!(source.exception_type, "ModuleNotFoundError");
    assert_eq!(source.operation, "import stonesoup.measures");
    assert!(source.traceback.as_ref().unwrap().contains("find_spec"));
}

#[test]
fn malformed_scan_inputs_fail_before_hypothesis_generation() {
    Python::initialize();
    let cases: Vec<String> = Python::attach(|py| {
        fixture_module(py)
            .call_method0("malformed_cases")
            .unwrap()
            .extract()
            .unwrap()
    });
    for case in cases {
        let fixture = Fixture::new(Some(&case));
        let error = fixture.invoke().unwrap_err();
        assert!(
            matches!(error, BridgeError::AlgorithmError(_)),
            "{case}: {error:?}"
        );
        assert!(fixture.calls().is_empty(), "{case}");
    }
}

#[test]
fn repeated_component_objects_fail_before_copy_and_preserve_inputs() {
    for collection in ["prior", "births"] {
        let source = format!(
            "f.{collection}.append(f.{collection}[0]); f.original_collection = f.{collection}"
        );
        let fixture = Fixture::new(Some(&source));
        let error = fixture.invoke().unwrap_err();
        let BridgeError::AlgorithmError(message) = error else {
            panic!("expected identity validation error, got {error:?}");
        };
        assert!(message.contains("distinct Python objects"), "{message}");
        assert!(fixture.calls().is_empty());
        Python::attach(|py| {
            let object = fixture.object.bind(py);
            let components = object.getattr(collection).unwrap();
            assert!(components.is(object.getattr("original_collection").unwrap()));
            assert!(
                components
                    .get_item(0)
                    .unwrap()
                    .is(components.get_item(1).unwrap())
            );
            assert_eq!(
                object
                    .getattr("copy_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
            let prior_count = if collection == "prior" { 2 } else { 1 };
            let birth_count = if collection == "births" { 2 } else { 1 };
            fixture
                .module
                .bind(py)
                .call_method1("assert_preserved", (object, prior_count, birth_count))
                .unwrap();
        });
    }
}

#[test]
fn distinct_same_tag_components_may_share_arrays_and_ancestry() {
    for births in [false, true] {
        let source = format!(
            "with_distinct_shared_components(f, {})",
            if births { "True" } else { "False" }
        );
        let fixture = Fixture::new(Some(&source));
        let result = fixture.invoke().unwrap();
        assert_eq!(fixture.calls(), ["hypothesise", "update", "reduce"]);
        Python::attach(|py| {
            let object = fixture.object.bind(py);
            let collection = if births { "births" } else { "prior" };
            let original = object.getattr(collection).unwrap();
            let original_first = original.get_item(0).unwrap();
            let original_second = original.get_item(1).unwrap();
            assert!(!original_first.is(&original_second));
            assert!(
                original_first
                    .getattr("tag")
                    .unwrap()
                    .eq(original_second.getattr("tag").unwrap())
                    .unwrap()
            );
            let components = result.bind(py).getattr("components").unwrap();
            assert_eq!(components.len().unwrap(), 2);
            let first = components.get_item(0).unwrap();
            let second = components.get_item(1).unwrap();
            assert!(!first.is(&second));
            assert_shared_component_graph(&first, &second);
            assert_shared_component_graph(&original_first, &original_second);
            assert_float_attr(&first, "weight", 0.5);
            assert_float_attr(&second, "weight", 0.5);
        });
    }
}

fn assert_shared_component_graph(first: &Bound<'_, PyAny>, second: &Bound<'_, PyAny>) {
    for name in ["state_vector", "covar", "ancestry"] {
        assert!(
            first
                .getattr(name)
                .unwrap()
                .is(second.getattr(name).unwrap()),
            "shared {name}"
        );
    }
}

#[test]
fn one_shot_inputs_are_rejected_without_consumption() {
    for collection in ["prior", "births", "detections"] {
        let source = format!(
            "f.original = list(f.{collection}); f.{collection} = (item for item in f.original)"
        );
        let fixture = Fixture::new(Some(&source));
        assert!(matches!(
            fixture.invoke(),
            Err(BridgeError::AlgorithmError(_))
        ));
        Python::attach(|py| {
            let object = fixture.object.bind(py);
            let remaining = py
                .import("builtins")
                .unwrap()
                .call_method1("list", (object.getattr(collection).unwrap(),))
                .unwrap();
            assert!(remaining.eq(object.getattr("original").unwrap()).unwrap());
        });
        assert!(fixture.calls().is_empty());
    }
    let fixture = Fixture::new(Some(
        "f.original = f.prior[:]; f.prior = (item for item in f.prior); f.births = None",
    ));
    assert!(matches!(
        fixture.invoke(),
        Err(BridgeError::AlgorithmError(_))
    ));
    Python::attach(|py| {
        let object = fixture.object.bind(py);
        assert!(
            py.import("builtins")
                .unwrap()
                .call_method1("list", (object.getattr("prior").unwrap(),))
                .unwrap()
                .eq(object.getattr("original").unwrap())
                .unwrap()
        );
    });
    assert_earlier_collections_are_not_advanced();
}

fn assert_earlier_collections_are_not_advanced() {
    for source in [
        "f.prior = CountedCollection(f.prior); f.births = CountedCollection(f.births); f.detections = (item for item in f.detections)",
        "f.prior = CountedCollection(f.prior); f.births = CountedCollection(f.births); f.detections = None",
    ] {
        let fixture = Fixture::new(Some(source));
        assert!(matches!(
            fixture.invoke(),
            Err(BridgeError::AlgorithmError(_))
        ));
        Python::attach(|py| {
            let object = fixture.object.bind(py);
            for name in ["prior", "births"] {
                assert_eq!(
                    object
                        .getattr(name)
                        .unwrap()
                        .getattr("advances")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    0,
                    "preflight consumed earlier {name} collection"
                );
            }
        });
        assert!(fixture.calls().is_empty());
    }
}

#[test]
fn graph_dispatch_deep_copies_once_and_retains_provenance() {
    let fixture = Fixture::new(Some("f.mutate = True"));
    let result = fixture.invoke().unwrap();
    assert_eq!(fixture.calls(), ["hypothesise", "update", "reduce"]);
    fixture.preserved();
    Python::attach(|py| {
        let object = fixture.object.bind(py);
        assert!(result.bind(py).is(object.getattr("result").unwrap()));
        assert!(
            result
                .bind(py)
                .getattr("hypothesis")
                .unwrap()
                .is(object.getattr("hypotheses").unwrap())
        );
        assert!(
            result
                .bind(py)
                .getattr("components")
                .unwrap()
                .is(object.getattr("reduced").unwrap())
        );
    });
}

#[test]
fn mutating_graph_failures_preserve_caller_inputs_and_python_causes() {
    for stage in ["hypothesis", "update", "reduction"] {
        for kind in ["ValueError", "TypeError"] {
            let source =
                format!("f.mutate = True; f.failure_stage = '{stage}'; f.failure_kind = {kind}");
            let fixture = Fixture::new(Some(&source));
            let error = fixture.invoke().unwrap_err();
            assert_python_cause(
                error,
                &format!("PHD {stage}"),
                kind,
                "synthetic stonesoup numpy stage failure",
            );
            fixture.preserved();
            let expected = match stage {
                "hypothesis" => 1,
                "update" => 2,
                _ => 3,
            };
            assert_eq!(fixture.calls().len(), expected);
        }
    }
}

fn assert_python_cause(error: BridgeError, operation: &str, kind: &str, message: &str) {
    let BridgeError::PythonError(source) = error else {
        panic!("expected PythonError: {error:?}")
    };
    assert_eq!(source.operation, operation);
    assert_eq!(source.exception_type, kind);
    assert_eq!(source.message, message);
    assert!(source.traceback.is_some());
}

#[test]
fn graph_missing_methods_are_operation_specific_api_errors() {
    for (object, method, operation) in [
        ("hypothesiser", "hypothesise", "PHD hypothesis lookup"),
        ("phd_updater", "update", "PHD update lookup"),
        ("reducer", "reduce", "PHD reduction lookup"),
    ] {
        let fixture = Fixture::new(Some(&format!("del f.{object}.{method}")));
        let error = fixture.invoke().unwrap_err();
        let BridgeError::UnsupportedApi { api, source } = error else {
            panic!("{error:?}")
        };
        assert_eq!(api, method);
        assert_eq!(source.operation, operation);
        assert_eq!(source.exception_type, "AttributeError");
        fixture.preserved();
    }
}

#[test]
fn component_copy_failure_is_structured_and_never_dispatches() {
    let fixture = Fixture::new(Some("with_copy_failure(f)"));
    assert_python_cause(
        fixture.invoke().unwrap_err(),
        "PHD component copy",
        "RuntimeError",
        "synthetic component copy failure",
    );
    assert!(fixture.calls().is_empty());
}

#[test]
fn valid_collection_types_and_intensity_weights_are_not_normalized() {
    for source in [
        "f.prior[0].weight = 2.5; f.births[0].weight = 0.0; f.prior = tuple(f.prior); f.births = tuple(f.births)",
        "f.prior[0].weight = 2.5; f.births[0].weight = 0.0; with_mixture_prior(f)",
    ] {
        let fixture = Fixture::new(Some(source));
        let result = fixture.invoke().unwrap();
        Python::attach(|py| {
            let components = result.bind(py).getattr("components").unwrap();
            assert_float_attr(&components.get_item(0).unwrap(), "weight", 0.0);
            assert_float_attr(&components.get_item(1).unwrap(), "weight", 2.5);
        });
    }
}

#[test]
fn unexpected_validation_and_result_errors_keep_their_context() {
    let fixture = Fixture::new(Some("f.prior[0].weight = UnexpectedWeight()"));
    assert_python_cause(
        fixture.invoke().unwrap_err(),
        "PHD input validation",
        "ValueError",
        "synthetic weight inspection failure",
    );
    assert!(fixture.calls().is_empty());

    let fixture = Fixture::new(Some("unwritable_result(f)"));
    assert_python_cause(
        fixture.invoke().unwrap_err(),
        "PHD result",
        "ValueError",
        "synthetic result assignment failure",
    );
    fixture.preserved();

    let fixture = Fixture::new(Some("f.phd_updater.update = lambda hypotheses: object()"));
    let error = fixture.invoke().unwrap_err();
    let BridgeError::UnsupportedApi { api, source } = error else {
        panic!("{error:?}");
    };
    assert_eq!(api, "components");
    assert_eq!(source.operation, "PHD components lookup");
    fixture.preserved();
}

fn reduce(config: PhdConfig, inputs: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    Python::initialize();
    Python::attach(|py| {
        let module = fixture_module(py);
        let components = module
            .call_method1("reducer_inputs", (inputs.to_vec(),))
            .unwrap();
        // Use exactly the constructor and dispatch helper used in production.
        let reducer = build_reducer(py, &config).unwrap();
        let reduced = call_step(reducer.bind(py), "reduce", &components, "PHD reduction").unwrap();
        module
            .call_method1("numeric_components", (reduced,))
            .unwrap()
            .extract()
            .unwrap()
    })
}

fn neutral_reducer() -> PhdConfig {
    PhdConfig {
        prune_threshold: 0.0,
        merge_threshold: 0.0,
        max_components: 100,
        ..PhdConfig::default()
    }
}

fn assert_components(actual: &[(f64, f64, f64)], expected: &[(f64, f64, f64)]) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum_residual = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        for (actual, expected) in [
            (actual.0, expected.0),
            (actual.1, expected.1),
            (actual.2, expected.2),
        ] {
            assert!(actual.is_finite());
            let residual = (actual - expected).abs();
            maximum_residual = maximum_residual.max(residual);
            assert!(
                residual <= 1e-12 + 1e-10 * expected.abs(),
                "actual={actual}, expected={expected}, residual={residual:e}"
            );
        }
    }
    eprintln!("GM-PHD reducer maximum absolute residual: {maximum_residual:.17e}");
}

#[test]
fn reducer_pruning_redistributes_and_pruning_everything_is_empty() {
    let config = PhdConfig {
        prune_threshold: 0.1,
        ..neutral_reducer()
    };
    assert_components(
        &reduce(
            config.clone(),
            &[(0.05, 0.0, 1.0), (0.2, 1.0, 1.0), (0.3, 2.0, 1.0)],
        ),
        &[(0.225, 1.0, 1.0), (0.325, 2.0, 1.0)],
    );
    assert!(reduce(config, &[(0.05, 0.0, 1.0), (0.09, 1.0, 1.0)]).is_empty());
}

#[test]
fn reducer_merging_matches_moments_and_clips_weight_at_one() {
    let config = PhdConfig {
        merge_threshold: 4.0,
        ..neutral_reducer()
    };
    assert_components(
        &reduce(config.clone(), &[(0.2, 0.0, 1.0), (0.3, 1.0, 1.0)]),
        &[(0.5, 0.6, 1.24)],
    );
    // The pin deliberately does not conserve intensity for this pair.
    assert_components(
        &reduce(config, &[(0.7, 0.0, 1.0), (0.6, 0.0, 1.0)]),
        &[(1.0, 0.0, 1.0)],
    );
}

#[test]
fn reducer_merge_uses_strict_squared_candidate_covariance_distance() {
    let config = PhdConfig {
        merge_threshold: 4.0,
        ..neutral_reducer()
    };
    assert_components(
        &reduce(config, &[(0.2, 0.0, 1.0), (0.3, 2.0, 1.0)]),
        &[(0.2, 0.0, 1.0), (0.3, 2.0, 1.0)],
    );
    let config = PhdConfig {
        merge_threshold: 2.0,
        ..neutral_reducer()
    };
    // Lower-weight candidate covariance 4 gives d²=1, so merge. Reversing
    // covariances gives d²=4 and excludes the pair despite the best's variance 4.
    assert_components(
        &reduce(config.clone(), &[(0.2, 0.0, 4.0), (0.3, 2.0, 1.0)]),
        &[(0.5, 1.2, 3.16)],
    );
    assert_components(
        &reduce(config, &[(0.2, 0.0, 1.0), (0.3, 2.0, 4.0)]),
        &[(0.2, 0.0, 1.0), (0.3, 2.0, 4.0)],
    );
}

#[test]
fn reducer_truncation_redistributes_removed_weight() {
    let config = PhdConfig {
        max_components: 2,
        ..neutral_reducer()
    };
    assert_components(
        &reduce(config, &[(0.1, 0.0, 1.0), (0.2, 1.0, 1.0), (0.3, 2.0, 1.0)]),
        &[(0.25, 1.0, 1.0), (0.35, 2.0, 1.0)],
    );
}
