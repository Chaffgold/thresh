//! Wrapper around Stone Soup's JPDA (Joint Probabilistic Data Association) data associator.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

/// Configuration for the JPDA data associator.
#[derive(Debug, Clone)]
pub struct JpdaConfig {
    /// Finite gate probability strictly between zero and one (default 0.99).
    ///
    /// Stone Soup converts this probability to a squared Mahalanobis threshold
    /// using the measurement dimension's chi-squared distribution. It is not
    /// itself a squared-distance threshold.
    pub gate_probability: f64,
    /// Finite probability of detection in `[0, 1]` (default 0.85).
    pub detection_probability: f64,
    /// Positive finite clutter density in inverse measurement-volume units
    /// (default 1e-6). All likelihoods and this density must describe one
    /// compatible measurement space; the default is not a unit conversion.
    pub clutter_spatial_density: f64,
}

impl Default for JpdaConfig {
    fn default() -> Self {
        Self {
            gate_probability: 0.99,
            detection_probability: 0.85,
            clutter_spatial_density: 1e-6,
        }
    }
}

impl JpdaConfig {
    fn validate(&self) -> BridgeResult<()> {
        if !self.gate_probability.is_finite()
            || self.gate_probability <= 0.0
            || self.gate_probability >= 1.0
        {
            return Err(BridgeError::AlgorithmError(
                "JPDA gate_probability must be finite and strictly between zero and one".into(),
            ));
        }
        if !self.detection_probability.is_finite()
            || !(0.0..=1.0).contains(&self.detection_probability)
        {
            return Err(BridgeError::AlgorithmError(
                "JPDA detection_probability must be finite and in [0, 1]".into(),
            ));
        }
        if !self.clutter_spatial_density.is_finite() || self.clutter_spatial_density <= 0.0 {
            return Err(BridgeError::AlgorithmError(
                "JPDA clutter_spatial_density must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// JPDA data associator backed by Stone Soup.
pub struct JpdaAssociator {
    /// The Python JPDA object, held across calls.
    py_associator: Py<PyAny>,
}

impl JpdaAssociator {
    /// Create a new JPDA associator.
    ///
    /// Builds a normalized, gated Stone Soup `PDAHypothesiser` and exact `JPDA`
    /// associator using the supplied compatible predictor and updater objects.
    /// The host must already have initialized Python. Invalid numeric
    /// configuration fails before interpreter attachment or package imports.
    pub fn new(
        config: &JpdaConfig,
        predictor: &Py<PyAny>,
        updater: &Py<PyAny>,
    ) -> BridgeResult<Self> {
        config.validate()?;
        with_python("JPDA construction", |py| {
            let error = |error| BridgeError::from_python(py, error, "JPDA construction");
            let ss_hyp = import_module(py, "stonesoup.hypothesiser.probability")?;
            let ss_assoc = import_module(py, "stonesoup.dataassociator.probability")?;

            let hyp_kwargs = PyDict::new(py);
            hyp_kwargs
                .set_item("predictor", predictor.bind(py))
                .map_err(error)?;
            hyp_kwargs
                .set_item("updater", updater.bind(py))
                .map_err(error)?;
            hyp_kwargs
                .set_item("prob_gate", config.gate_probability)
                .map_err(error)?;
            hyp_kwargs
                .set_item("prob_detect", config.detection_probability)
                .map_err(error)?;
            hyp_kwargs
                .set_item("clutter_spatial_density", config.clutter_spatial_density)
                .map_err(error)?;
            hyp_kwargs.set_item("include_all", false).map_err(error)?;
            hyp_kwargs.set_item("normalise", true).map_err(error)?;
            let hyp_cls = required_attribute(
                ss_hyp.as_any(),
                "PDAHypothesiser",
                "JPDA hypothesiser lookup",
            )?;
            let hypothesiser = hyp_cls.call((), Some(&hyp_kwargs)).map_err(error)?;

            let assoc_kwargs = PyDict::new(py);
            assoc_kwargs
                .set_item("hypothesiser", hypothesiser)
                .map_err(error)?;
            let assoc_cls =
                required_attribute(ss_assoc.as_any(), "JPDA", "JPDA associator lookup")?;
            let associator = assoc_cls.call((), Some(&assoc_kwargs)).map_err(error)?;

            Ok(Self {
                py_associator: associator.unbind(),
            })
        })
    }

    /// Run the JPDA association on a set of tracks and detections.
    ///
    /// `tracks` and `detections` are Python objects (Stone Soup types).
    /// Returns the association mapping as a Python dict, retaining track and
    /// detection identity. Detected predictions use each detection's observation
    /// timestamp; missed predictions use `timestamp`. Times pass through without
    /// a fallback for `None` or an out-of-sequence policy, and upstream may predict
    /// backward. Supported reference fixtures use timezone-aware UTC timestamps.
    pub fn associate(
        &self,
        tracks: &Py<PyAny>,
        detections: &Py<PyAny>,
        timestamp: &Py<PyAny>,
    ) -> BridgeResult<Py<PyAny>> {
        with_python("JPDA association", |py| {
            let method = required_attribute(
                self.py_associator.bind(py),
                "associate",
                "JPDA association lookup",
            )?;
            method
                .call1((tracks.bind(py), detections.bind(py), timestamp.bind(py)))
                .map(Bound::unbind)
                .map_err(|error| BridgeError::from_python(py, error, "JPDA association"))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use pyo3::types::PyModule;

    use super::*;
    use crate::runtime::step_test_support::assert_step_contract;

    #[test]
    fn real_constructor_preserves_components_and_configuration() {
        Python::initialize();
        Python::attach(|py| {
            let fixtures = PyModule::from_code(
                py,
                &CString::new(include_str!("../tests/support/jpda_fixture.py")).unwrap(),
                c"jpda_fixture.py",
                c"jpda_fixture",
            )
            .unwrap();
            let (predictor, updater): (Py<PyAny>, Py<PyAny>) = fixtures
                .getattr("components")
                .unwrap()
                .call0()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(JpdaConfig::default().gate_probability, 0.99);
            assert_eq!(JpdaConfig::default().detection_probability, 0.85);
            assert_eq!(JpdaConfig::default().clutter_spatial_density, 1e-6);
            for detection_probability in [0.0, 0.85, 1.0] {
                let config = JpdaConfig {
                    gate_probability: 0.93,
                    detection_probability,
                    clutter_spatial_density: 0.2,
                };
                let wrapper = JpdaAssociator::new(&config, &predictor, &updater).unwrap();
                assert_graph(py, &wrapper, &config, &predictor, &updater);
            }
        });
    }

    fn assert_graph(
        py: Python<'_>,
        wrapper: &JpdaAssociator,
        config: &JpdaConfig,
        predictor: &Py<PyAny>,
        updater: &Py<PyAny>,
    ) {
        let associator = wrapper.py_associator.bind(py);
        assert_eq!(associator.get_type().name().unwrap(), "JPDA");
        let hypothesiser = associator.getattr("hypothesiser").unwrap();
        assert_eq!(hypothesiser.get_type().name().unwrap(), "PDAHypothesiser");
        assert!(hypothesiser.getattr("predictor").unwrap().is(predictor));
        assert!(hypothesiser.getattr("updater").unwrap().is(updater));
        for (name, expected) in [
            ("prob_gate", config.gate_probability),
            ("prob_detect", config.detection_probability),
            ("clutter_spatial_density", config.clutter_spatial_density),
        ] {
            let actual = hypothesiser
                .getattr(name)
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert_eq!(actual, expected);
        }
        assert!(
            !hypothesiser
                .getattr("include_all")
                .unwrap()
                .is_truthy()
                .unwrap()
        );
        assert!(
            hypothesiser
                .getattr("normalise")
                .unwrap()
                .is_truthy()
                .unwrap()
        );
    }

    #[test]
    fn initialized_associate_preserves_dispatch_and_errors() {
        assert_step_contract(
            "associate",
            "JPDA association",
            "JPDA association lookup",
            3,
            |py_associator| JpdaAssociator { py_associator },
            |wrapper, arguments| wrapper.associate(&arguments[0], &arguments[1], &arguments[2]),
        );
    }
}
