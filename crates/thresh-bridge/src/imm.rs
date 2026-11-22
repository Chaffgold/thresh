//! Wrapper around Stone Soup's IMM (Interacting Multiple Model) filter.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::{BridgeError, BridgeResult};
use crate::runtime::{import_module, required_attribute, with_python};

/// Configuration for the IMM filter.
#[derive(Debug, Clone)]
pub struct ImmConfig {
    /// Transition probability matrix (row-major, N x N for N models).
    pub transition_matrix: Vec<Vec<f64>>,
    /// Initial model probabilities (length N).
    pub model_probabilities: Vec<f64>,
}

impl Default for ImmConfig {
    fn default() -> Self {
        // Default: two models (constant velocity + constant acceleration)
        // with high self-transition probability.
        Self {
            transition_matrix: vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            model_probabilities: vec![0.5, 0.5],
        }
    }
}

/// IMM filter backed by Stone Soup.
pub struct ImmFilter {
    /// The Python IMM predictor object.
    py_predictor: Py<PyAny>,
}

impl ImmFilter {
    /// Create a new IMM filter.
    ///
    /// `models` is a Python list of Stone Soup transition model objects.
    pub fn new(
        py: Python<'_>,
        config: &ImmConfig,
        models: &Bound<'_, PyList>,
    ) -> BridgeResult<Self> {
        let error = |error| BridgeError::from_python(py, error, "IMM construction");
        let np = import_module(py, "numpy")?;

        // Build transition matrix as numpy array.
        let trans_list = PyList::empty(py);
        for row in &config.transition_matrix {
            let py_row = PyList::new(py, row).map_err(error)?;
            trans_list.append(py_row).map_err(error)?;
        }
        let array = required_attribute(np.as_any(), "array", "IMM array lookup")?;
        let trans_matrix = array.call1((trans_list,)).map_err(error)?;

        let prob_list = PyList::new(py, &config.model_probabilities).map_err(error)?;
        let probs = array.call1((prob_list,)).map_err(error)?;

        let ss_imm = import_module(py, "stonesoup.predictor.interacting")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("sub_predictors", models).map_err(error)?;
        kwargs
            .set_item("transition_matrix", trans_matrix)
            .map_err(error)?;
        kwargs
            .set_item("model_probabilities", probs)
            .map_err(error)?;

        let imm_cls = required_attribute(ss_imm.as_any(), "IMMPredictor", "IMM predictor lookup")?;
        let predictor = imm_cls.call((), Some(&kwargs)).map_err(error)?;

        Ok(Self {
            py_predictor: predictor.unbind(),
        })
    }

    /// Predict the state forward using the IMM filter.
    ///
    /// `prior` is a Stone Soup `GaussianState`.
    /// `timestamp` is a Python datetime for the prediction target time.
    /// Returns the predicted state.
    pub fn predict(&self, prior: &Py<PyAny>, timestamp: &Py<PyAny>) -> BridgeResult<Py<PyAny>> {
        with_python("IMM prediction", |py| {
            let method = required_attribute(
                self.py_predictor.bind(py),
                "predict",
                "IMM prediction lookup",
            )?;
            method
                .call1((prior.bind(py), timestamp.bind(py)))
                .map(Bound::unbind)
                .map_err(|error| BridgeError::from_python(py, error, "IMM prediction"))
        })
    }
}
