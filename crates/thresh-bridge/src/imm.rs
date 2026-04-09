//! Wrapper around Stone Soup's IMM (Interacting Multiple Model) filter.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::{BridgeError, BridgeResult};

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
        let np = py.import("numpy").map_err(BridgeError::from)?;

        // Build transition matrix as numpy array.
        let trans_list = PyList::empty(py);
        for row in &config.transition_matrix {
            let py_row = PyList::new(py, row)?;
            trans_list.append(py_row)?;
        }
        let trans_matrix = np.call_method1("array", (trans_list,))?;

        let prob_list = PyList::new(py, &config.model_probabilities)?;
        let probs = np.call_method1("array", (prob_list,))?;

        let ss_imm = py
            .import("stonesoup.predictor.interacting")
            .map_err(BridgeError::from)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("sub_predictors", models)?;
        kwargs.set_item("transition_matrix", trans_matrix)?;
        kwargs.set_item("model_probabilities", probs)?;

        let imm_cls = ss_imm.getattr("IMMPredictor")?;
        let predictor = imm_cls.call((), Some(&kwargs))?;

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
        Python::with_gil(|py| {
            let result = self
                .py_predictor
                .call_method1(py, "predict", (prior.bind(py), timestamp.bind(py)))
                .map_err(BridgeError::from)?;
            Ok(result)
        })
    }
}
