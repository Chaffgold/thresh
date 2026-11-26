# Stone Soup bridge runtime contract

The `stonesoup` feature enables a boundary to an already initialized Python
interpreter. It does not initialize or finalize Python. Standalone Rust hosts
must initialize explicitly; Python-hosted applications reuse their interpreter.

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thresh_bridge::jpda::{JpdaAssociator, JpdaConfig};
use thresh_bridge::phd::{PhdFilter, PhdConfig};

fn main() {
    Python::initialize(); // Host-owned, idempotent when already initialized.
    Python::attach(|py| {
        let objects = PyDict::new(py);
        py.run(c"\
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
predictor = KalmanPredictor(ConstantVelocity(0.1))
updater = KalmanUpdater(LinearGaussian(ndim_state=2, mapping=[0], noise_covar=[[1.0]]))
", Some(&objects), None).unwrap();
        let predictor = objects.get_item("predictor").unwrap().unwrap().unbind();
        let updater = objects.get_item("updater").unwrap().unwrap().unbind();
        let _jpda = JpdaAssociator::new(&JpdaConfig::default(), &predictor, &updater).unwrap();
        let _phd = PhdFilter::new(py, PhdConfig::default(), predictor.bind(py), updater.bind(py))
            .unwrap();
    });
}
```

JPDA and GM-PHD now require explicit compatible predictor/updater components;
the example supplies invented dynamics and observation noise, not defaults
chosen by the bridge. Their linear-Gaussian reference tests exercise real
construction and calls. The bridge provides JPDA and GM-PHD references only; it
has no MHT or IMM entry point, because those contracts are native
(`mht-tracking`, `imm-estimation`) and both have interim gaps. None of these
checks certifies full-track accuracy or deployment readiness. See
[bridge-references.md](bridge-references.md) for the scan APIs and compatibility
limits. Missing dependencies and ordinary upstream failures remain structured
errors, not evidence of a missing interpreter.

JPDA's constructor now accepts Python-owned objects, so it cannot be exercised
with valid inputs before host initialization. The only standalone bridge entry
point is `measurement_to_detection`; fresh-process startup assertions remain on
it and on the shared attachment helper. Initialized JPDA/GM-PHD calls verify host
reuse without path or environment mutation.

Token-free calls use `Python::try_attach`, returning
`BridgeError::PythonNotAvailable` when attachment is unavailable. This remains
true if another dependency enables PyO3 auto-initialization: `try_attach` never
initializes. Methods that accept an existing `Python` token use that token.
The library does not edit `sys.path` or process environment variables. Configure
the matching interpreter and package paths in the host or CI before startup.

Keep Python alive while bridge objects exist. Concurrent host finalization,
subinterpreters, and free-threaded interpreters are not certified by the initial
CPython 3.12 lane. PyO3's attachment checks during shutdown are best-effort, not a
guarantee that racing finalization is safe.

## Error API migration

`From<PyErr>` is removed: it could reattach while formatting an exception. When
a token is already held, use
`BridgeError::from_python(py, error, "operation description")`. At a specifically
requested attribute lookup, use `from_python_api(py, error, operation, api)`.
Do not use the API mapper for arbitrary Python execution.

`PythonError` and `StoneSoupNotInstalled` now contain `PythonException`, not a
string. It exposes `operation`, `exception_type`, `message`, and optional
`traceback`; these are captured while attached and can subsequently be formatted
without Python access. Python-backed errors also expose the cause through
`std::error::Error::source`.

| Category | Meaning |
|---|---|
| `PythonNotAvailable` | Host interpreter cannot be attached to |
| `StoneSoupNotInstalled` | `ModuleNotFoundError.name` is exactly `stonesoup` |
| `MissingDependency { name, source }` | Another required module is missing, including NumPy or a transitive dependency |
| `UnsupportedApi { api, source }` | A requested Stone Soup submodule or explicitly looked-up symbol is unavailable |
| `PythonError` | Other exceptions, retaining their original type and message |
| `ConversionError` | Invalid Rust/context/numeric conversion input |
| `AlgorithmError` | Explicit algorithm-specific validation errors |

Classification uses exception type and missing-module identity, never message
substrings. A TypeError mentioning `stonesoup`, for example, remains a Python
error. A missing module exception without a usable `name` also remains a Python
error instead of guessing its package identity.

## Deployment and verification limits

The host must be able to load the Python shared library selected at build time,
and that interpreter must be able to locate the pinned runtime packages. A
missing shared library can prevent process startup **before Rust can return an
error**. This change does not add dynamic loading and does not fulfill the
existing broader graceful-deployment promise by itself.

Lifecycle and dependency tests run in fresh child processes. Import hooks are
test-local; they never uninstall packages or mutate the user's environment.
Boundary conformance does not imply numerical algorithm parity or model-release
acceptance, and it does not require provider data.
