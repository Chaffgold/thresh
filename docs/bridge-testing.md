# Stone Soup boundary and reference verification

The `Stone Soup boundary` CI job runs on every pull request, including PRs to
`develop`, without path-based skipping. It verifies the Rust/Python boundary and
bounded linear-Gaussian JPDA/GM-PHD reference graphs. It does **not** certify
native/reference parity, full-track accuracy, nonlinear algorithm behavior,
MHT/IMM reference graphs, trained-model acceptance, provider-data permissions, or
deployment without a linked Python shared library. All inputs are synthetic;
no OpenSky access or data is needed. See [bridge-references.md](bridge-references.md)
for the explicit APIs and pinned compatibility semantics.

## Exact isolated runtime

Use **CPython 3.12.13**, the workspace-locked **PyO3 0.29.0**, **Stone Soup 1.9.1**,
**NumPy 2.5.3**, and **SciPy 1.18.1**. All direct/transitive Python versions and
distribution hashes are recorded in
`crates/thresh-bridge/tests/requirements.txt`, independently of the training lock.
No OR-Tools is required. Run from the repository root in Bash; validation failures
stop the procedure:

```bash
set -euo pipefail
python3.12 --version  # Must report 3.12.13.
bridge_env="$(mktemp -d)/venv"
python3.12 -m venv "$bridge_env"
"$bridge_env/bin/python" -m pip install --require-hashes -r crates/thresh-bridge/tests/requirements.txt
export PYO3_PYTHON="$bridge_env/bin/python"
"$PYO3_PYTHON" -m unittest discover -s crates/thresh-bridge/tests -p test_runner.py -v
"$PYO3_PYTHON" crates/thresh-bridge/tests/run_boundary_tests.py
PYTHONPATH="$("$PYO3_PYTHON" -c 'import os, site; print(os.pathsep.join(site.getsitepackages()))')"
export PYTHONPATH
bridge_library_path="$("$PYO3_PYTHON" -c '
import os, runpy
environment = os.environ.copy()
runner = runpy.run_path("crates/thresh-bridge/tests/run_boundary_tests.py")
runner["configure_shared_library"](environment)
print(environment.get("LD_LIBRARY_PATH", ""))
')"
if [ -n "$bridge_library_path" ]; then
  export LD_LIBRARY_PATH="$bridge_library_path"
fi
cargo test --locked -p thresh-bridge --no-default-features --features stonesoup --doc
```

The runner prints the interpreter, locked PyO3 version, and every installed
Python package version. It rejects a mismatched interpreter, a missing package,
an incorrect pinned version, a failed package import, or a non-virtualenv host.
It sets `PYO3_PYTHON` and `PYTHONPATH` only for the Cargo/test child processes.
On Linux it also adds the selected interpreter's configured shared-library
directory to child `LD_LIBRARY_PATH`, because the test executables run directly
rather than through Cargo's runtime search-path setup.
The manual and CI doctest steps reuse that library-location validation before
exporting the Linux path in their own shell, preserving any existing value;
the runner's child settings do not modify the calling shell. Non-Linux library
paths remain unchanged.
`PYTHONPATH` is necessary because an embedded interpreter does not automatically
find a virtualenv's site-packages. These environment settings never move into
the bridge library; the host still owns Python initialization and lifetime.
On macOS with multiple developer tools, select the working Command Line Tools
installation through `DEVELOPER_DIR` if required by your local toolchain.

## Checked test manifest

`crates/thresh-bridge/tests/boundary-test-manifest.json` records exact test names
for the library unit tests and every integration-test target. The runtime command
currently requires **106 passed tests across eight targets, zero ignored/failed**:

| Target | Required tests |
| --- | ---: |
| Library unit tests | 36 |
| `context_contract` | 11 |
| `detection_contract` | 17 |
| `numeric_contract` | 9 |
| `jpda_reference` | 11 |
| `phd_reference` | 10 |
| `runtime_contract` | 10 |
| `stonesoup_integration` | 2 |

The last target contains the original import and nalgebra round-trip tests; they
are no longer ignored and initialize their host explicitly. The library target
includes physical-time, covariance, overflow, and interpreter-attachment tests.
Initialized-host wrapper-step tests use Python-owned dispatch doubles to
verify ordered argument and return-object identity, repeated host reuse,
unchanged host path/environment, missing-method diagnostics, and ordinary Python
exception details. Dispatch doubles are plumbing checks, not algorithm evidence.
JPDA and GM-PHD additionally execute real constructor/scan tests with independent
numerical expectations: exhaustive exclusive-event marginals for JPDA and
hand-derived prediction, intensity, birth, and reduction fixtures for GM-PHD.
The PHD step tests also assert input preservation across the deep-copy boundary
and operation-specific failures. MHT/IMM remain unresolved reference graphs.
The integration targets cover five observation profiles, frame/pose/covariance,
canonical and large finite azimuths, metadata, ndarray-subclass conversion,
array rejection, lifecycle, and structured errors. Fresh-child process
tests prevent imported modules or prior interpreter initialization from hiding
missing-runtime/package cases. Feature-enabled executable documentation examples
run as a separate gate after the manifest.

The runner first checks Cargo's complete integration-target inventory against the
manifest, builds all targets with `--locked`, then lists every executable's tests.
Only an exact match can proceed. It executes each complete target with stable
libtest text output, verifies every named result is `ok`, and checks the single
summary has the exact expected passed count and zero failed, ignored, measured,
or filtered tests. No test-name filtering is exposed. Any added, removed, renamed,
ignored, or silently unselected test fails the gate until the checked manifest is
deliberately updated. `--manifest PATH` allows testing an alternate manifest; CI
always uses the checked default.

The **14 stdlib runner self-tests** exercise success and fail-closed behavior for
empty selection, missing/renamed/unlisted tests, ignored/failed assertions, failed
children, missing/incorrect dependencies, zero execution, filtered/duplicate
results, missing test targets, and Linux shared-library discovery. They require no
third-party Python packages.
An ordinary `cargo test` exit code of zero is not a substitute for this gate.

## Recorded reference verification (2026-11-22)

The JPDA/GM-PHD repair was verified locally with the exact runtime above and
unchanged dependency locks. The complete manifest passed all 106 tests with zero
ignored, failed, or filtered tests; all 14 runner self-tests and seven
feature-enabled doctests also passed. The library inventory includes 15 PHD
contract/reducer tests and two JPDA construction/dispatch tests.

Observed maximum absolute residuals across the independent numerical runs were
`3.3306690738754696e-16` for JPDA marginals, `2.220446049250313e-16` for PHD
prediction/update fixtures, and `0` for the reducer fixtures. General fixture
tolerances are `atol=1e-12`, `rtol=1e-10`; endpoint epsilon assertions and
deliberately constructed gate-equality cases are checked separately. A negative
control changed the expected scalar detected weight from `0.5` to `0.51` and
failed the oracle assertion (Cargo exit 101); the perturbation was removed and
the suite passed again. These residuals describe tiny synthetic fixtures, not
sequence-level tracking accuracy.

Independent source/test review checked all six added requirements and 18
scenarios. A missing preflight-order regression check was added: when detections
is a generator, counted reusable prior/birth collections remain unread and the
generator remains unconsumed. No review findings remain. Caller-model
compatibility, including updater-owned fallback models, remains an explicit
precondition documented in [bridge-references.md](bridge-references.md).

Warning-strict bridge Clippy/docs, workspace formatting, normal workspace tests,
and native dependency-tree/build/tests with Python discovery disabled passed.
The native tree contains no PyO3. Ordinary workspace tests retain pre-existing
ignored tests; they are separate from the zero-ignored required bridge manifest.
OpenSpec 1.13.2 strict validation passed 52 live/spec items and 29 archives, with
valid metadata for all four active changes. This is local verification, not a
claim that a GitHub PR or remote CI has completed.

## Native workspace without Python

The separate `Native workspace without Python` CI job excludes only `thresh-py`,
whose PyO3 dependency is intentionally unconditional. All existing Python-binding,
training, lint, documentation, benchmark, and OpenSpec jobs remain intact.
Run the native check from a fresh shell, or explicitly clear a boundary virtualenv:

```bash
set -euo pipefail
unset PYTHONPATH VIRTUAL_ENV
export PYO3_NO_PYTHON=1
export PYO3_PYTHON=/__thresh_python_discovery_disabled__
export PYTHON_SYS_EXECUTABLE=/__thresh_python_discovery_disabled__
dependency_tree="$(cargo tree --color never --locked --workspace --exclude thresh-py --no-default-features --prefix none)"
native_grep_status=0
grep -Eq '^pyo3([ -]|$)' <<< "$dependency_tree" || native_grep_status=$?
case "$native_grep_status" in
  0)
    echo 'Unexpected PyO3 dependency in native workspace' >&2
    exit 1
    ;;
  1) ;;
  *)
    echo "Native dependency check failed (grep exit $native_grep_status)" >&2
    exit "$native_grep_status"
    ;;
esac
cargo build --locked --workspace --exclude thresh-py --no-default-features
cargo test --locked --workspace --exclude thresh-py --no-default-features
```

The dependency check rejects PyO3 and its `pyo3-*` support crates. Only grep's
no-match status is accepted; dependency-tree and grep execution errors fail the
check. Invalid Python discovery paths and `PYO3_NO_PYTHON=1` prevent a
Python-enabled build script from
quietly relying on the machine's interpreter. This lane does not disable the
independent `thresh-py` integration job or weaken any feature-enabled boundary test.
