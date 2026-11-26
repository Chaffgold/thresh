# Native reference tests

The native MHT and IMM capabilities are verified against independent, Rust-only
references that run in the `native-no-python` CI lane, with no Python, Stone
Soup or GPU. Because `cargo test` passes silently when a test is ignored,
renamed or filtered out, those reference targets are also protected by an
exact-name gate.

| Package | Test target | Required tests | What it verifies |
|---|---|---|---|
| `thresh-association` | `mht_reference` | 30 | `mht-tracking` "Exhaustive reference verification": a std-only exhaustive enumerator of global association hypotheses (C1–C5 fixtures), checked against a second enumeration order and hand-derived anchors. |
| `thresh-filter` | `imm_reference` | 46 | `imm-estimation`: exact-rational and given-likelihood cases, an independent nalgebra-only oracle, native conformance on configurations that depend on no unspecified policy, and configuration validation. |
| `thresh-filter` | `imm_current_policy` | 13 | Current-policy characterisation (`current_policy_*`). Not specified behaviour and not cited as verification; these change deliberately in `correct-native-imm-mode-policies`. |
| **Total** | | **89** | |

The references share no code with the production code they check: each oracle
source is scanned by an `include_str!` test that rejects any `thresh_` path.
Neither reference certifies native MHT conformance; see the `mht-tracking`
scenario "Reference evidence is not native conformance".

## The gate

- **Manifest:** `scripts/native-reference-manifest.txt`, one
  `<package> <test-target> <test-name>` entry per line.
- **Checker:** `scripts/check-native-reference-tests.sh` (bash and cargo only).
  For each manifest target it:
  1. rejects an empty, duplicated or malformed manifest, or one with no entry
     for a protected target;
  2. compares `cargo test --locked -p <package> --no-default-features --test <target> -- --list --format terse`
     with the manifest, failing on any missing or unlisted test;
  3. runs the target and requires every listed test to report `ok` (a
     `#[should_panic]` test reports `test <name> - should panic ... ok`), and
     exactly one summary of
     `N passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`, where N is
     the manifest count.
- **Self-test:** `--self-test` runs the checker's own logic against canned
  libtest output and manifests. It must fail on missing, unlisted, ignored,
  failed, filtered-out and zero-test output, on a wrong summary, and on an
  empty, duplicated or malformed manifest, and must pass on a matching
  inventory that includes a should-panic line.

Run locally:

```sh
bash scripts/check-native-reference-tests.sh --self-test
PYO3_NO_PYTHON=1 bash scripts/check-native-reference-tests.sh
```

CI runs both steps in the `native-no-python` job, after the native workspace
tests, with Python discovery disabled and no path filtering.

The checker must stay portable to bash 3.2 (macOS `/bin/bash`) and to both BSD
and GNU userlands: no associative arrays, `mapfile`/`readarray` or `${x,,}`,
and no GNU-only flags such as `grep -P`, suffixless `sed -i` or `sort -V`.

## Changing a protected test

Adding, removing or renaming a test in a protected target requires the same
edit to the manifest, so the change is visible in review. Never mark a
reference test `#[ignore]`; the gate fails if any test is ignored.

## Recorded verification (2026-11-26)

Verified locally on macOS (bash 3.2 for the checker) with Python discovery
disabled (`PYO3_NO_PYTHON=1`) and `RUSTFLAGS=-Dwarnings`:

- `cargo tree --locked --workspace --exclude thresh-py --no-default-features`
  contains no PyO3; the native workspace builds, and its tests pass (1130
  passed, 0 failed; pre-existing ignored tests are outside the protected
  targets).
- The checker self-test passed all 19 cases, and the gate passed **89** required
  tests across 3 targets (30 `mht_reference`, 46 `imm_reference`, 13
  `imm_current_policy`) with zero ignored. Temporarily marking one
  `imm_reference` test `#[ignore]` made the gate fail
  (`44 passed; 0 failed; 1 ignored`, before a 46th test was added); the change
  was reverted.

Maximum observed residuals:

| Comparison | Target | Max absolute | Max relative | Comparisons |
|---|---|---|---|---|
| Two MHT enumeration orders (per-scan and cumulative scores) | `mht_reference` | 3.6e-15 | — | all C1–C5 hypotheses |
| Native vs IMM oracle | `imm_reference` | 6.8e-13 | 2.6e-14 | 211 |
| IMM oracle vs hand-derived exact rationals | `imm_reference` | 1.6e-12 | 1.8e-14 | 52 |
| Native vs hand-derived exact rationals | `imm_reference` | 1.4e-14 | 1.2e-15 | 24 |
| Native vs IMM oracle (current-policy emulation) | `imm_current_policy` | 1.1e-13 | 1.8e-14 | 194 |
| Native and oracle vs 10-decimal characterisation pins | `imm_current_policy` | 4.8e-11 | 4.1e-11 | 65 |

Relative residuals are against `max(|expected|, 1)`. The mean mixing invariant
is checked separately, with its tolerance scaled by the magnitude of the summed
terms, and is not counted here. The required tolerance is
`atol = 1e-10` plus `rtol = 1e-9`; the characterisation pins are rounded to ten
decimals, which accounts for their larger residuals. To reproduce:

```sh
cargo test -p thresh-association --test mht_reference enumeration_orders_agree -- --nocapture
IMM_REFERENCE_RESIDUALS=1 cargo test -p thresh-filter --test imm_reference -- --nocapture
IMM_REFERENCE_RESIDUALS=1 cargo test -p thresh-filter --test imm_current_policy -- --nocapture
```

This is local verification, not a claim that a GitHub PR or remote CI has
completed.
