# Orbital & Ballistic Filter Models

## Why

thresh claims coverage "from UAVs to ballistic missiles," but no filter can actually predict orbital or ballistic dynamics: the physics (two-body + J2 + exponential-drag RK4) exists only inside `thresh-synth`'s truth generator, while `TrackHead::ballistic()` tracks missiles with a generic 9D constant-acceleration model (20 m/s² process noise) and the orbital benchmark passes with pipeline-smoke baselines (`mota = -2.0` on the ISS scenario, `-5.0` on the Starlink train). Closing this gap gives trackers physically correct prediction between measurements — the single highest-leverage accuracy improvement for the orbital and ballistic target classes — and it must land before the follow-on `astro-time-and-frames` and orbit-propagation-fidelity changes, which refine (not create) this dynamics layer. Stone Soup deprecated all of its orbital functionality in v1.9 (removal in v1.10), so there is no upstream to lean on; only its `OrbitalState` type design (element-representation enum, parameterized gravitational constant) is worth imitating — with the frame discipline it lacked.

## What Changes

- **thresh-filter**: add a Kepler+J2 orbital motion model (6D ECI Cartesian, nonlinear, EKF/UKF/CKF-compatible via the existing `MotionModel` trait) and a 7D ballistic reentry model ([position, velocity, drag-beta]) with altitude-dependent exponential atmospheric density. Optionally extend the IMM bank for boost/midcourse/reentry mode switching.
- **Shared force math**: move the two-body/J2/drag acceleration functions out of `thresh-synth/src/orbital.rs` into a shared home (thresh-core or similar — design decision) so filters and truth generation use identical physics. `thresh-synth` keeps re-export shims where feasible.
- **Orbital state type**: a frame-disciplined orbital state with an element-representation enum (Cartesian / Keplerian / TLE / Equinoctial), lazy conversion between representations, and a **parameterized** gravitational parameter (no hardcoded Earth mu in the new types).
- **thresh-synth**: phased ballistic truth generation — boost (constant thrust + gravity turn), midcourse (wired to the *existing* J2 RK4 propagator), and beta-parameterized reentry — over a round rotating Earth. **BREAKING**: replaces the toy flat-Earth `SegmentType::Ballistic { drag_coefficient }` (constructors in `trajectory.rs` tests and `crates/thresh/tests/integration.rs` must migrate).
- **thresh-tracker**: **BREAKING** — `TrackHead::ballistic()` switches from 9D CA to the 7D reentry model (state dimension and covariance layout change for downstream consumers); a new orbital head uses the Kepler+J2 model.
- **thresh-data**: one ballistic benchmark scenario TOML; tighten the orbital CI benchmark from the smoke baselines (`mota = -2.0` ISS, `-5.0` Starlink train) to a real accuracy gate once the orbital model lands.
- **Validation**: golden-vector fixtures checked into the repo, generated offline (AIAA SGP4 verification vectors via the Rust `sgp4` crate's validated output; Vallado worked examples for Kepler/J2 propagation) — deterministic, no Python in CI. Stone Soup's element-conversion tests (MIT) ported as Rust test vectors.
- **docs/reference**: new self-contained derivation doc for the orbital and ballistic dynamics (Vallado/Farrell style, matching the repo's existing reference-doc pattern).

**Out of scope** (stated explicitly): 6DOF missile aerodynamics (JSBSim stays aircraft-only), staging/RV-deployment realism, thrust-profile libraries, learned dynamics, association/fusion changes, maneuver detection, SRP/third-body/higher-order gravity (later orbit-propagation-fidelity change), time-scale and frame overhaul including UT1/precession-nutation (later `astro-time-and-frames` change), atmospheric measurement refraction (later `atmospheric-measurement-propagation` change), and any Stone Soup bridge changes.

## Capabilities

### New Capabilities

- `orbital-motion-model`: Kepler+J2 orbital motion model in thresh-filter — 6D ECI Cartesian state, nonlinear transition with Jacobian for EKF and sigma-point compatibility for UKF/CKF, force math shared with thresh-synth, validated against Vallado worked examples and SGP4-derived golden vectors.
- `ballistic-reentry-model`: 7D ballistic reentry motion model — [position, velocity, ballistic-coefficient beta] with altitude-dependent exponential density over a round rotating Earth, filter-compatible via the `MotionModel` trait.
- `orbital-state-representation`: frame-disciplined orbital state type with element-representation enum (Cartesian/Keplerian/TLE/Equinoctial), lazy conversion, and parameterized gravitational constant — the Stone Soup `OrbitalState` shape, improved with frame tags.
- `orbital-ballistic-benchmarks`: ballistic tracking benchmark scenario plus a real accuracy baseline for the existing orbital CI benchmark gate (replacing the `mota = -2.0` / `-5.0` pipeline-smoke thresholds).

### Modified Capabilities

- `synthetic-data`: the "Configurable target trajectory generation" requirement's ballistic behavior upgrades from a flat-Earth gravity+drag toy segment to phased boost/midcourse/reentry truth generation over a round rotating Earth, reusing the existing J2 RK4 propagator for midcourse.
- `track-management`: the "Class-specific track heads" requirement strengthens — the ballistic head MUST use the physics-based reentry model (not generic CA), and a dedicated orbital head using the Kepler+J2 model MUST exist.

Note: `state-estimation` is intentionally **not** modified — its "Custom motion model" scenario already requires filters to accept new `MotionModel` implementations without filter-code changes, which this change exercises rather than alters. `hifi-orbital` (nyx-space full force models) is also untouched; this change is the low-fidelity, filter-side stepping stone toward it.

## Impact

- **Crates**: `thresh-core` (or a new shared module) gains the common acceleration functions and the orbital state type; `thresh-filter` gains two motion models (+ optional IMM bank extension); `thresh-synth` reworks `SegmentType::Ballistic` and delegates midcourse to its existing propagator; `thresh-tracker` reworks the ballistic head and adds an orbital head; `thresh-data` adds one scenario TOML and a tightened baseline.
- **Breaking API surface**: `SegmentType::Ballistic` fields (**BREAKING**, constructors: `thresh-synth/src/trajectory.rs` tests, `crates/thresh/tests/integration.rs`); `TrackHead::ballistic()` state dimension 9 → 7 (**BREAKING** for anything reading its state layout). Both are pre-1.0, GitHub-org-internal — no crates.io consumers.
- **CI**: the "Orbital benchmark gate" baselines tighten from their smoke values (`mota = -2.0` in `crates/thresh-data/scenarios/orbital-iss.toml`, `-5.0` in `orbital-starlink-train.toml`); golden-vector fixtures run as plain `cargo test` — no Python, no new feature gates (pure Rust math via nalgebra).
- **Dependencies**: none added; explicitly does **not** depend on Stone Soup (orbital support deprecated upstream in v1.9) or nyx-space (deferred to the hifi-orbital work).
- **Conventions**: clippy `-D warnings`; SonarCloud cognitive complexity ≤ 15 via phase-helper decomposition (worked example: `rk4_stage` in `orbital.rs`).
- **Follow-on changes**: `astro-time-and-frames` (ECI/TEME GMST-only conflation stays as-is here by design), orbit-propagation-fidelity (SRP/third-body/higher-order gravity), atmospheric-measurement-propagation.
