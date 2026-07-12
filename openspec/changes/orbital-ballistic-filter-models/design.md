# Design — Orbital & Ballistic Filter Models

## Context

The physics needed to predict orbital and ballistic motion already exists in the repo — but only on the truth-generation side. `crates/thresh-synth/src/orbital.rs` has two-body + J2 gravity (`acceleration_j2`), a 28-row piecewise-exponential atmosphere (`ATMOSPHERE_TABLE` + `atmosphere_density`), co-rotating-atmosphere drag (`acceleration_drag`), and a fixed-step RK4 integrator (`rk4_stage` / `rk4_step` / `propagate`). None of it is reachable from `thresh-filter`, which offers only kinematic models (CV / CA / CTRV / CoordinatedTurn) behind the `MotionModel` trait:

```rust
pub trait MotionModel {
    fn state_dim(&self) -> usize;
    fn predict(&self, state: &DVector<f64>, dt: f64) -> DVector<f64>;
    fn jacobian(&self, state: &DVector<f64>, dt: f64) -> DMatrix<f64>;
    fn process_noise(&self, dt: f64) -> DMatrix<f64>;
}
```

The EKF consumes `jacobian` for covariance propagation only (`ekf.rs:22` — `self.x = model.predict(...)`, `self.p = F·P·Fᵀ + Q`); UKF/CKF consume `predict` through sigma points and never call `jacobian`. So a new nonlinear model is EKF/UKF/CKF-compatible for free once those four methods exist — the `state-estimation` spec's "Custom motion model" scenario already guarantees this.

Downstream, `TrackHead::ballistic()` (`crates/thresh-tracker/src/heads.rs:38-49`) is a config record (`state_dim: 9`, `process_noise_sigma: 20.0`, 2-of-3 confirmation) with **no model dispatch behind it**: the tracker's single-model path hardcodes `ConstantVelocity::new(5.0)` for every track regardless of head (`tracker.rs:358`). The orbital CI benchmark passes with explicit pipeline-smoke values: `mota = -2.0` (`crates/thresh-data/scenarios/orbital-iss.toml:35`) and `mota = -5.0` (`orbital-starlink-train.toml`). Truth-side, `SegmentType::Ballistic { drag_coefficient }` in `trajectory.rs` is a flat-Earth toy (`g = 9.81`, `-k·v·|v|` drag) with no boost, no round Earth, no β = m/(C_d·A).

Constraints inherited from the repo: dependency flow is `thresh-core ← thresh-filter ← thresh-tracker ← thresh-synth/thresh-data` (lower depends on higher); `thresh-core` depends only on nalgebra + serde/serde_json; clippy `-D warnings`; SonarCloud cognitive complexity ≤ 15 (phase-helper decomposition); ECI/TEME are deliberately conflated under a GMST-only rotation (`thresh-core/src/eci.rs`) until the `astro-time-and-frames` change; Stone Soup's orbital functionality is deprecated upstream (v1.9, removal v1.10) — only its `orbitalstate.py` type shape is borrowed, never its code path.

## Goals / Non-Goals

**Goals:**

- Give the filter family physically correct orbital (Kepler+J2) and ballistic-reentry (β-parameterized drag) prediction via two new `MotionModel` implementations.
- Single source of truth for force math: filters and synthetic truth use byte-identical acceleration functions.
- A frame-disciplined, mu-parameterized orbital state representation (Cartesian / Keplerian / TLE / Equinoctial) usable by ingest, head initialization, and analysis code.
- Phased ballistic truth generation (boost / midcourse / reentry) over a round rotating Earth, reusing the existing RK4 J2 propagator.
- Real accuracy baselines on the orbital CI gate plus one new ballistic benchmark scenario.
- Golden-vector validation that runs as plain `cargo test` — deterministic, offline, no Python.

**Non-Goals:**

- 6DOF missile aerodynamics, staging/RV-deployment realism, thrust-profile libraries (JSBSim stays aircraft-only).
- SRP, third-body, higher-order gravity harmonics (later orbit-propagation-fidelity change).
- Fixing the ECI/TEME GMST-only conflation, UT1, precession/nutation (later `astro-time-and-frames` change).
- Atmospheric measurement refraction (later `atmospheric-measurement-propagation` change).
- Association/fusion changes, maneuver detection, learned dynamics, Stone Soup bridge changes.
- Adaptive-step or high-order integrators — fixed-step RK4 matches the existing propagator and is sufficient at tracker dt scales.

## Decisions

### Decision 1: Shared force math lives in a new `thresh_core::orbital` module

**Decision:** Move the acceleration functions and their constants out of `thresh-synth/src/orbital.rs` into a new `thresh-core/src/orbital/` module (submodules `gravity.rs`, `atmosphere.rs`, `integrate.rs`):

- `GravityModel { mu: f64, j2: f64, equatorial_radius: f64 }` with a `GravityModel::EARTH_WGS84` const — **mu is a field, never a hardcoded constant inside the math** (the `GM_EARTH` / `J2` / `EARTH_RADIUS` consts in `thresh-synth/src/orbital.rs:12-18` become fields of this const).
- `two_body_acceleration(pos, &GravityModel)`, `j2_acceleration(pos, &GravityModel)` (the body of today's `acceleration_j2`).
- `atmosphere_density(alt_m)` + `ATMOSPHERE_TABLE` moved verbatim.
- `drag_acceleration(pos, vel, inv_beta)` — the body of today's `acceleration_drag`, re-parameterized on inverse ballistic coefficient `inv_beta = C_d·A/m = 1/β` (co-rotating atmosphere kept as-is; the local `omega_e = 7.292_115e-5` literal becomes a reference to `thresh_core::eci::EARTH_ROTATION_RATE`, which has the identical value, preserving bitwise identity).
- `rk4_step` / `rk4_stage` generalized to take the acceleration as a closure `impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>` instead of `&PropagatorConfig`.

Core functions use `nalgebra::Vector3<f64>` (already a `thresh-core` dependency); `thresh-synth` keeps its `[f64; 3]`-based public signatures as thin shims that convert and delegate (`pub fn acceleration_j2(pos: &[f64; 3]) -> [f64; 3]` calls `thresh_core::orbital::j2_acceleration(..., &GravityModel::EARTH_WGS84)`). `PropagatorConfig`, `DragConfig`, `propagate`, `apply_maneuver`, and the visibility helpers stay in `thresh-synth` — they are truth-generation orchestration, not shared physics. The synth shim computes `inv_beta = cd * area_m2 / mass_kg` from `DragConfig`, so `DragConfig` is unchanged and all 10 existing `orbital.rs` tests pass without edits (the refactor is arithmetic-identical, and the tolerance-based tests — energy conservation `< 1e-8`, ISS altitude band, GEO drift — are insensitive to `[f64;3]`↔`Vector3` conversion).

**Alternatives considered:**

- *New `thresh-astro` crate.* Cleanest isolation, but a 14th workspace crate for ~300 lines of pure math that has no dependencies beyond what `thresh-core` already carries is overhead without benefit. Feature gates are reserved for heavy deps (project convention); this is neither heavy nor optional.
- *Host in `thresh-filter`, have `thresh-synth` depend on it.* Works with the dependency flow (synth sits below filter), but couples truth generation to the filter crate and puts frame/physics math in a crate whose concern is estimation. `thresh-core` already owns the adjacent concerns (`eci.rs`, `geodetic.rs`, `coords.rs`).
- *Duplicate the math in `thresh-filter`.* Rejected outright: divergence between the filter's dynamics and the truth generator's dynamics is exactly the class of bug this change exists to prevent, and it would make the "filter matches truth propagator" golden test (Decision 8) meaningless.

### Decision 2: Orbital state representation — element enum in `thresh-core`, on-demand conversion, no memoization

**Decision:** New type in `thresh-core/src/orbital/state.rs` (borrowing the shape of Stone Soup's MIT-licensed `orbitalstate.py`, not its code):

```rust
pub enum OrbitalElements {
    Cartesian { position: Vector3<f64>, velocity: Vector3<f64> },        // m, m/s
    Keplerian { sma: f64, ecc: f64, inc: f64, raan: f64, argp: f64, true_anomaly: f64 },
    Equinoctial { sma: f64, h: f64, k: f64, p: f64, q: f64, mean_longitude: f64 },
    Tle { line1: String, line2: String },                                 // mean elements, SGP4 theory
}

pub enum OrbitalFrame { EciGmst, Teme }   // tag, not a type parameter

pub struct OrbitalState {
    pub elements: OrbitalElements,
    pub mu: f64,                          // gravitational parameter, m³/s² — REQUIRED, no default
    pub frame: OrbitalFrame,
    pub epoch_jd: f64,
}
```

Conversion is **on demand via pure methods** (`as_cartesian() -> Result<(Vector3, Vector3), ElementError>`, `as_keplerian()`, `as_equinoctial()`), not lazily memoized. Stone Soup memoizes because Python attribute access is its API idiom; in Rust, interior-mutability caching (`OnceCell`) would complicate `Clone`/`Serialize` for a conversion that costs nanoseconds of trigonometry. The existing `from_keplerian` / `to_keplerian` math in `thresh-synth/src/orbital.rs:114-281` (including the `compute_raan`/`compute_argp`/`compute_true_anomaly` phase helpers) moves into these methods, with `GM_EARTH` replaced by `self.mu`.

**Frame discipline (the improvement over Stone Soup):** every `OrbitalState` carries an `OrbitalFrame` tag. Under today's GMST-only rotation `EciGmst` and `Teme` are numerically identical *by documented design* (see `eci.rs`); the tag exists so TLE-derived states are honestly labeled `Teme` and the later `astro-time-and-frames` change can make the distinction real without an API break. Mixing frames in arithmetic helpers is a debug-assert now, an error later.

**TLE variant conversion:** `as_cartesian()` on a `Tle` variant returns `Err(ElementError::RequiresSgp4)` in `thresh-core` — SGP4 lives in the `sgp4` crate behind `thresh-data`'s `orbital` feature, and core must not grow that dependency. `thresh-data` provides the conversion (`tle_to_cartesian(&OrbitalState) -> Result<OrbitalState>`) next to its existing TLE/SGP4 pipeline. The variant still belongs in core: it lets ingest code carry a TLE-born state through frame-tagged plumbing without stringly-typed side channels.

**Mapping to the flat filter state:** `OrbitalState::to_filter_state() -> DVector<f64>` emits the 6D interleaved layout of Decision 3 (and its inverse constructor takes a `DVector` + epoch + mu). `OrbitalState` is the *ingest/initialization/analysis* type; filters continue to operate on bare `DVector<f64>` per the `MotionModel` trait — no filter code changes.

**Relationship to `thresh_synth::OrbitalState`:** the synth struct (`position: [f64;3], velocity: [f64;3], epoch_jd`) remains as the propagator's lightweight Cartesian sample type — it appears in serialized outputs and a dozen synth/data call sites, and forcing the enum type into the inner RK4 loop buys nothing. `From` conversions in both directions are provided (synth→core assumes `EciGmst` + `EARTH_WGS84.mu`). Full consolidation is deferred to `astro-time-and-frames`, which will touch every one of those call sites anyway.

**Alternatives considered:**

- *Typestate frames (`OrbitalState<Eci>`).* Compile-time frame safety is attractive, but with only one real frame convention in the codebase today it would be speculative generality, and it poisons every container (`Vec<OrbitalState<?>>` for mixed ingest). Revisit in `astro-time-and-frames`.
- *Struct-of-Option caching all representations (Stone Soup's effective shape).* Invalidation hazards on mutation; rejected per above.
- *Default mu = Earth.* Rejected — the entire point of parameterizing mu is that a silent Earth default reintroduces the hardcoding bug in every forgotten constructor call. `GravityModel::EARTH_WGS84.mu` is one explicit token at the call site.

### Decision 3: `KeplerJ2` motion model — interleaved 6D ECI state, sub-stepped RK4 predict, numeric Jacobian

**Decision:** `thresh-filter/src/models/kepler_j2.rs`:

```rust
pub struct KeplerJ2 {
    pub gravity: GravityModel,   // mu, j2, equatorial_radius — parameterized per Decision 1
    pub max_step_s: f64,         // RK4 sub-step ceiling, default 10.0
    pub sigma_accel: f64,        // unmodeled-acceleration PSD, m/s² (default ~1e-3)
}
```

- **State layout:** `[x, vx, y, vy, z, vz]` — **interleaved**, matching `ConstantVelocity`, the IMM common space (`imm.rs`: "common 6D representation `[x, vx, y, vy, z, vz]`"), and the tracker's existing 3×6 position observation matrices. The math internally unpacks to `(Vector3 pos, Vector3 vel)` pairs; a blocked `[r | v]` layout is more conventional in astrodynamics texts but would force every consumer (observation matrices, `birth_track`'s H-based init at `tracker.rs:757`, IMM mappings) to special-case it. The reference doc states the permutation explicitly.
- **Frame contract:** state is ECI (GMST convention) in metres / m/s. This is a documented contract on the type, not enforced — the `MotionModel` trait is frame-blind. Consumers that hold ENU measurements convert before update (Decision 7/8).
- **`predict`:** sub-steps the interval with the shared `rk4_step` (Decision 1) using `a(r) = two_body + J2`: `n = ceil(dt / max_step_s)` equal sub-steps. At `max_step_s = 10 s` this matches the fidelity the synth ISS test already validates (`dt_s: 10.0`, 1-day altitude-band check), and a 5 s tracker cadence costs exactly one RK4 step. Degenerate guard: if `‖r‖ < equatorial_radius / 2` the state is physically meaningless (filter divergence); clamp the radius used in the force evaluation to avoid NaN poisoning and let the covariance blow-up surface the problem — never panic inside `predict`.
- **Jacobian: numeric central differences of the exact `predict`,** not an analytic STM. Per-column step `h_i = ε·max(|x_i|, s_i)` with `ε = cbrt(f64::EPSILON) ≈ 6e-6` and floor scales `s_i` (1 m for position, 1e-3 m/s for velocity). Cost: 12 extra propagations per EKF predict — microseconds at tracker rates, and UKF/CKF never call it. Rationale: (a) EKF consistency requires the Jacobian of *the discretized map actually used*, and differencing the sub-stepped RK4 delivers that automatically, whereas the analytic route needs either the variational equations integrated alongside the state (13-dim augmented RK4) or a first-order `F ≈ I + A(x)·dt` that degrades over multi-step dt; (b) it generalizes unchanged to the 7D ballistic model. The analytic continuous-time `A(x) = ∂a/∂r` for two-body+J2 **is** derived in the reference doc (Decision 9) and used in a unit test that cross-checks the numeric Jacobian against `I + A·dt + O(dt²)` at small dt — analytic math validates, numeric math ships. The differencing lives in a shared `numeric_jacobian(f, x, dt, scales)` helper in `thresh-filter` (phase-helper style, own unit tests against CV's exact `F`).
- **Process noise:** discretized continuous white-noise acceleration per axis — the standard `Q = σ_a²·[[dt³/3, dt²/2],[dt²/2, dt]]` blocks in the interleaved layout (same structure as `ConstantVelocity`). `sigma_accel` absorbs exactly what the model omits: drag, SRP, higher harmonics, small maneuvers — order 1e-4…1e-3 m/s² for LEO defaults, exposed for tuning. A dynamics-shaped Q (rotated along-track/cross-track) is deliberately deferred to orbit-propagation-fidelity.

**Alternatives considered:** analytic STM via variational equations (correctness equal at best, triple the code, blocks the ballistic model from sharing the approach); `F ≈ I + A·dt` (cheap but inconsistent with the multi-step RK4 predict at radar-gap dt values — precisely the sparse-visibility regime the orbital benchmark exercises); Keplerian-element state with J2 secular rates (better-conditioned covariance, but nonlinear measurement mapping for every Cartesian sensor and singular at low e/i — Cartesian is the right first model, elements can come later as an alternative parameterization).

### Decision 4: 7D ballistic reentry model — `[x, vx, y, vy, z, vz, β]`, direct β with floor, random-walk β noise

**Decision:** `thresh-filter/src/models/ballistic_reentry.rs`:

```rust
pub struct BallisticReentry {
    pub gravity: GravityModel,
    pub max_step_s: f64,        // default 1.0 (reentry dynamics are fast)
    pub sigma_accel: f64,       // m/s², unmodeled lift/attitude effects (default ~5.0)
    pub sigma_beta: f64,        // kg/m² per √s, β random-walk intensity
    pub beta_floor: f64,        // default 10.0 kg/m²
}
```

- **State layout:** `[x, vx, y, vy, z, vz, β]` — the first six components identical to the interleaved 6D convention, β appended. This makes the IMM `StateMapping` trivial (drop/append β, exactly the `CaMapping` pattern of projecting extra components in `imm.rs:74-114`) and lets position-observation H matrices be the 6D ones padded with a zero column.
- **Frame:** ECI, same contract as `KeplerJ2`. The rotating round Earth enters through the force model, not the frame: gravity is two-body + J2 from the shared `GravityModel`, and drag uses the **co-rotating atmosphere relative velocity** already implemented in `drag_acceleration` (`v_rel = v − ω_⊕ × r`), so Coriolis/centrifugal effects are captured without an ECEF state. Altitude for density is geometric `‖r‖ − equatorial_radius` — the same spherical approximation the existing synth drag uses; the sub-1% density error from oblateness is far below exponential-model uncertainty (noted in the reference doc).
- **Dynamics:** `a = a_twobody+J2(r) + drag_acceleration(r, v, 1/β)` with `ρ` from the shared `atmosphere_density` (`ATMOSPHERE_TABLE` reused unmodified); `β̇ = 0`. Same sub-stepped RK4 `predict` as Decision 3 via the closure-based integrator.
- **β parameterization: direct β, clamped to `beta_floor` inside `predict`.** Estimating `1/β` or `ln β` was considered — `1/β` enters the dynamics linearly and `ln β` guarantees positivity — but direct β is what operators and the truth generator specify (kg/m²), keeps the state physically readable in track outputs, and the floor removes the divide-by-zero/negative-β failure mode. The reference doc records the log-β alternative for a future revision if filter consistency tests show β posterior skew.
- **Jacobian:** the shared `numeric_jacobian` helper (Decision 3) with a β column scale of `max(|β|, 100.0)·ε` — β sensitivities are well-conditioned since `∂a_drag/∂β = −a_drag/β`. No hand-derived 7×7 matrix to maintain.
- **Process noise:** 6D white-noise-acceleration blocks as in Decision 3 with `sigma_accel` sized for reentry (unmodeled lift, attitude oscillation: several m/s²), plus an independent β random walk `Q[6,6] = sigma_beta²·dt`. A nonzero β process noise is essential: effective β genuinely varies through the flight regime (Farrell's classic reentry-tracking treatment), and it keeps the β covariance from collapsing during the exo-atmospheric portion where β is unobservable (drag ≈ 0 ⇒ zero information about β; the random walk correctly re-inflates uncertainty until the vehicle hits sensible atmosphere).

**Alternatives considered:** ENU/ECEF-frame model (needs explicit Coriolis/centrifugal terms and breaks force-math sharing with the orbital model); 9D with estimated lift components (out of scope — 6DOF aero excluded); fixed known β (defeats the purpose — β is the primary unknown that distinguishes RVs from debris/boosters).

### Decision 5: Phased ballistic truth — new `thresh_synth::ballistic` generator in ECI; `SegmentType::Ballistic` is removed (BREAKING)

**Decision:** phased generation cannot live inside `trajectory.rs`'s segment machinery: segments step `Vector3` pos/vel in an **anchorless flat local frame** (no Earth center, no epoch), so "round rotating Earth" is unrepresentable there. Instead:

- New module `thresh-synth/src/ballistic.rs` with a `BallisticProfile` config: launch geodetic position, launch azimuth, boost parameters (`thrust_accel` m/s² constant magnitude, `burn_time_s`, `pitch_over_s`, `pitch_kick_rad`), `beta` (kg/m²), plus epoch. `generate(&BallisticProfile, dt) -> Vec<OrbitalState>` (synth's Cartesian sample type) integrates **in ECI** with the shared closure-based RK4 (Decision 1), switching acceleration by phase:
  - **Boost** (`t < burn_time_s`): gravity + thrust. Vertical rise until `pitch_over_s`, an instantaneous pitch kick of `pitch_kick_rad` downrange, then a **zero-lift gravity turn**: thrust aligned with the atmosphere-relative velocity `v_rel = v − ω_⊕ × r` (the textbook gravity-turn condition; aligning with inertial v was rejected because it makes the turn geometry epoch-dependent in ECI). Constant thrust *acceleration* — no mass depletion, no staging (out of scope), which is the fidelity floor that still produces a correct loft/range shape.
  - **Midcourse** (burnout → sensible atmosphere): exactly the **existing** J2 RK4 path — the phase's acceleration closure is `two_body + J2`, identical to what `propagate` with `include_j2: true, drag: None` evaluates. No new physics.
  - **Reentry** (altitude < 100 km descending): gravity + J2 + `drag_acceleration(r, v, 1/β)` — the same closure the filter model of Decision 4 integrates, guaranteeing truth/filter physics identity. Terminate at the WGS-84 ellipsoid (`‖r‖` vs geodetic surface radius is approximated spherically, consistent with the drag altitude convention).
  - Phase boundaries are time/altitude events evaluated per RK4 step (phase-helper decomposition: `boost_accel`, `phase_of`, `integrate_profile` each ≤ 15 complexity).
- **Output contract:** ECI samples plus a station-projection helper reusing `orbital_to_enu` / `eci_to_enu`, and an adapter emitting the `TrajectoryPoint` grid the existing radar generator consumes — the ballistic scenario (Decision 8) plugs into the benchmark pipeline exactly like the SGP4 orbital source does (ECI → ENU → radar RAE).
- **`SegmentType::Ballistic { drag_coefficient }` is deleted (BREAKING),** not re-fielded: any parameter set rich enough for phased flight cannot be honestly evaluated in the flat anchorless segment frame, and keeping a flat-Earth "ballistic" variant alongside a real generator invites silent misuse. Migration for the two known constructor sites — `trajectory.rs:289` (test) and `crates/thresh/tests/integration.rs:179,191` — is `SegmentType::Ca { acceleration: [0.0, 0.0, -9.81] }` where they only needed "falls under gravity", or the new `BallisticProfile` where they meant an actual missile. Grep-clean is the acceptance test.

**Alternatives considered:** extend the variant in place with phase fields (frame is wrong, see above); generate in ECEF (Coriolis/centrifugal terms must then be added to every phase's dynamics, and midcourse could no longer reuse the ECI J2 propagator verbatim — the single strongest reason to stay in ECI); keep the toy variant deprecated (rejected: pre-1.0, org-internal, and the proposal explicitly flags the break).

### Decision 6: Track heads gain explicit model dispatch; ballistic head → 7D reentry, new orbital head; IMM bank extension deferred

**Decision:** `TrackHead` currently *implies* a model that the tracker never builds (`predict_single_model` hardcodes `ConstantVelocity::new(5.0)`). Add an explicit selector:

```rust
pub enum HeadModel {
    Cv,                                      // existing behavior
    Reentry { sigma_beta: f64, beta_init: f64 },
    KeplerJ2 { max_step_s: f64 },
}
```

as a new `TrackHead.model: HeadModel` field (default `Cv` for every existing head — zero behavior change for aircraft/UAV/automotive), plus a `TrackHead::build_model(&self) -> Box<dyn MotionModel>` factory that combines `HeadModel` with `process_noise_sigma`. The tracker's predict path dispatches per-track through the head's model with an EKF-style covariance propagation (the `LinearModel` fast path stays for `Cv`).

- **`TrackHead::ballistic()` (BREAKING):** `state_dim: 9 → 7`, `model: HeadModel::Reentry { sigma_beta: …, beta_init: 1000.0 }`, `initial_covariance` becomes 7 entries — wide position/velocity priors as today plus a large β variance (β is a priori unknown within ~[50, 10000] kg/m²). Confirmation stays 2-of-3 / delete-3: reentry passes are short and the existing policy was already tuned for that.
- **New `TrackHead::orbital()`:** `TargetClass::Orbital` already exists (`thresh-core/src/track.rs`), but `HeadRegistry::default()` registers no head for it, so orbital-classified tracks silently fall back to the `Unknown` head today — no core enum change is needed, only the head. `state_dim: 6`, `model: HeadModel::KeplerJ2 { max_step_s: 10.0 }`, position priors of order (1 km)², velocity (100 m/s)² to match radar-initialized LEO births, confirmation 2-of-3 with a longer deletion window (5) because pass-edge dropouts are routine at 5 s sampling. Registered in `HeadRegistry::default()`.
- **IMM bank: extension deferred, mapping shipped.** The 4-model `cv_ca_ctrv_ct` bank stays untouched because (a) it is index-aligned with `MotionModeLabel` (`thresh-core/src/motion_mode.rs`) and (b) the `learned-imm` classifier path hard-requires exactly that bank (`tracker.rs:199-228`, `NUM_MODES` guard) — appending modes breaks a trained, checked-in ONNX contract for a speculative benefit. What ships instead: a `Reentry7Mapping: StateMapping` (drop/append β + covariance row-col, the `CaMapping` pattern) so users can compose e.g. a boost(CA)/reentry two-mode bank through the public `ImmConfig` today, and phase switching for the benchmark tracker is handled deterministically by the head (reentry model is valid exo-atmospherically too — drag term vanishes with ρ → 0, so one 7D model covers midcourse + reentry without mode logic; boost tracking is initialized late enough post-burnout in the scenario that no boost mode is needed). A dedicated boost/midcourse/reentry IMM preset is recorded as a follow-on candidate in the spec, not built here.

**Alternatives considered:** dispatch on `state_dim` implicitly (fragile — 6 is already CV *and* KeplerJ2); a `Box<dyn Fn() -> Box<dyn MotionModel>>` factory field on `TrackHead` (breaks `Clone + Debug + Serialize`-friendliness of the config record; the enum keeps heads data-only); extending `MotionModeLabel` and retraining the IMM classifier (out of scope, GPU work, separate change).

### Decision 7: Benchmark gate — calibrated absolute MOTA floors, not relative-improvement assertions

**Decision:** keep the gate shape (per-scenario `[baselines] mota = X` in the TOML, asserted by the existing runner) and change only the numbers and the tracker underneath:

- `run_orbital_benchmark` gains an orbital-head path: measurements are converted station-ENU → ECI position (inverting `eci_to_enu` with the known station geodetics + GMST epoch — both already available in the pipeline) and fed to a 6D `KeplerJ2` EKF track per Decision 6. This is the piece that makes a real accuracy floor honest: the current CV-in-ENU tracker needs `gate_threshold = 100000.0` and smoke baselines of `-2.0` / `-5.0` precisely because straight-line prediction across 35 km inter-scan arcs shreds association.
- **Calibration procedure (written into tasks, not guessed now):** the scenario noise is deterministically seeded, so a run produces one MOTA value per platform; run the gate scenarios on the three CI targets, take the minimum observed MOTA, and set the baseline to `min_observed − 0.15`. The 0.15 margin absorbs cross-platform float-drift-induced association flips (the only nondeterminism left), which historically move MOTA by ≪ 0.1 on seeded scenarios. The TOML comment must state the observed value and the margin so future regressions are interpretable. Expectation to verify at calibration time: ISS/Starlink floors land ≥ 0.5 (vs the −2.0/−5.0 smoke values); if a physically-correct tracker cannot clear 0.0 on these scenarios, that is a finding to document, not a number to hide.
- The runner-errors-on-breakage property (TLE parse, propagation failure) remains the smoke layer; the MOTA floor becomes a genuine accuracy layer on top. `gate_threshold` tightens alongside (Mahalanobis gating with a J2-consistent innovation no longer needs a 1e5 escape hatch).
- **New ballistic scenario TOML** (`crates/thresh-data/scenarios/ballistic-mrbm.toml`): one `BallisticProfile` truth (MRBM-class: ~300 s flight, ~1000 km range, β = 2000 kg/m²), a radar station downrange with visibility from upper midcourse through impact, 1 s sampling, seeded noise; source variant `[source.Ballistic]` wired to the Decision 5 generator through the same ENU/radar path. Gate: MOTA floor calibrated by the same procedure. Single target — MOTA on a one-target scenario is effectively a track-continuity/precision metric, which is what a dynamics-model gate should measure; multi-target ballistic stress belongs to association changes, not this one.

**Alternatives considered:** relative gate "orbital head beats CV head by ≥ Δ" (doubles CI runtime, and a diff of two noisy quantities flakes worse than either — the CV number would exist only to be subtracted); RMSE/MOTP-based gate (MOTP in this pipeline mixes measurement noise into the score and the repo's gate infrastructure is MOTA-keyed; MOTP can be *reported* without being gated); leaving the −2.0/−5.0 smoke values with a TODO (explicitly what the proposal exists to end).

### Decision 8: Golden-vector validation — three tiers, fixtures in `test-data/golden/orbital/`

**Decision:** validation is layered by what each source can actually certify, all running as plain `cargo test`:

1. **Element-conversion vectors (exact math):** Vallado worked examples (rv2coe / coe2rv, e.g. Example 2-5/2-6 4th ed.) and Stone Soup's element-conversion test vectors (MIT, ported as Rust constants with attribution comment) — hardcoded inline in `thresh-core` unit tests. Tolerances: 1e-6 relative on sma, 1e-9 rad on angles (matching the precision the sources print), consistent with the existing `keplerian_roundtrip` test tightness.
2. **Kepler/J2 propagation vs Vallado (analytic anchor):** Vallado's Kepler-propagation worked example as an inline test on `KeplerJ2` with J2 disabled (tolerance ≤ 10 m — the example's printed precision), plus the truth-identity test: `KeplerJ2::predict` over a 60 s LEO arc vs `thresh_synth::propagate` with matched step — same shared force math, same integrator, so tolerance 1e-6 m; this test is the enforcement mechanism for Decision 1's "identical physics" claim.
3. **SGP4 envelope vectors (physics sanity, not equality):** a fixture-generator binary in `thresh-data` (behind the existing `orbital` feature, run manually, output committed) takes AIAA-verified `sgp4`-crate propagation for 2–3 LEO objects, samples TEME position/velocity at t₀ and t₀ + {60, 300, 600} s, and writes JSON to `test-data/golden/orbital/`. The `cargo test` side initializes `KeplerJ2` from the t₀ state and asserts position error vs the SGP4 samples stays inside a documented envelope (≤ 5 km at 600 s for LEO) — generous because SGP4 carries drag and short-periodic terms J2-only physics legitimately lacks; the test certifies "right physics regime", the tiers above certify exactness. JSON fixtures + a `PROVENANCE.md` recording the TLEs, sgp4 crate version, and regeneration command.

Path precedent: tests already reach `test-data/` via `concat!(env!("CARGO_MANIFEST_DIR"), "/../../test-data/…")` (`tracker.rs:1292`). No Python anywhere; the generator is Rust and its output is committed, so CI never regenerates.

**Alternatives considered:** generating SGP4 vectors at test time (adds `sgp4` to default test deps and makes golden values float with crate upgrades — committed fixtures pin them); Python-generated fixtures via python-training tree (violates the no-Python-in-CI constraint for a task Rust already does with validated AIAA vectors); asserting against SGP4 tightly (category error — J2-osculating ≠ SGP4 mean-element theory; a tight tolerance would either fail honestly or pass by tuning, both bad).

### Decision 9: Reference doc — `docs/reference/orbital-ballistic-dynamics-reference.md`

**Decision:** one self-contained derivation doc in the established `docs/reference/` pattern (the transformer-fusion trio is the style precedent: notation table, derivations from first principles, explicit mapping to code symbols). Contents: two-body acceleration from the potential; J2 acceleration derived from the zonal potential (matching `j2_acceleration` term-for-term, with the `(5z²/r² − 1)` / `(… − 3)` asymmetry called out); the piecewise-exponential atmosphere and the table's provenance (Vallado Table 8-4 lineage); β = m/(C_d·A) drag dynamics and the co-rotating relative velocity; the analytic `∂a/∂r` for two-body+J2 (the Jacobian cross-check of Decision 3); the gravity-turn boost equations; RK4 discretization and sub-step error order; process-noise discretization for the WNA and β-random-walk blocks. Citations: Vallado (*Fundamentals of Astrodynamics*), Farrell/reentry-tracking literature for the β model — cited as sources of the standard results, with every equation re-derived so the doc stands alone per repo pattern.

## Risks / Trade-offs

- **[Numeric Jacobian conditioning near phase boundaries]** — density varies by orders of magnitude across an RK4 sub-step near 80–120 km, so finite differences can bracket a regime change. → Mitigation: `max_step_s = 1 s` default for the reentry model keeps sub-step density variation bounded; Jacobian unit tests include a 100 km-altitude state; the β floor prevents the worst conditioning.
- **[ECI tracking makes ENU→ECI conversion a new correctness surface in the benchmark]** — a wrong GMST sign converts cleanly and tracks terribly. → Mitigation: round-trip unit test `eci_to_enu ∘ enu_to_eci = id` at benchmark epochs; the calibrated MOTA floor itself catches systematic frame errors loudly.
- **[Gate calibration could institutionalize a mediocre number]** — if the first implementation underperforms, `min_observed − 0.15` bakes that in. → Mitigation: calibration task requires recording observed values in the TOML comments and comparing against the CV baseline informally before setting the floor; the floor is a regression gate, and raising it later is a one-line change.
- **[`SegmentType::Ballistic` removal may break unknown external users]** — pre-1.0, org-internal per the release-decision memory; the two in-repo constructor sites are enumerated. → Mitigation: BREAKING flag in proposal + migration note in the change; grep in CI review.
- **[Two `OrbitalState` types coexist]** — core's representation enum and synth's Cartesian sample. → Mitigation: `From` impls both ways, doc comments pointing at each other, consolidation explicitly assigned to `astro-time-and-frames`.
- **[β unobservability exo-atmospherically]** — β covariance grows without bound during long midcourse tracking. → Mitigation: that is the *correct* Bayesian behavior; the β random walk is sized so variance growth over a full midcourse stays within the initial prior's order of magnitude; documented in the head's tuning comment.
- **[Cognitive-complexity gate on the phased generator]** — phase-switching integrators are exactly the shape SonarCloud punishes. → Mitigation: phase-helper decomposition planned up front (Decision 5 names the helpers); `rk4_stage` is the in-repo worked example.

## Migration Plan

1. `thresh-core::orbital` lands first (force math + `OrbitalState` type + golden tier 1/2 tests) with `thresh-synth` shims in the same PR — workspace tests prove arithmetic identity before anything consumes the new module.
2. `thresh-filter` models (`KeplerJ2`, `BallisticReentry`, `numeric_jacobian`, `Reentry7Mapping`) — purely additive.
3. `thresh-tracker` head dispatch (`HeadModel`, breaking `ballistic()` change, new `orbital()` head) + `thresh-synth` phased generator with the `SegmentType::Ballistic` removal and the two test-site migrations.
4. `thresh-data`: ENU→ECI benchmark path, ballistic scenario, SGP4 fixture generator + committed fixtures; run calibration, set floors, tighten `orbital-iss.toml` / `orbital-starlink-train.toml` baselines last — the gate only tightens once the tracker it measures exists.
5. Reference doc alongside step 1–2 (the Jacobian cross-check test cites it).

Rollback: steps are independently revertible; the gate baselines are data-only and can be restored to the old smoke values (`-2.0` ISS, `-5.0` Starlink train) without code changes if the new tracker path must be reverted.

## Open Questions

- **Orbital head birth velocity:** a single radar position fixes 3 of 6 states; the current `birth_track` zero-velocity init is hopeless for a 7.5 km/s LEO target. Options: two-point differencing at confirmation, or circular-orbit velocity prior from the position. Decide during implementation of Decision 7's benchmark path; the head's velocity prior width depends on the answer.
- **Ballistic scenario visibility window:** whether the MRBM radar sees upper midcourse (β unobservable, tests covariance honesty) or only reentry (β observable throughout) — pick after first truth-generation runs show realistic pass geometry.
