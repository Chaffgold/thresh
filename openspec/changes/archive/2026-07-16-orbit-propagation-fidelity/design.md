# Design — Orbit Propagation Fidelity

## Context

The shared force math in `thresh-core/src/orbital/` is two-body + J2 (`gravity.rs`), a 12-band exponential atmosphere (`atmosphere.rs`), and fixed-step RK4 (`integrate.rs`) — 530 lines consumed via position/velocity closures by the `KeplerJ2`/`BallisticReentry` filter models and the thresh-synth truth generators. `astro-time-and-frames` (#139) landed typed `Epoch`s, GCRF/TEME/ITRF transforms, and the `FrameProvider` seam. The live `hifi-orbital` spec predates the AGPL rejection of nyx-space and mandates it (plus an Orekit PyO3 fallback) — it is unimplementable as written and is superseded by this change.

Constraints: pure Rust, no new dependencies (coefficient tables embedded as const data), no AGPL, deterministic CI with tier-3 golden fixtures (no Python/network, default features), SonarCloud complexity ≤ 15, calibrated benchmark floors must not move.

## Goals / Non-Goals

**Goals**
- A configurable force stack above J2: J3/J4 zonals, truncated EGM96 harmonics, Harris-Priester density, cannonball SRP with eclipse, lunisolar third-body — each force toggleable and independently validated.
- Analytic Sun/Moon ephemerides good enough for third-body/SRP/diurnal-bulge use (arcminute-class), with measured accuracy against astropy.
- An adaptive Dormand–Prince RK5(4) integrator with PI step control and grid-exact sampling.
- GCRF/Epoch discipline: time-aware acceleration closures, Earth-fixed force legs through the `FrameProvider`.
- Retire `hifi-orbital` with per-requirement migration notes.

**Non-Goals**
- NRLMSISE-00/JB2008 (needs space-weather ingestion), maneuvers, ephemeris-file providers (ANISE later), filter-model consumption of the new forces (follow-on change), Gauss–Jackson/symplectic integrators, conical shadow (cylindrical only, documented), EGM96 beyond the embedded subset.

## Decisions

### Decision 1 — Time-aware closure seam alongside the existing one, not replacing it

The existing integrator seam is `Fn(&Vector3, &Vector3) -> Vector3` — no time argument, which is exactly why the current stack stops at J2 + non-rotating drag. The new machinery adds a time-aware seam: accelerations are `Fn(f64 /* seconds past the arc's epoch */, &Vector3, &Vector3) -> Vector3`, with the arc's `Epoch` bound at closure construction. The fixed-step RK4 and every existing consumer keep the old seam untouched (filter models and calibrated benchmarks depend on it — zero-baseline-movement is a hard constraint); the adaptive integrator and the high-fidelity force stack use the new one. *Alternative* — retrofitting time into the existing seam — breaks `KeplerJ2`/`BallisticReentry`/synth call sites for no benefit until the filter-consumption change actually wants it.

### Decision 2 — `ForceModelConfig` builder composing into one closure

```rust
pub struct ForceModelConfig {
    pub gravity: GravityFidelity,        // TwoBody | J2 | Zonal { j3, j4 } | Harmonics { degree, order }
    pub drag: Option<DragConfig>,        // HarrisPriester { inv_beta } | Exponential { inv_beta }
    pub srp: Option<SrpConfig>,          // cannonball: cr_area_over_mass, cylindrical eclipse
    pub third_body: ThirdBodyConfig,     // sun: bool, moon: bool
}
```
`build(epoch: Epoch, provider: &impl FrameProvider) -> impl Fn(f64, &Vector3, &Vector3) -> Vector3` sums the enabled terms. Each force is its own module-level function (phase-helper style, individually testable); the builder is a linear composition. Earth-fixed legs (harmonics, Harris-Priester co-rotation) rotate GCRF→ITRF at `epoch + t` through the provider and rotate the acceleration back. *Alternative* — a `Force` trait object vec — dynamic dispatch buys nothing here and complicates the closure seam the filters may later consume.

### Decision 3 — EGM96 truncated to 12×12, normalized Legendre recursion, cross-checked against the analytic zonals

Normalized C̄/S̄ coefficients to degree/order 12 embedded as a const table (~90 pairs, cited to the NASA EGM96 distribution — fetched at implementation time, never transcribed from memory), evaluated with the standard normalized-associated-Legendre recursion (Montenbruck & Gill Alg. 3.2 lineage) in the ITRF frame. J2/J3/J4 also exist as closed-form zonal accelerations (cheap tier for consumers that want just-above-J2 without the table). Mandatory consistency tests: harmonics-with-only-C̄₂₀ agrees with the existing analytic `j2_acceleration` to relative 1e-12 (same J2, two code paths), and zonal-only harmonics agree with the closed-form J3/J4. Degree 12 over 8: the marginal cost is table size only (the recursion is the same code), and 12×12 comfortably covers the sub-J2 signal for LEO truth arcs; going higher is a data edit, not a design change.

### Decision 4 — Harris-Priester with the standard 100–1000 km table, bulge exponent n = 2 fixed

The classic H-P min/max density table (Montenbruck & Gill Table 3.1 lineage, fetched at implementation time) with the diurnal-bulge interpolation `ρ = ρ_min + (ρ_max − ρ_min)·cos^n(ψ/2)`, apex 30° east of the subsolar point, n = 2 fixed (the inclination-dependent 2–6 refinement is deferred — documented). Sun direction from Decision 5's ephemeris. Below 100 km / above 1000 km: clamp-with-doc (the existing exponential table remains the reentry path's model — `BallisticReentry` is untouched). Deterministic by construction (no space-weather inputs).

### Decision 5 — Meeus/Vallado low-precision analytic Sun and Moon, accuracy measured not assumed

Sun: Meeus Ch. 25 low-precision series (~0.01° ≈ 36″); Moon: the truncated ELP-lineage series Vallado presents (~0.3° class). Both produce ecliptic-of-date longitudes → MOD via mean obliquity → GCRF via the frames module's precession. That accuracy is orders beyond what third-body tides (~1e-6 of total acceleration), SRP direction, and the H-P bulge need — but the golden generator measures the actual Sun/Moon position error against astropy at the fixture epochs and records it, so the claim is evidence, not folklore. *Alternative* — ANISE/DE ephemerides — file-backed, deferred to the provider seam by explicit non-goal.

### Decision 6 — Dormand–Prince RK5(4) with PI control and grid-exact sampling via step clamping

The 7-stage FSAL DP pair with standard PI step-size control (safety 0.9, standard exponents, min/max step-scale clamps, configurable abs/rel tolerances defaulting 1e-9/1e-9). Output sampling is **grid-exact by step clamping**: when the next natural step would cross a requested output time, the step lands on it exactly — interpolant-free, deterministic, and exact at every sample. *Alternative* — the dopri5 dense-output interpolant — more coefficients to transcribe and test for ~zero benefit at truth-generation grid densities; recorded as the upgrade path if profiling ever shows clamping-induced step churn. Eclipse boundary discontinuity (Decision 2's cylindrical shadow is a step function) is handled by the controller's natural step rejection; a doc note records the conical-penumbra upgrade path.

### Decision 7 — Golden validation: independent python force stack + scipy DOP853, plus fetched published spot values

`test-data/golden/propagation/` fixtures from the established manual generator pattern (PEP-723 pinned, PROVENANCE.md): (a) **component goldens** — Sun/Moon positions vs astropy at fixture epochs; EGM96 acceleration spot values vs an independently-implemented python evaluation of the same fetched coefficient file; H-P density at published table nodes; (b) **trajectory goldens** — 3 arcs (LEO ~400 km full-force, MEO harmonics+lunisolar, HEO SRP-dominant with eclipse crossings) integrated by an *independently written* python force stack (astropy ephemerides, own harmonics from the fetched EGM96 file) under scipy DOP853 at tight tolerance; envelope tolerances = measured cross-implementation delta × margin, every number's provenance recorded. Orekit was considered as the authority and rejected for the generator toolchain (JVM/conda dependency breaks the uv-run reproducibility pattern); published Vallado/Montenbruck spot values are fetched and pinned where available. This validates transcription and composition (the realistic failure mode), while model-choice validity rests on the cited published models themselves.

### Decision 8 — `hifi-orbital` retirement semantics

The delta spec REMOVEs all six requirements with per-requirement Reason/Migration: nyx-space mandate and Orekit fallback → unimplementable under the AGPL rejection, superseded by the pure-Rust stack; configurable force models → `orbital-force-models`; higher-accuracy-than-SGP4 → the trajectory-golden requirement in `adaptive-orbit-propagation` (accuracy demonstrated against independent integration rather than a propagator-vs-propagator race); covariance propagation → the closure seam remains numeric-Jacobian-compatible (the existing filter pattern), full high-fidelity filter models being the follow-on change; maneuvers → deferred until a mission-planning consumer exists. At archive time the sync deletes `openspec/specs/hifi-orbital/`.

## Risks / Trade-offs

- **[Coefficient transcription]** EGM96 table, H-P table, DP tableau, Meeus series → every table fetched from an authoritative source at implementation time with the source cited inline; checksum tripwire tests (the nutation-table pattern from #139); component goldens catch value errors independently of trajectories.
- **[Normalization confusion]** normalized vs unnormalized harmonics is the classic silent-error → the C̄₂₀-vs-analytic-J2 identity test makes the normalization contract executable.
- **[Eclipse step churn]** shadow discontinuity can make the controller reject steps around crossings → bounded by min-step clamp; HEO golden arc deliberately crosses shadow so the behavior is measured, not hypothetical.
- **[Per-step provider cost]** GCRF↔ITRF per force evaluation → provider rotations are closed-form (no allocation); if profiling shows pain, a per-step rotation cache slots into the closure without API change.
- **[Complexity gate]** recursions and series are loop-heavy → table-driven loops + per-force phase helpers, the pattern SonarCloud already accepts for the nutation series.

## Migration Plan

1. Ephemerides module (Sun/Moon + tests vs fetched spot values) — additive.
2. Force models one at a time (zonals → harmonics → H-P → SRP/eclipse → third-body), each with component tests; consistency identities (C̄₂₀ ≡ J2) in the same commits — additive.
3. Adaptive DP integrator + PI controller + grid clamping, validated on problems with known solutions (Kepler orbit energy/period, exponential decay) — additive.
4. `ForceModelConfig` composition + the thresh-synth high-fidelity option + trajectory goldens + envelope tests; benchmark-invariance verification run (all four calibrated scenarios bitwise unchanged).
5. Wrap-up gates; `hifi-orbital` retirement happens via the delta spec at archive time (no code step).

Every step additive and independently revertible; nothing edits existing force functions, filter models, or benchmark defaults.

## Open Questions

- **Third-body formulation** — RESOLVED (task 2.5): `third_body_acceleration` uses Battin's cancellation-free form `a = −GM_b/‖r−s‖³·(r + F(q)·s)` with `F(q) = q(3+3q+q²)/(1+(1+q)^{3/2})`, `q = r·(r−2s)/(s·s)` — algebraically identical to the direct-minus-indirect standard (the fixture generator's convention), transcribed from Battin (1999, rev. ed., pp. 448–450) via the statement in Proietti & Pontani, arXiv:2104.01240 Eq. (15) (fetched at implementation time) and pinned by executable equivalence tests. The decision test (`stabilized_form_beats_naive_cancellation_at_leo`) compares both forms against a colinear-geometry closed form that is cancellation-free by construction, at LEO radii against the Sun: measured worst relative error 4.0e-16 (stabilized) vs 1.6e-12 (naive) — the naive form's ~4-digit cancellation loss is real but not catastrophic; F(q) was adopted because it removes the loss at zero cost. At GEO the two forms agree to ≤ 1e-12 relative (also asserted).
- **Synth exposure** — RESOLVED (task 4.1): **API-only**. The high-fidelity truth option is `thresh_synth::orbital::propagate_high_fidelity` + `HighFidelityConfig` (a `ForceModelConfig` force stack paired with the `DpConfig` integrator settings), mirroring the existing `propagate` seam's sample-vector shape (initial state first, grid-cadence samples, final sample exactly at the arc end). No scenario-TOML knob is added until a benchmark scenario actually wants one — no TOML schema churn before a consumer exists — so every calibrated scenario keeps consuming the default RK4/J2 truth path unchanged (re-verified digit-for-digit in task 4.4). The option documents its convention deltas at the API: states are read as GCRF (not the default path's TEME-consistent "ECI") and Earth-fixed legs rotate through the zero-EOP `Iau76Fk5Provider`; callers needing explicit EOP drive `ForceModelConfig::build`/`propagate_on_grid` directly.
- **SRP solar-pressure constant** — RESOLVED (task 4.2, golden generator): fixed nominal pressure at 1 AU scaled by the inverse-square *actual Sun→spacecraft distance*: `P(d) = (S₀/c)·(AU/d)²` with S₀ = 1361 W/m² (IAU 2015 Resolution B3 nominal total solar irradiance), c = 299 792 458 m/s (exact), AU = 149 597 870 700 m (IAU 2012 B1). Both stacks compute `P₁AU = 1361.0 / 299792458.0` identically in IEEE-754 (≈ 4.53980733564685e-6 N/m²); the convention and constants are recorded in every SRP-bearing fixture's `force_config` block and in `test-data/golden/propagation/PROVENANCE.md`.

## Implementation-Time Divergences (task 5.3)

- **Single constant-set discipline for gravity tiers.** `GravityFidelity`'s non-harmonic tiers (TwoBody/J2/Zonal) draw their constants from `egm96_gravity_model()` (EGM96 μ/radius/J2) rather than a caller-supplied `GravityModel`, so every identity test and every fixture comparison shares one constant set — the C̄₂₀ ≡ J2 contract is meaningless across mixed constants. The existing `EARTH_WGS84`-based functions are untouched; the baseline-config-matches-existing-math test is bitwise under the EGM96 set.
- **Checksum tripwires vary by table representation**: the ephemeris series use scaled-integer column sums (their sources' fixed-decimal coefficients are not representation-exact as raw f64 sums); the Harris-Priester table pins exact-f64 checksums (its fetched values are exactly representable); the DP tableau uses sequential-sum checksums plus structural order-condition tests and a cross-source `E = b − b̂` identity. All serve the nutation-table pattern's tripwire property.
- **Min-step behavior on discontinuities**: the DP controller force-accepts steps at or below `min_step_s` regardless of the error estimate (implements the design's "bounded by min-step clamp" risk note for the shadow discontinuity); documented on `DpConfig::min_step_s`.
- **Golden tolerance methodology strengthened**: per-arc tolerance = 3 × (integrator delta [1e-12 vs 1e-10 re-integration] + measured ephemeris-swap delta [ERFA → the Rust stack's analytic series] + bulge-apex convention delta) + 1 m floor — the ephemeris term dominates (e.g. 10.8 m on the MEO arc vs 3.4e-7 m integrator), so the design's suggested pure re-integration delta would have been unpassable by construction.
- **Fixture-recorded shared constants asserted within 1 ulp in envelope tests** (not bitwise): serde_json's default float parsing is not correctly-rounded for all 17-digit decimals (measured); the true bit-level identities (P₁AU, GM values, EGM96 constants) stay pinned against rustc-parsed literals in unit tests instead. Enabling serde_json's `float_roundtrip` workspace-wide was rejected — it would alter default JSON paths elsewhere.
- **Ephemeris sourcing**: Meeus Ch. 25 coefficients entered via the book-faithful soniakeys/meeus Go transcription (the book text itself is not fetchable), cited as such; Vallado's Moon uses the frames module's obl80 obliquity (more digits, same lineage) and TT where Vallado writes TDB (|TT−TDB| < 2 ms, far below series class) — both documented in module docs.
