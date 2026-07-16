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

- **Third-body formulation**: direct minus indirect term (Battin/Vallado standard) is settled math, but whether to use the numerically-stabilized Battin F(q) form to avoid small-difference cancellation at LEO — decide at implementation with a cancellation test at LEO radius.
- **Synth exposure**: does the high-fidelity option get a scenario-TOML knob now, or stay API-only until a benchmark scenario actually wants it? Leaning API-only (no TOML schema churn before a consumer exists).
- **SRP solar-pressure constant**: fixed 1361 W/m² AU-scaled vs the fixture epochs' actual distance-scaled irradiance — pick whichever the independent python stack can mirror exactly, document the choice.
