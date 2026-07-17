# Tasks — Element-Space Orbital Filtering

> Phases follow design.md's Migration Plan: GVE math first, then the cross-formulation equivalence suite that validates it, then the filter model with its angular-handling proofs, then the Monte-Carlo demonstration with the standing benchmark-invariance verification. The falsifiable exit criterion is task 4.3.

## 1. GVE — `thresh-core/src/orbital/gve.rs` (design Decisions 1–2)

- [x] 1.1 Fetch the authoritative element-rate equations for the direct equinoctial set `(a, h, k, p, q, λ)` (DSST/Cefola lineage or Vallado's equinoctial VOP; cite what is actually fetched); resolve design Decision 1's fallback — if only Walker-set `(p, f, g, h, k, L)` equations are fetchable in usable form, implement internally in that set with exact boundary conversions — and RECORD the route taken in design.md's Open Questions.
- [x] 1.2 Implement the element rates as per-element phase helpers (complexity ≤ 15 each) taking the RSW perturbation components; the two-body secular term `dλ/dt ⊇ √(μ/a³)` analytic; RSW basis construction (R̂ = r̂, Ŵ ∝ r×v, Ŝ = Ŵ×R̂) with orthonormality/right-handedness test (spec: "RSW rotation is orthonormal and right-handed"); the perturbing-acceleration seam is a time-aware inertial closure rotated internally (design Decision 2 — J2 enters as `j2_acceleration` alone, never the two-body composite). Tests: zero perturbation moves only λ at exactly the mean motion (spec: "Unperturbed motion moves only the mean longitude"); fetched published spot values where available (spec: "Published spot values reproduced"); near-singular domain e = 1e-8, i = 1e-8 finite (spec: "Near-circular near-equatorial evaluation is finite") with singular denominators identified at the code.
- [x] 1.3 Element-space propagation: sub-stepped fixed RK4 over the element rates (`max_step_s` configurable, the KeplerJ2 pattern). Tests: bitwise-deterministic repeated propagation (spec: "Deterministic element propagation"); step-halving convergence at the integrator's order on a J2 LEO arc (spec: "Convergence under step halving").

## 2. Cross-formulation equivalence (design Decision 4)

- [x] 2.1 Equivalence suite: identical J2 orbits through GVE-in-elements and the existing Cartesian two-body+J2 path, converted and compared at the endpoints, both directions (element-born and Cartesian-born), three regimes — near-circular LEO, MEO, eccentric e ∈ [0.3, 0.7] through multiple perigee passages — over multi-revolution arcs; tolerances measured at the chosen steps and documented at each test (specs: "LEO equivalence both directions", "Eccentric-orbit equivalence through perigee"). No python generator (design Decision 4's deliberate record).

## 3. Filter model — `thresh-filter/src/models/equinoctial.rs` (design Decision 3)

- [x] 3.1 `MotionModel` impl over the 6D element state (state order = the stored `OrbitalElements::Equinoctial` field order), `mu` + `max_step_s` parameterized; UKF sigma-point prediction test: predicted mean matches direct propagation within sigma-point tolerance, covariance symmetric PD (spec: "Sigma-point prediction through the element dynamics").
- [x] 3.2 Unwrapped-λ convention (design Decision 3): documented on the model; wrap-straddle test — mean λ just below 2π·k with sigma spread crossing, predicted mean/covariance continuous and equal (to tolerance) to a 2π-shifted run (spec: "Wrap-straddling sigma points produce no artifact"); long-horizon test — λ accumulating hundreds of radians still matches the Cartesian path within the cross-formulation tolerance, with the f64 sin/cos precision note documented (spec: "Long-horizon accumulation stays accurate").
- [x] 3.3 Measurement mapping elements → inertial Cartesian position, exactly equal to the existing conversion (spec: "Measurement mapping matches the conversion").
- [x] 3.4 Resolve the element-space process-noise open question (design: RSW acceleration PSD mapped through the GVE input matrix vs tuned diagonal) with the demonstration-fairness requirement in mind; implement, and RECORD the decision in design.md.

## 4. Demonstration + invariance (design Decision 5)

- [x] 4.1 The Monte-Carlo coast-gap experiment in thresh-eval (dev-dep on thresh-filter already exists): seeded truth arcs (Cartesian two-body+J2), noisy ECI position measurements, warmup tracking, coast gap of duration T, ANEES at gap end over N runs — UKF-over-equinoctial vs EKF-over-KeplerJ2, both sharing the physical process-noise assumption from 3.4; sweep T; resolve the sweep-values open question for the cleanest separation and RECORD it in design.md.
- [x] 4.2 Assertions: at least one recorded gap duration where the Cartesian baseline's ANEES is outside the two-sided 95% band while the equinoctial filter's is inside, margins documented (spec: "Element filter outlasts the Cartesian filter through coast"); bitwise-identical reruns (spec: "Demonstration is deterministic"); full sweep values recorded at the test.
- [x] 4.3 Exit criterion (numeric, falsifiable): the cross-formulation equivalence suite passes at its documented tolerances in all three regimes and both directions; the wrap-straddle and long-horizon λ tests hold; and the demonstration sweep records the crossover — Cartesian ANEES out of band, equinoctial in band — at a stated gap duration.
- [x] 4.4 Benchmark invariance: the four calibrated scenarios digit-for-digit identical before vs after (spec: "Calibrated benchmarks are bitwise unchanged"); record the comparison in the PR description.

## 5. Wrap-up

- [x] 5.1 Full gates: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, rustdoc `-Dwarnings`, `openspec validate --all --strict --no-interactive`; complexity spot-check on the element-rate helpers and the sweep harness. Also `cargo test -p thresh-data --features adsb` (the feature-gated-manifest lesson from #145).
- [x] 5.2 Update proposal.md/design.md with implementation-time divergences and the resolutions of the three design Open Questions (GVE source/set route, element-space Q, sweep values).
