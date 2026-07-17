# Design — Element-Space Orbital Filtering

## Context

`OrbitalState` stores the direct/prograde equinoctial set `(a, h = e·sin(ω+Ω), k = e·cos(ω+Ω), p = tan(i/2)·sin Ω, q = tan(i/2)·cos Ω, λ = M + ω + Ω)` with lazy conversions to Cartesian/Keplerian (nonsingular at e = 0, i = 0; documented singular at i = π). Filtering, however, happens exclusively in Cartesian: `KeplerJ2` propagates a 6D ECI state with sub-stepped RK4 inside `MotionModel::predict`, and its EKF covariance goes non-Gaussian (banana) within a fraction of an orbit of coast. The propagation tier supplies everything the element-space alternative needs: shared force closures (`j2_acceleration`), the time-aware seam and DP integrator, `Epoch`/frames, UKF/CKF sigma-point machinery, and — decisively — the NEES/ANEES consistency machinery that turns "elements stay Gaussian longer" from folklore into a measured, falsifiable claim.

Constraints: pure Rust, no new dependencies, deterministic CI, complexity ≤ 15, fetched-source provenance for published equations/values, calibrated benchmarks bitwise-invariant.

## Goals / Non-Goals

**Goals**
- GVE for the stored direct equinoctial set, driven by perturbing RSW accelerations from the shared force closures.
- An equinoctial `MotionModel` + measurement mapping usable by the existing UKF/CKF, with mean-longitude angular handling that provably survives sigma-point statistics.
- Cross-formulation equivalence validation (GVE-in-elements vs Cartesian `KeplerJ2`, both directions, LEO/MEO/eccentric).
- The seeded Monte-Carlo coast-gap ANEES bracket demonstrating element-space consistency outlasting Cartesian.

**Non-Goals**
- GEqOE (J2-absorbing generalized elements), adaptive GMM splitting (rides on this change; own follow-up), retrograde (i = π) handling, tracker-head consumption, TLE mean-element filtering, ENU/RAE measurement wiring beyond ECI positions (the demonstration's measurement space).

## Decisions

### Decision 1 — GVE in the stored set, equations transcribed from a fetched authority, with the Walker set as the recorded fallback

The GVE are implemented for the exact set `OrbitalState` stores (no additional conversion layer in the dynamics hot path). The element-rate equations for the direct equinoctial set MUST be transcribed from an authoritative source fetched at implementation time (Danielson et al.'s DSST report, Cefola-lineage papers, or Vallado's equinoctial VOP presentation — whatever is actually fetchable), cited inline, and never reproduced from memory. **Recorded fallback**: if only the *modified* equinoctial (Walker/Ireland/Owens `(p, f, g, h, k, L)`) equations prove fetchable in usable form, the dynamics may run internally in the Walker set with exact conversions at the model boundary — the conversion between the two sets is algebraic and cheap; whichever route is taken is recorded here at implementation time. Either way, the structural validation is the same: the cross-formulation equivalence check (Decision 4) catches transcription errors in whichever equations ship.

### Decision 2 — Perturbing-acceleration seam: inertial closure in, RSW rotation inside

GVE consume the *perturbing* acceleration only (two-body is the analytic secular term `dλ/dt ⊇ n(a) = √(μ/a³)`). The GVE evaluator takes a time-aware inertial perturbation closure `Fn(t, &r, &v) -> a_pert` (for this change: `j2_acceleration` alone — NOT the two-body+J2 composite) and rotates it into RSW internally (R̂ = r̂, Ŵ = r̂×v̂ normalized, Ŝ = Ŵ×R̂), evaluating r, v from the current elements via the existing conversions. This keeps the seam identical in kind to the force-stack seam (any perturbation the force config can express can later drive the GVE — SRP/third-body element filtering comes free when wanted), and makes "J2 only" an argument, not a hard-coding. Integration: sub-stepped fixed RK4 inside `predict` (the `KeplerJ2` pattern, `max_step_s` configurable) — the DP integrator remains available through the time-aware seam but the filter default matches the Cartesian sibling for a fair consistency comparison.

### Decision 3 — Mean longitude is unwrapped in filter state space; conversions own periodicity

The classic element-filter failure is angle averaging across the 2π wrap in sigma-point means and residuals. Rather than weighted circular means or UKF API changes, the filter-state convention is: **λ is continuous/unwrapped** — it grows monotonically under the secular rate and is never reduced mod 2π in state space; the element→Cartesian conversion applies periodicity internally (trigonometric functions are periodic anyway). Sigma points generated around an unwrapped mean stay contiguous (spreads are small relative to 2π), so plain arithmetic means remain correct, and position-space measurement updates never touch λ directly. A dedicated test constructs the near-wrap straddle case (mean λ just below a multiple of 2π, sigma spread crossing it) and proves the predicted mean and covariance are wrap-artifact-free; a long-horizon test proves nothing degrades as λ accumulates hundreds of radians (f64 precision note: at λ ~ 1e3 rad, sin/cos precision loss is ~1e-13 — documented, negligible for the demonstrated horizons). *Alternatives rejected*: circular statistics in the UKF (API change rippling into every other model for a problem only the fast variable has); wrapping λ with residual-aware subtraction (correct but scatters wrap logic across mean/residual/cross-covariance code — the historical bug farm).

### Decision 4 — Validation is cross-formulation equivalence plus fetched spot values; no python generator

The same physics is already in-tree in an independent formulation: Cartesian RK4 two-body+J2 (`KeplerJ2`/`propagate`). Equivalence tests propagate identical initial conditions through both formulations across LEO (near-circular), MEO, and an eccentric (e ≈ 0.3–0.7) regime for multi-revolution arcs, converting at the endpoints, in both directions (start-in-elements and start-in-Cartesian), with tolerances measured at the chosen step sizes and documented (truncation behavior differs between formulations — the tolerance is empirical, recorded, and tight enough to catch any term/sign error, which is the realistic failure mode). Published spot values (element rates at a documented state) are fetched and pinned where available. A python fixture generator adds nothing here — the two-formulation check is stronger than a third transcription of the same equations — recorded deliberately (the atmospheric change's precedent).

### Decision 5 — The demonstration lives in thresh-eval, reusing the task-2.4 Monte-Carlo pattern

thresh-eval already dev-depends on thresh-filter (the eval-consistency falsifiability bracket), so the coast-gap experiment lives beside it: seeded truth arcs (Cartesian two-body+J2), noisy ECI position measurements at fixed cadence, a warmup tracking arc, then a measurement-free coast gap of duration T, then ANEES-at-gap-end over N seeded runs against the truth — for the UKF-over-equinoctial model vs the EKF-over-`KeplerJ2` baseline, at a sweep of T. Success criterion (spec-level, falsifiable): there exists a recorded gap duration at which the Cartesian filter's ANEES exits the two-sided 95% χ² band while the element filter's remains inside; all sweep numbers recorded at the test. The measurement space is ECI positions (the orbital benchmark's post-conversion space) — ENU/RAE wiring is out of scope.

### Decision 6 — Placement and shape

`thresh-core/src/orbital/gve.rs`: pure element-rate math + RSW rotation + the integrator loop over elements (mirrors `integrate.rs`'s role). `thresh-filter/src/models/equinoctial.rs`: the `MotionModel` impl (state layout = the six stored elements in `OrbitalState` order), `max_step_s` + `mu` parameterized like `KeplerJ2`, plus the elements→ECI-position measurement helper. No new pub surface elsewhere; no UKF/CKF changes anticipated (Decision 3 avoids them) — if implementation contradicts that, the spec delta for `state-estimation` is added at that point (recorded in the proposal).

## Risks / Trade-offs

- **[Equation transcription]** GVE terms are long and sign-sensitive → fetched-source rule + the cross-formulation equivalence check, which shares no code with the GVE path beyond the J2 closure.
- **[Near-singular geometries]** e → 0 and i → 0 are exactly where equinoctial elements shine, but denominators like `1 + √(1−h²−k²)` and `1 + p² + q²` must be checked for the regimes tested; the eccentric regime probes the other end. Explicit domain notes on every helper.
- **[Fast-variable stiffness]** λ moves ~2π per revolution while the slow elements barely move → the sub-step ceiling is set by λ accuracy; the equivalence tests at eccentric perigee are the stress case; step-halving convergence test pins the order.
- **[Demonstration flakiness]** Monte-Carlo ANEES near band edges → seeded runs, N sized so the recorded margins are decisive (the eval-consistency 2.4 pattern), bitwise-deterministic reruns.
- **[Complexity gate]** the GVE right-hand side is term-heavy → per-element phase helpers with per-helper tests, table-driven where natural.

## Migration Plan

1. `gve.rs`: element rates + RSW rotation + integration loop, per-helper unit tests + fetched spot values — additive.
2. Cross-formulation equivalence suite (three regimes, both directions, measured tolerances) — additive.
3. `equinoctial.rs` filter model: `MotionModel` impl + measurement helper + wrap/straddle and long-horizon λ tests — additive.
4. The thresh-eval Monte-Carlo coast-gap demonstration + recorded sweep; benchmark-invariance verification run.
5. Wrap-up gates; divergence/open-question records.

Every step additive and independently revertible; nothing touches existing models, trackers, or benchmark defaults.

## Open Questions

- **Which authority's GVE form ships** (direct-set from DSST/Cefola lineage vs Walker-set internally with boundary conversions) — resolved at implementation by what is actually fetchable; recorded per Decision 1.
- **Process noise in element space**: a diagonal Q on elements is not physically meaningful the way acceleration-driven Q is in Cartesian; options are Van-Loan-style Q from an RSW acceleration PSD mapped through the GVE input matrix, or a small tuned diagonal for the demonstration. Leaning: map an RSW acceleration PSD through the GVE B-matrix (honest, and reuses the demonstration's fairness argument — both filters then share the same physical Q assumption); decide at implementation with the fairness requirement in mind and record.
- **Demonstration gap sweep values** (LEO minutes vs fractions of a period) — pick at implementation for the cleanest recorded separation.
