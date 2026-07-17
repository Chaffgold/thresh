# Element-Space Orbital Filtering

## Why

Cartesian ECI covariances go banana-shaped within a fraction of an orbit during coast: along-track uncertainty wraps around the curved trajectory, the Gaussian assumption fails, and the Cartesian `KeplerJ2` EKF's covariance turns dishonest exactly when coast prediction matters most (sensor gaps, beam revisit, handover). Equinoctial elements keep orbital uncertainty near-Gaussian over far longer arcs — the advanced-methods survey's space-accuracy centerpiece recommendation — and with the propagation tier complete, thresh can now *measure* that claim with its own NEES/ANEES consistency machinery instead of citing folklore: the payoff demonstration is a seeded Monte-Carlo coast-gap experiment where the element-space filter stays inside the two-sided χ² band at gap lengths where the Cartesian filter has left it.

## What Changes

- **thresh-core/orbital**: Gauss variational equations (GVE) for the direct/prograde equinoctial set (already `OrbitalState`'s representation — nonsingular at e = 0 and i = 0; the i = π retrograde singularity stays documented and out of scope) — element rates driven by a perturbing acceleration expressed in the RSW/LVLH frame, plus the two-body secular rate on the fast variable (mean longitude); J2 supplied by the existing shared force closures rotated into RSW; integrated on the existing fixed-step RK4 (with the DP adaptive integrator available through the time-aware seam).
- **thresh-filter**: an equinoctial motion model (`MotionModel` impl over the 6D element state) usable by the existing UKF/CKF sigma-point machinery, with **explicit angular handling of the mean longitude** — wrapping in sigma-point means and residuals, the classic element-filter failure mode, elevated to a spec requirement — and a measurement model mapping elements → Cartesian ECI → the existing station/ENU measurement space through the existing lazy conversions.
- **Validation by cross-formulation equivalence**: the same J2 orbit propagated by the existing Cartesian `KeplerJ2` path and by GVE-in-elements SHALL agree after conversion, in both directions, across LEO/MEO/eccentric regimes — a genuine independent check because the two formulations share only the force model; published GVE spot values fetched at implementation time where available.
- **The payoff demonstration**: seeded Monte-Carlo coast-gap ANEES bracket — element-space sigma-point filter vs Cartesian `KeplerJ2` EKF at growing gap durations, numbers recorded; the element filter SHALL remain χ²-consistent at gaps where the Cartesian filter is not.
- **Benchmarks untouched by default** (bitwise-invariance verification per the standing pattern); tracker-head consumption of the element filter is explicitly a later change.
- **Out of scope** (each recorded in design): GEqOE J2-absorbing elements, adaptive GMM splitting for coast gaps (rides on this change; its own follow-up), retrograde singularity handling, tracker/association changes, TLE mean-element filtering.

## Capabilities

### New Capabilities

- `gauss-variational-equations`: equinoctial element rates under perturbing RSW accelerations with the two-body mean-longitude secular term, wired to the shared force closures, integrator-agnostic, validated by cross-formulation equivalence against the Cartesian propagation path and fetched published spot values.
- `equinoctial-filter-model`: the element-space motion + measurement models for sigma-point filters — mean-longitude angular handling in sigma-point statistics, element→ENU measurement mapping, and the Monte-Carlo coast-gap consistency demonstration against the Cartesian baseline.

### Modified Capabilities

_None. `orbital-state-representation` (element conversions), `state-estimation` (UKF/CKF), and the consistency machinery are consumed as-is; if sigma-point angular handling requires a UKF/CKF API extension rather than a model-side treatment, the design records it and the spec delta is added then._

## Impact

- **Crates**: `thresh-core` (new `orbital/gve.rs` — pure math over the existing closure seam), `thresh-filter` (new equinoctial model alongside `KeplerJ2`; possibly small sigma-point-statistics hooks if the design requires them). No tracker, synth-default, benchmark-schema, or dependency changes.
- **Benchmarks**: bitwise-invariant (verification run required).
- **Downstream**: unblocks adaptive GMM splitting for coast gaps and the launch→orbit handoff work; gives any future SDA-flavored tracking an honest long-coast filter.
