# hifi-orbital Specification (Delta)

## REMOVED Requirements

### Requirement: Numerical propagation with nyx-space
**Reason**: nyx-space is AGPL-3.0 and rejected by standing project decision; the requirement is unimplementable as written.
**Migration**: Pure-Rust adaptive Dormand–Prince propagation over the in-tree force stack — see `adaptive-orbit-propagation` ("Dormand-Prince adaptive integration", "GCRF and Epoch-disciplined propagation entry points").

### Requirement: Configurable force models
**Reason**: Superseded — the intent carries forward without the nyx-space backend.
**Migration**: `orbital-force-models` ("Configurable force-model stack" and its per-force requirements: truncated EGM96 harmonics, Harris-Priester density, SRP with eclipse, lunisolar third-body).

### Requirement: Higher accuracy than SGP4
**Reason**: Superseded — the propagator-vs-propagator race against SGP4 presumed the nyx-space/precise-ephemeris tooling; accuracy is now demonstrated against independent integration instead.
**Migration**: `adaptive-orbit-propagation` ("Trajectory golden validation", including the "Higher fidelity than the J2 baseline" scenario, with measured tolerances recorded in fixture provenance).

### Requirement: Maneuver modeling
**Reason**: Deferred — no mission-planning consumer exists in thresh; carrying an untested maneuver surface violates the falsifiability bar the benchmark gates set.
**Migration**: None yet; reopen as its own change when a consumer materializes (the time-aware closure seam of `orbital-force-models` is where thrust terms would compose).

### Requirement: Covariance propagation for uncertainty quantification
**Reason**: Superseded — filter-side covariance propagation already exists via the numeric-Jacobian EKF pattern over shared force closures; a nyx-space-specific STM surface is unnecessary.
**Migration**: The time-aware closure seam remains numeric-Jacobian-compatible (`orbital-force-models`, "Configurable force-model stack"); high-fidelity filter-model consumption is an explicit follow-on change.

### Requirement: Orekit fallback via PyO3
**Reason**: Retired — a JVM-backed PyO3 bridge contradicts the Rust-native preference and the deterministic-CI constraint; with nyx-space gone there is no "primary" for it to back stop.
**Migration**: None; independent cross-validation lives in the golden-fixture generators (manual-only python tooling), not in a runtime fallback.
