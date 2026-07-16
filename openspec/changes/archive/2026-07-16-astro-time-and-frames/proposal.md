# Astrodynamic Time Scales and Reference Frames

## Why

thresh's space-domain accuracy now hinges on two silently wrong foundations: time is a bare `f64` with no scale (`thresh_core::time::Timestamp(pub f64)`, `OrbitalState::epoch_jd: f64` — UTC? TT? nobody can say, and leap seconds are unrepresentable), and `eci.rs` rotates by GMST alone while conflating J2000 with TEME — the frame SGP4 actually outputs. TEME↔J2000 differs by up to ~0.8° of precession/nutation (kilometres at LEO radius), an error class the `OrbitalFrame` tag machinery landed by `orbital-ballistic-filter-models` can *name* but not *fix*: the tag exists, the transform doesn't. Every queued space change (`orbit-propagation-fidelity`, `element-space-orbital-filtering`, initial orbit determination) is explicitly blocked on this discipline existing.

## What Changes

- **thresh-core**: a time-scale-aware `Epoch` type backed by the `hifitime` crate (UTC/TAI/GPS/TT, leap-second correct, serde-compatible), replacing `epoch_jd: f64` on `OrbitalState` (**BREAKING**) and offered alongside the retained `Timestamp` relative-seconds type with explicit conversion between them.
- **thresh-core**: IAU-76/FK5 reference-frame transforms — TEME↔GCRF(≈J2000)↔ITRF via precession, nutation, GAST/GMST sidereal rotation (polar motion optional, defaulting to off) — pure Rust, no new heavy dependencies (nyx-space remains AGPL-rejected; ANISE is a possible later feature-gated provider, not a dependency of this change).
- **thresh-core**: every frame transform rotates covariance alongside state — position and velocity blocks through the full 6×6 chain, including the Earth-rotation rate term in inertial↔ITRF velocity transforms — so a track handed across frames keeps an honest P.
- **thresh-core**: a `FrameProvider` trait seam sized so a later ANISE-backed (cislunar) provider can plug in without breaking the API; the IAU-76/FK5 implementation is the default provider.
- **`OrbitalFrame` enforcement hardens**: the existing `FrameMismatch` error variant (currently only debug-asserted) becomes the enforced behavior for mixed-frame operations, and the tag vocabulary extends to the frames this change makes real (TEME, GCRF, ITRF).
- **thresh-data / thresh-synth**: SGP4 ingest output is tagged TEME (today it is implicitly treated as generic ECI); the benchmark ENU↔ECI path and synth propagators migrate to typed epochs. Calibrated benchmark baselines are expected to move only if a real frame error is being corrected — any shift is measured and documented, never silently re-floored.
- **Validation**: golden transformation vectors (Vallado worked examples + astropy/Skyfield-generated fixtures) committed as tier-3 Rust fixtures with provenance, exactly like `test-data/golden/orbital/` — no Python, no network, default features in CI.

## Capabilities

### New Capabilities

- `astro-time`: time-scale-aware epochs (UTC/TAI/GPS/TT) with leap-second correctness, conversions to/from Julian dates and the relative `Timestamp` seconds used by trackers, and serde round-tripping — the input contract every frame transform and propagator epoch consumes.
- `reference-frame-transforms`: TEME↔GCRF↔ITRF transforms via IAU-76/FK5 with full 6×6 covariance rotation (including the ω⊕ rate term), a `FrameProvider` trait seam for future ephemeris-backed providers, golden-vector validation, and integration of the existing ENU chain so ground-station projections compose with the new frames.

### Modified Capabilities

- `orbital-state-representation`: the "Frame-disciplined orbital state" requirement strengthens — mixed-frame operations SHALL fail via the enforced `FrameMismatch` path (not debug-assert), explicit conversions between tagged frames SHALL exist (they currently do not), and the state's epoch becomes a time-scale-aware `Epoch` rather than a bare Julian-date float.

## Impact

- **Crates**: `thresh-core` gains `hifitime` (MPL-2.0 — weak file-level copyleft, acceptable as an unmodified dependency, unlike the AGPL nyx-space; pure Rust, no transitive heaviness) plus a new `frames` module; `thresh-synth`, `thresh-data`, `thresh-tracker` migrate `epoch_jd`/implicit-frame call sites (**BREAKING** internally; pre-1.0, org-internal, no external consumers).
- **Deliberately out of scope**: polar-motion EOP ingestion (needs an EOP data source — recorded as the ITRF accuracy limit), IAU-2006/2000A CIO-based transforms (IAU-76/FK5 is the SGP4-consistent choice), cislunar frames and ANISE integration (the provider trait is sized for them; implementing them is not this change), superseding the stale `hifi-orbital` spec's nyx-space mandate (that spec is `orbit-propagation-fidelity`'s to retire — noted here because its "ECI J2000" language inherits this change's frame vocabulary when that happens), and any propagator force-model work.
- **Downstream unblocked**: `orbit-propagation-fidelity` (needs typed epochs + GCRF), `element-space-orbital-filtering`, IOD (both meaningless without frame/time discipline per the advanced-methods integration analysis).
