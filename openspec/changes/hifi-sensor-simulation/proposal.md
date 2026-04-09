## Why

The current thresh-synth pipeline generates trajectories from simple kinematic models (CV, CA, CTRV, ballistic) and radar measurements from basic Gaussian noise. This is adequate for unit testing and algorithm development but insufficient for validating the tracker against realistic sensor physics. Real radar detection depends on target RCS (which varies with aspect angle and fluctuates statistically), the radar equation (SNR vs range), atmospheric effects, and clutter environments. Similarly, realistic aircraft and missile trajectories require aerodynamic constraints, not just kinematic primitives. Bridging this fidelity gap is essential before the tracker can be trusted on real-world data.

## What Changes

- Add Swerling I-IV RCS fluctuation models (pure Rust, no external dependency)
- Implement the full radar equation: SNR = (Pt G² λ² σ) / ((4π)³ R⁴ kTB) → P_d via Albersheim or Shnidman approximation
- Add RCS lookup tables: pre-computed RCS vs aspect angle for target archetypes (fighter, airliner, missile, UAV, satellite), loadable from JSON
- Integrate PyPOFacets or Open RCS (PyO3 bridge, feature-gated) for computing RCS tables from 3D geometry
- Integrate JSBSim (PyO3 bridge, feature-gated) for 6DOF aircraft/missile trajectory generation with realistic dynamics
- Integrate nyx-space (Rust native) or Orekit (PyO3 bridge) for high-fidelity orbital propagation replacing SGP4
- Add IR/EO sensor physics: blackbody target signature, atmospheric transmission (ITU-R P.676), sensor NETD → detection range
- Integrate RadarSimPy or mpar-sim (PyO3 bridge, feature-gated) for full radar scene simulation with beam scheduling

## Capabilities

### New Capabilities
- `swerling-rcs`: Swerling I-IV fluctuation models and RCS lookup tables with aspect-angle dependence
- `radar-equation`: Full radar equation with configurable transmitter parameters, Albersheim P_d approximation, atmospheric attenuation
- `rcs-computation`: PyO3 bridge to PyPOFacets or Open RCS for computing RCS tables from 3D target models (offline)
- `jsbsim-trajectories`: PyO3 bridge to JSBSim for 6DOF aircraft and missile trajectory generation with aerodynamic constraints
- `hifi-orbital`: nyx-space (Rust) or Orekit (PyO3) integration for high-fidelity orbital propagation with drag, SRP, gravitational harmonics
- `eoir-physics`: IR/EO sensor modeling with blackbody signatures, atmospheric transmission, and detection probability
- `radar-scene-sim`: PyO3 bridge to RadarSimPy or mpar-sim for full radar scene simulation with clutter and multi-path

### Modified Capabilities
- `synthetic-data`: Upgrade measurement generators to use radar equation + Swerling instead of fixed P_d + Gaussian noise

## Impact

- `thresh-synth` gains new modules for radar equation, Swerling, EO/IR physics (pure Rust, always available)
- New optional dependencies: `nyx-space` (Rust crate), PyO3 bridges to JSBSim, PyPOFacets/Open RCS, RadarSimPy
- Feature flags: `jsbsim`, `rcs-compute`, `hifi-orbital`, `radar-scene`
- RCS lookup tables stored as JSON data files in the repo (small, ~10 KB per target type)
- Existing synthetic scenarios remain valid; new high-fidelity scenarios layer on top
