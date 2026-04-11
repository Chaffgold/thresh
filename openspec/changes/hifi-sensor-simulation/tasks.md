## 1. Swerling RCS Models (Pure Rust)

- [ ] 1.1 Implement Swerling I model: slow fluctuation, exponential RCS distribution (chi-squared 2 DOF), constant across pulses within a dwell
- [ ] 1.2 Implement Swerling II model: fast fluctuation, exponential RCS distribution, independent pulse-to-pulse
- [ ] 1.3 Implement Swerling III model: slow fluctuation, chi-squared 4 DOF (dominant scatterer + background), constant within dwell
- [ ] 1.4 Implement Swerling IV model: fast fluctuation, chi-squared 4 DOF, independent pulse-to-pulse
- [ ] 1.5 Implement Swerling case 0 (non-fluctuating, deterministic RCS) as baseline
- [ ] 1.6 Define `RcsProfile` struct: mean RCS (dBsm), Swerling type, aspect-angle table (optional)
- [ ] 1.7 Implement RCS lookup table: load from JSON, interpolate between azimuth/elevation entries
- [ ] 1.8 Create default RCS tables: fighter (~1 m², nose/tail/beam variation), airliner (~100 m²), cruise missile (~0.01-0.1 m²), UAV (~0.01-1 m²), satellite (~1-10 m²)
- [ ] 1.9 Write tests: Swerling I mean matches configured sigma, variance matches chi-squared 2 DOF over 10K samples
- [ ] 1.10 Write tests: RCS lookup interpolation returns correct values at and between table entries

## 2. Radar Equation (Pure Rust)

- [ ] 2.1 Define `RadarParameters` struct: Pt, G, lambda, bandwidth, noise_figure, system_losses, antenna pattern
- [ ] 2.2 Implement monostatic radar equation: SNR = (Pt * G² * λ² * σ) / ((4π)³ * R⁴ * k * T_sys * B * L)
- [ ] 2.3 Implement system noise temperature: T_sys = T_antenna + T_receiver, T_receiver = (NF - 1) * T_0
- [ ] 2.4 Implement Albersheim's approximation: P_d from SNR and P_fa (single pulse)
- [ ] 2.5 Implement Shnidman's equation for P_d with N integrated pulses and Swerling fluctuation
- [ ] 2.6 Implement atmospheric attenuation: ITU-R P.676 specific attenuation (dB/km) vs frequency and elevation angle
- [ ] 2.7 Implement range-dependent measurement noise: sigma_range, sigma_angle proportional to 1/sqrt(SNR)
- [ ] 2.8 Integrate radar equation into measurement generator: replace fixed P_d with computed P_d(R, σ, atmospheric_loss)
- [ ] 2.9 Create preset radar configs: X-band surveillance (AN/TPS-80), S-band search (AN/SPY-1), C-band tracking
- [ ] 2.10 Write tests: P_d decreases monotonically with range for fixed RCS
- [ ] 2.11 Write tests: Albersheim matches published P_d tables within 0.5 dB
- [ ] 2.12 Write tests: atmospheric attenuation increases with frequency

## 3. RCS Computation Bridge (Feature-gated)

- [x] 3.1 `RcsComputeBridge` in `crates/thresh-synth/src/rcs_compute.rs` under `#[cfg(feature = "rcs-compute")]` wraps PyPOFacets via pyo3 0.24.
- [x] 3.2 `RcsComputeBridge::load_geometry()` calls `pofacets.load_stl()` and returns the facet count.
- [x] 3.3 `RcsComputeBridge::sweep_azimuth()` produces a full azimuth sweep at a fixed elevation.
- [x] 3.4 `RcsComputeBridge::sweep_hemisphere()` produces an azimuth × elevation grid.
- [x] 3.5 `RcsSweepResult::{to_json, write_json}` and `compute_and_save_rcs()` export sweep data as JSON compatible with `RcsLookupTable` (via the non-gated `sweep_to_lookup_table` helper).
- [x] 3.6 `thresh-rcs-compute` binary at `crates/thresh-synth/src/bin/rcs_compute.rs` exposes `--stl / --freq / --step / --output` (plus `--az-start / --az-end / --el / --polarization`). Gated by `required-features = ["rcs-compute"]`. Arg parser lives in `rcs_compute::cli` with 11 unit tests covering help, required-flag errors, polarization validation, az range checks, sample counting, and hemisphere configs — all runnable without a Python runtime.
- [x] 3.7 `sphere_rcs_matches_analytical` test in `rcs_compute.rs` checks the σ = πr² result against PyPOFacets output within 2 dB. Marked `#[ignore]` because it requires a Python environment and a `sphere.stl` fixture — not a gap in the test itself.

## 4. JSBSim Trajectory Bridge (Feature-gated)

- [x] 4.1 Set up PyO3 bridge to JSBSim, gated behind `jsbsim` feature
- [x] 4.2 Implement aircraft initialization: select model (F-16, 737, C172), set initial conditions (position, altitude, speed, heading)
- [x] 4.3 Implement waypoint-based autopilot: heading/altitude/speed commands at specified times
- [x] 4.4 Implement simulation runner: advance JSBSim in fixed timesteps, extract state (position, velocity, attitude, acceleration)
- [x] 4.5 Convert JSBSim geodetic output (lat/lon/alt) to thresh ENU coordinates
- [x] 4.6 Implement maneuver library: level turn at specified g-load, climb/descend at specified rate, acceleration/deceleration
- [x] 4.7 Export trajectory as thresh `Trajectory` with `Waypoint` structs
- [x] 4.8 Write test: F-16 level turn produces correct bank angle and turn radius for 4g turn
- [x] 4.9 Write test: 737 climb performance matches published specs within 10%

## 5. High-Fidelity Orbital Propagation

- [ ] 5.1 Add `nyx-space` dependency to thresh-synth (Rust native, always available)
- [ ] 5.2 Implement orbit initialization from TLE (convert TLE elements to nyx-space state)
- [ ] 5.3 Implement orbit initialization from Cartesian state vector (position + velocity at epoch)
- [ ] 5.4 Configure force models: Earth gravity (degree/order configurable), drag (with Cd, area, mass), SRP, Sun/Moon
- [ ] 5.5 Implement propagation: generate position/velocity at configurable time steps
- [ ] 5.6 Implement covariance propagation alongside state (for uncertainty quantification)
- [ ] 5.7 Implement maneuver insertion: impulsive delta-V at specified epoch
- [ ] 5.8 Convert nyx-space ECI output to ENU relative to ground station
- [ ] 5.9 Implement ground station visibility: elevation mask, slant range computation
- [ ] 5.10 Write tests: ISS propagation matches SGP4 within 10 km over 1 day (both use same TLE)
- [ ] 5.11 Write tests: GEO satellite remains near assigned longitude over 24 hours
- [ ] 5.12 Write tests: drag causes LEO orbit to decay (semi-major axis decreases)

## 6. EO/IR Sensor Physics (Pure Rust)

- [ ] 6.1 Implement Planck blackbody spectral radiance: L(λ, T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
- [ ] 6.2 Implement band-integrated radiance for MWIR (3-5 μm) and LWIR (8-12 μm) windows
- [ ] 6.3 Define target IR signature profiles: engine exhaust temperature, skin temperature, projected area vs aspect
- [ ] 6.4 Implement atmospheric transmission: Beer-Lambert with band-specific extinction coefficients, range-dependent
- [ ] 6.5 Define IR sensor parameters: NETD, IFOV, aperture diameter, detector spectral response, integration time
- [ ] 6.6 Implement detection range computation: target irradiance at sensor → contrast vs background → SNR → P_d
- [ ] 6.7 Implement EO/IR measurement generator using physics model (replace fixed P_d Gaussian)
- [ ] 6.8 Create preset configs: MWIR search sensor, LWIR tracking sensor, visible-band camera
- [ ] 6.9 Write tests: blackbody peak wavelength matches Wien's law (λ_max * T = 2898 μm·K)
- [ ] 6.10 Write tests: P_d decreases with range, approaches zero beyond detection horizon
- [ ] 6.11 Write tests: MWIR detects hot exhaust at longer range than LWIR (for afterburner temps)

## 7. Radar Scene Simulation Bridge (Feature-gated)

- [x] 7.1 Set up PyO3 bridge to RadarSimPy, gated behind `radar-scene` feature
- [x] 7.2 Define radar scene: transmitter position/parameters, target list (position, velocity, RCS), clutter model
- [x] 7.3 Implement scene simulation: run RadarSimPy, extract raw detections (range, angle, Doppler, SNR)
- [x] 7.4 Convert RadarSimPy detections to thresh `Measurement::Radar`
- [x] 7.5 Implement clutter configuration: surface clutter (land/sea σ₀), volume clutter (weather)
- [x] 7.6 Implement CFAR threshold configuration: CA-CFAR, OS-CFAR parameters
- [x] 7.7 Implement multi-radar scene: multiple transmitters with different parameters observing the same targets
- [x] 7.8 Write test: single target in free space produces detection at correct range/angle
- [x] 7.9 Write test: target below noise floor is not detected (validates CFAR)

## 8. Integration

- [ ] 8.1 Create high-fidelity benchmark scenario: JSBSim F-16 trajectory → radar equation with Swerling I → tracker → eval
- [ ] 8.2 Create orbital scenario: nyx-space ISS propagation → ground station radar measurements → tracker → eval
- [ ] 8.3 Create multi-sensor scenario: JSBSim aircraft trajectory → radar + MWIR EO/IR → centralized fusion → tracker → eval
- [ ] 8.4 Compare tracker performance: Level 0 (simple) vs Level 1 (radar equation) vs Level 2 (full RadarSimPy) on same trajectory
- [ ] 8.5 Document fidelity levels and when to use each in README
