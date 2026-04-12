## 1. Swerling RCS Models (Pure Rust)

- [x] 1.1 Swerling I model (slow fluctuation, χ² 2 DOF, constant within dwell) in `crates/thresh-synth/src/swerling.rs`.
- [x] 1.2 Swerling II model (fast fluctuation, χ² 2 DOF, independent pulse-to-pulse).
- [x] 1.3 Swerling III model (slow fluctuation, χ² 4 DOF, dominant scatterer + background).
- [x] 1.4 Swerling IV model (fast fluctuation, χ² 4 DOF, independent pulse-to-pulse).
- [x] 1.5 Swerling case 0 (non-fluctuating deterministic RCS) as baseline.
- [x] 1.6 `RcsProfile` struct (mean RCS in dBsm, Swerling type, optional aspect-angle table).
- [x] 1.7 `RcsLookupTable` with bilinear interpolation; JSON load/save compatible with the `thresh-rcs-compute` CLI output (§3.6).
- [x] 1.8 Preset profiles: fighter, airliner, cruise missile, UAV, satellite (constructors on `RcsProfile`).
- [x] 1.9 Swerling I mean/variance tests at 10 k samples.
- [x] 1.10 RCS lookup interpolation tests at and between grid entries.

## 2. Radar Equation (Pure Rust)

- [x] 2.1 `RadarParameters` struct (Pt, G, λ, bandwidth, noise figure, system losses).
- [x] 2.2 Monostatic radar equation `SNR = (Pt·G²·λ²·σ) / ((4π)³·R⁴·k·Tsys·B·L)` in `radar_equation::compute_snr_enhanced`.
- [x] 2.3 System noise temperature `Tsys = Tant + (NF−1)·T₀` in `system_noise_temp`.
- [x] 2.4 `albersheim_pd()` — Albersheim's P_d approximation from SNR and P_fa.
- [x] 2.5 `shnidman_pd()` — Shnidman's equation for P_d with N integrated pulses and Swerling type.
- [x] 2.6 `atmospheric_attenuation_db_per_km` and `atmospheric_loss_db` — ITU-R P.676-style attenuation vs frequency and elevation angle.
- [x] 2.7 `measurement_noise` — range / angle sigma scaled by `1/√SNR`.
- [x] 2.8 `generate_radar_full` integrates the radar equation end-to-end, replacing the fixed `p_detection` with computed P_d.
- [x] 2.9 Preset configs: X-band surveillance, S-band search, C-band tracking (constructors on `RadarParameters`).
- [x] 2.10 Test: P_d decreases monotonically with range for fixed RCS.
- [x] 2.11 Test: Albersheim matches published tables within tolerance.
- [x] 2.12 Test: atmospheric attenuation increases with frequency.

## 3. RCS Computation Bridge (Feature-gated)

- [x] 3.1 `RcsComputeBridge` in `crates/thresh-synth/src/rcs_compute.rs` under `#[cfg(feature = "rcs-compute")]` wraps PyPOFacets via pyo3 0.24.
- [x] 3.2 `RcsComputeBridge::load_geometry()` calls `pofacets.load_stl()` and returns the facet count.
- [x] 3.3 `RcsComputeBridge::sweep_azimuth()` produces a full azimuth sweep at a fixed elevation.
- [x] 3.4 `RcsComputeBridge::sweep_hemisphere()` produces an azimuth × elevation grid.
- [x] 3.5 `RcsSweepResult::{to_json, write_json}` and `compute_and_save_rcs()` export sweep data as JSON compatible with `RcsLookupTable` (via the non-gated `sweep_to_lookup_table` helper).
- [x] 3.6 `thresh-rcs-compute` binary at `crates/thresh-synth/src/bin/rcs_compute.rs` exposes `--stl / --freq / --step / --output` (plus `--az-start / --az-end / --el / --polarization`). Gated by `required-features = ["rcs-compute"]`. Arg parser lives in `rcs_compute::cli` with 15 unit tests covering help, required-flag errors, polarization validation, NaN/inf rejection, az range checks, sample counting, and hemisphere configs — all runnable without a Python runtime.
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

- [x] 5.1 Chose to ship a pure-Rust RK4 propagator in `crates/thresh-synth/src/orbital.rs` rather than pull in `nyx-space`. Rationale: nyx-space adds heavy dependencies the default CI build doesn't need, and J2 + drag is enough fidelity for the tracking scenarios in §8; nyx-space can still be added later under a feature gate if higher-fidelity force models are required.
- [x] 5.2 `OrbitalState::from_keplerian` initializes from Keplerian elements; SGP4 → Cartesian conversion for TLE input lives in `thresh-data/src/orbital.rs::propagate_tle`.
- [x] 5.3 `OrbitalState::from_cartesian` initializes directly from position/velocity at epoch.
- [x] 5.4 Force models: `PropagatorConfig` carries Earth gravity (J2), `DragConfig { cd, area, mass }`, and the RK4 step size. SRP / Sun / Moon deferred behind a future feature — not needed for the LEO tracking scenarios.
- [x] 5.5 `propagate(initial, duration_s, config, output_dt_s)` produces position/velocity samples at configurable rate.
- [x] 5.6 Covariance propagation deferred: the tracker-side UKF handles uncertainty, and synthetic ground truth is deterministic.
- [x] 5.7 `apply_maneuver` inserts an impulsive delta-V at an epoch.
- [x] 5.8 `orbital_to_enu` converts ECI state to local ENU relative to a ground station (via `thresh-core::eci::eci_to_enu`).
- [x] 5.9 `is_visible` (elevation mask) and `slant_range` computed from the ENU vector.
- [x] 5.10 Test: ISS propagation stays within bounds over extended propagation in `orbital::tests`.
- [x] 5.11 Test: GEO stability (orbit radius conservation) in `orbital::tests`.
- [x] 5.12 Test: LEO orbit decay under drag (semi-major axis decreases) in `orbital::tests`.

## 6. EO/IR Sensor Physics (Pure Rust)

- [x] 6.1 `planck_radiance` in `crates/thresh-synth/src/eoir_physics.rs` implements `L(λ, T) = (2hc²/λ⁵) / (exp(hc/λkT) − 1)`.
- [x] 6.2 `band_radiance` integrates Planck over MWIR (3–5 μm) and LWIR (8–12 μm) spectral bands.
- [x] 6.3 `IrSignature` struct with exhaust / skin / body temperatures and area; presets `fighter_afterburner`, `fighter_military`, `airliner`, `uav_electric`, `ballistic_reentry`.
- [x] 6.4 `atmospheric_transmission` — Beer-Lambert with band-specific extinction coefficients.
- [x] 6.5 `IrSensorConfig` — NETD, IFOV, aperture diameter, detector spectral response, integration time.
- [x] 6.6 `ir_detection_probability` — target irradiance → contrast vs background → SNR → Albersheim P_d.
- [x] 6.7 `generate_eoir_physics` — physics-based EO/IR measurement generator replacing fixed-P_d Gaussian.
- [x] 6.8 Preset sensors: `IrSensorConfig::{mwir_search, lwir_tracking, visible_camera}`.
- [x] 6.9 Test: blackbody peak wavelength matches Wien's law.
- [x] 6.10 Test: P_d monotonic in range, approaches zero beyond horizon.
- [x] 6.11 Test: MWIR detects hot exhaust at longer range than LWIR.

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

- [x] 8.1 `hifi_radar_swerling_scenario` in `crates/thresh/tests/hifi_integration.rs` — fighter CV + CTRV trajectory, X-band surveillance radar with Swerling I fluctuation via `generate_radar_full`, MOTA > 0.3 asserted.
- [x] 8.2 `hifi_orbital_radar_scenario` in the same file — ISS-like overhead pass (408 km altitude, 6.8 km/s ground speed), range-dependent P_d, >20 detections + >50% track maintenance asserted.
- [x] 8.3 `hifi_multisensor_radar_eoir` — maneuvering aircraft with radar + MWIR EO/IR physics fusion (radar preferred, EO/IR fallback), MOTA > 0.3 asserted.
- [x] 8.4 `fidelity_level_comparison` — same trajectory at Level 0 (fixed P_d) and Level 1 (radar equation + Swerling); both assert MOTA > 0.2 and the delta is logged for diagnostics. Level 2 (RadarSimPy) runs via the feature-gated `radar_scene_integration` test suite.
- [x] 8.5 README "Sensor fidelity levels" section documents Level 0 / Level 1 / Level 2 with a when-to-use table, points at the integration test file, and references the `thresh-rcs-compute` CLI for precomputing RCS tables.
