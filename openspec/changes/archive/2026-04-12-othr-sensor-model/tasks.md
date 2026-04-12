## 1. Core Types

- [x] 1.1 `Measurement::Othr` variant in `crates/thresh-core/src/measurement.rs` carries ground_range_m, azimuth_rad, doppler_m_s, propagation_mode, time, sensor_id.
- [x] 1.2 `PropagationMode` enum in `measurement.rs`: `ELayer`, `FLayer`, `MultiHop(u8)`.
- [x] 1.3 `OthrSensorRegistration` struct in `crates/thresh-core/src/othr.rs` with transmitter lat/lon/alt and operating frequency.
- [x] 1.4 Serialization roundtrip tests in `measurement.rs` and `othr.rs`.

## 2. Ionospheric Propagation Model

- [x] 2.1 `chapman_density(height_km, n_max, hm_km, scale_height_km)` in `crates/thresh-synth/src/ionosphere.rs`.
- [x] 2.2 `fo_f2_diurnal(base_fo_f2_mhz, solar_local_time_hours)` with noon peak / midnight trough.
- [x] 2.3 `muf(fo_f2_mhz, ground_range_km, virtual_height_km)` via secant law.
- [x] 2.4 `skip_zone_range_km(freq_mhz, params)` minimum ground range for single-hop.
- [x] 2.5 `virtual_height_km(ground_range_km, layer)` with E-layer constant 110 km and F-layer 250–350 km model.
- [x] 2.6 `sounder_measurement(params)` returns foF2 and hmF2.
- [x] 2.7 `oblique_ionogram(ground_range_km, params, freq_range, n_points)` for group-path vs frequency.
- [x] 2.8 MUF-vs-foF2 and skip-zone-vs-frequency tests in `ionosphere.rs`.
- [x] 2.9 Virtual-height E/F tests in `ionosphere.rs`.
- [x] 2.10 Diurnal foF2 noon/midnight tests in `ionosphere.rs`.

## 3. Coordinate Registration

- [x] 3.1 `vincenty_direct(lat1, lon1, azimuth, distance)` in `crates/thresh-core/src/othr.rs`.
- [x] 3.2 `vincenty_inverse(lat1, lon1, lat2, lon2)` with near-antipodal fallback.
- [x] 3.3 `othr_to_geodetic(reg, ground_range_m, azimuth_rad)` via Vincenty direct from transmitter position.
- [x] 3.4 `othr_to_enu(reg, ground_range_m, azimuth_rad, estimated_alt_m, ref_lat_rad, ref_lon_rad, ref_alt_m)` via geodetic→ECEF→ENU chain.
- [x] 3.5 `estimated_target_altitude_m(PropagationMode, ground_range_km)` in `crates/thresh-synth/src/ionosphere.rs`: mid-path tangent height from virtual reflection height minus spherical-Earth sagitta, multi-hop aware, clamped to a 15 km aircraft ceiling. Unit tests cover the clamp, long-range non-negativity, multi-hop per-hop geometry, and the `MultiHop(0)` defensive case (landed in PR #34).
- [x] 3.6 Vincenty direct→inverse roundtrip tests at multiple ranges within 1 cm tolerance in `othr.rs`.
- [x] 3.7 OTHR registration end-to-end tests at 2000+ km against known great-circle endpoints in `othr.rs`.

## 4. OTHR Measurement Noise Model

- [x] 4.1 `OthrNoiseConfig` struct in `crates/thresh-synth/src/othr_noise.rs`: range_sigma_m, azimuth_sigma_rad, doppler_sigma_m_s, range_bias_sigma_m.
- [x] 4.2 `range_dependent_noise(base_sigma, range_m)` with sqrt scaling.
- [x] 4.3 `ionospheric_bias_m(virtual_height_uncertainty_km, ground_range_km)` for systematic range error.
- [x] 4.4 Noise-statistics validation tests (mean / variance over 10k samples) in `othr_noise.rs`.

## 5. Multi-Path Disambiguation

- [x] 5.1 `multipath_ranges(target_range_km, params)` returns both E-layer and F-layer apparent ranges in `crates/thresh-synth/src/multipath.rs`.
- [x] 5.2 `disambiguate(observed_range_km, freq_mhz, params)` compares observed range against predicted E/F modes and returns the most consistent `PropagationMode`.
- [x] 5.3 `is_multihop_viable(ground_range_km, freq_mhz, params)` detects multi-hop at extended ranges.
- [x] 5.4 Disambiguation correctness tests (F-layer when E-layer MUF exceeded).
- [x] 5.5 Multi-hop tests (2-hop returns at ~2× single-hop ground range).

## 6. Synthetic OTHR Generator

- [x] 6.1 `OthrConfig` struct in `crates/thresh-synth/src/othr_generator.rs`: transmitter, frequency, bandwidth, PRF, coherent integration time, noise parameters.
- [x] 6.2 `doppler_detection_probability(radial_velocity_m_s, clutter_doppler_m_s)` — P_d depends on Doppler separation from clutter.
- [x] 6.3 Skip-zone blanking enforced in `generate_othr()`.
- [x] 6.4 `diurnal_factor(solar_local_time_hours)` cosine model in [0.6, 1.0].
- [x] 6.5 `generate_othr(target, config, time, solar_local_time)` main measurement generator.
- [x] 6.6 Preset configs `OthrConfig::rothr()` and `OthrConfig::jorn()` for ROTHR-class and JORN-class systems.
- [x] 6.7 Bounds tests (range / azimuth / Doppler) in `othr_generator.rs`.
- [x] 6.8 Skip-zone exclusion tests in `othr_generator.rs`.

## 7. Tracker Integration

- [x] 7.1 `othr_observation_jacobian()` maps state `[x, vx, y, vy, z, vz]` → `[range, azimuth, doppler]` in `crates/thresh-tracker/src/othr_integration.rs`.
- [x] 7.2 Analytical Jacobian for EKF nonlinear ground-range / azimuth mapping in `othr_integration.rs`.
- [x] 7.3 `othr_cartesian_noise()` conservative Mahalanobis gating to account for coarse OTHR resolution.
- [x] 7.4 OTHR + conventional-radar fusion scenario in `othr_fusion.rs`.
- [x] 7.5 OTHR-only CV-tracking convergence tests.
- [x] 7.6 OTHR + radar fusion position-error comparison tests.
- [x] 7.7 Multi-target OTHR + radar MOTA / IDF1 integration test.

## 8. Long-Range Tracking Frames

### 8.A ECEF Tracking Variant

- [x] 8.A.1 `EcefMotionModel` with centrifugal + Coriolis terms in `crates/thresh-tracker/src/ecef_tracker.rs`.
- [x] 8.A.2 `MultiObjectTrackerEcef` full tracker variant with ECEF state and observation models.
- [x] 8.A.3 OTHR observation matrix for ECEF state (great-circle range, initial bearing, radial velocity).
- [x] 8.A.4 Conventional radar observation matrix for ECEF state.
- [x] 8.A.5 Track output conversion from ECEF state to ENU at user-supplied reference point.
- [x] 8.A.6 3000 km cross-coverage transit accuracy test.
- [x] 8.A.7 Great-circle aircraft path 1-hour tracking test.
- [x] 8.A.8 ECEF vs ENU long-traverse MOTA benchmark.

### 8.B Great-Circle Motion Model

- [x] 8.B.1 `GreatCircleState` (lat, lon, alt, ground_speed, heading, climb_rate) in `crates/thresh-tracker/src/great_circle_tracker.rs`.
- [x] 8.B.2 `GreatCircleMotionModel` predict step using Vincenty direct.
- [x] 8.B.3 Geodetic state Jacobian via finite differences for EKF.
- [x] 8.B.4 `MultiObjectTrackerGreatCircle` full variant.
- [x] 8.B.5 Single-detection initialization with assumed altitude.
- [x] 8.B.6 >1000 km constant-heading flight test.
- [x] 8.B.7 Polar region (longitude wraparound) test.
- [x] 8.B.8 Great-circle vs ENU long-duration benchmark.

### 8.C Recentered ENU Tracker

- [x] 8.C.1 Per-track ENU origin in `crates/thresh-tracker/src/recentered_enu_tracker.rs`.
- [x] 8.C.2 Origin recentering policy at ~200 km centroid drift threshold.
- [x] 8.C.3 State + covariance transformation across recentering.
- [x] 8.C.4 Per-track measurement conversion into local ENU frame.
- [x] 8.C.5 Recentering continuity tests (no filter-state jumps).
- [x] 8.C.6 Accuracy-parity-with-ECEF long-traverse tests.

### 8.D Local Stereographic Projection

- [x] 8.D.1 `stereographic_project(lat, lon, center)` in `crates/thresh-tracker/src/stereographic_tracker.rs`.
- [x] 8.D.2 `stereographic_inverse(plane_coords, center)`.
- [x] 8.D.3 `MultiObjectTrackerStereographic` tracker variant with stereographic 2D + altitude state.
- [x] 8.D.4 OTHR observation matrix for stereographic state.
- [x] 8.D.5 `recommended_center()` for single or multiple OTHR transmitters.
- [x] 8.D.6 Stereographic projection roundtrip tests (< 1 m at OTHR coverage ranges).
- [x] 8.D.7 Full coverage-area tracking accuracy tests (long-traverse test in `stereographic_tracker_tests.rs` — cognitive complexity refactored in PR #33).
- [x] 8.D.8 Stereographic vs ENU vs ECEF benchmark.

### 8.E Tracker Selection and Documentation

- [x] 8.E.1 `TrackerVariant` enum in `crates/thresh-tracker/src/tracker_variant.rs` with all four variants.
- [x] 8.E.2 `TrackerVariant::recommend()` decision logic with module-level documentation.
- [x] 8.E.3 Scenario-driven selection wired into the benchmark runner in `thresh-data/src/benchmark.rs`.
- [x] 8.E.4 End-to-end comparison via the benchmark harness and the `fidelity_level_comparison` integration test.
