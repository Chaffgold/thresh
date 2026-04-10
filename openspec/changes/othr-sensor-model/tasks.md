## 1. Core Types

- [ ] 1.1 Add `Measurement::Othr` variant to thresh-core: ground_range_m, azimuth_rad, doppler_m_s, propagation_mode, time, sensor_id
- [ ] 1.2 Define `PropagationMode` enum: ELayer, FLayer, MultiHop(u8)
- [ ] 1.3 Define `OthrSensorRegistration` struct: transmitter position, receiver position (may be co-located or bistatic-capable), operating frequency
- [ ] 1.4 Write tests: Measurement::Othr serialization roundtrip, PropagationMode variants

## 2. Ionospheric Propagation Model

- [ ] 2.1 Implement Chapman-layer electron density profile: N(h) = N_max * exp(0.5 * (1 - z - exp(-z))) where z = (h - hmF2) / H
- [ ] 2.2 Implement critical frequency computation: foF2 from peak electron density, diurnal sinusoidal variation with solar local time
- [ ] 2.3 Implement Maximum Usable Frequency (MUF): MUF = foF2 * sec(θ_i) where θ_i is incidence angle
- [ ] 2.4 Implement skip zone computation: minimum ground range for single-hop at given frequency
- [ ] 2.5 Implement virtual reflection height: h_virtual(ground_range, frequency) for E-layer (~110 km) and F-layer (~250-350 km)
- [ ] 2.6 Implement ionospheric sounder model: vertical incidence sounding to determine real-time foF2 and hmF2 from sounder measurements
- [ ] 2.7 Implement oblique ionogram synthesis: compute group path vs frequency for target detection validation
- [ ] 2.8 Write tests: MUF increases with foF2, skip zone increases with frequency
- [ ] 2.9 Write tests: virtual height varies correctly between E and F layers
- [ ] 2.10 Write tests: diurnal foF2 peaks at local noon, minimum at midnight

## 3. Coordinate Registration

- [ ] 3.1 Implement Vincenty's direct formula: given (lat, lon, azimuth, distance) → (lat2, lon2)
- [ ] 3.2 Implement Vincenty's inverse formula: given (lat1, lon1, lat2, lon2) → (distance, azimuth)
- [ ] 3.3 Implement OTHR ground-range/azimuth → geodetic (lat, lon) using Vincenty direct from transmitter position
- [ ] 3.4 Implement OTHR ground-range/azimuth → ENU using geodetic→ECEF→ENU chain
- [ ] 3.5 Implement altitude estimation from ionospheric model: assign target altitude based on propagation geometry (mid-path tangent height)
- [ ] 3.6 Write tests: Vincenty roundtrip (direct → inverse) within 1 cm at multiple ranges
- [ ] 3.7 Write tests: OTHR registration at 2000 km matches known great-circle endpoint

## 4. OTHR Measurement Noise Model

- [ ] 4.1 Define `OthrNoiseConfig`: range_sigma_m (10-30 km), azimuth_sigma_rad (0.5-2°), doppler_sigma_m_s (~1 m/s)
- [ ] 4.2 Implement range-dependent noise: range accuracy degrades with ionospheric instability and range
- [ ] 4.3 Implement ionospheric bias model: systematic ground-range error from virtual height uncertainty
- [ ] 4.4 Write tests: noise statistics match configured sigmas over 10K samples

## 5. Multi-Path Disambiguation

- [ ] 5.1 Implement E-layer and F-layer ground-range computation for the same target: two different virtual heights → two different apparent ground ranges
- [ ] 5.2 Implement disambiguation using ionospheric model: compare observed range with predicted E/F ranges, select most consistent
- [ ] 5.3 Implement multi-hop detection: 1-hop vs 2-hop at extended ranges (>3000 km)
- [ ] 5.4 Write tests: disambiguation correctly selects F-layer when E-layer MUF is exceeded
- [ ] 5.5 Write tests: multi-hop produces returns at approximately 2x single-hop ground range

## 6. Synthetic OTHR Generator

- [ ] 6.1 Define `OthrConfig`: transmitter position, operating frequency, bandwidth, PRF, coherent integration time, noise config
- [ ] 6.2 Implement Doppler-based detection: P_d depends on target radial velocity relative to clutter (sea/ground returns near zero Doppler)
- [ ] 6.3 Implement skip-zone blanking: no detections within minimum range
- [ ] 6.4 Implement diurnal coverage variation: detection range changes with ionospheric conditions
- [ ] 6.5 Implement synthetic OTHR measurement generator: target position → ionospheric path → ground range/azimuth/Doppler + noise
- [ ] 6.6 Create preset OTHR configs: ROTHR-class (5-28 MHz, 1000-3500 km), JORN-class (6-30 MHz)
- [ ] 6.7 Write tests: measurements within expected range/azimuth/Doppler bounds
- [ ] 6.8 Write tests: skip zone correctly prevents close-range detections

## 7. Tracker Integration

- [ ] 7.1 Implement OTHR observation matrix: state [x, vx, y, vy, z, vz] → [ground_range, azimuth, doppler]
- [ ] 7.2 Implement OTHR observation Jacobian for EKF (nonlinear ground-range/azimuth mapping)
- [ ] 7.3 Implement OTHR-specific gating: account for coarse resolution in Mahalanobis distance
- [ ] 7.4 Implement OTHR + conventional radar fusion scenario: OTHR cueing → conventional radar track refinement
- [ ] 7.5 Write tests: OTHR-only tracking of CV target converges (with higher position uncertainty)
- [ ] 7.6 Write tests: OTHR + radar fusion reduces position error compared to OTHR alone
- [ ] 7.7 Write integration test: multi-target scenario with OTHR and radar, compute MOTA/IDF1
