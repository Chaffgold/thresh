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

- [x] 7.1 Implement OTHR observation matrix: state [x, vx, y, vy, z, vz] → [ground_range, azimuth, doppler]
- [x] 7.2 Implement OTHR observation Jacobian for EKF (nonlinear ground-range/azimuth mapping)
- [x] 7.3 Implement OTHR-specific gating: account for coarse resolution in Mahalanobis distance
- [x] 7.4 Implement OTHR + conventional radar fusion scenario: OTHR cueing → conventional radar track refinement
- [x] 7.5 Write tests: OTHR-only tracking of CV target converges (with higher position uncertainty)
- [x] 7.6 Write tests: OTHR + radar fusion reduces position error compared to OTHR alone
- [x] 7.7 Write integration test: multi-target scenario with OTHR and radar, compute MOTA/IDF1

## 8. Long-Range Tracking Frames

### 8.A ECEF Tracking Variant

- [x] 8.A.1 Implement `EcefMotionModel`: 6-state CV in ECEF coordinates with proper centrifugal/Coriolis terms for Earth-fixed frame
- [x] 8.A.2 Implement `MultiObjectTrackerEcef`: tracker variant with ECEF state and observation models
- [x] 8.A.3 Implement OTHR observation matrix for ECEF state: ground_range from great-circle, azimuth from initial bearing, doppler from radial velocity
- [x] 8.A.4 Implement conventional radar observation matrix for ECEF state (range/azimuth/elevation from sensor ECEF position)
- [x] 8.A.5 Implement track output conversion: ECEF state → ENU at user-supplied reference point for visualization/eval
- [x] 8.A.6 Write tests: ECEF tracker maintains position accuracy on 3000 km cross-coverage transit
- [x] 8.A.7 Write tests: ECEF tracker correctly tracks a great-circle aircraft path over 1 hour
- [x] 8.A.8 Write benchmark: ECEF vs ENU tracker on long-traverse scenario, compare MOTA

### 8.B Great-Circle Motion Model

- [x] 8.B.1 Define `GreatCircleState`: lat, lon, alt, ground_speed, heading, climb_rate (6-state geodetic)
- [x] 8.B.2 Implement `GreatCircleMotionModel`: predict step uses Vincenty direct formula to advance lat/lon along current heading
- [x] 8.B.3 Implement geodetic state Jacobian for EKF (analytical or numerical via finite differences)
- [x] 8.B.4 Implement `MultiObjectTrackerGreatCircle`: tracker variant with geodetic state and OTHR/radar observation models
- [x] 8.B.5 Implement initialization: convert single OTHR detection to initial geodetic state with assumed altitude and zero velocity
- [x] 8.B.6 Write tests: great-circle tracker correctly maintains aircraft constant-heading flight over 1000+ km
- [x] 8.B.7 Write tests: great-circle tracker handles polar regions without singularity (longitude wraparound)
- [x] 8.B.8 Write benchmark: great-circle vs ENU tracker on long-duration aircraft track

### 8.C Recentered ENU Tracker

- [x] 8.C.1 Implement per-track ENU origin tracking: each track stores its own ENU reference point
- [x] 8.C.2 Implement origin recentering policy: when track centroid drifts more than threshold (e.g., 200 km), recenter ENU at current centroid
- [x] 8.C.3 Implement state transformation across recentering: rotate state vector and covariance into new ENU frame
- [x] 8.C.4 Implement measurement transformation per track: convert measurement to track's local ENU before update
- [x] 8.C.5 Write tests: recentering preserves filter state continuity (no jumps in track output)
- [x] 8.C.6 Write tests: recentered ENU tracker matches ECEF tracker accuracy on long-traverse scenario

### 8.D Local Stereographic Projection

- [ ] 8.D.1 Implement stereographic projection: geodetic (lat, lon) → 2D plane preserving distances from a center point
- [ ] 8.D.2 Implement inverse stereographic projection: 2D plane → geodetic
- [ ] 8.D.3 Define `MultiObjectTrackerStereographic`: tracker variant using stereographic 2D + altitude state
- [ ] 8.D.4 Implement OTHR observation matrix for stereographic state (range/azimuth direct mapping)
- [ ] 8.D.5 Implement projection center selection: place at OTHR transmitter or coverage centroid
- [ ] 8.D.6 Write tests: stereographic projection roundtrip within 1 m at OTHR coverage ranges
- [ ] 8.D.7 Write tests: stereographic tracker accurately tracks targets across full OTHR coverage area
- [ ] 8.D.8 Write benchmark: stereographic vs ENU vs ECEF on representative OTHR scenarios

### 8.E Tracker Selection and Documentation

- [x] 8.E.1 Define `TrackerVariant` enum and factory function for selecting appropriate tracker based on scenario
- [x] 8.E.2 Document selection guidance: ENU for short tracks <500 km traverse, ECEF for ballistic/orbital, great-circle for aircraft >1000 km, stereographic for area surveillance
- [x] 8.E.3 Add scenario-driven tracker selection to benchmark runner
- [x] 8.E.4 Write end-to-end comparison: same OTHR scenario tracked by all 4 variants, document accuracy/runtime tradeoffs
