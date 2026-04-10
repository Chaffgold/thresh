## Context

Thresh currently supports three sensor types: conventional radar (range/azimuth/elevation), EO/IR (bearing-only), and ADS-B (position reports). OTHR introduces a fourth modality with fundamentally different measurement geometry — 2D ground-range/azimuth with no elevation — and unique propagation physics that must be modeled for realistic simulation and correct tracker integration.

The key challenge is that OTHR measurements cannot be treated as degraded conventional radar. The propagation path curves through the ionosphere, ground range follows great-circle geometry (not straight-line slant range), and the ionospheric reflection height introduces systematic biases that vary diurnally and seasonally. Fusion of OTHR with conventional sensors requires explicit handling of these differences.

## Goals / Non-Goals

Goals:
- Implement OTHR measurement type compatible with the existing sensor fusion pipeline
- Model ionospheric HF skywave propagation at sufficient fidelity for tracking simulation
- Support great-circle coordinate registration for OTHR ground-range/azimuth measurements
- Enable OTHR + conventional radar fusion for wide-area cueing scenarios
- Provide realistic synthetic OTHR data generation for tracker evaluation
- Support E-layer and F-layer multi-path with disambiguation

Non-Goals:
- Full ionospheric ray-tracing (e.g., PHaRLAP-level fidelity) — use simplified virtual mirror model
- Surface-wave OTHR modeling (coastal/maritime) — focus on skywave only
- Real-time ionospheric data ingestion (ionosonde, GPS TEC) — use parametric models
- OTHR signal processing (waveform design, clutter suppression, CFAR) — model at detection level
- Bistatic or multistatic OTHR configurations

## Decisions

### 1. Measurement representation

**Decision:** Add `Measurement::Othr` variant with `ground_range_m`, `azimuth_rad`, `doppler_m_s`, `propagation_mode`, `time`, `sensor_id`.

**Rationale:** OTHR measures ground range (along the Earth's surface), not slant range. Azimuth is measured but elevation is unobservable. Doppler is critical for detection and provides velocity information. Propagation mode (E/F layer, number of hops) determines the coordinate registration path.

**Alternatives considered:** Reusing `Measurement::Radar` with `elevation = None` was considered but rejected because the range semantics are fundamentally different (ground range vs slant range) and would require conditional logic throughout the tracker.

### 2. Ionospheric model

**Decision:** Use a Chapman-layer virtual mirror model with configurable critical frequency (foF2), layer height (hmF2), and semi-thickness. Diurnal variation modeled as sinusoidal with solar local time.

**Rationale:** Provides sufficient fidelity for tracking simulation without requiring ray-tracing. The virtual mirror model is standard in OTHR system analysis and produces realistic skip zones and MUF/LUF behavior.

**Alternatives considered:** IRI-2020 ionospheric model would be more accurate but requires large coefficient databases and adds significant complexity. Ray-tracing (e.g., Jones-Stephenson) gives path-level fidelity but is computationally expensive for Monte Carlo simulation.

### 3. Coordinate registration

**Decision:** Implement Vincenty's formulae for great-circle ground-range/azimuth to geodetic lat/lon conversion, then use existing WGS84→ECEF→ENU pipeline.

**Rationale:** Great-circle geometry is essential for OTHR — at 2000 km range, flat-Earth errors exceed 50 km. Vincenty's formulae handle the WGS84 ellipsoid correctly at all distances and azimuths.

**Alternatives considered:** Haversine formula is simpler but assumes a sphere (errors of ~0.3% vs ellipsoid). Direct Cartesian projection would introduce unacceptable errors at OTHR ranges.

### 4. Tracker integration

**Decision:** Implement a separate OTHR observation matrix mapping state [x, vx, y, vy, z, vz] to [ground_range, azimuth, doppler]. Doppler maps to radial velocity. No elevation row.

**Rationale:** OTHR provides no elevation information, so the observation matrix has fewer rows (3 vs 4 for conventional radar). This naturally increases the track state covariance in the vertical dimension, which is correct physics. Including Doppler in the observation provides velocity observability that partially compensates for missing elevation.

**Alternatives considered:** Treating OTHR as a 2D sensor (range/azimuth only, ignoring Doppler) would work but discards valuable velocity information that aids association and reduces filter convergence time.

### 5. Multi-path handling

**Decision:** Model E-layer and F-layer returns as separate detections with different ground ranges. Provide a disambiguation function that uses ionospheric state to select the most likely propagation path.

**Rationale:** Multi-path is a fundamental OTHR phenomenology. Both layers can produce valid returns for the same target, and the tracker must handle this or it will create ghost tracks. Explicit modeling in the synthetic generator ensures the tracker is tested against this scenario.

## Risks / Trade-offs

**[Trade-off] Simplified ionosphere vs fidelity** → The Chapman-layer model won't capture sporadic-E, traveling ionospheric disturbances, or polar effects. Mitigation: model is parameterizable, so users can inject measured ionospheric parameters for specific scenarios.

**[Risk] Coordinate registration accuracy** → Errors in virtual reflection height directly bias ground-range estimation. At 2000 km range, a 10 km height error produces ~5 km ground-range bias. Mitigation: model the height uncertainty in the measurement noise covariance.

**[Trade-off] No surface-wave support** → Limits maritime OTHR applications. Mitigation: surface-wave can be added later as a separate propagation mode without changing the measurement type.

**[Risk] Tracker performance with coarse OTHR resolution** → 10-30 km range resolution may cause association ambiguity in dense target environments. Mitigation: Doppler discrimination significantly helps, and cascaded association (OTHR first pass, then conventional sensor refinement) is a natural operational pattern.

## Open Questions

- Should OTHR Doppler be modeled as radial velocity or as ground-range-rate (they differ due to Earth curvature)?
- Should ionospheric parameters be time-varying within a scenario, or fixed per scenario run?
- Should we support frequency-agile OTHR (multiple operating frequencies within a dwell)?
- What altitude should be assumed for OTHR targets when initializing tracks (since elevation is unobservable)?
