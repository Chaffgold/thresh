## Why

Over-the-Horizon Radar (OTHR) is a critical sensor modality for wide-area surveillance of aircraft, ships, and ballistic missiles at ranges of 1,000–3,500 km. Systems like AN/TPS-71 ROTHR, Jindalee (JORN), and Nostradamus exploit HF skywave propagation via ionospheric refraction to detect targets far beyond line-of-sight radar horizons.

OTHR presents fundamentally different measurement geometry and error characteristics compared to conventional microwave radar. Signals propagate via ionospheric skip (E-layer at ~110 km, F-layer at ~250-350 km), producing measurements in ground range and azimuth with no direct elevation observable. Range resolution is coarse (10–30 km), azimuth accuracy is ~0.5–2°, and multi-path propagation creates range ambiguities. Doppler velocity is often the primary detection discriminant, unlike conventional radar where range-based detection dominates.

No existing Rust tracking framework supports OTHR phenomenology. Adding OTHR to thresh enables heterogeneous fusion of OTHR wide-area cueing with conventional sensor precision tracking — a key operational pattern for aerospace defense.

## What Changes

- Introduce `Measurement::Othr` variant with ground-range, azimuth, Doppler, and ionospheric mode metadata
- Implement ionospheric propagation model: virtual reflection height, great-circle ground range, skip zone computation
- Implement OTHR-specific measurement noise model with range/azimuth/Doppler uncertainties
- Implement coordinate registration: OTHR ground-range/azimuth to Cartesian ENU via great-circle geometry
- Implement multi-path disambiguation for E-layer and F-layer returns
- Build synthetic OTHR measurement generator with realistic phenomenology
- Implement OTHR-aware data association accounting for coarse resolution and missing elevation
- Create preset OTHR system configurations for representative systems

## Capabilities

### New Capabilities

- `othr-measurement-type`: New `Measurement::Othr` variant in thresh-core with ground range, azimuth, Doppler velocity, and propagation mode metadata
- `ionospheric-propagation`: HF skywave propagation model including virtual reflection height, skip zone, MUF/LUF, and diurnal ionospheric variation
- `othr-coordinate-registration`: Great-circle geometry transforms from OTHR ground-range/azimuth to ECEF/ENU, accounting for Earth curvature
- `othr-measurement-noise`: Range-dependent noise model reflecting OTHR resolution limits (10–30 km range, 0.5–2° azimuth, ~1 m/s Doppler)
- `multi-path-disambiguation`: Resolve E-layer vs F-layer propagation ambiguity using range consistency and ionospheric models
- `othr-synthetic-data`: Synthetic OTHR measurement generator with ionospheric effects, skip zones, and Doppler-based detection
- `othr-data-association`: Modified gating and cost computation for OTHR's coarse resolution and 2D-only (no elevation) geometry

### Modified Capabilities

- `thresh-core::measurement::Measurement`: Add `Othr` variant
- `thresh-tracker`: Support OTHR observation matrices (2D ground-range/azimuth, no elevation)
- `thresh-fusion`: Handle OTHR measurements in centralized and information-filter fusion with conventional sensors

## Impact

**Modified crates:**
- `thresh-core` — new Measurement variant, OTHR coordinate types
- `thresh-synth` — OTHR measurement generator, ionospheric model
- `thresh-tracker` — OTHR-compatible observation matrix and gating
- `thresh-fusion` — OTHR sensor registration and cross-sensor fusion

**Dependencies:**
- No new external dependencies — ionospheric models and great-circle geometry are pure Rust math

**Build requirements:**
- None beyond existing workspace setup

**Deployment:**
- OTHR capability available via standard crate imports, no feature gating required (core sensor type)
