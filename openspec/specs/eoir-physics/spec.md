# eoir-physics Specification

## Purpose
TBD - created by archiving change hifi-sensor-simulation. Update Purpose after archive.
## Requirements
### Requirement: Target IR signature from thermal emission
The system MUST compute target infrared spectral radiance using Planck's blackbody radiation law for configurable thermal source components: engine exhaust (afterburner ~1800-2200 K, military power ~900-1200 K, cruise ~600-900 K), skin heating from aerodynamic friction (function of Mach number and altitude), and exhaust plume emission (CO2 and H2O band emission at 4.3 and 2.7 micrometers). The total in-band irradiance at the sensor MUST be computed by integrating spectral radiance over the sensor's spectral band and projecting through the target's emitting area and range.

#### Scenario: Afterburner vs cruise engine signature
- **GIVEN** a fighter aircraft target with engine exhaust area 0.5 m^2, afterburner temperature 2000 K, and cruise temperature 800 K
- **WHEN** the in-band irradiance is computed in the MWIR (3-5 micrometer) band at a range of 50 km
- **THEN** the afterburner irradiance MUST be at least 10x greater than the cruise irradiance, reflecting the strong temperature dependence of blackbody emission (Stefan-Boltzmann T^4 scaling)

#### Scenario: Skin heating at high Mach
- **GIVEN** an aircraft flying at Mach 2.0 at 30000 ft altitude (ambient temperature ~220 K)
- **WHEN** the stagnation temperature is computed as T_stag = T_ambient * (1 + 0.2 * M^2)
- **THEN** the skin temperature MUST be approximately 220 * (1 + 0.2*4) = 396 K, and the LWIR skin emission MUST be computed from this temperature over the projected aircraft skin area

### Requirement: Atmospheric transmission modeling
The system MUST model atmospheric transmission as a function of wavelength, path length, and atmospheric conditions. The model MUST include molecular absorption from CO2 (strong band at 4.3 micrometers), H2O (bands at 2.7 and 6.3 micrometers), and other atmospheric gases. Aerosol scattering and absorption MUST be modeled using a Beer-Lambert exponential extinction law with configurable extinction coefficients for standard atmosphere conditions (clear, haze, fog). The transmission MUST be computed for slant paths accounting for altitude-dependent density profiles.

#### Scenario: MWIR transmission through CO2 absorption band
- **GIVEN** a horizontal sea-level path of 10 km length in the MWIR band
- **WHEN** atmospheric transmission is computed at 4.3 micrometers (CO2 absorption center)
- **THEN** the transmission MUST be near zero (less than 0.01) due to strong CO2 absorption, while transmission at 4.0 micrometers (within the MWIR window but away from CO2) MUST be greater than 0.5

#### Scenario: Clear vs haze visibility
- **GIVEN** a 20 km slant path at 4.0 micrometers wavelength
- **WHEN** transmission is computed for clear atmosphere (visibility 23 km) and haze (visibility 5 km)
- **THEN** the clear-atmosphere transmission MUST be at least 3x higher than the haze transmission, and both MUST follow Beer-Lambert exponential decay with range

### Requirement: Sensor noise model with NETD
The system MUST model EO/IR sensor performance using noise equivalent temperature difference (NETD) as the primary sensitivity metric. The sensor model MUST include instantaneous field of view (IFOV) defining the angular resolution, detector spectral response curve (quantum efficiency vs wavelength), integration time, and f-number (optics speed). The sensor noise MUST be computed as the signal level corresponding to a scene temperature change equal to NETD, and the signal-to-noise ratio MUST be SNR = delta_T / NETD where delta_T is the target-to-background temperature contrast.

#### Scenario: Detection SNR for a warm target
- **GIVEN** a sensor with NETD = 50 mK, IFOV = 0.1 mrad, operating in MWIR (3-5 micrometer)
- **WHEN** observing a target with apparent temperature contrast of 5 K above background
- **THEN** the sensor SNR MUST be approximately 5.0 / 0.050 = 100, and the detection MUST be declared if this exceeds the configured detection threshold

### Requirement: Detection range from contrast and SNR
The system MUST compute maximum detection range by solving for the range at which the target irradiance (after atmospheric attenuation) produces an SNR equal to the detection threshold. The computation MUST account for: target in-band radiant intensity (W/sr), atmospheric transmission over the path, sensor aperture collecting area, sensor NETD, and background clutter level. The detection probability MUST be derived from SNR using P_d = f(SNR, P_fa) analogous to radar detection models.

#### Scenario: Detection range of afterburner target
- **GIVEN** a fighter aircraft in afterburner (radiant intensity ~1000 W/sr in MWIR) observed by a sensor with NETD=30 mK, aperture diameter 200 mm, detection threshold SNR=6
- **WHEN** maximum detection range is computed in clear atmosphere
- **THEN** the detection range MUST be between 80 km and 200 km (depending on atmospheric conditions), and the range MUST decrease significantly (by at least 50%) when haze conditions are applied

#### Scenario: Detection range of cruise target vs afterburner
- **GIVEN** the same sensor observing a fighter in cruise (radiant intensity ~50 W/sr) vs afterburner (~1000 W/sr)
- **WHEN** maximum detection ranges are computed for both
- **THEN** the afterburner detection range MUST be at least 3x the cruise detection range, reflecting the sqrt(intensity) scaling of detection range (since irradiance falls as 1/R^2)

### Requirement: MWIR and LWIR band selection
The system MUST support both MWIR (3-5 micrometer) and LWIR (8-12 micrometer) spectral bands with correct atmospheric transmission windows, detector response characteristics, and background radiation levels for each band. The system MUST allow the user to select the operating band and MUST compute band-specific quantities: MWIR is better for hot targets (engine exhaust, afterburner) while LWIR is better for warm targets (aircraft skin, ground vehicles) due to Wien's displacement law.

#### Scenario: Band selection for engine vs skin detection
- **GIVEN** a fighter aircraft with engine exhaust at 1200 K and skin at 350 K
- **WHEN** the target signature is computed in both MWIR (3-5 micrometer) and LWIR (8-12 micrometer)
- **THEN** the engine exhaust signal MUST be stronger in MWIR (peak emission near 2.4 micrometers, more in-band energy in MWIR than LWIR), while the skin signal MUST be stronger in LWIR (peak emission near 8.3 micrometers), consistent with Wien's displacement law

#### Scenario: Background contrast in MWIR vs LWIR
- **GIVEN** a daytime scenario with sky background temperature of 250 K and a target at 350 K
- **WHEN** the target-to-background contrast is computed in both bands
- **THEN** the LWIR contrast MUST be higher than the MWIR contrast for this 350 K target because the Planck function difference between 350 K and 250 K is proportionally larger in the LWIR band

