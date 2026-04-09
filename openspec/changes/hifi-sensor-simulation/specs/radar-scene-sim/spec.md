## Capability: Radar Scene Simulation via PyO3 Bridge

### Overview
Provides a PyO3-based bridge to RadarSimPy or mpar-sim for full radar scene simulation including multi-target environments, ground and volume clutter, jamming, beam scheduling, pulse-Doppler processing, and CFAR detection. This capability enables end-to-end radar simulation from transmitted waveform through signal processing to detection output, producing detections with range, angle, Doppler, and SNR. The entire capability is feature-gated behind `radar-scene` to keep the core library free of heavy simulation dependencies.

## ADDED Requirements

### Requirement: Feature-gated radar scene simulation module
The system MUST gate all radar scene simulation functionality behind the `radar-scene` Cargo feature flag. When the feature is not enabled, no Python, PyO3, or RadarSimPy/mpar-sim dependencies MUST be compiled or required. When enabled, the module MUST link against PyO3 and require a Python environment with RadarSimPy or mpar-sim installed.

#### Scenario: Build without radar-scene feature
- **GIVEN** the thresh crate is compiled with default features (no `radar-scene`)
- **WHEN** the build completes
- **THEN** the binary MUST compile successfully without any Python installation and MUST NOT contain radar scene simulation entry points

#### Scenario: Build with radar-scene feature
- **GIVEN** the thresh crate is compiled with `--features radar-scene` and Python 3.9+ with RadarSimPy is available
- **WHEN** the build completes
- **THEN** the radar scene simulation API MUST be accessible and functional

### Requirement: Multi-target scene with clutter and jamming
The system MUST support defining radar scenes containing multiple targets (each with position, velocity, and RCS), ground clutter (surface reflectivity map or statistical model with specified clutter-to-noise ratio), volume clutter (rain, chaff with configurable reflectivity), and jamming sources (noise jamming with specified effective radiated power, or deceptive jamming with false target parameters). The scene definition MUST be translatable to the RadarSimPy or mpar-sim input format.

#### Scenario: Multi-target scene with ground clutter
- **GIVEN** a scene with 5 aircraft targets at various ranges (50-200 km), ground clutter with sigma_0 = -20 dBsm/m^2, and no jamming
- **WHEN** the scene is simulated for one radar scan
- **THEN** the output MUST contain target detections (subject to P_d) embedded in clutter returns, with clutter power varying with range and grazing angle consistent with the specified sigma_0

#### Scenario: Noise jamming effects on detection
- **GIVEN** a scene with 3 targets and a noise jammer at 100 km with ERP = 1 kW in the radar bandwidth
- **WHEN** the scene is simulated
- **THEN** target detections in the jammer's angular sector MUST show degraded SNR (reduced by the jammer-to-noise ratio), and targets behind the jammer MUST have lower P_d than targets outside the jammed sector

### Requirement: Beam scheduling patterns
The system MUST support configurable radar beam scheduling including search patterns (raster scan, bar scan with configurable bars and scan rate), track beams (dedicated dwells on tracked targets with configurable revisit interval), and confirmation beams (follow-up dwells on tentative detections). The scheduler MUST allocate dwell time across search, track, and confirm tasks based on configurable priority rules and total timeline budget.

#### Scenario: Search and track interleaving
- **GIVEN** a radar with 60% search timeline budget and 40% track budget, searching a 120-degree azimuth sector with 2-degree beamwidth, and tracking 5 targets with 2-second revisit
- **WHEN** beam scheduling is executed for 10 seconds
- **THEN** the search pattern MUST cover the full sector at least once, track beams MUST revisit each target within 2.5 seconds on average, and total dwell time MUST not exceed the timeline budget

#### Scenario: Confirmation beam on new detection
- **GIVEN** a search beam detects a new target at range 80 km, azimuth 45 degrees
- **WHEN** the confirmation scheduling logic is triggered
- **THEN** a confirmation dwell MUST be scheduled at the detected position within 3 seconds of the initial detection, with dwell time sufficient to achieve P_d >= 0.9 for the estimated target SNR

### Requirement: Pulse-Doppler processing
The system MUST simulate pulse-Doppler radar signal processing including: coherent pulse integration over a dwell (N pulses at PRF), FFT-based Doppler processing to produce a range-Doppler map, range gating with resolution determined by waveform bandwidth, and Doppler resolution determined by coherent processing interval. The range-Doppler map MUST contain signal returns from targets, clutter, and noise with correct relative power levels.

#### Scenario: Range-Doppler map generation
- **GIVEN** a radar with PRF=10 kHz, N=128 pulses per dwell, bandwidth=1 MHz, and a target at range 75 km with radial velocity 300 m/s
- **WHEN** pulse-Doppler processing is performed
- **THEN** the range-Doppler map MUST show a peak in the range bin corresponding to 75 km (range resolution = c/(2B) = 150 m) and the Doppler bin corresponding to 300 m/s (Doppler resolution = lambda*PRF/N), with the peak SNR matching the radar equation prediction

#### Scenario: Clutter Doppler spread
- **GIVEN** a ground-based radar with antenna scanning at omega_s rad/s producing clutter Doppler spread
- **WHEN** pulse-Doppler processing is performed
- **THEN** the ground clutter MUST appear concentrated at low Doppler frequencies (near zero for stationary clutter) with a Doppler spread proportional to the antenna scan rate and beamwidth, and moving targets outside the clutter Doppler band MUST be separable

### Requirement: CFAR detection thresholding
The system MUST implement constant false alarm rate (CFAR) detection on the range-Doppler map. The implementation MUST support cell-averaging CFAR (CA-CFAR) with configurable guard cells and reference cells, and the threshold multiplier MUST be set to achieve the desired P_fa. The CFAR detector MUST output a list of detections that exceed the adaptive threshold, each with range bin, Doppler bin, and measured SNR.

#### Scenario: CA-CFAR detection in homogeneous noise
- **GIVEN** a range-Doppler map with Gaussian noise (no clutter), 3 targets at known locations, P_fa = 1e-6, 16 reference cells, and 2 guard cells
- **WHEN** CA-CFAR is applied
- **THEN** all 3 targets with SNR above the detection threshold MUST be detected, the number of false alarms across the map MUST be approximately N_cells * P_fa (within a factor of 3 for statistical variation), and the CFAR threshold MUST adapt to the local noise level

#### Scenario: CFAR in clutter edge
- **GIVEN** a range-Doppler map with a sharp clutter boundary (strong clutter on one side, noise on the other)
- **WHEN** CA-CFAR is applied near the clutter edge
- **THEN** the CFAR threshold MUST elevate near the clutter edge (due to clutter leaking into reference cells), potentially masking weak targets near the edge, demonstrating the known limitation of CA-CFAR in non-homogeneous environments

### Requirement: Detection output with range, angle, Doppler, and SNR
The system MUST output each detection as a structured record containing: measured range (meters), measured azimuth angle (radians), measured elevation angle (radians), measured Doppler velocity (m/s), measured SNR (dB), detection timestamp, and beam index. Measurement values MUST include realistic noise consistent with the radar equation and processing parameters. The output format MUST be compatible with thresh measurement ingest for tracker evaluation.

#### Scenario: Detection record completeness
- **GIVEN** a scene simulation producing detections from a scanning radar
- **WHEN** the output is inspected
- **THEN** each detection record MUST contain all required fields (range, azimuth, elevation, Doppler, SNR, timestamp, beam_id), and the measurement noise standard deviations MUST be consistent with the theoretical values (sigma_range = c/(2B*sqrt(2*SNR)), sigma_angle = beamwidth/(k_a*sqrt(2*SNR)))

#### Scenario: Import detections into thresh measurement pipeline
- **GIVEN** a completed radar scene simulation producing 500 detections over a 30-second scenario
- **WHEN** the detections are converted to thresh measurement format
- **THEN** all 500 detections MUST be successfully ingested with correct timestamps, measurement vectors, and measurement noise covariance matrices populated from the SNR-derived uncertainties

## Reference Information

### CFAR Parameters
| Parameter | Description | Typical Value |
|-----------|------------|---------------|
| Guard cells | Cells adjacent to CUT excluded from noise estimate | 2-4 |
| Reference cells | Cells used to estimate local noise level | 16-32 |
| P_fa | Desired false alarm probability | 1e-6 to 1e-4 |
| Threshold multiplier | alpha = N_ref * (P_fa^(-1/N_ref) - 1) | Derived from P_fa |

### Key References
- Richards, M.A. "Fundamentals of Radar Signal Processing," McGraw-Hill, 2005
- RadarSimPy: https://github.com/radarsimx/radarsimpy
- Guerci, J.R. "Space-Time Adaptive Processing for Radar," Artech House, 2003
- Skolnik, M. "Introduction to Radar Systems," 3rd Edition, Chapter 3
