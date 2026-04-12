## Capability: Radar Equation and Detection Probability

### Overview
Implements the full monostatic radar range equation for computing signal-to-noise ratio (SNR) as a function of radar parameters, target RCS, and range, along with detection probability models (Albersheim approximation and Shnidman equation). Includes atmospheric attenuation modeling per ITU-R P.676 and range-dependent measurement noise that grows with range as predicted by the radar equation. This enables realistic radar detection simulation where detection probability and measurement accuracy degrade with range and vary with target RCS.

## ADDED Requirements

### Requirement: Monostatic radar range equation SNR computation
The system MUST compute received signal-to-noise ratio using the monostatic radar range equation: SNR = (Pt * G^2 * lambda^2 * sigma) / ((4*pi)^3 * R^4 * k * T * B * L), where Pt is peak transmit power (W), G is antenna gain (linear), lambda is wavelength (m), sigma is target RCS (m^2), R is range (m), k is Boltzmann's constant, T is system noise temperature (K), B is receiver bandwidth (Hz), and L is total system losses (linear). The computation MUST correctly handle all units and produce SNR in linear scale and dB.

#### Scenario: SNR computation for known radar parameters
- **GIVEN** a radar with Pt=1 MW, G=40 dB (10000 linear), lambda=0.03 m (X-band), B=1 MHz, T=290 K, L=6 dB (4.0 linear)
- **WHEN** SNR is computed for a target with RCS sigma=1 m^2 at range R=100 km
- **THEN** the computed SNR MUST match the analytic result within 0.1 dB (expected approximately 18.4 dB)

### Requirement: Configurable radar parameter set
The system MUST accept a radar configuration struct containing: peak transmit power Pt (W), antenna gain G (dB), operating frequency or wavelength (Hz or m), receiver bandwidth B (Hz), noise figure NF (dB), system losses L (dB), antenna aperture or beamwidth, and pulse repetition parameters. The system MUST derive dependent quantities (e.g., lambda from frequency, T_sys from NF via T_sys = T_0 * (NF_linear - 1) + T_0).

#### Scenario: Frequency to wavelength derivation
- **GIVEN** a radar configured with operating frequency f=10 GHz and noise figure NF=3 dB
- **WHEN** the radar parameter set is constructed
- **THEN** the derived wavelength MUST be lambda = c/f = 0.03 m and the system noise temperature MUST be T_sys = 290*(10^(3/10) - 1) + 290 = approximately 580 K

### Requirement: Detection probability via Albersheim approximation
The system MUST compute single-pulse detection probability P_d from SNR and false alarm probability P_fa using the Albersheim approximation: SNR_required(dB) = -5.2 + 6.2 + 4.54/sqrt(N) * log10(A), where A is a function of P_d and P_fa. The approximation MUST be valid for P_fa in [1e-12, 1e-2] and P_d in [0.1, 0.999] for non-fluctuating (Swerling 0) targets and MUST support N coherently integrated pulses.

#### Scenario: P_d for a given SNR and P_fa
- **GIVEN** a single-pulse SNR of 13.2 dB and P_fa = 1e-6
- **WHEN** P_d is computed via the Albersheim approximation
- **THEN** P_d MUST be approximately 0.5 (within +/- 0.05) consistent with standard detection curves for a non-fluctuating target

### Requirement: Detection probability via Shnidman equation for fluctuating targets
The system MUST implement the Shnidman equation (or equivalent numerical method) to compute required SNR for a given P_d and P_fa for Swerling I-IV fluctuating targets. This MUST account for the detection loss due to RCS fluctuation: Swerling I/II targets require higher SNR than Swerling 0 for the same P_d, while Swerling III/IV may require less SNR than Swerling I/II for high P_d values.

#### Scenario: Swerling I detection loss relative to non-fluctuating
- **GIVEN** P_d = 0.9 and P_fa = 1e-6
- **WHEN** the required SNR is computed for Swerling 0 and Swerling I targets
- **THEN** the Swerling I required SNR MUST be higher than the Swerling 0 required SNR by approximately 8 dB (the fluctuation loss), within +/- 2 dB

### Requirement: False alarm probability configuration
The system MUST support configurable false alarm probability P_fa that determines the detection threshold. P_fa MUST be settable per radar and MUST support values from 1e-12 to 1e-2. The system MUST also support computing the detection threshold voltage from P_fa assuming Gaussian noise (threshold = sqrt(-2 * ln(P_fa)) * noise_rms for Rayleigh envelope detection).

#### Scenario: Detection threshold from P_fa
- **GIVEN** P_fa = 1e-6 and noise RMS = 1.0
- **WHEN** the Rayleigh envelope detection threshold is computed
- **THEN** the threshold MUST be approximately sqrt(-2 * ln(1e-6)) = approximately 5.26 noise standard deviations

### Requirement: Atmospheric attenuation via ITU-R P.676
The system MUST model one-way atmospheric attenuation as a function of frequency, elevation angle, and path length using the ITU-R P.676 model for gaseous absorption (oxygen and water vapor). The two-way attenuation MUST be applied to the radar equation as an additional loss factor. The model MUST cover frequencies from 1 GHz to 350 GHz and elevation angles from 0 to 90 degrees.

#### Scenario: X-band atmospheric loss at low elevation
- **GIVEN** a radar operating at 10 GHz with target at elevation angle 5 degrees and slant range 200 km
- **WHEN** atmospheric attenuation is computed via ITU-R P.676
- **THEN** the two-way atmospheric loss MUST be computed as approximately 0.02-0.1 dB/km (depending on humidity) integrated over the path, yielding a total two-way loss of approximately 4-20 dB for the full path

#### Scenario: Millimeter-wave absorption peak
- **GIVEN** a radar operating at 60 GHz (oxygen absorption band)
- **WHEN** atmospheric attenuation is computed for a horizontal path at sea level
- **THEN** the specific attenuation MUST be approximately 15 dB/km reflecting the O2 resonance peak, dramatically reducing detection range compared to X-band

### Requirement: Range-dependent measurement noise
The system MUST model radar measurement noise that increases with range as predicted by the radar equation and thermal noise theory. Range measurement noise MUST scale as sigma_range proportional to c/(2*B*sqrt(2*SNR)). Angle measurement noise MUST scale as sigma_angle proportional to beamwidth/(k_a*sqrt(2*SNR)) where k_a is a monopulse slope constant. The system MUST apply these range-dependent errors when generating synthetic measurements.

#### Scenario: Measurement noise growth with range
- **GIVEN** a radar with B=1 MHz and beamwidth=2 degrees tracking a sigma=1 m^2 target
- **WHEN** measurements are generated at ranges 50 km and 200 km
- **THEN** the range measurement standard deviation at 200 km MUST be approximately 4x larger than at 50 km (since SNR drops as R^4 and sigma_range scales as 1/sqrt(SNR), giving sqrt(R^4) = R^2 scaling, so 4^2 = 16 ratio in SNR maps to 4x in sigma)

## Reference Information

### Radar Equation Parameters
| Symbol | Description | Typical X-band Value |
|--------|------------|---------------------|
| Pt | Peak transmit power | 1 MW |
| G | Antenna gain | 30-40 dB |
| lambda | Wavelength | 0.03 m (10 GHz) |
| B | Bandwidth | 1-10 MHz |
| NF | Noise figure | 2-4 dB |
| L | System losses | 4-8 dB |

### Key References
- Skolnik, M. "Introduction to Radar Systems," 3rd Edition, Chapters 1-2
- Albersheim, W.J. "A Closed-Form Approximation to Robertson's Detection Characteristics," Proc. IEEE, 1981
- ITU-R P.676: "Attenuation by atmospheric gases and related effects"
