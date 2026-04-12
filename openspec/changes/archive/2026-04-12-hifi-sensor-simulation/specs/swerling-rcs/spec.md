## Capability: Swerling RCS Fluctuation Models

### Overview
Implements Swerling I through IV radar cross-section fluctuation models and RCS lookup tables for use in high-fidelity radar measurement generation. Swerling models capture the statistical variation of target RCS due to aspect angle changes and scintillation effects. RCS lookup tables provide deterministic aspect-angle-dependent cross sections for known target archetypes. All implementations are pure Rust with no external dependencies.

## ADDED Requirements

### Requirement: Swerling I slow fluctuation model
The system MUST implement Swerling I RCS fluctuation following a chi-squared distribution with 2 degrees of freedom (exponential distribution), where the RCS value is constant within a single scan (dwell) but fluctuates independently from scan to scan (slow fluctuation). The probability density function MUST be p(sigma) = (1/sigma_avg) * exp(-sigma/sigma_avg) where sigma_avg is the mean RCS.

#### Scenario: Swerling I scan-to-scan decorrelation
- **GIVEN** a target with mean RCS sigma_avg = 1.0 m^2 using Swerling I model
- **WHEN** RCS samples are drawn for 1000 independent scans
- **THEN** the samples MUST follow an exponential distribution with mean 1.0 m^2 (verified by Kolmogorov-Smirnov test at p=0.05) and samples within the same scan MUST return the identical RCS value

### Requirement: Swerling II fast fluctuation model
The system MUST implement Swerling II RCS fluctuation following a chi-squared distribution with 2 degrees of freedom, where the RCS value fluctuates independently from pulse to pulse within a single scan (fast fluctuation). The PDF MUST be the same exponential distribution as Swerling I but with decorrelation at the pulse level rather than the scan level.

#### Scenario: Swerling II pulse-to-pulse independence
- **GIVEN** a target with mean RCS sigma_avg = 5.0 m^2 using Swerling II model
- **WHEN** RCS samples are drawn for 100 pulses within a single scan
- **THEN** each pulse MUST receive an independently drawn RCS value from the exponential distribution, and the sample mean over 10000 pulses MUST converge to sigma_avg within 10% relative error

### Requirement: Swerling III dominant-scatterer slow fluctuation model
The system MUST implement Swerling III RCS fluctuation following a chi-squared distribution with 4 degrees of freedom (representing a target with one dominant scatterer plus smaller contributors), with slow (scan-to-scan) decorrelation. The PDF MUST be p(sigma) = (4*sigma/sigma_avg^2) * exp(-2*sigma/sigma_avg).

#### Scenario: Swerling III reduced variance compared to Swerling I
- **GIVEN** two targets with identical mean RCS sigma_avg = 2.0 m^2, one using Swerling I and one using Swerling III
- **WHEN** 10000 RCS samples are drawn for each
- **THEN** the Swerling III sample variance MUST be approximately half the Swerling I sample variance (chi-squared 4 DOF variance = sigma_avg^2/2 vs chi-squared 2 DOF variance = sigma_avg^2)

### Requirement: Swerling IV fast fluctuation model
The system MUST implement Swerling IV RCS fluctuation following a chi-squared distribution with 4 degrees of freedom with fast (pulse-to-pulse) decorrelation. The statistical distribution MUST match Swerling III but with independent samples at each pulse rather than each scan.

#### Scenario: Swerling IV pulse-level chi-squared 4 DOF
- **GIVEN** a target with mean RCS sigma_avg = 0.5 m^2 using Swerling IV model
- **WHEN** RCS samples are drawn for 500 pulses within a single dwell
- **THEN** each pulse MUST receive an independent chi-squared 4 DOF sample and the empirical CDF MUST match the theoretical chi-squared 4 DOF CDF within Kolmogorov-Smirnov bounds at p=0.05

### Requirement: RCS lookup table with aspect angle dependence
The system MUST support RCS lookup tables that map aspect angle (azimuth and elevation relative to the target body frame) to RCS in dBsm. Tables MUST be loadable from JSON files with a defined schema containing azimuth grid, elevation grid, and RCS matrix. Interpolation between grid points MUST use bilinear interpolation.

#### Scenario: Load and query RCS table from JSON
- **GIVEN** a JSON file containing an RCS table with 1-degree azimuth resolution (0-360) and 1-degree elevation resolution (-90 to 90)
- **WHEN** the system loads the table and queries RCS at azimuth=45.5 deg, elevation=10.3 deg
- **THEN** the returned RCS MUST be bilinearly interpolated from the four surrounding grid points and converted from dBsm to linear m^2

### Requirement: Built-in target archetype RCS tables
The system MUST provide built-in RCS lookup tables for standard target archetypes: fighter aircraft (mean ~1 m^2, 0 dBsm), commercial airliner (mean ~100 m^2, 20 dBsm), cruise missile (mean ~0.01-0.1 m^2, -20 to -10 dBsm), UAV (mean ~0.01-1 m^2, -20 to 0 dBsm), and satellite (mean ~1-10 m^2, 0 to 10 dBsm). Each archetype MUST include aspect-angle variation reflecting physical characteristics (e.g., broadside flash for aircraft, reduced nose-on RCS for stealth-shaped targets).

#### Scenario: Fighter archetype nose-on vs broadside RCS
- **GIVEN** the built-in fighter aircraft RCS archetype is loaded
- **WHEN** RCS is queried at nose-on aspect (azimuth=0 deg) and broadside aspect (azimuth=90 deg)
- **THEN** the broadside RCS MUST be at least 10 dB higher than the nose-on RCS, reflecting the typical aspect-angle signature of a fighter-sized target

### Requirement: Pure Rust implementation with no external dependencies
All Swerling model implementations and RCS table handling MUST be implemented in pure Rust without external C/C++/Python dependencies. Random number generation MUST use the rand and rand_distr crates (or equivalent pure-Rust crates) for chi-squared and exponential sampling.

#### Scenario: Build without optional features
- **GIVEN** the thresh crate is built with default features only
- **WHEN** the Swerling RCS module is compiled
- **THEN** compilation MUST succeed without requiring any system-level libraries, Python installation, or C/C++ toolchain beyond the standard Rust toolchain

## Reference Information

### Swerling Model Summary
| Model | Distribution | DOF | Decorrelation | Physical Interpretation |
|-------|-------------|-----|---------------|------------------------|
| I | Chi-squared | 2 | Scan-to-scan | Many equal scatterers, slow |
| II | Chi-squared | 2 | Pulse-to-pulse | Many equal scatterers, fast |
| III | Chi-squared | 4 | Scan-to-scan | Dominant + small scatterers, slow |
| IV | Chi-squared | 4 | Pulse-to-pulse | Dominant + small scatterers, fast |

### Key References
- Swerling, P. "Probability of Detection for Fluctuating Targets," IRE Transactions, 1960
- Skolnik, M. "Introduction to Radar Systems," 3rd Edition, Chapter 2
