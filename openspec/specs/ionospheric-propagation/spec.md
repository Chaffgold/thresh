# ionospheric-propagation Specification

## Purpose
TBD - created by archiving change othr-sensor-model. Update Purpose after archive.
## Requirements
### Requirement: Chapman-layer electron density
The system SHALL implement a Chapman-layer electron density profile parameterized by peak density (NmF2), peak height (hmF2), and scale height (H). The system MUST support this.

#### Scenario: Peak density at layer height
- **WHEN** the electron density is computed at height hmF2
- **THEN** the density SHALL equal NmF2 (the configured peak)

#### Scenario: Density decreases above and below peak
- **WHEN** the density is computed at hmF2 ± 50 km
- **THEN** the density SHALL be less than NmF2

### Requirement: Maximum Usable Frequency
The system SHALL compute MUF as a function of foF2 and incidence angle. The system MUST support this.

#### Scenario: MUF at vertical incidence
- **WHEN** the incidence angle is 0° (vertical)
- **THEN** MUF SHALL equal foF2

#### Scenario: MUF increases with oblique angles
- **WHEN** the incidence angle increases from 0° to 70°
- **THEN** MUF SHALL increase monotonically

### Requirement: Skip zone computation
The system SHALL compute the minimum ground range at which a given frequency can propagate via ionospheric reflection. The system MUST support this.

#### Scenario: Higher frequency increases skip distance
- **WHEN** the operating frequency increases while ionospheric conditions remain constant
- **THEN** the skip zone minimum range SHALL increase

### Requirement: Virtual reflection height
The system SHALL compute the effective virtual reflection height for E-layer and F-layer propagation as a function of ground range and frequency. The system MUST support this.

#### Scenario: E-layer vs F-layer heights
- **WHEN** virtual heights are computed for E-layer and F-layer
- **THEN** the E-layer height SHALL be approximately 110 km and F-layer SHALL be approximately 250-350 km

### Requirement: Ionospheric sounder model
The system SHALL model vertical-incidence ionospheric sounding to determine real-time foF2 and hmF2 from sounder measurements. The system MUST support this.

#### Scenario: Sounder determines critical frequency
- **WHEN** a vertical-incidence sounder measurement is processed
- **THEN** the system SHALL extract foF2 as the maximum frequency at which a reflection is returned

### Requirement: Diurnal variation
The system SHALL model diurnal variation of ionospheric parameters with solar local time. The system MUST support this.

#### Scenario: foF2 peaks at local solar noon
- **WHEN** foF2 is computed over a 24-hour period at a fixed location
- **THEN** the peak SHALL occur near local solar noon and the minimum near local midnight

