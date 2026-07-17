# ionospheric-group-delay Specification

## Purpose
Frequency-dependent ionospheric group delay for microwave radar ranging — the closed-form `Δr = K·STEC/f²` range lengthening from a configured vertical TEC through the standard thin-shell obliquity mapping, a convenience bridge integrating the existing OTHR Chapman electron-density profile to a vertical TEC, and an exact inverse correction with documented residuals under TEC mismatch, applicable before RAE→Cartesian conversion.

## Requirements

### Requirement: Frequency-dependent group delay from configured TEC
The system SHALL compute the ionospheric range delay for microwave radar as `Δr = K·STEC/f²`, with the constant `K`'s exact value and units taken from a fetched published derivation (cited at the constant), slant TEC obtained from a configured vertical TEC through the standard thin-shell obliquity mapping (configurable shell height, published default cited), and the delay lengthening the measured range. Ionospheric elevation bending at microwave frequencies SHALL be documented as negligible and not modeled.

#### Scenario: Inverse-square frequency scaling
- **WHEN** the delay is evaluated at the same geometry and TEC for two frequencies an octave apart
- **THEN** the delays SHALL scale as 1/f² exactly

#### Scenario: Band magnitudes are physical
- **WHEN** the delay is evaluated at a representative mid-latitude TEC for L-, S-, and X-band frequencies at moderate elevation
- **THEN** the magnitudes SHALL fall in the documented metre / decimetre / centimetre regimes respectively

#### Scenario: Obliquity grows toward the horizon
- **WHEN** the slant delay is evaluated from zenith down to low elevation at fixed vertical TEC
- **THEN** the delay SHALL increase monotonically with the thin-shell mapping factor

### Requirement: Chapman-profile TEC bridge
The system SHALL provide a bridge integrating the existing OTHR Chapman electron-density profile to a vertical TEC (with explicit electrons/m² → TECU unit handling), so scenarios using both regimes can be made mutually consistent. The bridge SHALL be a convenience constructor for the configured-TEC path, not a coupling of the two capabilities.

#### Scenario: Hand-integrated profile matches
- **WHEN** the bridge integrates a Chapman profile whose integral is independently hand-computed in the test
- **THEN** the resulting vertical TEC SHALL match within the documented quadrature tolerance, in TECU

### Requirement: Delay correction with bounded residual
The system SHALL provide an inverse correction removing the modeled delay from a measured range before RAE→Cartesian conversion. Bias-then-correct with true parameters SHALL recover the unbiased range exactly (the model is closed-form), and correcting with a vertical TEC mis-set by ±25% SHALL leave a residual documented and asserted proportionally smaller than the uncorrected delay.

#### Scenario: Closed-form round trip
- **WHEN** a range is biased by the delay model and corrected with identical parameters
- **THEN** the original range SHALL be recovered to floating-point precision

#### Scenario: Mismatched TEC correction still helps
- **WHEN** the correction runs with vertical TEC mis-set by ±25%
- **THEN** the residual range error SHALL equal the mis-set fraction of the true delay, recorded at the test
