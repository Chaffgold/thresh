# tropospheric-refraction Specification (Delta)

## ADDED Requirements

### Requirement: Exponential-profile refraction model
The system SHALL compute tropospheric refraction — apparent-elevation bending and excess range — for a ground station observing an elevated target, from a Bean–Dutton exponential refractivity profile `N(h) = N_s·e^(−h/H)` integrated along the spherically-stratified ray path with a deterministic fixed quadrature. Profile constants SHALL be configurable with published defaults whose source is fetched and cited, and repeated evaluation SHALL be bitwise deterministic.

#### Scenario: Published worked values reproduced
- **WHEN** refraction is evaluated at the fetched published reference conditions (surface refractivity, elevation, target height from the cited source's worked example)
- **THEN** the computed bending and range error SHALL match the published values within the documented tolerance

#### Scenario: Bending grows toward the horizon
- **WHEN** refraction is evaluated at decreasing elevations from 45° toward the validity floor
- **THEN** the elevation bending and range error SHALL increase monotonically

### Requirement: 4/3-Earth cheap tier with cross-check
The system SHALL provide the classic 4/3-Earth effective-radius refraction model as a cheap tier, and the two tiers SHALL be cross-checked: at moderate elevations they agree within a documented band, with the exponential tier authoritative.

#### Scenario: Tiers agree at moderate elevation
- **WHEN** both tiers evaluate the same geometry at elevations between 5° and 45°
- **THEN** their apparent-elevation corrections SHALL agree within the documented cross-check band

### Requirement: Explicit validity band
Refraction functions SHALL declare a minimum-elevation validity floor (default 1°, configurable), clamp below it with documented behavior, and state that ducting/super-refraction regimes are out of scope.

#### Scenario: Below-floor evaluation clamps
- **WHEN** refraction is requested below the validity floor
- **THEN** the result SHALL equal the floor evaluation (clamped) and the API documentation SHALL state this behavior

### Requirement: Refraction correction with bounded residual
The system SHALL provide an inverse-model correction mapping an apparent (measured) elevation and range to refraction-corrected values, suitable for application before RAE→Cartesian conversion. A bias-then-correct round trip with true parameters SHALL recover the unbiased geometry within quadrature tolerance, and correcting with deliberately mismatched profile parameters SHALL leave a residual approximately equal to the mis-set fraction of the uncorrected bias — the bias is near-linear in surface refractivity, so a parameter mis-set by a given fraction leaves that fraction of the bias uncorrected — measured, documented, and recorded at the test.

#### Scenario: Round trip with true parameters
- **WHEN** a geometric measurement is biased by the model and then corrected using the same parameters
- **THEN** elevation and range SHALL be recovered within the documented quadrature tolerance

#### Scenario: Mismatched correction still helps
- **WHEN** the correction runs with surface refractivity mis-set by ±10%
- **THEN** the residual elevation and range errors SHALL be approximately the mis-set fraction (≈10%) of the uncorrected bias, with the measured residual ratios recorded at the test; the full bias is removed only when correcting with the true parameters
