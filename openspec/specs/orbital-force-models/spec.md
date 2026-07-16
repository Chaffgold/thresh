# orbital-force-models Specification

## Purpose
Composable, individually toggleable perturbation-force stack for high-fidelity orbit propagation — gravity fidelity tiers up to truncated normalized EGM96 spherical harmonics, Harris-Priester atmospheric density, cannonball solar-radiation pressure with cylindrical Earth-shadow eclipse, and lunisolar third-body accelerations from embedded analytic Sun/Moon ephemerides — composed into time-aware acceleration closures that evaluate Earth-fixed legs through the frame-provider seam and reduce to the existing two-body/J2 behavior when every optional force is disabled.

## Requirements

### Requirement: Configurable force-model stack
The system SHALL provide a force-model configuration that composes selectable perturbation accelerations — gravity fidelity (two-body, J2, closed-form J3/J4 zonals, or truncated spherical-harmonic), atmospheric drag (Harris-Priester or the existing exponential model), cannonball solar-radiation pressure with Earth-shadow eclipse, and lunisolar third-body — into a single time-aware acceleration closure compatible with the shared integrator seam. Each force SHALL be individually toggleable, and disabling every optional force SHALL reproduce the existing two-body/J2 behavior.

#### Scenario: Forces are individually toggleable
- **WHEN** two configurations differ only in one enabled force (e.g. SRP on vs off)
- **THEN** the difference between their evaluated accelerations SHALL equal that force's contribution alone

#### Scenario: Baseline configuration matches the existing math
- **WHEN** the configuration enables only two-body + J2 gravity
- **THEN** the composed acceleration SHALL agree with the existing `two_body_acceleration` + `j2_acceleration` functions — evaluated under the force stack's single gravitational constant set (the EGM96-derived `GravityModel`, so the harmonic-consistency identities remain meaningful) — to floating-point equivalence at the same state

### Requirement: Truncated spherical-harmonic gravity with executable normalization contract
The system SHALL evaluate Earth gravity from embedded normalized EGM96 coefficients to a selectable degree and order (at least 12×12), using a normalized associated-Legendre recursion evaluated in the Earth-fixed frame, with coefficients transcribed from the authoritative EGM96 distribution and cited at the table. Consistency with the closed-form zonals SHALL be tested: a harmonics evaluation restricted to C̄₂₀ SHALL reproduce the analytic J2 acceleration, and zonal-only harmonics SHALL reproduce the closed-form J3/J4 terms, within tight relative tolerance.

#### Scenario: C20-only harmonics reproduce analytic J2
- **WHEN** the harmonic evaluation is restricted to the C̄₂₀ coefficient at a LEO-magnitude position
- **THEN** the resulting acceleration SHALL match the closed-form J2 acceleration within a relative tolerance of 1e-10 or better

#### Scenario: Harmonic acceleration matches independent evaluation
- **WHEN** the committed golden fixtures' harmonic-acceleration spot values (computed by an independent implementation from the same fetched coefficient file) are compared with the Rust evaluation
- **THEN** each component SHALL agree within the fixture's documented tolerance

### Requirement: Harris-Priester atmospheric density
The system SHALL provide Harris-Priester atmospheric density — the standard min/max table with diurnal-bulge interpolation toward an apex 30° east of the subsolar direction — as a deterministic, space-weather-free drag density model for the 100–1000 km band, alongside (not replacing) the existing exponential model. Outside the table's altitude span the model SHALL clamp with documented behavior.

#### Scenario: Density between bulge extremes
- **WHEN** density is evaluated at the same altitude at the bulge apex and at the anti-apex
- **THEN** the apex density SHALL equal the table's maximum interpolation, the anti-apex the minimum, and intermediate solar angles SHALL fall monotonically between them

#### Scenario: Published table values reproduced
- **WHEN** density is evaluated exactly at committed table-node altitudes
- **THEN** the min/max values SHALL match the fetched published Harris-Priester table cited in the fixtures

### Requirement: Solar-radiation pressure with eclipse
The system SHALL provide cannonball SRP acceleration parameterized by a reflectivity-area-to-mass coefficient, directed along the Sun-to-spacecraft line, scaled by the inverse-square solar distance, and multiplied by a cylindrical Earth-shadow factor (0 in umbra, 1 in sunlight). The shadow model's cylindrical approximation SHALL be documented at the API.

#### Scenario: SRP vanishes in shadow
- **WHEN** the spacecraft position is inside the cylindrical Earth shadow for the epoch's Sun direction
- **THEN** the SRP contribution SHALL be exactly zero, and outside the cylinder it SHALL be nonzero and anti-sunward

### Requirement: Lunisolar third-body acceleration
The system SHALL provide point-mass third-body accelerations for the Sun and Moon (individually toggleable) using the standard direct-minus-indirect formulation, with body positions from the analytic ephemerides. The implementation SHALL NOT suffer catastrophic cancellation at LEO radii (validated by a dedicated test at LEO magnitude).

#### Scenario: Third-body magnitude at GEO-class radius
- **WHEN** lunar third-body acceleration is evaluated at a GEO-magnitude position
- **THEN** its magnitude SHALL fall within the documented expected range (order 1e-6 m/s²) and point consistently with the Moon's fixture position

### Requirement: Analytic Sun and Moon ephemerides with measured accuracy
The system SHALL compute Sun and Moon positions from embedded analytic series (Meeus/Vallado lineage, no ephemeris files, no network), delivered in GCRF via the reference-frame machinery, with epochs as the time-scale-aware `Epoch`. The series' accuracy SHALL be measured against an independent authority in the committed fixtures — not assumed — and recorded with the fixtures.

#### Scenario: Sun and Moon positions match the independent authority
- **WHEN** the committed fixtures' Sun/Moon GCRF positions (computed by an independent astronomy library at the fixture epochs) are compared with the Rust ephemerides
- **THEN** the angular error SHALL be within the per-body tolerance recorded in the fixture provenance

### Requirement: Earth-fixed force legs use the frame provider
The high-fidelity force terms that are naturally Earth-fixed (spherical-harmonic gravity, Harris-Priester atmosphere co-rotation) SHALL evaluate through the `FrameProvider` seam at the arc epoch plus elapsed time — never through an ad-hoc rotation — so provider substitution (e.g. explicit EOP) propagates into force evaluation without code changes. The legacy exponential-drag tier deliberately retains the existing inertial co-rotation approximation for continuity with the pre-existing model and SHALL document that approximation at its API.

#### Scenario: Provider parameters shift Earth-fixed forces
- **WHEN** the same harmonic-gravity evaluation runs under the zero-default provider and under a provider with non-zero ΔUT1
- **THEN** the resulting accelerations SHALL differ, reflecting the rotated Earth-fixed frame
