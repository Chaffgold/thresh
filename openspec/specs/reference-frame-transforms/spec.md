# reference-frame-transforms Specification

## Purpose
Pure-Rust IAU-76/FK5 reference-frame transform chain between GCRF, MOD, TOD, TEME, PEF, and ITRF — precession, nutation, sidereal rotation, and optional polar motion — with covariance rotation through every transform, a swappable frame-provider seam, committed golden-vector validation, and consistent composition with the ground-station ENU projection.

## Requirements

### Requirement: IAU-76/FK5 transform chain
The system SHALL provide reference-frame transforms between GCRF, MOD, TOD, TEME, PEF, and ITRF via the IAU-76/FK5 reduction — precession (IAU-76), nutation (full IAU-1980 series), sidereal rotation (GMST/GAST with the equation of the equinoxes), and optional polar motion — implemented in pure Rust with no network or ephemeris-file dependency. Composed transforms (e.g. TEME→ITRF) SHALL be available without the caller chaining legs manually.

#### Scenario: TEME and GCRF differ by precession and nutation
- **WHEN** a LEO-magnitude position is transformed from TEME to GCRF at a modern epoch
- **THEN** the transformed position SHALL differ from the input by a kilometre-scale displacement consistent with accumulated precession/nutation, and transforming back SHALL recover the original within numerical tolerance

#### Scenario: Round trip through the full chain
- **WHEN** a state is transformed GCRF→ITRF and back at the same epoch
- **THEN** position and velocity SHALL round-trip within millimetre and micrometre-per-second tolerance respectively

### Requirement: Covariance rotation through every transform
Every frame transform SHALL offer a form that maps state and covariance together, applying the 6×6 Jacobian — block rotation for position and velocity, including the Earth-rotation rate term in inertial↔ITRF velocity transforms. Inertial↔inertial transforms MAY treat the rotation-rate block as zero, and this approximation SHALL be documented at the API.

#### Scenario: Rotating a covariance into ITRF
- **WHEN** a state with an anisotropic 6×6 covariance is transformed from GCRF to ITRF
- **THEN** the position block SHALL equal R·P_rr·Rᵀ and the velocity block SHALL include the ω⊕ rate coupling, with the result symmetric positive semi-definite

#### Scenario: Covariance round trip preserves the matrix
- **WHEN** a covariance is transformed GCRF→TEME→GCRF
- **THEN** the recovered covariance SHALL equal the original within numerical tolerance

### Requirement: Frame provider seam
Frame rotations SHALL be produced through a provider trait taking (source frame, target frame, epoch) and returning the rotation and its rate, with the IAU-76/FK5 implementation as the default provider. Provider parameters that require external Earth-orientation data (ΔUT1, polar motion) SHALL be explicit with zero defaults, and the accuracy consequence of the defaults SHALL be documented.

#### Scenario: Custom provider substitutes without call-site changes
- **WHEN** a caller supplies a non-default provider implementing the trait to a transform entry point
- **THEN** the transform SHALL use the supplied provider's rotations, with no other code change required

#### Scenario: Explicit EOP parameters are honored
- **WHEN** the IAU-76/FK5 provider is constructed with a non-zero ΔUT1 and polar-motion pair
- **THEN** inertial↔ITRF results SHALL reflect those parameters, differing measurably from the zero-default result

### Requirement: Golden-vector validation
The transform chain SHALL be validated against committed golden vectors: Vallado's worked IAU-76/FK5 reduction (with its published EOP inputs) agreeing in position to ≤ 1 m at every leg, and independently generated (astropy/Skyfield) TEME↔GCRF↔ITRF vectors at additional epochs agreeing within documented tolerances. Fixtures SHALL be committed under `test-data/golden/frames/` with provenance recording sources, versions, and regeneration commands.

#### Scenario: Vallado reduction reproduced
- **WHEN** the committed Vallado fixture's ITRF state is transformed to PEF, TOD, MOD, GCRF, and TEME with the fixture's EOP values
- **THEN** each leg's position SHALL match the published values within 1 m

#### Scenario: Deterministic CI execution
- **WHEN** the golden frame tests run twice under default features
- **THEN** they SHALL require no network and no Python and produce bitwise-identical results

### Requirement: ENU composition with tagged frames
The existing ground-station ENU projection SHALL compose with the tagged-frame vocabulary: projecting a TEME state via the GMST-based path SHALL remain supported (it is the TEME→PEF rotation), and projecting a GCRF state SHALL route through the full reduction so both arrive in the same ENU frame consistently.

#### Scenario: TEME and GCRF states project consistently
- **WHEN** the same physical state expressed once in TEME and once in GCRF is projected into a ground station's ENU frame
- **THEN** the two ENU results SHALL agree within the transform-chain tolerance
