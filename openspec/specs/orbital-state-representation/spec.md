# orbital-state-representation Specification

## Purpose
Frame-disciplined orbital state type with lazy conversion between Cartesian, Keplerian, TLE mean-element, and Equinoctial representations, carrying an explicit gravitational parameter and reference-frame tag.

## Requirements

### Requirement: Element-representation enum with lazy conversion
The system SHALL provide an orbital state type that stores its elements in one of four representations — Cartesian, Keplerian, TLE mean elements, or Equinoctial — and converts between representations on demand (lazily) rather than eagerly at construction. Conversions SHALL be mutually consistent: converting to another representation and back SHALL recover the original elements within numerical tolerance.

#### Scenario: Keplerian round trip
- **WHEN** an orbital state constructed from Keplerian elements is converted to Cartesian and back to Keplerian
- **THEN** the recovered elements SHALL match the originals within a tight numerical tolerance for a non-degenerate elliptical orbit

#### Scenario: Conversion is deferred until requested
- **WHEN** an orbital state is constructed in one representation and only that representation is read back
- **THEN** no cross-representation conversion SHALL be performed

#### Scenario: Ported element-conversion test vectors
- **WHEN** the element-conversion tests run against vectors ported from Stone Soup's MIT-licensed orbital-state test suite
- **THEN** each conversion SHALL reproduce the expected values within the tolerance recorded alongside the vectors

### Requirement: Parameterized gravitational constant
The orbital state type SHALL carry its gravitational parameter as data supplied at construction. No Earth-specific mu SHALL be hardcoded inside the type or its conversion routines; Earth values SHALL be available only as explicit named constants callers opt into.

#### Scenario: Conversions honor the supplied mu
- **WHEN** two orbital states are constructed from the same Cartesian elements but with different gravitational parameters (e.g., Earth and lunar mu)
- **THEN** their derived Keplerian elements (e.g., semi-major axis via the vis-viva relation) SHALL differ according to the supplied mu

### Requirement: Frame-disciplined orbital state
Every orbital state SHALL carry an explicit reference-frame tag drawn from the shared frame vocabulary (at minimum TEME, GCRF, and ITRF alongside the existing tags), and the API SHALL prevent silent mixing of frames: combining or comparing states with different frame tags SHALL fail with the frame-mismatch error — enforced in release builds, not only debug-asserted — never coerced implicitly. Explicit conversions between tagged frames SHALL exist via the reference-frame transform chain, producing a state whose tag reflects the target frame.

#### Scenario: Mixed-frame operation rejected
- **WHEN** an operation combines a GCRF-tagged state with a TEME-tagged state without an explicit frame conversion
- **THEN** the operation SHALL fail with the frame-mismatch error in both debug and release builds rather than produce a numerically silent wrong answer

#### Scenario: Frame tag survives representation conversion
- **WHEN** a frame-tagged orbital state is converted between element representations
- **THEN** the resulting state SHALL carry the same frame tag as the original

#### Scenario: Explicit conversion changes tag and elements together
- **WHEN** a TEME-tagged orbital state is explicitly converted to GCRF
- **THEN** the result SHALL carry the GCRF tag and its Cartesian elements SHALL be the transform of the originals — tag and numbers never disagree

### Requirement: Time-scale-aware epoch on orbital states
The orbital state type SHALL carry its epoch as the time-scale-aware epoch type rather than a bare Julian-date float, and frame conversions SHALL evaluate their rotations at that epoch.

#### Scenario: Epoch survives conversions with its scale intact
- **WHEN** an orbital state constructed with a UTC epoch is converted between representations and frames
- **THEN** the resulting state's epoch SHALL be identical to the original, including its time scale

#### Scenario: Frame conversion uses the state's own epoch
- **WHEN** the same TEME elements are converted to GCRF under two states differing only in epoch by one year
- **THEN** the two GCRF results SHALL differ, reflecting epoch-dependent precession/nutation evaluated at each state's own epoch
