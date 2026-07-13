# orbital-state-representation Specification (Delta)

## ADDED Requirements

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
Every orbital state SHALL carry an explicit reference-frame tag (e.g., ECI, TEME), and the API SHALL prevent silent mixing of frames: combining or comparing states with different frame tags SHALL be rejected or require an explicit conversion, never coerced implicitly.

#### Scenario: Mixed-frame operation rejected
- **WHEN** an operation combines an ECI-tagged state with a TEME-tagged state without an explicit frame conversion
- **THEN** the operation SHALL fail (at compile time or with a runtime error) rather than produce a numerically silent wrong answer

#### Scenario: Frame tag survives representation conversion
- **WHEN** a frame-tagged orbital state is converted between element representations
- **THEN** the resulting state SHALL carry the same frame tag as the original
