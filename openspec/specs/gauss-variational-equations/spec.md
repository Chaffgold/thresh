# gauss-variational-equations Specification

## Purpose
Gauss variational equations over the direct/prograde equinoctial element set `(a, h, k, p, q, λ)` — element rates driven by a time-aware inertial perturbing-acceleration closure rotated into the RSW frame, plus the analytic two-body secular rate on the mean longitude — with deterministic sub-stepped propagation over the integrator seam, cross-formulation equivalence against the Cartesian two-body+J2 path, and an explicitly documented nonsingular validity domain.

## Requirements

### Requirement: Equinoctial element rates under perturbing acceleration
The system SHALL compute Gauss variational element rates for the stored direct/prograde equinoctial set `(a, h, k, p, q, λ)` driven by a perturbing acceleration supplied as a time-aware inertial closure and rotated into the RSW frame internally, plus the two-body secular rate on the mean longitude. The equations SHALL be transcribed from an authoritative source fetched at implementation time and cited inline — never from memory — and the two-body term SHALL be analytic, not part of the perturbation input.

#### Scenario: Unperturbed motion moves only the mean longitude
- **WHEN** the element rates are evaluated with a zero perturbing acceleration
- **THEN** the rates of `a, h, k, p, q` SHALL be exactly zero and `dλ/dt` SHALL equal the two-body mean motion `√(μ/a³)`

#### Scenario: Published spot values reproduced
- **WHEN** element rates are evaluated at fetched published reference conditions (state and perturbation from the cited source's worked values, where fetchable)
- **THEN** each rate SHALL match the published value within the documented tolerance

#### Scenario: RSW rotation is orthonormal and right-handed
- **WHEN** the RSW basis is constructed from any non-degenerate position/velocity pair
- **THEN** the basis SHALL be orthonormal with W parallel to the angular-momentum vector and S completing the right-handed triad

### Requirement: Element-space propagation over the integrator seam
The system SHALL propagate equinoctial elements over an arc by integrating the element rates with the sub-stepped fixed-step integrator (configurable maximum step), and integration SHALL be deterministic: identical inputs produce bitwise-identical elements. Step-halving SHALL exhibit the integrator's design order on a J2-perturbed orbit.

#### Scenario: Deterministic element propagation
- **WHEN** the same initial elements are propagated twice over the same arc with the same configuration
- **THEN** the resulting elements SHALL be bitwise identical

#### Scenario: Convergence under step halving
- **WHEN** a J2-perturbed LEO arc is propagated at a step size and at half that step size
- **THEN** the difference from a much-finer reference propagation SHALL shrink consistently with the integrator's order

### Requirement: Cross-formulation equivalence with the Cartesian path
Propagating the same J2-perturbed orbit through the Gauss variational equations in element space and through the existing Cartesian two-body+J2 path SHALL agree after conversion — in both directions (element-born and Cartesian-born initial states) and across at least three regimes (near-circular LEO, MEO, and an eccentric orbit) over multi-revolution arcs — within tolerances that are measured at the chosen step sizes and documented with the tests. The two formulations SHALL share only the underlying force closure, so agreement validates the element-rate transcription independently.

#### Scenario: LEO equivalence both directions
- **WHEN** a near-circular LEO state is propagated for several revolutions through both formulations, starting once from elements and once from Cartesian
- **THEN** the converted end states SHALL agree in position within the documented tolerance in both runs

#### Scenario: Eccentric-orbit equivalence through perigee
- **WHEN** an eccentric orbit (documented e in the 0.3–0.7 band) is propagated through multiple perigee passages by both formulations
- **THEN** the converted end states SHALL agree within the documented tolerance (the fast-variable stress case)

### Requirement: Nonsingular domain is explicit
Every Gauss-variational helper SHALL document its validity domain: nonsingular at zero eccentricity and zero inclination, singular at i = π (retrograde), with the singular denominators identified at the code and exercised by tests at the near-singular ends of the valid domain (e ≈ 0, i ≈ 0).

#### Scenario: Near-circular near-equatorial evaluation is finite
- **WHEN** element rates are evaluated at e = 1e-8 and i = 1e-8 under a J2 perturbation
- **THEN** every rate SHALL be finite and the propagation SHALL remain finite over an arc
