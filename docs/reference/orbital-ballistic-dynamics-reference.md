# Orbital and Ballistic Dynamics for Filter Prediction and Truth Generation

### A Mathematical Reference for the Shared Force Math in `thresh_core::orbital`

> **Covers:** two-body and J2 gravity from the potential, the piecewise-exponential
> atmosphere, ballistic-coefficient drag with a co-rotating atmosphere, the analytic
> gravity-gradient Jacobian, interleaved-vs-blocked state layouts, gravity-turn boost
> equations, RK4 error order, and process-noise discretization.
>
> **Audience:** contributors to the `KeplerJ2` / `BallisticReentry` motion models,
> the phased ballistic truth generator, and their tests. Every equation is re-derived
> from first principles so the doc stands alone (repo reference-doc pattern); Vallado,
> Montenbruck & Gill, and the reentry-tracking literature are cited as sources of the
> standard results.
>
> All state vectors in this doc are ECI under the repo's GMST-only rotation convention
> (`crates/thresh-core/src/eci.rs`; ECI/TEME are deliberately conflated until the
> `astro-time-and-frames` change). SI units throughout: metres, seconds, kilograms, radians.

## Key Equations Summary

### Gravity
- Two-body: a = −μ r / ‖r‖³
- J2 (total = two-body + perturbation, the shape `j2_acceleration` returns):
  - a_x = −μx/r³ + (3/2)J2·μ·R_e²/r⁵ · x·(5z²/r² − 1)
  - a_y = −μy/r³ + (3/2)J2·μ·R_e²/r⁵ · y·(5z²/r² − 1)
  - a_z = −μz/r³ + (3/2)J2·μ·R_e²/r⁵ · z·(5z²/r² − 3)
- Vis-viva: v² = μ(2/r − 1/a); orbital energy ε = v²/2 − μ/r = −μ/(2a)
- J2 secular rates: Ω̇ = −(3/2)·n·J2·(R_e/p)²·cos i,  ω̇ = (3/4)·n·J2·(R_e/p)²·(5cos²i − 1)

### Atmosphere and drag
- Piecewise exponential: ρ(h) = ρ₀·exp(−(h − h₀)/H) per table bracket (Vallado Table 8-4 lineage)
- Drag: a_D = −ρ‖v_rel‖·v_rel / (2β),  β = m/(C_d·A),  v_rel = v − ω_⊕ × r
- Drag magnitude: ‖a_D‖ = ρ(h)·‖v_rel‖² / (2β)

### Jacobian (gravity gradient, see the dedicated section)
- Two-body: G₂ᵦ = (μ/r⁵)(3 r rᵀ − r² I)
- J2: G_J2 = (3/2)J2·μ·R_e²/r⁵ · M(x, y, z) — full matrix below; G = G₂ᵦ + G_J2 is symmetric and traceless

### Integration and noise
- RK4: local error O(h⁵) per step, global O(h⁴); sub-stepped predict with h = dt/n, n = ⌈dt/max_step_s⌉
- CWNA per-axis block: Q = σ_a²·[[dt³/3, dt²/2], [dt²/2, dt]]
- DWNA per-axis block (the `ConstantVelocity` variant): Q = σ_a²·[[dt⁴/4, dt³/2], [dt³/2, dt²]]
- β random walk: Q_ββ = σ_β²·dt

## Notation

| Symbol | Meaning | Units | Code symbol |
|---|---|---|---|
| r = (x, y, z) | ECI position vector | m | `pos: Vector3<f64>` |
| v | ECI velocity vector | m/s | `vel: Vector3<f64>` |
| r = ‖r‖ | geocentric radius | m | — |
| μ | gravitational parameter GM | m³/s² | `GravityModel::mu` |
| J2 | second zonal harmonic coefficient | — | `GravityModel::j2` |
| R_e | central-body equatorial radius | m | `GravityModel::equatorial_radius` |
| ω_⊕ | Earth rotation rate, 7.292115e-5 | rad/s | `thresh_core::eci::EARTH_ROTATION_RATE` |
| φ | geocentric latitude, sin φ = z/r | rad | — |
| u | z²/r² (so 5u = the code's `z2_r2`) | — | `z2_r2 / 5.0` |
| h | geometric altitude ‖r‖ − R_e | m | `alt` in `drag_acceleration` |
| ρ(h) | atmospheric density | kg/m³ | `atmosphere_density` |
| h₀, ρ₀, H | bracket base altitude, base density, scale height | km, kg/m³, km | `ATMOSPHERE_TABLE` rows |
| β | ballistic coefficient m/(C_d·A) | kg/m² | state component 6 of `BallisticReentry` |
| 1/β | inverse ballistic coefficient C_d·A/m | m²/kg | `inv_beta` |
| v_rel | atmosphere-relative velocity v − ω_⊕ × r | m/s | `v_rel` |
| a_T | thrust acceleration magnitude (boost) | m/s² | `BallisticProfile::thrust_accel` (Decision 5) |
| γ | flight-path angle above local horizontal | rad | — |
| G | gravity-gradient matrix ∂a/∂r | 1/s² | — (validation anchor for `numeric_jacobian`) |
| A(x) | continuous-time dynamics Jacobian ∂f/∂x | mixed | — |
| F, Φ | discrete state-transition matrix / STM | mixed | `MotionModel::jacobian` output |
| Q | discrete process-noise covariance | mixed | `MotionModel::process_noise` output |
| σ_a | white-noise-acceleration PSD | m/s²·(per √Hz) | `sigma_accel` (Decisions 3, 4) |
| σ_β | β random-walk intensity | kg/m² per √s | `sigma_beta` (Decision 4) |
| dt, h_step | predict interval, RK4 sub-step | s | `dt`, `max_step_s` |
| n | mean motion √(μ/a³) | rad/s | — |
| a, e, i, Ω, ω, ν | Keplerian elements (sma, ecc, inc, RAAN, argp, true anomaly) | m, —, rad | `to_keplerian` tuple |
| p | semi-latus rectum a(1 − e²) | m | — |
| ε_mach | f64 machine epsilon ≈ 2.22e-16 | — | `f64::EPSILON` |

Earth values (WGS-84, `GravityModel::EARTH_WGS84` in
`crates/thresh-core/src/orbital/gravity.rs`): μ = 3.986004418e14 m³/s²,
J2 = 1.08263e-3, R_e = 6 378 137 m.

## Two-Body Acceleration from the Potential

Take the gravitational potential per unit mass with the astrodynamics sign
convention a = ∇U (Vallado 4th ed., Ch. 1):

> U₂ᵦ(r) = μ / r

The gradient of 1/r follows from ∂r/∂x = x/r:

> ∂(1/r)/∂x = −(1/r²)·(x/r) = −x/r³, and likewise for y, z

so

> **a₂ᵦ = ∇U₂ᵦ = −μ r / r³**,  componentwise a_i = −(μ/r³)·r_i

This is exactly `two_body_acceleration` (`crates/thresh-core/src/orbital/gravity.rs`):
the code computes `mu_r3 = μ/(r²·r)` once and negates each component. The acceleration
points from the field point toward the central body with magnitude μ/r²
(unit test `two_body_points_inward_with_mu_over_r2_magnitude`).

Two standard corollaries used by the test suites:

- **Vis-viva:** v² = μ(2/r − 1/a). Conservation of ε = v²/2 − μ/r = −μ/(2a) is the
  energy invariant asserted by `rk4_step_conserves_two_body_energy_over_one_orbit`
  (`crates/thresh-core/src/orbital/integrate.rs`), and vis-viva is what makes
  "identical Cartesian state, different μ ⇒ different sma" a valid test of
  μ-parameterization (tasks 2.4 / 3.3).
- **Circular speed:** v_circ = √(μ/r), used to construct closed test orbits.

Because μ is a field of `GravityModel` rather than a constant inside the math, the same
function serves Earth, lunar, or any central body — accelerations at fixed geometry
scale exactly with μ (unit test `two_body_uses_supplied_mu`).

## J2 Acceleration from the Zonal Potential

The axially-symmetric gravitational potential of an oblate body, truncated after the
second zonal harmonic (Vallado 4th ed., §8.6.1; Montenbruck & Gill Ch. 3):

> U(r, φ) = (μ/r)·[1 − J2·(R_e/r)²·P₂(sin φ)],  P₂(s) = (3s² − 1)/2

With geocentric latitude sin φ = z/r, the J2 disturbing potential is

> U_J2 = −(μ·J2·R_e²/2)·(3z²/r² − 1)/r³ = A·(3z²·r⁻⁵ − r⁻³),  A ≝ −μ·J2·R_e²/2

Differentiate using ∂(r⁻ⁿ)/∂x_i = −n·x_i·r⁻⁽ⁿ⁺²⁾. For x (and identically y), z² is
constant with respect to the differentiation, so **only the chain rule through r
contributes**:

> ∂U_J2/∂x = A·(3z²·(−5x·r⁻⁷) − (−3x·r⁻⁵)) = 3A·x·r⁻⁵·(1 − 5z²/r²)

Substituting A and defining k ≝ (3/2)·μ·J2·R_e² (so the code's `j2_coeff` = k/r⁵):

> **a_x^J2 = (k/r⁵)·x·(5z²/r² − 1)**,  **a_y^J2 = (k/r⁵)·y·(5z²/r² − 1)**

For z there is an **additional direct term** from the explicit z² in 3z²·r⁻⁵:

> ∂U_J2/∂z = A·(6z·r⁻⁵ + 3z²·(−5z·r⁻⁷) + 3z·r⁻⁵) = 3A·z·r⁻⁵·(3 − 5z²/r²)

> **a_z^J2 = (k/r⁵)·z·(5z²/r² − 3)**

### Term-for-term mapping to `j2_acceleration`

`j2_acceleration` (`crates/thresh-core/src/orbital/gravity.rs`) returns the **total**
gravitational acceleration — two-body plus the J2 perturbation, *not* the perturbation
alone (it is the moved body of the former `thresh_synth::orbital::acceleration_j2`;
consumers composing "two-body + J2" use `j2_acceleration` by itself, never summed with
`two_body_acceleration`):

| Code | Math |
|---|---|
| `mu_r3 = gravity.mu / (r2 * r)` | μ/r³ |
| `j2_coeff = 1.5 * gravity.j2 * gravity.mu * gravity.equatorial_radius² / r5` | k/r⁵ = (3/2)·J2·μ·R_e²/r⁵ |
| `z2_r2 = 5.0 * z * z / r2` | 5z²/r² = 5u |
| `-mu_r3 * x + j2_coeff * x * (z2_r2 - 1.0)` | −μx/r³ + (k/r⁵)·x·(5u − 1) |
| `-mu_r3 * y + j2_coeff * y * (z2_r2 - 1.0)` | −μy/r³ + (k/r⁵)·y·(5u − 1) |
| `-mu_r3 * z + j2_coeff * z * (z2_r2 - 3.0)` | −μz/r³ + (k/r⁵)·z·(5u − 3) |

### The (5z²/r² − 1) vs (5z²/r² − 3) asymmetry

The x and y components carry the factor (5z²/r² − 1) while z carries (5z²/r² − 3).
The extra −2 in the z factor is not a typo and not symmetry-breaking sloppiness: x and y
enter U_J2 **only through r**, whereas z also appears **explicitly** in the 3z²·r⁻⁵ term.
Its direct derivative contributes A·6z·r⁻⁵ = −2·(k/r⁵)·z on top of the chain-rule part
(k/r⁵)·z·(5u − 1), giving (5u − 1) − 2 = (5u − 3). Physical sanity checks (both are unit
tests in `gravity.rs`):

- **Equator** (z = 0): perturbation = −(3/2)·J2·μ·R_e²/r⁴ along the radial direction —
  the oblate bulge pulls equatorial satellites inward *more* strongly
  (`j2_perturbation_at_equator_matches_closed_form`).
- **Pole** (x = y = 0, z = r): 5u = 5 so the z factor is +2, perturbation
  = +3·J2·μ·R_e²/r⁴ radially *outward* — net gravity is weaker over the poles because
  the bulge mass sits far away at the equator
  (`j2_perturbation_at_pole_matches_closed_form`).

### Secular J2 rates (used by the propagation tests)

Averaging the J2 perturbation over one revolution gives the classical secular drift of the
node and perigee (Vallado 4th ed., Ch. 9 — general perturbations; results quoted, not
re-derived, since only tests consume them):

> Ω̇ = −(3/2)·n·J2·(R_e/p)²·cos i
> ω̇ = +(3/4)·n·J2·(R_e/p)²·(5cos²i − 1)

with n = √(μ/a³) and p = a(1 − e²). The nodal-regression rate Ω̇ is the analytic anchor
for the multi-revolution `KeplerJ2` golden test (task 3.5): a prograde LEO regresses
westward (cos i > 0 ⇒ Ω̇ < 0) by ≈ −5.8e-5 deg/s at ISS-like elements
(a ≈ 6778 km, i = 51.6°) — about −5 deg/day, easily resolved over a few orbits.

## Piecewise-Exponential Atmosphere

### Model and provenance

Density between table base altitudes follows a locally-fitted exponential:

> **ρ(h) = ρ₀ · exp(−(h − h₀)/H)**

where (h₀, ρ₀, H) are the base altitude, nominal density, and scale height of the bracket
containing h. The 28-row `ATMOSPHERE_TABLE` in
`crates/thresh-core/src/orbital/atmosphere.rs` (moved verbatim from
`thresh-synth/src/orbital.rs` per design Decision 1) is the exponential atmospheric model
of Vallado 4th ed., §8.6.2, **Table 8-4** — base altitudes 0, 25, 30, …, 900, 1000 km with
the familiar sea-level row (1.225 kg/m³, H = 7.249 km). Vallado's table in turn condenses
the *U.S. Standard Atmosphere, 1976* at low altitude and CIRA-72 above (with a moderate
exospheric temperature assumption for the 500–1000 km rows). Accuracy expectations follow
that lineage: this is a static, spherically-symmetric mean atmosphere — no diurnal bulge,
no solar-flux (F10.7) or geomagnetic response, so real thermospheric density can differ
from it by a factor of a few during solar maxima. That model error is why the filter's
`sigma_accel` / `sigma_beta` process noise exists (see the process-noise section), and it
is far larger than the sub-1% error from using geometric altitude h = ‖r‖ − R_e over an
oblate Earth (design Decision 4).

### Lookup semantics in `atmosphere_density`

The implementation defines behavior at and beyond the table edges — tests and consumers
rely on all four properties:

1. **h < 0:** clamps to the sea-level density 1.225 kg/m³ (no extrapolation blow-up when
   a truth trajectory or a filter sigma point dips below the spherical surface).
2. **h > 1000 km:** returns exactly 0. This hard ceiling is what makes drag vanish
   identically in the exo-atmospheric regime — the "Exoatmospheric limit" spec scenario
   and the β-unobservability argument both hinge on it.
3. **Bracket choice:** the last row with h₀ ≤ h (linear scan), so at a row boundary the
   exponential restarts at exp(0) = 1 and ρ equals the row's nominal density exactly
   (unit test `density_equals_table_value_at_bracket_base`).
4. **Within a bracket** density decreases monotonically since every H > 0
   (`density_decreases_with_altitude`).

Note the density is *not* continuous across bracket boundaries (adjacent exponentials
meet only approximately) nor at the 1000 km ceiling (3.019e-15 kg/m³ steps to 0); both
discontinuities are inherited from the tabulated model and are negligible against its
physical error, but they are worth knowing when differencing the dynamics numerically —
see the sub-step discussion under the Jacobian section.

## Ballistic-Coefficient Drag with a Co-Rotating Atmosphere

The drag acceleration on a body of mass m, drag coefficient C_d, and reference area A is

> a_D = −(1/2)·ρ·(C_d·A/m)·‖v_rel‖·v_rel = **−ρ·‖v_rel‖·v_rel / (2β)**

with the **ballistic coefficient** β ≝ m/(C_d·A) (kg/m²) collapsing the three body
parameters into the single quantity that the dynamics can actually observe: large β
(dense, streamlined — RVs, ~1000–5000 kg/m²) decelerates weakly; small β (light, blunt —
debris, deployed decoys) decelerates strongly. The shared implementation
`drag_acceleration(pos, vel, inv_beta)` (`crates/thresh-core/src/orbital/atmosphere.rs`)
is parameterized on the **inverse** ballistic coefficient `inv_beta` = C_d·A/m = 1/β
(m²/kg), which enters the force linearly; the synth shim
`thresh_synth::orbital::acceleration_drag` computes `inv_beta = cd * area_m2 / mass_kg`
from its `DragConfig`, and the `BallisticReentry` filter model passes 1/β from its
seventh state component.

**Co-rotating atmosphere.** The atmosphere rotates with the Earth, so the velocity that
matters for drag is relative to the local air, not inertial:

> **v_rel = v − ω_⊕ × r**,  ω_⊕ = (0, 0, ω_⊕) ⇒ ω_⊕ × r = (−ω_⊕·y, +ω_⊕·x, 0)

which is exactly the code's `v_atm` subtraction, with ω_⊕ =
`thresh_core::eci::EARTH_ROTATION_RATE` = 7.2921150e-5 rad/s. Working in ECI with a
co-rotating v_rel captures every rotating-Earth drag effect (the equivalent
ECEF formulation would instead add explicit Coriolis and centrifugal terms to *all*
accelerations — design Decision 4 rejected that trade). At LEO the correction is not
small: ‖ω_⊕ × r‖ ≈ 465 m/s at the equatorial surface, ~6% of orbital speed, and its
direction relative to v is what makes drag on prograde vs retrograde orbits asymmetric.

Magnitude identity asserted by the spec ("Drag deceleration matches the beta
parameterization") and by unit test `drag_magnitude_matches_half_rho_v2_over_beta`:

> ‖a_D‖ = ρ(h)·‖v_rel‖² / (2β), directed exactly opposite v_rel

Edge behavior of `drag_acceleration`: returns zero outside 0 ≤ h ≤ 1000 km and for
‖v_rel‖ < 1e-10 m/s (a body moving *with* the atmosphere feels no drag —
`drag_vanishes_when_moving_with_the_atmosphere`).

**Reentry dynamics (Decision 4).** The `BallisticReentry` model integrates

> r̈ = a_gravity(r) + a_D(r, v, 1/β),  β̇ = 0

where a_gravity is the full two-body + J2 field (`j2_acceleration`) and β is a state
component estimated by the filter — constant in the deterministic dynamics, driven only
by process noise (β random walk, below). Treating β as slowly varying rather than known
is the classic reentry-tracking formulation (Athans–Wishner–Bertolini 1968; Farrell 2008;
Ristic–Arulampalam–Gordon 2004): effective β genuinely drifts through the flight regime
(Mach-dependent C_d, attitude), and the filter refines it from observed deceleration.

## Analytic Jacobian of the two-body + J2 field

> This section is the validation anchor for the numeric Jacobian: the "Jacobian
> consistency" test (task 3.4) cites it by this heading. The matrix below was verified
> against central differences of the shipped `j2_acceleration` at equatorial, polar, and
> generic LEO points (relative error ~1e-9, at the finite-difference truncation floor).

The gravity-only dynamics ṙ = v, v̇ = a(r) have the continuous-time Jacobian

```
A(x) = ∂f/∂x = [ ∂ṙ/∂r  ∂ṙ/∂v ]   [ 0   I ]
               [ ∂v̇/∂r  ∂v̇/∂v ] = [ G   0 ]      (blocked [r | v] layout)
```

where **G = ∂a/∂r** is the 3×3 *gravity-gradient* matrix. G is symmetric (it is the
Hessian of the potential U) and traceless away from the origin (U is harmonic:
∇²U = 0 — Laplace's equation), and both properties are free assertions for any test.

### Two-body gradient

Differentiate a_i = −μ·x_i/r³ with ∂(r⁻³)/∂x_j = −3·x_j·r⁻⁵:

> ∂a_i/∂x_j = −μ·(δ_ij·r⁻³ − 3·x_i·x_j·r⁻⁵)

> **G₂ᵦ = (μ/r⁵)·(3·r rᵀ − r²·I)**

Symmetric by inspection; trace = (μ/r⁵)(3r² − 3r²) = 0. ✓

### J2 gradient

Write the J2 perturbation (derived above) as a_x = k·x·P, a_y = k·y·P, a_z = k·z·Q with

> k = (3/2)·μ·J2·R_e²,  P = 5z²·r⁻⁷ − r⁻⁵,  Q = 5z²·r⁻⁷ − 3r⁻⁵

The needed partials, again via ∂(r⁻ⁿ)/∂x_j = −n·x_j·r⁻⁽ⁿ⁺²⁾:

> ∂P/∂x = 5x·r⁻⁷·(1 − 7z²/r²)   ∂P/∂y = 5y·r⁻⁷·(1 − 7z²/r²)   ∂P/∂z = 5z·r⁻⁷·(3 − 7z²/r²)
> ∂Q/∂x = 5x·r⁻⁷·(3 − 7z²/r²)   ∂Q/∂y = 5y·r⁻⁷·(3 − 7z²/r²)   ∂Q/∂z = 5z·r⁻⁷·(5 − 7z²/r²)

Assembling ∂a_i^J2/∂x_j by the product rule and factoring out k/r⁵ (the code's
`j2_coeff`), with u ≝ z²/r²:

> **G_J2 = (3/2)·J2·μ·R_e²/r⁵ · M**, where M is symmetric with entries

```
M_xx = (5u − 1) + (5x²/r²)·(1 − 7u)
M_yy = (5u − 1) + (5y²/r²)·(1 − 7u)
M_zz = (5u − 3) + (5z²/r²)·(5 − 7u)
M_xy = M_yx = (5xy/r²)·(1 − 7u)
M_xz = M_zx = (5xz/r²)·(3 − 7u)
M_yz = M_zy = (5yz/r²)·(3 − 7u)
```

The off-diagonal symmetry M_xz = M_zx (i.e. ∂a_x/∂z = ∂a_z/∂x) is a nontrivial check —
the two entries come from *different* product-rule splits (x·∂P/∂z vs z·∂Q/∂x) and agree
only because G_J2 is a Hessian. Trace check:
tr M = [2(5u−1) + (5u−3)] + (5/r²)[(x²+y²)(1−7u) + z²(5−7u)]
= (15u − 5) + 5(1 − 3u) = 0. ✓ (Same asymmetry story as the acceleration: the z-row
factors (3−7u), (5−7u) differ from the x/y factors because z appears explicitly in P
and Q, not only through r.)

The total gravity gradient consumed by the consistency test is

> **G = G₂ᵦ + G_J2**

### Closed-form spot checks

At two axis-aligned points G is diagonal, giving hand-checkable values (k = (3/2)·μ·J2·R_e²):

| Point | G (diagonal) |
|---|---|
| Equatorial, r = (r, 0, 0), u = 0 | diag( 2μ/r³ + 4k/r⁵,  −μ/r³ − k/r⁵,  −μ/r³ − 3k/r⁵ ) |
| Polar, r = (0, 0, r), u = 1 | diag( −μ/r³ + 4k/r⁵,  −μ/r³ + 4k/r⁵,  2μ/r³ − 8k/r⁵ ) |

Both are traceless. The 2μ/r³ tension along the radial direction against −μ/r³
compression transverse is the classical tidal structure; the J2 correction at LEO is
~3·J2·(R_e/r)² ≈ 3e-3 of it. Order of magnitude at r = R_e + 500 km:
μ/r³ ≈ 1.22e-6 s⁻².

### Embedding in the interleaved 6D state

The filter state is interleaved, s = [x, vx, y, vy, z, vz] (see the next section), so the
blocked A above must be permuted. Indexing 0-based with position component i at slot 2i
and velocity component i at slot 2i+1, the interleaved continuous-time Jacobian is

```
            x     vx    y     vy    z     vz
        ┌ 0     1     0     0     0     0  ┐   x
        │ G₀₀   0     G₀₁   0     G₀₂   0  │   vx
A_int = │ 0     0     0     1     0     0  │   y
        │ G₁₀   0     G₁₁   0     G₁₂   0  │   vy
        │ 0     0     0     0     0     1  │   z
        └ G₂₀   0     G₂₁   0     G₂₂   0  ┘   vz
```

i.e. A_int[2i, 2i+1] = 1 and A_int[2i+1, 2j] = G_ij, all other entries zero.

### How the numeric Jacobian is cross-checked (task 3.4)

The shipped `MotionModel::jacobian` for `KeplerJ2` / `BallisticReentry` is **numeric**
(design Decision 3): central differences of the exact sub-stepped RK4 `predict`, with
per-column step h_i = ε·max(|x_i|, s_i), ε = ∛ε_mach ≈ 6.06e-6, floor scales s_i (1 m
position, 1e-3 m/s velocity, and max(|β|, 100) for the β column). The cube-root choice
is the standard optimum for central differences: truncation error O(h²) balances rounding
error O(ε_mach/h) at h ~ ε_mach^(1/3).

The variational (state-transition-matrix) equation for the flow map φ_dt of ẋ = f(x) is
Φ̇ = A(x(t))·Φ, Φ(0) = I, whose short-time expansion is

> **Φ(dt) = I + A(x₀)·dt + O(dt²)**

(the O(dt²) term collects both A² and the along-trajectory drift Ȧ). The consistency
test therefore evaluates the numeric Jacobian at a representative LEO state with a small
dt, builds A_int from the analytic G of this section, and asserts

> ‖F_numeric(dt) − (I + A_int·dt)‖ = O(dt²)

element-wise with a tolerance budgeted for the dt² term — analytic math validates,
numeric math ships. Halving dt should shrink the residual ~4×, a cheap second assertion
that the error really is second-order.

**Drag terms are deliberately not part of this anchor.** For `BallisticReentry` the
dynamics add ∂a_D/∂r (density gradient ∂ρ/∂h = −ρ/H along r̂, plus the −ω_⊕× coupling
inside v_rel), ∂a_D/∂v = −(ρ/2β)·(‖v_rel‖·I + v_rel·v_relᵀ/‖v_rel‖), and

> ∂a_D/∂β = −a_D/β

(from a_D ∝ 1/β — well-conditioned, the basis for the β column scale). The gravity-only
G above remains the exact analytic reference in the exo-atmospheric regime where drag
vanishes; inside the atmosphere the density gradient is steep (H as small as 5.4 km near
90 km altitude ⇒ e-folding per ~5 km of altitude), which is why the reentry model's
default sub-step is `max_step_s = 1 s` and why the Jacobian unit tests include a 100 km
state (design Risks).

## State Layout: Interleaved vs Blocked

Astrodynamics texts write the state blocked, y = [x, y, z, vx, vy, vz]. The repo's
filter convention is **interleaved**:

> s = [x, vx, y, vy, z, vz]

matching `ConstantVelocity` (`crates/thresh-filter/src/models/cv.rs`), the IMM common
6D space (`crates/thresh-filter/src/imm.rs`), and the tracker's 3×6 position observation
matrices. The two layouts are related by the permutation y = Π·s:

```
        ┌ 1 0 0 0 0 0 ┐        (rows: blocked x, y, z, vx, vy, vz;
        │ 0 0 1 0 0 0 │         cols: interleaved x, vx, y, vy, z, vz)
    Π = │ 0 0 0 0 1 0 │
        │ 0 1 0 0 0 0 │        position i:  interleaved slot 2i   → blocked slot i
        │ 0 0 0 1 0 0 │        velocity i:  interleaved slot 2i+1 → blocked slot 3+i
        └ 0 0 0 0 0 1 ┘
```

Π is orthogonal (Π⁻¹ = Πᵀ), so any blocked-layout matrix from the literature converts as

> A_int = Πᵀ·A_blk·Π,  F_int = Πᵀ·F_blk·Π,  Q_int = Πᵀ·Q_blk·Π

and vectors as s = Πᵀ·y. The 7D reentry state appends β after the interleaved six:
s₇ = [x, vx, y, vy, z, vz, β] — chosen so the position-only H matrices are the 6D ones
padded with a zero column and the IMM `StateMapping` between 6D and 7D is a plain
drop/append of the last row and column (the `CaMapping` pattern in `imm.rs`;
`Reentry7Mapping`, design Decision 4 / task 3.9).

## Gravity-Turn Boost Dynamics

The phased truth generator (design Decision 5, `thresh_synth::ballistic`,
`BallisticProfile`) integrates boost in ECI with a thrust of **constant acceleration
magnitude** a_T (no mass depletion or staging — out of scope) in three sub-phases:

1. **Vertical rise** (t < `pitch_over_s`): thrust along the local vertical r̂,

   > r̈ = a_gravity(r) + a_T·r̂

2. **Pitch-over** (instantaneous, at t = `pitch_over_s`): the velocity/thrust direction
   is rotated by `pitch_kick_rad` downrange (toward the launch azimuth). This small kick
   (a few degrees) seeds the turn; without it a vertical rocket stays vertical forever.

3. **Zero-lift gravity turn** (until t = `burn_time_s`): thrust aligned with the
   **atmosphere-relative** velocity,

   > r̈ = a_gravity(r) + a_T·v_rel/‖v_rel‖,  v_rel = v − ω_⊕ × r

   Aligning with v_rel (not inertial v) is the textbook zero-angle-of-attack condition —
   the vehicle flies at zero aerodynamic incidence, so gravity alone rotates the velocity
   vector. Design Decision 5 also notes the practical reason: inertial-v alignment would
   make the turn geometry epoch-dependent in ECI.

The classical planar form (Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed.,
rocket-vehicle-dynamics chapter) makes the "gravity turns the trajectory" mechanism
explicit. In the flight plane
with speed v = ‖v_rel‖ and flight-path angle γ above the local horizontal:

> v̇ = a_T − g·sin γ  (thrust minus the along-track gravity component)
> v·γ̇ = −(g − v²/r)·cos γ  (gravity bends the path down; v²/r is the centrifugal relief)

At liftoff γ = π/2 and γ̇ = 0 (cos γ = 0): the equations *cannot* leave the vertical —
hence the pitch kick. As v grows, the v²/r term erodes the turn rate until at orbital
speed (v² = g·r) the path stops bending: the turn "ends" naturally. The ECI vector form
above is what the code integrates (the planar form drops the J2 and ω_⊕ details); both
describe the same dynamics.

After burnout the acceleration closure drops to exactly `j2_acceleration` (midcourse —
identical to `thresh_synth::orbital::propagate` with `include_j2: true, drag: None`), and
below 100 km descending it adds `drag_acceleration(r, v, 1/β)` — the same closure the
`BallisticReentry` filter model integrates, which is what makes the truth-vs-filter
identity tests (tasks 4.7 / 3.5) meaningful.

## RK4 Integration and Error Order

The shared integrator (`rk4_step` / `rk4_stage`,
`crates/thresh-core/src/orbital/integrate.rs`) is the classical explicit 4th-order
Runge–Kutta scheme applied to the coupled first-order system ṙ = v, v̇ = a(r, v), with
the acceleration supplied as a closure. Each stage returns the pair (k_r, k_v) — the
position derivative (stage velocity) and velocity derivative (stage acceleration)
evaluated at the stage point:

> k₁ at the current state; k₂, k₃ at half-step points advanced by the previous stage's
> slopes; k₄ at the full step; combined with weights (1, 2, 2, 1)/6.

Standard order results (Hairer–Nørsett–Wanner, Ch. II):

- **Local truncation error O(h⁵)** per step of size h; **global error O(h⁴)** over a
  fixed interval (one order lost to the ~1/h step count).
- Sub-stepped filter predict (Decisions 3, 4): n = ⌈dt/max_step_s⌉ equal sub-steps of
  h = dt/n ≤ max_step_s, so the error over one predict interval scales as
  n·O(h⁵) = O(dt·h⁴) — tightening `max_step_s` by 2× buys 16× accuracy at 2× cost.
- With the O(h⁵) local error, solutions polynomial in time up to degree 4 are
  reproduced exactly; in particular constant-acceleration motion (quadratic position,
  linear velocity) is reproduced to rounding (unit test
  `rk4_step_is_exact_for_constant_acceleration`) — the degenerate case the kinematic
  CV/CA models also cover.
- RK4 is **not symplectic** — two-body orbital energy drifts secularly, but slowly:
  the regression test `rk4_step_conserves_two_body_energy_over_one_orbit` bounds the
  relative energy drift below 1e-8 over one LEO orbit at h = 10 s, the same regime the
  synth ISS test validates over a full day. Defaults: `max_step_s = 10 s` (orbital,
  smooth field) and `1 s` (reentry, steep density gradients — see the Jacobian section).

## Process-Noise Discretization

The deterministic dynamics above omit real forces (drag on the orbital model, lift and
attitude effects on the reentry model, density model error, small maneuvers). Filters
absorb the omission as process noise. Both new models use per-axis blocks in the
interleaved layout, exactly the block placement of `ConstantVelocity::process_noise`.

### Continuous white-noise acceleration (CWNA)

Model each axis as a double integrator driven by continuous white acceleration noise
(Bar-Shalom–Li–Kirubarajan, Ch. 6):

> d/dt [x; v] = [[0, 1], [0, 0]]·[x; v] + [0; 1]·w(t),  E[w(t)·w(τ)] = σ_a²·δ(t − τ)

The per-axis STM is Φ(τ) = [[1, τ], [0, 1]], and the discretized noise covariance is

> Q(dt) = ∫₀^dt Φ(τ)·B·σ_a²·Bᵀ·Φ(τ)ᵀ dτ,  B = [0; 1]

With Φ(τ)·B = [τ; 1] the integrand is σ_a²·[[τ², τ], [τ, 1]], so

> **Q = σ_a² · [[dt³/3, dt²/2], [dt²/2, dt]]**  (per axis, [position, velocity] block)

This is the block Decisions 3 and 4 specify for `KeplerJ2` and `BallisticReentry`
(σ_a = `sigma_accel`; order 1e-4…1e-3 m/s² for LEO where the omitted forces are drag,
SRP, and higher harmonics — several m/s² for reentry where they are lift and attitude
oscillation). Strictly, Q should be propagated through the full linearized dynamics
(Φ built from A(x) of the Jacobian section), but over ≤ 10 s sub-steps the
double-integrator approximation errs far below the tuning uncertainty in σ_a itself; a
dynamics-shaped (along-track/cross-track) Q is deferred to orbit-propagation-fidelity.

### Discrete white-noise acceleration (DWNA) — the `ConstantVelocity` variant

The existing `ConstantVelocity::process_noise` (`crates/thresh-filter/src/models/cv.rs`)
uses the *piecewise-constant* variant: an acceleration a_k ~ N(0, σ_a²) held constant
across each step enters through Γ = [dt²/2; dt], giving the rank-one block

> Q = Γ·σ_a²·Γᵀ = σ_a² · [[dt⁴/4, dt³/2], [dt³/2, dt²]]

The two parameterizations differ in σ_a's units (PSD vs per-step standard deviation) and
in dt scaling, and coincide in effect only near dt ≈ 1 s; they are not interchangeable
constants. The doc records the distinction so nobody "fixes" one to match the other: CV
keeps DWNA (its shipped, tuned behavior), the new physical models use CWNA because their
σ_a genuinely represents a continuous unmodeled-force density.

### β random walk

The reentry model's seventh component has trivial deterministic dynamics (β̇ = 0) driven
by an independent scalar white noise of intensity σ_β² (units kg/m² per √s):

> **Q_ββ = σ_β² · dt**,  cross-terms to the kinematic block zero

The nonzero σ_β is essential, not cosmetic (Farrell 2008): effective β varies through
the flight regime, and — more subtly — β is **unobservable exo-atmospherically**
(ρ = 0 ⇒ a_D = 0 ⇒ ∂a/∂β = 0: measurements carry no β information). Without process
noise the β variance would freeze at whatever the filter last inferred; the random walk
correctly re-inflates it during midcourse so the filter re-learns β when the vehicle
hits sensible atmosphere. Variance growth σ_β²·T over a full midcourse should stay
within the initial prior's order of magnitude — the head's tuning comment records the
sizing. Positivity is enforced by clamping β to `beta_floor` (default 10 kg/m²) inside
`predict`, removing the divide-by-zero/negative-β failure mode.

### Log-β: recorded future alternative

Estimating b = ln β instead of β would guarantee positivity structurally (β = eᵇ > 0),
give a well-scaled sensitivity ∂a_D/∂b = β·∂a_D/∂β = −a_D, and make the random walk
multiplicative — arguably the right geometry for a prior spanning [50, 10000] kg/m².
Direct β ships instead (design Decision 4) because it is what operators and the truth
generator specify, keeps track outputs physically readable, and the floor already
removes the failure mode. **Revisit trigger:** if filter consistency tests show a
skewed β posterior (NEES/NIS failures traceable to the β marginal), switch the state
component to ln β — the `Reentry7Mapping` boundary is the only seam that changes.

## Code-Symbol Map

Shipped symbols (verified in-tree):

| Math | Code symbol | Location |
|---|---|---|
| μ, J2, R_e | `GravityModel { mu, j2, equatorial_radius }` | `crates/thresh-core/src/orbital/gravity.rs` |
| WGS-84 Earth values | `GravityModel::EARTH_WGS84` | `crates/thresh-core/src/orbital/gravity.rs` |
| a₂ᵦ = −μr/r³ | `two_body_acceleration(pos, gravity)` | `crates/thresh-core/src/orbital/gravity.rs` |
| a₂ᵦ + a_J2 (total) | `j2_acceleration(pos, gravity)` | `crates/thresh-core/src/orbital/gravity.rs` |
| (h₀, ρ₀, H) rows | `ATMOSPHERE_TABLE` | `crates/thresh-core/src/orbital/atmosphere.rs` |
| ρ(h) | `atmosphere_density(alt_m)` | `crates/thresh-core/src/orbital/atmosphere.rs` |
| a_D(r, v, 1/β) | `drag_acceleration(pos, vel, inv_beta)` | `crates/thresh-core/src/orbital/atmosphere.rs` |
| ω_⊕ | `EARTH_ROTATION_RATE` | `crates/thresh-core/src/eci.rs` |
| GMST(jd) | `gmst_from_jd` | `crates/thresh-core/src/eci.rs` |
| one RK4 stage (k_r, k_v) | `rk4_stage` | `crates/thresh-core/src/orbital/integrate.rs` |
| one RK4 step | `rk4_step` | `crates/thresh-core/src/orbital/integrate.rs` |
| [f64; 3] shim of a₂ᵦ + a_J2 | `acceleration_j2(pos)` | `crates/thresh-synth/src/orbital.rs` |
| [f64; 3] shim of a_D (computes 1/β) | `acceleration_drag(pos, vel, cd, area_m2, mass_kg)` | `crates/thresh-synth/src/orbital.rs` |
| C_d, A, m | `DragConfig { cd, area_m2, mass_kg }` | `crates/thresh-synth/src/orbital.rs` |
| truth propagation | `propagate`, `PropagatorConfig` | `crates/thresh-synth/src/orbital.rs` |
| ENU projection | `orbital_to_enu`, `eci_to_enu` | `crates/thresh-synth/src/orbital.rs`, `crates/thresh-core/src/eci.rs` |
| interleaved 6D precedent | `ConstantVelocity` (state doc comment) | `crates/thresh-filter/src/models/cv.rs` |
| 6D common space / mappings | `StateMapping`, `CaMapping` | `crates/thresh-filter/src/imm.rs` |
| 6D Kepler+J2 model, σ_a, sub-step | `KeplerJ2 { gravity, max_step_s, sigma_accel }` | `crates/thresh-filter/src/models/kepler_j2.rs` |
| 7D reentry model, σ_β, β floor | `BallisticReentry { gravity, max_step_s, sigma_accel, sigma_beta, beta_floor }` | `crates/thresh-filter/src/models/ballistic_reentry.rs` |
| central-difference F | `numeric_jacobian(f, x, dt, scales)` | `crates/thresh-filter/src/numeric.rs` |
| 6D↔7D IMM mapping | `Reentry7Mapping` | `crates/thresh-filter/src/imm.rs` |

Symbols introduced elsewhere in this change (names fixed by design Decision 5; see
`openspec/changes/orbital-ballistic-filter-models/design.md`):

| Math | Design symbol | Planned location |
|---|---|---|
| a_T, burn/pitch parameters, β | `BallisticProfile` | `crates/thresh-synth/src/ballistic.rs` |

## References

- **Vallado, D. A.**, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
  Microcosm Press / Springer, 2013. Ch. 1 (two-body problem from the potential,
  vis-viva); Ch. 2 (Kepler's problem; Algorithms 9/10 with Examples 2-5/2-6 — the
  rv2coe/coe2rv golden vectors of tier 1); Ch. 8 (special perturbation techniques:
  §8.6.1 aspherical gravitational potential, §8.6.2 atmospheric drag with **Table 8-4**,
  the exponential atmosphere reproduced verbatim as `ATMOSPHERE_TABLE`); Ch. 9 (general
  perturbations; secular J2 rates).
- **Montenbruck, O., Gill, E.**, *Satellite Orbits: Models, Methods and Applications*,
  Springer, 2000. Ch. 3 (force model: geopotential, drag); Ch. 7 (linearization,
  variational equations — the gravity-gradient/STM machinery of the Jacobian section).
- **Bar-Shalom, Y., Li, X. R., Kirubarajan, T.**, *Estimation with Applications to
  Tracking and Navigation*, Wiley, 2001. Ch. 6 (kinematic models; continuous vs
  discrete white-noise-acceleration discretization).
- **Farrell, W. J., III**, "Interacting Multiple Model Filter for Tactical Ballistic
  Missile Tracking," *IEEE Transactions on Aerospace and Electronic Systems*, 44(2),
  2008. β-parameterized reentry dynamics with β process noise — the direct source of
  the Decision 4 model shape.
- **Athans, M., Wishner, R. P., Bertolini, A.**, "Suboptimal State Estimation for
  Continuous-Time Nonlinear Systems from Discrete Noisy Measurements," *IEEE
  Transactions on Automatic Control*, 13(5), 1968. The classic reentry β-estimation
  benchmark problem.
- **Ristic, B., Arulampalam, S., Gordon, N.**, *Beyond the Kalman Filter: Particle
  Filters for Tracking Applications*, Artech House, 2004. Ballistic-object tracking
  treatment (reentry vehicle dynamics and β observability).
- **Curtis, H. D.**, *Orbital Mechanics for Engineering Students*, 3rd ed.,
  Butterworth-Heinemann, 2014. Rocket-vehicle-dynamics chapter (gravity-turn
  trajectory equations).
- **Hairer, E., Nørsett, S. P., Wanner, G.**, *Solving Ordinary Differential Equations
  I: Nonstiff Problems*, 2nd ed., Springer, 1993. Ch. II (Runge–Kutta order theory).
- *U.S. Standard Atmosphere, 1976* (NOAA/NASA/USAF) and *CIRA-72* (COSPAR International
  Reference Atmosphere) — the underlying data of Vallado Table 8-4.
