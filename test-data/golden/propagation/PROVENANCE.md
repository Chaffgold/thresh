# Orbit-Propagation Golden Fixture Provenance

Tier-3 golden fixtures for design Decision 7 of the
`orbit-propagation-fidelity` change: component goldens (analytic
Sun/Moon vs an independent authority, EGM96 spot accelerations from an
independent evaluation, the published Harris-Priester table) and three
trajectory goldens integrated by an independently written Python force
stack. Consumed by default-feature envelope tests (no Python, no
network). Units are metres, m/s, seconds; epochs are ISO-8601 UTC
strings.

## Generation record

- Generator: `python/scripts/gen_propagation_fixtures.py` (manual-only, never CI)
- Regeneration command (from the repository root, then commit the output):

```sh
uv run python/scripts/gen_propagation_fixtures.py
```

- The run fetches the EGM96 coefficient file once and verifies its
  SHA-256 against the pinned digest before trusting the embedded
  subset (set `THRESH_EGM96_GFC` to a local copy to skip the network;
  the digest check still runs). Repeated runs at the same package
  versions are bit-identical.
- Toolchain at generation time:

  - python 3.12.13
  - numpy 2.2.6
  - scipy 1.15.3
  - astropy 7.0.1
  - pyerfa 2.0.1.5
  - mpmath 1.3.0
  - astropy-iers-data 0.2026.7.13.0.54.2

## Frames and time conventions (both stacks)

Zero-EOP IAU-76/FK5 exactly matching the Rust default
`Iau76Fk5Provider`: dUT1 = 0 (UT1 = UTC), no polar motion, and
GCRF -> ITRF = R3(GAST) . N(IAU-1980) . P(IAU-76) with
GAST = GMST-1982 + EqEq-1994 (generator side: erfa
pmat76/nutm80/gmst82/eqeq94 — the same reduction the frames golden
fixtures validated against astropy). dAT is constant across every arc
(no leap second since 2017; asserted at generation time). The ITRF
velocity transport term uses omega_earth = 7.2921150e-5 rad/s
(`thresh_core::eci::EARTH_ROTATION_RATE`).

## Sources (all fetched this generation session)

- **icgem-egm96**: ICGEM (GFZ) EGM96 gfc file, http://icgem.gfz-potsdam.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc — SHA-256 5247a9e9c316dd2c8f8fd491d53be0e163cb5cb3676b021754240cc9e44cb43b. Header: earth_gravity_constant 0.3986004415E+15 m^3/s^2, radius 0.6378136300E+07 m, tide_system tide_free. Citation in file header: Lemoine et al., NASA/TP-1998-206861.
- **nga-egm96**: NGA EGM96 spherical-harmonic package, https://earth-info.nga.mil/php/download.php?file=egm-96spherical — inner file `EGM96` SHA-256 1e5e6c30343989b8e2eda0bb96bde06ef05981eec69934d6e44aace4f0d6a9d5 (original 1996 NASA/NIMA distribution); cross-check source for the degree-2..12 subset (bit-equal, all 88 rows).
- **orekit-hp**: CS-SI/Orekit HarrisPriester.java (table ALT_RHO and the density formula), commit 77cfa41667457f0db879a0fc7e892e7248c1e516, https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/models/earth/atmosphere/HarrisPriester.java — cites Montenbruck & Gill, 'Satellite Orbits', Springer 2005.
- **satkit-hp**: JuliaSpace/SatelliteToolboxAtmosphericModels.jl constants.jl (_HARRIS_PRIESTER_ALT_RHO), commit 268c2645923d5a1c4206785c9c98f1ff2eccd0e4, https://github.com/JuliaSpace/SatelliteToolboxAtmosphericModels.jl/blob/main/src/harrispriester/constants.jl — independent transcription of the same table.
- **meeus-sun**: Meeus, 'Astronomical Algorithms', 2nd ed., Ch. 25 'Solar Coordinates', low-accuracy method (Eqs. 25.2-25.5, ~0.01 deg class) — the Sun series the Rust ephemerides implement. This generator mirrors the crates/thresh-core/src/orbital/ephemeris.rs Meeus Ch. 25 transcription coefficient for coefficient (the Rust file transcribed them from the book-faithful soniakeys/meeus v3 Go implementation, solar/solar.go, and pins Meeus Example 25.a as a spot test — re-asserted here at generation time).
- **vallado-sun**: CelesTrak/fundamentals-of-astrodynamics, software/matlab/sun.m (Vallado 2022, p. 285, Algorithm 29 — low-precision Sun, ~0.01 deg), commit 4555fdc4f611817cf55a8b3158860bdab0b552ee, https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/matlab/sun.m — reference-only: this is NOT the Sun series the Rust ephemerides implement (that is meeus-sun); its deltas are recorded as informational and enter no tolerance.
- **vallado-moon**: CelesTrak/fundamentals-of-astrodynamics, software/matlab/moon.m (Vallado 2022, p. 294, Algorithm 31 — truncated ELP-lineage Moon), commit 102fe0182ea0c20d02fdf18b285faa6869067feb, https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/matlab/moon.m
- **jpl-astro-par**: JPL SSD Astrodynamic Parameters, https://ssd.jpl.nasa.gov/astro_par.html — GM_Sun = 1.32712440041279419e20 m^3/s^2, GM_Moon = 4902.800118 km^3/s^2 (DE440), au = 149597870700 m (IAU 2012 B1), c = 299792458 m/s (CODATA 2018).
- **iau2015b3**: IAU 2015 Resolution B3 (Prsa et al. 2016, AJ 152:41; arXiv:1510.07674) — nominal total solar irradiance 1361 W/m^2.
- **legendre-recursion**: Fully-normalized associated Legendre recursion: seeds P00=1, P11=sqrt(3)*u; sectoral Pmm = u*sqrt((2m+1)/(2m))*P(m-1,m-1); non-sectoral Pnm = a_nm*t*P(n-1,m) - b_nm*P(n-2,m) with a_nm = sqrt((2n-1)(2n+1)/((n-m)(n+m))), b_nm = sqrt((2n+1)(n+m-1)(n-m-1)/((n-m)(n+m)(2n-3))); t = sin(lat), u = cos(lat); normalization sqrt(k(2n+1)(n-m)!/(n+m)!), k=1 (m=0) else 2. Fetched from the MITgcm geoid cookbook sec. 3.7 (http://mitgcm.org/~mlosch/geoidcookbook/node11.html), which follows Holmes & Featherstone 2002 (J. Geodesy 76:279) and Torge 1991.
- **barthelmes**: Barthelmes, 'Definition of Functionals of the Geopotential ...' (GFZ STR09/02, revised Jan 2013), Eq. (108): W = (GM/r) * sum_l sum_m (R/r)^l * Plm(sin phi) * (Clm cos(m lambda) + Slm sin(m lambda)), spherical geocentric coordinates, fully normalized. Fetched from https://gfzpublic.gfz.de/pubman/item/item_104132_4/component/file_104133/0902-2.pdf

EGM96 coefficients were consumed from: http://icgem.gfz-potsdam.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc (local sha256-verified copy via THRESH_EGM96_GFC).

## `sun-moon-ephemeris.json`

Authority: astropy `get_body(..., ephemeris='builtin')` (ERFA epv00 /
moon98) — independent of the analytic series the Rust ephemerides
implement. The generator mirrors those series and measures them
against the authority per epoch:

- **Sun**: Meeus Ch. 25 low-accuracy series, mirrored
  coefficient-for-coefficient from
  `crates/thresh-core/src/orbital/ephemeris.rs` (which cites Meeus AA
  2nd ed. Eqs. 25.2-25.5 via the book-faithful soniakeys/meeus
  transcription), evaluated on **TT Julian centuries** — the Rust
  time argument — with the IAU-1980 (obl80) mean obliquity and the
  IAU-76 precession transpose: the exact ephemeris.rs chain, so the
  measured Sun error IS the Rust stack's error at these epochs.
  Meeus Example 25.a (the Rust unit test's spot values) is
  re-asserted at generation time.
- **Moon**: Vallado Algorithm 31 per the fetched moon.m,
  coefficient-identical to the Rust transcription, evaluated on the
  UTC Julian date with moon.m's truncated obliquity as fetched (the
  Rust evaluates TT and obl80; TT-UTC = 69.184 s is ~38 arcsec of
  lunar motion — well inside one series error and the 3x margin).
- The sibling **Vallado Algorithm 29 Sun** (which differs from the
  Meeus Ch. 25 Sun by ~20 arcsec at these epochs) is kept
  reference-only; its deltas are recorded per epoch as
  `informational_vallado_alg29_sun_error` and enter no tolerance.

Per-body tolerances = 3x the worst measured error, rounded up to two
significant figures:

- Sun: 170.0 arcsec (distance rel 0.00013)
- Moon: 2300.0 arcsec (distance rel 0.0047)

The 3x margin buys headroom against future-epoch drift, the
authority's light-time-corrected (vs geometric) positions, and the
Moon's residual evaluation-input differences above. A transcription
typo shows up at degrees scale and fails the envelope by orders of
magnitude.

## `egm96-spot-accelerations.json`

This generator evaluates the potential of Barthelmes STR09/02
Eq. (108) with its own fully-normalized Legendre recursion (sources
above) and takes the analytic spherical gradient. Generation-time
self-checks, all hard-fail:

1. fetched-file SHA-256 + bit-equality of the embedded 88-row subset
   (plus the NGA original-distribution cross-check recorded above);
2. recursion magnitudes vs `mpmath.legenp` for all n,m <= 12 at three
   latitudes (< 1e-30);
3. analytic gradient vs 60-digit mpmath central differences of the
   same potential at every spot (< 5e-12 relative, measured value
   recorded per spot);
4. Laplacian-harmonicity residual at every spot (< 1e-9 relative —
   a recursion-coefficient error breaks harmonicity);
5. C20-only evaluation == closed-form J2 (J2 = -sqrt(5)*C20 = 0.0010826266835531513)
   to <= 1e-12 relative — the executable normalization contract
   (design Decision 3).

Envelope tolerance: 1e-9 relative per component (basis recorded in the
fixture).

## `harris-priester-table.json`

The 50-node min/max table is fetched-published data (two independent
transcriptions, Orekit and SatelliteToolbox.jl, agree on every row;
both cite Montenbruck & Gill, 'Satellite Orbits'). The interpolation +
bulge convention (exponential node interpolation, cos^2(psi/2) bulge,
apex 30 deg east of the subsolar point about the ITRF z-axis, geodetic
WGS-84 height, n = 2 per design Decision 4) is recorded in the fixture
so the Rust implementation mirrors it exactly. Worked examples are
tool-computed from the table with that convention.

## Trajectory goldens

Integrated with scipy `solve_ivp` DOP853 at rtol = atol = 1e-12,
max_step 300 s, sampled by `t_eval` on each arc's fixed grid. The
force stack is written independently of the Rust one: ERFA
epv00/moon98 ephemerides, this generator's own EGM96 evaluation,
Harris-Priester per the fetched table/convention, cannonball SRP with
the cylindrical shadow, direct-minus-indirect third body.

SRP convention (resolves the design.md open question): P(d) =
(TSI/c) * (au/d)^2 with the IAU 2015 B3 nominal TSI 1361 W/m^2,
c = 299792458 m/s, au = 149597870700 m, d = Sun->spacecraft distance;
P at 1 au = 4.53980733564685e-06 N/m^2 (= 1361/299792458, exactly mirrorable
in IEEE-754 by both stacks).

Per-arc tolerance = 3 x (sum of measured sensitivities) + 1 m floor
(velocities: + 1 mm/s), rounded up to two significant figures.
Sensitivities, each the max state delta over the sample grid:

- **integrator**: baseline re-run at rtol/atol 1e-10 (the committed
  samples' own numerical envelope);
- **ephemeris swap**: baseline re-run with the analytic Sun/Moon
  series the Rust stack implements (Meeus Ch. 25 Sun mirrored from
  ephemeris.rs on TT centuries, Vallado Alg 31 Moon) substituted for
  ERFA — bounds the Rust stack's ephemeris difference end to end
  (dominant term wherever third-body or bulge terms matter);
- **bulge-apex convention**: baseline re-run applying the 30 deg apex
  rotation about GCRF z instead of ITRF z (drag arcs only).

Measured at generation time:

| arc | integrator (m) | ephemeris swap (m) | apex conv (m) | rtol 1e-9 (m, info) | tolerance (m) | tolerance (m/s) |
|-----|----------------|--------------------|---------------|---------------------|---------------|-----------------|
| leo-full-force | 6.387e-01 | 7.871e-02 | 2.013e-02 | 1.235e-02 | 3.3 | 0.0035 |
| meo-harmonics-lunisolar | 3.438e-07 | 1.084e+01 | 0.000e+00 | 1.633e-07 | 34.0 | 0.0056 |
| srp-shadow-crossing | 8.159e-02 | 1.019e-01 | 0.000e+00 | 3.356e-02 | 1.6 | 0.0015 |

- Shadow crossings on `srp-shadow-crossing`: **2**
  0<->1 transitions of the cylindrical shadow factor (>= 2 asserted at
  generation time; times recorded in the fixture). The LEO arc's own
  crossings are recorded in its fixture for reference.
- Fidelity headroom on `leo-full-force`: J2-only propagation misses the
  golden by 385.433 m at arc end (>= 20x the position
  tolerance, asserted) — the numeric basis for the 'Higher fidelity
  than the J2 baseline' scenario.

## Value-source summary

- **Fetched-published**: EGM96 coefficients/GM/radius (ICGEM + NGA),
  the Harris-Priester table (Orekit + SatelliteToolbox.jl, Montenbruck
  & Gill lineage), the Meeus Ch. 25 Sun coefficients (mirrored from
  crates/thresh-core/src/orbital/ephemeris.rs, whose transcription
  cites the book-faithful soniakeys/meeus solar.go), the Vallado Moon
  series constants and the reference-only Alg 29 Sun (CelesTrak
  companion code), GM_Sun/GM_Moon/au/c (JPL SSD), TSI (IAU 2015 B3),
  the Legendre recursion and potential formulas (MITgcm geoid
  cookbook / Holmes & Featherstone; Barthelmes STR09/02).
- **Tool-computed** (packages above): astropy Sun/Moon authority
  positions, EGM96 spot accelerations, Harris-Priester worked
  examples, all trajectory samples, every measured sensitivity and
  self-check residual.
- **Defined inputs (not references)**: arc epochs, initial states,
  drag/SRP area-to-mass parameters, grid steps, the shadow-cylinder
  radius choice (WGS-84 equatorial), and omega_earth (chosen to match
  the Rust constant).
