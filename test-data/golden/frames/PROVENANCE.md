# Frame-Transform Golden Fixture Provenance

Tier-3 golden vectors for design Decision 7 of the `astro-time-and-frames`
change: the IAU-76/FK5 reduction chain (GCRF/MOD/TOD/TEME/PEF/ITRF) is
validated against Vallado's published worked reduction plus independently
generated astropy/Skyfield vectors. Consumed by the default-feature envelope
test (no network, no Python). Units in the JSON files are metres and m/s;
epochs are ISO-8601 UTC strings.

## Generation record

- Generator: `python/scripts/gen_frame_fixtures.py` (manual-only, never CI)
- Regeneration command (from the workspace root, then commit the output):

```sh
uv run python/scripts/gen_frame_fixtures.py
```

- Toolchain at generation time:

  - python 3.12.13
  - numpy 2.2.6
  - astropy 7.0.1
  - pyerfa 2.0.1.5
  - skyfield 1.49
  - astropy-iers-data 0.2026.7.13.0.54.2

No IERS *values* reach any transform: dUT1 is forced at the IERS-table
layer (a constant-valued `earth_orientation_table` science-state override,
so astropy's finite-difference velocity machinery sees the fixture dUT1 on
its shifted obstimes too - a per-Time `delta_ut1_utc` override does NOT
cover those and was measured to contaminate velocities by ~omega^2*dUT1*r),
and polar motion is forced at the transform layer (patched
`get_polar_motion`). Repeated runs at the same versions are bit-identical.

## `vallado-example-3-15.json` - published reduction

All published numbers were fetched and verified this generation session from:

1. Vallado, Seago, Seidelmann, 'Implementation Issues Surrounding the New IAU Reference Frameworks' (AAS 06-675), LEO test case, p. 17. https://www.agi.com/getmedia/2a33d303-9a4e-4520-9da0-86a4d5e45a10/Implementation-Issues-Surrounding-the-New-IAU-Reference-Systems-for-Astrodynamics.pdf
   - ITRF input state and the 'PEF iau76', 'TOD iau76', 'MOD iau76',
     'J2000 iau76', 'GCRF iau76 w corr', 'GCRF CIO dx,dy=0' rows, and the
     time table (jdtt 2453101.82815474550).
2. Vallado, Crawford, Hujsak, Kelso, 'Revisiting Spacetrack Report #3' (AIAA 2006-6753, Rev 3), Appendix C - TEME Coordinate System, p. 32. https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf
   - TEME position/velocity at the same epoch.
3. CelesTrak/fundamentals-of-astrodynamics companion code, software/matlab/ex3_15.m (input epoch and EOP for Example 3-15). https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/software/matlab/ex3_15.m
   - Exact EOP inputs: dut1 = -0.4399619 s, dat = 32 s, xp = -0.140682",
     yp = 0.333309", lod = 0.0015563 s, ddpsi = -0.052195", ddeps = -0.003875".

This is the same LEO test case as book Example 3-15 (Vallado, Fundamentals
of Astrodynamics and Applications - ex3_15.m is its companion code); the
papers above are the fetchable authoritative records of its outputs.

The fixture's `GCRF` leg is the **'J2000 iau76'** row: the IAU-76/FK5
reduction *without* the ddpsi/ddeps EOP nutation corrections, which is what
`Iau76Fk5Provider` (dUT1 + polar motion only) implements. The corrected row
('GCRF iau76 w corr') is recorded informationally; the two differ by ~0.9 m.

Envelope tolerance: **1.0 m and 0.01 m/s per leg** (spec 'Vallado reduction
reproduced'). Generation-time self-check - an independent pyerfa IAU-76/FK5
chain reproduced the published rows to:

| leg | dr (m) | dv (m/s) |
|-----|--------|----------|
| PEF | 0.0001 | 0.000000 |
| TOD | 0.0086 | 0.000005 |
| MOD | 0.0086 | 0.000005 |
| GCRF | 0.0086 | 0.000005 |
| TEME | 0.0800 | 0.000048 |

The fixture also records astropy's GCRS/TEME outputs at the same epoch/EOP
(IAU-2006/2000A theory) with their measured deltas from the published
IAU-76/FK5 rows, so the envelope test can assert against either set.
astropy's GCRS with the fixture EOP was itself checked against the paper's
'GCRF CIO dx,dy=0' row (< 1 m) before being recorded.

## `zero-eop-*.json` - astropy-computed vectors at provider-default EOP

Input TEME states are arbitrary LEO-magnitude vectors (defined in the
generator, not published values). Targets (GCRF, ITRF states) are computed
by astropy's frame machinery (TEME -> GCRS, TEME -> ITRS; equinox-of-date
TEME via GMST-1982, ITRS/GCRS via the IAU-2006/2000A CIO chain + frame
bias) under the zero-EOP configuration matching the Rust provider defaults:
UT1 = UTC (dUT1 = 0 forced at the IERS-table layer) and xp = yp = 0
(patched polar-motion lookup).

Because the Rust side implements equinox-based IAU-76/FK5, each comparison
records `measured_iau76_vs_astropy_m`: the disagreement of a reference
pyerfa IAU-76/FK5 chain (same theory as the Rust provider) with the astropy
target at that epoch - i.e. the pure theory difference (frame bias
~0.023 arcsec plus 1980-vs-2000A nutation, growing with distance from
J2000). Per-comparison tolerance = 1.5 x measured + 1 m (velocities:
1.5 x measured + 5 mm/s): a correct IAU-76 implementation passes with
margin, while the ~kilometre TEME/GCRF conflation this change removes
fails by orders of magnitude. Every comparison is measured in the exact
direction the envelope test runs it (input = `states[from_frame]`,
expected = `states[to_frame]`): notably GCRF -> ITRF starts from the
astropy-computed GCRF state, so the inertial-orientation theory delta
shows up there too and does NOT cancel. TEME -> ITRF is theory-free under
zero EOP (both sides are the GMST-1982 spin), so its measured delta is ~0.

Skyfield cross-check: the same TEME states pushed through Skyfield
(`Timescale(delta_t = 32.184 + dAT)` so UT1 = UTC; no polar-motion table)
agree with the astropy targets to the `skyfield_vs_astropy_m` field of each
comparison (sub-metre; Skyfield's sgp4lib TEME and IAU-2000A-family ITRS).

Measured agreement at generation time:

| case | transform | iau76 vs astropy (m) | skyfield vs astropy (m) | tolerance (m) |
|------|-----------|----------------------|-------------------------|---------------|
| zero-eop-2020-01-01 | TEME->GCRF | 1.015 | 0.000 | 2.5 |
| zero-eop-2020-01-01 | GCRF->ITRF | 1.015 | 0.000 | 2.5 |
| zero-eop-2020-01-01 | TEME->ITRF | 0.000 | 0.000 | 1.0 |
| zero-eop-2015-06-30 | TEME->GCRF | 0.979 | 0.000 | 2.5 |
| zero-eop-2015-06-30 | GCRF->ITRF | 0.979 | 0.000 | 2.5 |
| zero-eop-2015-06-30 | TEME->ITRF | 0.000 | 0.000 | 1.0 |
| zero-eop-2024-03-20 | TEME->GCRF | 0.826 | 0.000 | 2.2 |
| zero-eop-2024-03-20 | GCRF->ITRF | 0.826 | 0.000 | 2.2 |
| zero-eop-2024-03-20 | TEME->ITRF | 0.000 | 0.000 | 1.0 |

## Value-source summary

- Fetched-published: every number under `published`, `input`,
  `gcrf_with_eop_corrections`, and the `eop` block of
  `vallado-example-3-15.json` (sources 1-3 above).
- Tool-computed (astropy/pyerfa/Skyfield at the versions above): the
  `astropy_iau2006_2000a` blocks, all `states` marked GCRF/ITRF and all
  `comparisons`/`measured_*` fields in `zero-eop-*.json`, and the
  generator self-check residuals.
- Defined inputs (not references): the TEME states marked
  `origin: defined input` in `zero-eop-*.json`.
