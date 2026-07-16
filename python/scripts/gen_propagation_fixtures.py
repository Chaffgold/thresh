#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy==2.2.6",
#     "scipy==1.15.3",
#     "astropy==7.0.1",
#     "pyerfa==2.0.1.5",
#     "mpmath==1.3.0",
# ]
# ///
"""Golden-fixture generator for `test-data/golden/propagation/`.

orbit-propagation-fidelity, task 4.2 (design Decision 7). MANUAL-ONLY —
never runs in CI. CI consumes the committed fixtures with no Python and no
network (default features).

Regenerate from the repository root with:

    uv run python/scripts/gen_propagation_fixtures.py

Requires network access once per run: the EGM96 coefficient file is fetched
from ICGEM and SHA-256-verified against the pinned digest before the embedded
degree-2..12 subset is trusted (set THRESH_EGM96_GFC to a pre-downloaded copy
to skip the download; the SHA-256 check still runs).

Outputs (all built in memory first, then written together):

  test-data/golden/propagation/sun-moon-ephemeris.json
      Sun/Moon GCRF positions from astropy (authority) at six epochs, with
      the measured angular error of this script's mirror of the analytic
      series the Rust ephemerides implement (Meeus Ch. 25 low-accuracy Sun,
      Vallado Alg 31 Moon) — the per-body tolerance basis, measured not
      assumed (design Decision 5). The sibling Vallado Alg 29 Sun is kept
      for reference and its deltas recorded as informational only.
  test-data/golden/propagation/egm96-spot-accelerations.json
      EGM96 12x12 accelerations at four ITRF positions from this script's
      own normalized-Legendre evaluation of the fetched coefficient file,
      cross-checked against 60-digit mpmath numerical gradients.
  test-data/golden/propagation/harris-priester-table.json
      The fetched published Harris-Priester min/max density table (50 nodes,
      100-1000 km) plus worked bulge/interpolation examples.
  test-data/golden/propagation/{leo-full-force,meo-harmonics-lunisolar,
      srp-shadow-crossing}.json
      Three trajectory goldens integrated by this independent force stack
      (ERFA ephemerides, own harmonics, Harris-Priester, cannonball SRP with
      cylindrical shadow) under scipy DOP853 at rtol=atol=1e-12, sampled on
      fixed grids, with per-arc tolerances derived from measured integrator/
      ephemeris/convention sensitivities.
  test-data/golden/propagation/PROVENANCE.md
      Sources (URL + SHA-256/commit), toolchain versions, methods, measured
      sensitivities, and the value-source classification of every number.

Frames convention: zero-EOP IAU-76/FK5 exactly like the Rust default
`Iau76Fk5Provider` — dUT1 = 0 (UT1 = UTC), no polar motion, GCRF -> ITRF =
R3(GAST[GMST-1982 + EqEq-1994]) . N(IAU-1980) . P(IAU-76).

Every numerical table below cites the authoritative source it was fetched
from this generation session; nothing is transcribed from memory. Self-checks
hard-fail the run (nothing is written) if a transcription disagrees with the
fetched file or an independent evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import erfa
import mpmath
import numpy as np
from astropy import units as u
from astropy.coordinates import get_body, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
from scipy.integrate import solve_ivp

# Fixtures must be reproducible offline: never consult (or download) IERS data.
iers.conf.auto_download = False
# pyerfa flags post-leap-second-table years as "dubious"; no leap second is
# scheduled through the fixture epochs (none since 2017), so dAT = 37 s is
# correct and the warning is noise. Correctness is asserted per-arc below
# (dAT constant across each arc).
warnings.filterwarnings("ignore", category=erfa.core.ErfaWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "test-data" / "golden" / "propagation"

# ---------------------------------------------------------------------------
# Physical constants (every value fetched this session; source cited)
# ---------------------------------------------------------------------------
# JPL SSD "Astrodynamic Parameters" (https://ssd.jpl.nasa.gov/astro_par.html):
#   GM_Sun = 1.32712440041279419e20 m^3/s^2
#   GM_Moon = 4902.800118 km^3/s^2 (DE440)
#   au = 149597870700 m (IAU 2012 Resolution B1)
#   c = 299792458 m/s (CODATA 2018, exact)
GM_SUN = 1.32712440041279419e20  # m^3/s^2
GM_MOON = 4902.800118e9  # m^3/s^2
AU_M = 149_597_870_700.0  # m
C_LIGHT = 299_792_458.0  # m/s
# IAU 2015 Resolution B3 (Prsa et al. 2016, AJ 152:41; arXiv:1510.07674):
# nominal total solar irradiance S_sun = 1361 W/m^2.
TSI_W_M2 = 1361.0
# SRP convention (resolves the design.md open question; both stacks mirror it
# exactly): P(d) = (TSI/c) * (AU/d)^2, d = Sun->spacecraft distance.
P_SRP_1AU = TSI_W_M2 / C_LIGHT  # N/m^2 at 1 au

# Earth rotation rate used for the ITRF velocity transport term, matching the
# Rust side (`thresh_core::eci::EARTH_ROTATION_RATE`, rad/s).
OMEGA_EARTH = 7.292_115_0e-5
# Cylindrical Earth-shadow radius (WGS-84 equatorial radius, m) — a modeling
# choice, recorded in every SRP force_config block so Rust mirrors it.
R_SHADOW_M = 6_378_137.0

# EGM96 model constants from the fetched ICGEM gfc header
# (earth_gravity_constant / radius keys; same values in NASA TP-1998-206861):
EGM96_GM = 3.986004415e14  # m^3/s^2
EGM96_RADIUS = 6.3781363e6  # m
EGM96_DEGREE = 12

# Vallado companion-code constants (constastro.m, CelesTrak repo, fetched):
#   re = 6378.1363 km  (Earth radii -> km for the Moon series)
#   au = 149597870.7 km (equals AU_M)
RE_VALLADO_M = 6.3781363e6

ARCSEC = math.pi / (180.0 * 3600.0)

# ---------------------------------------------------------------------------
# EGM96 normalized coefficients, degree 2..12 (fully normalized, tide-free).
#
# Source (fetched + SHA-256-verified at generation time, see
# fetch_and_verify_egm96):
#   ICGEM EGM96.gfc — http://icgem.gfz-potsdam.de/getmodel/gfc/
#   971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc
#   (redirects to icgem.gfz.de), header: earth_gravity_constant 0.3986004415E+15,
#   radius 0.6378136300E+07, tide_system tide_free, errors formal.
# Cross-verified this session (all 88 rows bit-equal) against the original
# NASA/NIMA distribution: NGA "egm-96spherical" archive, file `EGM96`
# (https://earth-info.nga.mil/php/download.php?file=egm-96spherical).
# The gfc file also carries the degree-0 row (C00 = 1); degree-1 terms are
# absent (geocentric origin). Values below are the gfc columns C, S verbatim.
# ---------------------------------------------------------------------------
EGM96_DEG12_TEXT = """\
2 0 -0.484165371736e-03 0.000000000000e+00
2 1 -0.186987635955e-09 0.119528012031e-08
2 2 0.243914352398e-05 -0.140016683654e-05
3 0 0.957254173792e-06 0.000000000000e+00
3 1 0.202998882184e-05 0.248513158716e-06
3 2 0.904627768605e-06 -0.619025944205e-06
3 3 0.721072657057e-06 0.141435626958e-05
4 0 0.539873863789e-06 0.000000000000e+00
4 1 -0.536321616971e-06 -0.473440265853e-06
4 2 0.350694105785e-06 0.662671572540e-06
4 3 0.990771803829e-06 -0.200928369177e-06
4 4 -0.188560802735e-06 0.308853169333e-06
5 0 0.685323475630e-07 0.000000000000e+00
5 1 -0.621012128528e-07 -0.944226127525e-07
5 2 0.652438297612e-06 -0.323349612668e-06
5 3 -0.451955406071e-06 -0.214847190624e-06
5 4 -0.295301647654e-06 0.496658876769e-07
5 5 0.174971983203e-06 -0.669384278219e-06
6 0 -0.149957994714e-06 0.000000000000e+00
6 1 -0.760879384947e-07 0.262890545501e-07
6 2 0.481732442832e-07 -0.373728201347e-06
6 3 0.571730990516e-07 0.902694517163e-08
6 4 -0.862142660109e-07 -0.471408154267e-06
6 5 -0.267133325490e-06 -0.536488432483e-06
6 6 0.967616121092e-08 -0.237192006935e-06
7 0 0.909789371450e-07 0.000000000000e+00
7 1 0.279872910488e-06 0.954336911867e-07
7 2 0.329743816488e-06 0.930667596042e-07
7 3 0.250398657706e-06 -0.217198608738e-06
7 4 -0.275114355257e-06 -0.123800392323e-06
7 5 0.193765507243e-08 0.177377719872e-07
7 6 -0.358856860645e-06 0.151789817739e-06
7 7 0.109185148045e-08 0.244415707993e-07
8 0 0.496711667324e-07 0.000000000000e+00
8 1 0.233422047893e-07 0.590060493411e-07
8 2 0.802978722615e-07 0.654175425859e-07
8 3 -0.191877757009e-07 -0.863454445021e-07
8 4 -0.244600105471e-06 0.700233016934e-07
8 5 -0.255352403037e-07 0.891462164788e-07
8 6 -0.657361610961e-07 0.309238461807e-06
8 7 0.672811580072e-07 0.747440473633e-07
8 8 -0.124092493016e-06 0.120533165603e-06
9 0 0.276714300853e-07 0.000000000000e+00
9 1 0.143387502749e-06 0.216834947618e-07
9 2 0.222288318564e-07 -0.322196647116e-07
9 3 -0.160811502143e-06 -0.742287409462e-07
9 4 -0.900179225336e-08 0.194666779475e-07
9 5 -0.166165092924e-07 -0.541113191483e-07
9 6 0.626941938248e-07 0.222903525945e-06
9 7 -0.118366323475e-06 -0.965152667886e-07
9 8 0.188436022794e-06 -0.308566220421e-08
9 9 -0.477475386132e-07 0.966412847714e-07
10 0 0.526222488569e-07 0.000000000000e+00
10 1 0.835115775652e-07 -0.131314331796e-06
10 2 -0.942413882081e-07 -0.515791657390e-07
10 3 -0.689895048176e-08 -0.153768828694e-06
10 4 -0.840764549716e-07 -0.792806255331e-07
10 5 -0.493395938185e-07 -0.505370221897e-07
10 6 -0.375885236598e-07 -0.795667053872e-07
10 7 0.811460540925e-08 -0.336629641314e-08
10 8 0.404927981694e-07 -0.918705975922e-07
10 9 0.125491334939e-06 -0.376516222392e-07
10 10 0.100538634409e-06 -0.240148449520e-07
11 0 -0.509613707522e-07 0.000000000000e+00
11 1 0.151687209933e-07 -0.268604146166e-07
11 2 0.186309749878e-07 -0.990693862047e-07
11 3 -0.309871239854e-07 -0.148131804260e-06
11 4 -0.389580205051e-07 -0.636666511980e-07
11 5 0.377848029452e-07 0.494736238169e-07
11 6 -0.118676592395e-08 0.344769584593e-07
11 7 0.411565188074e-08 -0.898252808977e-07
11 8 -0.598410841300e-08 0.243989612237e-07
11 9 -0.314231072723e-07 0.417731829829e-07
11 10 -0.521882681927e-07 -0.183364561788e-07
11 11 0.460344448746e-07 -0.696662308185e-07
12 0 0.377252636558e-07 0.000000000000e+00
12 1 -0.540654977836e-07 -0.435675748979e-07
12 2 0.142979642253e-07 0.320975937619e-07
12 3 0.393995876403e-07 0.244264863505e-07
12 4 -0.686908127934e-07 0.415081109011e-08
12 5 0.309411128730e-07 0.782536279033e-08
12 6 0.341523275208e-08 0.391765484449e-07
12 7 -0.186909958587e-07 0.356131849382e-07
12 8 -0.253769398865e-07 0.169361024629e-07
12 9 0.422880630662e-07 0.252692598301e-07
12 10 -0.617619654902e-08 0.308375794212e-07
12 11 0.112502994122e-07 -0.637946501558e-08
12 12 -0.249532607390e-08 -0.111780601900e-07
"""

EGM96_ICGEM_URL = (
    "http://icgem.gfz-potsdam.de/getmodel/gfc/"
    "971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc"
)
EGM96_ICGEM_SHA256 = "5247a9e9c316dd2c8f8fd491d53be0e163cb5cb3676b021754240cc9e44cb43b"
EGM96_NGA_URL = "https://earth-info.nga.mil/php/download.php?file=egm-96spherical"
EGM96_NGA_FILE_SHA256 = "1e5e6c30343989b8e2eda0bb96bde06ef05981eec69934d6e44aace4f0d6a9d5"

# ---------------------------------------------------------------------------
# Harris-Priester min/max density table: altitude (m), rho_min, rho_max
# (kg/m^3), mean solar activity, 100-1000 km.
#
# Sources (both fetched this session; the two transcriptions agree on all
# 50 rows, diff-verified):
#   1. CS-SI/Orekit, src/main/java/org/orekit/models/earth/atmosphere/
#      HarrisPriester.java (ALT_RHO), commit 77cfa41667457f0db879a0fc7e892e7248c1e516,
#      which cites Montenbruck & Gill, "Satellite Orbits", Springer 2005.
#   2. JuliaSpace/SatelliteToolboxAtmosphericModels.jl,
#      src/harrispriester/constants.jl (_HARRIS_PRIESTER_ALT_RHO),
#      commit 268c2645923d5a1c4206785c9c98f1ff2eccd0e4.
# ---------------------------------------------------------------------------
HP_TABLE: list[tuple[float, float, float]] = [
    (100_000.0, 4.974e-07, 4.974e-07),
    (120_000.0, 2.490e-08, 2.490e-08),
    (130_000.0, 8.377e-09, 8.710e-09),
    (140_000.0, 3.899e-09, 4.059e-09),
    (150_000.0, 2.122e-09, 2.215e-09),
    (160_000.0, 1.263e-09, 1.344e-09),
    (170_000.0, 8.008e-10, 8.758e-10),
    (180_000.0, 5.283e-10, 6.010e-10),
    (190_000.0, 3.617e-10, 4.297e-10),
    (200_000.0, 2.557e-10, 3.162e-10),
    (210_000.0, 1.839e-10, 2.396e-10),
    (220_000.0, 1.341e-10, 1.853e-10),
    (230_000.0, 9.949e-11, 1.455e-10),
    (240_000.0, 7.488e-11, 1.157e-10),
    (250_000.0, 5.709e-11, 9.308e-11),
    (260_000.0, 4.403e-11, 7.555e-11),
    (270_000.0, 3.430e-11, 6.182e-11),
    (280_000.0, 2.697e-11, 5.095e-11),
    (290_000.0, 2.139e-11, 4.226e-11),
    (300_000.0, 1.708e-11, 3.526e-11),
    (320_000.0, 1.099e-11, 2.511e-11),
    (340_000.0, 7.214e-12, 1.819e-11),
    (360_000.0, 4.824e-12, 1.337e-11),
    (380_000.0, 3.274e-12, 9.955e-12),
    (400_000.0, 2.249e-12, 7.492e-12),
    (420_000.0, 1.558e-12, 5.684e-12),
    (440_000.0, 1.091e-12, 4.355e-12),
    (460_000.0, 7.701e-13, 3.362e-12),
    (480_000.0, 5.474e-13, 2.612e-12),
    (500_000.0, 3.916e-13, 2.042e-12),
    (520_000.0, 2.819e-13, 1.605e-12),
    (540_000.0, 2.042e-13, 1.267e-12),
    (560_000.0, 1.488e-13, 1.005e-12),
    (580_000.0, 1.092e-13, 7.997e-13),
    (600_000.0, 8.070e-14, 6.390e-13),
    (620_000.0, 6.012e-14, 5.123e-13),
    (640_000.0, 4.519e-14, 4.121e-13),
    (660_000.0, 3.430e-14, 3.325e-13),
    (680_000.0, 2.632e-14, 2.691e-13),
    (700_000.0, 2.043e-14, 2.185e-13),
    (720_000.0, 1.607e-14, 1.779e-13),
    (740_000.0, 1.281e-14, 1.452e-13),
    (760_000.0, 1.036e-14, 1.190e-13),
    (780_000.0, 8.496e-15, 9.776e-14),
    (800_000.0, 7.069e-15, 8.059e-14),
    (840_000.0, 4.680e-15, 5.741e-14),
    (880_000.0, 3.200e-15, 4.210e-14),
    (920_000.0, 2.210e-15, 3.130e-14),
    (960_000.0, 1.560e-15, 2.360e-14),
    (1_000_000.0, 1.150e-15, 1.810e-14),
]
HP_BULGE_EXPONENT = 2  # design Decision 4: n = 2 fixed
HP_APEX_LAG_DEG = 30.0  # bulge apex 30 deg east of the subsolar point

SOURCES = {
    "icgem-egm96": (
        "ICGEM (GFZ) EGM96 gfc file, " + EGM96_ICGEM_URL + f" — SHA-256 {EGM96_ICGEM_SHA256}. "
        "Header: earth_gravity_constant 0.3986004415E+15 m^3/s^2, radius "
        "0.6378136300E+07 m, tide_system tide_free. Citation in file header: "
        "Lemoine et al., NASA/TP-1998-206861."
    ),
    "nga-egm96": (
        "NGA EGM96 spherical-harmonic package, " + EGM96_NGA_URL + " — inner file `EGM96` "
        f"SHA-256 {EGM96_NGA_FILE_SHA256} (original 1996 NASA/NIMA distribution); "
        "cross-check source for the degree-2..12 subset (bit-equal, all 88 rows)."
    ),
    "orekit-hp": (
        "CS-SI/Orekit HarrisPriester.java (table ALT_RHO and the density formula), "
        "commit 77cfa41667457f0db879a0fc7e892e7248c1e516, "
        "https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/"
        "models/earth/atmosphere/HarrisPriester.java — cites Montenbruck & Gill, "
        "'Satellite Orbits', Springer 2005."
    ),
    "satkit-hp": (
        "JuliaSpace/SatelliteToolboxAtmosphericModels.jl constants.jl "
        "(_HARRIS_PRIESTER_ALT_RHO), commit 268c2645923d5a1c4206785c9c98f1ff2eccd0e4, "
        "https://github.com/JuliaSpace/SatelliteToolboxAtmosphericModels.jl/blob/main/"
        "src/harrispriester/constants.jl — independent transcription of the same table."
    ),
    "meeus-sun": (
        "Meeus, 'Astronomical Algorithms', 2nd ed., Ch. 25 'Solar Coordinates', "
        "low-accuracy method (Eqs. 25.2-25.5, ~0.01 deg class) — the Sun series "
        "the Rust ephemerides implement. This generator mirrors the "
        "crates/thresh-core/src/orbital/ephemeris.rs Meeus Ch. 25 transcription "
        "coefficient for coefficient (the Rust file transcribed them from the "
        "book-faithful soniakeys/meeus v3 Go implementation, solar/solar.go, and "
        "pins Meeus Example 25.a as a spot test — re-asserted here at "
        "generation time)."
    ),
    "vallado-sun": (
        "CelesTrak/fundamentals-of-astrodynamics, software/matlab/sun.m "
        "(Vallado 2022, p. 285, Algorithm 29 — low-precision Sun, ~0.01 deg), "
        "commit 4555fdc4f611817cf55a8b3158860bdab0b552ee, "
        "https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/"
        "software/matlab/sun.m — reference-only: this is NOT the Sun series the "
        "Rust ephemerides implement (that is meeus-sun); its deltas are "
        "recorded as informational and enter no tolerance."
    ),
    "vallado-moon": (
        "CelesTrak/fundamentals-of-astrodynamics, software/matlab/moon.m "
        "(Vallado 2022, p. 294, Algorithm 31 — truncated ELP-lineage Moon), "
        "commit 102fe0182ea0c20d02fdf18b285faa6869067feb, "
        "https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/"
        "software/matlab/moon.m"
    ),
    "jpl-astro-par": (
        "JPL SSD Astrodynamic Parameters, https://ssd.jpl.nasa.gov/astro_par.html — "
        "GM_Sun = 1.32712440041279419e20 m^3/s^2, GM_Moon = 4902.800118 km^3/s^2 "
        "(DE440), au = 149597870700 m (IAU 2012 B1), c = 299792458 m/s (CODATA 2018)."
    ),
    "iau2015b3": (
        "IAU 2015 Resolution B3 (Prsa et al. 2016, AJ 152:41; arXiv:1510.07674) — "
        "nominal total solar irradiance 1361 W/m^2."
    ),
    "legendre-recursion": (
        "Fully-normalized associated Legendre recursion: seeds P00=1, P11=sqrt(3)*u; "
        "sectoral Pmm = u*sqrt((2m+1)/(2m))*P(m-1,m-1); non-sectoral "
        "Pnm = a_nm*t*P(n-1,m) - b_nm*P(n-2,m) with "
        "a_nm = sqrt((2n-1)(2n+1)/((n-m)(n+m))), "
        "b_nm = sqrt((2n+1)(n+m-1)(n-m-1)/((n-m)(n+m)(2n-3))); t = sin(lat), "
        "u = cos(lat); normalization sqrt(k(2n+1)(n-m)!/(n+m)!), k=1 (m=0) else 2. "
        "Fetched from the MITgcm geoid cookbook sec. 3.7 "
        "(http://mitgcm.org/~mlosch/geoidcookbook/node11.html), which follows "
        "Holmes & Featherstone 2002 (J. Geodesy 76:279) and Torge 1991."
    ),
    "barthelmes": (
        "Barthelmes, 'Definition of Functionals of the Geopotential ...' "
        "(GFZ STR09/02, revised Jan 2013), Eq. (108): W = (GM/r) * sum_l sum_m "
        "(R/r)^l * Plm(sin phi) * (Clm cos(m lambda) + Slm sin(m lambda)), "
        "spherical geocentric coordinates, fully normalized. Fetched from "
        "https://gfzpublic.gfz.de/pubman/item/item_104132_4/component/"
        "file_104133/0902-2.pdf"
    ),
}

# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------


def rot3_frame(theta: float) -> np.ndarray:
    """Vallado ROT3 (frame rotation about +z): maps of-date axes by +theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def rotz_active(theta: float) -> np.ndarray:
    """Active rotation about +z (RA increases by +theta for a vector)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def angle_between_rad(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    # atan2 form is well-conditioned for small angles.
    return math.atan2(float(np.linalg.norm(np.cross(ua, ub))), float(np.dot(ua, ub)))


def vec(v: np.ndarray) -> list[float]:
    return [float(x) for x in v]


# ---------------------------------------------------------------------------
# EGM96: fetch + verify, parse, evaluate
# ---------------------------------------------------------------------------


def parse_embedded_egm96() -> dict[tuple[int, int], tuple[float, float]]:
    coeffs: dict[tuple[int, int], tuple[float, float]] = {}
    for line in EGM96_DEG12_TEXT.strip().splitlines():
        n_s, m_s, c_s, s_s = line.split()
        coeffs[(int(n_s), int(m_s))] = (float(c_s), float(s_s))
    if len(coeffs) != 88:
        raise AssertionError(f"embedded EGM96 subset has {len(coeffs)} rows, expected 88")
    return coeffs


def fetch_egm96_gfc_bytes() -> tuple[bytes, str]:
    """Return (file bytes, description of where they came from)."""
    override = os.environ.get("THRESH_EGM96_GFC")
    if override:
        return Path(override).read_bytes(), f"local copy {override} (THRESH_EGM96_GFC)"
    req = urllib.request.Request(
        EGM96_ICGEM_URL, headers={"User-Agent": "thresh-golden-generator/1.0"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read(), EGM96_ICGEM_URL


def fetch_and_verify_egm96(embedded: dict[tuple[int, int], tuple[float, float]]) -> str:
    """Fetch the ICGEM EGM96 gfc, pin its SHA-256, and require the embedded
    degree-2..12 subset to equal the fetched values exactly. Hard-fails on any
    disagreement so a transcription typo can never reach the fixtures."""
    data, origin = fetch_egm96_gfc_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EGM96_ICGEM_SHA256:
        raise AssertionError(
            f"EGM96 gfc SHA-256 mismatch: got {digest}, pinned {EGM96_ICGEM_SHA256} "
            f"(source: {origin})"
        )
    fetched: dict[tuple[int, int], tuple[float, float]] = {}
    header_gm = header_radius = None
    for raw in data.decode("ascii", errors="replace").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if parts[0] == "earth_gravity_constant":
            header_gm = float(parts[1])
        elif parts[0] == "radius":
            header_radius = float(parts[1])
        elif parts[0] == "gfc" and 2 <= int(parts[1]) <= EGM96_DEGREE:
            fetched[(int(parts[1]), int(parts[2]))] = (float(parts[3]), float(parts[4]))
    if header_gm != EGM96_GM or header_radius != EGM96_RADIUS:
        raise AssertionError(f"EGM96 header GM/radius mismatch: {header_gm}, {header_radius}")
    if fetched != embedded:
        bad = [k for k in embedded if fetched.get(k) != embedded[k]]
        raise AssertionError(f"embedded EGM96 subset disagrees with fetched file at {bad[:5]}")
    print(f"EGM96 verified against {origin} (SHA-256 ok, 88 rows bit-equal)")
    return origin


def legendre_norm_dnorm(
    nmax: int, sphi: float, cphi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fully-normalized P̄nm(sin phi) and d P̄nm / d phi up to degree/order nmax.

    Recursion per SOURCES['legendre-recursion'] with t = sin(phi), u = cos(phi);
    derivatives by mechanical chain rule on the same recursion
    (dt/dphi = u, du/dphi = -t).
    """
    p = np.zeros((nmax + 1, nmax + 1))
    dp = np.zeros((nmax + 1, nmax + 1))
    p[0, 0] = 1.0
    if nmax >= 1:
        s3 = math.sqrt(3.0)
        p[1, 1] = s3 * cphi
        dp[1, 1] = -s3 * sphi
    for m in range(2, nmax + 1):
        c = math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        p[m, m] = c * cphi * p[m - 1, m - 1]
        dp[m, m] = c * (cphi * dp[m - 1, m - 1] - sphi * p[m - 1, m - 1])
    for m in range(nmax + 1):
        for n in range(m + 1, nmax + 1):
            a = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
            b = 0.0
            if n - m >= 2:
                b = math.sqrt(
                    (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
                    / ((n - m) * (n + m) * (2.0 * n - 3.0))
                )
            p[n, m] = a * sphi * p[n - 1, m] - b * p[n - 2, m]
            dp[n, m] = a * (cphi * p[n - 1, m] + sphi * dp[n - 1, m]) - b * dp[n - 2, m]
    return p, dp


def legendre_norm_mp(nmax: int, t, u) -> list[list]:
    """The same recursion in mpmath arithmetic (values only, no derivatives)."""
    p = [[mpmath.mpf(0)] * (nmax + 1) for _ in range(nmax + 1)]
    p[0][0] = mpmath.mpf(1)
    if nmax >= 1:
        p[1][1] = mpmath.sqrt(3) * u
    for m in range(2, nmax + 1):
        p[m][m] = mpmath.sqrt(mpmath.mpf(2 * m + 1) / (2 * m)) * u * p[m - 1][m - 1]
    for m in range(nmax + 1):
        for n in range(m + 1, nmax + 1):
            a = mpmath.sqrt(mpmath.mpf((2 * n - 1) * (2 * n + 1)) / ((n - m) * (n + m)))
            b = mpmath.mpf(0)
            if n - m >= 2:
                b = mpmath.sqrt(
                    mpmath.mpf((2 * n + 1) * (n + m - 1) * (n - m - 1))
                    / ((n - m) * (n + m) * (2 * n - 3))
                )
            p[n][m] = a * t * p[n - 1][m] - b * p[n - 2][m]
    return p


@dataclass
class Egm96Model:
    coeffs: dict[tuple[int, int], tuple[float, float]]
    degree: int = EGM96_DEGREE
    gm: float = EGM96_GM
    radius: float = EGM96_RADIUS

    def c(self, n: int, m: int) -> float:
        if n == 0 and m == 0:
            return 1.0
        return self.coeffs.get((n, m), (0.0, 0.0))[0]

    def s(self, n: int, m: int) -> float:
        return self.coeffs.get((n, m), (0.0, 0.0))[1]


def egm96_accel_itrf(model: Egm96Model, r_itrf: np.ndarray) -> np.ndarray:
    """Total gravitational acceleration (central + harmonics through the model
    degree) at an ITRF position, m/s^2.

    Potential per SOURCES['barthelmes'] Eq. (108); gradient in spherical
    geocentric coordinates:
      a = dU/dr * r_hat + (1/r) dU/dphi * phi_hat + (1/(r cos phi)) dU/dlambda
          * lambda_hat.
    Verified at generation time against 60-digit mpmath central differences of
    the same potential and against the closed-form J2 identity.
    """
    x, y, z = float(r_itrf[0]), float(r_itrf[1]), float(r_itrf[2])
    r = math.sqrt(x * x + y * y + z * z)
    rxy = math.hypot(x, y)
    sphi, cphi = z / r, rxy / r
    lam = math.atan2(y, x)
    p, dp = legendre_norm_dnorm(model.degree, sphi, cphi)

    du_dr = 0.0
    du_dphi = 0.0
    du_dlam = 0.0
    ratio = model.radius / r
    for n in range(model.degree + 1):
        if n == 1:
            continue  # degree-1 terms are zero (geocentric origin)
        rn = ratio**n
        sum_r = sum_phi = sum_lam = 0.0
        for m in range(n + 1):
            cm, sm = model.c(n, m), model.s(n, m)
            if cm == 0.0 and sm == 0.0:
                continue
            cml, sml = math.cos(m * lam), math.sin(m * lam)
            t1 = cm * cml + sm * sml
            sum_r += p[n, m] * t1
            sum_phi += dp[n, m] * t1
            sum_lam += m * p[n, m] * (sm * cml - cm * sml)
        du_dr += -(model.gm / (r * r)) * (n + 1.0) * rn * sum_r
        du_dphi += (model.gm / r) * rn * sum_phi
        du_dlam += (model.gm / r) * rn * sum_lam

    a_r = du_dr
    a_phi = du_dphi / r
    a_lam = du_dlam / (r * cphi)
    coslam, sinlam = math.cos(lam), math.sin(lam)
    r_hat = np.array([cphi * coslam, cphi * sinlam, sphi])
    phi_hat = np.array([-sphi * coslam, -sphi * sinlam, cphi])
    lam_hat = np.array([-sinlam, coslam, 0.0])
    return a_r * r_hat + a_phi * phi_hat + a_lam * lam_hat


def egm96_potential_mp(model: Egm96Model, x, y, z):
    """Eq. (108) potential in mpmath arithmetic (for numerical gradients)."""
    r = mpmath.sqrt(x * x + y * y + z * z)
    rxy = mpmath.sqrt(x * x + y * y)
    sphi, cphi = z / r, rxy / r
    lam = mpmath.atan2(y, x)
    p = legendre_norm_mp(model.degree, sphi, cphi)
    total = mpmath.mpf(0)
    ratio = mpmath.mpf(model.radius) / r
    for n in range(model.degree + 1):
        if n == 1:
            continue
        rn = ratio**n
        s = mpmath.mpf(0)
        for m in range(n + 1):
            cm, sm = model.c(n, m), model.s(n, m)
            if cm == 0.0 and sm == 0.0:
                continue
            s += p[n][m] * (cm * mpmath.cos(m * lam) + sm * mpmath.sin(m * lam))
        total += rn * s
    return mpmath.mpf(model.gm) / r * total


def mp_gradient_accel(model: Egm96Model, r_itrf: np.ndarray, h: float = 1.0) -> np.ndarray:
    """60-digit central-difference gradient of the mpmath potential (m/s^2)."""
    with mpmath.workdps(60):
        base = [mpmath.mpf(repr(float(c))) for c in r_itrf]
        out = []
        for axis in range(3):
            plus = list(base)
            minus = list(base)
            plus[axis] += h
            minus[axis] -= h
            up = egm96_potential_mp(model, *plus)
            um = egm96_potential_mp(model, *minus)
            out.append(float((up - um) / (2 * h)))
    return np.array(out)


def mp_laplacian_residual(model: Egm96Model, r_itrf: np.ndarray, h: float = 10.0) -> float:
    """Relative Laplacian residual — the truncated field must be harmonic.

    A transcription error in the recursion coefficients breaks the degree
    consistency between the radial factor and the angular function and shows
    up as a non-zero Laplacian; a correct field gives ~0 to the finite-
    difference truncation level.
    """
    with mpmath.workdps(60):
        base = [mpmath.mpf(repr(float(c))) for c in r_itrf]
        u0 = egm96_potential_mp(model, *base)
        second_sum = mpmath.mpf(0)
        second_max = mpmath.mpf(0)
        for axis in range(3):
            plus = list(base)
            minus = list(base)
            plus[axis] += h
            minus[axis] -= h
            upp = egm96_potential_mp(model, *plus)
            umm = egm96_potential_mp(model, *minus)
            d2 = (upp - 2 * u0 + umm) / (h * h)
            second_sum += d2
            second_max = max(second_max, abs(d2))
        return float(abs(second_sum) / second_max)


def j2_closed_form_accel(r: np.ndarray, mu: float, re: float, j2: float) -> np.ndarray:
    """Two-body + J2 closed form, mirroring `thresh-core/src/orbital/gravity.rs`
    `j2_acceleration` (x/y carry (5z^2/r^2 - 1), z carries (5z^2/r^2 - 3))."""
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    r2 = x * x + y * y + z * z
    rn = math.sqrt(r2)
    r5 = r2 * r2 * rn
    mu_r3 = mu / (r2 * rn)
    j2c = 1.5 * j2 * mu * re * re / r5
    z2 = 5.0 * z * z / r2
    return np.array(
        [
            -mu_r3 * x + j2c * x * (z2 - 1.0),
            -mu_r3 * y + j2c * y * (z2 - 1.0),
            -mu_r3 * z + j2c * z * (z2 - 3.0),
        ]
    )


def verify_legendre_vs_mpmath_legenp(nmax: int = EGM96_DEGREE) -> None:
    """Magnitude cross-check of the recursion against mpmath.legenp.

    mpmath's Ferrers-function conventions may differ from the geodetic one by
    the Condon-Shortley phase (a sign), so magnitudes are compared; the sign
    contract is executed separately by the C20/J2 identity and the Laplacian
    harmonicity check.
    """
    with mpmath.workdps(40):
        for phi_deg in (-62.0, 17.0, 49.0):
            t = mpmath.sin(mpmath.radians(phi_deg))
            u_ = mpmath.cos(mpmath.radians(phi_deg))
            p = legendre_norm_mp(nmax, t, u_)
            for n in range(nmax + 1):
                for m in range(n + 1):
                    k = 1 if m == 0 else 2
                    norm = mpmath.sqrt(
                        mpmath.mpf(k)
                        * (2 * n + 1)
                        * mpmath.factorial(n - m)
                        / mpmath.factorial(n + m)
                    )
                    ref = norm * mpmath.legenp(n, m, t)
                    err = abs(abs(p[n][m]) - abs(ref))
                    scale = max(abs(ref), mpmath.mpf(1))
                    if err / scale > mpmath.mpf("1e-30"):
                        raise AssertionError(
                            f"Legendre recursion vs mpmath.legenp at n={n} m={m} "
                            f"phi={phi_deg}: |{p[n][m]}| vs |{ref}|"
                        )
    print("Legendre recursion verified against mpmath.legenp (n,m <= 12, 3 latitudes)")


def verify_c20_identity(model: Egm96Model) -> float:
    """C20-only harmonics must equal the closed-form J2 with J2 = -sqrt(5)*C20
    (the executable normalization contract, design Decision 3)."""
    c20 = model.coeffs[(2, 0)][0]
    j2 = -math.sqrt(5.0) * c20
    only_c20 = Egm96Model(coeffs={(2, 0): (c20, 0.0)}, degree=2)
    worst = 0.0
    for pos in (
        np.array([6.778137e6, 1.2e6, 3.4e6]),
        np.array([-2.5e6, 5.9e6, -3.1e6]),
        np.array([4.2e6, -4.2e6, 4.2e6]),
    ):
        a_h = egm96_accel_itrf(only_c20, pos)
        a_j2 = j2_closed_form_accel(pos, model.gm, model.radius, j2)
        worst = max(worst, float(np.max(np.abs(a_h - a_j2)) / np.max(np.abs(a_j2))))
    if worst > 1e-12:
        raise AssertionError(f"C20-only vs closed-form J2 identity broke: rel {worst:.3e}")
    print(f"C20 == J2 identity holds (worst relative {worst:.3e}, J2 = {j2!r})")
    return j2


# ---------------------------------------------------------------------------
# Time and frames (zero-EOP IAU-76/FK5, matching the Rust Iau76Fk5Provider)
# ---------------------------------------------------------------------------


@dataclass
class ArcClock:
    """Julian-date bookkeeping for one arc: epoch + seconds-past-epoch access.

    dUT1 = 0 (UT1 = UTC) and no polar motion — the Rust provider's zero
    defaults. dAT is asserted constant across the arc so TT = UTC + const.
    """

    epoch_iso: str
    time: Time
    tt1: float
    tt2: float
    ut11: float
    ut12: float
    dat: float

    def tt(self, ts: float) -> tuple[float, float]:
        return self.tt1, self.tt2 + ts / 86400.0

    def ut1(self, ts: float) -> tuple[float, float]:
        return self.ut11, self.ut12 + ts / 86400.0

    def jd_utc(self, ts: float) -> float:
        return self.ut11 + self.ut12 + ts / 86400.0


def make_clock(epoch_iso: str, duration_s: float = 0.0) -> ArcClock:
    t = Time(epoch_iso, scale="utc", format="isot")
    t.delta_ut1_utc = 0.0  # zero-EOP: UT1 = UTC
    y, mo, d = (int(x) for x in epoch_iso[:10].split("-"))
    dat0 = erfa.dat(y, mo, d, 0.0)
    t_end = t + (duration_s * u.s)
    iso_end = t_end.isot
    y2, mo2, d2 = (int(x) for x in iso_end[:10].split("-"))
    if erfa.dat(y2, mo2, d2, 0.0) != dat0:
        raise AssertionError(f"leap second inside arc starting {epoch_iso}")
    return ArcClock(
        epoch_iso=epoch_iso,
        time=t,
        tt1=t.tt.jd1,
        tt2=t.tt.jd2,
        ut11=t.jd1,
        ut12=t.jd2,
        dat=float(dat0),
    )


def gcrf_to_itrf_matrix(clock: ArcClock, ts: float) -> np.ndarray:
    """R such that r_itrf = R @ r_gcrf: R3(GAST) @ N(1980) @ P(IAU-76),
    GAST = GMST-1982 + EqEq-1994, zero polar motion — the exact chain of the
    Rust `Iau76Fk5Provider` (erfa pmat76/nutm80/gmst82/eqeq94)."""
    tt1, tt2 = clock.tt(ts)
    ut11, ut12 = clock.ut1(ts)
    prec = erfa.pmat76(tt1, tt2)
    nut = erfa.nutm80(tt1, tt2)
    gast = erfa.gmst82(ut11, ut12) + erfa.eqeq94(tt1, tt2)
    return rot3_frame(gast) @ nut @ prec


# ---------------------------------------------------------------------------
# Ephemerides
# ---------------------------------------------------------------------------


def astropy_body_gcrf_m(body: str, t: Time) -> np.ndarray:
    """Authority Sun/Moon GCRS positions (astropy builtin ephemeris: ERFA
    epv00 for the Sun lineage, ERFA moon98 for the Moon)."""
    with solar_system_ephemeris.set("builtin"):
        return np.asarray(get_body(body, t).cartesian.xyz.to_value(u.m), dtype=float)


def erfa_sun_gcrf_m(clock: ArcClock, ts: float) -> np.ndarray:
    """Geometric geocentric Sun in GCRS (m): minus the heliocentric Earth
    position from erfa.epv00 (BCRS/ICRS axes; TDB ~ TT here, sub-milliarcsec)."""
    tt1, tt2 = clock.tt(ts)
    pvh, _pvb = erfa.epv00(tt1, tt2)
    return -np.asarray(pvh[0], dtype=float) * AU_M


def erfa_moon_gcrf_m(clock: ArcClock, ts: float) -> np.ndarray:
    """Geocentric Moon in GCRS (m) from erfa.moon98 (Meeus-lineage, full
    series — the independent high-accuracy reference, not the Vallado
    truncation)."""
    tt1, tt2 = clock.tt(ts)
    pv = erfa.moon98(tt1, tt2)
    return np.asarray(pv[0], dtype=float) * AU_M


def meeus_sun_ecliptic_of_date(t_tt: float) -> tuple[float, float]:
    """Meeus Ch. 25 low-accuracy Sun (SOURCES['meeus-sun']): geometric true
    longitude (radians, mean equinox of date; ecliptic latitude is zero in
    this method) and Earth-Sun distance (metres).

    Mirrors crates/thresh-core/src/orbital/ephemeris.rs
    `sun_ecliptic_of_date` coefficient for coefficient, including the Horner
    evaluation order, so this IS the series the Rust stack evaluates. The
    time argument is Julian centuries of **TT** since J2000.0 — the same
    argument the Rust `julian_centuries_tt` supplies (Meeus specifies
    TD = TT).
    """
    # Meeus Eq. 25.2 — geometric mean longitude, mean equinox of date.
    mean_longitude_deg = 280.46646 + t_tt * (36000.76983 + t_tt * 0.0003032)
    # Meeus Eq. 25.3 — mean anomaly.
    mean_anomaly_deg = 357.52911 + t_tt * (35999.05029 + t_tt * -0.0001537)
    # Meeus Eq. 25.4 — eccentricity of Earth's orbit.
    eccentricity = 0.016708634 + t_tt * (-0.000042037 + t_tt * -0.0000001267)
    # Equation of center (the unnumbered expression between Eqs. 25.4/25.5).
    m_rad = math.radians(mean_anomaly_deg)
    center_deg = (
        (1.914602 + t_tt * (-0.004817 + t_tt * -0.000014)) * math.sin(m_rad)
        + (0.019993 + t_tt * -0.000101) * math.sin(2.0 * m_rad)
        + 0.000289 * math.sin(3.0 * m_rad)
    )
    true_longitude_deg = mean_longitude_deg + center_deg
    true_anomaly_rad = math.radians(mean_anomaly_deg + center_deg)
    # Meeus Eq. 25.5 — radius vector in AU.
    distance_au = (
        1.000001018
        * (1.0 - eccentricity * eccentricity)
        / (1.0 + eccentricity * math.cos(true_anomaly_rad))
    )
    return math.radians(true_longitude_deg % 360.0), distance_au * AU_M


def meeus_sun_gcrf_m(clock: ArcClock, ts: float) -> np.ndarray:
    """The Rust stack's Sun in GCRF (metres): Meeus Ch. 25 low-accuracy
    series carried through the exact ephemeris.rs chain — series on TT
    Julian centuries -> ecliptic-of-date (zero latitude) -> mean-of-date via
    the IAU-1980 mean obliquity (erfa.obl80, the same polynomial as the Rust
    `mean_obliquity_iau1980`) -> GCRF via the IAU-76 precession transpose."""
    tt1, tt2 = clock.tt(ts)
    t_tt = ((tt1 - 2451545.0) + tt2) / 36525.0
    lon_rad, dist_m = meeus_sun_ecliptic_of_date(t_tt)
    eps = float(erfa.obl80(tt1, tt2))
    r_mod = dist_m * np.array(
        [
            math.cos(lon_rad),
            math.cos(eps) * math.sin(lon_rad),
            math.sin(eps) * math.sin(lon_rad),
        ]
    )
    return of_date_to_gcrf(clock, ts, r_mod)


def verify_meeus_sun_transcription() -> None:
    """Meeus Example 25.a (JDE 2448908.5 TD = 1992 Oct 13.0): true longitude
    199.90987 deg, R = 0.99766 AU — the same fetched spot values the Rust
    unit test `sun_series_matches_meeus_example_25a` pins. A mirror typo
    hard-fails the run before anything is written."""
    t_tt = (2_448_908.5 - 2_451_545.0) / 36_525.0
    lon_rad, dist_m = meeus_sun_ecliptic_of_date(t_tt)
    lon_deg = math.degrees(lon_rad)
    if abs(lon_deg - 199.90987) > 1e-4:
        raise AssertionError(f"Meeus Sun mirror: longitude {lon_deg} != 199.90987 deg")
    if abs(dist_m / AU_M - 0.99766) > 1e-5:
        raise AssertionError(f"Meeus Sun mirror: radius {dist_m / AU_M} != 0.99766 AU")


def vallado_sun_of_date(jd_utc: float) -> np.ndarray:
    """Vallado Algorithm 29 (sun.m, SOURCES['vallado-sun']) — position vector
    on the mean equator/equinox of date, metres. Series constants verbatim
    from the fetched sun.m. Reference-only: the Rust Sun implements the
    sibling Meeus Ch. 25 series (see meeus_sun_gcrf_m); Alg 29 deltas are
    recorded as informational and enter no tolerance."""
    tut1 = (jd_utc - 2451545.0) / 36525.0
    meanlong = math.fmod(280.460 + 36000.771285 * tut1, 360.0)
    ttdb = tut1
    meananomaly = math.fmod(math.radians(357.528 + 35999.0509575 * ttdb), 2.0 * math.pi)
    if meananomaly < 0.0:
        meananomaly += 2.0 * math.pi
    eclplong = math.fmod(
        meanlong
        + 1.914666471 * math.sin(meananomaly)
        + 0.019994643 * math.sin(2.0 * meananomaly),
        360.0,
    )
    obliquity = math.radians(23.439291 - 0.0130042 * ttdb)
    eclplong_r = math.radians(eclplong)
    magr_au = (
        1.000140612
        - 0.016708617 * math.cos(meananomaly)
        - 0.000139589 * math.cos(2.0 * meananomaly)
    )
    return magr_au * AU_M * np.array(
        [
            math.cos(eclplong_r),
            math.cos(obliquity) * math.sin(eclplong_r),
            math.sin(obliquity) * math.sin(eclplong_r),
        ]
    )


def vallado_moon_of_date(jd_utc: float) -> np.ndarray:
    """Vallado Algorithm 31 (moon.m, SOURCES['vallado-moon']) — position on
    the mean equator/equinox of date, metres (Earth radii scaled by the
    companion-code re = 6378.1363 km). Series constants verbatim from the
    fetched moon.m."""
    d2r = math.pi / 180.0
    ttdb = (jd_utc - 2451545.0) / 36525.0
    eclplong = (
        218.32
        + 481267.8813 * ttdb
        + 6.29 * math.sin((134.9 + 477198.85 * ttdb) * d2r)
        - 1.27 * math.sin((259.2 - 413335.38 * ttdb) * d2r)
        + 0.66 * math.sin((235.7 + 890534.23 * ttdb) * d2r)
        + 0.21 * math.sin((269.9 + 954397.70 * ttdb) * d2r)
        - 0.19 * math.sin((357.5 + 35999.05 * ttdb) * d2r)
        - 0.11 * math.sin((186.6 + 966404.05 * ttdb) * d2r)
    )
    eclplat = (
        5.13 * math.sin((93.3 + 483202.03 * ttdb) * d2r)
        + 0.28 * math.sin((228.2 + 960400.87 * ttdb) * d2r)
        - 0.28 * math.sin((318.3 + 6003.18 * ttdb) * d2r)
        - 0.17 * math.sin((217.6 - 407332.20 * ttdb) * d2r)
    )
    hzparal = (
        0.9508
        + 0.0518 * math.cos((134.9 + 477198.85 * ttdb) * d2r)
        + 0.0095 * math.cos((259.2 - 413335.38 * ttdb) * d2r)
        + 0.0078 * math.cos((235.7 + 890534.23 * ttdb) * d2r)
        + 0.0028 * math.cos((269.9 + 954397.70 * ttdb) * d2r)
    )
    eclplong_r = math.fmod(eclplong * d2r, 2.0 * math.pi)
    eclplat_r = math.fmod(eclplat * d2r, 2.0 * math.pi)
    hzparal_r = math.fmod(hzparal * d2r, 2.0 * math.pi)
    obliquity = (23.439291 - 0.0130042 * ttdb) * d2r
    li = math.cos(eclplat_r) * math.cos(eclplong_r)
    mi = math.cos(obliquity) * math.cos(eclplat_r) * math.sin(eclplong_r) - math.sin(
        obliquity
    ) * math.sin(eclplat_r)
    ni = math.sin(obliquity) * math.cos(eclplat_r) * math.sin(eclplong_r) + math.cos(
        obliquity
    ) * math.sin(eclplat_r)
    magr = (1.0 / math.sin(hzparal_r)) * RE_VALLADO_M
    return magr * np.array([li, mi, ni])


def of_date_to_gcrf(clock: ArcClock, ts: float, r_of_date: np.ndarray) -> np.ndarray:
    """Mean-of-date -> GCRF via the transpose of the IAU-76 precession matrix
    (the Vallado low-precision vectors use the mean obliquity, i.e. the mean
    equator/equinox of date; the interpretation is recorded in provenance)."""
    tt1, tt2 = clock.tt(ts)
    prec = erfa.pmat76(tt1, tt2)
    return prec.T @ r_of_date


def vallado_sun_gcrf_m(clock: ArcClock, ts: float) -> np.ndarray:
    return of_date_to_gcrf(clock, ts, vallado_sun_of_date(clock.jd_utc(ts)))


def vallado_moon_gcrf_m(clock: ArcClock, ts: float) -> np.ndarray:
    return of_date_to_gcrf(clock, ts, vallado_moon_of_date(clock.jd_utc(ts)))


def verify_fast_ephemeris(epochs: list[str]) -> None:
    """The trajectory stack's direct-ERFA Sun/Moon must agree with the astropy
    authority to well under the analytic-series tolerances."""
    for iso in epochs:
        clock = make_clock(iso)
        sun_ref = astropy_body_gcrf_m("sun", clock.time)
        moon_ref = astropy_body_gcrf_m("moon", clock.time)
        d_sun = angle_between_rad(erfa_sun_gcrf_m(clock, 0.0), sun_ref) / ARCSEC
        d_moon = angle_between_rad(erfa_moon_gcrf_m(clock, 0.0), moon_ref) / ARCSEC
        if d_sun > 60.0 or d_moon > 10.0:
            raise AssertionError(
                f"fast ERFA ephemeris drifted from astropy at {iso}: "
                f"sun {d_sun:.2f} arcsec, moon {d_moon:.2f} arcsec"
            )


# ---------------------------------------------------------------------------
# Harris-Priester density
# ---------------------------------------------------------------------------


def hp_density_at_height(h_m: float, cos_psi: float) -> float:
    """Harris-Priester density (kg/m^3) at height h with bulge angle psi.

    Formula per SOURCES['orekit-hp'] (Montenbruck & Gill lineage):
    exponential interpolation between table nodes,
    rho = rho_min + (rho_max - rho_min) * cos^n(psi/2) with
    cos^2(psi/2) = (1 + cos psi)/2 and n = 2 (design Decision 4).
    Clamp-with-doc outside the band: 0 above 1000 km; below 100 km this
    generator refuses (the golden arcs never descend there).
    """
    if h_m > HP_TABLE[-1][0]:
        return 0.0
    if h_m < HP_TABLE[0][0]:
        raise AssertionError(f"altitude {h_m} m below the Harris-Priester band")
    i = 0
    while i < len(HP_TABLE) - 2 and h_m > HP_TABLE[i + 1][0]:
        i += 1
    h_i, min_i, max_i = HP_TABLE[i]
    h_j, min_j, max_j = HP_TABLE[i + 1]
    dh = (h_i - h_m) / (h_i - h_j)
    rho_min = min_i * (min_j / min_i) ** dh
    rho_max = max_i * (max_j / max_i) ** dh
    c2 = max(0.0, (1.0 + cos_psi) / 2.0)
    cos_pow = c2 ** (HP_BULGE_EXPONENT / 2.0) if HP_BULGE_EXPONENT != 2 else c2
    return rho_min + (rho_max - rho_min) * cos_pow


def hp_density_itrf(r_itrf: np.ndarray, apex_itrf: np.ndarray) -> float:
    """Density at an ITRF position: geodetic (WGS-84) height via erfa.gc2gd,
    bulge angle against the supplied apex direction."""
    _lon, _lat, height = erfa.gc2gd(1, np.asarray(r_itrf, dtype=float))  # 1 = WGS84
    cos_psi = float(np.dot(unit(r_itrf), unit(apex_itrf)))
    return hp_density_at_height(float(height), cos_psi)


# ---------------------------------------------------------------------------
# Shadow + SRP + third body
# ---------------------------------------------------------------------------


def cylindrical_shadow_factor(r_gcrf: np.ndarray, sun_gcrf: np.ndarray) -> float:
    """1 in sunlight, exactly 0 inside the cylindrical umbra: the spacecraft
    is shadowed iff it is on the anti-Sun side and within R_SHADOW_M of the
    Earth-Sun axis."""
    s_hat = unit(sun_gcrf)
    along = float(np.dot(r_gcrf, s_hat))
    if along >= 0.0:
        return 1.0
    transverse = r_gcrf - along * s_hat
    return 0.0 if float(np.linalg.norm(transverse)) < R_SHADOW_M else 1.0


def srp_accel(r_gcrf: np.ndarray, sun_gcrf: np.ndarray, cr_a_over_m: float) -> np.ndarray:
    """Cannonball SRP: a = nu * (TSI/c) * (AU/d)^2 * (Cr A/m) * unit(r - r_sun),
    d = |r - r_sun| (Sun -> spacecraft), nu = cylindrical shadow factor."""
    nu_f = cylindrical_shadow_factor(r_gcrf, sun_gcrf)
    if nu_f == 0.0:
        return np.zeros(3)
    d = r_gcrf - sun_gcrf
    dist = float(np.linalg.norm(d))
    return nu_f * P_SRP_1AU * (AU_M / dist) ** 2 * cr_a_over_m * (d / dist)


def third_body_accel(r_gcrf: np.ndarray, body_gcrf: np.ndarray, gm: float) -> np.ndarray:
    """Point-mass third body, direct minus indirect:
    a = GM_b * ((r_b - r)/|r_b - r|^3 - r_b/|r_b|^3)."""
    d = body_gcrf - r_gcrf
    dn = float(np.linalg.norm(d))
    bn = float(np.linalg.norm(body_gcrf))
    return gm * (d / dn**3 - body_gcrf / bn**3)


# ---------------------------------------------------------------------------
# Arc definitions and the force-stack RHS
# ---------------------------------------------------------------------------


def kepler_to_rv(
    mu: float, a: float, e: float, inc: float, raan: float, argp: float, nu_ta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Classical elements -> GCRF cartesian (angles in radians), with
    self-checks (radius equation, vis-viva, angular momentum, inclination)."""
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * math.cos(nu_ta))
    r_pqw = np.array([r_mag * math.cos(nu_ta), r_mag * math.sin(nu_ta), 0.0])
    v_pqw = math.sqrt(mu / p) * np.array([-math.sin(nu_ta), e + math.cos(nu_ta), 0.0])
    rotm = rotz_active(raan) @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(inc), -math.sin(inc)],
            [0.0, math.sin(inc), math.cos(inc)],
        ]
    ) @ rotz_active(argp)
    r = rotm @ r_pqw
    v = rotm @ v_pqw
    if abs(np.linalg.norm(r) - r_mag) > 1e-6:
        raise AssertionError("kepler_to_rv radius check failed")
    vis_viva = mu * (2.0 / r_mag - 1.0 / a)
    if abs(float(np.dot(v, v)) - vis_viva) / vis_viva > 1e-12:
        raise AssertionError("kepler_to_rv vis-viva check failed")
    h_vec = np.cross(r, v)
    if abs(float(np.linalg.norm(h_vec)) - math.sqrt(mu * p)) / math.sqrt(mu * p) > 1e-12:
        raise AssertionError("kepler_to_rv angular-momentum check failed")
    if abs(float(h_vec[2] / np.linalg.norm(h_vec)) - math.cos(inc)) > 1e-12:
        raise AssertionError("kepler_to_rv inclination check failed")
    return r, v


@dataclass
class ArcSpec:
    name: str
    description: str
    epoch_iso: str
    duration_s: float
    grid_step_s: float
    r0: np.ndarray
    v0: np.ndarray
    gravity: str  # "egm96" | "two_body"
    drag_inv_beta: float | None = None  # Cd*A/m, m^2/kg
    srp_cr_a_over_m: float | None = None  # Cr*A/m, m^2/kg
    third_body_sun: bool = False
    third_body_moon: bool = False
    extra_notes: list[str] = field(default_factory=list)


def make_rhs(
    spec: ArcSpec,
    clock: ArcClock,
    model: Egm96Model,
    ephem: str = "erfa",
    apex_frame: str = "itrf",
):
    """Build the time-aware acceleration RHS for solve_ivp.

    ephem: 'erfa' (baseline authority stack) or 'analytic' (swap in the
    analytic series the Rust stack implements — Meeus Ch. 25 Sun mirrored
    from ephemeris.rs plus Vallado Alg 31 Moon — used to measure ephemeris
    sensitivity).
    apex_frame: where the +30 deg bulge-apex rotation about +z is applied —
    'itrf' (baseline, Orekit/M&G convention) or 'gcrf' (convention-sensitivity
    variant).
    """
    sun_fn = erfa_sun_gcrf_m if ephem == "erfa" else meeus_sun_gcrf_m
    moon_fn = erfa_moon_gcrf_m if ephem == "erfa" else vallado_moon_gcrf_m
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    lag = math.radians(HP_APEX_LAG_DEG)
    needs_sun = spec.drag_inv_beta is not None or spec.srp_cr_a_over_m is not None
    needs_sun = needs_sun or spec.third_body_sun

    def rhs(ts: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        sun = sun_fn(clock, ts) if needs_sun else None
        a = np.zeros(3)

        rot = None
        if spec.gravity == "egm96" or spec.drag_inv_beta is not None:
            rot = gcrf_to_itrf_matrix(clock, ts)
        if spec.gravity == "egm96":
            r_itrf = rot @ r
            a += rot.T @ egm96_accel_itrf(model, r_itrf)
        else:
            rn = float(np.linalg.norm(r))
            a += -model.gm * r / rn**3

        if spec.drag_inv_beta is not None:
            r_itrf = rot @ r
            v_itrf = rot @ v - np.cross(omega_vec, r_itrf)
            if apex_frame == "itrf":
                apex_itrf = rotz_active(lag) @ unit(rot @ sun)
            else:
                apex_itrf = rot @ (rotz_active(lag) @ unit(sun))
            rho = hp_density_itrf(r_itrf, apex_itrf)
            vmag = float(np.linalg.norm(v_itrf))
            a += rot.T @ (-0.5 * rho * vmag * spec.drag_inv_beta * v_itrf)

        if spec.srp_cr_a_over_m is not None:
            a += srp_accel(r, sun, spec.srp_cr_a_over_m)

        if spec.third_body_sun:
            a += third_body_accel(r, sun, GM_SUN)
        if spec.third_body_moon:
            a += third_body_accel(r, moon_fn(clock, ts), GM_MOON)

        return np.concatenate((v, a))

    return rhs


MAX_STEP_S = 300.0


def integrate_arc(
    spec: ArcSpec,
    clock: ArcClock,
    model: Egm96Model,
    rtol: float,
    ephem: str = "erfa",
    apex_frame: str = "itrf",
    dense: bool = False,
):
    grid = np.arange(0.0, spec.duration_s + 0.5 * spec.grid_step_s, spec.grid_step_s)
    rhs = make_rhs(spec, clock, model, ephem=ephem, apex_frame=apex_frame)
    y0 = np.concatenate((spec.r0, spec.v0))
    sol = solve_ivp(
        rhs,
        (0.0, spec.duration_s),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=rtol,
        max_step=MAX_STEP_S,
        t_eval=grid,
        dense_output=dense,
    )
    if not sol.success:
        raise AssertionError(f"solve_ivp failed on {spec.name} (rtol={rtol}): {sol.message}")
    return sol


def max_state_delta(sol_a, sol_b) -> tuple[float, float]:
    dr = np.max(np.linalg.norm(sol_a.y[:3] - sol_b.y[:3], axis=0))
    dv = np.max(np.linalg.norm(sol_a.y[3:] - sol_b.y[3:], axis=0))
    return float(dr), float(dv)


def round_up_2sig(x: float) -> float:
    """Round up to two significant figures (tolerances only ever grow)."""
    if x <= 0.0:
        return 0.0
    exp = math.floor(math.log10(x))
    scale = 10.0 ** (exp - 1)
    # Re-parse through the 2-significant-digit decimal form so the recorded
    # tolerance prints cleanly (e.g. 0.00015, not 0.00015000000000000001).
    return float(f"{math.ceil(x / scale) * scale:.1e}")


def count_shadow_crossings(
    spec: ArcSpec, clock: ArcClock, sol_dense, step_s: float = 10.0
) -> dict:
    """Count 0<->1 transitions of the cylindrical shadow factor along the
    baseline trajectory (dense output sampled every `step_s`)."""
    times = np.arange(0.0, spec.duration_s + 0.5 * step_s, step_s)
    factors = []
    for ts in times:
        r = sol_dense.sol(ts)[:3]
        factors.append(cylindrical_shadow_factor(r, erfa_sun_gcrf_m(clock, float(ts))))
    transitions = [
        {"t_s": float(times[k + 1]), "into_shadow": factors[k + 1] == 0.0}
        for k in range(len(factors) - 1)
        if factors[k] != factors[k + 1]
    ]
    return {
        "count": len(transitions),
        "resolution_s": step_s,
        "transitions": transitions,
        "note": (
            "0<->1 transitions of the cylindrical shadow factor sampled every "
            f"{step_s:.0f} s along the baseline trajectory; entry/exit times are "
            "accurate to the sampling resolution."
        ),
    }


# ---------------------------------------------------------------------------
# Fixture assembly — components
# ---------------------------------------------------------------------------

EPHEMERIS_EPOCHS = [
    "2025-09-01T00:00:00.000",
    "2026-02-01T06:30:00.000",  # LEO arc epoch
    "2026-03-20T12:00:00.000",  # SRP arc epoch
    "2026-05-10T00:00:00.000",  # MEO arc epoch
    "2026-11-15T18:00:00.000",
    "2027-06-30T12:00:00.000",
]


def build_ephemeris_fixture() -> tuple[dict, dict]:
    """Sun/Moon authority positions + measured errors of the Rust series."""
    entries = []
    worst = {"sun_arcsec": 0.0, "moon_arcsec": 0.0, "sun_dist_rel": 0.0, "moon_dist_rel": 0.0}
    for iso in EPHEMERIS_EPOCHS:
        clock = make_clock(iso)
        sun_ref = astropy_body_gcrf_m("sun", clock.time)
        moon_ref = astropy_body_gcrf_m("moon", clock.time)
        sun_val = meeus_sun_gcrf_m(clock, 0.0)
        moon_val = vallado_moon_gcrf_m(clock, 0.0)
        sun_err = angle_between_rad(sun_val, sun_ref) / ARCSEC
        moon_err = angle_between_rad(moon_val, moon_ref) / ARCSEC
        sun_dist = abs(np.linalg.norm(sun_val) - np.linalg.norm(sun_ref)) / np.linalg.norm(
            sun_ref
        )
        moon_dist = abs(np.linalg.norm(moon_val) - np.linalg.norm(moon_ref)) / np.linalg.norm(
            moon_ref
        )
        sun_v29_err = angle_between_rad(vallado_sun_gcrf_m(clock, 0.0), sun_ref) / ARCSEC
        worst["sun_arcsec"] = max(worst["sun_arcsec"], sun_err)
        worst["moon_arcsec"] = max(worst["moon_arcsec"], moon_err)
        worst["sun_dist_rel"] = max(worst["sun_dist_rel"], float(sun_dist))
        worst["moon_dist_rel"] = max(worst["moon_dist_rel"], float(moon_dist))
        entries.append(
            {
                "epoch_utc": iso + "Z",
                "sun_gcrf_m": vec(sun_ref),
                "moon_gcrf_m": vec(moon_ref),
                "measured_rust_series_error": {
                    "sun_arcsec": round(sun_err, 3),
                    "moon_arcsec": round(moon_err, 3),
                    "sun_distance_rel": float(f"{sun_dist:.3e}"),
                    "moon_distance_rel": float(f"{moon_dist:.3e}"),
                    "note": (
                        "this generator's mirror of the series the Rust "
                        "ephemerides implement: Meeus Ch. 25 Sun (mirrored from "
                        "ephemeris.rs, TT centuries, obl80 obliquity) and "
                        "Vallado Alg 31 Moon (coefficient-identical to the Rust "
                        "transcription) — the tolerance basis"
                    ),
                },
                "informational_vallado_alg29_sun_error": {
                    "sun_arcsec": round(sun_v29_err, 3),
                    "note": (
                        "reference-only: the sibling Vallado Alg 29 Sun series, "
                        "NOT the series the Rust stack implements; enters no "
                        "tolerance"
                    ),
                },
            }
        )
    # Sanity gates: the mirrored series must sit inside the design's
    # documented accuracy classes (Sun ~0.01 deg = 36", Moon ~0.3 deg = 1080")
    # with margin; larger means a series transcription bug.
    if worst["sun_arcsec"] > 108.0 or worst["moon_arcsec"] > 3240.0:
        raise AssertionError(f"analytic series error out of class: {worst}")
    tolerances = {
        "sun_angular_arcsec": round_up_2sig(3.0 * worst["sun_arcsec"]),
        "moon_angular_arcsec": round_up_2sig(3.0 * worst["moon_arcsec"]),
        "sun_distance_rel": round_up_2sig(3.0 * worst["sun_dist_rel"]),
        "moon_distance_rel": round_up_2sig(3.0 * worst["moon_dist_rel"]),
        "basis": (
            "3x the worst measured error of this generator's mirror of the "
            "series the Rust ephemerides implement, against the astropy "
            "authority over the six fixture epochs. Sun: Meeus Ch. 25 "
            "low-accuracy series mirrored coefficient-for-coefficient from "
            "crates/thresh-core/src/orbital/ephemeris.rs, evaluated on TT "
            "Julian centuries with the IAU-1980 (obl80) mean obliquity and the "
            "IAU-76 precession transpose — the exact Rust chain, so the "
            "measured Sun error is the Rust stack's own error at these epochs "
            "(residual mirror differences are numerical noise). Moon: Vallado "
            "Alg 31 per the fetched moon.m, coefficient-identical to the Rust "
            "transcription but evaluated on the UTC Julian date with moon.m's "
            "truncated obliquity as fetched; the Rust evaluates TT and obl80 — "
            "TT-UTC = 69.184 s is ~38 arcsec of lunar motion, well inside one "
            "series error and covered by the 3x margin. The 3x margin "
            "otherwise buys headroom against future-epoch drift and "
            "authority-convention differences (geometric series vs the "
            "authority's light-time-corrected positions)."
        ),
    }
    fixture = {
        "case": "sun-moon-ephemeris",
        "description": (
            "Sun and Moon geocentric GCRF positions from an independent "
            "authority (astropy builtin ephemeris: ERFA epv00 / moon98) at six "
            "epochs, with the measured angular error of the analytic series "
            "the Rust ephemerides implement (Sun: Meeus Ch. 25 low-accuracy "
            "series mirrored from crates/thresh-core/src/orbital/ephemeris.rs; "
            "Moon: Vallado Alg 31). Spec scenario: 'Sun and Moon positions "
            "match the independent authority'."
        ),
        "frame": "GCRF (astropy GCRS; frame bias vs IAU-76 J2000 ~0.023 arcsec, "
        "far below every tolerance here)",
        "authority": "astropy get_body(..., ephemeris='builtin') — positions in metres",
        "epochs": entries,
        "tolerances": tolerances,
        "sources": {
            "sun_series": SOURCES["meeus-sun"],
            "moon_series": SOURCES["vallado-moon"],
            "sun_series_informational_reference": SOURCES["vallado-sun"],
        },
    }
    return fixture, tolerances


EGM96_SPOT_POSITIONS = [
    ("leo-equatorial", [6_778_137.0, 0.0, 0.0]),
    (
        "leo-mid-latitude",
        [
            6_778_137.0 * math.cos(math.radians(45.0)) * math.cos(math.radians(60.0)),
            6_778_137.0 * math.cos(math.radians(45.0)) * math.sin(math.radians(60.0)),
            6_778_137.0 * math.sin(math.radians(45.0)),
        ],
    ),
    (
        "high-latitude-1000km",
        [
            7_378_137.0 * math.cos(math.radians(80.0)) * math.cos(math.radians(-120.0)),
            7_378_137.0 * math.cos(math.radians(80.0)) * math.sin(math.radians(-120.0)),
            7_378_137.0 * math.sin(math.radians(80.0)),
        ],
    ),
    (
        "meo",
        [
            26_560_000.0 * math.cos(math.radians(-30.0)) * math.cos(math.radians(150.0)),
            26_560_000.0 * math.cos(math.radians(-30.0)) * math.sin(math.radians(150.0)),
            26_560_000.0 * math.sin(math.radians(-30.0)),
        ],
    ),
]


def build_egm96_fixture(model: Egm96Model, egm96_origin: str) -> dict:
    spots = []
    worst_mp = 0.0
    worst_lap = 0.0
    for name, pos in EGM96_SPOT_POSITIONS:
        r = np.array(pos)
        a_total = egm96_accel_itrf(model, r)
        a_central = -model.gm * r / float(np.linalg.norm(r)) ** 3
        a_mp = mp_gradient_accel(model, r)
        rel_mp = float(np.max(np.abs(a_total - a_mp)) / np.max(np.abs(a_total)))
        lap = mp_laplacian_residual(model, r)
        worst_mp = max(worst_mp, rel_mp)
        worst_lap = max(worst_lap, lap)
        spots.append(
            {
                "name": name,
                "position_itrf_m": vec(r),
                "acceleration_total_m_s2": vec(a_total),
                "acceleration_central_m_s2": vec(a_central),
                "acceleration_perturbation_m_s2": vec(a_total - a_central),
                "generator_self_check": {
                    "mpmath_gradient_rel_delta": float(f"{rel_mp:.3e}"),
                    "laplacian_residual_rel": float(f"{lap:.3e}"),
                },
            }
        )
    if worst_mp > 5e-12:
        raise AssertionError(f"analytic vs mpmath EGM96 gradient drifted: {worst_mp:.3e}")
    if worst_lap > 1e-9:
        raise AssertionError(f"EGM96 field not harmonic (recursion bug?): {worst_lap:.3e}")
    return {
        "case": "egm96-spot-accelerations",
        "description": (
            "EGM96 12x12 gravitational accelerations at fixed ITRF positions, "
            "computed by this generator's own fully-normalized Legendre "
            "evaluation of the fetched coefficient file (independent of the "
            "Rust implementation). Spec scenario: 'Harmonic acceleration "
            "matches independent evaluation'."
        ),
        "model": {
            "name": "EGM96",
            "degree": model.degree,
            "order": model.degree,
            "gm_m3_s2": model.gm,
            "reference_radius_m": model.radius,
            "tide_system": "tide_free",
            "normalization": "fully normalized (geodetic, no Condon-Shortley phase)",
            "includes_central_term": True,
            "degree_1_terms": "absent (geocentric origin)",
            "coefficient_source": SOURCES["icgem-egm96"],
            "coefficient_cross_check": SOURCES["nga-egm96"],
            "fetched_from": egm96_origin,
            "coefficients": [
                {"n": n, "m": m, "c_bar": c, "s_bar": s}
                for (n, m), (c, s) in sorted(model.coeffs.items())
            ],
        },
        "evaluation": {
            "potential": SOURCES["barthelmes"],
            "legendre_recursion": SOURCES["legendre-recursion"],
            "gradient": (
                "analytic spherical gradient (r, geocentric latitude, longitude "
                "components), cross-checked per spot against 60-digit mpmath "
                "central differences of the same potential and against a "
                "Laplacian-harmonicity residual; C20-only evaluation matches "
                "the closed-form J2 to <= 1e-12 relative (see PROVENANCE.md)."
            ),
        },
        "tolerance": {
            "relative_per_component": 1e-9,
            "basis": (
                "both stacks evaluate the identical truncated model from the "
                "same verified coefficients; this generator's float64 "
                "evaluation agrees with 60-digit arithmetic to "
                f"{worst_mp:.1e} relative (measured, worst spot), so 1e-9 "
                "leaves >100x headroom for the Rust recursion's own float64 "
                "noise while catching any single-coefficient or normalization "
                "error (smallest such error observed >= 1e-5 relative)."
            ),
        },
        "spots": spots,
    }


def build_hp_fixture() -> dict:
    # Worked examples: node-exact altitudes at apex/anti-apex/quadrature plus
    # one off-node altitude that exercises the exponential interpolation.
    examples = []
    for h_m, cos_psi, label in (
        (400_000.0, 1.0, "node 400 km, bulge apex (psi = 0): rho = rho_max"),
        (400_000.0, -1.0, "node 400 km, anti-apex (psi = 180 deg): rho = rho_min"),
        (400_000.0, 0.0, "node 400 km, psi = 90 deg: rho_min + (rho_max-rho_min)/2"),
        (410_000.0, 1.0, "off-node 410 km, apex: exponential interpolation check"),
        (410_000.0, -1.0, "off-node 410 km, anti-apex: exponential interpolation check"),
    ):
        examples.append(
            {
                "height_m": h_m,
                "cos_psi": cos_psi,
                "density_kg_m3": hp_density_at_height(h_m, cos_psi),
                "note": label,
            }
        )
    return {
        "case": "harris-priester-table",
        "description": (
            "The published Harris-Priester min/max density table (mean solar "
            "activity, 100-1000 km) with the interpolation/bulge convention. "
            "Spec scenario: 'Published table values reproduced'."
        ),
        "sources": {
            "table_primary": SOURCES["orekit-hp"],
            "table_cross_check": SOURCES["satkit-hp"],
            "agreement": "all 50 rows identical between the two fetched transcriptions",
        },
        "convention": {
            "bulge_exponent_n": HP_BULGE_EXPONENT,
            "apex": (
                f"Sun direction rotated +{HP_APEX_LAG_DEG:.0f} deg about the ITRF "
                "+z axis (right ascension + 30 deg, declination preserved)"
            ),
            "density_formula": (
                "rho(h, psi) = rho_min(h) + (rho_max(h) - rho_min(h)) * "
                "cos^n(psi/2), cos^2(psi/2) = (1 + cos psi)/2"
            ),
            "height_interpolation": (
                "exponential between bracketing nodes: rho_x(h) = rho_x(h_i) * "
                "(rho_x(h_{i+1})/rho_x(h_i))^((h_i - h)/(h_i - h_{i+1})) for "
                "x in {min, max}"
            ),
            "altitude_definition": (
                "geodetic height above the WGS-84 ellipsoid (erfa.gc2gd "
                "equivalent) in the trajectory stack; the table nodes and the "
                "worked examples below take height directly"
            ),
            "clamp": "rho = 0 above 1000 km; below 100 km out of the model's band",
        },
        "table": [
            {"altitude_m": h, "rho_min_kg_m3": lo, "rho_max_kg_m3": hi}
            for h, lo, hi in HP_TABLE
        ],
        "worked_examples": {
            "note": (
                "tool-computed from the table with the convention above "
                "(generator-evaluated, not published values)"
            ),
            "values": examples,
        },
    }


# ---------------------------------------------------------------------------
# Fixture assembly — trajectory arcs
# ---------------------------------------------------------------------------


def define_arcs(model: Egm96Model) -> list[ArcSpec]:
    mu = model.gm
    # LEO ~400 km, full force stack, two orbital periods.
    r0, v0 = kepler_to_rv(
        mu,
        a=6_778_137.0,
        e=0.001,
        inc=math.radians(51.6),
        raan=math.radians(40.0),
        argp=math.radians(20.0),
        nu_ta=0.0,
    )
    leo = ArcSpec(
        name="leo-full-force",
        description=(
            "LEO ~400 km, full force stack: EGM96 12x12 gravity, "
            "Harris-Priester drag, cannonball SRP with cylindrical shadow, "
            "Sun+Moon third body. Two orbital periods."
        ),
        epoch_iso="2026-02-01T06:30:00.000",
        duration_s=11_100.0,
        grid_step_s=60.0,
        r0=r0,
        v0=v0,
        gravity="egm96",
        drag_inv_beta=0.011,
        srp_cr_a_over_m=0.0065,
        third_body_sun=True,
        third_body_moon=True,
    )
    # MEO (GPS-class), harmonics + lunisolar, half a day.
    r0, v0 = kepler_to_rv(
        mu,
        a=26_560_000.0,
        e=0.005,
        inc=math.radians(55.0),
        raan=math.radians(120.0),
        argp=math.radians(30.0),
        nu_ta=math.radians(45.0),
    )
    meo = ArcSpec(
        name="meo-harmonics-lunisolar",
        description=(
            "MEO (GPS-class), EGM96 12x12 gravity plus Sun+Moon third body "
            "(no drag, no SRP). 12 hours."
        ),
        epoch_iso="2026-05-10T00:00:00.000",
        duration_s=43_200.0,
        grid_step_s=300.0,
        r0=r0,
        v0=v0,
        gravity="egm96",
        third_body_sun=True,
        third_body_moon=True,
    )
    # SRP-dominant GTO-class arc engineered to sweep through the Earth
    # shadow near apogee: perigee toward the epoch Sun, apogee anti-Sun.
    clock = make_clock("2026-03-20T12:00:00.000")
    s_hat = unit(erfa_sun_gcrf_m(clock, 0.0))
    r_p, r_a = 6_878_137.0, 42_164_000.0
    a_srp = 0.5 * (r_p + r_a)
    v_p = math.sqrt(mu * (2.0 / r_p - 1.0 / a_srp))
    q_hat = unit(np.cross(np.array([0.0, 0.0, 1.0]), s_hat))
    srp_arc = ArcSpec(
        name="srp-shadow-crossing",
        description=(
            "SRP-dominant GTO-class arc: two-body gravity plus cannonball SRP "
            "with cylindrical Earth shadow only (SRP is the sole "
            "perturbation). Perigee points at the epoch Sun so the arc sweeps "
            "through the shadow cylinder around apogee; one orbital period."
        ),
        epoch_iso="2026-03-20T12:00:00.000",
        duration_s=38_400.0,
        grid_step_s=120.0,
        r0=r_p * s_hat,
        v0=v_p * q_hat,
        gravity="two_body",
        srp_cr_a_over_m=0.05,
        extra_notes=[
            "initial state constructed from the epoch Sun direction "
            "(perigee radius 6878137 m sunward, apogee radius 42164000 m "
            "anti-Sun, velocity along z x sun_hat); the r0/v0 numbers above "
            "are the contract — the construction is descriptive only."
        ],
    )
    return [leo, meo, srp_arc]


def force_config_block(spec: ArcSpec, model: Egm96Model, j2_from_c20: float) -> dict:
    """Everything the Rust envelope test needs to mirror the arc exactly."""
    gravity: dict = {"model": spec.gravity}
    if spec.gravity == "egm96":
        gravity.update(
            {
                "degree": model.degree,
                "order": model.degree,
                "gm_m3_s2": model.gm,
                "reference_radius_m": model.radius,
                "includes_central_term": True,
                "coefficients": "see egm96-spot-accelerations.json (identical subset)",
                "evaluation_frame": "ITRF via the zero-EOP IAU-76/FK5 chain",
            }
        )
    else:
        gravity.update({"gm_m3_s2": model.gm, "note": "point-mass central body only"})
    drag = None
    if spec.drag_inv_beta is not None:
        drag = {
            "model": "harris_priester",
            "inv_beta_m2_per_kg": spec.drag_inv_beta,
            "inv_beta_definition": "Cd * A / m",
            "acceleration": "a = -0.5 * rho * |v_rel| * v_rel * inv_beta",
            "relative_velocity": (
                "ITRF-frame velocity: v_itrf = R v_gcrf - omega x (R r_gcrf), "
                f"omega = [0, 0, {OMEGA_EARTH!r}] rad/s "
                "(thresh_core::eci::EARTH_ROTATION_RATE); acceleration rotated "
                "back with R^T"
            ),
            "density": "see harris-priester-table.json (table + full convention)",
            "bulge_exponent_n": HP_BULGE_EXPONENT,
            "apex_lag_deg": HP_APEX_LAG_DEG,
            "sun_ephemeris": "each stack's own (see tolerance.sensitivities)",
        }
    srp = None
    if spec.srp_cr_a_over_m is not None:
        srp = {
            "model": "cannonball_cylindrical_shadow",
            "cr_area_over_mass_m2_per_kg": spec.srp_cr_a_over_m,
            "pressure_at_1au_n_m2": P_SRP_1AU,
            "pressure_convention": (
                "P(d) = (TSI/c) * (au/d)^2 with TSI = 1361 W/m^2 (IAU 2015 "
                "B3), c = 299792458 m/s, au = 149597870700 m; d = "
                "Sun->spacecraft distance; direction unit(r - r_sun)"
            ),
            "shadow": {
                "model": "cylinder",
                "radius_m": R_SHADOW_M,
                "rule": (
                    "factor = 0 iff dot(r, sun_hat) < 0 and "
                    "|r - dot(r, sun_hat) sun_hat| < radius, else 1"
                ),
            },
        }
    third_body = {
        "sun": spec.third_body_sun,
        "moon": spec.third_body_moon,
    }
    if spec.third_body_sun or spec.third_body_moon:
        third_body.update(
            {
                "gm_sun_m3_s2": GM_SUN,
                "gm_moon_m3_s2": GM_MOON,
                "gm_source": SOURCES["jpl-astro-par"],
                "formulation": "direct minus indirect: GM_b ((r_b - r)/|r_b - r|^3 - r_b/|r_b|^3)",
                "ephemeris": (
                    "each stack's own Sun/Moon (generator: ERFA epv00/moon98; "
                    "Rust: Meeus Ch. 25 Sun + Vallado Alg 31 Moon analytic "
                    "series) — the swap effect is measured in "
                    "tolerance.sensitivities.ephemeris_swap"
                ),
            }
        )
    return {
        "gravity": gravity,
        "drag": drag,
        "srp": srp,
        "third_body": third_body,
        "frames": {
            "provider": "IAU-76/FK5, zero EOP (dUT1 = 0, no polar motion)",
            "gcrf_to_itrf": "R3(GAST[GMST-1982 + EqEq-1994]) . N(IAU-1980) . P(IAU-76)",
        },
        "j2_baseline_for_fidelity_comparison": {
            "mu_m3_s2": model.gm,
            "equatorial_radius_m": model.radius,
            "j2": j2_from_c20,
            "j2_definition": "-sqrt(5) * C20_bar from the fixture coefficients",
        },
    }


def build_trajectory_fixture(
    spec: ArcSpec, model: Egm96Model, j2_from_c20: float
) -> tuple[dict, dict]:
    clock = make_clock(spec.epoch_iso, spec.duration_s)
    print(f"[{spec.name}] integrating baseline (rtol=atol=1e-12) ...")
    base = integrate_arc(spec, clock, model, rtol=1e-12, dense=True)
    print(f"[{spec.name}] sensitivity runs ...")
    loose10 = integrate_arc(spec, clock, model, rtol=1e-10)
    loose9 = integrate_arc(spec, clock, model, rtol=1e-9)
    d_int, dv_int = max_state_delta(base, loose10)
    d_loose9, dv_loose9 = max_state_delta(base, loose9)

    uses_bodies = spec.third_body_sun or spec.third_body_moon
    uses_bodies = uses_bodies or spec.srp_cr_a_over_m is not None
    uses_bodies = uses_bodies or spec.drag_inv_beta is not None
    if uses_bodies:
        eph = integrate_arc(spec, clock, model, rtol=1e-12, ephem="analytic")
        d_eph, dv_eph = max_state_delta(base, eph)
    else:
        d_eph = dv_eph = 0.0
    if spec.drag_inv_beta is not None:
        conv = integrate_arc(spec, clock, model, rtol=1e-12, apex_frame="gcrf")
        d_conv, dv_conv = max_state_delta(base, conv)
    else:
        d_conv = dv_conv = 0.0

    tol_pos = round_up_2sig(3.0 * (d_int + d_eph + d_conv) + 1.0)
    tol_vel = round_up_2sig(3.0 * (dv_int + dv_eph + dv_conv) + 1e-3)

    shadow = None
    if spec.srp_cr_a_over_m is not None:
        shadow = count_shadow_crossings(spec, clock, base)
        if spec.name == "srp-shadow-crossing" and shadow["count"] < 2:
            raise AssertionError(f"SRP arc has {shadow['count']} shadow crossings, need >= 2")

    j2_info = None
    if spec.name == "leo-full-force":

        def j2_rhs(_ts: float, y: np.ndarray) -> np.ndarray:
            a = j2_closed_form_accel(y[:3], model.gm, model.radius, j2_from_c20)
            return np.concatenate((y[3:], a))

        grid = np.arange(0.0, spec.duration_s + 0.5 * spec.grid_step_s, spec.grid_step_s)
        sol_j2 = solve_ivp(
            j2_rhs,
            (0.0, spec.duration_s),
            np.concatenate((spec.r0, spec.v0)),
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            max_step=MAX_STEP_S,
            t_eval=grid,
        )
        if not sol_j2.success:
            raise AssertionError("J2-only comparison run failed")
        d_j2, _ = max_state_delta(base, sol_j2)
        d_j2_final = float(np.linalg.norm(base.y[:3, -1] - sol_j2.y[:3, -1]))
        if d_j2_final < 20.0 * tol_pos:
            raise AssertionError(
                f"fidelity headroom too small: J2-only final delta {d_j2_final:.1f} m "
                f"vs tolerance {tol_pos:.1f} m"
            )
        j2_info = {
            "max_position_delta_m": round(d_j2, 3),
            "final_position_delta_m": round(d_j2_final, 3),
            "note": (
                "informational: two-body+J2-only propagation (parameters in "
                "force_config.j2_baseline_for_fidelity_comparison) vs this "
                "golden, integrated the same way. The spec scenario 'Higher "
                "fidelity than the J2 baseline' expects the full stack to "
                "land inside tolerance.position_m while J2-only misses by "
                "this margin."
            ),
        }

    samples = [
        {
            "t_s": float(base.t[k]),
            "r_m": vec(base.y[:3, k]),
            "v_mps": vec(base.y[3:, k]),
        }
        for k in range(base.t.size)
    ]
    sens = {
        "integrator_rtol_1e-10_max_delta": {"position_m": d_int, "velocity_mps": dv_int},
        "integrator_rtol_1e-9_max_delta": {
            "position_m": d_loose9,
            "velocity_mps": dv_loose9,
            "note": "informational only — not part of the tolerance sum",
        },
        "ephemeris_swap": {
            "position_m": d_eph,
            "velocity_mps": dv_eph,
            "note": (
                "baseline re-integrated with the analytic Sun/Moon series the "
                "Rust stack implements (Meeus Ch. 25 Sun mirrored from "
                "ephemeris.rs on TT centuries, Vallado Alg 31 Moon) in place "
                "of ERFA epv00/moon98 — bounds the ephemeris-choice effect "
                "end to end"
            ),
        },
        "bulge_apex_convention": {
            "position_m": d_conv,
            "velocity_mps": dv_conv,
            "note": (
                "baseline re-integrated with the Harris-Priester bulge apex "
                "rotation applied about the GCRF +z axis instead of ITRF +z "
                "(zero when the arc has no drag)"
            ),
        },
    }
    fixture = {
        "case": spec.name,
        "description": spec.description,
        "epoch_utc": spec.epoch_iso + "Z",
        "frame": "GCRF",
        "units": {"position": "m", "velocity": "m/s", "time": "seconds past epoch_utc"},
        "initial_state": {"r_m": vec(spec.r0), "v_mps": vec(spec.v0)},
        "force_config": force_config_block(spec, model, j2_from_c20),
        "generator_integrator": {
            "method": "scipy.integrate.solve_ivp DOP853",
            "rtol": 1e-12,
            "atol": 1e-12,
            "max_step_s": MAX_STEP_S,
            "sampling": "t_eval on the fixed grid below",
        },
        "tolerance": {
            "position_m": tol_pos,
            "velocity_mps": tol_vel,
            "method": (
                "3 x (integrator sensitivity [rtol 1e-12 vs 1e-10 re-run] + "
                "ephemeris-swap sensitivity + bulge-apex convention "
                "sensitivity) + 1 m (or 1 mm/s) floor, rounded up to two "
                "significant figures; each term is the max delta over the "
                "sample grid, measured at generation time and recorded below"
            ),
            "sensitivities": sens,
        },
        "grid_step_s": spec.grid_step_s,
        "samples": samples,
    }
    if shadow is not None:
        fixture["shadow_crossings"] = shadow
    if j2_info is not None:
        fixture["j2_only_comparison"] = j2_info
    if spec.extra_notes:
        fixture["notes"] = spec.extra_notes
    measured = {
        "name": spec.name,
        "tol_pos": tol_pos,
        "tol_vel": tol_vel,
        "d_int": d_int,
        "d_eph": d_eph,
        "d_conv": d_conv,
        "d_loose9": d_loose9,
        "shadow_count": None if shadow is None else shadow["count"],
        "j2_final": None if j2_info is None else j2_info["final_position_delta_m"],
    }
    print(
        f"[{spec.name}] tol {tol_pos} m / {tol_vel} m/s "
        f"(int {d_int:.2e}, eph {d_eph:.2e}, conv {d_conv:.2e}); "
        f"shadow crossings: {measured['shadow_count']}"
    )
    return fixture, measured


# ---------------------------------------------------------------------------
# PROVENANCE.md
# ---------------------------------------------------------------------------


def package_versions() -> dict:
    packages = ("numpy", "scipy", "astropy", "pyerfa", "mpmath", "astropy-iers-data")
    return {
        "python": platform.python_version(),
        **{p: importlib.metadata.version(p) for p in packages},
    }


def provenance_markdown(
    versions: dict,
    eph_tols: dict,
    arc_measured: list[dict],
    c20_j2: float,
    egm96_origin: str,
) -> str:
    lines = [
        "# Orbit-Propagation Golden Fixture Provenance",
        "",
        "Tier-3 golden fixtures for design Decision 7 of the",
        "`orbit-propagation-fidelity` change: component goldens (analytic",
        "Sun/Moon vs an independent authority, EGM96 spot accelerations from an",
        "independent evaluation, the published Harris-Priester table) and three",
        "trajectory goldens integrated by an independently written Python force",
        "stack. Consumed by default-feature envelope tests (no Python, no",
        "network). Units are metres, m/s, seconds; epochs are ISO-8601 UTC",
        "strings.",
        "",
        "## Generation record",
        "",
        "- Generator: `python/scripts/gen_propagation_fixtures.py` (manual-only, never CI)",
        "- Regeneration command (from the repository root, then commit the output):",
        "",
        "```sh",
        "uv run python/scripts/gen_propagation_fixtures.py",
        "```",
        "",
        "- The run fetches the EGM96 coefficient file once and verifies its",
        "  SHA-256 against the pinned digest before trusting the embedded",
        "  subset (set `THRESH_EGM96_GFC` to a local copy to skip the network;",
        "  the digest check still runs). Repeated runs at the same package",
        "  versions are bit-identical.",
        "- Toolchain at generation time:",
        "",
    ]
    lines += [f"  - {name} {ver}" for name, ver in versions.items()]
    lines += [
        "",
        "## Frames and time conventions (both stacks)",
        "",
        "Zero-EOP IAU-76/FK5 exactly matching the Rust default",
        "`Iau76Fk5Provider`: dUT1 = 0 (UT1 = UTC), no polar motion, and",
        "GCRF -> ITRF = R3(GAST) . N(IAU-1980) . P(IAU-76) with",
        "GAST = GMST-1982 + EqEq-1994 (generator side: erfa",
        "pmat76/nutm80/gmst82/eqeq94 — the same reduction the frames golden",
        "fixtures validated against astropy). dAT is constant across every arc",
        "(no leap second since 2017; asserted at generation time). The ITRF",
        "velocity transport term uses omega_earth = 7.2921150e-5 rad/s",
        "(`thresh_core::eci::EARTH_ROTATION_RATE`).",
        "",
        "## Sources (all fetched this generation session)",
        "",
    ]
    lines += [f"- **{key}**: {text}" for key, text in SOURCES.items()]
    lines += [
        "",
        f"EGM96 coefficients were consumed from: {egm96_origin}.",
        "",
        "## `sun-moon-ephemeris.json`",
        "",
        "Authority: astropy `get_body(..., ephemeris='builtin')` (ERFA epv00 /",
        "moon98) — independent of the analytic series the Rust ephemerides",
        "implement. The generator mirrors those series and measures them",
        "against the authority per epoch:",
        "",
        "- **Sun**: Meeus Ch. 25 low-accuracy series, mirrored",
        "  coefficient-for-coefficient from",
        "  `crates/thresh-core/src/orbital/ephemeris.rs` (which cites Meeus AA",
        "  2nd ed. Eqs. 25.2-25.5 via the book-faithful soniakeys/meeus",
        "  transcription), evaluated on **TT Julian centuries** — the Rust",
        "  time argument — with the IAU-1980 (obl80) mean obliquity and the",
        "  IAU-76 precession transpose: the exact ephemeris.rs chain, so the",
        "  measured Sun error IS the Rust stack's error at these epochs.",
        "  Meeus Example 25.a (the Rust unit test's spot values) is",
        "  re-asserted at generation time.",
        "- **Moon**: Vallado Algorithm 31 per the fetched moon.m,",
        "  coefficient-identical to the Rust transcription, evaluated on the",
        "  UTC Julian date with moon.m's truncated obliquity as fetched (the",
        "  Rust evaluates TT and obl80; TT-UTC = 69.184 s is ~38 arcsec of",
        "  lunar motion — well inside one series error and the 3x margin).",
        "- The sibling **Vallado Algorithm 29 Sun** (which differs from the",
        "  Meeus Ch. 25 Sun by ~20 arcsec at these epochs) is kept",
        "  reference-only; its deltas are recorded per epoch as",
        "  `informational_vallado_alg29_sun_error` and enter no tolerance.",
        "",
        "Per-body tolerances = 3x the worst measured error, rounded up to two",
        "significant figures:",
        "",
        f"- Sun: {eph_tols['sun_angular_arcsec']} arcsec "
        f"(distance rel {eph_tols['sun_distance_rel']})",
        f"- Moon: {eph_tols['moon_angular_arcsec']} arcsec "
        f"(distance rel {eph_tols['moon_distance_rel']})",
        "",
        "The 3x margin buys headroom against future-epoch drift, the",
        "authority's light-time-corrected (vs geometric) positions, and the",
        "Moon's residual evaluation-input differences above. A transcription",
        "typo shows up at degrees scale and fails the envelope by orders of",
        "magnitude.",
        "",
        "## `egm96-spot-accelerations.json`",
        "",
        "This generator evaluates the potential of Barthelmes STR09/02",
        "Eq. (108) with its own fully-normalized Legendre recursion (sources",
        "above) and takes the analytic spherical gradient. Generation-time",
        "self-checks, all hard-fail:",
        "",
        "1. fetched-file SHA-256 + bit-equality of the embedded 88-row subset",
        "   (plus the NGA original-distribution cross-check recorded above);",
        "2. recursion magnitudes vs `mpmath.legenp` for all n,m <= 12 at three",
        "   latitudes (< 1e-30);",
        "3. analytic gradient vs 60-digit mpmath central differences of the",
        "   same potential at every spot (< 5e-12 relative, measured value",
        "   recorded per spot);",
        "4. Laplacian-harmonicity residual at every spot (< 1e-9 relative —",
        "   a recursion-coefficient error breaks harmonicity);",
        f"5. C20-only evaluation == closed-form J2 (J2 = -sqrt(5)*C20 = {c20_j2!r})",
        "   to <= 1e-12 relative — the executable normalization contract",
        "   (design Decision 3).",
        "",
        "Envelope tolerance: 1e-9 relative per component (basis recorded in the",
        "fixture).",
        "",
        "## `harris-priester-table.json`",
        "",
        "The 50-node min/max table is fetched-published data (two independent",
        "transcriptions, Orekit and SatelliteToolbox.jl, agree on every row;",
        "both cite Montenbruck & Gill, 'Satellite Orbits'). The interpolation +",
        "bulge convention (exponential node interpolation, cos^2(psi/2) bulge,",
        "apex 30 deg east of the subsolar point about the ITRF z-axis, geodetic",
        "WGS-84 height, n = 2 per design Decision 4) is recorded in the fixture",
        "so the Rust implementation mirrors it exactly. Worked examples are",
        "tool-computed from the table with that convention.",
        "",
        "## Trajectory goldens",
        "",
        "Integrated with scipy `solve_ivp` DOP853 at rtol = atol = 1e-12,",
        "max_step 300 s, sampled by `t_eval` on each arc's fixed grid. The",
        "force stack is written independently of the Rust one: ERFA",
        "epv00/moon98 ephemerides, this generator's own EGM96 evaluation,",
        "Harris-Priester per the fetched table/convention, cannonball SRP with",
        "the cylindrical shadow, direct-minus-indirect third body.",
        "",
        "SRP convention (resolves the design.md open question): P(d) =",
        "(TSI/c) * (au/d)^2 with the IAU 2015 B3 nominal TSI 1361 W/m^2,",
        "c = 299792458 m/s, au = 149597870700 m, d = Sun->spacecraft distance;",
        f"P at 1 au = {P_SRP_1AU!r} N/m^2 (= 1361/299792458, exactly mirrorable",
        "in IEEE-754 by both stacks).",
        "",
        "Per-arc tolerance = 3 x (sum of measured sensitivities) + 1 m floor",
        "(velocities: + 1 mm/s), rounded up to two significant figures.",
        "Sensitivities, each the max state delta over the sample grid:",
        "",
        "- **integrator**: baseline re-run at rtol/atol 1e-10 (the committed",
        "  samples' own numerical envelope);",
        "- **ephemeris swap**: baseline re-run with the analytic Sun/Moon",
        "  series the Rust stack implements (Meeus Ch. 25 Sun mirrored from",
        "  ephemeris.rs on TT centuries, Vallado Alg 31 Moon) substituted for",
        "  ERFA — bounds the Rust stack's ephemeris difference end to end",
        "  (dominant term wherever third-body or bulge terms matter);",
        "- **bulge-apex convention**: baseline re-run applying the 30 deg apex",
        "  rotation about GCRF z instead of ITRF z (drag arcs only).",
        "",
        "Measured at generation time:",
        "",
        "| arc | integrator (m) | ephemeris swap (m) | apex conv (m) | rtol 1e-9 (m, info) "
        "| tolerance (m) | tolerance (m/s) |",
        "|-----|----------------|--------------------|---------------|---------------------"
        "|---------------|-----------------|",
    ]
    for m in arc_measured:
        lines.append(
            f"| {m['name']} | {m['d_int']:.3e} | {m['d_eph']:.3e} | {m['d_conv']:.3e} "
            f"| {m['d_loose9']:.3e} | {m['tol_pos']} | {m['tol_vel']} |"
        )
    srp_m = next(m for m in arc_measured if m["name"] == "srp-shadow-crossing")
    leo_m = next(m for m in arc_measured if m["name"] == "leo-full-force")
    lines += [
        "",
        f"- Shadow crossings on `srp-shadow-crossing`: **{srp_m['shadow_count']}**",
        "  0<->1 transitions of the cylindrical shadow factor (>= 2 asserted at",
        "  generation time; times recorded in the fixture). The LEO arc's own",
        "  crossings are recorded in its fixture for reference.",
        "- Fidelity headroom on `leo-full-force`: J2-only propagation misses the",
        f"  golden by {leo_m['j2_final']} m at arc end (>= 20x the position",
        "  tolerance, asserted) — the numeric basis for the 'Higher fidelity",
        "  than the J2 baseline' scenario.",
        "",
        "## Value-source summary",
        "",
        "- **Fetched-published**: EGM96 coefficients/GM/radius (ICGEM + NGA),",
        "  the Harris-Priester table (Orekit + SatelliteToolbox.jl, Montenbruck",
        "  & Gill lineage), the Meeus Ch. 25 Sun coefficients (mirrored from",
        "  crates/thresh-core/src/orbital/ephemeris.rs, whose transcription",
        "  cites the book-faithful soniakeys/meeus solar.go), the Vallado Moon",
        "  series constants and the reference-only Alg 29 Sun (CelesTrak",
        "  companion code), GM_Sun/GM_Moon/au/c (JPL SSD), TSI (IAU 2015 B3),",
        "  the Legendre recursion and potential formulas (MITgcm geoid",
        "  cookbook / Holmes & Featherstone; Barthelmes STR09/02).",
        "- **Tool-computed** (packages above): astropy Sun/Moon authority",
        "  positions, EGM96 spot accelerations, Harris-Priester worked",
        "  examples, all trajectory samples, every measured sensitivity and",
        "  self-check residual.",
        "- **Defined inputs (not references)**: arc epochs, initial states,",
        "  drag/SRP area-to-mass parameters, grid steps, the shadow-cylinder",
        "  radius choice (WGS-84 equatorial), and omega_earth (chosen to match",
        "  the Rust constant).",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def render_json(obj: dict) -> str:
    # allow_nan=False: a non-finite value anywhere in a fixture is a generator
    # bug — fail here rather than emit JSON Rust cannot parse.
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    json.loads(text)  # parse-back check on the exact rendered bytes
    return text


def main() -> int:
    versions = package_versions()
    print(f"generator toolchain: {versions}")

    embedded = parse_embedded_egm96()
    egm96_origin = fetch_and_verify_egm96(embedded)
    model = Egm96Model(coeffs=embedded)

    verify_legendre_vs_mpmath_legenp()
    j2_from_c20 = verify_c20_identity(model)
    verify_meeus_sun_transcription()
    print("Meeus Ch. 25 Sun mirror verified against Example 25.a spot values")
    verify_fast_ephemeris(EPHEMERIS_EPOCHS)
    print("fast ERFA ephemeris verified against astropy at all fixture epochs")

    # Build and render every output in memory first; write only after all
    # succeed, so a failure partway never leaves a mixed fixture set on disk.
    outputs: dict[str, str] = {}

    eph_fixture, eph_tols = build_ephemeris_fixture()
    outputs["sun-moon-ephemeris.json"] = render_json(eph_fixture)
    print(f"sun-moon-ephemeris.json rendered; tolerances {eph_tols}")

    outputs["egm96-spot-accelerations.json"] = render_json(
        build_egm96_fixture(model, egm96_origin)
    )
    print("egm96-spot-accelerations.json rendered")

    outputs["harris-priester-table.json"] = render_json(build_hp_fixture())
    print("harris-priester-table.json rendered")

    arc_measured = []
    for spec in define_arcs(model):
        fixture, measured = build_trajectory_fixture(spec, model, j2_from_c20)
        outputs[f"{spec.name}.json"] = render_json(fixture)
        arc_measured.append(measured)

    outputs["PROVENANCE.md"] = provenance_markdown(
        versions, eph_tols, arc_measured, j2_from_c20, egm96_origin
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    print(f"all {len(outputs)} outputs written atomically to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
