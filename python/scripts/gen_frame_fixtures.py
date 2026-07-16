#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy==2.2.6",
#     "astropy==7.0.1",
#     "pyerfa==2.0.1.5",
#     "skyfield==1.49",
# ]
# ///
"""Golden-vector generator for `test-data/golden/frames/` (astro-time-and-frames, task 3.1).

MANUAL-ONLY. Never runs in CI (design Decision 7: fixtures are tier-3 golden
vectors, committed to the tree; CI consumes them with no Python and no network).

Regenerate from the repository root with:

    uv run python/scripts/gen_frame_fixtures.py

Outputs:
  test-data/golden/frames/vallado-example-3-15.json  - Vallado's worked IAU-76/FK5
      reduction (ITRF -> PEF -> TOD -> MOD -> GCRF, plus TEME) at
      2004-04-06 07:51:28.386009 UTC with the published EOP inputs.
  test-data/golden/frames/zero-eop-*.json            - astropy-computed
      TEME->GCRF, GCRF->ITRF, TEME->ITRF vectors at three epochs with
      dUT1 = 0 (UT1 = UTC) and zero polar motion, matching the Rust
      Iau76Fk5Provider zero defaults.
  test-data/golden/frames/PROVENANCE.md              - sources, versions,
      regeneration command, measured cross-check agreement.

Published reference values are transcribed as constants below; every constant
cites its source. The script re-derives the whole chain with an independent
pyerfa IAU-76/FK5 implementation and hard-fails if the transcription and the
theory disagree, so a typo in a constant cannot reach the committed fixtures.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, TEME, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers

# Fixtures must be reproducible offline: never consult (or download) IERS data.
# All EOP values are either the fixture's published inputs or exact zeros.
iers.conf.auto_download = False

import astropy.coordinates.builtin_frames.intermediate_rotation_transforms as irt  # noqa: E402

ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
# Earth rotation rate, rad/s (Vallado, Fundamentals of Astrodynamics and
# Applications, 4th ed., Eq. 3-40 vicinity; standard IAU-76/FK5 value).
OMEGA_EARTH = 7.292115146706979e-5

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "test-data" / "golden" / "frames"

# ---------------------------------------------------------------------------
# Vallado Example 3-15 published values (all verified against fetched sources
# at generation time; see PROVENANCE.md for URLs).
#
# Inputs: CelesTrak/fundamentals-of-astrodynamics, software/matlab/ex3_15.m
# (D. Vallado's own companion code), and AAS 06-675 "Implementation Issues
# Surrounding the New IAU Reference Frameworks" (Vallado, Seago, Seidelmann),
# LEO test case, p. 17.
# ---------------------------------------------------------------------------
VALLADO_EPOCH_UTC = "2004-04-06T07:51:28.386009"
VALLADO_DUT1_S = -0.4399619  # ex3_15.m; AAS 06-675 rounds to -0.439962
VALLADO_DAT_S = 32
VALLADO_XP_ARCSEC = -0.140682
VALLADO_YP_ARCSEC = 0.333309
VALLADO_LOD_S = 0.0015563
VALLADO_DDPSI_ARCSEC = -0.052195  # recorded only; IAU-76 provider does not apply
VALLADO_DDEPS_ARCSEC = -0.003875  # recorded only; IAU-76 provider does not apply
# AAS 06-675 time table, p. 17: jdtt 2453101.82815474550 (sanity check target)
VALLADO_JD_TT = 2453101.82815474550

# Published state rows, km and km/s. Source: AAS 06-675, LEO test case output
# table, p. 17 (rows named exactly as in the paper). The "*_iau76" rows are the
# classical IAU-76/FK5 reduction WITHOUT the ddpsi/ddeps EOP corrections, which
# is precisely what thresh's Iau76Fk5Provider implements.
VALLADO_ITRF_R = [-1033.4793830, 7901.2952754, 6380.3565958]
VALLADO_ITRF_V = [-3.225636520, -2.872451450, 5.531924446]
VALLADO_PEF_R = [-1033.4750313, 7901.3055856, 6380.3445328]  # row "PEF iau76"
VALLADO_PEF_V = [-3.225632747, -2.872442511, 5.531931288]
VALLADO_TOD_R = [5094.5147804, 6127.3664612, 6380.3445328]  # row "TOD iau76"
VALLADO_TOD_V = [-4.746088567, 0.786077222, 5.531931288]
VALLADO_MOD_R = [5094.0290167, 6127.8709363, 6380.2478885]  # row "MOD iau76"
VALLADO_MOD_V = [-4.746262495, 0.786014149, 5.531791025]
VALLADO_J2000_R = [5102.5096000, 6123.0115200, 6378.1363000]  # row "J2000 iau76"
VALLADO_J2000_V = [-4.743219600, 0.790536600, 5.533756190]
# Row "GCRF iau76 w corr" (ddpsi/ddeps applied) - informational only.
VALLADO_GCRF_CORR_R = [5102.5089579, 6123.0114007, 6378.1369282]
VALLADO_GCRF_CORR_V = [-4.743220157, 0.790536497, 5.533755727]
# Row "GCRF CIO dx,dy=0" (IAU-2000 CIO chain, no dX/dY corrections) - used as
# the self-check target for astropy's IAU-2006/2000A machinery.
VALLADO_GCRF_CIO0_R = [5102.5089592, 6123.0114033, 6378.1369247]
# TEME at the same epoch. Source: AIAA 2006-6753 Rev 3 "Revisiting Spacetrack
# Report #3" (Vallado, Crawford, Hujsak, Kelso), Appendix C, p. 32
# ("Converting through PEF to TEME, we find"). That appendix quotes
# dUT1 = -0.439961 s (1e-7 s coarser than ex3_15.m; sub-mm effect).
VALLADO_TEME_R = [5094.18010720, 6127.64470520, 6380.34453270]
VALLADO_TEME_V = [-4.746131494, 0.785817998, 5.531931288]

# ---------------------------------------------------------------------------
# Zero-EOP cases: LEO-magnitude TEME input states (defined here, not published
# values) at three epochs. dUT1 = 0 (UT1 = UTC), xp = yp = 0 to match the Rust
# Iau76Fk5Provider defaults. Targets are astropy-computed.
# ---------------------------------------------------------------------------
ZERO_EOP_CASES = [
    {
        "name": "zero-eop-2020-01-01",
        "epoch_utc": "2020-01-01T12:00:00.000000",
        "comment": "Plain modern epoch.",
        "r_teme_km": [-4743.211, 4241.734, 2500.410],
        "v_teme_kmps": [-5.530, -4.720, 1.620],
    },
    {
        "name": "zero-eop-2015-06-30",
        "epoch_utc": "2015-06-30T23:59:00.000000",
        "comment": "One minute before the 2015-06-30T23:59:60 UTC leap second (dAT = 35 s).",
        "r_teme_km": [6045.000, -3490.000, 2500.000],
        "v_teme_kmps": [2.500, 6.700, -1.800],
    },
    {
        "name": "zero-eop-2024-03-20",
        "epoch_utc": "2024-03-20T03:06:00.000000",
        "comment": "Near the March 2024 equinox (2024-03-20 ~03:06 UTC).",
        "r_teme_km": [-2145.000, -6653.000, 1500.000],
        "v_teme_kmps": [7.000, -2.400, 0.900],
    },
]

SOURCES = {
    "aas-06-675": (
        "Vallado, Seago, Seidelmann, 'Implementation Issues Surrounding the New "
        "IAU Reference Frameworks' (AAS 06-675), LEO test case, p. 17. "
        "https://www.agi.com/getmedia/2a33d303-9a4e-4520-9da0-86a4d5e45a10/"
        "Implementation-Issues-Surrounding-the-New-IAU-Reference-Systems-for-"
        "Astrodynamics.pdf"
    ),
    "aiaa-2006-6753": (
        "Vallado, Crawford, Hujsak, Kelso, 'Revisiting Spacetrack Report #3' "
        "(AIAA 2006-6753, Rev 3), Appendix C - TEME Coordinate System, p. 32. "
        "https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf"
    ),
    "ex3_15.m": (
        "CelesTrak/fundamentals-of-astrodynamics companion code, "
        "software/matlab/ex3_15.m (input epoch and EOP for Example 3-15). "
        "https://github.com/CelesTrak/fundamentals-of-astrodynamics/blob/main/"
        "software/matlab/ex3_15.m"
    ),
}


# ---------------------------------------------------------------------------
# Small math/formatting helpers
# ---------------------------------------------------------------------------
def rot3(theta_rad: float) -> np.ndarray:
    """Vallado ROT3 (frame rotation about +z by theta)."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def km_to_m(vec_km) -> list[float]:
    return [round(float(x) * 1000.0, 4) for x in vec_km]


def kmps_to_mps(vec_kmps) -> list[float]:
    return [round(float(x) * 1000.0, 7) for x in vec_kmps]


def delta_m(a_km, b_km) -> float:
    return float(np.linalg.norm((np.asarray(a_km, float) - np.asarray(b_km, float)) * 1000.0))


def delta_mps(a_kmps, b_kmps) -> float:
    return float(np.linalg.norm((np.asarray(a_kmps, float) - np.asarray(b_kmps, float)) * 1000.0))


def position_tolerance_m(measured_m: float) -> float:
    """Envelope tolerance: 1.5x the measured theory delta plus 1 m headroom.

    The +1 m absorbs small implementation choices on the Rust side (e.g. the
    two ~0.003" complementary equation-of-the-equinoxes terms subtend ~0.1 m at
    LEO radius; published-row rounding is 0.1 mm) while staying orders of
    magnitude below the ~kilometre TEME/GCRF conflation this change fixes.
    """
    return round(1.5 * measured_m + 1.0, 3)


def velocity_tolerance_mps(measured_mps: float) -> float:
    """1.5x measured plus 5 mm/s headroom (astropy rotating-frame velocities
    come from finite differencing, good to ~mm/s at LEO rates)."""
    return round(1.5 * measured_mps + 0.005, 4)


def utc_time(iso_utc: str, dut1_s: float) -> Time:
    t = Time(iso_utc, scale="utc", format="isot")
    t.delta_ut1_utc = dut1_s  # explicit value: astropy will not consult IERS
    return t


class ConstantDut1Table(iers.IERS):
    """IERS-table stand-in returning a constant UT1-UTC everywhere.

    astropy's rotating-frame velocity machinery finite-differences transforms
    over shifted copies of ``obstime``; those copies do NOT inherit a per-object
    ``Time.delta_ut1_utc`` override and would silently fall back to real IERS
    UT1-UTC (measured effect: an omega^2 * dUT1 * r velocity contamination,
    ~25 mm/s at LEO for the 2015 bundled dUT1 of -0.68 s). Overriding the
    ``earth_orientation_table`` science state forces every ``Time`` in the
    process - shifted copies included - onto the fixture's dUT1.

    ``pm_xy`` raises: polar motion must flow through the patched
    ``get_polar_motion`` (see ``forced_eop``); reaching the IERS table for it
    would mean the patch no longer covers some transform route.
    """

    dut1_s = 0.0

    def ut1_utc(self, jd1, jd2=0.0, return_status=False):
        del jd2
        if np.shape(jd1):
            value = u.Quantity(np.full(np.shape(jd1), self.dut1_s), u.s)
        else:
            value = self.dut1_s * u.s
        if return_status:
            return value, np.full(np.shape(jd1), iers.FROM_IERS_B)
        return value

    def pm_xy(self, *args, **kwargs):
        raise RuntimeError("unexpected IERS pm_xy call: forced_eop polar-motion patch coverage gap")


@contextmanager
def forced_eop(dut1_s: float, xp_rad: float, yp_rad: float):
    """Force dUT1 (IERS-table layer) and polar motion (transform layer) at once."""
    table = ConstantDut1Table()
    table.dut1_s = dut1_s
    with iers.earth_orientation_table.set(table), forced_polar_motion(xp_rad, yp_rad):
        yield


# ---------------------------------------------------------------------------
# Independent IAU-76/FK5 chain (pyerfa) - the same reduction the Rust provider
# implements. Used to (a) verify the transcribed published rows and (b) measure
# the IAU-76/FK5 vs IAU-2006/2000A theory delta that sets each tolerance.
# ---------------------------------------------------------------------------
def iau76_angles(t: Time) -> dict:
    """Precession/nutation/sidereal quantities for the IAU-76/FK5 reduction."""
    jd_tt1, jd_tt2 = t.tt.jd1, t.tt.jd2
    ttt = ((jd_tt1 - 2451545.0) + jd_tt2) / 36525.0
    dpsi, deps = erfa.nut80(jd_tt1, jd_tt2)
    eps_m = erfa.obl80(jd_tt1, jd_tt2)
    prec = erfa.pmat76(jd_tt1, jd_tt2)  # r_MOD = prec @ r_J2000
    nut = erfa.numat(eps_m, dpsi, deps)  # r_TOD = nut @ r_MOD
    gmst = erfa.gmst82(t.ut1.jd1, t.ut1.jd2)
    # Equation of the equinoxes, Vallado eqeterms=2 form (post-1997 kinematic
    # complementary terms; Vallado 4th ed. Eq. 3-79 vicinity).
    om_deg = 125.04452222 + (-6962890.5390 * ttt + 7.455 * ttt * ttt + 0.008 * ttt**3) / 3600.0
    om = math.radians(om_deg % 360.0)
    eqe_geometric = dpsi * math.cos(eps_m)
    kinematic = (0.00264 * math.sin(om) + 0.000063 * math.sin(2.0 * om)) * ARCSEC_TO_RAD
    eqe_full = eqe_geometric + kinematic
    return {
        "prec": prec,
        "nut": nut,
        "gmst": gmst,
        "gast": gmst + eqe_full,
        "eqe_geometric": eqe_geometric,
        "eqe_full": eqe_full,
    }


def pef_from_itrf(r_itrf, v_itrf, xp_rad: float, yp_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Undo polar motion: ERFA pom00 maps V(TRS) = rpom @ V(PEF), so PEF = rpom^T @ ITRF."""
    rpom = erfa.pom00(xp_rad, yp_rad, 0.0)  # sp = 0 in the IAU-76/FK5 reduction
    return rpom.T @ np.asarray(r_itrf, float), rpom.T @ np.asarray(v_itrf, float)


def inertial_legs_from_pef(r_pef, v_pef, ang: dict, lod_s: float) -> dict:
    """PEF -> TOD -> MOD -> J2000 and PEF -> TEME, with the omega x r velocity terms."""
    omega = np.array([0.0, 0.0, OMEGA_EARTH * (1.0 - lod_s / 86400.0)])
    r_tod = rot3(-ang["gast"]) @ r_pef
    v_tod = rot3(-ang["gast"]) @ (v_pef + np.cross(omega, r_pef))
    r_mod = ang["nut"].T @ r_tod
    v_mod = ang["nut"].T @ v_tod
    r_j2000 = ang["prec"].T @ r_mod
    v_j2000 = ang["prec"].T @ v_mod
    r_teme = rot3(-ang["gmst"]) @ r_pef
    v_teme = rot3(-ang["gmst"]) @ (v_pef + np.cross(omega, r_pef))
    return {
        "PEF": (r_pef, v_pef),
        "TOD": (r_tod, v_tod),
        "MOD": (r_mod, v_mod),
        "GCRF": (r_j2000, v_j2000),
        "TEME": (r_teme, v_teme),
    }


def iau76_chain_from_itrf(t: Time, r_itrf_km, v_itrf_kmps, xp_arcsec, yp_arcsec, lod_s) -> dict:
    ang = iau76_angles(t)
    xp_rad, yp_rad = xp_arcsec * ARCSEC_TO_RAD, yp_arcsec * ARCSEC_TO_RAD
    r_pef, v_pef = pef_from_itrf(r_itrf_km, v_itrf_kmps, xp_rad, yp_rad)
    return inertial_legs_from_pef(r_pef, v_pef, ang, lod_s)


def iau76_chain_from_teme(t: Time, r_teme_km, v_teme_kmps) -> dict:
    """Zero-EOP direction: TEME -> PEF(=ITRF) -> ... -> J2000. UT1 = UTC, no polar motion."""
    ang = iau76_angles(t)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    r_pef = rot3(ang["gmst"]) @ np.asarray(r_teme_km, float)
    v_pef = rot3(ang["gmst"]) @ np.asarray(v_teme_kmps, float) - np.cross(omega, r_pef)
    return inertial_legs_from_pef(r_pef, v_pef, ang, 0.0)


def iau76_itrf_from_gcrf(t: Time, r_gcrf_km, v_gcrf_kmps) -> tuple[np.ndarray, np.ndarray]:
    """Zero-EOP GCRF -> ITRF(=PEF) leg of the IAU-76/FK5 chain.

    Needed separately from `iau76_chain_from_teme`: the fixture's GCRF->ITRF
    comparison starts from the astropy-computed GCRF state, so the IAU-76 vs
    IAU-2006/2000A inertial-orientation difference does NOT cancel the way it
    does when both chains start from the same TEME input.
    """
    ang = iau76_angles(t)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    r_tod = ang["nut"] @ (ang["prec"] @ np.asarray(r_gcrf_km, float))
    v_tod = ang["nut"] @ (ang["prec"] @ np.asarray(v_gcrf_kmps, float))
    r_pef = rot3(ang["gast"]) @ r_tod
    v_pef = rot3(ang["gast"]) @ v_tod - np.cross(omega, r_pef)
    return r_pef, v_pef


# ---------------------------------------------------------------------------
# astropy states (IAU-2006/2000A machinery) with forced EOP
# ---------------------------------------------------------------------------
@contextmanager
def forced_polar_motion(xp_rad: float, yp_rad: float):
    """Patch astropy's polar-motion lookup so transforms use exactly (xp, yp).

    astropy's ITRS/TEME matrices call the module-local `get_polar_motion`
    (astropy 7.0.1, astropy/coordinates/builtin_frames/
    intermediate_rotation_transforms.py); patching that symbol covers every
    route we exercise. Version-pinned, manual-only script.
    """
    assert hasattr(irt, "get_polar_motion"), "astropy internals moved; re-pin or update patch"

    def _forced(time):
        del time  # signature fixed by astropy
        return xp_rad, yp_rad

    original = irt.get_polar_motion
    irt.get_polar_motion = _forced
    try:
        yield
    finally:
        irt.get_polar_motion = original


def cartesian_state(frame_obj) -> tuple[np.ndarray, np.ndarray]:
    cart = frame_obj.cartesian
    r_km = cart.xyz.to_value(u.km)
    v_kmps = cart.differentials["s"].d_xyz.to_value(u.km / u.s)
    return np.asarray(r_km, float), np.asarray(v_kmps, float)


def astropy_transform(
    t: Time, source_frame, r_km, v_kmps, targets: dict, dut1_s: float, xp_rad: float, yp_rad: float
) -> dict:
    """Transform one cartesian state into each target frame under forced EOP."""
    rep = CartesianRepresentation(
        np.asarray(r_km, float) * u.km,
        differentials=CartesianDifferential(np.asarray(v_kmps, float) * (u.km / u.s)),
    )
    out = {}
    with forced_eop(dut1_s, xp_rad, yp_rad):
        coo = source_frame(rep, obstime=t)
        for name, target in targets.items():
            out[name] = cartesian_state(coo.transform_to(target(obstime=t)))
    return out


# ---------------------------------------------------------------------------
# Skyfield cross-check (zero-EOP cases only)
# ---------------------------------------------------------------------------
def skyfield_from_teme(epoch_utc: str, r_teme_km, v_teme_kmps) -> dict:
    """TEME -> GCRS/ITRS via Skyfield with UT1 forced equal to UTC.

    Skyfield's builtin timescale accepts a fixed delta_t = TT - UT1; choosing
    delta_t = 32.184 + dAT makes UT1 == UTC. Its default Timescale carries no
    polar-motion table, so xp = yp = 0 already matches the zero-EOP config.
    """
    from skyfield.api import load
    from skyfield.framelib import itrs as sf_itrs
    from skyfield.positionlib import ICRF
    from skyfield.sgp4lib import TEME as SF_TEME
    from skyfield.units import Distance, Velocity

    y, mo, rest = epoch_utc.split("-", 2)
    d, hms = rest.split("T")
    hh, mm, ss = hms.split(":")
    dat = erfa.dat(int(y), int(mo), int(d), 0.0)  # dAT at the epoch's day
    ts = load.timescale(delta_t=32.184 + dat)
    t = ts.utc(int(y), int(mo), int(d), int(hh), int(mm), float(ss))
    pos = ICRF.from_time_and_frame_vectors(
        t,
        SF_TEME,
        Distance(km=np.asarray(r_teme_km, float)),
        Velocity(km_per_s=np.asarray(v_teme_kmps, float)),
    )
    r_gcrs = np.asarray(pos.position.km, float)
    v_gcrs = np.asarray(pos.velocity.km_per_s, float)
    d_itrs, v_itrs = pos.frame_xyz_and_velocity(sf_itrs)
    return {
        "GCRF": (r_gcrs, v_gcrs),
        "ITRF": (np.asarray(d_itrs.km, float), np.asarray(v_itrs.km_per_s, float)),
    }


# ---------------------------------------------------------------------------
# Self-checks (hard failures: a bad transcription or convention bug stops here)
# ---------------------------------------------------------------------------
def verify_vallado_transcription(chain: dict) -> dict:
    """The independent pyerfa IAU-76 chain must reproduce every published row.

    Published rows carry 0.1 mm resolution; agreement is expected at the cm
    level or better (residual = published rounding + minor eqeq conventions).
    A transcription typo or a transpose/sign bug shows up as metres-to-km here.
    """
    published = {
        "PEF": (VALLADO_PEF_R, VALLADO_PEF_V),
        "TOD": (VALLADO_TOD_R, VALLADO_TOD_V),
        "MOD": (VALLADO_MOD_R, VALLADO_MOD_V),
        "GCRF": (VALLADO_J2000_R, VALLADO_J2000_V),
        "TEME": (VALLADO_TEME_R, VALLADO_TEME_V),
    }
    residuals = {}
    for frame, (r_pub, v_pub) in published.items():
        r_chk, v_chk = chain[frame]
        dr, dv = delta_m(r_chk, r_pub), delta_mps(v_chk, v_pub)
        residuals[frame] = {"dr_m": round(dr, 6), "dv_mps": round(dv, 6)}
        if dr > 0.15 or dv > 0.001:
            raise AssertionError(
                f"pyerfa IAU-76 chain vs published {frame}: dr={dr:.6f} m dv={dv:.6f} m/s - "
                "transcription or convention error, refusing to write fixtures"
            )
    return residuals


def verify_astropy_eop_mechanics(t: Time, astro_gcrf_km) -> None:
    """Prove the dUT1/polar-motion overrides actually reached astropy.

    (a) astropy GCRS with the fixture EOP must land on the published
        'GCRF CIO dx,dy=0' row (same IAU-2000-family theory, same EOP) to <1 m.
    (b) Recomputing with dUT1=0 must move the result by roughly
        |dUT1| * omega * r ~ 0.44 s * 465 m/s ~ 200 m, proving delta_ut1_utc
        survives astropy's frame plumbing.
    """
    d_cio = delta_m(astro_gcrf_km, VALLADO_GCRF_CIO0_R)
    if d_cio > 1.0:
        raise AssertionError(f"astropy GCRS vs published 'GCRF CIO dx,dy=0': {d_cio:.3f} m > 1 m")
    t0 = utc_time(VALLADO_EPOCH_UTC, 0.0)
    out0 = astropy_transform(
        t0,
        ITRS,
        VALLADO_ITRF_R,
        VALLADO_ITRF_V,
        {"GCRF": GCRS},
        0.0,
        VALLADO_XP_ARCSEC * ARCSEC_TO_RAD,
        VALLADO_YP_ARCSEC * ARCSEC_TO_RAD,
    )
    shift = delta_m(out0["GCRF"][0], astro_gcrf_km)
    if not 100.0 < shift < 300.0:
        raise AssertionError(f"dUT1 override ineffective: 0 vs -0.44 s moved GCRS {shift:.1f} m")
    # jd(TT) sanity against the AAS 06-675 time table
    jd_tt = float(t.tt.jd1 + t.tt.jd2)
    if abs(jd_tt - VALLADO_JD_TT) > 1e-8:
        raise AssertionError(f"jd(TT) {jd_tt!r} != published {VALLADO_JD_TT!r}")


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------
def published_leg(frame, r_km, v_kmps, source, note, astro=None, extra=None) -> dict:
    leg = {
        "frame": frame,
        "published": {
            "r_m": km_to_m(r_km),
            "v_mps": kmps_to_mps(v_kmps),
            "source": source,
            "note": note,
        },
        "tolerance_m": 1.0,  # spec "Vallado reduction reproduced": <= 1 m per leg
        "tolerance_mps": 0.01,
    }
    if astro is not None:
        r_a, v_a = astro
        leg["astropy_iau2006_2000a"] = {
            "r_m": km_to_m(r_a),
            "v_mps": kmps_to_mps(v_a),
            "delta_vs_published_m": round(delta_m(r_a, r_km), 4),
            "delta_vs_published_mps": round(delta_mps(v_a, v_kmps), 6),
            "note": (
                "astropy GCRS/TEME with the fixture EOP forced (dUT1 at the IERS-table "
                "layer, patched polar-motion lookup); IAU-2006/2000A theory, so it "
                "differs from the IAU-76/FK5 published rows by frame bias + "
                "nutation-theory deltas."
            ),
        }
    if extra:
        leg.update(extra)
    return leg


def build_vallado_fixture() -> tuple[dict, dict]:
    t = utc_time(VALLADO_EPOCH_UTC, VALLADO_DUT1_S)
    chain = iau76_chain_from_itrf(
        t, VALLADO_ITRF_R, VALLADO_ITRF_V, VALLADO_XP_ARCSEC, VALLADO_YP_ARCSEC, VALLADO_LOD_S
    )
    residuals = verify_vallado_transcription(chain)

    astro = astropy_transform(
        t,
        ITRS,
        VALLADO_ITRF_R,
        VALLADO_ITRF_V,
        {"GCRF": GCRS, "TEME": TEME},
        VALLADO_DUT1_S,
        VALLADO_XP_ARCSEC * ARCSEC_TO_RAD,
        VALLADO_YP_ARCSEC * ARCSEC_TO_RAD,
    )
    verify_astropy_eop_mechanics(t, astro["GCRF"][0])

    legs = [
        published_leg(
            "PEF",
            VALLADO_PEF_R,
            VALLADO_PEF_V,
            SOURCES["aas-06-675"] + " (row 'PEF iau76')",
            "Polar motion removed (W^T with the fixture xp/yp).",
        ),
        published_leg(
            "TOD",
            VALLADO_TOD_R,
            VALLADO_TOD_V,
            SOURCES["aas-06-675"] + " (row 'TOD iau76')",
            "PEF spun to true-of-date by GAST-1982 (GMST82 + full equation of the equinoxes).",
        ),
        published_leg(
            "MOD",
            VALLADO_MOD_R,
            VALLADO_MOD_V,
            SOURCES["aas-06-675"] + " (row 'MOD iau76')",
            "IAU-1980 nutation removed (full 106-term series, no ddpsi/ddeps corrections).",
        ),
        published_leg(
            "GCRF",
            VALLADO_J2000_R,
            VALLADO_J2000_V,
            SOURCES["aas-06-675"] + " (row 'J2000 iau76')",
            "IAU-76 precession removed. This is the classical IAU-76/FK5 'J2000' result "
            "WITHOUT ddpsi/ddeps EOP corrections - the exact chain Iau76Fk5Provider "
            "implements. It differs from true GCRF by the ~0.023 arcsec frame bias "
            "(~0.7 m at this radius); see gcrf_with_eop_corrections for the corrected row.",
            astro=astro["GCRF"],
        ),
        published_leg(
            "TEME",
            VALLADO_TEME_R,
            VALLADO_TEME_V,
            SOURCES["aiaa-2006-6753"],
            "TEME = R3(-GMST82) applied to PEF (equivalently R3(+EqEquinox) applied to TOD).",
            astro=astro["TEME"],
        ),
    ]
    fixture = {
        "case": "vallado-example-3-15",
        "description": (
            "Vallado's worked IAU-76/FK5 reduction (Fundamentals of Astrodynamics and "
            "Applications, Example 3-15; identical LEO test case published in AAS 06-675 "
            "and AIAA 2006-6753 Rev 3). Input is the ITRF state; every leg's published "
            "position must be reproduced to <= 1 m with the fixture's EOP inputs."
        ),
        "epoch_utc": VALLADO_EPOCH_UTC + "Z",
        "eop": {
            "dut1_s": VALLADO_DUT1_S,
            "dat_s": VALLADO_DAT_S,
            "xp_arcsec": VALLADO_XP_ARCSEC,
            "yp_arcsec": VALLADO_YP_ARCSEC,
            "lod_s": VALLADO_LOD_S,
            "ddpsi_arcsec": VALLADO_DDPSI_ARCSEC,
            "ddeps_arcsec": VALLADO_DDEPS_ARCSEC,
            "note": (
                "dut1/xp/yp from ex3_15.m (AAS 06-675 rounds dut1 to -0.439962 s; AIAA "
                "2006-6753 Rev 3 to -0.439961 s - sub-mm effect). lod affects only the "
                "omega magnitude in velocity terms (~mm/s here); ddpsi/ddeps are the "
                "EOP nutation corrections, recorded for completeness but NOT applied by "
                "the IAU-76/FK5 provider under test (compare against the *_iau76 rows)."
            ),
        },
        "input": {
            "frame": "ITRF",
            "r_m": km_to_m(VALLADO_ITRF_R),
            "v_mps": kmps_to_mps(VALLADO_ITRF_V),
            "source": (
                SOURCES["aas-06-675"] + " (row 'ITRF'); inputs also in " + SOURCES["ex3_15.m"]
            ),
        },
        "legs": legs,
        "gcrf_with_eop_corrections": {
            "r_m": km_to_m(VALLADO_GCRF_CORR_R),
            "v_mps": kmps_to_mps(VALLADO_GCRF_CORR_V),
            "source": SOURCES["aas-06-675"] + " (row 'GCRF iau76 w corr')",
            "note": "Informational: IAU-76/FK5 WITH ddpsi/ddeps applied (true GCRF to ~1 cm).",
        },
        "generator_self_check_residuals": {
            "note": (
                "Independent pyerfa IAU-76/FK5 chain vs the published rows at generation "
                "time (guards the transcription; not a fixture target)."
            ),
            **residuals,
        },
    }
    return fixture, residuals


def zero_eop_comparison(from_frame, to_frame, measured_m, measured_mps, sky_m, sky_mps) -> dict:
    comp = {
        "from_frame": from_frame,
        "to_frame": to_frame,
        "tolerance_m": position_tolerance_m(measured_m),
        "tolerance_mps": velocity_tolerance_mps(measured_mps),
        "measured_iau76_vs_astropy_m": round(measured_m, 4),
        "measured_iau76_vs_astropy_mps": round(measured_mps, 6),
    }
    if sky_m is not None:
        comp["skyfield_vs_astropy_m"] = round(sky_m, 4)
        comp["skyfield_vs_astropy_mps"] = round(sky_mps, 6)
    return comp


def build_zero_eop_fixture(case: dict) -> dict:
    t = utc_time(case["epoch_utc"], 0.0)
    r_teme, v_teme = case["r_teme_km"], case["v_teme_kmps"]

    astro = astropy_transform(t, TEME, r_teme, v_teme, {"GCRF": GCRS, "ITRF": ITRS}, 0.0, 0.0, 0.0)
    iau76 = iau76_chain_from_teme(t, r_teme, v_teme)
    sky = skyfield_from_teme(case["epoch_utc"], r_teme, v_teme)

    # Each comparison is measured exactly as the envelope test will run it:
    # IAU-76/FK5 applied to states[from_frame], compared against states[to_frame].
    # PEF == ITRF under zero polar motion, so iau76["PEF"] is the TEME->ITRF leg;
    # GCRF->ITRF starts from the astropy GCRF state (theory delta does not cancel).
    iau76_per_comparison = {
        ("TEME", "GCRF"): iau76["GCRF"],
        ("GCRF", "ITRF"): iau76_itrf_from_gcrf(t, astro["GCRF"][0], astro["GCRF"][1]),
        ("TEME", "ITRF"): iau76["PEF"],
    }
    comparisons = []
    for (from_f, to_f), chk in iau76_per_comparison.items():
        ref = astro[to_f]
        sky_ref = sky.get(to_f)
        comparisons.append(
            zero_eop_comparison(
                from_f,
                to_f,
                delta_m(chk[0], ref[0]),
                delta_mps(chk[1], ref[1]),
                None if sky_ref is None else delta_m(sky_ref[0], ref[0]),
                None if sky_ref is None else delta_mps(sky_ref[1], ref[1]),
            )
        )

    return {
        "case": case["name"],
        "description": (
            "Zero-EOP TEME/GCRF/ITRF golden vectors, astropy-computed (IAU-2006/2000A "
            "machinery) with UT1 = UTC and zero polar motion to match Iau76Fk5Provider "
            "defaults. " + case["comment"]
        ),
        "epoch_utc": case["epoch_utc"] + "Z",
        "eop": {
            "dut1_s": 0.0,
            "xp_arcsec": 0.0,
            "yp_arcsec": 0.0,
            "dat_s": int(erfa.dat(*[int(x) for x in case["epoch_utc"][:10].split("-")], 0.0)),
            "note": "UT1 = UTC and no polar motion: the Rust provider's documented zero defaults.",
        },
        "states": {
            "TEME": {
                "r_m": km_to_m(r_teme),
                "v_mps": kmps_to_mps(v_teme),
                "origin": "defined input (arbitrary LEO-magnitude state, not a published value)",
            },
            "GCRF": {
                "r_m": km_to_m(astro["GCRF"][0]),
                "v_mps": kmps_to_mps(astro["GCRF"][1]),
                "origin": "astropy TEME->GCRS (computed, not published)",
            },
            "ITRF": {
                "r_m": km_to_m(astro["ITRF"][0]),
                "v_mps": kmps_to_mps(astro["ITRF"][1]),
                "origin": "astropy TEME->ITRS (computed, not published)",
            },
        },
        "comparison_semantics": (
            "For each comparison: input = states[from_frame], expected = states[to_frame]; "
            "assert |transform(input) - expected| <= tolerance. measured_iau76_vs_astropy_* "
            "was produced the same way with a reference pyerfa IAU-76/FK5 chain; "
            "skyfield_vs_astropy_* cross-checks the expected (target) state itself."
        ),
        "comparisons": comparisons,
        "theory_note": (
            "Targets are astropy's IAU-2006/2000A + frame-bias reduction; the Rust side "
            "implements equinox-based IAU-76/FK5 without EOP corrections. The measured_* "
            "fields give the actual disagreement of a reference IAU-76/FK5 (pyerfa) chain "
            "at this epoch; tolerance = 1.5 x measured + 1 m (or + 5 mm/s), so the "
            "envelope both passes a correct IAU-76 implementation and fails the "
            "kilometre-scale TEME/GCRF conflation this change removes."
        ),
    }


# ---------------------------------------------------------------------------
# PROVENANCE.md
# ---------------------------------------------------------------------------
def package_versions() -> dict:
    packages = ("numpy", "astropy", "pyerfa", "skyfield", "astropy-iers-data")
    return {
        "python": platform.python_version(),
        **{p: importlib.metadata.version(p) for p in packages},
    }


def provenance_markdown(versions: dict, vallado_residuals: dict, zero_fixtures: list[dict]) -> str:
    lines = [
        "# Frame-Transform Golden Fixture Provenance",
        "",
        "Tier-3 golden vectors for design Decision 7 of the `astro-time-and-frames`",
        "change: the IAU-76/FK5 reduction chain (GCRF/MOD/TOD/TEME/PEF/ITRF) is",
        "validated against Vallado's published worked reduction plus independently",
        "generated astropy/Skyfield vectors. Consumed by the default-feature envelope",
        "test (no network, no Python). Units in the JSON files are metres and m/s;",
        "epochs are ISO-8601 UTC strings.",
        "",
        "## Generation record",
        "",
        "- Generator: `python/scripts/gen_frame_fixtures.py` (manual-only, never CI)",
        "- Regeneration command (from the workspace root, then commit the output):",
        "",
        "```sh",
        "uv run python/scripts/gen_frame_fixtures.py",
        "```",
        "",
        "- Toolchain at generation time:",
        "",
    ]
    lines += [f"  - {name} {ver}" for name, ver in versions.items()]
    lines += [
        "",
        "No IERS *values* reach any transform: dUT1 is forced at the IERS-table",
        "layer (a constant-valued `earth_orientation_table` science-state override,",
        "so astropy's finite-difference velocity machinery sees the fixture dUT1 on",
        "its shifted obstimes too - a per-Time `delta_ut1_utc` override does NOT",
        "cover those and was measured to contaminate velocities by ~omega^2*dUT1*r),",
        "and polar motion is forced at the transform layer (patched",
        "`get_polar_motion`). Repeated runs at the same versions are bit-identical.",
        "",
        "## `vallado-example-3-15.json` - published reduction",
        "",
        "All published numbers were fetched and verified this generation session from:",
        "",
        f"1. {SOURCES['aas-06-675']}",
        "   - ITRF input state and the 'PEF iau76', 'TOD iau76', 'MOD iau76',",
        "     'J2000 iau76', 'GCRF iau76 w corr', 'GCRF CIO dx,dy=0' rows, and the",
        "     time table (jdtt 2453101.82815474550).",
        f"2. {SOURCES['aiaa-2006-6753']}",
        "   - TEME position/velocity at the same epoch.",
        f"3. {SOURCES['ex3_15.m']}",
        "   - Exact EOP inputs: dut1 = -0.4399619 s, dat = 32 s, xp = -0.140682\",",
        "     yp = 0.333309\", lod = 0.0015563 s, ddpsi = -0.052195\", ddeps = -0.003875\".",
        "",
        "This is the same LEO test case as book Example 3-15 (Vallado, Fundamentals",
        "of Astrodynamics and Applications - ex3_15.m is its companion code); the",
        "papers above are the fetchable authoritative records of its outputs.",
        "",
        "The fixture's `GCRF` leg is the **'J2000 iau76'** row: the IAU-76/FK5",
        "reduction *without* the ddpsi/ddeps EOP nutation corrections, which is what",
        "`Iau76Fk5Provider` (dUT1 + polar motion only) implements. The corrected row",
        "('GCRF iau76 w corr') is recorded informationally; the two differ by ~0.9 m.",
        "",
        "Envelope tolerance: **1.0 m and 0.01 m/s per leg** (spec 'Vallado reduction",
        "reproduced'). Generation-time self-check - an independent pyerfa IAU-76/FK5",
        "chain reproduced the published rows to:",
        "",
        "| leg | dr (m) | dv (m/s) |",
        "|-----|--------|----------|",
    ]
    lines += [
        f"| {leg} | {res['dr_m']:.4f} | {res['dv_mps']:.6f} |"
        for leg, res in vallado_residuals.items()
    ]
    lines += [
        "",
        "The fixture also records astropy's GCRS/TEME outputs at the same epoch/EOP",
        "(IAU-2006/2000A theory) with their measured deltas from the published",
        "IAU-76/FK5 rows, so the envelope test can assert against either set.",
        "astropy's GCRS with the fixture EOP was itself checked against the paper's",
        "'GCRF CIO dx,dy=0' row (< 1 m) before being recorded.",
        "",
        "## `zero-eop-*.json` - astropy-computed vectors at provider-default EOP",
        "",
        "Input TEME states are arbitrary LEO-magnitude vectors (defined in the",
        "generator, not published values). Targets (GCRF, ITRF states) are computed",
        "by astropy's frame machinery (TEME -> GCRS, TEME -> ITRS; equinox-of-date",
        "TEME via GMST-1982, ITRS/GCRS via the IAU-2006/2000A CIO chain + frame",
        "bias) under the zero-EOP configuration matching the Rust provider defaults:",
        "UT1 = UTC (dUT1 = 0 forced at the IERS-table layer) and xp = yp = 0",
        "(patched polar-motion lookup).",
        "",
        "Because the Rust side implements equinox-based IAU-76/FK5, each comparison",
        "records `measured_iau76_vs_astropy_m`: the disagreement of a reference",
        "pyerfa IAU-76/FK5 chain (same theory as the Rust provider) with the astropy",
        "target at that epoch - i.e. the pure theory difference (frame bias",
        "~0.023 arcsec plus 1980-vs-2000A nutation, growing with distance from",
        "J2000). Per-comparison tolerance = 1.5 x measured + 1 m (velocities:",
        "1.5 x measured + 5 mm/s): a correct IAU-76 implementation passes with",
        "margin, while the ~kilometre TEME/GCRF conflation this change removes",
        "fails by orders of magnitude. Every comparison is measured in the exact",
        "direction the envelope test runs it (input = `states[from_frame]`,",
        "expected = `states[to_frame]`): notably GCRF -> ITRF starts from the",
        "astropy-computed GCRF state, so the inertial-orientation theory delta",
        "shows up there too and does NOT cancel. TEME -> ITRF is theory-free under",
        "zero EOP (both sides are the GMST-1982 spin), so its measured delta is ~0.",
        "",
        "Skyfield cross-check: the same TEME states pushed through Skyfield",
        "(`Timescale(delta_t = 32.184 + dAT)` so UT1 = UTC; no polar-motion table)",
        "agree with the astropy targets to the `skyfield_vs_astropy_m` field of each",
        "comparison (sub-metre; Skyfield's sgp4lib TEME and IAU-2000A-family ITRS).",
        "",
        "Measured agreement at generation time:",
        "",
        "| case | transform | iau76 vs astropy (m) | skyfield vs astropy (m) | tolerance (m) |",
        "|------|-----------|----------------------|-------------------------|---------------|",
    ]
    for fx in zero_fixtures:
        for comp in fx["comparisons"]:
            sky = comp.get("skyfield_vs_astropy_m")
            lines.append(
                f"| {fx['case']} | {comp['from_frame']}->{comp['to_frame']} | "
                f"{comp['measured_iau76_vs_astropy_m']:.3f} | "
                f"{'-' if sky is None else f'{sky:.3f}'} | {comp['tolerance_m']:.1f} |"
            )
    lines += [
        "",
        "## Value-source summary",
        "",
        "- Fetched-published: every number under `published`, `input`,",
        "  `gcrf_with_eop_corrections`, and the `eop` block of",
        "  `vallado-example-3-15.json` (sources 1-3 above).",
        "- Tool-computed (astropy/pyerfa/Skyfield at the versions above): the",
        "  `astropy_iau2006_2000a` blocks, all `states` marked GCRF/ITRF and all",
        "  `comparisons`/`measured_*` fields in `zero-eop-*.json`, and the",
        "  generator self-check residuals.",
        "- Defined inputs (not references): the TEME states marked",
        "  `origin: defined input` in `zero-eop-*.json`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def render_json(obj: dict) -> str:
    # allow_nan=False: a non-finite value anywhere in a fixture is a
    # generator bug — fail here rather than emit JSON Rust cannot parse.
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    json.loads(text)  # parse-back check on the exact rendered bytes
    return text


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    print(f"generator toolchain: {versions}")

    # Build and render every output in memory first; write only after all
    # succeed, so a failure partway never leaves a mixed fixture set on
    # disk.
    outputs: dict[str, str] = {}

    vallado, residuals = build_vallado_fixture()
    outputs["vallado-example-3-15.json"] = render_json(vallado)
    print("vallado-example-3-15.json rendered; pyerfa-vs-published residuals:")
    for leg, res in residuals.items():
        print(f"  {leg:5s} dr={res['dr_m']:.6f} m  dv={res['dv_mps']:.6f} m/s")

    zero_fixtures = []
    for case in ZERO_EOP_CASES:
        fx = build_zero_eop_fixture(case)
        zero_fixtures.append(fx)
        outputs[f"{case['name']}.json"] = render_json(fx)
        print(f"{case['name']}.json rendered; comparisons:")
        for comp in fx["comparisons"]:
            print(
                f"  {comp['from_frame']:>4s}->{comp['to_frame']:<4s} "
                f"iau76-vs-astropy {comp['measured_iau76_vs_astropy_m']:.3f} m "
                f"({comp['measured_iau76_vs_astropy_mps'] * 1000:.3f} mm/s), "
                f"skyfield-vs-astropy {comp.get('skyfield_vs_astropy_m', float('nan')):.3f} m, "
                f"tolerance {comp['tolerance_m']:.1f} m / {comp['tolerance_mps']:.4f} m/s"
            )

    outputs["PROVENANCE.md"] = provenance_markdown(versions, residuals, zero_fixtures)

    for name, text in outputs.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    print(f"all {len(outputs)} outputs written atomically to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
