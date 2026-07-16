# Design — Astrodynamic Time Scales and Reference Frames

## Context

Time in thresh is a bare `f64`: `thresh_core::time::Timestamp(pub f64)` (relative seconds, zero consumers outside its own file) and `OrbitalState::epoch_jd: f64` (a Julian date of unspecified time scale, consumed by nine files across thresh-core/-synth/-filter/-data). Frames are half-disciplined: `orbital-ballistic-filter-models` landed the `OrbitalFrame` tag enum and a `FrameMismatch` error variant, but the only rotation in the tree is `eci.rs`'s GMST spin, which conflates J2000 with TEME — the frame SGP4 actually emits. The gap is kilometres at LEO radius (TEME↔J2000 precession/nutation reaches ~0.8°). The benchmark pipeline is *internally consistent* in the wrong frame: truth and measurements both flow through the same GMST-only chain, so MOT metrics never noticed.

Constraints: pure Rust, no AGPL (nyx-space rejected), no GPU, deterministic CI with no network/Python (golden tier-3 fixture pattern), SonarCloud complexity ≤ 15, calibrated benchmark floors must not silently move.

## Goals / Non-Goals

**Goals**
- Time-scale-aware epochs (UTC/TAI/GPS/TT; leap-second correct) as the single epoch currency for the space lane.
- TEME↔GCRF↔ITRF transforms via IAU-76/FK5, with full 6×6 covariance rotation, behind a provider seam.
- Harden `OrbitalFrame` enforcement from debug-assert to real errors, with real conversions to point to.
- Golden-vector validation to sub-arcsecond rotation accuracy, committed as fixtures.

**Non-Goals**
- EOP ingestion (ΔUT1, polar motion time series) — parameters exist with documented zero defaults; feeding them live data is future work.
- IAU-2006/2000A CIO transforms, cislunar/ANISE providers, propagator force models, retiring the `hifi-orbital` spec (owned by `orbit-propagation-fidelity`).
- Migrating tracker-lane `f64` relative time (`dt` seconds in `step()`) — trackers keep relative seconds; only the astro path needs absolute epochs.

## Decisions

### Decision 1 — hifitime 4.x as the epoch backbone, wrapped in a thresh-core newtype

`hifitime` (MPL-2.0, pure Rust, no transitive weight, `serde` feature) provides TAI-anchored epochs with leap-second tables and UTC/TAI/GPS/TT conversions at sub-nanosecond precision. thresh-core wraps it: `pub struct Epoch(hifitime::Epoch)` in `time.rs`, exposing exactly the surface thresh needs — constructors from UTC calendar/ISO-8601/JD-with-scale, accessors `to_jde_utc_days()`/`to_jde_tt_days()`, `Sub` yielding seconds (`f64`), serde as ISO-8601 UTC string. The newtype keeps hifitime out of the public API (swappable, and controls the serialized form). *Alternatives*: `chrono` (no leap seconds, no TT — wrong tool); hand-rolled two-part JD (reinvents leap-second bookkeeping we would then own forever); re-exporting `hifitime::Epoch` directly (leaks a semver-uncontrolled type into every downstream signature).

License note for the record: nyx-space (AGPL-3.0) stays rejected; hifitime is the same author but MPL-2.0 — file-level copyleft that does not encumber the workspace when used unmodified.

### Decision 2 — IAU-76/FK5 with the full 1980 nutation series, table-driven

The reduction chain is Vallado's: GCRF —P(zeta, theta, z)→ MOD —N(Δψ, Δε, 1980 series)→ TOD —R₃(GAST)→ PEF —W(xₚ, yₚ)→ ITRF, and TEME joins at the true-equator/mean-equinox point: TEME = R₃(−Eq_equinox)·TOD, TEME —R₃(GMST)→ PEF. The 1980 nutation model ships as a const 106-term table (data, not code — a table-driven loop keeps cognitive complexity trivial and transcription errors are exactly what the golden vectors exist to catch). Truncated 4-term nutation was rejected: it saves nothing meaningful in Rust and its ~0.5″ error is the same order as the effects this change exists to stop ignoring. IAU-76/FK5 over IAU-2006/2000A because it is the reduction SGP4/TEME literature is defined against and Vallado's worked examples validate it end-to-end; the provider seam (Decision 4) is where a CIO-based implementation would slot later.

### Decision 3 — Covariance transforms are part of every frame transform, with the rate term where it matters

Each transform maps `(r, v, P₆ₓ₆)`: the 6×6 Jacobian is block `[[R, 0], [Ṙ, R]]`. For inertial↔ITRF, `Ṙ` carries the Earth-rotation term (ω⊕ × — the ~465 m/s equatorial velocity swing); for inertial↔inertial (GCRF↔TEME/MOD/TOD), precession/nutation rates are ~1e-12 rad/s and `Ṙ` is taken as zero — documented at the API, tested by asserting a GCRF↔TEME round trip preserves P to tolerance while GCRF↔ITRF visibly rotates the velocity block. Position-only 3×3 variants exist for measurement-space use (the ENU chain). *Alternative* — state-only transforms with covariance left to callers — is exactly how R·P·Rᵀ gets forgotten; the integration analysis called for covariance rotation to be explicit in scope.

### Decision 4 — `FrameProvider` trait: rotations are data, frames are an enum

```rust
pub trait FrameProvider {
    fn rotation(&self, from: Frame, to: Frame, epoch: &Epoch) -> Result<FrameRotation, FrameError>;
}
```
`Frame` is a plain enum (`Gcrf`, `Mod`, `Tod`, `Teme`, `Pef`, `Itrf` — plus the existing ENU story composing on top); `FrameRotation` carries `R: Matrix3` and `r_dot: Matrix3` so covariance handling (Decision 3) is provider-independent. `Iau76Fk5Provider { delta_ut1: f64, polar_motion: Option<(f64, f64)> }` is the default, with zero defaults documented as the accuracy floor (|ΔUT1| ≤ 0.9 s ⇒ ≤ ~430 m equatorial ECEF longitude error; polar motion ≤ ~15 m). A later ANISE/cislunar provider implements the same trait without touching call sites. Free functions (`teme_to_gcrf(state, cov, epoch)` etc.) wrap the default provider for the common path. *Alternative* — methods on `OrbitalState` — couples the state type to one reduction theory and makes the ANISE seam a breaking change.

### Decision 5 — `OrbitalFrame` hardening and the epoch migration are one atomic breaking step

`OrbitalState.epoch_jd: f64` → `epoch: Epoch` (**BREAKING**), and the arithmetic helpers' debug-asserts become returned `FrameMismatch` errors, in the same commit that gives callers real conversions to reach for. The nine `epoch_jd` call sites migrate mechanically (`Epoch::from_jde_utc(...)` at construction, `.to_jde_utc_days()` where a JD is still needed — notably `gmst()`, which gains an `Epoch`-taking form and keeps a doc note that its argument is UT1≈UTC). The existing `OrbitalFrame` enum grows the real vocabulary (TEME/GCRF/ITRF) or is unified with `Frame` — unify: one enum, re-exported under both names during the change, single source of truth after.

### Decision 6 — SGP4 ingest tags TEME; the benchmark chain stays TEME end-to-end (zero expected baseline movement)

`thresh-data`'s SGP4 output is tagged `Frame::Teme` (it always was TEME — now the type says so). The orbital benchmark's ENU projection keeps its GMST rotation, which is *correct for TEME→PEF* — meaning today's internally-consistent chain was consistent in TEME, not J2000, and stays bitwise unchanged. This is the design's benchmark-invariance argument: this change re-labels the existing chain truthfully and adds new conversions; it does not alter any number the calibrated floors (ISS 0.74 / Starlink 0.77 / ballistic 0.84, ANEES/ANIS on synth-cv-clean) see. A determinism check before/after the migration commit enforces that; any observed drift is a bug in the migration, not a re-calibration event.

### Decision 7 — Golden vectors: Vallado's worked reduction + astropy cross-check, tier-3 fixtures

`test-data/golden/frames/` gets (a) Vallado Example 3-15 (the canonical IAU-76/FK5 reduction of an ITRF state to GCRF/MOD/TOD/TEME/PEF at 2004-04-06 07:51:28.386 UTC, with ΔUT1/xₚ/yₚ given — exercises every leg including polar motion and ΔUT1 as *explicit parameters*), and (b) astropy/Skyfield-generated TEME↔GCRF↔ITRS vectors at 2–3 additional epochs with zero-EOP settings matching the provider defaults. A manual-only generator script (python/ tree, never CI) plus `PROVENANCE.md` records versions and regeneration commands; the envelope test runs under default features, no network, asserting position agreement ≤ 1 m (Vallado, with his EOP inputs) and ≤ documented bounds for the zero-EOP fixtures. Leap-second sanity: an epoch pair straddling 2016-12-31 asserts the 37th leap second through `Epoch` arithmetic.

## Risks / Trade-offs

- **[Nutation-table transcription]** 106 × 9 coefficients hand-carried from the IAU-1980 table → golden vectors catch any wrong term at the sub-arcsecond level; the table cites its source edition inline.
- **[hifitime major-version churn]** 4.x API is stable but the newtype (Decision 1) contains any future migration to one file.
- **[UT1≈UTC floor]** ≤ ~430 m ECEF longitude ambiguity until EOP ingestion exists → parameters are explicit with documented defaults; fixtures that need ΔUT1 (Vallado) pass it explicitly, so the code path is proven even though the default is zero.
- **[Hidden frame assumptions downstream]** some consumer may compensate for the TEME/J2000 conflation implicitly → the tag hardening turns those into loud `FrameMismatch` errors during migration, and the benchmark-invariance check (Decision 6) catches numeric drift.
- **[Complexity gate]** precession/nutation/sidereal formulas are polynomial-heavy → phase-helper decomposition per CLAUDE.md (one function per rotation leg, table-driven nutation loop, worked example: `rk4_stage`).

## Migration Plan

1. `astro-time`: `Epoch` newtype + conversions + serde + leap-second tests (thresh-core only, additive).
2. `frames` module: rotation legs (P, N, sidereal, W), TEME junction, provider trait + IAU-76/FK5 provider, covariance transforms, golden fixtures + envelope tests (additive).
3. **BREAKING commit**: `OrbitalState.epoch` migration + `FrameMismatch` enforcement + `OrbitalFrame`/`Frame` unification + nine call-site migrations + benchmark-invariance verification (Decision 6).
4. SGP4 TEME tagging + `gmst(Epoch)` form + doc sweeps; full gates.

Each step is independently revertible; step 3 is the only one touching calibrated paths and carries the bitwise-invariance check in the same commit.

## Open Questions

- **Serde format for `Epoch` in golden fixtures**: ISO-8601 UTC string (readable, scale-explicit) vs keeping `epoch_jd` floats in existing SGP4 fixture JSON (no regeneration). Leaning: new frame fixtures use ISO strings; existing SGP4 fixtures keep their floats with the loader converting (documented scale: UTC) — decide at task time by whichever keeps `test-data/golden/orbital/` byte-identical.
- **`Timestamp`'s fate**: zero consumers outside its file; deprecate now or leave until a tracker-lane change needs absolute time? Leaning: leave, with a doc pointing to `Epoch` for absolute time.

Both open questions resolved as leaned (2026-07-16): new frame fixtures serialize epochs as ISO-8601 UTC strings while `test-data/golden/orbital/` stayed byte-identical (loader converts the `epoch_jd` floats, documented scale UTC); `Timestamp` is left untouched with a doc note routing absolute-time users to `Epoch`.

## Implementation-Time Divergences (task 5.4)

- **Synth-internal time stays raw f64 JD where bitwise invariance demanded it.** `thresh_synth::OrbitalState.epoch_jd`, `BallisticProfile.epoch_jd`, and the benchmark-internal JD plumbing keep raw UTC Julian-date floats: JD↔Epoch round trips measured non-bit-exact through hifitime's internal representation, and Decision 6's invariance contract outranks type purity inside the truth-generation chain. `Epoch` appears at every API boundary (synth↔core `From`/`TryFrom` conversions, `Tle::epoch()`, `Sgp4Fixture::epoch()`), and each raw-JD site documents its scale and the rationale.
- **`OrbitalFrame::EciGmst` removed, not kept.** Decision 5 said "unify"; the unification remapped the former `EciGmst` tag to `Frame::Teme` — the GMST-only chain always was TEME-consistent (Decision 6's argument), so a separate variant would perpetuate the ambiguity this change exists to end. `pub use Frame as OrbitalFrame` preserves the old name during the transition.
- **The GCRF golden leg is FK5-J2000 (no δΔψ/δΔε corrections).** The Vallado fixture's target row is "J2000 iau76" — exactly what `Iau76Fk5Provider` implements; the EOP-corrected "GCRF iau76 w corr" row is recorded informationally. The ~0.7 m frame-bias gap between FK5-J2000 and true GCRF sits inside the 1 m envelope and is documented in the fixture provenance.
- **No `lib.rs` re-export for `Epoch`.** thresh-core exposes module paths only (no top-level `pub use` anywhere); `thresh_core::time::Epoch` follows the crate's convention rather than the task's "lib.rs (re-export only)" allowance.
- **Cross-theory golden tolerances are measured, not assumed.** The zero-EOP fixtures compare an IAU-76/FK5 implementation against astropy's IAU-2006/2000A chain; per-comparison tolerances are 1.5× the *measured* theory delta plus 1 m (2.2–2.5 m inertial legs, 1.0 m TEME→ITRF), each measured in the direction the envelope test runs, cross-checked by Skyfield to ≤ 0.4 mm.
- **One pre-existing rustdoc failure fixed in passing**: `run_orbital_benchmark`'s public doc linked the private `effective_cartesian_sigma` (failed `-Dwarnings` under `--features orbital`); demoted to a code span.
