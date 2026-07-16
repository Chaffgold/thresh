# Design — Atmospheric Measurement Propagation

## Context

The synthetic radar path (`thresh-synth/measurement_gen.rs::generate_radar`) produces geometric RAE truth plus Gaussian noise; the only atmosphere anywhere in the microwave chain is ITU-R P.676 attenuation feeding SNR/P_d (radar-equation spec). Real radar measurements carry two deterministic bias families this change adds: tropospheric refraction (elevation bending of order 0.1–10 mrad and range error of order metres–tens of metres, exploding at low elevation) and ionospheric group delay (metres at L-band scaling as 1/f²). The OTHR machinery (`thresh-synth/ionosphere.rs`, `ionospheric-propagation` spec) is HF sky-wave physics — untouched. The eval-consistency gates (#136) can now *measure* what an unmodeled bias does to a tracker, which makes the demonstration falsifiable rather than rhetorical.

Constraints: pure Rust, no new dependencies, deterministic CI, complexity ≤ 15, calibrated benchmarks bitwise-invariant by default (the standing pattern, held through three consecutive changes).

## Goals / Non-Goals

**Goals**
- Deterministic tropospheric refraction (Bean–Dutton exponential profile; 4/3-Earth cheap tier) and ionospheric group delay (configured VTEC, thin-shell mapping) as pure functions in thresh-core.
- Inverse-model correction hooks at the RAE→Cartesian seam, with residual-bias-after-mismatched-correction bounded in tests.
- Opt-in bias application in synthetic radar generation, default OFF.
- The corrected-vs-uncorrected consistency demonstration through the real benchmark runner.

**Non-Goals**
- MODTRAN/NRLMSISE, live TEC or space-weather ingestion, OTHR sky-wave changes, multipath, attenuation changes, filter-side bias-state estimation (future work the survey's §3.4 covers), scenario-TOML exposure of the bias knob (API/test-only until a committed scenario wants it — the `tracker_noise_sigma` precedent).

## Decisions

### Decision 1 — Module home: `thresh-core/src/propagation/` (refraction + iono_delay)

Measurement-domain propagation physics is neither `orbital` (dynamics) nor `geodetic` (coordinates): a new `propagation` module with `refraction.rs` and `iono_delay.rs` submodules, pure functions over scalars/geodetics, no state. thresh-synth and any tracker preprocessing consume the same functions — the bitwise-shared-physics discipline the orbital module established.

### Decision 2 — Refraction: fixed-quadrature ray integral through the Bean–Dutton exponential profile, 4/3-Earth as the cheap tier and cross-check

`N(h) = N_s·e^(−h/H)` (CRPL exponential reference atmosphere lineage; defaults `N_s = 313`, `H ≈ 7 km` — exact constants fetched at implementation time and cited). The high tier integrates the standard spherically-stratified ray equations (Bouguer's rule `n·r·cos E = const`) with a fixed-step composite quadrature from station height to target height, yielding both the elevation bending δE and the excess range `∫(n−1)ds` plus geometric path difference — fixed step count for determinism and complexity bounds. The cheap tier is the classic 4/3-Earth effective-radius model. Mandatory cross-check test: at moderate elevations (5°–45°) the two tiers agree within the documented band; published worked values (Bean & Dutton / radar-handbook lineage, fetched) pin the high tier. **Validity band explicit**: below a configurable minimum elevation (default 1°) the functions clamp and document — low-elevation ducting/super-refraction is out of scope.

### Decision 3 — Ionospheric delay: `Δr = K·STEC/f²`, thin-shell obliquity, configured VTEC only

`K = 40.308 m·Hz²/TECU-ish` (exact constant and units fetched from a published derivation and cited; the familiar "40.3" is a rounding). `STEC = VTEC·M(E)` with the standard thin-shell mapping `M(E) = 1/√(1 − (R_e·cos E/(R_e+h_s))²)`, shell height default 400 km (configurable, cited). Group delay lengthens measured range; microwave elevation bending from the ionosphere is negligible and documented as such. Config: `{ vtec_tecu, shell_height_m, frequency_hz }`. An optional bridge integrates the existing OTHR Chapman profile to a VTEC (unit-checked: electrons/m³ → TECU) so the two regimes can be made mutually consistent in scenarios that use both — the bridge is a convenience constructor, not a coupling. Frequency-band sanity tests pin L/S/X-band magnitudes (metres / decimetres / centimetres at representative TEC).

### Decision 4 — Bias-then-noise, applied to true RAE in generation; corrections at the RAE→Cartesian seam

Synthetic generation order: geometric RAE → apply refraction bending + range excess and iono range delay (the *apparent* measurement) → add Gaussian noise. The knob is an `Option<AtmosphereBiasConfig>` on the radar generation path, `None` by default — additive, serde-defaulted, every existing constructor and TOML unchanged. Corrections are the inverse models applied to a measured RAE before Cartesian conversion (`correct_refraction`, `correct_iono_delay`, composed `correct_atmosphere`); correcting the *apparent* elevation uses the same profile inversely (iterate once — the bending is small, one fixed-point step suffices and is documented). **Residual-honesty tests**: correcting with deliberately mismatched parameters (N_s ±10%, VTEC ±25%) leaves a residual bias that is measured and asserted to be an order smaller than the uncorrected bias — corrections are models, and the tests say exactly how good.

### Decision 5 — The consistency demonstration runs through the real benchmark runner

A code-constructed low-elevation scenario (the `tracker_noise_sigma` test pattern from eval-consistency 6.6): biased truth with an uncorrected tracker SHALL measurably degrade — MOTP inflated and ANEES/ANIS shifted beyond the honest run's values — while the same scenario with the correction applied at the conversion seam recovers to near-honest statistics. Geometry chosen at implementation for maximal signal (low elevation, L-band-ish frequency, documented); recorded as the demonstration the change exists to enable. No committed scenario TOML gains bounds or knobs — the demonstration is a test, not a new CI gate (calibrating a biased-scenario gate is future work once a consumer wants it).

### Decision 6 — Validation is component-level with fetched spot values; no python fixture generator

Refraction and iono delay are closed forms plus a fixed quadrature — a cross-implementation trajectory generator adds nothing here. Validation: fetched published worked values pin the refraction integral and the mapping function; the 4/3-Earth cross-check bounds the exponential tier; the iono formula is exact given its constant (provenance-pinned); checksum tripwires guard any embedded table. This is deliberately lighter than the last two changes' golden machinery — matched to the math's complexity, and recorded here so review doesn't mistake the absence of fixtures for an omission.

## Risks / Trade-offs

- **[Low-elevation divergence]** cot-like growth below ~2° → explicit validity band with clamping and documentation (Decision 2); the demonstration scenario stays inside the band.
- **[Sign/convention errors]** apparent-vs-true elevation and delay sign are classic silent errors → conventions stated once at the module doc (measured = apparent = true + bias; correction subtracts), round-trip tests (bias then correct with true parameters recovers truth to quadrature tolerance).
- **[Double-counting with attenuation]** none — attenuation is amplitude/SNR, this is geometry/timing; stated in docs.
- **[Chapman-bridge unit slip]** electrons/m³ integrated to TECU (1 TECU = 1e16 el/m²) → dedicated unit test with a hand-integrated profile.
- **[Benchmark drift]** default-OFF plus the standing digit-for-digit invariance verification.

## Migration Plan

1. `propagation` module: refraction (both tiers) + iono delay + corrections, component tests with fetched constants — additive.
2. thresh-synth opt-in bias application (bias-then-noise), generation-side tests — additive, default OFF.
3. Chapman→VTEC bridge + unit test — additive.
4. Demonstration test through the benchmark runner (corrected vs uncorrected vs honest); benchmark-invariance verification run.
5. Wrap-up gates; divergence/open-question records.

Each step additive and independently revertible.

## Open Questions

- **Demonstration geometry**: low-elevation aircraft scenario vs a LEO pass through low elevation — pick whichever produces the cleaner ANEES/MOTP separation at implementation time and record the numbers.
- **Where the correction hook lives for the benchmark runner**: a wrapper around `measurement_to_cartesian` vs a preprocessing step in the runner — decide when wiring the demonstration; the correction functions themselves are seam-agnostic.
- **Refraction quadrature step count**: fixed count vs fixed step-length — pick by measuring the published-value reproduction error at both, document the choice.
