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

`K = 40.308 m·Hz²/TECU-ish` (exact constant and units fetched from a published derivation and cited; the familiar "40.3" is a rounding). `STEC = VTEC·M(E)` with the standard thin-shell mapping `M(E) = 1/√(1 − (R_e·cos E/(R_e+h_s))²)`, shell height default 450 km — the implementation-time citations (GLONASS IAC single-layer convention; Ren et al. 2019's 350–450 km practice band) superseded this decision's original 400 km sketch (configurable, cited). Group delay lengthens measured range; microwave elevation bending from the ionosphere is negligible and documented as such. Config: `{ vtec_tecu, shell_height_m, frequency_hz }`. An optional bridge integrates the existing OTHR Chapman profile to a VTEC (unit-checked: electrons/m³ → TECU) so the two regimes can be made mutually consistent in scenarios that use both — the bridge is a convenience constructor, not a coupling. Frequency-band sanity tests pin L/S/X-band magnitudes (metres / decimetres / centimetres at representative TEC).

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

- **Demonstration geometry**: low-elevation aircraft scenario vs a LEO pass through low elevation — pick whichever produces the cleaner ANEES/MOTP separation at implementation time and record the numbers. **RESOLVED (task 4.2): low-elevation aircraft.** A new `"low-elevation"` synthetic scenario type (four aircraft-like CV targets at 37–50 km slant range, 4°–6° elevation, flying outbound so the bias is time-varying) driven by the RAE radar model with a tightened angular sigma (`/250 000`, so random cross-range error stays a few metres and the deterministic bias dominates), biased by CRPL refraction at sea level. A 25 TECU L-band (1.3 GHz) ionospheric term is configured but **inert**: aircraft altitudes sit far below the 450 km thin shell, so the target-altitude step gate withholds the full-traversal delay on both the forward and the correction path — the demonstration bias is refraction-only in effect (elevation bending dominates: ~0.05° maps to tens of metres of cross-range error at this slant range, on top of the ~10–12 m excess range), and the demo doubles as an end-to-end check that no unphysical iono bias is attributed to aircraft. Chosen over a LEO pass because it runs under **default features** (no `orbital`/`sgp4`), so the demonstration test lives in the standard `cargo test --workspace` gate. Recorded three-run values through the real `run_synthetic_benchmark` (fixed seed; all three runs share the noise realisation and differ only by the bias):

  | run          | MOTA   | IDF1   | MOTP (m) | ANEES  | ANIS   |
  |--------------|--------|--------|----------|--------|--------|
  | honest       | 0.9512 | 0.9750 | 34.808   | 1.5440 | 1.5947 |
  | uncorrected  | 0.9512 | 0.9750 | 84.648   | 6.2900 | 1.5948 |
  | corrected    | 0.9512 | 0.9750 | 34.803   | 1.5438 | 1.5947 |

  The tropospheric bias alone inflates MOTP 2.43× and ANEES 4.07× while leaving MOTA and IDF1 bit-identical (the systematic offset is invisible to detection). **ANIS is essentially unmoved** — the smoothly-varying bias is absorbed into the CV filter's state, so innovation self-consistency cannot see it while estimate-vs-truth consistency (ANEES) can; ANEES is therefore the sensitive statistic and the demonstration is scored on the ANEES/MOTP separation. Correction (same parameters inverted at the seam, with the same altitude gate) recovers MOTP to within 0.02% and ANEES to within 0.02% of honest. See `atmospheric_bias_demonstration_through_runner` in `crates/thresh-data/src/benchmark.rs`.
- **Where the correction hook lives for the benchmark runner**: a wrapper around `measurement_to_cartesian` vs a preprocessing step in the runner — decide when wiring the demonstration; the correction functions themselves are seam-agnostic. **RESOLVED (task 4.1): wrapper around `measurement_to_cartesian`.** A `measurement_to_cartesian_corrected(measurement, &AtmosphereBiasConfig)` helper inverts the RAE bias with `thresh_core::propagation::correct_atmosphere` (the same parameters the generator used) and then runs the identical `range·(cos/sin)` reconstruction; the runner's measurement loop selects it only when a new serde-defaulted `ScenarioParameters::atmosphere_correction` flag is set *and* an `atmosphere_bias` config is present. Chosen over runner preprocessing because it keeps the default (correction-off) path byte-identical — the plain `measurement_to_cartesian` call is unchanged — so the benchmark-invariance guarantee is preserved by construction. No committed scenario TOML gains knobs or bounds; `atmosphere_bias`/`atmosphere_correction` default to off exactly like the `tracker_noise_sigma` precedent.
- **Refraction quadrature step count**: fixed count vs fixed step-length — pick by measuring the published-value reproduction error at both, document the choice. **RESOLVED (task 1.2): fixed count** — `RAY_QUADRATURE_INTERVALS = 1024` composite-Simpson intervals over the capped `[h_s, ATMOSPHERE_TOP_M=80 km]` domain. Measured against ITU-R P.834-8 Eq. (9) (fetched 2026-07-16): the ray-turning bending reproduces Eq. (9) within 6% over 3°–10° (the band where Eq. 9 is authoritative) and the zenith excess matches the analytic column delay `N_s·H·1e-6` (Eq. 20) to `<0.1 mm`, converged to `~1e-6` relative by 512 intervals. Fixed step-length was rejected: it scales the interval count and cost with the target-height integration span while giving no accuracy benefit on the bounded, exponentially-concentrated (surface-weighted) integrand; capping the domain at 80 km makes the fixed count height-independent. Fixed count gives deterministic `O(1)` cost, bitwise-repeatable output, and a bounded-complexity loop (mirrors the ionospheric `CHAPMAN_SIMPSON_INTERVALS` precedent).

## Implementation-Time Divergences (task 5.2)

- **Bias threading**: the opt-in lives in a new `generate_radar_biased` entry point (plus a benchmark-local `run_scenario_biased` helper), not a field on `RadarConfig` — a required field would have forced edits to out-of-scope full struct literals, and the entry-point shape keeps every existing constructor byte-for-byte unchanged.
- **Shell-height default is 450 km, not Decision 3's original 400 km sketch** (Decision 3's text now carries the correction): the fetchable authoritative convention (GLONASS IAC single-layer method) pins 450 km, inside the published 350–450 km practice band (Ren et al. 2019); configurable via `IonoDelayConfig::shell_height_m`; thin-shell geometry uses the IUGG mean radius (6371 km), matching the single-layer literature and the OTHR module.
- **ANIS is structurally blind to the bias** (measured: 1.5947 → 1.5948 uncorrected): a smoothly-varying measurement bias is absorbed into the CV filter's state, so innovation self-consistency cannot see what truth-referenced ANEES sees (4.07× inflation) — the demonstration satisfies the spec's "ANEES and/or ANIS" via ANEES and turns the NEES-vs-NIS distinction into a measured example.
- **The ionospheric leg is a step function of target altitude** (not sketched in Decision 4, which assumed full-traversal geometry): `apply_to_rae` adds the thin-shell slant delay only when the target altitude is at or above the configured shell height — exo-ionospheric targets traverse effectively all of the TEC, sub-ionospheric targets (aircraft) essentially none, and fractional traversal for targets inside the ionosphere is out of scope (documented at `IonoDelayConfig` and `apply_to_rae`). The benchmark correction seam gates the inverse identically, so no delay that was never applied gets subtracted. Consequence for the demonstration: the configured iono term correctly contributes zero at 4 km, making the demo refraction-only in effect (fresh numbers in the resolved open question above).
- **Refraction bending is exact for exo-atmospheric targets, a conservative bound (≤ 2×) for targets embedded in the troposphere**: the exact in-atmosphere apparent-minus-true value needs a two-point boundary-value ray solve, deferred and documented; `apply_refraction`/`correct_refraction` are self-consistent so round trips are unaffected.
- **Published-value anchor is ITU-R P.834-8** (closed-form bending Eq. 9, excess Eqs. 15/16/20; read visually from the fetched PDF) rather than Bean–Dutton book tables, which are scanned-image PDFs that do not extract; CRPL default constants cite the MathWorks/Bean–Thayer lineage. Zenith excess reproduces the canonical ~2.31 m dry-delay figure to < 0.1 mm.
- **Refraction mismatch floor is the mis-set fraction**: N_s ±10% leaves ≈ 10% of the bias (measured 9.7%/10.3%) because the bias is near-linear in N_s — same intrinsic floor as the iono sibling. The tropospheric-refraction delta spec now states this truthful guarantee directly (the residual approximately equals the mis-set fraction of the uncorrected bias, mirroring the iono spec's phrasing), so this is a recorded property, not a divergence; the test pins the residual ratio to the [0.05, 0.13] band.
