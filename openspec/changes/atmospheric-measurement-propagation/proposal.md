# Atmospheric Measurement Propagation

## Why

thresh's microwave radar path models the atmosphere only as an SNR/P_d loss (ITU-R P.676 attenuation in the radar equation) — the range and elevation *measurements* themselves carry zero tropospheric refraction and zero ionospheric group delay, so synthetic truth is unphysically clean and trackers never face the dominant real-world bias class (tens of metres of range bias and milliradians of elevation bending at low elevation angles). With the propagation tier otherwise complete and the eval-consistency gates able to *measure* covariance honesty, this last roadmap change closes the loop: physically-biased measurements, the correction hooks trackers need, and an end-to-end demonstration that the consistency layer catches the bias when it goes uncorrected.

## What Changes

- **thresh-core**: deterministic tropospheric refraction — Bean–Dutton exponential refractivity profile `N(h) = N_s·e^(−h/H)` driving elevation-angle bending and range error for a ground radar observing an elevated target, with the classic 4/3-Earth effective-radius model as the cheap tier and cross-check; constants fetched from published sources with checksum tripwires.
- **thresh-core**: ionospheric group delay for microwave bands — `Δrange = 40.3·STEC/f²`, slant TEC from a configured vertical TEC through a thin-shell obliquity mapping; frequency-dependent (L/S/X-band regimes), elevation-dependent. Static configured TEC only (space-weather ingestion stays out of scope); an optional consistency bridge integrates the existing OTHR Chapman profile to a vertical TEC.
- **thresh-core**: inverse-model correction hooks applicable at the RAE→Cartesian conversion step, so a tracker can compensate before filtering; corrections are honest about being models — the residual bias after correcting with mismatched parameters is documented and bounded in tests.
- **thresh-synth**: radar measurement generation optionally applies the biases to generated RAE measurements (config knob, **default OFF** — every calibrated benchmark stays bitwise unchanged, the invariance pattern of the last three changes).
- **Falsifiability demonstration**: an end-to-end test where biased truth + an uncorrected tracker measurably degrades (MOTP/ANEES beyond the honest run's values) while the corrected tracker recovers — the eval-consistency layer catching a physics-modeling gap, which is exactly what it was built for.
- **Untouched**: the OTHR/HF sky-wave machinery (Chapman/MUF/skip-zone — a different regime), the radar-equation attenuation path (SNR is its concern), multipath, and any tracker-side bias-state estimation (future filter work).

## Capabilities

### New Capabilities

- `tropospheric-refraction`: Bean–Dutton exponential-profile refraction (elevation bending + range error) with the 4/3-Earth tier, published-constant provenance, correction hooks, and residual-bias bounds.
- `ionospheric-group-delay`: frequency- and elevation-dependent microwave range delay from configured vertical TEC via thin-shell obliquity mapping, with the 40.3 coefficient's provenance, the optional Chapman-integration bridge, and correction hooks.
- `biased-measurement-generation`: opt-in application of the atmospheric biases in synthetic radar generation, default-off with bitwise benchmark invariance, plus the corrected-vs-uncorrected consistency demonstration.

### Modified Capabilities

_None. The `ionospheric-propagation` (OTHR sky-wave) and `radar-equation` (attenuation/SNR) specs are deliberately untouched; this change adds sibling capabilities for the microwave measurement-bias regime._

## Impact

- **Crates**: `thresh-core` gains an atmospheric-propagation module (refraction + group delay + corrections — pure functions, no new dependencies); `thresh-synth` gains the opt-in bias application in radar generation; `thresh-data`/tests gain the demonstration scenario path (API-level; whether any scenario TOML exposes the knob is a design open question, leaning API-only per precedent).
- **Benchmarks**: bitwise-invariant by default (verification run required, per the standing pattern).
- **Out of scope**: MODTRAN/NRLMSISE, live TEC/space-weather data, OTHR changes, multipath, attenuation changes, automatic bias estimation in filters (the survey's §3.4 bias-state work).
- **Downstream**: completes the four-change propagation tier; gives future bias-estimation filter work (augmented-state or consider-state) a physically meaningful bias to estimate.
