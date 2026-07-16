# Tasks — Atmospheric Measurement Propagation

> Phases follow design.md's Migration Plan: the propagation module first (pure functions, additive), then the opt-in synth application, then the Chapman bridge, then the consistency demonstration with the standing benchmark-invariance verification. The falsifiable exit criterion is task 4.4.

## 1. Propagation module — `thresh-core/src/propagation/` (design Decisions 1–3)

- [ ] 1.1 Create the `propagation` module (`refraction.rs`, `iono_delay.rs`); module docs state the conventions once: measured = apparent = true + bias, corrections subtract, attenuation is explicitly not this module's concern.
- [ ] 1.2 Refraction, exponential tier: Bean–Dutton profile `N(h) = N_s·e^(−h/H)` with published default constants fetched at implementation time (cited inline, checksum-tripwired if tabular); spherically-stratified ray integral (Bouguer's rule) via deterministic fixed quadrature returning elevation bending and excess range; resolve the design open question fixed-count-vs-fixed-step by measuring published-value reproduction at both, recording the decision in design.md. Tests: fetched published worked values within documented tolerance (spec: "Published worked values reproduced"); monotonic growth toward the horizon (spec: "Bending grows toward the horizon"); bitwise-deterministic repeated evaluation.
- [ ] 1.3 Refraction, 4/3-Earth tier + cross-check test at 5°–45° within the documented band (spec: "Tiers agree at moderate elevation"); validity floor (default 1°) with clamp-and-document behavior and test (spec: "Below-floor evaluation clamps").
- [ ] 1.4 Iono delay: `Δr = K·STEC/f²` with K's exact value/units from a fetched published derivation (cited at the constant); thin-shell obliquity mapping with configurable shell height (published default cited); doc note that microwave iono bending is negligible and unmodeled. Tests: exact 1/f² scaling (spec: "Inverse-square frequency scaling"); L/S/X-band magnitude regimes (spec: "Band magnitudes are physical"); monotonic obliquity toward the horizon (spec: "Obliquity grows toward the horizon").
- [ ] 1.5 Corrections: `correct_refraction` (one documented fixed-point step on the inverse), `correct_iono_delay` (closed-form), composed `correct_atmosphere`. Tests: bias-then-correct round trip with true parameters — refraction within quadrature tolerance, iono to floating-point (specs: "Round trip with true parameters", "Closed-form round trip"); mismatched-parameter residuals measured and recorded — N_s ±10% leaves residual ≥ 10× smaller than uncorrected (spec: "Mismatched correction still helps"), VTEC ±25% leaves exactly the mis-set fraction (spec: "Mismatched TEC correction still helps").

## 2. Synth opt-in application — `thresh-synth` (design Decision 4)

- [ ] 2.1 `AtmosphereBiasConfig` (refraction profile + iono config + frequency) as an `Option` on the radar generation path, `None` default, serde-defaulted; bias applied to true RAE before noise (bias-then-noise). Every existing constructor, test, and TOML unchanged — grep-verify additivity.
- [ ] 2.2 Generation tests: config-absent output bitwise identical to the pre-change generator at the same seed (spec: "Default generation is unchanged"); enabled-with-zero-noise output equals truth plus exactly the modeled biases (spec: "Bias applied before noise").

## 3. Chapman → VTEC bridge (design Decision 3)

- [ ] 3.1 Bridge integrating the existing OTHR Chapman profile to vertical TEC with explicit electrons/m² → TECU conversion; unit test against a hand-integrated profile within documented quadrature tolerance (spec: "Hand-integrated profile matches"). OTHR code untouched.

## 4. Demonstration + invariance (design Decision 5)

- [ ] 4.1 Wire the correction seam for the demonstration (design open question: wrapper around `measurement_to_cartesian` vs runner preprocessing — decide, record in design.md); no committed scenario TOML gains knobs or bounds.
- [ ] 4.2 Demonstration geometry: pick low-elevation aircraft vs LEO-pass (design open question) for the cleanest separation; record the choice and the numbers in design.md.
- [ ] 4.3 The three-run demonstration test through the real benchmark runner: honest / biased-uncorrected / biased-corrected — uncorrected degrades MOTP and shifts ANEES and/or ANIS beyond the honest values by documented margins; corrected recovers within documented bands; deterministic across same-seed runs (specs: "Uncorrected bias degrades the tracker measurably", "Correction recovers near-honest statistics", "Demonstration is deterministic"). All three runs' values recorded at the test.
- [ ] 4.4 Exit criterion (numeric, falsifiable): the published refraction worked values reproduce within tolerance; the bias-then-correct round trips hold; and the demonstration's three-run table shows uncorrected degradation and corrected recovery by its documented margins — the bias class is real, visible to the evaluation layer, and recoverable.
- [ ] 4.5 Benchmark invariance: the four calibrated scenarios digit-for-digit identical before vs after (spec: "Invariance verification"); record the comparison in the PR description.

## 5. Wrap-up

- [ ] 5.1 Full gates: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, rustdoc `-Dwarnings`, `openspec validate --all --strict`; complexity spot-check on the ray-integral and mapping helpers.
- [ ] 5.2 Update proposal.md/design.md with implementation-time divergences and the resolutions of the three design Open Questions (demonstration geometry, correction-seam home, quadrature discretization).
