# Advanced-Methods Doc — Integration Analysis

**Source doc:** `thresh-advanced-methods.md` (this worktree root)
**Codebase snapshot:** `develop` @ `d33f243` (worktree `advanced-methods`)
**Cross-referenced against:** live code in `crates/` and `python/`, live specs in `openspec/specs/`, archived changes in `openspec/changes/archive/`, the **in-flight** change `orbital-ballistic-filter-models` (implementing now), and the three **planned** changes `astro-time-and-frames`, `orbit-propagation-fidelity`, `atmospheric-measurement-propagation`.
**Method:** five parallel analysts, one per doc-section group (1.x / 2–3 / 4–5 / 6–7–11 / 8–9–10), each verifying every named technique with repo-wide greps and source reads.

---

## 1. Duplication summary

**81 unique techniques** were assessed after deduplicating cross-section repeats (NEES/NIS appears in §2, §9, §11.1; RTS in §7 and §10.3; GLR/CUSUM in §4 and §11.4; PMBM/TPMBM in §5 and §10.3; OOSM in §6 and §10.4; GMM splitting in §2 and §10.2; SR/UD forms in §3.1 and §11.3; GGIW in §1.1, §5, §11.2).

| Bucket | Count | Share |
|---|---|---|
| Already implemented (fully) | 1 | 1% |
| In-flight (`orbital-ballistic-filter-models`) | 2 | 2% |
| Planned (the 3 queued changes) | 2 | 2% |
| Partial (real half exists in-tree) | 22 | 27% |
| Genuinely new | 54 | 67% |

Headline: **two-thirds of the doc is genuinely new**, but the duplication that does exist is **concentrated in the doc's own priority table**, which is why the Section 12 roadmap cannot be adopted as written:

- **(a) vs existing code/specs.** CKF is fully implemented (`crates/thresh-filter/src/ckf.rs`, live spec, archived change, pluggable IMM leaf) yet the doc's P2 row proposes adding it. Native Rust **JPDA and MHT** ship today (`thresh-association/src/{jpda,mht}.rs`, `AssociationStrategy::{Jpda,Mht}`, live specs) yet §5 presents them as "the next tier." Joseph-form updates already exist in KF/EKF (spec-mandated) — the P0 row's remaining gap is only the UKF/CKF sigma-point path. T2T fusion already has the Bar-Shalom optimal rule with known P12 + CI fallback (`t2t.rs`); only the Campo P12 *reconstruction* is missing. The information filter already does additive batched multi-sensor updates (`information.rs::fuse_sensors`); only the predict step and tracker wiring are missing. §8 never mentions the existing, battle-tested **learned-IMM lane** (`imm_adapter.rs`, `python/training/imm_model.py`, Decision-26 bracketing) that all its proposals should extend.
- **(b) vs in-flight work.** The doc's boundary-crossing centerpiece — **ballistic-phase models** (§1.2, doc P2) — is landing *right now* in `orbital-ballistic-filter-models`: KeplerJ2 coast, 7D reentry with online β estimation, phased boost/coast/reentry truth generation, IMM composability via `Reentry7Mapping`. **Full-arc sigma/cubature propagation** (§2) arrives *for free* with the same change (sub-stepped RK4 inside `MotionModel::predict` ⇒ `ukf.predict(keplerj2, dt)` is full-arc propagation, zero filter changes). The change also lays enabler substrate for four more doc items (equinoctial representation, frame tags, closure-based integrator seam, numeric-Jacobian flow).
- **(c) vs planned changes.** The doc's P1 space row (Cowell + force models + adaptive integrators) **is** the planned `orbit-propagation-fidelity` change, minus conscious cuts (EGM96-truncated not EGM2008; no Gauss–Jackson) — and its "evaluate nyx-space" advice violates the standing AGPL rejection. The §4 frame-transform item **is** the planned `astro-time-and-frames` change. Deterministic bias *models* (adjacent to §3.4 bias estimation) are the planned `atmospheric-measurement-propagation` change.

Notable stale-baseline claims (full list in §4 below): the doc believes thresh uses **hifitime** (it doesn't — `Timestamp` is a bare f64), recommends **nyx-space** (rejected, AGPL), pins the baseline at **v0.2.0** (workspace is 0.3.0-dev), omits CKF/JPDA/MHT/learned-IMM, and twice describes a **Zenoh middleware layer that does not exist**. One repo-side inconsistency was discovered during verification: the live `openspec/specs/hifi-orbital/spec.md` still *mandates* nyx-space and needs a superseding amendment.

---

## 2. Duplication matrix

Verdicts: **done** = already implemented · **in-flight** = covered by `orbital-ballistic-filter-models` · **planned** = covered by a queued change · **partial** = a real half exists · **new** = greenfield.

### §1.1 Automotive motion models

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| CTRA | 1.1 | P0 | new | No `ctra` hits anywhere; models/ has only cv/ca/ctrv/ct | `maneuver-motion-models` change; compose via ImmConfig+StateMapping (learned-imm bank frozen) |
| Kinematic bicycle (filter-side) | 1.1 | — | partial | Truth-gen only: `thresh-synth/src/trajectory.rs` `step_bicycle`; no filter model | Defer — CTRA captures most benefit; follow-on if cut-in prediction needed |
| Frenet/curvilinear road-frame | 1.1 | — | new | Zero hits; no map/centerline ingestion in thresh-data | Defer — needs nuScenes map layer; 2 Hz keyframe association doesn't reward it yet |
| Road-network constrained filtering | 1.1 | — | new | No constraint projection/truncation code | Defer with Frenet; VS-IMM half sequences after in-flight change |
| GGIW / random-matrix extent | 1.1 (+5, 11.2) | P3 | new | Extent is pass-through data only (`measurement.rs:74-77`); point-kinematics tracker | Defer — detector already resolves boxes; pays only on raw returns; after PMBM if ever |

### §1.2 Airborne motion models

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| Singer | 1.2 | P0 | new | No exponentially-correlated-accel model | `maneuver-motion-models`; natural first Van Loan user |
| Current Statistical (CS) | 1.2 | P0 | new | Zero hits | Follow-on to Singer; needs one-time state-dependent-Q trait decision |
| Jerk (Mehrotra–Mahapatra) | 1.2 | — | new | ca.rs has white-jerk *noise*, not jerk state | Defer — niche high-g targets the benchmarks don't exercise |
| 3D coordinated turn | 1.2 | P0 | partial | Planar CT with estimated ω exists twice (coordinated_turn.rs, ctrv.rs); no vertical channel | `maneuver-motion-models` (CV-in-z variant first); real ADS-B climbing-turn gap |
| Ballistic-phase models | 1.2 | P2 | **in-flight** | `orbital-ballistic-filter-models`: KeplerJ2, 7D reentry + online β, phased truth, Reentry7Mapping | Landing now; boost *filter* model + boost/coast/reentry IMM preset are recorded follow-ons |
| Hypersonic glide | 1.2 | — | new | Zero hits; lift explicitly out of in-flight scope (Decision 4) | Defer — hard, unvalidatable with current synth; revisit after ballistic gate proves out |

### §1.3 Space propagation

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| Cowell numerical propagation + force models | 1.3 | P1 | **planned** | RK4 two-body+J2+drag exists (`orbital.rs`, moving to thresh-core in-flight); full tier = `orbit-propagation-fidelity` | Fold into `orbit-propagation-fidelity`; replace nyx-space advice (AGPL-rejected, Rust-native); supersede stale hifi-orbital spec |
| Atmospheric density (NRLMSISE-00/JB2008/DTM) | 1.3 | P1 | partial | 29-band exponential table exists; Harris-Priester planned; NRLMSISE nowhere | Harris-Priester per plan; NRLMSISE deferred (needs F10.7/Ap plumbing); consider-states first |
| Equinoctial / GEqOE elements | 1.3 | P2 | partial | Representation in-flight (Decision 2); element-space *filtering* + GEqOE nowhere | `element-space-orbital-filtering` change after in-flight + astro-time-and-frames |
| Gauss variational equations | 1.3 | — | new | Zero hits | Pull in exactly with element-space filtering; no standalone value |
| DSST semi-analytical | 1.3 | — | new | Zero hits | **Demote/skip** — catalog-scale SDA workload doesn't exist |
| Integrators (adaptive DP / Gauss–Jackson / symplectic) | 1.3 | P1 | partial | Fixed-step RK4 in-tree (closure-based refactor in-flight); adaptive DP8/RKF7(8) planned | Adaptive per `orbit-propagation-fidelity`; Gauss–Jackson/symplectic demoted (filter-horizon arcs) |

### §1.4 Continuous-discrete formulation

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| SDE + Van Loan Q | 1.4 | P0 | new | No Van Loan utility; Q conventions verified inconsistent (cv.rs discrete-WNA labeled "spectral density"; ctrv.rs ad hoc) | `van-loan-process-noise` change; **must not interleave** with in-flight gate calibration (task 6.5) — re-baseline once after |
| Continuous-discrete EKF/UKF | 1.4 | — | partial | In-flight models sub-step the mean + Jacobian of full flow; no covariance-ODE facility | Defer — sub-stepped flow captures most practical benefit; refine post-landing if NEES says so |

### §2 Uncertainty propagation

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| State Transition Tensors | 2 | P3 | new | In-flight Decision 3 explicitly rejects analytic STM for numeric Jacobians | Defer — research tier behind GMM splitting |
| Differential Algebra | 2 | P3 | new | Zero hits; nalgebra-only workspace | Defer — publishable-scale effort; only if SDA/conjunction pivot |
| Adaptive GMM splitting | 2 (+10.2) | P2 | new | No mixture-density machinery; IMM mixing is mode-fixed | `element-space-orbital-filtering` companion; share mixture container with any GSF |
| Polynomial Chaos Expansion | 2 | P3 | new | Zero hits incl. python/ | **Demote/skip** — offline studies better done with off-the-shelf Python tooling |
| Full-arc sigma/cubature propagation | 2 | — | **in-flight** | UKF/CKF already propagate points through `model.predict`; KeplerJ2 sub-stepped RK4 = full-arc for free | No action — delivered implicitly by in-flight change |
| NEES/NIS + validated Q | 2 (+9, 11.1) | P0 | new | Zero NEES/NIS hits repo-wide; thresh-eval is MOT-metrics only | `eval-consistency-metrics` change — land with/right behind in-flight change, before gate re-baselining |

### §3 Estimation

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| SR/UD-factorized filters | 3.1 (+11.3) | P0 | new | No factor-propagating filter; mitigation stack instead (Joseph, ensure_psd eigen-repair, LU solves, Cholesky-with-jitter) | `square-root-filter-forms` change right after in-flight lands (ECI/7D-reentry conditioning stress); retires cov.rs band-aid |
| Joseph-form update | 3.1 | P0 | partial | KF/EKF done + spec-mandated; UKF/CKF use plain form + PSD repair | Fold into SR change (SR subsumes it); do not do both |
| CKF | 3.2 | P2 | **done** | `ckf.rs`, live spec, archived 2026-05-18 change, pluggable IMM leaf | Already done — drop from any roadmap |
| Iterated EKF / IPLF | 3.2 | P2 | new | Single-pass EKF update only | Ride along `robust-adaptive-updates` as LeafFilter kinds; pays off at close-range radar/angles-only |
| Gaussian Sum Filters | 3.2 | — | new | IMM mixture ≠ GSF (fixed count, collapse per cycle) | Defer — share mixture container with GMM splitting when that lands |
| RBPF | 3.3 | — | new | Explicitly out of scope in archived imm-filter change; β handled by 7D augmentation instead | Defer — augmentation + IMM already cover cited use cases; log-β is the recorded fallback |
| Student's-t / VB-adaptive | 3.3 | P2 | new | Fixed Gaussian R everywhere | `robust-adaptive-updates` after NEES/NIS exists (validate adaptivity) |
| Huber / M-estimation | 3.3 | — | new | Gating rejects, nothing downweights | Bundle into `robust-adaptive-updates` (~50 lines) |
| Schmidt–Kalman consider filter | 3.4 | — | new | Zero hits; in-flight estimates β rather than considers it | Defer until orbit-propagation-fidelity introduces density/Cd params; co-design with SR refactor |
| Online sensor bias estimation | 3.4 | P1 | partial | Deterministic known-transform registration only (`sensor.rs`, `registration.rs`); zero estimated-bias states | Fold into / fast-follow `atmospheric-measurement-propagation` (estimate residual after deterministic models) |
| Lie-group / Invariant EKF | 3.5 | — | new | No attitude states anywhere; quaternions are known inputs | **Demote/skip** — point targets in vector spaces; revisit only with extended-object heading or platform pose |

### §4 Maneuvers, mode switching, boundary crossing

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| VS-IMM / LMS / EMA | 4 | P2 | new | Fixed bank + static TPM; model-set selection scoped out of archived imm change | `imm-mode-set-adaptation` after in-flight archives; **learned-imm ONNX contract pins the 4-model bank** — needs non-learned path or retraining |
| Jump-Markov MM-RFS | 4 | — | partial | IMM (the standard JMLS filter) fully implemented; MM-PMBM/GLMB absent | Defer — PMBM follow-on only |
| GLR/CUSUM maneuver detection | 4 (+11.4) | — | new | Zero hits; out of in-flight scope | Small change sharing the innovation/S hook built for NEES/NIS |
| Rigorous frame transforms + covariance rotation | 4 | P2 | **planned** | Partial in-tree (eci.rs GMST-only; R·P·Rᵀ on ENU re-center); full chain = `astro-time-and-frames` | Fold into `astro-time-and-frames`; confirm full covariance rotation is explicit in scope |
| Launch→orbit / reentry handoff transforms | 4 | P2 | partial | Reentry7Mapping + deterministic per-head phase switch in-flight; UT-of-state handoff + GMM transition absent | Two changes deep — after element-space filtering exists |
| Sensor-region handover (T2T + stitching) | 4 | P2 | partial | T2T fusion rich (`t2t.rs`); no smoother, no segment stitching | Fusion half done; stitching = `track-segment-association` after RTS |
| Constrained state estimation | 4 | — | new | No projection/truncation; β-floor clamp in-flight is a first primitive | Defer — low urgency vs P0/P1 items |

### §5 Data association & MOT architecture

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| JPDA / JIPDA | 5 | — | partial | **JPDA done** (native jpda.rs + spec + tracker wiring); JIPDA existence probability absent | JIPDA vs PMBM is one explicit decision — do not build both |
| PMBM / TPMBM | 5 (+10.3) | P1 | new | Zero RFS hits anywhere | `pmbm-tracker-variant` after Murty + after in-flight tracker-head churn; **demoted P1→P2** (multi-change effort vs three active lanes) |
| GLMB / LMB | 5 | — | new | Zero hits | **Demote/skip** — decide PMBM-over-GLMB once in the proposal |
| PHD / CPHD | 5 | — | partial | Stone Soup bridge only (`thresh-bridge/src/phd.rs`), dev/test-gated | Defer — native GM-PHD only as future PMBM birth/clutter front-end |
| BP/SPA association | 5 | P3 | new | Zero hits; drops into existing `JpdaResult` marginal interface | **Promoted P3→P2** — cheap integration, dense nuScenes win, honestly resolves jpda spec's joint-event language |
| Network-flow MHT / GNN association | 5 | — | new | mht.rs is Reid-style enumerate-then-truncate; no min-cost-flow; no association GNN in python/ | Defer — best future use is offline eval/auto-labeling |
| GGIW-PMBM | 5 | P3 | new | Double dependency (PMBM + GGIW), neither exists | Defer — only if raw-return tracking replaces detector-first pipeline |
| Murty's k-best + hypothesis management | 5 | P1 | new | Recorded open question in archived jpda-mht design (chose sorted truncation); mht.rs blowup warning documents symptom | `murty-k-best-assignment` — cleanest standalone change; unblocks PMBM, fixes shipped MHT |

### §6 Multi-sensor & distributed fusion

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| OOSM retrodiction | 6 (+10.4) | P1 | new | `streaming.rs` TemporalBinner time-misassigns or drops late data — the exact warned failure | `oosm-retrodiction` change; co-design fixed-lag augmentation with RTS smoother |
| ICI / Ellipsoidal Intersection | 6 | — | new | Classic CI only (`covariance_intersection.rs`) | Small quality upgrade when multi-site T2T is exercised; drop-in FusionMode |
| Chernoff / GCI density fusion | 6 | — | new | No GM/RFS density types | Defer — strictly after PMBM |
| Consensus / diffusion filters | 6 | — | new | No comms-graph or middleware layer (Zenoh claim false) | **Demote/skip** — single-process reality; hub case covered by FederatedFusionManager |
| Bar-Shalom–Campo P12 reconstruction | 6 | — | partial | `fuse_optimal` + OptimalWithCrossCovariance + CI fallback exist; nothing *computes* P12 | Small change — Campo recursion or document CI-default; currently a dead code path |
| Asynchronous multi-rate fusion | 6 | — | partial | Per-arrival sequential updates + arbitrary-dt models exist; no general Van Loan; t2t alignment hand-builds F,Q | Finish via `van-loan-process-noise` + wire extrapolate_track to MotionModel |

### §7 Smoothing & batch

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| RTS / fixed-lag smoothers | 7 (+10.3) | — (unlisted!) | new | No smoother of any kind workspace-wide | `rts-smoother` change, **pulled forward to P1** — enabler for TSA + better training labels (Track A/B) + eval trajectories |
| Factor-graph / sliding-window MAP | 7 | P3 | new | No batch NLS infra; nalgebra-only | Defer — second estimation backbone; only if point solutions (OOSM, RTS) prove insufficient |
| Initial Orbit Determination | 7 | — | new | Zero hits; in-flight assumes initialized states | Defer as fast-follow to `astro-time-and-frames` (meaningless without frame/time discipline) |

### §8 Learning-augmented estimation

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| KalmanNet / learned gain/Q/R | 8 | P3 | new | No learned-gain code; learned-IMM seam is the template (imm_adapter.rs) | Defer — Track A/B evidence says data quality gates learned components; frame as learned-imm-seam extension |
| Learned motion models / trajectory predictor | 8 | — | partial | `learned-predictor` designed-but-deferred with written spec (flight-data change); GP models nowhere | Spec-complete the deferred predictor when OpenSky data unblocks; GP variant deferred |
| E2E transformer MOT adversary | 8 | — | new | Detection-level bracketing (Decision 26) already practices the idea | Defer — heavy engineering for a comparison number; revisit for publication needs |
| Learned birth/clutter intensity | 8 | P3 | new | Constant scalar clutter_density; no Poisson birth (no PMBM) | Defer — strictly after PMBM |

### §9 Evaluation & consistency

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| NEES/NIS chi-squared tests | 9 | P0 | new | (see §2 row) thresh-eval is MOT-only; filters compute but don't expose innovation/S | `eval-consistency-metrics` — highest value/cost item in the whole doc |
| GOSPA / trajectory-GOSPA | 9 | P0 | new | Zero hits; matching.rs is threshold-MOT only | Bundle into `eval-consistency-metrics`; right target function for future PMBM |
| PCRLB | 9 | — | new | Zero hits (team already does hand CRLB-style bounds, task 7.9) | Defer — useful offline diagnostic, not blocking |

### §10 Sporadic data & gaps

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| Sinopoli λ_c diagnostic | 10.1 | P3 | new | Zero hits; i.i.d. Bernoulli dropout substrate exists in synth | Defer — cheap thresh-eval add-on alongside consistency suite |
| Gilbert–Elliott bursty gaps | 10.1 | — | new | All dropout i.i.d. (`measurement_gen.rs`) | Fold into `gap-aware-tracking` as its test harness (few dozen lines in synth) |
| Deterministic gap prediction Pd(t) | 10.1 | P1 | partial | Visibility math exists (elevation masks orbital.rs:542, Albersheim Pd, EO/IR horizon); tracker side constant-Pd + M-of-N | `gap-aware-tracking` — mostly plumbing; prevents M-of-N killing orbital tracks in predictable gaps |
| GMM splitting for coast gaps | 10.2 | P2 | new | Consumer of §2 GMM infra | Rides `element-space-orbital-filtering` |
| Gap-duration Q growth / mode diffusion | 10.2 | P1 | partial | Exact dt-scaled Q + per-cycle re-predict exist; no envelope inflation or coast-driven diffusion | `gap-aware-tracking`, driven by existing `coast_count`; keep in analytic ImmFilter so learned variant inherits |
| Ps/Pd existence (JIPDA-style) | 10.2 | P1 | partial | Pd already in JPDA weights; lifecycle deterministic M-of-N; no Ps, no existence prob | `gap-aware-tracking` (Pd side) + PMBM/JIPDA decision (existence side) |
| Track-before-detect | 10.2 | P3 | new | Zero hits; radar_scene.rs provides the sub-threshold data source | **Demote/defer** — niche low-SNR differentiator; after mainstream gap work |
| Track segment association (TSA) | 10.3 | P2 | new | Only offline ID-keyed gap-splitting (stitching.py / trajectory_ingest.rs); no smoother for backward half | `track-segment-association` after `rts-smoother` |
| TPMBM native gap handling | 10.3 | — | new | Downstream of PMBM | Defer — JIPDA existence + TSA covers the need years sooner |
| Negative information (dwell misses) | 10.4 | P1 | new | Misses only bump coast_count; no dwell model, no existence prob | Last brick of the Ps/Pd wall — after `gap-aware-tracking` + existence modeling |

### §11 High-frequency data

| Technique | Doc § | Doc pri | Verdict | Evidence | Disposition |
|---|---|---|---|---|---|
| Time-correlated noise handling | 11.1 | P2 | new | White-R assumed everywhere | Defer until high-rate benchmarks exist; pair with NIS whiteness test |
| NIS whiteness (Ljung–Box) | 11.1 | P2 | new | Rides the NEES/NIS suite | Extension of `eval-consistency-metrics` |
| Tracklet / equivalent-measurement compression | 11.2 | P1 | new | TemporalBinner batches but every measurement costs full update | `high-rate-measurement-path` — **demoted P1→P2**, no benchmark drives it yet; bundle with info-form predict |
| Information-form batched updates | 11.2 | P1 | partial | fuse_sensors already additive-batches; missing predict step + tracker wiring | Bundle with tracklet compression (one change) |
| Extended-object measurement rate | 11.2 | P3 | new | Point-target assumption throughout | Defer with GGIW |
| SR/UD forms at rate | 11.3 | P0 | (dup §3.1) | — | See `square-root-filter-forms` |
| Auction + sparse gated matrices | 11.3 | P2 | new | Explicitly descoped by archived performance-optimization change pending profiling | Defer — wait for profiling evidence, per precedent |
| Rate-adaptive processing | 11.3 | — | partial | TemporalBinner is half; no tracklet interface layer | Design constraint on the tracklet change, not separate work |
| Stiff integrators | 11.3 | — | new | Explicit RK only, in-flight sub-stepping is the deliberate answer | Defer — only with profiling evidence, after orbit-propagation-fidelity |
| GLR/CUSUM at rate | 11.4 | — | (dup §4) | — | See §4 row |
| Doppler/range-rate fusion | 11.4 | — | partial | OTHR tracker fuses doppler jointly; `range_rate` carried in Measurement but unused by mainline radar path | Small P2 quick win — wire existing field into radar update; conditioning refinements wait for SR |

---

## 3. Corrected roadmap

Priorities are **corrected assignments**; the doc's original is noted where different. Sequencing anchors: `orbital-ballistic-filter-models` (in-flight, archives first), then `astro-time-and-frames` → `orbit-propagation-fidelity` → `atmospheric-measurement-propagation` (planned). Standing decisions respected: no AGPL nyx-space, Rust-native preference, no GPU in CI, learned-imm 4-model bank frozen.

| Pri | Item | Packaging | Rationale |
|---|---|---|---|
| P0 | NEES/NIS + GOSPA + innovation/S exposure API | new change: `eval-consistency-metrics` | In-flight change adds unfalsifiable Q params (σ_accel, σ_β) and tightens gates — this is the acceptance instrument; pure Rust, no conflicts; land before gate re-baselining |
| P0 | SR-UKF/SR-CKF + Joseph-equivalent sigma-point update | new change: `square-root-filter-forms` | ECI-scale + mixed-unit 7D reentry states stress conditioning hardest; sequence right after in-flight lands; retires cov.rs eigen-repair; KF/EKF Joseph already done |
| P0 | Van Loan Q utility + CV/CTRV/CT Q retrofit | new change: `van-loan-process-noise` | Verified convention mismatch in-tree; shifts every calibrated baseline so must not interleave with in-flight gate calibration (task 6.5) — land after archive, re-baseline once |
| P1 (doc P0) | Singer + CTRA + 3D CT(ω) motion models (CS follow-on) | new change: `maneuver-motion-models` | Airborne (ADS-B climbing turns) + automotive (accelerating turns) gaps; compose via ImmConfig+StateMapping only — learned-imm bank frozen; demoted below the falsifiability/numerics tier that validates them |
| P1 | Murty's k-best assignment | new change: `murty-k-best-assignment` | Recorded open question in archived jpda-mht design; fixes shipped MHT's expand-then-prune blowup; prerequisite for PMBM; builds on hungarian.rs |
| P1 | Gap-aware Pd(t)/Ps + gap-duration Q growth + Gilbert–Elliott synth dropout | new change: `gap-aware-tracking` | Visibility math already in synth — mostly plumbing; prevents M-of-N deleting orbital tracks in predictable station gaps; directly serves the benchmark scenarios being tightened now |
| P1 (doc unlisted) | RTS + fixed-lag smoother | new change: `rts-smoother` | Absent workspace-wide yet missing from the doc's own roadmap table; enabler for TSA, better training labels for the blocked Track A/B lane, eval-grade trajectories; cheap and standard |
| P1 | OOSM retrodiction (Bl1/Al1 or fixed-lag augmentation) | new change: `oosm-retrodiction` | streaming.rs currently time-misassigns or drops late data; multi-sensor latencies (radar/EO-IR/ADS-B) are the aerospace reality; co-design augmentation with the smoother |
| P1 | Online sensor bias/registration estimation | fold into `atmospheric-measurement-propagation` (or fast-follow) | Registration today is static known transforms; estimate the residual the planned deterministic refraction/TEC models leave behind — natural pairing |
| P1 | Numerical propagation force models + adaptive integrators | fold into `orbit-propagation-fidelity` | Already the planned change; replace doc's nyx-space row (AGPL-rejected → Rust-native); also supersede the stale hifi-orbital spec that still mandates nyx-space |
| P2 | Frame transforms with full covariance rotation | fold into `astro-time-and-frames` | Already planned; confirm covariance rotation through the whole chain is explicit in scope when the change opens |
| P2 (doc P1) | PMBM (point-target, GM, Murty-based) | new change: `pmbm-tracker-variant` | Still the biggest architectural upgrade, but multi-change effort competing with three active lanes; sequence after Murty + in-flight tracker-head churn; decide PMBM-vs-JIPDA-vs-GLMB once — skip GLMB and standalone JIPDA if committed |
| P2 | Equinoctial/GEqOE element-space filtering + GVE + adaptive GMM splitting | new change: `element-space-orbital-filtering` | Doc's space-accuracy centerpiece is right; in-flight delivers only the representation; needs astro-time-and-frames first; share mixture container with any future GSF |
| P2 | VS-IMM / boost-coast-reentry IMM preset + handoff | new change: `imm-mode-set-adaptation` | The in-flight change ships Reentry7Mapping and defers the preset — natural first use case; learned-imm contract requires a non-learned path or classifier retraining |
| P2 (doc P3) | BP/SPA association | new change: `bp-spa-association` | Promoted: drops into the existing JpdaResult marginal interface (only the probability computation swaps); dense-scene nuScenes win; resolves jpda spec's joint-event gap honestly |
| P2 | Student-t/VB adaptive R + Huber + IPLF leaf | new change: `robust-adaptive-updates` | Scene-dependent detector noise is the hybrid architecture's known pain; sequence after NEES/NIS so adaptivity is validated, not guessed |
| P2 | Track segment association (two-sided stitch) | new change: `track-segment-association` | ADS-B/OpenSky fragmentation is a live pain point; blocked on `rts-smoother`; one-sided variant possible today at reduced selectivity |
| P2 (doc P1) | Tracklet compression + information-form predict + range-rate wiring | new change: `high-rate-measurement-path` | Demoted: info-form batching half-exists (fuse_sensors) and range_rate is already carried but unused — cheap completions; full tracklet layer waits for a high-rate benchmark to drive it |
| P3 | IOD (Gauss/Gooding, Herrick-Gibbs) | defer — fast-follow to `astro-time-and-frames` | Real space-lane gap (birthing tracks from uncued sensors) but meaningless while Timestamp is a bare f64 with no frame discipline |
| P3 | Demoted research tier: STT/DA, PCE, DSST, hypersonic glide, TBD, factor-graph backend, consensus/Chernoff fusion, GGIW(+GGIW-PMBM), GLMB, Lie-group IEKF, RBPF, Schmidt-Kalman, KalmanNet/learned birth-clutter/E2E-MOT adversary, Frenet/road-network, auction, stiff integrators, Jerk, filter-side bicycle | defer — see rationale | Each lacks a current consumer, is blocked on absent infrastructure (PMBM, GMM, maps, middleware, attitude states), was descoped by precedent (auction), or is gated on data/pivots (learned items per Track A/B evidence; DSST/DA on an SDA workload that doesn't exist) |

**Explicit demotions from the doc's table:** CKF (P2 → already done); PMBM (P1 → P2, sequencing + effort); tracklet/info-form row (P1 → P2, no driving benchmark); nyx-space evaluation (P1 → removed, AGPL); "boost–coast–reentry model chain" P2 row (→ largely in-flight already; remainder is the IMM-preset follow-on); auction (P2 → wait-for-profiling per archived precedent); GGIW rows, STT/DA/PCE, BP-factor-graph-backend half, KalmanNet/learned-intensities, TBD, λ_c (all P3 → confirmed deferred); DSST, hypersonic glide, Frenet/road-network, GLMB, consensus/Chernoff, Lie-group IEKF, RBPF, Jerk, bicycle, stiff integrators (unlisted → explicitly skipped for now). **Promotions:** RTS smoother (unlisted → P1), BP/SPA (P3 → P2), GLR/CUSUM (unlisted → small P2 rider on the consistency hook), Gilbert–Elliott (unlisted → folded into gap-aware-tracking).

---

## 4. Stale claims in the doc

1. **hifitime**: lines 40 and 104 claim thresh uses hifitime ("already ecosystem-adjacent to hifitime, which thresh uses"; "you already have TAI/GPS via hifitime"). False — no Cargo.toml depends on hifitime; `thresh_core::time::Timestamp` is a bare `pub struct Timestamp(pub f64)`. hifitime is only *proposed* in the planned `astro-time-and-frames` change.
2. **nyx-space**: recommended in §1.3 and the P1 roadmap row ("evaluate nyx-space"). Rejected on AGPL licensing grounds (standing decision); direction is Rust-native in-house (`orbit-propagation-fidelity`). Repo-side corollary: the live `openspec/specs/hifi-orbital/spec.md` still mandates nyx-space and needs a superseding amendment.
3. **Baseline version**: "thresh v0.2.0, develop" — workspace is at 0.3.0-dev.
4. **CKF**: baseline omits it and §3.2 presents it as "essentially free to add"; the P2 row proposes adding it. CKF has been implemented since 2026-05-18 (`ckf.rs`, live spec, archived change, pluggable IMM leaf).
5. **JPDA/MHT**: baseline omits them and §5 presents them as "the next tier." Both ship natively (`jpda.rs`, `mht.rs`, `AssociationStrategy::{Jpda,Mht}`, live specs, archived 2026-04-25 change, plus Stone Soup bridges). Only JIPDA existence probability is missing.
6. **Joseph form**: §3.1 "everywhere a plain form exists today" overstates — KF/EKF already use Joseph form per the state-estimation spec; the remaining gap is only the UKF/CKF sigma-point update.
7. **Zenoh middleware**: §6 and §11.3 describe consensus iterations and rate-adaptive processing mapping onto "your middleware layer" / "a Zenoh topology." No middleware or pub/sub layer exists anywhere in the workspace.
8. **Information filter**: §11.2 "extend it to batched updates" understates — `InformationState::fuse_sensors` already accumulates arbitrarily many contributions additively with one deferred inversion; what is missing is the predict step and tracker wiring.
9. **T2T cross-covariance**: §6 implies thresh lacks it — `fuse_optimal` (Bar-Shalom with known P12), `FusionMode::OptimalWithCrossCovariance`, a P12 store, and CI fallback all exist; only the Campo P12 *reconstruction* is missing (currently a dead code path in practice).
10. **Learned-IMM omission**: §8 presents learning-augmented estimation as entirely prospective; a GRU mode classifier is trained/exported (`python/training/imm_model.py`), integrated behind `learned-imm` (`imm_adapter.rs` with analytic fallback), and A/B-bracketed (Decision 26) — §8's items should extend this seam, not start greenfield.
11. **"Level-1/Level-2 radar fidelity stack"** (§10.2, P3 row): invented naming — no such taxonomy exists; the underlying capabilities do (`radar_equation.rs`, `radar_scene.rs` per the radar-scene-sim spec).
12. **P1 OOSM row's "sensor bias/registration states"**: overstates the baseline — `registration.rs` applies static known transforms only; estimated bias states are entirely greenfield.

---

## 5. Suggested next OpenSpec changes

In dependency order (after `orbital-ballistic-filter-models` archives):

1. **`eval-consistency-metrics`** (P0) — NEES/NIS + GOSPA in thresh-eval, innovation/S exposure in thresh-filter (UKF/CKF accessors to match KF/EKF). Small, no conflicts, and it is the acceptance instrument for the orbital gate re-baselining. Optionally include the Ljung–Box whiteness test and the Sinopoli λ_c diagnostic as stretch tasks.
2. **`square-root-filter-forms`** (P0) — SR-UKF/SR-CKF propagating factors, subsuming the sigma-point Joseph gap and retiring the `cov.rs` eigen-repair; conditioning tests against the new KeplerJ2/BallisticReentry models. Co-design covariance storage with future Schmidt–Kalman partitioning in mind.
3. **`van-loan-process-noise`** (P0) — `van_loan(A, G, q̃, dt)` utility + retrofit CV/CTRV/CT Q conventions; explicitly scheduled outside the gate-calibration window with one coordinated baseline re-run.
4. **`maneuver-motion-models`** (P1) — Singer, CTRA, 3D CT (CV-in-z); CS as a follow-on task gated on the state-dependent-Q trait decision; all composed via public ImmConfig + StateMapping (default bank untouched).
5. **`murty-k-best-assignment`** (P1) — Murty over hungarian.rs; wire into MHT hypothesis generation; closes the archived design's open question. Also the moment to re-scope the jpda/mht spec-vs-implementation gaps found during this analysis (joint-event language; missing new-track/false-alarm branches in `EnumCtx::enumerate`).
6. **`gap-aware-tracking`** (P1) — visibility-predicted Pd(t) provider from synth, per-track Pd in JPDA, coast-driven Q growth/mode diffusion, Gilbert–Elliott bursty dropout in synth as the harness.
7. **`rts-smoother`** (P1) — RTS + fixed-lag over stored predict/update pairs (small filter-API extension); consumers: training-label quality (Track A/B), TSA, eval trajectories.
8. **`oosm-retrodiction`** (P1) — Bl1/Al1 or fixed-lag augmentation; fix TemporalBinner's silent time-misassignment; co-designed with the smoother's state-augmentation plumbing.

Then the P2 tier as separate proposals when their dependencies land: `pmbm-tracker-variant` (decision doc first: PMBM vs JIPDA vs GLMB), `element-space-orbital-filtering`, `imm-mode-set-adaptation`, `bp-spa-association`, `robust-adaptive-updates`, `track-segment-association`, `high-rate-measurement-path`.

Repo housekeeping surfaced by this analysis: supersede or amend **`openspec/specs/hifi-orbital/spec.md`** (still mandates AGPL-rejected nyx-space + Orekit-PyO3 fallback, conflicting with the Rust-native direction).
