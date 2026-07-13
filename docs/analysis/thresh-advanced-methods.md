# Advanced Mathematical Methods for `thresh`

**Scope:** Candidate techniques for multi-sensor fusion and multi-object tracking across automotive, airborne, and space domains — covering slow movers (pedestrians, loitering UAS, GEO objects), fast movers (fighters, hypersonics, LEO objects), and targets that **cross domain boundaries** (launch → orbit, reentry, air-to-ground handoff, sensor-region handover, road-network entry/exit).

**Baseline (thresh v0.2.0, develop):** KF/EKF/UKF; CV, CA, CTRV, Coordinated Turn motion models; IMM; Hungarian assignment, Mahalanobis gating, 2D/3D IoU, cascaded association; centralized fusion, information filter, covariance intersection, T2T fusion; M-of-N track lifecycle; ONNX transformer detection; SGP4 orbital adapter; MOTA/MOTP/IDF1/HOTA/AMOTA metrics.

Each section below identifies what exists, what's missing, and priority. Prioritization summary is at the end.

---

## 1. Motion & Propagation Models by Domain

Propagation fidelity is the single biggest driver of track accuracy during coast (no-measurement) intervals — which is exactly when boundary-crossing targets are hardest (sensor handover gaps, terrain masking, eclipse, beam revisit).

### 1.1 Automotive (slow, constrained)

| Technique | Why | Notes |
|---|---|---|
| **CTRA** (Constant Turn Rate & Acceleration) | Fills the gap between CTRV (no accel) and CA (no turn). Best single model for on-road vehicles | Nonlinear; UKF/CKF-friendly. Closed-form transition exists |
| **Kinematic bicycle / single-track model** | Physically correct low-speed vehicle motion; heading tied to steering geometry | Needed if you fuse with ego-vehicle or predict cut-ins; wheelbase as a per-class parameter |
| **Curvilinear (Frenet–Serret) road-frame models** | Propagate along-track/cross-track relative to lane centerline instead of Cartesian. Dramatically better long-horizon prediction on curved roads | Requires map/centerline input; boundary crossing = lane change / road exit handled as mode switch |
| **Constrained filtering on road networks** | Enforce that state lies on/near a road graph: (a) projection onto constraint manifold, (b) pseudo-measurements with small variance, (c) PDF truncation | Directly analogous to VS-IMM for ground targets in the GMTI literature (Kirubarajan & Bar-Shalom) — model set adapts at intersections |
| **Extended-object models (GGIW / random matrix)** | At automotive ranges a car is many lidar/radar returns, not a point. Gamma-Gaussian-Inverse-Wishart models kinematics + extent + measurement rate jointly | Koch's random-matrix model; Granström's GGIW-PHD/PMBM. Big accuracy win for nuScenes-class data |

### 1.2 Airborne (slow ↔ fast, maneuvering)

| Technique | Why | Notes |
|---|---|---|
| **Singer model** | Classic exponentially-correlated acceleration; the standard "maneuvering aircraft" model. One tunable maneuver time constant τ and acceleration variance | Cheap; belongs in the same family as CV/CA in `thresh-filter` |
| **Current Statistical (CS) model** (Zhou–Kumar) | Singer + adaptive, non-zero-mean acceleration centered on current estimate. Better for sustained maneuvers | Widely used in fire-control trackers |
| **Jerk model** (Mehrotra–Mahapatra) | Third-derivative state for high-g agile targets (missiles, aerobatic UAS) | Pairs naturally with IMM: {CV, CS, Jerk} |
| **3D coordinated turn with unknown turn rate** | Current CT is planar. Augment state with turn rate ω (and optionally turn-plane orientation) and estimate it | Nonlinear CT-UKF; standard for fighters. Li & Jilkov Survey Part I is the canonical taxonomy |
| **Ballistic-phase models** | Boost (thrusting, mass-varying), coast (Keplerian + J2), reentry (gravity + Allen–Eggers-style drag, ballistic coefficient β as augmented state) | This *is* the boundary-crossing problem for TBMs: mode-conditioned dynamics inside IMM, with β and drag estimated online |
| **Hypersonic glide models** | Equilibrium-glide + bank-reversal maneuver models; acceleration bounded by L/D envelope. Active research area (AMR-type adaptive models) | Relevant to Code Metal-adjacent mission space; hard because maneuvers are deliberate and non-Markovian |

### 1.3 Space (slow in LEO terms ≈ 7.5 km/s; GEO nearly static in ECEF)

| Technique | Why | Notes |
|---|---|---|
| **Numerical propagation (Cowell) with force models** | SGP4 is a TLE-consistency tool (~1 km class error), not a precision propagator. For track accuracy you need: spherical-harmonic gravity (EGM2008 to selectable degree/order), atmospheric drag, SRP with eclipse/shadow, luni-solar third-body | Rust: consider building on `nyx-space` (already ecosystem-adjacent to `hifitime`, which thresh uses) rather than reimplementing |
| **Atmospheric density models** | NRLMSISE-00 (general), JB2008 (storm-time, uses solar indices), DTM-2020. Density is the dominant LEO error source; 10–15% density error dominates the covariance below ~600 km | Treat density/Cd via consider states (§3.4) |
| **Element-set choice: equinoctial / GEqOE** | Cartesian ECI covariances go banana-shaped (non-Gaussian) within a fraction of an orbit. Equinoctial elements avoid singularities at e=0, i=0; Generalized Equinoctial Orbital Elements (GEqOE) absorb J2 so uncertainty stays near-Gaussian far longer | Directly improves how long a plain Gaussian remains valid → fewer GMM components needed (§3.3) |
| **Gauss variational equations** | Propagate element-space dynamics under perturbing accelerations; natural for element-space filters | |
| **Semi-analytical propagation (DSST)** | Mean-element propagation with short-periodic recovery; orders of magnitude faster than Cowell for catalog-scale problems | Only if thresh targets many-object SDA workloads |
| **Integrators** | Dormand–Prince RK5(4)/RK8(7) adaptive for general use; **Gauss–Jackson 8th-order** multistep is the space-surveillance standard for orbit propagation; symplectic integrators if long-arc energy behavior matters | Deterministic, testable — good fit for thresh's regression-test culture |

### 1.4 Cross-cutting: continuous-discrete formulation

- **Model dynamics as SDEs** (dx = f(x)dt + G dw) and derive the discrete-time process noise **exactly via the Van Loan method** rather than ad-hoc Q matrices. This makes Q consistent across variable dt — essential for asynchronous multi-sensor operation and for objects observed at wildly different revisit rates (automotive 10–20 Hz vs. space minutes–hours).
- **Continuous-discrete EKF/UKF**: integrate mean and covariance ODEs (or sigma points) between measurements instead of one-step discretization. Matters whenever dt × dynamics-rate is large — reentry, perigee passage, hard aircraft maneuvers between radar dwells.

---

## 2. Uncertainty Propagation for Accuracy

This is the "propagation models for accuracy" core. Linear covariance propagation (what EKF does) fails exactly where thresh's use cases live: long coast arcs, orbital nonlinearity, reentry.

| Technique | Idea | When it wins |
|---|---|---|
| **State Transition Tensors (STT)** | Keep 2nd/3rd-order terms of the Taylor expansion of the flow map, propagate moments through them | Orbit propagation over multiple revs; reentry. Analytic, no sampling |
| **Differential Algebra (DA) propagation** | Automatic high-order Taylor polynomial of the flow w.r.t. initial state (Berz/Armellin school). One integration yields a polynomial map you can evaluate for any initial deviation | Conjunction analysis, long coast, maneuver detection via polynomial residuals. Rust implementation is a real (publishable) engineering effort |
| **Adaptive Gaussian Mixture splitting** | Split a Gaussian along the direction of maximum nonlinearity when a linearity/entropy test fails (DeMars et al.); propagate each component with EKF/UKF; merge later | The workhorse for non-Gaussian orbital uncertainty; combines naturally with GEqOE (fewer splits) and with DA/STT (cheap per-component propagation) |
| **Polynomial Chaos Expansion (PCE)** | Spectral expansion of the state PDF in orthogonal polynomials of the input uncertainty | Highest accuracy for smooth dynamics; cost grows fast with dimension — best for offline covariance-realism studies |
| **Unscented/cubature propagation of full arcs** | Propagate sigma/cubature points through the *nonlinear integrator* over the whole coast arc, not step-by-step linearization | Cheap first upgrade; already 80% of the win over EKF covariance for moderate arcs |
| **Covariance realism machinery** | Whatever propagator you use, validated Q: energy-scaling process noise for drag, Cramér–von Mises / Mahalanobis-distance consistency tests (NEES/NIS) in `thresh-eval` | Without consistency testing, "accurate propagation" is unfalsifiable. Add NEES/NIS to the metrics crate |

**Recommendation:** the pragmatic stack for thresh is **GEqOE state representation + adaptive GMM splitting + UKF/CKF component propagation**, with STT/DA as a later accuracy tier. This mirrors current SSA practice and each piece is independently testable.

---

## 3. Estimation: Beyond KF/EKF/UKF

### 3.1 Numerically robust forms (do these first)
- **Square-root / UD-factorized filters** (SR-KF, SR-UKF, Bierman UD): propagate Cholesky/UD factors instead of P. Guarantees positive-definiteness, doubles effective precision. For a safety-critical Rust codebase this is table stakes — and it maps cleanly to your DO-178C instincts (bounded, deterministic numerics).
- **Joseph-form covariance update** everywhere a plain form exists today.

### 3.2 Better Gaussian filters
- **Cubature Kalman Filter (CKF)**: spherical-radial cubature; more numerically stable than UKF in high dimensions (no negative weights), essentially free to add next to the UKF.
- **Iterated EKF / Iterated Posterior Linearization Filter (IPLF)**: re-linearize about the posterior. Large wins for very nonlinear measurement models — monostatic radar r/az/el at close range, angles-only (EO/IR) tracking, TDOA.
- **Gaussian Sum Filters**: run a bank of Gaussians for multimodal posteriors (angles-only initial orbit determination, ghost tracks from multipath).

### 3.3 Non-Gaussian / robust
- **Rao-Blackwellized Particle Filter**: particles only over the nonlinear substate (e.g., ballistic coefficient, turn rate, mode), Kalman filters over the conditionally-linear rest. Practical PF variant for tracking; plain bootstrap PFs rarely earn their cost here.
- **Student's-t and variational-Bayes adaptive filters**: heavy-tailed measurement models absorb outliers (multipath, glint, mislabeled detections from the transformer front-end); VB methods jointly estimate unknown/varying R. Especially valuable when the ONNX detector's error statistics drift by scene.
- **Huber / M-estimation updates**: cheap robustification of the innovation step.

### 3.4 Consider states & parameter estimation
- **Schmidt–Kalman ("consider") filter**: account for uncertainty in nuisance parameters (sensor biases, drag coefficient, atmospheric density scale) without estimating them — keeps covariance honest.
- **Augmented-state bias estimation / sensor registration**: estimate per-sensor range/angle/time biases online. In multi-sensor fusion, uncorrected registration errors masquerade as maneuvers and kill T2T association. High practical ROI.

### 3.5 Geometry-aware filtering
- **Error-state / indirect KF and Invariant EKF (IEKF) on Lie groups** (SO(3), SE(3), SE₂(3)): when target attitude or a sensor platform's pose enters the problem (gimballed EO/IR, extended objects with heading), filtering on the manifold avoids linearization pathologies and gives provable consistency properties. The equivariant-filter literature is the current frontier.

---

## 4. Maneuvers, Mode Switching & Boundary Crossing

Thresh has IMM. The boundary-crossing use cases push toward the following:

- **Variable-Structure IMM (VS-IMM)**: the *active model set* changes with context — road network topology (automotive), flight envelope (airborne), mission phase (boost/coast/reentry). This is the canonical answer to "things that cross boundaries": the boundary event triggers a model-set switch, not just a probability shift. Likely-Model-Set (LMS) and Expected-Mode Augmentation are the standard variants.
- **Jump-Markov (nonlinear) systems**: formalize mode + state jointly; PMBM and GLMB both have multiple-model extensions (a 2025 multi-model trajectory-PMBM shows exactly this combination for maneuvering targets).
- **Maneuver detection as hypothesis testing**: GLR/CUSUM on innovation sequences to trigger model-set adaptation or covariance inflation; complements IMM rather than replacing it.
- **Domain/frame handoff mathematics**:
  - Rigorous **ENU ↔ ECEF ↔ ECI (GCRF/ITRF)** transformations with full covariance rotation, using IERS conventions and proper time scales (you already have TAI/GPS via hifitime — extend to UT1/polar motion for ITRF↔GCRF at accuracy).
  - **Launch → orbit**: boost-phase filter (thrust-acceleration state) hands off to element-space orbital filter; the handoff is a nonlinear state+covariance transform (unscented transform of the full state, not just the mean).
  - **Reentry**: element-space → Cartesian → drag-dominated ballistic model, with GMM representation through the non-Gaussian transition region.
  - **Sensor-region handover / track handover**: treat as T2T fusion with one-shot cross-covariance handling (§6), plus track-stitching over gaps using smoothing (§7).
- **Constrained state estimation**: equality/inequality constraints (on-road, on-orbit energy, terrain floor) via projection, PDF truncation, or interior-point moment matching.

---

## 5. Data Association & Multi-Target Architecture

The Hungarian + gating + cascade stack is a strong deterministic baseline. The next tier:

- **JPDA / JIPDA**: probabilistic weighting of associations; JIPDA adds existence probability (nice fit with M-of-N replacement).
- **PMBM / Trajectory-PMBM (TPMBM)**: the current state of the art in Bayesian MOT. Closed-form conjugate prior over sets of targets (or sets of *trajectories*), unifying track initiation, existence, association hypotheses, and termination in one Bayesian recursion; consistently outperforms δ-GLMB and MHT-class methods in benchmark studies. Gaussian-mixture implementations with Murty's algorithm for k-best hypotheses are the standard realization. This is the single biggest architectural upgrade available to `thresh-tracker`.
- **GLMB / LMB**: labeled-RFS alternative if explicit track labels inside the filter are preferred; heavier bookkeeping, similar accuracy.
- **PHD / CPHD**: cheap intensity-only filters; useful as a clutter/birth-density estimator feeding PMBM rather than as the primary tracker.
- **Belief-propagation (BP/SPA) data association** (Meyer, Williams et al.): scalable marginal association probabilities via message passing; near-JPDA accuracy at near-linear cost. Excellent for dense automotive scenes.
- **Graph/network-flow MHT and GNN-based association**: min-cost-flow formulations give globally optimal multi-frame association offline; learned (graph neural network) affinities can replace hand-tuned gates where training data exists — consistent with thresh's hybrid transformer+Bayesian philosophy.
- **Extended-object PMBM (GGIW-PMBM)**: merges §1.1 extent modeling with the PMBM architecture — the reference approach for automotive radar/lidar MOT.
- **Murty's algorithm + gated hypothesis management**: needed infrastructure for any of PMBM/MHT/GLMB (k-best assignments); a clean standalone addition to `thresh-association`.

---

## 6. Multi-Sensor & Distributed Fusion

Existing: centralized, information filter, CI, T2T. Additions:

- **Out-of-Sequence Measurements (OOSM)**: Bar-Shalom's Bl1/Al1 one-step retrodiction algorithms, or fixed-lag state augmentation. Mandatory once you fuse sensors with different latencies (radar vs. EO/IR vs. ADS-B) — silently dropping late data biases fast movers badly.
- **Inverse Covariance Intersection (ICI) and Ellipsoidal Intersection**: less conservative than CI when correlation structure is partially known; better fused accuracy at same safety.
- **Chernoff / Generalized CI fusion of full densities**: fuses Gaussian mixtures and RFS densities (including distributed PMB/PMBM), not just single Gaussians — the principled way to do T2T fusion between two PMBM trackers on different platforms.
- **Consensus / diffusion information filters**: fully distributed fusion over a comms graph (no fusion center); relevant to swarm/UxS and Zenoh-style pub/sub architectures — the consensus iterations map naturally onto your middleware layer.
- **Cross-covariance-aware T2T**: track fusion ignoring cross-correlation is optimistic; either reconstruct cross-covariance (Bar-Shalom–Campo) or use CI-family bounds deliberately.
- **Asynchronous multi-rate fusion**: exact handling of heterogeneous revisit rates via continuous-discrete propagation (§1.4) rather than resampling to a common clock.

---

## 7. Smoothing & Batch Estimation

- **RTS and fixed-lag smoothers**: near-free accuracy gain for any latency-tolerant output (forensics, training-data generation for the transformer, track-quality scoring). TPMBM already implies smoothing over trajectories.
- **Factor-graph / sliding-window MAP estimation** (GTSAM-style, iSAM2): batch nonlinear least squares over a window; the estimation backbone of modern SLAM applies directly to tracking with maneuvers, OOSM, and biases as graph nodes. Also the cleanest formulation for **track stitching across observation gaps** — the boundary-crossing handover problem restated.
- **Initial Orbit Determination (IOD)**: Gauss/Gooding angles-only, Herrick-Gibbs from range-resolved triplets, then batch least squares refinement — required to birth space tracks from uncued sensors rather than from TLEs.

---

## 8. Learning-Augmented Estimation (hybrid with the ONNX front end)

- **KalmanNet / differentiable Kalman filters**: learn the Kalman gain (or Q/R) from data while retaining the Bayesian structure — interpretable, certifiable-adjacent, and a natural thesis for a neuro-symbolic shop.
- **Learned motion models with quantified uncertainty**: GP motion models (GP-PMBM hybrids appear in the 2025 literature) and transformer trajectory predictors (automotive: multimodal prediction heads) used as *proposal/prior* inside the Bayesian filter rather than as the tracker itself.
- **End-to-end transformer MOT (MT3v2, MOTR-class)** as a benchmark adversary: keep the Bayesian stack as the certifiable path, use learned trackers to measure the gap on nuScenes and synthetic scenarios.
- **Learned birth/clutter intensity estimation**: feed detector confidence statistics into PMBM's Poisson birth model — closes the loop between `thresh-inference` and `thresh-tracker`.

---

## 9. Evaluation & Consistency (extends `thresh-eval`)

- **NEES / NIS consistency tests** with chi-squared bounds — the standard for covariance realism; without them, propagation-accuracy claims can't be validated.
- **GOSPA metric** (and trajectory-GOSPA): decomposes error into localization, missed, false, and (trajectory variant) track-switch terms; the RFS community's preferred metric over OSPA/MOTA and the right target function for PMBM development.
- **Cramér–Rao Lower Bounds (PCRLB)** for tracking scenarios: quantifies the best achievable accuracy for a given sensor/target geometry — turns "is the tracker good" into "how far from optimal."

---

## 10. Sporadic Data, Dropouts & Observation Gaps

Intermittent measurements are the norm, not the exception: terrain masking, eclipse, radar beam scheduling, occlusion in traffic, comms dropouts, low-Pd targets. The mathematics splits into four layers.

### 10.1 Theory: when does the filter stay bounded?

- **Kalman filtering with intermittent observations** (Sinopoli et al., 2004): if measurements arrive as a Bernoulli process with rate λ, there is a **critical arrival probability λ_c** below which the expected covariance diverges. λ_c depends on the dynamics' instability (spectral radius of F). This gives thresh a principled, computable answer to "how sporadic is too sporadic for this target class?" — a natural `thresh-eval` diagnostic and a design constraint for sensor scheduling.
- **Markov-modulated observation models**: gaps are rarely i.i.d. — occlusion and masking are bursty. A two-state Gilbert–Elliott (visible/occluded) Markov chain on detection probability models burst gaps and yields stability conditions for that regime.
- **Deterministic gap prediction**: many gaps are *knowable in advance* — terrain masks from DTED line-of-sight, orbital eclipse and ground-station visibility windows, scheduled radar dwells. Feed predicted Pd(t) into the tracker (PMBM's Pd is already per-target/per-time) so a miss during a predicted gap doesn't penalize existence probability. This is cheap and high-value.

### 10.2 Coasting well: what to do *during* the gap

- **Full nonlinear propagation of the density, not just the mean** — this is where §2 pays off. Over a long gap the Gaussian goes non-Gaussian; adaptive GMM splitting (triggered by the same linearity tests) keeps the reacquisition gate honest. A single inflated Gaussian gate either balloons (clutter floods in) or lies (reacquisition fails).
- **Gap-aware process noise**: Q tuned for 50 ms updates is wrong for a 30 s gap. Use maneuver-envelope-based Q growth (worst-case acceleration over the gap → covariance growth as a function of gap duration) or IMM mode-probability diffusion toward the maneuvering model as the gap lengthens.
- **Existence/survival modeling**: make survival probability Ps and detection probability Pd explicit functions of gap duration and predicted visibility rather than constants — JIPDA and PMBM both accept this directly. Prevents both premature track deletion and zombie tracks.
- **Track-before-detect (TBD)** for sporadic *weak* returns: when detections are intermittent because SNR hovers at threshold, dynamic-programming or particle-based TBD integrates sub-threshold energy across frames instead of tracking thresholded detections. Pairs naturally with the Level-1/Level-2 radar fidelity stack in `thresh-synth` for evaluation.

### 10.3 Reacquisition & stitching: what to do *after* the gap

- **Track segment association (TSA)**: globally associate track fragments across gaps using forward-predicted and backward-smoothed densities from each fragment — the two-sided Mahalanobis/likelihood test is far more selective than gating a stale forward prediction alone. Min-cost-flow or Murty's over segment pairs gives the global optimum.
- **Backward smoothing into the gap** (RTS from the post-gap segment) plus forward prediction (from the pre-gap segment) yields a stitched MAP trajectory through the unobserved interval — the factor-graph backend (§7) does this natively.
- **Trajectory-PMBM handles this inside the filter**: because TPMBM's Bernoullis live on *trajectories*, a target that vanishes and reappears is a hypothesis the filter carries natively, rather than a heuristic bolt-on.

### 10.4 Asynchronous & delayed arrivals

- Already in §6 (OOSM, asynchronous multi-rate fusion) — worth emphasizing that the continuous-discrete formulation (§1.4) is the *enabler*: exact Q(dt) for arbitrary dt means sporadic arrival times cost nothing in consistency.
- **Negative information**: a scheduled dwell that *should* have seen the target and didn't is a measurement (Pd-weighted likelihood of miss updates the posterior — standard in PMBM, absent in plain KF pipelines). Valuable for fast movers crossing sensor boundaries.

---

## 11. High-Frequency Data

High rate (automotive lidar/radar at 10–100 Hz, ESA radar at kHz dwell rates, high-speed EO) creates the *opposite* problems: correlated noise, numerical conditioning, and compute budgets.

### 11.1 The white-noise assumption breaks first

- At high rates, measurement errors are **temporally correlated** (thermal drift, platform vibration, detector rolling shutter, tracker-loop dynamics in the sensor itself). Treating correlated noise as white makes the filter overconfident — covariance shrinks as √N while true error doesn't.
- Remedies: **state augmentation** (model the noise as a Gauss–Markov process in the state), **measurement differencing** (Bryson–Henrikson), or **measurement thinning to the noise decorrelation time**. The decorrelation-time test belongs in `thresh-eval`'s consistency suite (NIS whiteness test — Ljung–Box on innovations).
- Similarly, process-noise discretization at tiny dt: use the exact Van Loan Q(dt) (§1.4); naive Q·dt scaling misbehaves at both extremes.

### 11.2 Measurement compression: don't run the full update N times

- **Equivalent-measurement / tracklet compression** (Drummond): batch k high-rate measurements into one statistically equivalent measurement (or a decorrelated tracklet) and update at a lower rate. Exact for linear-Gaussian, near-exact otherwise; the standard answer for fusing a 100 Hz sensor into a 1 Hz fusion node without double-counting.
- **Information-form sequential updates**: for many measurements per epoch (lidar clusters, multi-return radar), accumulate in information space (additive) rather than covariance space — one inversion per epoch instead of per measurement. Thresh already has an information filter; extend it to batched updates.
- **Extended-object measurement-rate modeling**: at high rate a target produces a Poisson-distributed *number* of returns per frame; the GGIW gamma component (§1.1) models this rate explicitly rather than treating each return as a separate association problem.

### 11.3 Numerical & real-time considerations

- **Square-root/UD forms become mandatory, not optional**, at high update rates — round-off accumulates per update, and P loses positive-definiteness fastest exactly when updates are frequent and informative.
- **Anytime/graceful-degradation association**: at high rate the association budget is the bottleneck, not the filter. Gated sparse cost matrices, auction algorithm (Bertsekas) instead of full Hungarian (near-linear on sparse problems, interruptible), and BP association (§5) which is iterative and can be truncated to the compute budget.
- **Rate-adaptive processing**: run association/track management at a management rate (e.g., 10 Hz) while filtering at sensor rate, or vice versa — the tracklet layer (§11.2) is the interface. This maps cleanly onto a Zenoh topology: high-rate edge filtering, low-rate fused picture.
- **Stiffness**: high-rate propagation of stiff dynamics (reentry drag, low-perigee) wants implicit or exponential integrators rather than tiny explicit RK steps.

### 11.4 Exploiting high rate (it's not only a burden)

- High-rate sequences make **maneuver onset detection** sharp: GLR/CUSUM on innovations (§4) localizes maneuver start to a few samples, enabling near-immediate model-set switching instead of IMM's soft lag.
- **Doppler/range-rate fusion at rate**: high-frequency radar gives velocity observability that collapses covariance in the along-range direction quickly; sequential-update ordering (position then rate, or joint) affects conditioning — use the square-root update.
- **Smoothing at rate is cheap**: fixed-lag smoothing over a short high-rate window (§7) gives near-batch accuracy with bounded latency — ideal for automotive outputs with a 100–200 ms latency budget.

---

## 12. Prioritized Roadmap

| Priority | Item | Crate(s) | Rationale |
|---|---|---|---|
| **P0** | Square-root/UD filter forms, Joseph updates, Van Loan Q discretization | `thresh-filter`, `thresh-core` | Numerical soundness underpins everything; cheap |
| **P0** | Singer + CS + CTRA + 3D CT(ω) motion models | `thresh-filter` | Fills the airborne/automotive model gaps inside existing IMM |
| **P0** | NEES/NIS + GOSPA | `thresh-eval` | Makes accuracy claims falsifiable |
| **P1** | OOSM handling + sensor bias/registration states | `thresh-fusion` | Multi-sensor reality; biases masquerade as maneuvers |
| **P1** | PMBM filter (point targets, GM implementation, Murty's) | `thresh-association`, `thresh-tracker` | SOTA MOT architecture; unifies lifecycle + association |
| **P1** | Numerical orbit propagation (Cowell + EGM + drag + SRP; Gauss–Jackson or DP8(7)); evaluate `nyx-space` | `thresh-data` / new `thresh-astro` | SGP4 is not accuracy-grade; core to space use case |
| **P2** | GEqOE/equinoctial state representation + adaptive GMM splitting for orbital uncertainty | `thresh-filter`, astro crate | The propagation-accuracy centerpiece for space |
| **P2** | VS-IMM / mode-set adaptation + boost–coast–reentry model chain with handoff transforms | `thresh-filter`, `thresh-tracker` | The boundary-crossing answer |
| **P2** | CKF, IPLF, VB-adaptive/Student's-t updates | `thresh-filter` | Nonlinear measurement + detector-noise robustness |
| **P3** | GGIW extended-object models (+ GGIW-PMBM) | `thresh-filter`, tracker | Automotive lidar/radar accuracy tier |
| **P3** | BP/SPA association; factor-graph smoothing backend | `thresh-association`, new | Scale + gap-stitching |
| **P3** | STT / Differential Algebra propagation; PCE for offline realism studies | astro crate | Highest-accuracy tier; research-grade differentiator |
| **P3** | KalmanNet-style learned gains; learned birth/clutter intensities | `thresh-inference` ↔ tracker | Neuro-symbolic hybrid story |
| **P1** | Gap-aware Ps/Pd (visibility-predicted), negative-information updates, gap-duration Q growth | `thresh-tracker`, `thresh-filter` | Sporadic-data robustness; mostly parameterization, low code cost |
| **P1** | Equivalent-measurement/tracklet compression + batched information-form updates | `thresh-fusion`, `thresh-filter` | High-rate sensors without double-counting or per-measurement cost |
| **P2** | Track segment association (two-sided stitch: forward predict + backward smooth) | `thresh-association` | Reacquisition across long gaps / sensor handover |
| **P2** | Time-correlated measurement noise handling (augmentation or differencing) + NIS whiteness test | `thresh-filter`, `thresh-eval` | High-frequency overconfidence is silent and severe |
| **P2** | Auction algorithm + sparse gated cost matrices | `thresh-association` | Anytime association under high-rate compute budgets |
| **P3** | Intermittent-observation stability diagnostic (critical arrival probability λ_c) | `thresh-eval` | Principled "how sporadic is too sporadic" per target class |
| **P3** | Track-before-detect (DP or particle) for threshold-SNR targets | new / `thresh-tracker` | Sporadic weak returns; pairs with synth fidelity Levels 1–2 |

---

## Key References (starting points)

- Li & Jilkov, "Survey of Maneuvering Target Tracking" Parts I–V (motion models, ballistic targets, measurement models, decision-based, multiple-model) — the canonical taxonomy.
- Bar-Shalom, Willett, Tian, *Tracking and Data Fusion* — OOSM, registration, T2T cross-covariance.
- García-Fernández, Williams, Granström, Svensson — PMBM filter and trajectory-PMBM papers (IEEE TAES / TSP, 2018–2023); Granström's extended-object (GGIW) tutorial.
- Meyer et al., "Message Passing Algorithms for Scalable Multitarget Tracking," *Proc. IEEE* 2018 — BP association.
- DeMars, Bishop, Jah — adaptive Gaussian mixture splitting for orbital uncertainty; Armellin & Di Lizia — differential algebra propagation; recent GEqOE + adaptive-GMM work (Remote Sensing 2023) and GMM+STT conjunction analysis (Acta Astronautica 2022).
- Vallado, *Fundamentals of Astrodynamics and Applications* — force models, Gauss–Jackson, coordinate/time systems.
- Barrau & Bonnabel — Invariant EKF; Sola — error-state KF and Lie theory for robotics.
- Crouse, "Basic tracking using nonlinear 3D monostatic and bistatic measurements" and the NRL Tracker Component Library papers — implementation-grade algorithm details.
- Sinopoli et al., "Kalman Filtering with Intermittent Observations," *IEEE TAC* 2004 — critical arrival probability / covariance boundedness.
- Drummond — tracklets and equivalent measurements for decorrelated track fusion (SPIE Signal & Data Processing of Small Targets series).
- Bryson & Henrikson, "Estimation Using Sampled Data Containing Sequentially Correlated Noise," *J. Spacecraft* 1968 — time-correlated measurement noise.
- Davey, Rutten, Cheung — track-before-detect survey (histogram-PMHT, DP-TBD, particle-TBD).
- Bertsekas, "The Auction Algorithm" — sparse assignment under compute budgets.
- Zhang, Li, Nevatia, "Global Data Association for MOT via Network Flows," CVPR 2008 — segment stitching as min-cost flow.
