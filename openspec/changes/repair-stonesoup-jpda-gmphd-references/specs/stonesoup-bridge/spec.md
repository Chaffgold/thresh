# Executable JPDA and GM-PHD Reference Delta

## ADDED Requirements

### Requirement: Explicit JPDA reference composition
The bridge SHALL construct an executable JPDA reference from caller-supplied compatible predictor and updater components. It SHALL expose gate probability, detection probability, and clutter spatial density explicitly, reject invalid numeric configuration before interpreter access, and retain host-owned interpreter lifecycle and structured dependency diagnostics.

#### Scenario: Supported construction and configuration
- **WHEN** the host supplies supported linear-Gaussian components and finite configuration with gate probability strictly between zero and one, detection probability in [0,1], and positive clutter density
- **THEN** construction succeeds and uses the supplied components and probabilities; the default detection probability is 0.85

#### Scenario: Invalid configuration
- **WHEN** a probability is outside its supported interval, density is nonpositive, or a configuration scalar is nonfinite
- **THEN** construction returns an algorithm-validation error without importing packages or silently clamping values

### Requirement: Exact JPDA marginal semantics
The JPDA reference SHALL return marginal association probabilities over mutually exclusive joint events, retain the missed-detection hypothesis, and use gate probability as a measurement-dimension-dependent probability rather than as a squared-distance threshold. Clutter density SHALL have units compatible with the likelihood in one declared measurement space. Association SHALL preserve supplied observation timestamps and object identity without substituting arrival time or retiming observations.

#### Scenario: Ambiguous measurements
- **WHEN** two or three tracks share eligible detections in a synthetic fixture
- **THEN** finite nonnegative marginals agree with independent exhaustive event enumeration, each track's probabilities including missed detection sum to one, and a detection's summed occupancy across tracks does not exceed one

#### Scenario: Empty or ineligible observations
- **WHEN** the track set is empty, or a nonempty track set has no detections, only gated-out detections, or zero detection probability
- **THEN** the result is respectively empty or assigns probability one to missed detection for each track

#### Scenario: Explicit observation times
- **WHEN** supported detections carry observation times distinct from the association time
- **THEN** detection predictions use their observation times and missed predictions use the association time, without a timestamp fallback or arrival-time substitution

### Requirement: Explicit GM-PHD scan inputs
The GM-PHD reference SHALL accept caller-supplied compatible predictor and updater components, prior Gaussian intensity components, explicit birth components, detections, and an explicit scan timestamp. Priors and births SHALL be tagged weighted Gaussian components with finite nonnegative intensity weights; every birth SHALL carry the reserved birth tag, and priors SHALL NOT carry it. All required times SHALL be timezone-aware UTC values. Prior components SHALL share a timestamp no later than the scan, and births and detections SHALL carry the scan timestamp. Invalid input contracts SHALL fail explicitly before hypothesis generation. Caller-owned input containers and component contents SHALL remain unchanged.

Prior, birth, and detection inputs SHALL be reusable collections, such as lists, tuples, sets, or Gaussian mixtures. One-shot iterators and generators SHALL be rejected before any input collection is consumed.

#### Scenario: Supported scan
- **WHEN** valid prior, birth, and detection inputs describe one compatible linear-Gaussian measurement space at the declared scan time
- **THEN** the reference predicts, forms hypotheses, updates the intensity, and reduces the mixture, returning Gaussian components with hypothesis provenance rather than fabricated target tracks

#### Scenario: Missing or inconsistent scan context
- **WHEN** an input lacks the required component type, tag, or timestamp, a prior is later than the scan, or a birth/detection time differs from the scan time
- **THEN** the reference returns an algorithm-validation error without silently retiming, inserting births, or modifying caller inputs

#### Scenario: Empty inputs
- **WHEN** any combination of prior, birth, and detection collections is empty
- **THEN** the scan still uses its explicit prediction time, detections alone do not create intensity, and an empty result reports an empty mixture whose component-derived timestamp is absent

#### Scenario: One-shot input cannot be preserved
- **WHEN** any prior, birth, or detection input is a generator or another one-shot iterator
- **THEN** the call returns an algorithm-validation error without consuming that iterator or any other input collection

### Requirement: GM-PHD intensity and birth compatibility
The reference SHALL preserve the pinned runtime's documented intensity semantics rather than normalizing mixture weights to sum to one. For supported ungated fixtures, an existing component of weight w SHALL contribute missed weight (1-Pd) Ps w and per-detection weight Pd Ps w q divided by the sum of clutter density and all eligible Pd Ps w q contributions. Detected births SHALL receive the pinned composition's survival factor and a non-birth tag; undetected births SHALL contribute no missed component. Exact zero updater weights SHALL receive the pinned runtime's machine-epsilon replacement before reduction, documented separately from mathematical zero; subsequent reduction SHALL retain its configured removal and redistribution behavior.

#### Scenario: One component and one detection
- **WHEN** a scalar component has mean 0, variance 1, weight 0.5, zero process noise, measurement variance 1, observation 0, survival probability 0.8, detection probability 0.5, and clutter density 0.2/sqrt(4*pi), with neutral reduction
- **THEN** the detected component has weight 0.5, mean 0, variance 0.5; the missed component has weight 0.2, mean 0, variance 1; total intensity is 0.7

#### Scenario: Birth behavior
- **WHEN** the same scan contains only a birth of weight 0.5
- **THEN** a detected birth has weight 0.5 and no missed component, while an empty detection scan produces an empty mixture

#### Scenario: Probability endpoints
- **WHEN** survival or detection probability is zero or one and reduction is neutral
- **THEN** independently calculated weights account explicitly for machine-epsilon substitution, without treating it as genuine positive target evidence

### Requirement: Explicit GM-PHD gating and reduction semantics
The reference SHALL distinguish measurement gating from mixture merging. Optional measurement gating SHALL use a finite positive squared Mahalanobis threshold, with omission meaning unbounded gating. Reduction SHALL use a finite nonnegative pruning threshold, finite nonnegative squared Mahalanobis merge threshold, and positive maximum component count. Probabilities SHALL be finite in [0,1] and clutter density finite and positive. Invalid numeric configuration SHALL return an algorithm-validation error before imports. The reference SHALL retain and document pinned pruning redistribution, strict merge comparison, pairwise merge-weight clipping at one, and truncation redistribution; it SHALL NOT claim unconditional intensity preservation.

#### Scenario: Squared merge threshold
- **WHEN** unit-variance scalar components at means 0 and 2 are reduced with merge threshold 4
- **THEN** their squared distance equals the threshold and they remain separate because the merge comparison is strict

#### Scenario: Prune and truncate redistribution
- **WHEN** weights [0.05,0.2,0.3] are pruned at 0.1 with other reductions neutral, or weights [0.1,0.2,0.3] are truncated to two with other reductions neutral
- **THEN** the respective surviving weights are [0.225,0.325] and [0.35,0.25], independent of output ordering

#### Scenario: Merge clipping
- **WHEN** coincident components of weights 0.7 and 0.6 are merged
- **THEN** their merged weight is one, not 1.3, and the result is reported as pinned reference behavior rather than mass conservation

#### Scenario: Invalid reduction or gate configuration
- **WHEN** a finite gate is nonpositive, a pruning or merge threshold is negative, a required scalar is nonfinite, or maximum component count is zero
- **THEN** construction rejects the configuration rather than passing a malformed graph to Python

### Requirement: Executed bounded algorithm verification
The required optional-runtime lane SHALL execute every new JPDA and GM-PHD reference test in its exact test manifest using the existing isolated pinned dependency baseline. Numerical expectations SHALL come from independent arithmetic or event enumeration, not another execution of the same reference graph. Tests SHALL use wholly invented synthetic inputs and preserve existing boundary and no-Python checks. Passing these tests SHALL certify only the specified linear-Gaussian reference behavior, not native-algorithm parity, full-track accuracy, MHT/IMM correctness, deployment readiness, or release permission.

#### Scenario: Independent numerical evidence
- **WHEN** the reference suite runs
- **THEN** it checks constructor success, JPDA marginals and gating, GM-PHD prediction/update/birth/intensity/reduction, probability endpoints, invalid input diagnostics, and input preservation against independently derived expectations

#### Scenario: Required tests cannot disappear
- **WHEN** a required reference test is missing, ignored, renamed without a manifest update, fails, or executes zero assertions because its dependencies are absent
- **THEN** the lane fails rather than treating a dependency skip or dispatch double as algorithm verification
