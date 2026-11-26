# MHT Exhaustive Reference Delta

## ADDED Requirements

### Requirement: Exhaustive reference verification
The project SHALL maintain an exhaustive reference enumeration of global association hypotheses for multi-scan MHT fixtures. The reference SHALL be authored independently of the production hypothesis tree and share no code with it.

For each scan, the reference SHALL enumerate every feasible global association event:
- each existing track is either missed or assigned at most one detection;
- each detection is assigned to at most one track, or labelled a false alarm, or labelled a new track when the fixture enables births;
- detections outside a fixture's gate are infeasible for that track.

Each hypothesis SHALL carry the sum of its per-scan log scores along its complete history. Every fixture SHALL declare its score, state, birth and gate models. A fixture's declared models define only that fixture's expected values; they do not specify the production tracker's scoring, birth initialisation or gating. The committed fixtures SHALL be wholly invented, and each SHALL have at most 100,000 global hypotheses so that exhaustive enumeration stays practical in the native test lane.

Expected values SHALL be committed at full float64 precision with hand-derived anchors and compared with an absolute tolerance of 1e-10 plus a relative tolerance of 1e-9.

Contract-neutral reference assertions SHALL be limited to properties implied by this capability's hypothesis-tree, k-best and extraction requirements:
- exact hypothesis sets and counts;
- exact cumulative scores;
- no detection reused within a scan;
- score ranking, compared as ordered groups of scores tied within 1e-9;
- the maximum-score hypothesis without pruning.

Assertions of an N-scan commit rule, a track-identity policy, a tie-break order or native conformance SHALL be added only together with requirements of this capability that define them.

The required reference tests SHALL run in the native test lane without Python. That lane SHALL fail when a required reference test is missing, renamed without a manifest update, ignored, failing, or when the target executes zero tests.

#### Scenario: Deferred-resolution fixture
- **WHEN** the fixture is configured as follows:
  - **Tracks and model:** two scalar random-walk tracks A and B with prior means 0 and 4 and prior variances 1, process variance 0.5 per scan and measurement variance 0.5.
  - **Detection and births:** detection probability 0.9 and false-alarm density 0.05, with no births and no gate.
  - **Scans:** one detection per scan, at 2.1, 3.0 and 5.0.
  - **Per-scan score:**
    - an assignment scores ln(detection probability × predicted Gaussian density / false-alarm density);
    - a miss scores ln(1 − detection probability);
    - a false alarm scores 0.
- **THEN** the reference enumerates exactly 27 hypotheses, and no hypothesis assigns one detection to two tracks
- **AND** the maximum-score history is A, A, B, with per-scan scores −1.7802254585825268, −1.228787824770912 and −1.0471246793032758, and cumulative score −4.056137962656715
- **AND** the runner-up is B, B, B with −4.36634826630236

#### Scenario: Gated, empty-scan and birth fixtures
- **WHEN** the deferred-resolution model is reused with these variations:
  - **Gated two-detection fixture:** gate 9 on the squared normalised innovation and scans {2.5}, {0.5}, {2.5, 5.0}.
  - **Gated non-nested fixture:** gate 9 and scans {2.5}, {5.0}, {−2.0}.
  - **Empty-final-scan fixture:** no gate and scans {2.1}, {3.0}, {}.
  - **Birth fixture:**
    - one track with prior mean 0 and variance 0.25;
    - false-alarm density 0.01 and birth density 0.02, with a new track scoring ln(birth density / false-alarm density) and starting at its detection with variance 0.5;
    - gate 9 and scans {0.1, 5.0}, {−0.1}, {0.0, 5.2}.
- **THEN** hypothesis counts and maximum cumulative scores are:

  | Fixture | Hypotheses | Maximum cumulative score |
  |---|---|---|
  | Gated two-detection | 51 | −1.1827855182382336 |
  | Gated non-nested | 13 | −4.919774326293078 |
  | Empty-final-scan | 9 | −6.741456196614258 |
  | Birth | 185 | 11.96617537989422 |
- **AND** the birth fixture's best history contains a track born from the scan-1 detection at 5.0, missed at scan 2 and re-detected at 5.2 in scan 3

#### Scenario: Tied cumulative scores
- **WHEN** a fixture contains distinct hypotheses whose cumulative scores are equal within 1e-9
- **THEN** ranking is asserted as ordered tie groups, with membership-only comparison within each group, never as an arbitrary order inside a group, and no asserted prefix ends inside a tie group

#### Scenario: Reference tests cannot disappear
- **WHEN** a required reference test is removed, renamed without a manifest update, marked ignored, filtered out, or fails, or the target runs zero tests
- **THEN** the native lane fails rather than reporting the reference as verified

#### Scenario: Reference evidence is not native conformance
- **WHEN** the reference suite passes
- **THEN** it certifies only the reference enumeration of the committed fixtures
- **AND** native hypothesis-tree conformance, N-scan commitment and track identity count as verified only through assertions that requirements of this capability define and that compare native output with this reference
