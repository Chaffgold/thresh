## ADDED Requirements

### Requirement: MOTA computation
The system SHALL compute Multi-Object Tracking Accuracy: MOTA = 1 - (sum_t FN_t + FP_t + IDSW_t) / (sum_t GT_t), where FN = false negatives, FP = false positives, IDSW = identity switches, GT = ground truth objects. MOTA can be negative.

#### Scenario: Perfect tracking
- **WHEN** all ground truth objects are tracked with correct identity and no false alarms
- **THEN** MOTA SHALL equal 1.0

#### Scenario: Tracking with identity switches
- **WHEN** two tracks swap identities mid-sequence
- **THEN** MOTA SHALL reflect the identity switches as penalties, and the IDSW count SHALL be 2

### Requirement: IDF1 computation
The system SHALL compute IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN), the global trajectory-level identity metric based on the optimal matching between predicted and ground truth trajectories.

#### Scenario: IDF1 with fragmented tracks
- **WHEN** a single ground truth trajectory is covered by 3 separate predicted tracks
- **THEN** IDF1 SHALL penalize the fragmentation through increased IDFN for the unmatched portions

### Requirement: HOTA computation
The system SHALL compute Higher Order Tracking Accuracy: HOTA = sqrt(DetA * AssA) integrated over IoU thresholds alpha in {0.05, 0.10, ..., 0.95}. DetA (detection accuracy) and AssA (association accuracy) SHALL be independently reportable.

#### Scenario: HOTA decomposition
- **WHEN** a tracker has perfect detection but poor association (many ID switches)
- **THEN** DetA SHALL be high, AssA SHALL be low, and HOTA SHALL reflect the geometric mean — penalizing the poor association

#### Scenario: IoU threshold sweep
- **WHEN** HOTA is computed
- **THEN** the system SHALL report HOTA values at each individual IoU threshold as well as the integrated (averaged) HOTA

### Requirement: AMOTA computation for 3D tracking
The system SHALL compute Average Multi-Object Tracking Accuracy by averaging MOTA across multiple recall thresholds, as defined by the nuScenes tracking benchmark. AMOTP (average localization error) SHALL also be computed.

#### Scenario: Distance-dependent recall
- **WHEN** tracking quality degrades with distance (more misses at long range)
- **THEN** AMOTA SHALL capture this degradation by averaging MOTA across recall operating points rather than at a single threshold

### Requirement: Evaluation report generation
The system SHALL produce a structured evaluation report (JSON and human-readable table) containing all computed metrics, per-class breakdowns, and summary statistics for a given tracker output against ground truth.

#### Scenario: Per-class metric breakdown
- **WHEN** evaluating a tracker on a multi-class scenario (aerodynamic, UAV, ballistic targets)
- **THEN** the report SHALL include MOTA, IDF1, HOTA, and AMOTA broken down by target class as well as aggregate values

#### Scenario: JSON output format
- **WHEN** evaluation completes
- **THEN** the system SHALL output a JSON document with all metrics keyed by metric name and class, parseable by downstream tools
