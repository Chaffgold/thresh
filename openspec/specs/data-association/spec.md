# data-association Specification

## Purpose
TBD - created by archiving change transformer-fusion-tracker. Update Purpose after archive.
## Requirements
### Requirement: Hungarian algorithm for optimal assignment
The system SHALL implement the Hungarian algorithm (or Jonker-Volgenant variant) solving the linear assignment problem on an N x M cost matrix in O(n^3) time, returning the optimal one-to-one assignment minimizing total cost.

#### Scenario: Square cost matrix assignment
- **WHEN** given a 5x5 cost matrix with known optimal assignment
- **THEN** the algorithm SHALL return the assignment matching the known minimum-cost solution

#### Scenario: Rectangular cost matrix
- **WHEN** given a cost matrix where N_tracks != M_detections (more detections than tracks or vice versa)
- **THEN** the algorithm SHALL handle unassigned tracks and unassigned detections, returning partial assignments and lists of unmatched indices

### Requirement: Mahalanobis distance gating
The system SHALL compute the Mahalanobis distance d^2_M = (z - Hx)^T S^{-1} (z - Hx) between each track prediction and each detection, and gate assignments at the chi-squared threshold for the measurement dimension m at a configurable significance level alpha.

#### Scenario: Gate rejection of unlikely associations
- **WHEN** a detection falls outside the chi-squared gate (e.g., chi^2_4(0.99) = 13.28 for 4D measurements at 99%)
- **THEN** the cost matrix entry for that track-detection pair SHALL be set to infinity (or excluded from assignment)

#### Scenario: Correct gating with different measurement dimensions
- **WHEN** sensor A provides 2D position measurements and sensor B provides 4D position+velocity measurements
- **THEN** the gating threshold SHALL automatically adjust based on the measurement dimension of each sensor

### Requirement: IoU-based cost computation
The system SHALL compute Intersection-over-Union (IoU) between predicted bounding boxes and detected bounding boxes, supporting both 2D IoU and 3D IoU (axis-aligned and rotated).

#### Scenario: 2D IoU cost matrix
- **WHEN** given predicted 2D bounding boxes from tracks and detected 2D boxes
- **THEN** the cost matrix SHALL contain 1 - IoU values, with IoU=0 for non-overlapping boxes and IoU=1 for identical boxes

#### Scenario: 3D rotated IoU
- **WHEN** given 3D bounding boxes with (x, y, z, l, w, h, yaw) parameterization
- **THEN** the system SHALL compute correct 3D IoU accounting for rotation in the ground plane

### Requirement: Fused cost matrix
The system SHALL support constructing a fused cost matrix C_ij = alpha * d_motion + (1 - alpha) * d_appearance where d_motion is the normalized Mahalanobis or IoU-based distance and d_appearance is an embedding cosine distance. The weighting alpha SHALL be configurable.

#### Scenario: Motion-only fallback
- **WHEN** appearance embeddings are not available (e.g., radar-only tracking)
- **THEN** the system SHALL fall back to motion-only cost (alpha = 1.0) without error

#### Scenario: Cascaded association
- **WHEN** configured for cascaded matching (high-confidence detections first, then low-confidence)
- **THEN** the system SHALL run assignment in stages, passing unmatched tracks and detections to subsequent stages with potentially different cost functions

