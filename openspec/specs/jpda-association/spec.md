# jpda-association Specification

## Purpose
TBD - created by archiving change jpda-mht-association. Update Purpose after archive.
## Requirements
### Requirement: Association probability computation

The system MUST compute marginal association probabilities for each track-detection pair by enumerating feasible joint association events within the gating region and summing event probabilities for each pair.

#### Scenario: Two tracks with overlapping gates in clutter

**WHEN** two tracks have overlapping gating regions containing three detections and one clutter hypothesis

**THEN** the JPDA enumerates all feasible joint events (respecting the constraint that each detection maps to at most one track), computes each event's probability from the individual likelihoods

**SHALL** produce marginal probabilities for each track-detection pair that sum to at most 1.0 per track (with the remainder being the missed-detection probability)

### Requirement: Merged-measurement state update

The system MUST update each track's state and covariance using a probability-weighted combination of innovations from all gated detections, including the spread-of-innovations term in the covariance update.

#### Scenario: Track updated with three gated detections

**WHEN** a track has three gated detections with association probabilities [0.5, 0.3, 0.2]

**THEN** the JPDA computes a merged innovation as the weighted sum of individual innovations, updates the state with this merged innovation, and augments the covariance with the spread-of-innovations term

**SHALL** produce a covariance that is larger than any single-detection update would produce, reflecting the association uncertainty

### Requirement: Missed-detection hypothesis

The system MUST include a missed-detection (no-association) hypothesis for each track, with probability derived from the detection probability and spatial density of false alarms.

#### Scenario: Low detection probability environment

**WHEN** the detection probability is set to 0.7 and the false alarm spatial density is 1e-6

**THEN** the missed-detection probability for each track MUST be computed from these parameters and included in the normalization of association probabilities

**SHALL** ensure all association probabilities for a track (including missed-detection) sum to 1.0

