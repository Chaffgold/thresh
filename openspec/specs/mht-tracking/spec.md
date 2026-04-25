# mht-tracking Specification

## Purpose
TBD - created by archiving change jpda-mht-association. Update Purpose after archive.
## Requirements
### Requirement: Hypothesis tree maintenance

The system MUST maintain a hypothesis tree where each node represents a global association hypothesis at one scan, with branches representing different feasible track-detection assignments.

#### Scenario: Expanding hypotheses at a new scan

**WHEN** a new scan arrives with M detections and T existing tracks

**THEN** the MHT generates child hypotheses for each parent hypothesis by enumerating feasible global assignments (each detection to at most one track, with new-track and false-alarm options)

**SHALL** assign a cumulative log-likelihood score to each child hypothesis based on the parent's score plus the assignment's likelihood

### Requirement: N-scan pruning

The system MUST implement N-scan pruning that collapses hypotheses agreeing on all associations older than N scans, reducing the tree to a manageable size while preserving recent ambiguity.

#### Scenario: Pruning with N=3

**WHEN** the hypothesis tree has accumulated 5 scans of data and N-scan depth is configured to 3

**THEN** the pruner identifies all hypotheses that agree on their association decisions at scans 1 and 2, merges them into single representative hypotheses

**SHALL** reduce the total hypothesis count while preserving all distinct association possibilities within the most recent 3 scans

### Requirement: K-best pruning

The system MUST implement k-best pruning that retains only the top k global hypotheses ranked by cumulative likelihood, discarding unlikely branches.

#### Scenario: Retaining top 50 hypotheses

**WHEN** hypothesis expansion produces 500 candidate hypotheses and k is configured to 50

**THEN** the pruner ranks all hypotheses by cumulative log-likelihood score and retains only the top 50

**SHALL** discard hypotheses with the lowest scores and reclaim their memory

### Requirement: Best-hypothesis track extraction

The system MUST extract track states from the highest-scoring global hypothesis for output to downstream consumers.

#### Scenario: Extracting confirmed tracks

**WHEN** the tracker is queried for current track states after an MHT update cycle

**THEN** it selects the highest-scoring global hypothesis and returns the track states and covariances associated with that hypothesis

**SHALL** provide consistent track IDs across timesteps, with ID continuity determined by the best hypothesis's association history

