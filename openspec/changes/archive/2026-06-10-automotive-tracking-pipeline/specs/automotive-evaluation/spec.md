## Capability: Automotive Evaluation

### Overview

Automotive evaluation for the tracker: per-class MOT metrics and nuScenes AMOTA
reported over held-out nuScenes scenes via the existing `NuScenesBridge`. Builds
on `thresh-eval`'s existing MOTA/MOTP/IDF1 and `compute_amota`.

## ADDED Requirements

### Requirement: Per-class MOT metrics

`thresh-eval` MUST report MOT metrics aggregated per `TargetClass` in addition to
the existing aggregate metrics.

#### Scenario: Metrics are broken down by class

**WHEN** an evaluation is run over frames containing multiple automotive classes

**THEN** MOTA/MOTP are reported per class as well as in aggregate.

**SHALL** leave the existing aggregate metric functions' behaviour unchanged.

### Requirement: nuScenes-benchmarked evaluation driver

There MUST be an evaluation driver that runs the automotive tracker over a
nuScenes split through `NuScenesBridge` and reports per-class MOTA and AMOTA.

#### Scenario: Tracker is evaluated on a nuScenes split

**WHEN** the automotive eval driver is run over a held-out nuScenes split
(e.g. the `mini` split)

**THEN** it ingests frames via `NuScenesBridge`, tracks them with the automotive
tracker, and emits per-class MOTA plus AMOTA.

#### Scenario: Heavy data stays out of CI

**WHEN** the default CI build runs

**THEN** it does not require the nuScenes dataset to be present; the nuScenes
evaluation is run on demand, with acquisition documented separately.
