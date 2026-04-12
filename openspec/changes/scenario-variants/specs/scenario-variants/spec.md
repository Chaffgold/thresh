## Capability: Synthetic Scenario Variants

### Overview

The benchmark runner supports multiple synthetic scenario types that exercise different tracking regimes, enabling repeatable evaluation across a range of operational conditions.

## ADDED Requirements

### Requirement: Multiple scenario type support

The benchmark runner MUST support at least 4 synthetic scenario types (cv-clean, maneuvering, heterogeneous, low-pd) selectable via the ScenarioParameters.

#### Scenario: Maneuvering scenario execution with metrics

**WHEN** a `synth-maneuvering` scenario is run

**THEN** the runner generates trajectories containing both CV and CTRV segments

**SHALL** generate trajectories with CV + CTRV segments and report MOTA/HOTA metrics
