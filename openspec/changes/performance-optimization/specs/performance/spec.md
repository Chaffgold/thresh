## Capability: Performance Optimization

### Overview

Criterion micro-benchmarks cover the critical hot paths in association, filtering, and tracking to provide stable, reproducible performance measurements for regression detection.

## ADDED Requirements

### Requirement: Criterion micro-benchmarks for hot paths

The system MUST provide criterion micro-benchmarks for the Hungarian assignment, Kalman filter update, and tracker step hot paths.

#### Scenario: Stable benchmark timings for core operations

**WHEN** criterion benchmarks are run

**THEN** each benchmark exercises a representative workload size

**SHALL** report stable per-iteration timings for association (100 targets), filter update (single track), and full tracker step (50 targets, 200 detections)
