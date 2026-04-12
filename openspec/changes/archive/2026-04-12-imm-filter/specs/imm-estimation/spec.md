## Capability: IMM State Estimation

### Overview

The Interacting Multiple Model (IMM) filter maintains a bank of model-conditioned Kalman filters running in parallel, combining their outputs via Markov-switching transition probabilities to track targets undergoing unknown motion-mode changes.

## ADDED Requirements

### Requirement: IMM parallel model-conditioned filtering

The system MUST provide an IMM filter that runs N model-conditioned Kalman filters in parallel with Markov-switching transition probabilities.

#### Scenario: Model probability shift on maneuver onset

**WHEN** a target transitions from straight flight to a coordinated turn

**THEN** the IMM filter updates model probabilities each cycle based on innovation likelihoods

**SHALL** shift probability mass to the CTRV model within 5 update cycles
