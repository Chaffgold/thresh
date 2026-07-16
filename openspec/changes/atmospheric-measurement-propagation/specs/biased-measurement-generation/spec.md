# biased-measurement-generation Specification (Delta)

## ADDED Requirements

### Requirement: Opt-in atmospheric bias in synthetic radar generation
Synthetic radar measurement generation SHALL optionally apply the tropospheric refraction and ionospheric group-delay biases to the true RAE geometry before measurement noise is added (bias-then-noise), controlled by an optional configuration that defaults to off. With the option absent, generation SHALL be byte-for-byte the existing behavior.

#### Scenario: Default generation is unchanged
- **WHEN** radar measurements are generated with the bias configuration absent
- **THEN** the outputs SHALL be bitwise identical to the pre-change generator for the same inputs and RNG seed

#### Scenario: Bias applied before noise
- **WHEN** the bias configuration is enabled with measurement noise set to zero
- **THEN** the generated RAE SHALL equal the true geometry plus exactly the modeled refraction and delay biases

### Requirement: Calibrated benchmarks are bitwise unchanged
Every calibrated benchmark scenario SHALL produce digit-for-digit identical metrics before and after this change (the bias path is opt-in and no committed scenario opts in). A verification run SHALL confirm this.

#### Scenario: Invariance verification
- **WHEN** the four calibrated benchmark scenarios run on the pre-change and post-change trees
- **THEN** every reported metric SHALL be digit-for-digit identical

### Requirement: Corrected-vs-uncorrected consistency demonstration
An end-to-end test through the real benchmark runner SHALL demonstrate the bias class is visible to the evaluation layer and recoverable by correction: with biased truth and an uncorrected tracker, MOTP SHALL degrade and the consistency statistics (ANEES and/or ANIS) SHALL shift measurably beyond the honest run's values; applying the atmospheric corrections at the RAE→Cartesian conversion SHALL recover statistics near the honest run's, with all three runs' values recorded at the test.

#### Scenario: Uncorrected bias degrades the tracker measurably
- **WHEN** the demonstration scenario runs with biases enabled and no correction
- **THEN** MOTP SHALL exceed the honest run's value and at least one consistency statistic SHALL shift beyond the honest run's, by documented margins

#### Scenario: Correction recovers near-honest statistics
- **WHEN** the same scenario runs with the corrections applied at the conversion seam
- **THEN** MOTP and the consistency statistics SHALL return to within documented bands of the honest run's values

#### Scenario: Demonstration is deterministic
- **WHEN** the demonstration runs twice with the same seed
- **THEN** all recorded statistics SHALL be bitwise identical between runs
