# Interacting Multiple Model Filter

## What

Add an Interacting Multiple Model (IMM) filter to thresh-filter that combines multiple motion models (CV, CA, CTRV, CT) with Markov-switching probabilities, enabling the tracker to handle targets that change motion mode mid-track (e.g., straight flight to coordinated turn). The IMM maintains a bank of model-conditioned filters and merges their estimates using mode probabilities updated at each timestep.

## Why

Single-model KF/EKF tracks maneuvering targets poorly because the assumed dynamics diverge from reality when the target changes behavior. A CV filter will lag behind a turning target; a CT filter will inject phantom maneuvers during straight flight. The IMM estimator is the standard solution for adaptive tracking in defense and aerospace applications. thresh already implements the four motion models (CV, CA, CTRV, CT) individually, but there is no mechanism to combine them adaptively. Adding IMM closes the gap between thresh's current single-model tracking and production-grade maneuvering target trackers.

## How

- Implement an `ImmFilter<const N: usize>` struct in thresh-filter that holds N model-conditioned filter instances and a transition probability matrix (TPM)
- Each timestep executes the IMM cycle: interaction (mix states/covariances using mixing probabilities), per-model predict, per-model update, mode probability update via likelihood, and output combination
- Expose a builder API for configuring the model set and initial mode probabilities
- Add an IMM-aware track head variant in thresh-tracker that wraps `ImmFilter` and reports the dominant mode per track
- Add integration tests using maneuvering synthetic trajectories that verify mode-switching detection and improved tracking accuracy vs single-model baselines

## Out of scope

- Variable-structure IMM (adding/removing models at runtime)
- Online learning of transition probabilities from data
- GPU acceleration of the IMM cycle
- Particle filter or other non-Gaussian model-mixing approaches

## Affected crates

- thresh-filter: IMM filter implementation, IMM cycle logic, builder API
- thresh-tracker: IMM-aware track head that selects and reports dominant motion mode
- thresh-eval: IMM-specific test scenarios and accuracy comparison benchmarks
