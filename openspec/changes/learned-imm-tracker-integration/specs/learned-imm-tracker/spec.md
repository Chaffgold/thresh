## Capability: Learned-IMM Tracker

### Overview

A tracker-level entry point that drives the per-track IMM mode-probability
update from a learned ONNX classifier instead of the analytic Markov update,
behind the `learned-imm` feature. It composes Track B's filter-level
`LearnedImmFilter` (from `flight-data-training-pipeline`) with
`MultiObjectTracker`, and exposes the learned path through the evaluation
harness so the analytic and learned pipelines can be compared on identical
inputs. Default (non-`learned-imm`) builds are unaffected.

## ADDED Requirements

### Requirement: Learned-IMM tracker constructor

The tracker MUST provide a constructor that builds an IMM tracker whose per-track
mode-probability update is supplied by an ONNX classifier, gated on the
`learned-imm` feature. The constructor SHALL validate its inputs up front and
fail (rather than silently degrade) when they are unusable.

#### Scenario: Constructing a learned-IMM tracker from a valid classifier

**WHEN** `MultiObjectTracker::new_imm_position_learned` is called with a 4-model
`cv_ca_ctrv_ct` config factory and a path to a loadable ONNX IMM mode classifier

**THEN** it returns a tracker that, for each new track, wraps the track's IMM
bank in a `LearnedImmFilter` driven by that classifier.

#### Scenario: Rejecting an unusable configuration or classifier

**WHEN** the constructor is given an invalid `ImmConfig`, a bank whose model
count is not the 4 the classifier expects, or a path to an ONNX model that
cannot be loaded

**THEN** it returns an error and does not construct a tracker.

### Requirement: Learned mode update with analytic fallback

A track created by the learned-IMM tracker MUST apply the classifier's mode
probabilities once enough history exists, and otherwise behave exactly like the
analytic IMM track.

#### Scenario: Learned tracker tracks a target end-to-end

**WHEN** a learned-IMM tracker is stepped over a constant-velocity target for
more than the classifier's window length

**THEN** the track is confirmed, follows the target, and exposes a dominant mode
and mode-probability vector — having used the analytic update during the warm-up
window and the learned update thereafter.

#### Scenario: Classifier error falls back to the analytic update

**WHEN** the classifier cannot be evaluated for a track (e.g. the model fails to
load at track birth or inference errors)

**THEN** the track falls back to the analytic IMM update so tracking continues
rather than dropping the track.

### Requirement: Runnable learned-vs-analytic evaluation

The evaluation harness MUST be able to run the learned tracker on the same
scenarios as the analytic tracker, behind the `learned-imm` feature, so the two
can be compared.

#### Scenario: eval-tracker runs the learned pipeline

**WHEN** `eval-tracker --learned-imm --model <path>` is run with the
`learned-imm` feature enabled

**THEN** the driver evaluates each scenario through the learned-IMM tracker and
reports MOTA / MOTP / IDF1, distinct from the analytic baseline.

#### Scenario: Learned evaluation requires a model path

**WHEN** `eval-tracker --learned-imm` is run without `--model`

**THEN** the driver exits with an error explaining that a classifier path is
required.
