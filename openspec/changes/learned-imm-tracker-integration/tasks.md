# Tasks — Learned-IMM Tracker Integration

## 1. Tracker constructor + seam

- [x] 1.1 Add a `learned-imm` feature to `thresh-tracker` (`= ["thresh-filter/learned-imm"]`). _`crates/thresh-tracker/Cargo.toml`._
- [x] 1.2 Add gated `MultiObjectTracker::new_imm_position_learned(config_factory, onnx_path, measurement_noise_sigma, gate_threshold) -> Result<Self, String>`. _Validates config, asserts the 4-model `cv_ca_ctrv_ct` bank, and loads the ONNX once up front to fail fast; each track then loads its own session at birth (sessions are not shared)._
- [x] 1.3 Add a gated `learned_imm_filters` map + `learned_classifier_path`; wire birth/predict/update/removal to prefer the learned map. _`src/tracker.rs`: `insert_imm_filter` wraps the bank in `LearnedImmFilter` when active, analytic fallback otherwise._

## 2. Eval harness wiring

- [x] 2.1 Extract a shared `run_eval_with_tracker` core; add gated `run_eval_harness_learned`. _`crates/thresh/src/eval_harness.rs`._
- [x] 2.2 Add a `learned-imm` feature to `thresh`; route `eval-tracker --learned-imm --model <path>` through the learned harness. _`crates/thresh/src/bin/eval-tracker.rs`; errors if `--learned-imm` is given without `--model`._

## 3. Tests, CI, verification

- [x] 3.1 Gated unit tests: constructor loads the stub, errors on a missing model, and a learned tracker confirms + follows a CV target end-to-end. _`src/tracker.rs` tests (run under `--features learned-imm`)._
- [x] 3.2 `onnx-tests` CI job runs `cargo test -p thresh-tracker --features learned-imm` and builds the umbrella `eval-tracker` with `learned-imm`. _`.github/workflows/ci.yml`._
- [x] 3.3 Verify the seam is live: `eval-tracker --learned-imm --model <stub>` produces metrics that differ from the analytic baseline (the random stub degrades maneuvering MOTA from 0.890 → 0.086), proving the learned classifier drives the mode update rather than being a no-op.

## 4. Follow-on (deferred)

- [ ] 4.1 Drop in a trained `imm_mode_classifier.onnx` and record a representative analytic-vs-learned A/B (gated on the GPU training run — `flight-data-training-pipeline` tasks 6.8 / 8.5).
