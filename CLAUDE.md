# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is thresh

Multi-sensor fusion multi-object tracking framework in Rust. Hybrid architecture: transformer-based detection (ONNX Runtime) + classical Bayesian state estimation (Kalman filter family). Targets heterogeneous aerospace objects from UAVs to ballistic missiles.

## Build & Test Commands

```sh
cargo build --workspace          # Build all crates
cargo test --workspace           # Run all tests
cargo test -p thresh-filter      # Test a single crate
cargo clippy --workspace --all-targets -- -D warnings  # Lint (CI-strict: warnings are errors)
cargo fmt --all                  # Format
cargo doc --workspace --no-deps  # Build docs (CI uses RUSTDOCFLAGS=-Dwarnings)
openspec validate --all --strict --no-interactive      # Validate OpenSpec artifacts
```

CI enforces: `RUSTFLAGS=-Dwarnings` globally — all warnings are compile errors.

## Pre-commit Hooks

Installed via `pre-commit install`. On commit: fmt, clippy, cargo check, openspec validate. On push: cargo test.

## Architecture

Cargo workspace with 10 crates. Dependency flow (lower depends on higher):

```
thresh-core          (types: state vectors, measurements, covariance, coords, sensors, tracks, time)
    ↓
thresh-filter        (KF, EKF, UKF + motion models: CV, CA, CTRV, CT)
thresh-association   (Hungarian, Mahalanobis gating, IoU, cascaded association)
thresh-fusion        (centralized fusion, information filter, covariance intersection)
    ↓
thresh-tracker       (track lifecycle, M-of-N confirmation, class-specific heads)
    ↓
thresh-inference     (ONNX Runtime pipeline — feature-gated: `onnx`)
thresh-bridge        (PyO3 → Stone Soup JPDA/MHT/IMM — feature-gated: `stonesoup`)
thresh-synth         (synthetic radar, EO/IR, ADS-B data generation)
thresh-eval          (MOT metrics: MOTA, MOTP, IDF1, HOTA, AMOTA)
    ↓
thresh               (umbrella re-export crate + integration tests)
```

Key design choices:
- **nalgebra** for all matrix/vector math (both static and dynamic sizing)
- Motion models and filters use traits for extensibility
- Feature gates isolate heavy optional deps (ONNX Runtime, PyO3)

## OpenSpec Workflow

Design specs live in `openspec/changes/`. Three active change sets: `transformer-fusion-tracker`, `test-data-pipeline`, `hifi-sensor-simulation`. Each contains proposal, design, tasks, and capability specs.

Claude Code commands for OpenSpec: `/opsx:explore`, `/opsx:propose`, `/opsx:apply`, `/opsx:archive`.

## Branch Strategy

Gitflow: `main` ← `develop` ← `feature/*` branches. PRs target `develop`. CI triggers on pushes to `main` and `develop`.

## Worktrees

Use git worktrees for parallel feature development. Worktrees live in `../thresh-worktrees/`.

```sh
# Create a worktree for a new feature branch off develop
git worktree add ../thresh-worktrees/<branch-name> -b feature/<name> develop

# Create a worktree for an existing remote branch
git worktree add ../thresh-worktrees/<branch-name> feature/<name>

# List active worktrees
git worktree list

# Remove a worktree after merging
git worktree remove ../thresh-worktrees/<branch-name>
```

Each worktree is a full working copy that shares the same `.git` — commits, branches, and stash are shared across all worktrees. Build artifacts (`target/`) are per-worktree.

## Reference Docs

Mathematical and algorithmic references are in `docs/reference/` — covering Kalman filter derivations, fusion math, data association, and transformer architectures.

## Style Guide: Phase-Helper Decomposition

SonarCloud rule `rust:S3776` enforces cognitive complexity ≤ 15 per function. When a function or test grows large enough to trip this gate, prefer **phase-helper decomposition** over early returns, flag variables, or loop unrolling:

1. **Read the function top-to-bottom and write down the logical phases** in a comment or scratchpad (e.g., `build_square_cost → reduce → greedy → augment → mark_cover → update → extract`). Each phase is typically one paragraph of code with a clear name.
2. **Extract each phase into its own private helper** with a descriptive verb name (`reduce_cost_matrix`, `augment_matching`, `interpolate_trajectory_to_grid`). Take `&mut` state by parameter rather than plumbing a context struct, unless you hit `clippy::too_many_arguments` — then group into a small `Ctx` struct.
3. **Keep the top-level function a linear sequence of phase calls** so its complexity is O(number of phases). No nested `if let` chains, no `while let Some(...)` loops at the top level — those belong in helpers.
4. **Each helper gets its own focused unit test** where possible (per-phase tests are cheaper to reason about than large end-to-end tests).
5. **Document preconditions explicitly** when a helper relies on invariants established by an earlier phase (see `mark_cover` in `crates/thresh-association/src/hungarian.rs` for a worked example — its doc comment states that it requires a maximum matching, which `run_munkres_loop` enforces by calling `augment_matching` first).

Worked examples in the tree: `hungarian.rs` (8 phases), `adsb.rs::extract_ground_truth` (4 phases), `rk4_stage` in `orbital.rs` (4 RK4 stages collapsed to one shared helper).

Avoid the alternatives: SonarCloud treats `if` / `else if` chains, nested matches, and early-return-with-cleanup patterns as **additional** complexity, so those rarely bring a function under 15 without also extracting helpers.
