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
