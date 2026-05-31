# thresh

Umbrella crate that re-exports the thresh workspace (`core`, `filter`,
`association`, `fusion`, `tracker`, `synth`, `inference`, `eval`, `bridge`) and
hosts the cross-crate integration tests and training-data tooling.

## `training-export` feature (off by default)

Adds the `arrow` / `parquet` dependencies and the Rust-native Parquet writers in
`training::parquet_export`, plus two binaries that bridge the synth pairing into
training datasets for the `flight-data-training-pipeline`:

```sh
# Track B — IMM filter-state features + analytic mode labels:
cargo run -p thresh --features training-export --bin gen-imm-dataset -- <out.parquet>

# Track A — point-cloud / 3D-box / class detector snapshots:
cargo run -p thresh --features training-export --bin gen-detector-dataset -- <out.parquet>
```

`training::generate_imm_training_samples` runs synth measurements → analytic IMM
tracker → 12-dim filter-state features; `training::detector::generate_detector_samples`
maps the synth `RadarSnapshot`s to detector samples. The Python training tree
under `python/` consumes the emitted Parquet (see `TRAINING.md`). The
`arrow`/`parquet` stack is heavy, hence the feature gate — it stays out of
default and CI builds.

## `stonesoup` feature

Enables the PyO3 → Stone Soup bridge via `thresh-bridge`.
