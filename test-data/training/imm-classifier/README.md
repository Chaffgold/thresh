# Synthetic IMM training fixture

`imm-samples.parquet` contains only built-in synthetic cruise and maneuver
trajectories generated through radar synthesis and the analytic IMM tracker.
No OpenSky, ADSBx, or other external captures are used.

Regenerate from the repository root:

```sh
cargo run -p thresh --features training-export --bin gen-imm-dataset -- \
  test-data/training/imm-classifier/imm-samples.parquet
```

Decision 30's elapsed-time schema (2026-09-22) stores 13 floats per `feature`:
`[x, vx, y, vy, z, vz, Pxx, Pvxvx, Pyy, Pvyvy, Pzz, Pvzvz, elapsed_seconds]`.
Rows are actual post-birth measurement updates, including tentative tracks;
birth initialization and prediction-only ticks are omitted. The first row for
each track has elapsed time zero. Later intervals equal consecutive `time_s`
differences, including gaps from missed updates. Python checks this before
constructing sliding windows; windows preserve their first row's interval.

The former 12-wide, confirmed/coasting-inclusive sample is incompatible and
must be regenerated, not silently assigned a rate. This fixture proves schema
and timing behavior, not representative classifier accuracy or release gates.
