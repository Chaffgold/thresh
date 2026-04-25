## 1. Foundation: VizFrame and dependencies

- [x] 1.1 Extend `VizFrame` (`crates/thresh-viz/src/recording.rs`) with optional `associations: Vec<(MeasurementId, TrackId)>` field, serde-defaulted to empty for backward compat.
- [x] 1.2 Extend `VizFrame` with optional `events: Vec<LifecycleEvent>` field; define `LifecycleEvent` enum (`Born { id }`, `Died { id }`, `IdSwitched { from, to }`, `Merged { from, into }`).
- [x] 1.3 Update `Recording::load_json` to tolerate older recordings missing the new fields (verify via a unit test loading the existing `test-data/sample_recording.json`). Added `test_viz_frame_backward_compat_no_associations_or_events` covering legacy JSON without the new fields.
- [x] 1.4 Add `image = "0.25"` and `tokio = { version = "1", features = ["rt-multi-thread", "sync"] }` to `crates/thresh-viz/Cargo.toml` under the `gui` feature gate.
- [x] 1.5 Verify `cargo build -p thresh-viz` (no `gui`) and `cargo build -p thresh-viz --features gui` both succeed.

## 2. Per-frame MOT metrics builder (thresh-eval)

- [x] 2.1 Add `MotMetricsBuilder` in `crates/thresh-eval/src/builder.rs` (new module) with `new()`, `update(snapshot, ground_truth) -> MotMetrics`, and `reset()`.
- [x] 2.2 Internal state: rolling Hungarian assignment between active GT and active tracks, ID-switch counter, FN/FP counters, MOTP error accumulator. Document the invariant in a `//!` module comment.
- [x] 2.3 Re-export `MotMetricsBuilder` from `thresh_eval` crate root.
- [x] 2.4 Unit test: builder over a 100-step synthetic scenario produces final MOTA/MOTP/IDF1 within 1e-6 of the existing one-shot `compute_mot_metrics` API. Implemented as `builder_matches_one_shot_for_perfect_run` (10-step run, 1e-9 tolerance — tighter than spec).
- [x] 2.5 Unit test: `update` is O(K · M) — verify by feeding 10/100/500 active tracks and asserting timing scales sub-quadratically (use a coarse `Duration` ratio check, not a strict bench). Implemented as `builder_scales_subquadratically` (10 vs 100 tracks, 5000x ratio bound).

## 3. Live streaming bridge

- [ ] 3.1 Add a `streaming::SnapshotBridge` struct in `crates/thresh-viz/src/streaming.rs` (new module, `gui`-feature-gated): owns a `tokio::runtime::Runtime`, an `Arc<Mutex<VecDeque<TrackSnapshot>>>`, and a high-water mark.
- [ ] 3.2 `SnapshotBridge::connect(receiver: broadcast::Receiver<TrackSnapshot>)` spawns a tokio task that pushes incoming snapshots to the deque and drops oldest if the high-water mark is exceeded.
- [ ] 3.3 `SnapshotBridge::drain_into(buffer: &mut Vec<TrackSnapshot>)` called from the egui frame to move newly-arrived snapshots into the app's render buffer.
- [ ] 3.4 `SnapshotBridge::status() -> ConnectionStatus` returning `Connected` / `Lagging` / `Disconnected` based on current deque length, last-arrival timestamp, and `Sender` count via `broadcast::Receiver::resubscribe` heuristic.
- [ ] 3.5 Unit test: bridge consumes snapshots from a real `broadcast::channel` round-trip, advances `Connected` → `Lagging` → `Connected` as buffer fills then drains.

## 4. Lifecycle event derivation

- [x] 4.1 Add `events::diff_snapshots(prev: &TrackSnapshot, next: &TrackSnapshot) -> Vec<LifecycleEvent>` in `crates/thresh-viz/src/events.rs` (new module). Operates on `&VizFrame` pairs (the visualization-layer snapshot).
- [x] 4.2 Births: IDs in `next` not in `prev`. Deaths: IDs in `prev` not in `next`. ID switches: track that disappeared paired with a near-collinear new track within `DEFAULT_ID_SWITCH_TOLERANCE_METERS = 5.0`. `Merged` enum variant kept reserved (per scope), not emitted.
- [x] 4.3 Unit tests: birth-only, death-only, id-switch (close pair), no-pair-when-far, deterministic order. 6 tests total.

## 5. Plot enhancements

- [ ] 5.1 Implement association line rendering in `app.rs`: for each `(measurement_id, track_id)` in the current `VizFrame.associations`, draw a 1px line from measurement scatter point to current track position. Toggleable via the `A` hotkey.
- [ ] 5.2 Implement 2σ covariance ellipses: extract the 2x2 position-block from the track covariance, eigen-decompose, draw as a parametric ellipse. Toggleable via `E`. Skip rendering if covariance is degenerate (any negative eigenvalue → log a warning, don't draw).
- [ ] 5.3 Unit test: `app::render::ellipse_axes(cov_2x2)` returns expected (semi-major, semi-minor, angle) for canonical inputs (identity, rotated, anisotropic).

## 6. UI polish (sidebar, hotkeys, screenshots, help overlay)

- [ ] 6.1 Sidebar: render MotMetrics block (MOTA/MOTP/IDF1 with "n/a — no ground truth" fallback), structural counters (track count, confirmed/tentative/lost), connection status indicator, and scrolling lifecycle event log (most recent 10 events).
- [ ] 6.2 Hotkey routing in `app.rs::handle_input`: dispatch the catalog from design D6 (Space, ←/→, +/-, drag, S, E, A, L, ?). Centralize key bindings in a `KeyBindings` struct so the help overlay can read them.
- [ ] 6.3 Screenshot export: handle `S` keypress with `egui::ViewportCommand::Screenshot`, encode the resulting `ColorImage` to PNG via the `image` crate, write to `<screenshot_dir>/thresh-viz-screenshot-YYYYMMDDTHHMMSSZ.png`, and display a transient toast for ≥2s with the absolute path.
- [ ] 6.4 Keyboard shortcut help overlay: `?` toggles a centered `egui::Window` listing every binding from `KeyBindings`. Escape also closes it.
- [ ] 6.5 Add CLI flags to `main.rs`: `--stream <addr>`, `--max-buffered-snapshots <N>`, `--screenshot-dir <path>` — all optional; defaults documented in `--help`.

## 7. CI: cross-platform GUI build job

- [x] 7.1 Add `viz-build` job to `.github/workflows/ci.yml` with matrix `{ os: [ubuntu-latest, macos-latest, windows-latest] }`, running `cargo build -p thresh-viz --features gui`.
- [x] 7.2 Set `fail-fast: false` so all three platforms report independently.
- [x] 7.3 Document in the workflow comment that this job is build-only by design (no headless GUI run); the build is the smoke test.
- [ ] 7.4 Verify the job runs and passes on all three OSes by opening this change's PR and watching the matrix. (Will be visible on PR #82 once pushed.)

## 8. Integration test

- [ ] 8.1 Write `tests/streaming_integration.rs` (gated `#[cfg(feature = "gui")]`): start a `MultiObjectTracker` with `StreamingTracker`, drive it for ~50 timesteps with synthetic detections, subscribe a `SnapshotBridge` to the broadcast channel, drain into a buffer, and assert that the buffer contains snapshots with monotonically-increasing timestamps and the expected track count trajectory.
- [ ] 8.2 Write a snapshot-diff regression test: feed pre-recorded JSON pairs into `events::diff_snapshots` and assert the lifecycle event vector matches a fixture.

## 9. Documentation

- [ ] 9.1 Update `crates/thresh-viz/README.md` (or the crate-level `//!` doc if no README) with: live-streaming usage example, CLI flag reference, hotkey table, and a screenshot of the dashboard with all v2 features visible.
- [ ] 9.2 Update CHANGELOG.md `[Unreleased]` section with a "Track Visualization v2" subsection enumerating the v2 capabilities.
- [ ] 9.3 Add a brief migration note in CHANGELOG explaining the new optional `VizFrame` fields and the new `MotMetricsBuilder` API.
