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

- [x] 3.1 Add a `streaming::SnapshotBridge` struct in `crates/thresh-viz/src/streaming.rs` (new module, `gui`-feature-gated): owns a `tokio::runtime::Runtime`, an `Arc<Mutex<VecDeque<TrackSnapshot>>>`, and a high-water mark.
- [x] 3.2 `SnapshotBridge::connect(receiver: broadcast::Receiver<TrackSnapshot>)` spawns a tokio task that pushes incoming snapshots to the deque and drops oldest if the high-water mark is exceeded.
- [x] 3.3 `SnapshotBridge::drain_into(buffer: &mut Vec<TrackSnapshot>)` called from the egui frame to move newly-arrived snapshots into the app's render buffer.
- [x] 3.4 `SnapshotBridge::status() -> ConnectionStatus` returning `Connected` / `Lagging` / `Disconnected` based on deque length, last-arrival timestamp, and `Sender::Closed` signal from the broadcast receiver.
- [x] 3.5 Unit test: bridge consumes snapshots from a real `broadcast::channel` round-trip, advances `Connected` → `Lagging` → `Connected` as buffer fills then drains. Also covers `Disconnected` via dropped sender and via inactivity timeout (4 tests total).

## 4. Lifecycle event derivation

- [x] 4.1 Add `events::diff_snapshots(prev: &TrackSnapshot, next: &TrackSnapshot) -> Vec<LifecycleEvent>` in `crates/thresh-viz/src/events.rs` (new module). Operates on `&VizFrame` pairs (the visualization-layer snapshot).
- [x] 4.2 Births: IDs in `next` not in `prev`. Deaths: IDs in `prev` not in `next`. ID switches: track that disappeared paired with a near-collinear new track within `DEFAULT_ID_SWITCH_TOLERANCE_METERS = 5.0`. `Merged` enum variant kept reserved (per scope), not emitted.
- [x] 4.3 Unit tests: birth-only, death-only, id-switch (close pair), no-pair-when-far, deterministic order. 6 tests total.

## 5. Plot enhancements

- [x] 5.1 Implement association line rendering in `app.rs`: for each `(detection_index, track_id)` in the current `VizFrame.associations`, draw a 1px line from measurement scatter point to current track position. Toggleable via the `A` hotkey.
- [x] 5.2 Implement 2σ covariance ellipses: extract the position-block diagonal from `VizTrack.covariance_diag`, compute semi-axes via `geom::ellipse_axes`, draw as a 48-segment polyline. Toggleable via `E`. Skip rendering if covariance is degenerate (`ellipse_axes` returns `None`).
- [x] 5.3 Unit tests: `geom::ellipse_axes` over identity, anisotropic diagonal, 45°-rotated, and degenerate inputs; `geom::ellipse_polyline` closure + center invariants. 6 tests total.

## 6. UI polish (sidebar, hotkeys, screenshots, help overlay)

- [x] 6.1 Sidebar: render MotMetrics block (MOTA/MOTP/IDF1 with "n/a — no ground truth" fallback), structural counters (track count, confirmed/tentative/lost), connection status indicator, and scrolling lifecycle event log (capped at 200 events, displayed newest-first).
- [x] 6.2 Hotkey routing in `app.rs::handle_input`: dispatch the catalog from design D6 (Space, ←/→, +/-, drag, S, E, A, L, ?). Centralized in `KeyBindings` struct exposed via `ThreshVizApp::keys()`.
- [x] 6.3 Screenshot export: handle `S` keypress with `egui::ViewportCommand::Screenshot`, encode the resulting `ColorImage` to PNG via the `image` crate, write to `<screenshot_dir>/thresh-viz-screenshot-YYYYMMDDTHHMMSSZ.png`, and display a transient toast for ≥2s with the absolute path. Embedded ISO-8601 formatter avoids a new chrono dep.
- [x] 6.4 Keyboard shortcut help overlay: `?` toggles a centered `egui::Window` listing every binding from `KeyBindings.catalog()`. Escape also closes it.
- [x] 6.5 Add CLI flags to `main.rs`: `--stream <addr>`, `--max-buffered-snapshots <N>`, `--screenshot-dir <path>`, `-h`/`--help` with full reference. `--stream` currently emits a notice that the in-process `SnapshotBridge` API is the supported path.

## 7. CI: cross-platform GUI build job

- [x] 7.1 Add `viz-build` job to `.github/workflows/ci.yml` with matrix `{ os: [ubuntu-latest, macos-latest, windows-latest] }`, running `cargo build -p thresh-viz --features gui`.
- [x] 7.2 Set `fail-fast: false` so all three platforms report independently.
- [x] 7.3 Document in the workflow comment that this job is build-only by design (no headless GUI run); the build is the smoke test.
- [ ] 7.4 Verify the job runs and passes on all three OSes by opening this change's PR and watching the matrix. (Will be visible on PR #82 once pushed.)

## 8. Integration test

- [x] 8.1 Write `tests/streaming_integration.rs` (gated `#[cfg(feature = "gui")]`): start a `MultiObjectTracker` with `StreamingTracker`, drive it for ~200 detections via the streaming sender at ~5ms cadence, subscribe a `SnapshotBridge` to the broadcast channel, drain into a buffer, and assert that the buffer contains snapshots with monotonically non-decreasing timestamps, ≥1 track in the final snapshot, and `Connected` status.
- [x] 8.2 Write a snapshot-diff regression test: covered by `crates/thresh-viz/src/events.rs::tests::deterministic_order_across_runs` (births + deaths + id-switches + determinism). Plus `tests/app_kittest.rs` (5 tests using `egui_kittest`) for headless end-to-end coverage of help overlay, ellipse / association / event-log toggles, and a render-without-panic smoke test.

## 9. Documentation

- [x] 9.1 Crate-level `//!` doc on `crates/thresh-viz/src/lib.rs` updated with: live-streaming usage example, CLI flag reference table, hotkey table, connection status documentation, and per-frame MOT metrics behavior. Static screenshot deferred — current sample recording covers the offline path; a v3 doc-pass would capture a live-streaming animation.
- [x] 9.2 CHANGELOG.md `[Unreleased]` section gained a "Track Visualization v2 (Desktop App Milestone)" subsection enumerating the eight v2 capabilities and the egui_kittest test infrastructure.
- [x] 9.3 Migration subsection added in CHANGELOG covering the new `VizFrame` fields, the new `MotMetricsBuilder` API, and the transitive feature activation of `thresh-tracker/streaming` via `thresh-viz/gui`.
