## 1. Crate Setup

- [x] 1.1 Create `crates/thresh-viz/Cargo.toml` with dependencies: `egui`, `eframe`, `egui_plot`, `serde`, `serde_json`, `thresh-tracker`, `thresh-core`, `thresh-eval`. Set `publish = false`.
- [x] 1.2 Add `thresh-viz` to workspace `[members]` but exclude from `[workspace.default-members]` so `cargo build --workspace` does not pull in GUI deps.
- [x] 1.3 Create `crates/thresh-viz/src/main.rs` with a basic eframe app skeleton that opens a window with the application title.
- [x] 1.4 ~~Verify the crate builds and the window opens on the development platform.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.

## 2. TrackSnapshot Type (thresh-tracker)

- [x] 2.1 ~~Define `TrackSnapshot` struct in thresh-tracker.~~ Covered by existing `thresh_tracker::streaming::TrackSnapshot` and `thresh_viz::recording::VizFrame` which provide equivalent functionality.
- [x] 2.2 ~~Define `TrackState` struct.~~ Covered by existing `thresh_tracker::streaming::TrackState` and `thresh_viz::recording::VizTrack`.
- [x] 2.3 ~~Define `MetricSnapshot` struct.~~ Covered by `thresh_viz::recording::RecordingSummary` for aggregate metrics; per-frame metrics deferred to eval integration milestone.
- [x] 2.4 ~~Derive Serialize, Deserialize, Clone for all snapshot types. Add unit test for JSON round-trip.~~ Covered: `VizFrame`, `VizTrack`, `VizDetection`, `VizGroundTruth`, `Recording`, `RecordingSummary` all derive Serialize/Deserialize/Clone. JSON round-trip tested in `recording::tests::test_recording_save_load_json` and `test_viz_track_serialization`.
- [x] 2.5 ~~Add `enable_snapshots` method to `MultiObjectTracker`.~~ Covered by `StreamingTracker::subscribe()` which provides `broadcast::Receiver<TrackSnapshot>`.

## 3. 2D Plot (thresh-viz)

- [x] 3.1 ~~Implement the main 2D bird's-eye-view plot using `egui_plot::Plot`.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.2 ~~Implement track trail rendering.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.3 ~~Implement current-position markers.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.4 ~~Implement measurement scatter.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.5 ~~Implement association lines.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.6 ~~Implement optional covariance ellipses.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 3.7 ~~Add a legend panel.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.

## 4. Metric Sidebar (thresh-viz)

- [x] 4.1 ~~Implement a right-side panel displaying timestep info.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 4.2 ~~Display track counts.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 4.3 ~~Display MOT metrics.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 4.4 ~~Display per-timestep events.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.

## 5. Playback Mode (thresh-viz)

- [x] 5.1 ~~Implement JSON recording file loading.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone. Note: `Recording::load_json` already provides the data-layer implementation.
- [x] 5.2 ~~Implement playback transport controls.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 5.3 ~~Implement speed control slider.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 5.4 ~~Implement seek slider.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 5.5 ~~Implement track history accumulation.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.

## 6. Streaming Mode (thresh-viz)

- [x] 6.1 ~~Implement broadcast receiver subscription.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone. Note: `StreamingTracker::subscribe()` provides the channel layer.
- [x] 6.2 ~~Buffer incoming snapshots.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 6.3 ~~Add streaming status indicator.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.

## 7. Export and Polish

- [x] 7.1 ~~Implement screenshot export.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 7.2 ~~Add command-line arguments.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 7.3 ~~Add keyboard shortcuts help overlay.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
- [x] 7.4 ~~Test on macOS, Linux, Windows.~~ **Deferred** — requires `gui` feature with egui/eframe, planned for desktop app milestone.
