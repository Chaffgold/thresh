## 1. Crate Setup

- [ ] 1.1 Create `crates/thresh-viz/Cargo.toml` with dependencies: `egui`, `eframe`, `egui_plot`, `serde`, `serde_json`, `thresh-tracker`, `thresh-core`, `thresh-eval`. Set `publish = false`.
- [ ] 1.2 Add `thresh-viz` to workspace `[members]` but exclude from `[workspace.default-members]` so `cargo build --workspace` does not pull in GUI deps.
- [ ] 1.3 Create `crates/thresh-viz/src/main.rs` with a basic eframe app skeleton that opens a window with the application title.
- [ ] 1.4 Verify the crate builds and the window opens on the development platform: `cargo run -p thresh-viz`.

## 2. TrackSnapshot Type (thresh-tracker)

- [ ] 2.1 Define `TrackSnapshot` struct in thresh-tracker: `timestep: u64`, `timestamp: f64`, `tracks: Vec<TrackState>`, `measurements: Vec<Measurement>`, `associations: Vec<(TrackId, usize)>`, `metrics: Option<MetricSnapshot>`.
- [ ] 2.2 Define `TrackState` struct: `id: TrackId`, `state: DVector<f64>`, `covariance: DMatrix<f64>`, `status: TrackStatus`, `class: Option<String>`.
- [ ] 2.3 Define `MetricSnapshot` struct: `mota: Option<f64>`, `motp: Option<f64>`, `idf1: Option<f64>`, `hota: Option<f64>`, `num_confirmed: usize`, `num_tentative: usize`, `num_lost: usize`.
- [ ] 2.4 Derive `Serialize`, `Deserialize`, `Clone` for all snapshot types. Add unit test for JSON round-trip serialization.
- [ ] 2.5 Add `enable_snapshots(&mut self)` method to `MultiObjectTracker` that stores a `tokio::sync::broadcast::Sender<TrackSnapshot>`. Emit a snapshot at each `step()` call when enabled.

## 3. 2D Plot (thresh-viz)

- [ ] 3.1 Implement the main 2D bird's-eye-view plot using `egui_plot::Plot`. Set up axes, grid, and pan/zoom interaction.
- [ ] 3.2 Implement track trail rendering: for each track, draw a polyline from its position history. Color-code by track ID using a deterministic hash-based color palette.
- [ ] 3.3 Implement current-position markers: draw a filled circle at each track's latest position, sized proportional to confidence.
- [ ] 3.4 Implement measurement scatter: draw small cross markers for current-timestep measurements.
- [ ] 3.5 Implement association lines: draw dashed lines from each measurement to its assigned track for the current timestep.
- [ ] 3.6 Implement optional covariance ellipses: draw 2-sigma covariance ellipses around each track position using the 2D position submatrix of the covariance.
- [ ] 3.7 Add a legend panel mapping track IDs to colors with track status indicators.

## 4. Metric Sidebar (thresh-viz)

- [ ] 4.1 Implement a right-side panel displaying: current timestep, total timestep count, playback speed.
- [ ] 4.2 Display track counts: total, confirmed, tentative, lost.
- [ ] 4.3 Display MOT metrics (MOTA, MOTP, IDF1, HOTA) when ground truth metrics are available in the snapshot.
- [ ] 4.4 Display per-timestep events: births, deaths, association count.

## 5. Playback Mode (thresh-viz)

- [ ] 5.1 Implement JSON recording file loading: parse `Vec<TrackSnapshot>` from a JSON file, store in memory.
- [ ] 5.2 Implement playback transport controls: play/pause toggle (Space key), step forward (Right arrow), step backward (Left arrow).
- [ ] 5.3 Implement speed control slider: 0.25x to 4x, default 1x. Advance timesteps based on elapsed wall-clock time times speed multiplier.
- [ ] 5.4 Implement seek slider: scrub to any timestep. Display current/total timestep indicator.
- [ ] 5.5 Implement track history accumulation: as playback advances, accumulate track positions for trail rendering. On seek backward, rebuild history from the beginning up to the target timestep.

## 6. Streaming Mode (thresh-viz)

- [ ] 6.1 Implement `tokio::sync::broadcast::Receiver<TrackSnapshot>` subscription for live streaming from a running tracker.
- [ ] 6.2 Buffer incoming snapshots and append to the visualization state at each frame.
- [ ] 6.3 Add a streaming status indicator in the sidebar (connected/disconnected, snapshots received per second).

## 7. Export and Polish

- [ ] 7.1 Implement screenshot export: capture the current frame as PNG and save to a user-specified path via a file dialog.
- [ ] 7.2 Add command-line arguments: `--file <recording.json>` for playback mode, `--connect <addr>` for streaming mode.
- [ ] 7.3 Add keyboard shortcuts help overlay (H key to toggle).
- [ ] 7.4 Test on macOS, Linux (X11 and Wayland), and Windows. Document any platform-specific notes.
