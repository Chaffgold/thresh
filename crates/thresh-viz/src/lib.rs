//! Track visualization dashboard for the thresh tracking framework.
//!
//! The data layer ([`recording`], [`events`], [`geom`]) works without
//! any GUI dependencies. Enable the `gui` feature to get the
//! interactive egui application and the live [`streaming`] bridge.
//!
//! # Usage
//!
//! ## Offline playback (default)
//!
//! ```sh
//! cargo run -p thresh-viz --features gui -- --recording session.json
//! ```
//!
//! ## Live streaming from a `MultiObjectTracker`
//!
//! ```ignore
//! use thresh_tracker::{streaming::{StreamingTracker, StreamingConfig}, tracker::MultiObjectTracker};
//! use thresh_viz::{app::ThreshVizApp, streaming::SnapshotBridge};
//!
//! let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
//! let (streaming, _handle) = StreamingTracker::new(tracker, StreamingConfig::default());
//! let bridge = SnapshotBridge::new()?;
//! bridge.connect(streaming.subscribe());
//!
//! // Drive the streaming tracker from your detection source via streaming.sender().
//! // Hand the bridge to the app and run eframe:
//! let app = ThreshVizApp::live(bridge);
//! eframe::run_native("thresh-viz", Default::default(), Box::new(|_cc| Ok(Box::new(app))))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # CLI flags
//!
//! | Flag | Description |
//! |---|---|
//! | `--recording <FILE.json>` | Load a recorded session for offline playback |
//! | `--stream <ADDR>` | (Reserved for a future cross-process bridge) |
//! | `--max-buffered-snapshots <N>` | Bridge high-water mark (default 64) |
//! | `--screenshot-dir <PATH>` | Where to write PNG screenshots (default cwd) |
//! | `-h`, `--help` | Print help |
//!
//! # Hotkeys
//!
//! | Key | Action |
//! |---|---|
//! | `Space` | Play / pause |
//! | `←` / `→` | Step backward / forward |
//! | `+` / `-` | Zoom in / out (or scroll wheel) |
//! | drag | Pan |
//! | `S` | Screenshot to PNG (`thresh-viz-screenshot-<UTC-ISO8601>.png`) |
//! | `A` | Toggle association lines |
//! | `E` | Toggle covariance ellipses (2σ) |
//! | `L` | Toggle lifecycle event log |
//! | `?` | Toggle help overlay (also dismissed by `Esc`) |
//!
//! # Live streaming connection states
//!
//! When connected to a `StreamingTracker`, the sidebar shows:
//!
//! - **Connected** — snapshots arriving within the latency budget
//! - **Lagging** — buffer exceeded high-water mark; oldest snapshots dropped
//! - **Disconnected** — broadcast sender dropped or no snapshot in 2 s
//!
//! # Per-frame MOT metrics
//!
//! When ground truth is available, the sidebar displays running
//! MOTA / MOTP / IDF1 / ID switches via [`thresh_eval::MotMetricsBuilder`].
//! Without ground truth, the GT-dependent fields display
//! `n/a — no ground truth` while the structural counters
//! (track count, confirmed/tentative/lost, lifecycle events) keep
//! updating.

pub mod events;
pub mod geom;
pub mod recording;

#[cfg(feature = "gui")]
pub mod app;

#[cfg(feature = "gui")]
pub mod streaming;
