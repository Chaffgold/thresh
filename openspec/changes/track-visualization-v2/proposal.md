# Track Visualization Dashboard — v2 (Desktop App Milestone)

## Why

The original `track-visualization` change (archived 2026-04-25) shipped the `thresh-viz` egui crate with a 2D bird's-eye plot, JSON recording playback, and a metric sidebar — but **eight items were explicitly deferred to a "desktop app milestone"** because they required either the `gui` feature integration, additional `VizFrame` data, or thresh-eval glue that wasn't ready. The result is a viewer that is useful for offline JSON playback but cannot drive a real tracking session: no live streaming, no per-frame metrics, no association lines, no covariance, no screenshots, no cross-platform CI.

This change closes those eight items so `thresh-viz` becomes the production-ready dashboard the v1 design called for.

## What Changes

- **Live streaming integration** — egui app subscribes to `StreamingTracker::subscribe()` (broadcast channel of `TrackSnapshot`), buffers incoming snapshots, renders them in real time, and shows a streaming connection status indicator (connected / lagging / disconnected).
- **Per-frame MOT metrics in the sidebar** — wire `thresh-eval` per-frame MOTA / MOTP / IDF1 computation through the dashboard. Running values update each timestep instead of only at the end of a recording.
- **Track event log** — surface track births, deaths, ID switches, and merges as a scrolling event list pinned to the metric sidebar, sourced from tracker lifecycle events.
- **Association lines** — draw lines from current-timestep measurements to their assigned tracks. Requires extending `VizFrame` with a per-frame association map.
- **Covariance ellipses** — render optional 2σ position-covariance ellipses per track, using the diagonal covariance already exposed in `VizFrame`. Toggleable via the keyboard shortcut overlay.
- **Screenshot export** — PNG export of the current viewport via hotkey (`S`) and menu option. Filename includes ISO-8601 timestamp.
- **Keyboard shortcuts overlay** — toggleable help panel listing all hotkeys (pan, zoom, screenshot, pause, ellipse toggle, etc.).
- **Cross-platform GUI build CI** — new CI job builds `thresh-viz` on Ubuntu, macOS, and Windows runners. Build is the smoke test (no headless run required).

## Capabilities

### New Capabilities
<!-- None — this change extends the existing track-dashboard capability. -->

### Modified Capabilities
- `track-dashboard`: adds requirements for live streaming subscription, per-frame metric updates, track lifecycle event log, association line rendering, covariance ellipses, screenshot export, keyboard shortcuts help, and cross-platform GUI build verification.

## Impact

- **Code:**
  - `crates/thresh-viz/src/app.rs` — egui app struct, frame loop, sidebar rendering, hotkey routing, screenshot export
  - `crates/thresh-viz/src/recording.rs` — extend `VizFrame` with per-frame associations and track lifecycle events
  - `crates/thresh-viz/Cargo.toml` — add `image` (PNG encoding) and `tokio` (broadcast subscription bridge) deps under the `gui` feature gate
  - `crates/thresh-tracker/src/streaming.rs` — `StreamingTracker::subscribe()` already exists; consume it from the GUI side
  - `crates/thresh-eval/` — expose a per-frame metrics builder API so the dashboard can compute MOTA/MOTP/IDF1 incrementally rather than only end-of-run
- **CI:**
  - `.github/workflows/ci.yml` — new `viz-build` job matrix on `ubuntu-latest`, `macos-latest`, `windows-latest` running `cargo build -p thresh-viz --features gui`
- **Out of scope (deferred to a future v3 change):**
  - Optional 3D orbit view
  - Full video / GIF recording (only static screenshots in v2)
  - Advanced association-graph visualizations beyond simple lines (e.g., probability-weighted JPDA edges)
  - Multi-tracker comparison view (two trackers side-by-side)
