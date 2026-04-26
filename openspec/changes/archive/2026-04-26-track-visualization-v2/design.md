## Context

The `track-visualization` v1 change shipped `crates/thresh-viz` as an `egui` + `eframe` desktop app with a 2D bird's-eye plot, JSON recording playback, and a metric sidebar. Eight items were deferred to a "desktop app milestone" because they required either thresh-eval glue, additional `VizFrame` data, or `tokio` runtime integration that wasn't ready in v1. This change closes those items.

The runtime architecture from v1 is intentionally simple: a single `eframe::App` impl driven by egui's immediate-mode loop. v1 reads from a JSON `Recording` file. v2 must additionally consume a live `tokio::sync::broadcast` channel from `StreamingTracker` without blocking the UI thread, and must compute per-frame MOT metrics on the fly.

`thresh-viz` is workspace-member-but-not-default-member (per the v0.3.0 release notes). v2 keeps that property — the `gui` feature gate stays the gate for everything that pulls in `egui`/`eframe`, so headless workspace builds remain fast.

## Goals / Non-Goals

**Goals:**
- Real-time consumption of a `StreamingTracker` broadcast feed in the GUI without UI-thread blocking.
- Per-frame MOT metrics (MOTA, MOTP, IDF1) updated each timestep, not only at end-of-recording.
- Track lifecycle event log (births / deaths / ID switches / merges) visible alongside metrics.
- Association lines and 2σ covariance ellipses on the 2D plot.
- PNG screenshot export with timestamped filenames.
- Toggleable keyboard-shortcut help overlay.
- CI confirmation that `thresh-viz --features gui` builds on Linux, macOS, and Windows.

**Non-Goals:**
- 3D orbit view (deferred to v3).
- Video / GIF recording — only static PNGs in v2.
- Probability-weighted association edges for JPDA / MHT — simple lines only in v2.
- Multi-tracker side-by-side comparison.
- Headless GUI rendering tests in CI — the build itself is the smoke test.

## Decisions

### D1. Bridging the broadcast channel into egui

`StreamingTracker::subscribe()` returns a `tokio::sync::broadcast::Receiver<TrackSnapshot>`. egui's update loop runs on the UI thread and is not async. Two options:

- **(A)** Spawn a background tokio task that drains the receiver into an `Arc<Mutex<VecDeque<TrackSnapshot>>>` shared with the egui app. The app pulls newly-arrived snapshots from the deque each frame.
- **(B)** Block the UI thread on `try_recv` each frame.

**Choice: (A).** Option (B) couples render frame rate to broadcast cadence and would lose snapshots whenever the GUI is paused. (A) decouples ingest from render and gives us a natural place to track lag — if the deque grows beyond a configured high-water mark, we mark the connection "lagging" and drop the oldest snapshots. This mirrors the `DropPolicy::DropOldest` policy already used in `StreamingTracker`.

The background task lives in a `tokio::runtime::Runtime` that the GUI builds at startup (single multi-thread runtime, default settings). `eframe::run_native` already cohabits cleanly with a tokio runtime when the user constructs it themselves.

### D2. Per-frame MOT metrics — incremental computation

`thresh-eval` today computes MOTA / MOTP / IDF1 over a complete trajectory. v2 needs incremental updates per timestep. Two options:

- **(A)** Recompute the full metric over the snapshot history each frame. Trivial to implement, O(N²) per frame in track count.
- **(B)** Add a stateful per-frame builder API to thresh-eval that maintains the GT-to-track Hungarian assignment, ID-switch counter, and FN/FP totals incrementally.

**Choice: (B).** (A) becomes a measurable load above ~50 tracks at 30 FPS, and the incremental pattern is what live tracking systems actually need. The API addition is small: `MotMetricsBuilder::new(...).update(snapshot, ground_truth) -> MotMetrics`.

If ground truth is unavailable (live streaming with no GT feed), we display only structural counters: track-count, confirmed/tentative/lost breakdown, ID-switch rate. The MOT metrics block becomes "n/a — no ground truth."

### D3. Lifecycle event source

`MultiObjectTracker` does not currently emit explicit "track born" / "track died" events — that information is implicit in the diff between consecutive snapshots. Two options:

- **(A)** Diff snapshots GUI-side: compare the set of track IDs between snapshot N and N+1.
- **(B)** Add an explicit `TrackEvent` enum to `thresh-tracker` and broadcast it on a side channel.

**Choice: (A) for v2, (B) flagged as future work.** Option (A) avoids invasive changes to thresh-tracker for an essentially cosmetic feature. We get births (new ID), deaths (ID disappears), and ID switches (gross GT-to-track reassignment) from snapshot diffs. (B) is cleaner architecturally — call it out in `## Open Questions` so a future change can revisit if downstream consumers (logs, REST API, etc.) want the same events.

### D4. Screenshot export

egui's `Context::request_repaint` interacts awkwardly with eframe's frame buffer access. Two paths:

- **(A)** Use `eframe::Frame::request_screenshot` (calls native screenshot API).
- **(B)** Use `egui::ViewportCommand::Screenshot` (newer, returns the framebuffer to the app for arbitrary processing).

**Choice: (B)** — the framebuffer comes back as `egui::ColorImage`, which we can encode to PNG via the `image` crate. Filename: `thresh-viz-screenshot-YYYYMMDDTHHMMSSZ.png` in the current working directory by default, configurable via a `--screenshot-dir` CLI flag.

### D5. Cross-platform CI strategy

Adding macOS + Windows runners to the existing `Test` job would multiply CI cost ~3x for every PR. Instead, add a focused `viz-build` job:

- Matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`
- Action: `cargo build -p thresh-viz --features gui`
- No tests run — just verify the GUI feature graph compiles. egui/eframe link errors are platform-specific and what we want to catch.

This keeps PR cost low (build-only on three OSes is cheap) while catching `thresh-viz` regressions before they hit develop.

### D6. Keyboard shortcut catalog

Bind in v2 (all toggleable in the help overlay):

| Key | Action |
|---|---|
| `Space` | Pause / resume playback or live feed |
| `→` / `←` | Step forward / backward (recordings only) |
| `+` / `-` | Zoom in / out |
| Drag | Pan |
| `S` | Screenshot to PNG |
| `E` | Toggle covariance ellipses |
| `A` | Toggle association lines |
| `L` | Toggle lifecycle event log panel |
| `?` | Toggle keyboard shortcuts overlay |

## Risks / Trade-offs

- **[Risk] tokio runtime + eframe interaction.** Some `eframe` versions have raised issues when a tokio runtime is constructed before `run_native`. → Mitigation: build the runtime, spawn the broadcast drainer onto it, then pass an `Arc<Runtime>` into the `App` so the runtime stays alive for the GUI's lifetime; do not call `Runtime::block_on` from the UI thread.
- **[Risk] Snapshot deque high-water mark default.** Set too low → drop snapshots even on a slightly stuttering GUI; too high → unbounded memory growth on a wedged GUI. → Mitigation: default to 64 snapshots (~2 seconds at 30 FPS), expose `--max-buffered-snapshots` CLI flag.
- **[Risk] Snapshot-diff lifecycle events miss simultaneous birth + death of the same ID.** ID reuse within a single timestep gap would look like nothing happened. → Mitigation: track the union of all IDs ever seen, never reuse — `MultiObjectTracker` already does this, so the risk is theoretical for in-tree trackers but exists for external consumers.
- **[Risk] `image` crate adds non-trivial compile time.** → Mitigation: gated behind the `gui` feature, only built when `thresh-viz` is built. Acceptable since `thresh-viz` already pulls in `egui` (bigger).
- **[Risk] macOS / Windows CI runner availability.** GitHub-hosted macos-latest runners can have minute-scale queue times during peak hours. → Mitigation: keep the job minimal (build-only, no tests); set `fail-fast: false` so a Windows queue spike doesn't fail the macOS leg.
- **[Risk] Per-frame metric recompute even with builder pattern.** If the snapshot history grows large, even the builder needs bounded state. → Mitigation: `MotMetricsBuilder` is a rolling-window structure — keeps only the assignment state and counters, not the full snapshot history.

## Migration Plan

No user-facing migration needed — `thresh-viz` is a binary, not a library API consumed downstream. Internal consumers should note:

- `VizFrame` gains optional `associations: Vec<(MeasurementId, TrackId)>` and `events: Vec<LifecycleEvent>` fields. Both default to empty for backward compatibility with existing `Recording` JSON files.
- `MotMetricsBuilder` is a new public type in `thresh-eval`; the existing one-shot `compute_mot_metrics` API continues to work unchanged.
- New CLI flags on the `thresh-viz` binary: `--stream <addr>`, `--max-buffered-snapshots <N>`, `--screenshot-dir <path>`. All optional.

## Open Questions

- Should we emit explicit `TrackEvent`s from `MultiObjectTracker` (D3 option B) as a follow-up change in v0.3.x, or wait until a downstream consumer requests it?
- For the live streaming case with no ground truth, should we surface a "synthetic GT" toggle that uses one of the trackers as ground truth to compute relative metrics? (Probably v3.)
- Where should screenshots land by default — current working directory, OS-standard "Pictures" folder, or `$XDG_DATA_HOME/thresh-viz/`? Defer to maintainer preference; current proposal says cwd.
