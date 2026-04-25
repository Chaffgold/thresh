## Context

thresh-tracker produces track state at each timestep, but there is no built-in way to visualize tracker behavior spatially. Developers currently debug by reading log output, serializing track state to JSON, or writing one-off Python plotting scripts. This slows iteration and makes it difficult to understand spatial relationships, association quality, and track lifecycle transitions at a glance.

The Rust GUI ecosystem has matured around `egui` (immediate-mode GUI) and `eframe` (native windowing backend). `egui_plot` provides interactive 2D plotting with pan, zoom, and hover inspection. These crates are pure Rust, cross-platform, and have no web or system GUI dependencies. thresh-tracker already has a broadcast channel mechanism for streaming tracker state to consumers.

## Goals / Non-Goals

**Goals:**
- Create a new `thresh-viz` crate for track visualization
- Implement a 2D bird's-eye-view plot with track trails, measurement scatter, and association lines
- Display real-time MOT metrics in a sidebar
- Support live streaming from the tracker's broadcast channel
- Support playback from recorded JSON files with full transport controls
- Keep the crate optional (not in default workspace build)
- Cross-platform: Linux, macOS, Windows

**Non-Goals:**
- Web-based visualization (native desktop only)
- Video recording/export
- Editing tracker parameters through the UI
- Map or terrain underlay
- Distributed/networked visualization
- 3D rendering beyond basic `egui_plot` orbit view

## Decisions

### 1. New crate `thresh-viz` outside default workspace members

**Decision:** Create `crates/thresh-viz/` as a workspace member but exclude it from `default-members` in the root Cargo.toml. Build with `cargo build -p thresh-viz`.

**Rationale:** The visualization tool has heavy GUI dependencies (egui, eframe, egui_plot) that most users do not need. Excluding from default-members means `cargo build --workspace` does not pull in these dependencies. Users who want the visualization explicitly opt in.

### 2. egui + eframe for the GUI framework

**Decision:** Use `egui` (immediate-mode GUI) with `eframe` (native windowing via winit + glow/wgpu) for the desktop application.

**Rationale:** egui is the most mature immediate-mode GUI in Rust. It produces native desktop windows on all three major platforms without web dependencies. Immediate-mode rendering simplifies state management -- the UI is re-rendered every frame from the current data, with no retained widget tree to synchronize. `egui_plot` provides interactive 2D plotting with pan, zoom, axis labels, and legend out of the box.

### 3. `TrackSnapshot` as the data exchange type

**Decision:** Define a `TrackSnapshot` struct in thresh-tracker that captures the full tracker state at a single timestep:

```
pub struct TrackSnapshot {
    pub timestep: u64,
    pub timestamp: f64,
    pub tracks: Vec<TrackState>,
    pub measurements: Vec<Measurement>,
    pub associations: Vec<(TrackId, usize)>,  // track -> measurement index
    pub metrics: Option<MetricSnapshot>,
}

pub struct TrackState {
    pub id: TrackId,
    pub state: DVector<f64>,
    pub covariance: DMatrix<f64>,
    pub status: TrackStatus,  // Tentative, Confirmed, Lost
    pub class: Option<String>,
}
```

Serializable via serde to JSON for recording and playback.

**Rationale:** A single snapshot type decouples the tracker from the visualizer. The tracker produces snapshots; the visualizer consumes them. This works for both live streaming (snapshots sent via broadcast channel) and recorded playback (snapshots serialized to a JSON array file).

### 4. 2D plot as the primary view

**Decision:** The main view is a 2D bird's-eye plot (X-Y plane) rendered with `egui_plot::Plot`. Elements:
- Track trails: polylines from track history, color-coded by track ID using a deterministic hash-based palette
- Current position: filled circle at the latest track state
- Measurement scatter: small markers for current-timestep measurements
- Association lines: dashed lines from each measurement to its assigned track
- Gating ellipses: optional 2-sigma covariance ellipses around each track
- Legend: track ID to color mapping

**Rationale:** Bird's-eye 2D is the most common tracking visualization. It shows spatial relationships, crossing tracks, association quality, and coverage gaps at a glance. The covariance ellipses show filter uncertainty, which is critical for understanding gating and association behavior.

### 5. Metric sidebar

**Decision:** A right-side panel displays:
- Current timestep / total timesteps
- Total track count (confirmed / tentative / lost)
- MOTA, MOTP (if ground truth available)
- IDF1, HOTA (if ground truth available)
- Track birth/death counts for current timestep
- Association strategy in use

Metrics update at each timestep. If no ground truth is available, only track counts are shown.

**Rationale:** Metrics provide quantitative context for the visual display. Seeing MOTA drop while watching tracks swap on the plot immediately connects the visual failure mode to the metric degradation.

### 6. Two input modes: streaming and playback

**Decision:**
- **Streaming:** `thresh-viz` subscribes to a `tokio::sync::broadcast::Receiver<TrackSnapshot>` and renders snapshots as they arrive. The tracker's `step()` method sends snapshots to the broadcast channel.
- **Playback:** `thresh-viz` loads a JSON file containing a `Vec<TrackSnapshot>` and provides transport controls (play, pause, step forward, step backward, speed slider, seek bar).

**Rationale:** Streaming enables real-time debugging during development. Playback enables post-hoc analysis of recorded scenarios, sharing interesting cases, and regression testing visualization against known scenarios.

### 7. Playback transport controls

**Decision:** The playback bar includes:
- Play/Pause toggle (Space key)
- Step forward (Right arrow)
- Step backward (Left arrow)
- Speed slider (0.25x to 4x, default 1x)
- Seek slider (scrub to any timestep)
- Timestep display (current / total)

**Rationale:** Standard media player controls are intuitive. Step forward/backward is essential for detailed analysis of specific association decisions. Speed control allows fast review of long scenarios.

### 8. Screenshot export

**Decision:** Add a "Screenshot" button that captures the current frame (plot + sidebar) and saves it as a PNG file to a user-specified path.

**Rationale:** Screenshots are needed for reports, documentation, and issue filing. PNG is universally supported. Full video recording is out of scope but screenshots of key frames cover most documentation needs.

## Risks / Trade-offs

**[Risk] egui rendering performance with many tracks.** Rendering hundreds of track trails (each with hundreds of points) may cause frame drops. Mitigation: implement trail length limiting (only render the last N points per track), level-of-detail reduction when zoomed out, and optional track filtering by ID or status.

**[Risk] Large JSON recording files.** A 1000-timestep scenario with 100 tracks produces a large JSON file. Mitigation: consider binary serialization (bincode, MessagePack) as an alternative to JSON for large recordings. JSON remains the default for human readability.

**[Trade-off] Separate crate vs integrated into thresh-tracker.** A separate crate keeps visualization dependencies isolated but adds a new crate to the workspace. Integrated visualization would avoid the new crate but would force all thresh-tracker users to depend on egui. The separate crate is the right trade-off for a heavyweight optional dependency.

**[Trade-off] No web UI.** A web-based visualizer would be more shareable (just open a URL) but adds web dependencies (wasm, web-sys) and a web server. The native desktop approach is simpler and performs better for real-time rendering.

**[Risk] Cross-platform GPU compatibility.** eframe uses glow (OpenGL) or wgpu for rendering. Some headless servers or VMs lack GPU support. Mitigation: eframe falls back to software rendering (llvmpipe) on systems without GPU. Document the requirement for a display server (X11, Wayland, or macOS WindowServer).

## Open Questions

- Should the 3D orbit view be included in v0.3.0 or deferred to a later version?
- Should the recording format support ground truth tracks alongside tracker output for overlay comparison?
- Should thresh-viz be a library crate (importable) or a binary crate (standalone executable), or both?
- Should the broadcast channel use `tokio::sync::broadcast` or `std::sync::mpsc` for environments without a tokio runtime?
- Should track colors be configurable, or is hash-based auto-coloring sufficient?
