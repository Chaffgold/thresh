# Track Visualization Dashboard

## What

Add a real-time track visualization tool that displays tracker state, track histories, measurement associations, and MOT metrics on a 2D/3D plot. Implemented as a new `thresh-viz` crate using `egui` + `eframe` for a native desktop application. Reads `TrackSnapshot` data from the streaming tracker's broadcast channel or from recorded scenario output (JSON format).

## Why

Debugging tracker behavior currently requires reading log output, inspecting serialized track state, or writing custom plotting scripts in Python. This slows development iteration and makes it difficult to understand spatial relationships, association decisions, and track lifecycle transitions. A built-in visualization tool accelerates development by providing immediate visual feedback on tracker behavior, makes thresh more accessible to new users who can see the tracker in action, and enables rapid identification of failure modes (track swaps, missed detections, false tracks, fragmentation).

## How

- Create a new `thresh-viz` crate in the workspace, not included in the default workspace build (listed under `[workspace]` members but excluded from default-members)
- Use `egui` (immediate-mode GUI) and `eframe` (native windowing) for the desktop application -- no web dependencies, cross-platform (Linux, macOS, Windows)
- Implement a 2D bird's-eye-view plot using `egui_plot` showing: track position trails (color-coded by track ID), measurement scatter points, association lines connecting measurements to tracks, gating ellipses
- Implement a metric sidebar displaying real-time MOT metrics (MOTA, MOTP, track count, confirmed/tentative/lost counts)
- Add a `TrackSnapshot` export mechanism to thresh-tracker that serializes the full tracker state (all tracks, associations, metrics) at each timestep
- Support two input modes: live streaming from the tracker's broadcast channel (for real-time visualization) and playback from a JSON recording file (for post-hoc analysis)
- Add playback controls: play/pause, step forward/backward, speed control, seek to timestep
- Add optional 3D orbit view using `egui_plot` for scenarios with altitude variation
- Add screenshot/frame export for reports and documentation

## Out of scope

- Web-based visualization (native desktop only)
- Video recording/export (screenshots only)
- Editing tracker parameters through the UI (read-only visualization)
- Map/terrain underlay (plots use abstract coordinate space)
- Distributed/networked visualization (single-machine only)

## Affected crates

- thresh-viz (new crate): visualization application, 2D/3D plotting, playback, metric display
- thresh-tracker: `TrackSnapshot` export, broadcast channel integration for live streaming
