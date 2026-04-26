# track-dashboard Specification

## Purpose
TBD - created by archiving change track-visualization. Update Purpose after archive.
## Requirements
### Requirement: 2D bird's-eye-view track plot

The system MUST render a 2D plot showing track position trails (color-coded by track ID), measurement scatter points, association lines connecting measurements to their assigned tracks, and optional 2σ position-covariance ellipses per track. Association line rendering and covariance ellipse rendering MUST each be toggleable independently via keyboard shortcut.

#### Scenario: Displaying a multi-target tracking scenario

- **WHEN** the visualization receives TrackSnapshot data containing 10 active tracks over 100 timesteps
- **THEN** it renders each track's position history as a color-coded trail on the 2D plot, displays current-timestep measurements as scatter points, and draws association lines from each measurement to its assigned track
- **AND** the display updates at a minimum of 30 frames per second during live streaming and allows pan/zoom interaction without dropping frames

#### Scenario: Toggling covariance ellipses

- **WHEN** a user presses the covariance-ellipse toggle hotkey while viewing tracks with non-zero position covariance
- **THEN** the system renders a 2σ ellipse around each track's current position derived from the position-block diagonal of the covariance matrix
- **AND** pressing the same hotkey again hides the ellipses without affecting other plot elements

#### Scenario: Toggling association lines

- **WHEN** a user presses the association-line toggle hotkey
- **THEN** the system stops drawing measurement-to-track lines on subsequent frames; pressing again restores them

### Requirement: Real-time metric sidebar

The system MUST display a metric sidebar showing per-frame MOT metrics (MOTA, MOTP, IDF1, track count, confirmed/tentative/lost breakdown) and a scrolling track lifecycle event log (births, deaths, ID switches, merges) updated at each timestep. When ground truth is unavailable (live streaming with no GT feed), structural counters MUST still update and the GT-dependent metrics MUST be displayed as "n/a — no ground truth" rather than stale values.

#### Scenario: Live metric updates during streaming

- **WHEN** the visualization is connected to a live streaming tracker via broadcast channel and ground truth is available
- **THEN** the metric sidebar updates at each timestep with current MOTA, MOTP, IDF1, total track count, counts of confirmed / tentative / lost tracks, and the most recent ten lifecycle events
- **AND** metrics display with no more than one-timestep latency from the tracker's broadcast

#### Scenario: Streaming with no ground truth

- **WHEN** the visualization is connected to a live streaming tracker but no ground truth source is configured
- **THEN** the structural counters (track count, confirmed/tentative/lost breakdown, lifecycle event log) MUST continue to update each timestep
- **AND** the MOTA, MOTP, and IDF1 fields MUST display "n/a — no ground truth" without showing stale values from a previous run

#### Scenario: Per-frame metric incremental computation

- **WHEN** the dashboard receives a snapshot at timestep T
- **THEN** the metrics builder MUST update its internal state and emit a fresh `MotMetrics` value in O(K · M) time, where K is the active track count and M is the active ground-truth count, without recomputing over the full history of timesteps 0..T

### Requirement: Playback from recorded scenarios

The system MUST support loading a JSON recording of TrackSnapshot data and providing playback controls including play, pause, step forward, step backward, speed control, and seek to a specific timestep.

#### Scenario: Stepping through a recorded scenario

**WHEN** a user loads a JSON recording file and clicks the step-forward button

**THEN** the visualization advances exactly one timestep, updating the plot and metrics to reflect the new timestep's data

**SHALL** allow stepping backward to any previously viewed timestep without reloading the recording file

### Requirement: TrackSnapshot export from tracker

The system MUST provide a `TrackSnapshot` type in thresh-tracker that captures the full tracker state at a single timestep (all tracks with state, covariance, status, associations, and optional metrics) and supports serialization to JSON.

#### Scenario: Recording a tracking session

**WHEN** a user enables snapshot recording on a `MultiObjectTracker` instance

**THEN** the tracker emits a `TrackSnapshot` at each `step()` call, serializable to JSON for later playback

**SHALL** include all track states, covariances, lifecycle statuses, and association decisions in each snapshot

### Requirement: Live streaming connection status

The system MUST display a streaming connection status indicator showing one of three states: `Connected` (snapshots arriving within the configured latency budget), `Lagging` (snapshot deque has exceeded the high-water mark and oldest snapshots are being dropped), or `Disconnected` (broadcast channel sender has been dropped or no snapshot has arrived within the timeout).

#### Scenario: Indicator transitions to lagging when buffer overflows

- **WHEN** the snapshot ingest deque exceeds its configured high-water mark (default 64 snapshots)
- **THEN** the status indicator MUST switch to `Lagging` within one render frame
- **AND** the system MUST drop the oldest snapshots to bring the deque back below the high-water mark

#### Scenario: Indicator transitions to disconnected when channel closes

- **WHEN** the upstream `StreamingTracker` is dropped or the broadcast `Sender` count reaches zero
- **THEN** the status indicator MUST switch to `Disconnected` within one render frame and remain until a new connection is established

### Requirement: PNG screenshot export

The system MUST support exporting a PNG screenshot of the current viewport via a keyboard hotkey and a menu option. The output filename MUST include an ISO-8601 UTC timestamp.

#### Scenario: Screenshot via hotkey

- **WHEN** a user presses the screenshot hotkey
- **THEN** the system writes a PNG file to the configured screenshot directory (default: current working directory) with filename matching `thresh-viz-screenshot-YYYYMMDDTHHMMSSZ.png`
- **AND** a transient on-screen confirmation message MUST display the absolute path of the saved file for at least 2 seconds

### Requirement: Keyboard shortcuts help overlay

The system MUST provide a toggleable on-screen help panel listing all keyboard shortcuts. The panel MUST be reachable via a single hotkey from any view.

#### Scenario: Toggling the help overlay

- **WHEN** a user presses the help-overlay toggle hotkey from any view
- **THEN** the system displays an overlay listing every bound shortcut with its action description
- **AND** pressing the same hotkey again or pressing Escape MUST hide the overlay without affecting any other UI state

### Requirement: Cross-platform GUI build verification

The CI pipeline MUST verify that `cargo build -p thresh-viz --features gui` succeeds on Ubuntu, macOS, and Windows runners on every pull request to develop.

#### Scenario: viz-build job runs on all three platforms

- **WHEN** a pull request is opened against develop that touches any file under `crates/thresh-viz/`, `crates/thresh-eval/`, `crates/thresh-tracker/src/streaming.rs`, or the workflow file itself
- **THEN** the `viz-build` CI job MUST run a build matrix across `ubuntu-latest`, `macos-latest`, and `windows-latest`
- **AND** the job MUST fail if any platform fails to build, with `fail-fast: false` so all three results are visible

