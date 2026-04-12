# Real-Time Streaming Tracker API — Design

## Context

`MultiObjectTracker::step(&mut self, detections: &[DVector<f64>], dt: f64)` is the current synchronous batch API. Callers must implement their own temporal binning (grouping measurements into frames), frame-rate management, backpressure handling, and missed-frame logic. The `Measurement` type in thresh-core has timestamps and sensor IDs but no async channel integration. Every live integration rebuilds the same buffering and timing scaffolding around `step()`.

## Goals / Non-Goals

**Goals:**
- `StreamingTracker` wrapper in thresh-tracker around `MultiObjectTracker`
- `tokio::sync::mpsc` for measurement input with configurable buffer depth
- `tokio::sync::broadcast` for track output so multiple consumers can subscribe
- `StreamingConfig`: frame_duration, max_latency, drop_policy
- Temporal binner: accumulate measurements within a frame window, then flush to `step()`
- Track output includes timestamp and all confirmed track states
- `tokio` as optional dependency behind `streaming` feature gate
- `step()` remains the synchronous API; streaming wraps it, does not replace it

**Non-Goals:**
- Network I/O, sockets, or protocol handling
- Distributed tracking across processes or machines
- GUI, dashboard, or visualization
- Guaranteed real-time scheduling or RTOS integration

## Decisions

### StreamingTracker architecture

```
                      mpsc::Sender<TimestampedMeasurement>
                              |
                              v
                    +-------------------+
                    | StreamingTracker  |
                    |                   |
                    |  temporal_binner  |  accumulates measurements
                    |       |           |  per frame window
                    |       v           |
                    |  tracker.step()   |  synchronous core
                    |       |           |
                    |       v           |
                    |  broadcast_tx     |  emits TrackSnapshot
                    +-------------------+
                              |
                    broadcast::Receiver<TrackSnapshot>
                         (N consumers)
```

`StreamingTracker` owns:
- A `MultiObjectTracker` instance
- A `tokio::sync::mpsc::Receiver<TimestampedMeasurement>` for inbound measurements
- A `tokio::sync::broadcast::Sender<TrackSnapshot>` for outbound track state
- A `TemporalBinner` for frame accumulation

### TimestampedMeasurement

```rust
pub struct TimestampedMeasurement {
    pub measurement: DVector<f64>,
    pub timestamp: f64,
    pub sensor_id: Option<u32>,
}
```

This is deliberately simpler than the existing `Measurement` enum. Callers convert their sensor-specific measurements to `DVector<f64>` before submitting to the streaming channel. This keeps the streaming layer sensor-agnostic.

### StreamingConfig

```rust
pub struct StreamingConfig {
    /// Duration of each frame window in seconds.
    pub frame_duration: f64,
    /// Maximum allowed processing latency in seconds.
    /// If the tracker falls behind by more than this, frames are dropped.
    pub max_latency: f64,
    /// What to do when the input channel is full.
    pub drop_policy: DropPolicy,
    /// Input channel buffer depth.
    pub channel_capacity: usize,
    /// Broadcast channel capacity for track output.
    pub broadcast_capacity: usize,
}

pub enum DropPolicy {
    /// Drop the oldest pending measurement when the channel is full.
    DropOldest,
    /// Block the sender until space is available.
    Block,
}
```

Default: `frame_duration = 0.1` (10 Hz), `max_latency = 0.5`, `drop_policy = DropOldest`, `channel_capacity = 1024`, `broadcast_capacity = 64`.

### Temporal binner

`TemporalBinner` accumulates measurements and decides when to flush. It maintains:
- `current_frame_start: f64` — timestamp of the current frame window start
- `pending: Vec<TimestampedMeasurement>` — measurements in the current window
- `frame_duration: f64`

When a measurement arrives with `timestamp >= current_frame_start + frame_duration`, the binner flushes: all pending measurements are converted to `&[DVector<f64>]` and passed to `tracker.step(detections, frame_duration)`. The frame window advances. If the gap between the new measurement and the frame start spans multiple windows, intermediate predict-only steps are inserted (no detections) to advance the tracker's internal time.

### Processing loop

`StreamingTracker::run()` is an async method that drives the processing loop:

```rust
pub async fn run(&mut self) -> Result<(), StreamingError> {
    while let Some(measurement) = self.rx.recv().await {
        self.binner.push(measurement);
        while let Some(frame) = self.binner.flush() {
            let dt = frame.duration;
            let detections = frame.measurements_as_dvectors();
            self.tracker.step(&detections, dt);
            let snapshot = self.capture_snapshot(frame.timestamp);
            let _ = self.broadcast_tx.send(snapshot);
        }
    }
    Ok(())
}
```

The loop exits when all senders are dropped (channel closed). `broadcast_tx.send` ignores errors from lagging receivers (broadcast semantics).

### Latency management

If `binner.current_frame_start + max_latency < wall_clock_time`, the tracker has fallen behind. In this case, drop all pending frames except the most recent one, predict forward to current time, and process only the latest frame. This ensures the tracker output stays within `max_latency` of real time at the cost of missing intermediate detections.

Wall clock time is obtained via `tokio::time::Instant` to keep the streaming layer testable (can substitute a mock clock in tests).

### TrackSnapshot output

```rust
pub struct TrackSnapshot {
    pub timestamp: f64,
    pub tracks: Vec<TrackState>,
}

pub struct TrackState {
    pub track_id: TrackId,
    pub state: DVector<f64>,
    pub covariance: DMatrix<f64>,
    pub lifecycle: TrackLifecycle,
    pub class: TargetClass,
}
```

Broadcast to all subscribers after each `step()`. Subscribers receive a clone (broadcast channel semantics). The snapshot is a frozen view; subscribers cannot modify tracker state.

### Feature gate

All streaming types and the `StreamingTracker` are behind `#[cfg(feature = "streaming")]` in thresh-tracker. The `tokio` dependency (with `sync` and `time` features) is added as an optional dependency. The synchronous `step()` API has zero dependency on tokio.

## Risks / Trade-offs

- **tokio runtime coupling**: Callers must run a tokio runtime. This is standard for async Rust but excludes non-tokio runtimes (async-std, smol). Acceptable because tokio is the dominant async runtime.
- **Broadcast channel memory**: Each `TrackSnapshot` is cloned per subscriber. With many tracks and many subscribers, this can be expensive. Mitigate by using `Arc<TrackSnapshot>` if profiling shows clone overhead.
- **Frame duration tuning**: Too short wastes CPU on many small steps; too long increases latency. No auto-tuning is provided; callers must configure based on their sensor rates.
- **Wall clock vs. measurement time**: The latency management uses wall clock time, but measurements have their own timestamps. If measurements arrive with significant delay (e.g., buffered network), the latency check may trigger spuriously. Callers should use measurement timestamps for the binner and wall clock only for the latency budget.

## Open Questions

1. Should `StreamingTracker::run()` return a `JoinHandle` or should the caller `tokio::spawn` it?
2. Should we provide a `StreamingTracker::step_one()` method for manual single-measurement processing without the async loop?
3. Should the temporal binner support variable frame durations (adaptive frame rate) or only fixed?
