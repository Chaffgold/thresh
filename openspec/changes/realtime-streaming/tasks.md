# Real-Time Streaming Tracker API — Tasks

## 1. Core streaming types

- [ ] 1.1 Add `TimestampedMeasurement` struct to `crates/thresh-tracker/src/streaming.rs` with `measurement: DVector<f64>`, `timestamp: f64`, `sensor_id: Option<u32>`
- [ ] 1.2 Add `StreamingConfig` struct with `frame_duration`, `max_latency`, `drop_policy`, `channel_capacity`, `broadcast_capacity` and sensible defaults
- [ ] 1.3 Add `DropPolicy` enum (`DropOldest`, `Block`)
- [ ] 1.4 Add `TrackSnapshot` and `TrackState` output structs for broadcast channel output

## 2. Temporal binner

- [ ] 2.1 Implement `TemporalBinner` struct with `current_frame_start`, `pending` buffer, and `frame_duration`
- [ ] 2.2 Implement `TemporalBinner::push(&mut self, measurement: TimestampedMeasurement)` to accumulate measurements
- [ ] 2.3 Implement `TemporalBinner::flush(&mut self) -> Option<Frame>` that returns accumulated measurements when the frame window elapses, inserting predict-only frames for gaps
- [ ] 2.4 Unit test: measurements within one window are grouped; measurements spanning two windows produce two flushes
- [ ] 2.5 Unit test: gap of 3x frame_duration produces two predict-only frames plus one detection frame

## 3. StreamingTracker

- [ ] 3.1 Implement `StreamingTracker::new(config: StreamingConfig, tracker: MultiObjectTracker) -> (Self, mpsc::Sender<TimestampedMeasurement>, broadcast::Receiver<TrackSnapshot>)`
- [ ] 3.2 Implement `StreamingTracker::run(&mut self) -> Result<(), StreamingError>` async processing loop: receive -> bin -> step -> broadcast
- [ ] 3.3 Implement `capture_snapshot` that reads confirmed tracks from `MultiObjectTracker` and builds `TrackSnapshot`
- [ ] 3.4 Implement latency management: detect when tracker falls behind `max_latency`, drop intermediate frames, predict forward

## 4. Feature gate and dependencies

- [ ] 4.1 Add `streaming` feature to `crates/thresh-tracker/Cargo.toml` with optional `tokio` dependency (features: `sync`, `time`, `rt`)
- [ ] 4.2 Gate all streaming module code behind `#[cfg(feature = "streaming")]`
- [ ] 4.3 Add `pub mod streaming;` to `crates/thresh-tracker/src/lib.rs` behind the feature gate

## 5. Testing

- [ ] 5.1 Integration test: spawn `StreamingTracker::run`, send 100 measurements via mpsc, receive track snapshots via broadcast, verify track confirmation
- [ ] 5.2 Integration test: send measurements with a 3-frame gap, verify predict-only frames advance tracker state
- [ ] 5.3 Integration test: verify `DropOldest` policy drops old measurements when channel is full
- [ ] 5.4 Integration test: verify clean shutdown when all senders are dropped

## 6. Documentation

- [ ] 6.1 Add module-level doc comment to `streaming.rs` with usage example showing channel creation, spawn, and subscription
- [ ] 6.2 Add `streaming` feature documentation to `crates/thresh-tracker/Cargo.toml` metadata
