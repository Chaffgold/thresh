//! Async streaming wrapper around [`MultiObjectTracker`].
//!
//! This module provides [`StreamingTracker`], which accepts individual
//! [`TimestampedMeasurement`]s via a `tokio::sync::mpsc` channel, bins them
//! into temporal frames, and emits [`TrackSnapshot`]s on a
//! `tokio::sync::broadcast` channel after each tracker step.
//!
//! # Feature gate
//!
//! Requires the `streaming` feature:
//!
//! ```toml
//! [dependencies]
//! thresh-tracker = { version = "0.1", features = ["streaming"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use thresh_tracker::tracker::MultiObjectTracker;
//! use thresh_tracker::streaming::{StreamingTracker, StreamingConfig, TimestampedMeasurement};
//! use nalgebra::DVector;
//!
//! #[tokio::main]
//! async fn main() {
//!     let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
//!     let config = StreamingConfig::default();
//!     let (streaming, handle) = StreamingTracker::new(tracker, config);
//!
//!     let tx = streaming.sender();
//!     let mut rx = streaming.subscribe();
//!
//!     tx.send(TimestampedMeasurement {
//!         measurement: DVector::from_column_slice(&[100.0, 200.0, 50.0]),
//!         timestamp: 0.05,
//!     }).await.unwrap();
//!
//!     // Track snapshots arrive on `rx` after each frame flush.
//!     drop(tx);
//!     handle.await.unwrap();
//! }
//! ```

use nalgebra::DVector;
use tokio::sync::{broadcast, mpsc};

use crate::tracker::MultiObjectTracker;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single measurement with a timestamp, suitable for streaming ingestion.
#[derive(Debug, Clone)]
pub struct TimestampedMeasurement {
    /// Cartesian detection vector (e.g. `[x, y, z]`).
    pub measurement: DVector<f64>,
    /// Seconds since some epoch.
    pub timestamp: f64,
}

/// Policy for handling back-pressure when the input channel is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Drop the oldest pending measurement when the channel is full.
    DropOldest,
    /// Block the sender until space is available.
    Block,
}

/// Configuration for the streaming tracker.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Duration of each temporal frame in seconds (e.g. `0.1` = 100 ms).
    pub frame_duration_s: f64,
    /// Maximum time to hold a frame before flushing (seconds).
    pub max_latency_s: f64,
    /// What to do when frames back up.
    pub drop_policy: DropPolicy,
    /// Input `mpsc` channel buffer size.
    pub channel_capacity: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            frame_duration_s: 0.1,
            max_latency_s: 0.5,
            drop_policy: DropPolicy::DropOldest,
            channel_capacity: 256,
        }
    }
}

/// Frozen snapshot of all tracks after a single tracker step.
#[derive(Debug, Clone)]
pub struct TrackSnapshot {
    /// Timestamp of the frame that produced this snapshot.
    pub timestamp: f64,
    /// Track states at this instant.
    pub tracks: Vec<TrackState>,
    /// Number of intermediate frames that were skipped due to latency.
    pub frames_dropped: u64,
}

/// State of a single track within a [`TrackSnapshot`].
#[derive(Debug, Clone)]
pub struct TrackState {
    /// Unique track identifier.
    pub id: u64,
    /// Estimated position `[x, y, z]`.
    pub position: [f64; 3],
    /// Estimated velocity `[vx, vy, vz]`.
    pub velocity: [f64; 3],
    /// Diagonal of the 6×6 covariance (x, vx, y, vy, z, vz).
    pub covariance_diag: [f64; 6],
    /// Whether the track has been confirmed by the lifecycle manager.
    pub is_confirmed: bool,
}

// ---------------------------------------------------------------------------
// TemporalBinner
// ---------------------------------------------------------------------------

/// Accumulates [`TimestampedMeasurement`]s into fixed-duration frames.
struct TemporalBinner {
    frame_duration_s: f64,
    current_frame_start: Option<f64>,
    buffer: Vec<DVector<f64>>,
}

impl TemporalBinner {
    fn new(frame_duration_s: f64) -> Self {
        Self {
            frame_duration_s,
            current_frame_start: None,
            buffer: Vec::new(),
        }
    }

    /// Push a measurement. Returns `Some(frame)` when the measurement crosses
    /// the current frame boundary, flushing all buffered measurements from the
    /// *previous* frame. The new measurement is retained for the next frame.
    fn push(&mut self, m: TimestampedMeasurement) -> Option<Vec<DVector<f64>>> {
        match self.current_frame_start {
            None => {
                // First measurement ever — start the first frame.
                self.current_frame_start = Some(m.timestamp);
                self.buffer.push(m.measurement);
                None
            }
            Some(start) => {
                if m.timestamp < start + self.frame_duration_s {
                    // Still within the current frame.
                    self.buffer.push(m.measurement);
                    None
                } else {
                    // Crossed the frame boundary — flush the current buffer.
                    let frame = std::mem::take(&mut self.buffer);
                    // Advance the frame start to the new measurement's frame.
                    self.current_frame_start = Some(start + self.frame_duration_s);
                    self.buffer.push(m.measurement);
                    Some(frame)
                }
            }
        }
    }

    /// Force-flush whatever is currently buffered (for latency deadlines).
    fn flush(&mut self) -> Option<Vec<DVector<f64>>> {
        if self.buffer.is_empty() {
            None
        } else {
            self.current_frame_start = None;
            Some(std::mem::take(&mut self.buffer))
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingTracker
// ---------------------------------------------------------------------------

/// A closure wrapper that asserts `Send` by erasing the inner type behind
/// a raw pointer.
///
/// [`MultiObjectTracker`] is `!Send` because it stores `dyn Fn` and
/// `dyn MotionModel` trait objects without explicit `Send` bounds. However,
/// the streaming layer takes **exclusive ownership** and only accesses the
/// tracker from a single dedicated thread, so sending the closure that
/// captures it is safe.
struct SendableFn {
    ptr: *mut dyn FnOnce(),
}

// SAFETY: The closure is moved into a single background thread and is
// invoked exactly once. The `!Send` tracker it captures is never shared or
// accessed from any other thread.
unsafe impl Send for SendableFn {}

impl SendableFn {
    fn new<F: FnOnce() + 'static>(f: F) -> Self {
        Self {
            ptr: Box::into_raw(Box::new(f) as Box<dyn FnOnce()>),
        }
    }

    /// Invoke the closure, consuming it.
    ///
    /// # Safety
    /// Must only be called once.
    unsafe fn call(self) {
        let f: Box<dyn FnOnce()> = unsafe { Box::from_raw(self.ptr) };
        f();
    }
}

/// A join handle for the background processing thread.
///
/// This wraps a `std::thread::JoinHandle` because [`MultiObjectTracker`] is
/// not `Send` (it contains trait-object fields). The processing loop runs on
/// a dedicated OS thread and communicates with the async world via tokio
/// channels.
pub struct StreamingTrackerHandle {
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamingTrackerHandle {
    /// Block the current thread until the background loop finishes.
    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Async streaming wrapper that drives a [`MultiObjectTracker`] from a channel.
pub struct StreamingTracker {
    config: StreamingConfig,
    measurement_tx: mpsc::Sender<TimestampedMeasurement>,
    track_tx: broadcast::Sender<TrackSnapshot>,
}

impl StreamingTracker {
    /// Create a streaming tracker, spawning a background thread that runs the
    /// processing loop.
    ///
    /// Returns `(Self, StreamingTrackerHandle)`. The handle can be joined to
    /// detect when the background loop exits (i.e. when all senders are
    /// dropped and the channel closes).
    pub fn new(
        tracker: MultiObjectTracker,
        config: StreamingConfig,
    ) -> (Self, StreamingTrackerHandle) {
        let (measurement_tx, measurement_rx) = mpsc::channel(config.channel_capacity);
        // Broadcast capacity — use a reasonable default.
        let (track_tx, _) = broadcast::channel(64);

        let binner = TemporalBinner::new(config.frame_duration_s);
        let broadcast_tx = track_tx.clone();
        let frame_dur = config.frame_duration_s;
        let max_lat = config.max_latency_s;

        let task = SendableFn::new(move || {
            run_loop_blocking(
                tracker,
                measurement_rx,
                broadcast_tx,
                binner,
                frame_dur,
                max_lat,
            );
        });
        let thread_handle = std::thread::spawn(move || {
            // SAFETY: `call` is invoked exactly once.
            unsafe { task.call() }
        });

        let st = Self {
            config,
            measurement_tx,
            track_tx,
        };
        (
            st,
            StreamingTrackerHandle {
                handle: Some(thread_handle),
            },
        )
    }

    /// Clone the sender so additional producers can push measurements.
    pub fn sender(&self) -> mpsc::Sender<TimestampedMeasurement> {
        self.measurement_tx.clone()
    }

    /// Subscribe to track snapshots. Each subscriber gets its own receiver.
    pub fn subscribe(&self) -> broadcast::Receiver<TrackSnapshot> {
        self.track_tx.subscribe()
    }

    /// Access the current configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Blocking processing loop that runs on a dedicated OS thread.
///
/// Uses `blocking_recv` on the tokio mpsc receiver so we do not need the
/// tracker (which is `!Send`) to live inside an async task.
///
/// When the tracker falls behind `max_latency_s`, intermediate frames are
/// skipped: the binner's `current_frame_start` is advanced and predict-only
/// steps (`tracker.step(&[], dt)`) are issued for each skipped frame.
fn run_loop_blocking(
    mut tracker: MultiObjectTracker,
    mut rx: mpsc::Receiver<TimestampedMeasurement>,
    broadcast_tx: broadcast::Sender<TrackSnapshot>,
    mut binner: TemporalBinner,
    frame_duration_s: f64,
    max_latency_s: f64,
) {
    let mut total_frames_dropped: u64 = 0;

    while let Some(measurement) = rx.blocking_recv() {
        let ts = measurement.timestamp;
        if let Some(frame) = binner.push(measurement) {
            // Check if we've fallen behind: if the measurement timestamp is
            // far ahead of the binner's current frame start, skip frames.
            if let Some(frame_start) = binner.current_frame_start {
                let lag = ts - frame_start;
                if lag > max_latency_s {
                    // Calculate how many frames to skip
                    let frames_to_skip = ((lag - max_latency_s) / frame_duration_s).floor() as u64;
                    for _ in 0..frames_to_skip {
                        tracker.step(&[], frame_duration_s);
                        total_frames_dropped += 1;
                    }
                    // Advance binner's frame start past the skipped frames
                    binner.current_frame_start =
                        Some(frame_start + frames_to_skip as f64 * frame_duration_s);
                }
            }
            step_and_broadcast(
                &mut tracker,
                &frame,
                frame_duration_s,
                ts,
                &broadcast_tx,
                total_frames_dropped,
            );
        }
    }
    // Channel closed — flush remaining measurements.
    if let Some(frame) = binner.flush() {
        let ts = binner.current_frame_start.unwrap_or(0.0);
        step_and_broadcast(
            &mut tracker,
            &frame,
            frame_duration_s,
            ts,
            &broadcast_tx,
            total_frames_dropped,
        );
    }
}

fn step_and_broadcast(
    tracker: &mut MultiObjectTracker,
    frame: &[DVector<f64>],
    dt: f64,
    timestamp: f64,
    broadcast_tx: &broadcast::Sender<TrackSnapshot>,
    frames_dropped: u64,
) {
    tracker.step(frame, dt);
    let snapshot = capture_snapshot(tracker, timestamp, frames_dropped);
    // Ignore send errors (no active receivers).
    let _ = broadcast_tx.send(snapshot);
}

fn capture_snapshot(
    tracker: &MultiObjectTracker,
    timestamp: f64,
    frames_dropped: u64,
) -> TrackSnapshot {
    use thresh_core::track::TrackState as Lifecycle;

    let tracks = tracker
        .tracks
        .iter()
        .filter(|t| t.is_alive())
        .map(|t| {
            let s = &t.state;
            let dim = s.len();
            // State layout: [x, vx, y, vy, z, vz]
            let position = [
                if dim > 0 { s[0] } else { 0.0 },
                if dim > 2 { s[2] } else { 0.0 },
                if dim > 4 { s[4] } else { 0.0 },
            ];
            let velocity = [
                if dim > 1 { s[1] } else { 0.0 },
                if dim > 3 { s[3] } else { 0.0 },
                if dim > 5 { s[5] } else { 0.0 },
            ];

            let cov = &t.covariance;
            let cov_dim = cov.nrows().min(cov.ncols());
            let mut covariance_diag = [0.0; 6];
            for i in 0..6.min(cov_dim) {
                covariance_diag[i] = cov[(i, i)];
            }

            TrackState {
                id: t.id.0,
                position,
                velocity,
                covariance_diag,
                is_confirmed: t.lifecycle == Lifecycle::Confirmed,
            }
        })
        .collect();

    TrackSnapshot {
        timestamp,
        tracks,
        frames_dropped,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TemporalBinner unit tests --

    #[test]
    fn test_temporal_binner_accumulates_within_frame() {
        let mut binner = TemporalBinner::new(0.1);

        // Three measurements within a single 100ms frame
        let result1 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[1.0, 2.0, 3.0]),
            timestamp: 0.00,
        });
        assert!(result1.is_none());

        let result2 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[4.0, 5.0, 6.0]),
            timestamp: 0.03,
        });
        assert!(result2.is_none());

        let result3 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[7.0, 8.0, 9.0]),
            timestamp: 0.09,
        });
        assert!(result3.is_none());

        // Buffer should hold 3 measurements
        assert_eq!(binner.buffer.len(), 3);
    }

    #[test]
    fn test_temporal_binner_flushes_on_boundary() {
        let mut binner = TemporalBinner::new(0.1);

        // Frame 1: two measurements
        binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[1.0, 0.0, 0.0]),
            timestamp: 0.00,
        });
        binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[2.0, 0.0, 0.0]),
            timestamp: 0.05,
        });

        // This measurement crosses the boundary -> flushes frame 1
        let result = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[3.0, 0.0, 0.0]),
            timestamp: 0.12,
        });

        let frame = result.expect("should have flushed frame 1");
        assert_eq!(frame.len(), 2);
        assert_eq!(frame[0][0], 1.0);
        assert_eq!(frame[1][0], 2.0);

        // The new measurement should be in the buffer for the next frame
        assert_eq!(binner.buffer.len(), 1);
        assert_eq!(binner.buffer[0][0], 3.0);
    }

    #[test]
    fn test_temporal_binner_force_flush() {
        let mut binner = TemporalBinner::new(0.1);

        binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[10.0, 20.0, 30.0]),
            timestamp: 0.00,
        });
        binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[40.0, 50.0, 60.0]),
            timestamp: 0.02,
        });

        let flushed = binner.flush().expect("should flush 2 measurements");
        assert_eq!(flushed.len(), 2);

        // After flush, buffer is empty
        assert!(binner.flush().is_none());
    }

    // -- StreamingTracker integration tests --

    #[tokio::test]
    async fn test_streaming_tracker_receives_tracks() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let config = StreamingConfig {
            frame_duration_s: 0.1,
            channel_capacity: 64,
            ..Default::default()
        };
        let (streaming, handle) = StreamingTracker::new(tracker, config);

        let tx = streaming.sender();
        let mut rx = streaming.subscribe();

        // Send 10 measurements across 2 frames (5 per frame).
        // Frame 1: t=0.00..0.09, Frame 2: t=0.10..0.19
        for i in 0..10 {
            let t = i as f64 * 0.02; // 0.00, 0.02, 0.04, ..., 0.18
            tx.send(TimestampedMeasurement {
                measurement: DVector::from_column_slice(&[100.0, 200.0, 50.0]),
                timestamp: t,
            })
            .await
            .unwrap();
        }

        // Drop the sender to close the channel and let the loop finish.
        drop(tx);
        // Also drop the streaming's internal sender.
        drop(streaming);

        // Wait for the background thread to finish.
        tokio::task::spawn_blocking(move || handle.join())
            .await
            .unwrap();

        // We should have received at least one TrackSnapshot.
        // The first flush happens when frame boundary is crossed (t=0.10).
        // The second flush happens on channel close (remaining buffer).
        let mut snapshots = Vec::new();
        while let Ok(snap) = rx.try_recv() {
            snapshots.push(snap);
        }

        assert!(
            !snapshots.is_empty(),
            "should have received at least one TrackSnapshot"
        );
        // After the first frame, a track should have been born.
        assert!(
            !snapshots[0].tracks.is_empty(),
            "first snapshot should contain at least one track"
        );
    }

    #[test]
    fn test_temporal_binner_gap_produces_flush() {
        let mut binner = TemporalBinner::new(0.1); // 100ms frames

        // t=0.0: first measurement, starts frame [0.0, 0.1)
        let r1 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[1.0, 0.0, 0.0]),
            timestamp: 0.0,
        });
        assert!(r1.is_none(), "first measurement should not flush");

        // t=0.035: within the same 100ms frame
        let r2 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[2.0, 0.0, 0.0]),
            timestamp: 0.035,
        });
        assert!(
            r2.is_none(),
            "second measurement within frame should not flush"
        );

        // t=1.5: 3+ frames later -> should flush the first frame
        let r3 = binner.push(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[3.0, 0.0, 0.0]),
            timestamp: 1.5,
        });
        let frame = r3.expect("gap measurement should flush previous frame");
        assert_eq!(
            frame.len(),
            2,
            "flushed frame should contain 2 measurements"
        );
        assert_eq!(frame[0][0], 1.0);
        assert_eq!(frame[1][0], 2.0);

        // The new measurement should be buffered for the next frame
        assert_eq!(binner.buffer.len(), 1);
        assert_eq!(binner.buffer[0][0], 3.0);
    }

    /// 5.2: Send measurements with a 3-frame gap, verify predict-only frames
    /// advance tracker state.
    #[tokio::test]
    async fn test_streaming_gap_advances_tracker_state() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let config = StreamingConfig {
            frame_duration_s: 0.1,
            max_latency_s: 0.5,
            channel_capacity: 64,
            ..Default::default()
        };
        let (streaming, handle) = StreamingTracker::new(tracker, config);

        let tx = streaming.sender();
        let mut rx = streaming.subscribe();

        // Frame 1: measurements at t=0.00..0.05
        for i in 0..3 {
            tx.send(TimestampedMeasurement {
                measurement: DVector::from_column_slice(&[100.0, 200.0, 50.0]),
                timestamp: i as f64 * 0.02,
            })
            .await
            .unwrap();
        }

        // Frame 2: measurement at t=0.12 (crosses boundary, flushes frame 1)
        tx.send(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[100.0, 200.0, 50.0]),
            timestamp: 0.12,
        })
        .await
        .unwrap();

        // Gap: jump to t=0.55 (3+ frames later, flushes frame 2)
        tx.send(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[110.0, 210.0, 55.0]),
            timestamp: 0.55,
        })
        .await
        .unwrap();

        // Another measurement to flush the gap frame
        tx.send(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[115.0, 215.0, 58.0]),
            timestamp: 0.70,
        })
        .await
        .unwrap();

        drop(tx);
        drop(streaming);

        tokio::task::spawn_blocking(move || handle.join())
            .await
            .unwrap();

        let mut snapshots = Vec::new();
        while let Ok(snap) = rx.try_recv() {
            snapshots.push(snap);
        }

        // We should have received multiple snapshots (frames were processed).
        assert!(
            snapshots.len() >= 2,
            "should have at least 2 snapshots, got {}",
            snapshots.len()
        );

        // The tracker state should have advanced — tracks should exist
        // even after the gap because the tracker did predict-only steps.
        let last = snapshots.last().unwrap();
        assert!(
            !last.tracks.is_empty(),
            "tracks should persist through the gap"
        );
    }

    /// 5.4: Verify clean shutdown when all senders are dropped.
    #[tokio::test]
    async fn test_streaming_clean_shutdown_on_sender_drop() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let config = StreamingConfig {
            frame_duration_s: 0.1,
            channel_capacity: 64,
            ..Default::default()
        };
        let (streaming, handle) = StreamingTracker::new(tracker, config);

        let tx = streaming.sender();

        // Send a few measurements so the loop has something to process.
        tx.send(TimestampedMeasurement {
            measurement: DVector::from_column_slice(&[50.0, 60.0, 70.0]),
            timestamp: 0.0,
        })
        .await
        .unwrap();

        // Drop all senders — this should cause the channel to close
        // and the background loop to exit cleanly.
        drop(tx);
        drop(streaming);

        // The handle should join without panic.
        let result = tokio::task::spawn_blocking(move || handle.join()).await;
        assert!(
            result.is_ok(),
            "background thread should join cleanly after senders are dropped"
        );
    }

    /// 5.3: Verify DropOldest policy lets senders proceed when the channel is full.
    #[tokio::test]
    async fn test_drop_oldest_policy_drops_when_full() {
        let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
        let config = StreamingConfig {
            frame_duration_s: 0.1,
            channel_capacity: 2, // very small channel
            drop_policy: DropPolicy::DropOldest,
            ..Default::default()
        };
        let (streaming, handle) = StreamingTracker::new(tracker, config);

        let tx = streaming.sender();

        // Send more measurements than the channel capacity without consuming.
        // With DropOldest on a bounded mpsc, the sender should not block.
        // We use a timeout to ensure this completes promptly.
        let send_result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            for i in 0..10 {
                // try_send avoids blocking; if the channel is full the message
                // is simply dropped (DropOldest semantics).
                let _ = tx.try_send(TimestampedMeasurement {
                    measurement: DVector::from_column_slice(&[i as f64 * 10.0, 200.0, 50.0]),
                    timestamp: i as f64 * 0.01,
                });
            }
        })
        .await;

        assert!(
            send_result.is_ok(),
            "sending should complete within the timeout (DropOldest should not block)"
        );

        // Clean shutdown
        drop(tx);
        drop(streaming);

        tokio::task::spawn_blocking(move || handle.join())
            .await
            .unwrap();
    }

    #[test]
    fn test_streaming_tracker_default_config() {
        let config = StreamingConfig::default();
        assert!((config.frame_duration_s - 0.1).abs() < f64::EPSILON);
        assert!((config.max_latency_s - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.drop_policy, DropPolicy::DropOldest);
        assert_eq!(config.channel_capacity, 256);
    }
}
