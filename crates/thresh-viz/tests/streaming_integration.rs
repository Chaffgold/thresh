//! Integration test exercising the live `SnapshotBridge` against a real
//! `MultiObjectTracker` driven by a simulated detection feed. Verifies
//! that snapshots flow end-to-end and that the bridge surfaces them in
//! order with the expected track count trajectory.
//!
//! Gated to the `gui` feature (which is what pulls in tokio + the
//! bridge module).

#![cfg(feature = "gui")]

use std::time::Duration;

use nalgebra::DVector;
use thresh_tracker::streaming::{
    DropPolicy, StreamingConfig, StreamingTracker, TimestampedMeasurement, TrackSnapshot,
};
use thresh_tracker::tracker::MultiObjectTracker;
use thresh_viz::streaming::{ConnectionStatus, SnapshotBridge};

/// Wait up to `timeout` for `pred`. Polls every 10ms.
fn wait_for(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if pred() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    pred()
}

#[test]
fn streaming_tracker_to_bridge_round_trip() {
    let tracker = MultiObjectTracker::new_cv_position(10.0, 50.0);
    let cfg = StreamingConfig {
        frame_duration_s: 0.05,
        max_latency_s: 0.5,
        channel_capacity: 256,
        drop_policy: DropPolicy::DropOldest,
    };
    let (streaming, _handle) = StreamingTracker::new(tracker, cfg);

    let bridge = SnapshotBridge::with_capacity(128).expect("bridge");
    bridge.connect(streaming.subscribe());
    let tx = streaming.sender();

    // Drive the tracker for ~200 detections at ~200Hz (5ms apart). The
    // streaming layer's 50ms binner will produce ~20 frames, plenty
    // for a round-trip assertion.
    for i in 0..200 {
        let measurement = DVector::from_column_slice(&[100.0 + i as f64, 200.0, 50.0]);
        tx.try_send(TimestampedMeasurement {
            measurement,
            timestamp: i as f64 * 0.005,
        })
        .expect("try_send");
        std::thread::sleep(Duration::from_millis(5));
    }

    // Allow the binner to flush the tail.
    std::thread::sleep(Duration::from_millis(200));

    let mut buf: Vec<TrackSnapshot> = Vec::new();
    let drained = bridge.drain_into(&mut buf);
    assert_eq!(drained, buf.len());
    assert!(
        !buf.is_empty(),
        "expected at least one snapshot from StreamingTracker → SnapshotBridge"
    );

    // Timestamps must be monotonically non-decreasing.
    for w in buf.windows(2) {
        assert!(
            w[0].timestamp <= w[1].timestamp,
            "snapshots out of order: {} > {}",
            w[0].timestamp,
            w[1].timestamp,
        );
    }

    // Track count should be > 0 for at least the later snapshots
    // (M-of-N confirmation needs a few hits).
    let last_track_count = buf.last().map(|s| s.tracks.len()).unwrap_or(0);
    assert!(
        last_track_count >= 1,
        "expected ≥1 track in the final snapshot; got {last_track_count}"
    );

    // Connection should report Connected (sender alive, recent arrivals).
    assert!(
        wait_for(Duration::from_millis(100), || bridge.status()
            == ConnectionStatus::Connected),
        "expected Connected status; got {:?}",
        bridge.status()
    );
}
