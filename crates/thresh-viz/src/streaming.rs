//! Live `TrackSnapshot` ingest for the dashboard.
//!
//! `eframe`'s update loop runs on the UI thread and is not async, but
//! `StreamingTracker::subscribe()` returns a tokio
//! `broadcast::Receiver<TrackSnapshot>` that needs to be drained
//! asynchronously. `SnapshotBridge` solves the impedance mismatch by:
//!
//! 1. Owning a multi-thread tokio runtime that lives for the GUI's
//!    lifetime.
//! 2. Spawning a drainer task that pushes each newly-arrived snapshot
//!    into a shared `Arc<Mutex<VecDeque<TrackSnapshot>>>`.
//! 3. Bounding the deque at a high-water mark — when exceeded, oldest
//!    snapshots are dropped (mirroring `DropPolicy::DropOldest` in
//!    `StreamingTracker`).
//!
//! The egui app calls [`SnapshotBridge::drain_into`] each frame to
//! transfer newly-arrived snapshots into a render buffer, then queries
//! [`SnapshotBridge::status`] for a connection indicator
//! (`Connected` / `Lagging` / `Disconnected`).
//!
//! Available only with the `gui` feature (the dependency on `tokio`).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::runtime::Runtime;
use tokio::sync::broadcast;

use thresh_tracker::streaming::TrackSnapshot;

/// Default maximum buffered snapshots before the bridge starts
/// dropping the oldest. ~2 seconds at 30 FPS.
pub const DEFAULT_HIGH_WATER_MARK: usize = 64;

/// If no snapshot has arrived in this long, the bridge reports
/// `Disconnected`.
pub const DEFAULT_DISCONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Connection status reported by [`SnapshotBridge::status`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Snapshots are arriving and the deque is below the high-water mark.
    Connected,
    /// Snapshots are arriving but the deque has exceeded the high-water
    /// mark; the bridge is dropping oldest snapshots to recover.
    Lagging,
    /// No snapshot has arrived recently or the broadcast sender has
    /// dropped.
    Disconnected,
}

#[derive(Debug)]
struct BridgeState {
    snapshots: VecDeque<TrackSnapshot>,
    last_arrival: Option<Instant>,
    sender_alive: bool,
    /// Set true when the drainer hits the high-water mark and starts
    /// dropping. Reset to false when the deque has been drained back to
    /// half the high-water mark.
    lagging: bool,
}

/// Bridges a `broadcast::Receiver<TrackSnapshot>` into a synchronous
/// deque consumed by the egui app.
pub struct SnapshotBridge {
    runtime: Arc<Runtime>,
    state: Arc<Mutex<BridgeState>>,
    high_water: usize,
    disconnect_timeout: Duration,
}

impl SnapshotBridge {
    /// Build a bridge with the default high-water mark.
    pub fn new() -> std::io::Result<Self> {
        Self::with_capacity(DEFAULT_HIGH_WATER_MARK)
    }

    /// Build a bridge with a custom high-water mark.
    pub fn with_capacity(high_water: usize) -> std::io::Result<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?,
        );
        let state = Arc::new(Mutex::new(BridgeState {
            snapshots: VecDeque::with_capacity(high_water.max(8)),
            last_arrival: None,
            sender_alive: false,
            lagging: false,
        }));
        Ok(Self {
            runtime,
            state,
            high_water,
            disconnect_timeout: DEFAULT_DISCONNECT_TIMEOUT,
        })
    }

    /// Override the disconnect timeout (default 2s).
    pub fn with_disconnect_timeout(mut self, timeout: Duration) -> Self {
        self.disconnect_timeout = timeout;
        self
    }

    /// Subscribe to a broadcast channel and start draining its
    /// snapshots into the deque. May be called multiple times to
    /// re-subscribe after a disconnect.
    pub fn connect(&self, mut receiver: broadcast::Receiver<TrackSnapshot>) {
        {
            let mut s = self.state.lock().expect("bridge state poisoned");
            s.sender_alive = true;
            // Treat reconnect as a fresh start for connection status.
            s.last_arrival = Some(Instant::now());
        }
        let state = Arc::clone(&self.state);
        let high_water = self.high_water;
        self.runtime.spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(snap) => push_snapshot(&state, snap, high_water),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // The broadcast channel lagged on its own; mark
                        // the bridge as lagging so the indicator
                        // reflects upstream pressure.
                        let mut s = state.lock().expect("bridge state poisoned");
                        s.lagging = true;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let mut s = state.lock().expect("bridge state poisoned");
                        s.sender_alive = false;
                        return;
                    }
                }
            }
        });
    }

    /// Move all newly-arrived snapshots into `buffer`.
    /// Returns the number of snapshots transferred.
    pub fn drain_into(&self, buffer: &mut Vec<TrackSnapshot>) -> usize {
        let mut s = self.state.lock().expect("bridge state poisoned");
        let n = s.snapshots.len();
        buffer.extend(s.snapshots.drain(..));
        // Re-arm the lagging flag once we've drained back below half
        // the high-water mark.
        if s.lagging && s.snapshots.len() <= self.high_water / 2 {
            s.lagging = false;
        }
        n
    }

    /// Current connection status.
    pub fn status(&self) -> ConnectionStatus {
        let s = self.state.lock().expect("bridge state poisoned");
        if !s.sender_alive {
            return ConnectionStatus::Disconnected;
        }
        if let Some(last) = s.last_arrival {
            if last.elapsed() > self.disconnect_timeout {
                return ConnectionStatus::Disconnected;
            }
        } else {
            return ConnectionStatus::Disconnected;
        }
        if s.lagging {
            ConnectionStatus::Lagging
        } else {
            ConnectionStatus::Connected
        }
    }

    /// Number of snapshots currently buffered.
    pub fn buffered_len(&self) -> usize {
        self.state
            .lock()
            .expect("bridge state poisoned")
            .snapshots
            .len()
    }
}

fn push_snapshot(state: &Arc<Mutex<BridgeState>>, snap: TrackSnapshot, high_water: usize) {
    let mut s = state.lock().expect("bridge state poisoned");
    s.last_arrival = Some(Instant::now());
    s.snapshots.push_back(snap);
    while s.snapshots.len() > high_water {
        s.snapshots.pop_front();
        s.lagging = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use thresh_tracker::streaming::{TrackSnapshot, TrackState};

    fn snap(ts: f64) -> TrackSnapshot {
        TrackSnapshot {
            timestamp: ts,
            tracks: vec![TrackState {
                id: 1,
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                covariance_diag: [1.0; 6],
                is_confirmed: true,
            }],
            frames_dropped: 0,
        }
    }

    /// Wait up to `timeout` for `pred` to become true. Polls every 10ms.
    fn wait_for(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if pred() {
                return true;
            }
            sleep(Duration::from_millis(10));
        }
        pred()
    }

    #[test]
    fn round_trip_via_real_broadcast_channel() {
        let bridge = SnapshotBridge::with_capacity(16).expect("bridge construction");
        let (tx, rx) = broadcast::channel::<TrackSnapshot>(32);
        bridge.connect(rx);

        for i in 0..5 {
            tx.send(snap(i as f64)).expect("send");
        }

        assert!(
            wait_for(Duration::from_secs(2), || bridge.buffered_len() >= 5),
            "expected at least 5 buffered snapshots; got {}",
            bridge.buffered_len()
        );

        let mut drained = Vec::new();
        let n = bridge.drain_into(&mut drained);
        assert_eq!(n, drained.len());
        assert_eq!(drained.len(), 5);
        assert_eq!(bridge.status(), ConnectionStatus::Connected);
    }

    #[test]
    fn lagging_when_buffer_overflows_and_recovers_after_drain() {
        let bridge = SnapshotBridge::with_capacity(8).expect("bridge construction");
        let (tx, rx) = broadcast::channel::<TrackSnapshot>(64);
        bridge.connect(rx);

        for i in 0..32 {
            tx.send(snap(i as f64)).expect("send");
        }

        // Wait for drainer to ingest enough that we've crossed high-water.
        assert!(
            wait_for(Duration::from_secs(2), || bridge.status()
                == ConnectionStatus::Lagging),
            "expected Lagging once buffer overflows; status={:?}",
            bridge.status()
        );
        // Buffer is bounded.
        assert!(bridge.buffered_len() <= 8);

        // Draining past the half-high-water mark must clear Lagging.
        let mut drained = Vec::new();
        bridge.drain_into(&mut drained);
        assert_eq!(bridge.status(), ConnectionStatus::Connected);
    }

    #[test]
    fn disconnects_when_sender_dropped() {
        let bridge = SnapshotBridge::with_capacity(4).expect("bridge construction");
        let (tx, rx) = broadcast::channel::<TrackSnapshot>(8);
        bridge.connect(rx);

        tx.send(snap(0.0)).expect("send");
        assert!(wait_for(Duration::from_secs(2), || bridge.buffered_len() >= 1));

        drop(tx);
        assert!(
            wait_for(Duration::from_secs(2), || bridge.status()
                == ConnectionStatus::Disconnected),
            "expected Disconnected after sender dropped; status={:?}",
            bridge.status()
        );
    }

    #[test]
    fn disconnects_when_no_recent_arrival() {
        let bridge = SnapshotBridge::with_capacity(4)
            .expect("bridge construction")
            .with_disconnect_timeout(Duration::from_millis(50));
        let (tx, rx) = broadcast::channel::<TrackSnapshot>(8);
        bridge.connect(rx);

        tx.send(snap(0.0)).expect("send");
        // Wait past the disconnect timeout without sending more.
        sleep(Duration::from_millis(150));
        assert_eq!(bridge.status(), ConnectionStatus::Disconnected);
    }
}
