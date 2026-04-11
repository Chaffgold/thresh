//! Multi-object tracker with per-track recentering ENU frames.
//!
//! Each track maintains its own local East-North-Up origin. When the track
//! centroid drifts more than [`RecenteringPolicy::drift_threshold_m`] from its
//! origin, the tracker picks a new origin (the track's current geodetic
//! position) and transforms the 6-D CV state and covariance into the new ENU
//! frame via an ECEF round-trip. This keeps the flat-Earth approximation
//! bounded per-track, which matters for OTHR-class sensors whose targets can
//! sit thousands of kilometres from the transmitter and traverse similar
//! distances over a track's lifetime.
//!
//! Compared to [`crate::stereographic_tracker::MultiObjectTrackerStereographic`]
//! (a single global projection) and
//! [`crate::great_circle_tracker::MultiObjectTrackerGreatCircle`] (a full
//! geodetic state), this variant keeps the per-track filter linear-Gaussian
//! on flat ENU axes and pays the cost of transforming each measurement into
//! the relevant track's local frame.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use thresh_association::hungarian::hungarian_assignment;

use crate::cost_matrix::{LinearTrack, build_cost_matrix, predict_linear, record_hit, record_miss};
use thresh_core::geodetic::{ecef_to_enu, ecef_to_wgs84, enu_to_ecef, wgs84_to_ecef};
use thresh_core::measurement::Measurement;
use thresh_core::othr::{OthrSensorRegistration, othr_to_geodetic};
use thresh_core::track::{TargetClass, TrackId, TrackState};
use thresh_filter::kf::KalmanFilter;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::{LinearModel, MotionModel};

// ── Constants ──────────────────────────────────────────────────────────────

/// Minimum hits for a tentative track to be confirmed.
const CONFIRM_HITS: usize = 3;
/// Maximum consecutive misses before an alive track is deleted.
const MAX_MISSES: usize = 5;

// ── Track (Task 8.C.1) ─────────────────────────────────────────────────────

/// A single track maintained by [`MultiObjectTrackerRecenteredEnu`].
///
/// State layout: `[x, vx, y, vy, z, vz]` in the track's local ENU frame
/// anchored at `(origin_lat_rad, origin_lon_rad, origin_alt_m)`.
#[derive(Debug, Clone)]
pub struct RecenteredEnuTrack {
    /// Unique track identifier.
    pub id: TrackId,
    /// 6-D state vector `[x, vx, y, vy, z, vz]` in the local ENU frame.
    pub state: DVector<f64>,
    /// 6x6 state covariance.
    pub covariance: DMatrix<f64>,
    /// Lifecycle state.
    pub lifecycle: TrackState,
    /// Classification label.
    pub class: TargetClass,
    /// Total associated hits over the track's lifetime.
    pub hits: usize,
    /// Consecutive misses.
    pub misses: usize,
    /// ENU origin latitude (radians) for this track.
    pub origin_lat_rad: f64,
    /// ENU origin longitude (radians) for this track.
    pub origin_lon_rad: f64,
    /// ENU origin altitude (metres) for this track.
    pub origin_alt_m: f64,
}

impl RecenteredEnuTrack {
    /// Build a new track rooted at the supplied ENU origin.
    pub fn new(
        state: DVector<f64>,
        covariance: DMatrix<f64>,
        class: TargetClass,
        origin_lat_rad: f64,
        origin_lon_rad: f64,
        origin_alt_m: f64,
    ) -> Self {
        Self {
            id: TrackId::new(),
            state,
            covariance,
            lifecycle: TrackState::Tentative,
            class,
            hits: 1,
            misses: 0,
            origin_lat_rad,
            origin_lon_rad,
            origin_alt_m,
        }
    }

    /// `true` if the track has not been marked for deletion.
    pub fn is_alive(&self) -> bool {
        self.lifecycle != TrackState::Deleted
    }

    /// Current geodetic position recovered from the local ENU state.
    ///
    /// Returns `(lat_rad, lon_rad, alt_m)`.
    pub fn geodetic_position(&self) -> (f64, f64, f64) {
        let enu = Vector3::new(self.state[0], self.state[2], self.state[4]);
        let ecef = enu_to_ecef(
            &enu,
            self.origin_lat_rad,
            self.origin_lon_rad,
            self.origin_alt_m,
        );
        ecef_to_wgs84(&ecef)
    }
}

impl LinearTrack for RecenteredEnuTrack {
    fn is_alive(&self) -> bool {
        self.is_alive()
    }
    fn state(&self) -> &DVector<f64> {
        &self.state
    }
    fn state_mut(&mut self) -> &mut DVector<f64> {
        &mut self.state
    }
    fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }
    fn covariance_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.covariance
    }
}

// ── Recentering policy (Task 8.C.2) ────────────────────────────────────────

/// Configuration for when a track should be recentered onto a fresh ENU origin.
#[derive(Debug, Clone, Copy)]
pub struct RecenteringPolicy {
    /// Horizontal drift threshold in metres. If the horizontal distance
    /// `sqrt(x² + y²)` of the track state from its current origin exceeds this
    /// value, the track is recentered.
    pub drift_threshold_m: f64,
}

impl Default for RecenteringPolicy {
    fn default() -> Self {
        Self {
            drift_threshold_m: 200_000.0,
        }
    }
}

impl RecenteringPolicy {
    /// Returns `true` when `track` has drifted beyond the configured threshold
    /// and should be recentered onto a new origin.
    pub fn should_recenter(&self, track: &RecenteredEnuTrack) -> bool {
        let x = track.state[0];
        let y = track.state[2];
        (x * x + y * y).sqrt() >= self.drift_threshold_m
    }
}

// ── ENU rotation helper ────────────────────────────────────────────────────

/// Build the ECEF→ENU rotation matrix at the given reference geodetic point.
///
/// Rows are East, North, Up respectively. This matches the convention used in
/// [`thresh_core::geodetic`]; we duplicate it here so the rotation can be
/// composed explicitly when transforming one ENU frame into another.
fn enu_rotation(lat_rad: f64, lon_rad: f64) -> Matrix3<f64> {
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();
    Matrix3::new(
        -sin_lon,
        cos_lon,
        0.0,
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    )
}

// ── Recenter operation (Task 8.C.3) ────────────────────────────────────────

/// Recenter a track onto a new ENU origin.
///
/// Both the position `[x, y, z]` and the velocity `[vx, vy, vz]` blocks of the
/// state are transformed: positions round-trip through ECEF, velocities rotate
/// by the ECEF-composed rotation matrix `R_new_ecef2enu * R_old_enu2ecef`.
/// The covariance is updated as `R P R^T` with the 6x6 block-diagonal rotation.
pub fn recenter_track(
    track: &mut RecenteredEnuTrack,
    new_origin_lat_rad: f64,
    new_origin_lon_rad: f64,
    new_origin_alt_m: f64,
) {
    // 1. Position: old ENU → ECEF → new ENU.
    let old_enu_pos = Vector3::new(track.state[0], track.state[2], track.state[4]);
    let ecef_pos = enu_to_ecef(
        &old_enu_pos,
        track.origin_lat_rad,
        track.origin_lon_rad,
        track.origin_alt_m,
    );
    let new_enu_pos = ecef_to_enu(
        &ecef_pos,
        new_origin_lat_rad,
        new_origin_lon_rad,
        new_origin_alt_m,
    );

    // 2. Velocity: rotate by R_new(ECEF→ENU) * R_old(ENU→ECEF).
    // enu_rotation returns ECEF→ENU; its transpose is ENU→ECEF.
    let r_old = enu_rotation(track.origin_lat_rad, track.origin_lon_rad);
    let r_new = enu_rotation(new_origin_lat_rad, new_origin_lon_rad);
    let rot3: Matrix3<f64> = r_new * r_old.transpose();

    let old_vel = Vector3::new(track.state[1], track.state[3], track.state[5]);
    let new_vel = rot3 * old_vel;

    // 3. Rebuild state in [x, vx, y, vy, z, vz] order.
    let mut new_state = DVector::<f64>::zeros(6);
    new_state[0] = new_enu_pos.x;
    new_state[1] = new_vel.x;
    new_state[2] = new_enu_pos.y;
    new_state[3] = new_vel.y;
    new_state[4] = new_enu_pos.z;
    new_state[5] = new_vel.z;

    // 4. Build the 6x6 block rotation respecting the interleaved
    // [pos, vel, pos, vel, pos, vel] layout.
    let r6 = interleaved_rotation6(&rot3);
    let new_cov = &r6 * &track.covariance * r6.transpose();

    track.state = new_state;
    track.covariance = new_cov;
    track.origin_lat_rad = new_origin_lat_rad;
    track.origin_lon_rad = new_origin_lon_rad;
    track.origin_alt_m = new_origin_alt_m;
}

/// Lift a 3x3 rotation into the 6x6 rotation that acts on a state vector with
/// layout `[x, vx, y, vy, z, vz]`. The rotation applies to positions and
/// velocities independently (same 3x3 on each) but the interleaving means the
/// 6x6 is not simply a block-diagonal of two 3x3 blocks.
fn interleaved_rotation6(r3: &Matrix3<f64>) -> DMatrix<f64> {
    let mut r6 = DMatrix::<f64>::zeros(6, 6);
    // Map from 6-D interleaved index to axis index: 0,2,4 are positions,
    // 1,3,5 are velocities.
    let pos_idx = [0usize, 2, 4];
    let vel_idx = [1usize, 3, 5];
    for i in 0..3 {
        for j in 0..3 {
            r6[(pos_idx[i], pos_idx[j])] = r3[(i, j)];
            r6[(vel_idx[i], vel_idx[j])] = r3[(i, j)];
        }
    }
    r6
}

// ── Measurement helper (Task 8.C.4) ────────────────────────────────────────

/// Convert an OTHR measurement into a 3-D Cartesian detection `[x, y, z]`
/// expressed in the supplied ENU frame.
///
/// The target geodetic position is obtained via Vincenty direct from the
/// transmitter and then lowered into ECEF / ENU. Returns `None` for any
/// non-OTHR measurement variant.
pub fn othr_to_local_enu(
    measurement: &Measurement,
    registration: &OthrSensorRegistration,
    assumed_alt_m: f64,
    origin_lat_rad: f64,
    origin_lon_rad: f64,
    origin_alt_m: f64,
) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            let (lat, lon) = othr_to_geodetic(registration, *ground_range_m, *azimuth_rad);
            let ecef = wgs84_to_ecef(lat, lon, assumed_alt_m);
            let enu = ecef_to_enu(&ecef, origin_lat_rad, origin_lon_rad, origin_alt_m);
            Some(DVector::from_column_slice(&[enu.x, enu.y, enu.z]))
        }
        _ => None,
    }
}

// ── Tracker (Task 8.C.4) ───────────────────────────────────────────────────

/// Multi-object tracker with per-track recentering ENU frames.
pub struct MultiObjectTrackerRecenteredEnu {
    /// Active tracks.
    pub tracks: Vec<RecenteredEnuTrack>,
    /// Mahalanobis-squared gate threshold.
    pub gate_threshold: f64,
    /// Recentering policy applied after each update.
    pub recentering_policy: RecenteringPolicy,
    /// Per-axis CV acceleration-noise standard deviation (m/s²).
    pub process_noise_sigma: f64,
    /// 1-sigma measurement noise for each ENU position axis (metres).
    pub measurement_noise_sigma: f64,
    /// OTHR sensor registration (used to lift measurements into geodetic).
    pub registration: OthrSensorRegistration,
    /// Assumed target altitude for lifting OTHR detections into 3-D.
    pub assumed_alt_m: f64,
}

impl MultiObjectTrackerRecenteredEnu {
    /// Construct a new tracker with sensible OTHR defaults.
    pub fn new(registration: OthrSensorRegistration, assumed_alt_m: f64) -> Self {
        Self {
            tracks: Vec::new(),
            gate_threshold: 50.0,
            recentering_policy: RecenteringPolicy::default(),
            process_noise_sigma: 5.0,
            measurement_noise_sigma: 20_000.0,
            registration,
            assumed_alt_m,
        }
    }

    /// Number of alive tracks.
    pub fn alive_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_alive()).count()
    }

    /// Number of confirmed tracks.
    pub fn confirmed_count(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .count()
    }

    /// Observation matrix mapping `[x, vx, y, vy, z, vz]` to `[x, y, z]`.
    fn observation_matrix() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            ],
        )
    }

    fn measurement_noise(&self) -> DMatrix<f64> {
        let var = self.measurement_noise_sigma * self.measurement_noise_sigma;
        DMatrix::from_diagonal(&DVector::from_column_slice(&[var, var, var]))
    }

    /// Run one predict → associate → update → recenter → lifecycle cycle.
    pub fn step(&mut self, measurements: &[Measurement], dt: f64) {
        // 1. Predict each alive track in its own ENU frame.
        let cv = ConstantVelocity::new(self.process_noise_sigma);
        let f = cv.transition_matrix(dt);
        let q = cv.process_noise(dt);
        for track in self.tracks.iter_mut() {
            if !track.is_alive() {
                continue;
            }
            let (s, c) = predict_linear(&track.state, &track.covariance, &f, &q);
            track.state = s;
            track.covariance = c;
        }

        // 2. Collect alive track indices.
        let alive: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_alive())
            .map(|(i, _)| i)
            .collect();

        let h = Self::observation_matrix();
        let r = self.measurement_noise();

        // 3. For each alive track, transform every OTHR measurement into that
        // track's local ENU frame. This yields a per-track detection vector
        // (so measurement indices align across tracks, with `None` placeholders
        // for non-OTHR measurements).
        let mut per_track_dets: Vec<Vec<DVector<f64>>> = Vec::with_capacity(alive.len());
        let mut meas_is_othr = vec![false; measurements.len()];
        for (mi, m) in measurements.iter().enumerate() {
            if matches!(m, Measurement::Othr { .. }) {
                meas_is_othr[mi] = true;
            }
        }
        let othr_indices: Vec<usize> = (0..measurements.len())
            .filter(|&i| meas_is_othr[i])
            .collect();

        for &ti in &alive {
            let track = &self.tracks[ti];
            let mut dets = Vec::with_capacity(othr_indices.len());
            for &mi in &othr_indices {
                if let Some(det) = othr_to_local_enu(
                    &measurements[mi],
                    &self.registration,
                    self.assumed_alt_m,
                    track.origin_lat_rad,
                    track.origin_lon_rad,
                    track.origin_alt_m,
                ) {
                    dets.push(det);
                } else {
                    // Shouldn't happen because we prefiltered, but keep the
                    // shape consistent.
                    dets.push(DVector::zeros(3));
                }
            }
            per_track_dets.push(dets);
        }

        // 4. Build a per-track cost matrix row by row. Because every track has
        // its own local detection vector, we cannot use the shared
        // `build_track_cost_matrix` helper; instead we build one row at a time
        // against a single-track predicted-observation / innovation-cov pair
        // via `build_cost_matrix`, then stitch the rows back together.
        let n_alive = alive.len();
        let n_det = othr_indices.len();
        let mut cost_matrix: Vec<Vec<f64>> = vec![vec![self.gate_threshold; n_det]; n_alive];
        for (ai, &ti) in alive.iter().enumerate() {
            let track = &self.tracks[ti];
            let z_hat = vec![&h * &track.state];
            let s = vec![&h * &track.covariance * h.transpose() + &r];
            let row = build_cost_matrix(&z_hat, &s, &per_track_dets[ai], self.gate_threshold);
            if let Some(first) = row.into_iter().next() {
                cost_matrix[ai] = first;
            }
        }

        // 5. Hungarian assignment.
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        let mut associated_tracks = vec![false; n_alive];
        let mut associated_dets = vec![false; n_det];

        for &(ai, dj) in &result.matches {
            if ai >= n_alive || dj >= n_det {
                continue;
            }
            if cost_matrix[ai][dj] >= self.gate_threshold {
                continue;
            }
            associated_tracks[ai] = true;
            associated_dets[dj] = true;

            let ti = alive[ai];
            let det = per_track_dets[ai][dj].clone();
            let mut kf = KalmanFilter::new(
                self.tracks[ti].state.clone(),
                self.tracks[ti].covariance.clone(),
            );
            kf.update(&det, &h, &r);
            let t = &mut self.tracks[ti];
            t.state = kf.x;
            t.covariance = kf.p;
            record_hit(&mut t.hits, &mut t.misses, &mut t.lifecycle, CONFIRM_HITS);
        }

        // 6. Lifecycle bookkeeping for unassociated tracks.
        for (ai, &ti) in alive.iter().enumerate() {
            if !associated_tracks[ai] {
                let t = &mut self.tracks[ti];
                record_miss(&mut t.misses, &mut t.lifecycle, MAX_MISSES);
            }
        }

        // 7. Recenter any track that has drifted beyond the policy threshold.
        for track in self.tracks.iter_mut() {
            if !track.is_alive() {
                continue;
            }
            if self.recentering_policy.should_recenter(track) {
                let (lat, lon, alt) = track.geodetic_position();
                recenter_track(track, lat, lon, alt);
            }
        }

        // 8. Birth new tracks from unassociated OTHR detections. A newborn
        // track uses the transmitter as its initial ENU origin, so the
        // associated local position is simply the ECEF→ENU of the Vincenty
        // ground-truth at the transmitter.
        for (dj, &mi) in othr_indices.iter().enumerate() {
            if associated_dets[dj] {
                continue;
            }
            if let Some(det) = othr_to_local_enu(
                &measurements[mi],
                &self.registration,
                self.assumed_alt_m,
                self.registration.transmitter_lat_rad,
                self.registration.transmitter_lon_rad,
                self.registration.transmitter_alt_m,
            ) {
                self.birth_track(&det);
            }
        }

        // 9. Drop deleted tracks.
        self.tracks.retain(|t| t.lifecycle != TrackState::Deleted);
    }

    fn birth_track(&mut self, detection: &DVector<f64>) {
        let mut state = DVector::zeros(6);
        state[0] = detection[0];
        state[2] = detection[1];
        state[4] = detection[2];
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[
            1.0e8, // x position (10 km std)
            1.0e4, // vx
            1.0e8, // y position
            1.0e4, // vy
            1.0e6, // z/alt (1 km std)
            1.0e2, // vz
        ]));
        self.tracks.push(RecenteredEnuTrack::new(
            state,
            cov,
            TargetClass::Unknown,
            self.registration.transmitter_lat_rad,
            self.registration.transmitter_lon_rad,
            self.registration.transmitter_alt_m,
        ));
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_reg() -> OthrSensorRegistration {
        OthrSensorRegistration {
            transmitter_lat_rad: 20.0_f64.to_radians(),
            transmitter_lon_rad: 0.0,
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        }
    }

    #[test]
    fn policy_triggers_beyond_threshold() {
        let policy = RecenteringPolicy::default();
        let mut state = DVector::<f64>::zeros(6);
        state[0] = 150_000.0;
        state[2] = 150_000.0; // sqrt(2)*150km ~= 212 km
        let track = RecenteredEnuTrack::new(
            state,
            DMatrix::identity(6, 6),
            TargetClass::Unknown,
            0.0,
            0.0,
            0.0,
        );
        assert!(policy.should_recenter(&track));
    }

    #[test]
    fn policy_silent_below_threshold() {
        let policy = RecenteringPolicy::default();
        let mut state = DVector::<f64>::zeros(6);
        state[0] = 50_000.0;
        state[2] = 0.0;
        let track = RecenteredEnuTrack::new(
            state,
            DMatrix::identity(6, 6),
            TargetClass::Unknown,
            0.0,
            0.0,
            0.0,
        );
        assert!(!policy.should_recenter(&track));
    }

    #[test]
    fn recenter_preserves_geodetic_position() {
        let reg = test_reg();
        // Start at the transmitter origin, put the state 300 km due east.
        let mut state = DVector::<f64>::zeros(6);
        state[0] = 300_000.0; // x (east)
        state[2] = 0.0; // y (north)
        state[4] = 10_000.0; // z (up / altitude)
        state[1] = 100.0; // vx
        state[3] = 50.0; // vy
        state[5] = 0.0; // vz
        let mut track = RecenteredEnuTrack::new(
            state,
            DMatrix::<f64>::identity(6, 6) * 1.0e6,
            TargetClass::Aircraft,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );

        let (lat_before, lon_before, alt_before) = track.geodetic_position();

        // Recenter onto the track's own geodetic position.
        recenter_track(&mut track, lat_before, lon_before, alt_before);

        // After recentering, the local ENU position should be (~0, ~0, ~0)
        // and the geodetic position should agree.
        assert!(
            track.state[0].abs() < 1.0,
            "x should collapse to 0, got {}",
            track.state[0]
        );
        assert!(
            track.state[2].abs() < 1.0,
            "y should collapse to 0, got {}",
            track.state[2]
        );
        let (lat_after, lon_after, alt_after) = track.geodetic_position();
        assert!((lat_after - lat_before).abs() < 1e-9);
        assert!((lon_after - lon_before).abs() < 1e-9);
        assert!((alt_after - alt_before).abs() < 1e-3);
    }

    fn make_othr(ground_range_m: f64, azimuth_rad: f64) -> Measurement {
        Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            doppler_m_s: 0.0,
            propagation_mode: thresh_core::measurement::PropagationMode::FLayer,
            time: 0.0,
            sensor_id: 0,
        }
    }

    #[test]
    fn step_births_track_from_othr() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        let m = make_othr(500_000.0, 0.5);
        tracker.step(&[m], 1.0);
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].lifecycle, TrackState::Tentative);
    }

    #[test]
    fn step_handles_empty_measurements() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        tracker.step(&[], 1.0);
        assert!(tracker.tracks.is_empty());
    }

    #[test]
    fn step_filters_non_othr_measurements() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        // Radar measurements should be ignored.
        let radar = Measurement::Radar {
            range: 1000.0,
            azimuth: 0.0,
            elevation: 0.0,
            range_rate: None,
            time: 0.0,
            sensor_id: 0,
        };
        tracker.step(&[radar], 1.0);
        assert!(tracker.tracks.is_empty());
    }

    #[test]
    fn track_confirms_after_repeated_associations() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        let m = make_othr(500_000.0, 0.5);
        for _ in 0..6 {
            tracker.step(std::slice::from_ref(&m), 1.0);
        }
        assert!(
            tracker
                .tracks
                .iter()
                .any(|t| t.lifecycle == TrackState::Confirmed),
            "at least one track should be confirmed after repeated associations"
        );
    }

    #[test]
    fn track_deletes_after_repeated_misses() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        let m = make_othr(500_000.0, 0.5);
        // Birth a track.
        tracker.step(&[m], 1.0);
        let initial = tracker.tracks.len();
        assert!(initial > 0);
        // Coast many frames with no detections.
        for _ in 0..20 {
            tracker.step(&[], 1.0);
        }
        // Track should be deleted (retained list is empty or all coasting/deleted)
        let alive = tracker
            .tracks
            .iter()
            .filter(|t| t.lifecycle != TrackState::Deleted)
            .count();
        assert_eq!(
            alive, 0,
            "all tracks should be deleted after sustained misses"
        );
    }

    #[test]
    fn confirmed_then_missed_goes_coasting() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerRecenteredEnu::new(reg, 10_000.0);
        let m = make_othr(500_000.0, 0.5);
        for _ in 0..6 {
            tracker.step(std::slice::from_ref(&m), 1.0);
        }
        // Now miss exactly once.
        tracker.step(&[], 1.0);
        let any_coasting = tracker
            .tracks
            .iter()
            .any(|t| t.lifecycle == TrackState::Coasting);
        assert!(
            any_coasting,
            "confirmed track should transition to coasting"
        );
    }

    #[test]
    fn othr_to_local_enu_returns_some_for_valid_othr() {
        let reg = test_reg();
        let m = make_othr(1_000_000.0, 0.5);
        let result = othr_to_local_enu(
            &m,
            &reg,
            10_000.0,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );
        assert!(result.is_some());
    }

    #[test]
    fn othr_to_local_enu_returns_none_for_radar() {
        let reg = test_reg();
        let radar = Measurement::Radar {
            range: 1000.0,
            azimuth: 0.0,
            elevation: 0.0,
            range_rate: None,
            time: 0.0,
            sensor_id: 0,
        };
        let result = othr_to_local_enu(
            &radar,
            &reg,
            10_000.0,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
            reg.transmitter_alt_m,
        );
        assert!(result.is_none());
    }

    #[test]
    fn enu_rotation_at_origin_is_orthogonal() {
        let r = enu_rotation(0.5, 1.0);
        let rt = r.transpose();
        let i = r * rt;
        let identity = nalgebra::Matrix3::<f64>::identity();
        for row in 0..3 {
            for col in 0..3 {
                assert!((i[(row, col)] - identity[(row, col)]).abs() < 1e-12);
            }
        }
    }
}
