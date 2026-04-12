//! Multi-object tracker in ECEF (Earth-Centered, Earth-Fixed) coordinates.
//!
//! This module provides an ECEF-native variant of the Cartesian multi-object
//! tracker. The advantage over a fixed-origin ENU tracker is that ECEF is
//! valid globally, which matters for long-traverse scenarios (e.g., OTHR
//! coverage spans of several thousand kilometres) where a single ENU tangent
//! plane incurs significant curvature error.
//!
//! The tracker uses a constant-velocity motion model in ECEF with an
//! isotropic process noise. For the short propagation intervals and
//! aircraft-class dynamics this crate targets (<1 minute per step), the
//! inertial pseudo-forces that arise from treating the Earth-fixed frame as
//! inertial (centrifugal, Coriolis) contribute errors well below the process
//! noise floor, so they are folded into the noise rather than modelled
//! explicitly. For longer propagation these terms should be included.
//!
//! The module is deliberately isolated from [`crate::tracker`] so the existing
//! Cartesian tracker is untouched.

use crate::cost_matrix::{alive_indices, build_track_cost_matrix, predict_all};
use nalgebra::{DMatrix, DVector, Vector3};
use thresh_association::hungarian::hungarian_assignment;
use thresh_core::geodetic::{ecef_to_enu, enu_to_ecef, wgs84_to_ecef};
use thresh_core::measurement::Measurement;
use thresh_core::othr::{OthrSensorRegistration, othr_to_geodetic};
use thresh_core::track::{TargetClass, TrackId, TrackState};
use thresh_filter::kf::KalmanFilter;

use crate::lifecycle::{ConfirmationPolicy, DeletionPolicy, update_lifecycle};
use crate::othr_integration::othr_observation_jacobian;
use crate::track::Track;

// ── Motion model ────────────────────────────────────────────────────────────

/// Constant-velocity motion model in ECEF coordinates.
///
/// State: `[x, vx, y, vy, z, vz]` in ECEF (metres, m/s).
///
/// Note: For short propagation intervals (<1 minute) and aircraft-class
/// targets, the inertial pseudo-forces (centrifugal, Coriolis) introduce
/// errors smaller than typical process noise, so this model uses pure CV.
/// For longer propagation, consider including these terms.
#[derive(Debug, Clone, Copy)]
pub struct EcefMotionModel {
    /// Acceleration noise standard deviation (m/s²).
    pub process_noise_sigma: f64,
}

impl EcefMotionModel {
    /// Create a new ECEF CV motion model.
    pub fn new(process_noise_sigma: f64) -> Self {
        Self {
            process_noise_sigma,
        }
    }

    /// State transition matrix `F(dt)` for the 6-dim CV state.
    pub fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
        let mut f = DMatrix::<f64>::identity(6, 6);
        f[(0, 1)] = dt;
        f[(2, 3)] = dt;
        f[(4, 5)] = dt;
        f
    }

    /// Discrete-time process noise covariance `Q(dt)` for piecewise white
    /// noise acceleration.
    pub fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let q = self.process_noise_sigma * self.process_noise_sigma;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        // Standard white-noise-acceleration Q is rank-deficient (3 noise sources
        // for 6 states). Add small diagonal regularization so it is strictly
        // positive-definite for Cholesky-based filter implementations.
        let reg = 1e-6 * q;

        let mut noise = DMatrix::<f64>::identity(6, 6) * reg;
        for i in 0..3 {
            let p = i * 2;
            noise[(p, p)] += dt4 / 4.0 * q;
            noise[(p, p + 1)] = dt3 / 2.0 * q;
            noise[(p + 1, p)] = dt3 / 2.0 * q;
            noise[(p + 1, p + 1)] += dt2 * q;
        }
        noise
    }
}

// ── Track type ──────────────────────────────────────────────────────────────

/// A single track maintained in ECEF state space.
#[derive(Debug, Clone)]
pub struct EcefTrack {
    /// Globally unique track ID.
    pub id: TrackId,
    /// State vector `[x, vx, y, vy, z, vz]` in ECEF (metres, m/s).
    pub state: DVector<f64>,
    /// State covariance.
    pub covariance: DMatrix<f64>,
    /// Lifecycle state.
    pub lifecycle: TrackState,
    /// Target class.
    pub class: TargetClass,
    /// Total measurement associations ("hits").
    pub hits: usize,
    /// Consecutive misses.
    pub misses: usize,
    /// Age in tracker steps.
    pub age: usize,
}

impl EcefTrack {
    /// Create a new tentative track from an ECEF position detection.
    pub fn from_position(det: &DVector<f64>, class: TargetClass) -> Self {
        assert_eq!(det.len(), 3, "ECEF detection must be 3-D");
        let mut state = DVector::<f64>::zeros(6);
        state[0] = det[0];
        state[2] = det[1];
        state[4] = det[2];
        // Position uncertainty ~ 50 km, velocity ~ 500 m/s — conservative init
        // for long-range (OTHR-scale) detections.
        let diag = DVector::from_column_slice(&[2.5e9, 2.5e5, 2.5e9, 2.5e5, 2.5e9, 2.5e5]);
        let covariance = DMatrix::from_diagonal(&diag);
        Self {
            id: TrackId::new(),
            state,
            covariance,
            lifecycle: TrackState::Tentative,
            class,
            hits: 1,
            misses: 0,
            age: 1,
        }
    }

    /// Convert the ECEF state position to ENU relative to a user-supplied
    /// reference point (WGS84).
    pub fn to_enu(&self, ref_lat_rad: f64, ref_lon_rad: f64, ref_alt_m: f64) -> [f64; 3] {
        let ecef = Vector3::new(self.state[0], self.state[2], self.state[4]);
        let enu = ecef_to_enu(&ecef, ref_lat_rad, ref_lon_rad, ref_alt_m);
        [enu.x, enu.y, enu.z]
    }

    /// Is this track still alive (not deleted)?
    pub fn is_alive(&self) -> bool {
        self.lifecycle != TrackState::Deleted
    }
}

impl crate::cost_matrix::LinearTrack for EcefTrack {
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

// ── Tracker ─────────────────────────────────────────────────────────────────

/// Multi-object tracker that maintains state in ECEF coordinates.
pub struct MultiObjectTrackerEcef {
    /// Active tracks.
    pub tracks: Vec<EcefTrack>,
    /// Gating threshold (chi-squared, 3 DOF).
    pub gate_threshold: f64,
    /// ECEF CV motion model.
    pub motion_model: EcefMotionModel,
    /// Measurement noise R for 3-D ECEF position detections.
    pub measurement_noise: DMatrix<f64>,
    /// Observation matrix mapping state `[x, vx, y, vy, z, vz]` to
    /// observation `[x, y, z]`.
    pub observation_matrix: DMatrix<f64>,
    /// Confirmation policy.
    pub confirmation: ConfirmationPolicy,
    /// Deletion policy.
    pub deletion: DeletionPolicy,
}

impl MultiObjectTrackerEcef {
    /// Create a new ECEF tracker with a linear position-observation model.
    ///
    /// * `measurement_noise_sigma` — 1-σ position uncertainty in ECEF (metres)
    /// * `gate_threshold` — chi-squared gate (3 DOF)
    /// * `process_noise_sigma` — acceleration noise (m/s²)
    pub fn new(
        measurement_noise_sigma: f64,
        gate_threshold: f64,
        process_noise_sigma: f64,
    ) -> Self {
        let h = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            ],
        );
        let r =
            DMatrix::<f64>::identity(3, 3) * (measurement_noise_sigma * measurement_noise_sigma);

        Self {
            tracks: Vec::new(),
            gate_threshold,
            motion_model: EcefMotionModel::new(process_noise_sigma),
            measurement_noise: r,
            observation_matrix: h,
            confirmation: ConfirmationPolicy::new(3, 5),
            deletion: DeletionPolicy::new(5),
        }
    }

    /// Run one tracking cycle: predict, associate, update, lifecycle, birth.
    ///
    /// `detections` are 3-D ECEF position vectors (metres).
    pub fn step(&mut self, detections: &[DVector<f64>], dt: f64) {
        // 1. Predict
        let f = self.motion_model.transition_matrix(dt);
        let q = self.motion_model.process_noise(dt);
        predict_all(&mut self.tracks, &f, &q);

        // 2. Cost matrix
        let alive = alive_indices(&self.tracks);
        let h = &self.observation_matrix;
        let r = &self.measurement_noise;
        let cost_matrix =
            build_track_cost_matrix(&self.tracks, &alive, h, r, detections, self.gate_threshold);

        // 3. Hungarian assignment.
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; detections.len()];

        // 4. Update matched tracks.
        for &(ai, dj) in &result.matches {
            associated_tracks[ai] = true;
            associated_dets[dj] = true;

            let ti = alive[ai];
            let track = &self.tracks[ti];
            let mut kf = KalmanFilter::new(track.state.clone(), track.covariance.clone());
            kf.update(&detections[dj], h, r);
            self.tracks[ti].state = kf.x;
            self.tracks[ti].covariance = kf.p;
        }

        // 5. Lifecycle updates. We reuse the Track-based helper by mirroring
        // state into a temporary Track, running the policy, then syncing the
        // lifecycle fields back — this keeps the confirmation/deletion
        // semantics consistent with the ENU tracker without duplicating the
        // policy logic.
        for (ai, &ti) in alive.iter().enumerate() {
            let was_associated = associated_tracks[ai];
            let et = &mut self.tracks[ti];
            let mut tmp = Track {
                id: et.id,
                state: et.state.clone(),
                covariance: et.covariance.clone(),
                lifecycle: et.lifecycle,
                class: et.class,
                hit_streak: if was_associated { 1 } else { 0 },
                total_hits: et.hits,
                coast_count: et.misses,
                age: et.age,
                history: Vec::new(),
                max_history: 1,
                dominant_mode: None,
                mode_probabilities: None,
                imm_key: None,
            };
            update_lifecycle(&mut tmp, was_associated, &self.confirmation, &self.deletion);
            et.hits = tmp.total_hits;
            et.misses = tmp.coast_count;
            et.age = tmp.age;
            et.lifecycle = tmp.lifecycle;
        }

        // 6. Birth new tracks from unassigned detections.
        for (dj, det) in detections.iter().enumerate() {
            if !associated_dets[dj] {
                self.tracks
                    .push(EcefTrack::from_position(det, TargetClass::Unknown));
            }
        }

        // 7. Remove deleted tracks.
        self.tracks.retain(|t| t.lifecycle != TrackState::Deleted);
    }

    /// Number of confirmed tracks.
    pub fn confirmed_count(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .count()
    }

    /// Number of alive tracks.
    pub fn alive_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_alive()).count()
    }
}

// ── Observation models ──────────────────────────────────────────────────────

/// Build the OTHR observation Jacobian (3x6) at a given ECEF state.
///
/// Maps state `[x, vx, y, vy, z, vz]` (ECEF) to observation
/// `[ground_range, azimuth, doppler]` measured from the OTHR transmitter.
///
/// Implementation: project the state (position and velocity) into the local
/// ENU frame at the transmitter and apply the flat-Earth OTHR Jacobian from
/// [`othr_observation_jacobian`]. This gives an accurate linearization for
/// targets within the nominal OTHR coverage (≤3500 km) where the flat-Earth
/// approximation around the transmitter is adequate.
pub fn othr_observation_jacobian_ecef(
    state: &DVector<f64>,
    transmitter_ecef: &[f64; 3],
    transmitter_lat_rad: f64,
    transmitter_lon_rad: f64,
) -> DMatrix<f64> {
    assert_eq!(state.len(), 6, "state must be 6-D");

    // Rotation from ECEF to ENU at the transmitter.
    let sin_lat = transmitter_lat_rad.sin();
    let cos_lat = transmitter_lat_rad.cos();
    let sin_lon = transmitter_lon_rad.sin();
    let cos_lon = transmitter_lon_rad.cos();
    let rot = nalgebra::Matrix3::new(
        -sin_lon,
        cos_lon,
        0.0,
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    );

    // Position and velocity in ENU at the transmitter.
    let dpos = Vector3::new(
        state[0] - transmitter_ecef[0],
        state[2] - transmitter_ecef[1],
        state[4] - transmitter_ecef[2],
    );
    let enu_pos = rot * dpos;
    let vel_ecef = Vector3::new(state[1], state[3], state[5]);
    let enu_vel = rot * vel_ecef;

    // Build a 6-D ENU state [x, vx, y, vy, z, vz] and evaluate the existing
    // flat-Earth OTHR Jacobian with the transmitter at the ENU origin.
    let enu_state = DVector::from_column_slice(&[
        enu_pos.x, enu_vel.x, enu_pos.y, enu_vel.y, enu_pos.z, enu_vel.z,
    ]);
    let h_enu = othr_observation_jacobian(&enu_state, &[0.0, 0.0, 0.0]);

    // Chain rule: dh/d(ECEF state) = dh/d(ENU state) * d(ENU state)/d(ECEF state).
    // The ENU-state depends linearly on the ECEF state via a block-diagonal
    // rotation that interleaves position and velocity rows:
    //   ENU[0] = rot * ECEF_pos, ENU_vel = rot * ECEF_vel
    // In the interleaved state layout this becomes a 6x6 matrix with the
    // rotation scattered across (row, col) = (2*i, 2*j) for position and
    // (2*i+1, 2*j+1) for velocity.
    let mut j = DMatrix::<f64>::zeros(6, 6);
    for i in 0..3 {
        for jj in 0..3 {
            j[(2 * i, 2 * jj)] = rot[(i, jj)];
            j[(2 * i + 1, 2 * jj + 1)] = rot[(i, jj)];
        }
    }

    h_enu * j
}

/// Convert an OTHR [`Measurement`] to an ECEF detection vector for use with
/// the ECEF tracker.
///
/// Returns `None` if the measurement is not an [`Measurement::Othr`] variant.
/// The `estimated_alt_m` argument is the assumed target altitude (since OTHR
/// has no elevation observable). The ground track is computed on the WGS84
/// ellipsoid via Vincenty.
pub fn othr_to_ecef(
    measurement: &Measurement,
    registration: &OthrSensorRegistration,
    estimated_alt_m: f64,
) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            let (lat, lon) = othr_to_geodetic(registration, *ground_range_m, *azimuth_rad);
            let ecef = wgs84_to_ecef(lat, lon, estimated_alt_m);
            Some(DVector::from_column_slice(&[ecef.x, ecef.y, ecef.z]))
        }
        _ => None,
    }
}

/// Convert a conventional [`Measurement::Radar`] measurement to an ECEF
/// detection vector. The radar's local ENU frame is defined by its
/// `lat/lon/alt` registration.
///
/// Returns `None` if the measurement is not a [`Measurement::Radar`] variant.
///
/// Steps: extract range/az/el from the Radar variant, convert spherical
/// coordinates to ENU at the radar, then convert ENU to ECEF.
pub fn radar_to_ecef(
    measurement: &Measurement,
    radar_lat_rad: f64,
    radar_lon_rad: f64,
    radar_alt_m: f64,
) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Radar {
            range,
            azimuth,
            elevation,
            ..
        } => {
            // Spherical (range, az clockwise from north, elevation above
            // horizon) → ENU.
            let cos_el = elevation.cos();
            let sin_el = elevation.sin();
            let sin_az = azimuth.sin();
            let cos_az = azimuth.cos();
            let east = range * cos_el * sin_az;
            let north = range * cos_el * cos_az;
            let up = range * sin_el;
            let enu = Vector3::new(east, north, up);
            let ecef = enu_to_ecef(&enu, radar_lat_rad, radar_lon_rad, radar_alt_m);
            Some(DVector::from_column_slice(&[ecef.x, ecef.y, ecef.z]))
        }
        _ => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use thresh_core::measurement::PropagationMode;

    #[test]
    fn motion_model_transition_is_cv() {
        let m = EcefMotionModel::new(1.0);
        let f = m.transition_matrix(2.0);
        assert_eq!(f.nrows(), 6);
        assert_eq!(f.ncols(), 6);
        // x_new = x + vx * dt
        assert!((f[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((f[(2, 3)] - 2.0).abs() < 1e-12);
        assert!((f[(4, 5)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn process_noise_is_positive_definite() {
        let m = EcefMotionModel::new(5.0);
        let q = m.process_noise(1.0);
        assert!(q.clone().cholesky().is_some());
    }

    #[test]
    fn tracker_births_and_confirms() {
        let mut tr = MultiObjectTrackerEcef::new(100.0, 50.0, 5.0);
        let det = DVector::from_column_slice(&[6_378_137.0, 0.0, 0.0]);
        for _ in 0..4 {
            tr.step(std::slice::from_ref(&det), 1.0);
        }
        assert!(tr.confirmed_count() >= 1);
    }

    #[test]
    fn radar_to_ecef_roundtrip() {
        // Radar at (lat=0, lon=0, alt=0). Target 10 km due north at 0° elev.
        let m = Measurement::Radar {
            range: 10_000.0,
            azimuth: 0.0,
            elevation: 0.0,
            range_rate: None,
            time: 0.0,
            sensor_id: 0,
        };
        let det = radar_to_ecef(&m, 0.0, 0.0, 0.0).expect("radar");
        // The target should be slightly above the equator, near the
        // prime meridian, with x near WGS84_A.
        assert!((det[0] - 6_378_137.0).abs() < 20.0, "x: {}", det[0]);
        assert!(det[1].abs() < 1.0, "y near 0: {}", det[1]);
        assert!(
            det[2] > 9_990.0 && det[2] < 10_010.0,
            "z ~ 10 km: {}",
            det[2]
        );
    }

    #[test]
    fn othr_to_ecef_basic() {
        let reg = OthrSensorRegistration {
            transmitter_lat_rad: 0.0,
            transmitter_lon_rad: 0.0,
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        };
        let m = Measurement::Othr {
            ground_range_m: 2_000_000.0,
            azimuth_rad: 0.0,
            doppler_m_s: 0.0,
            propagation_mode: PropagationMode::FLayer,
            time: 0.0,
            sensor_id: 0,
        };
        let det = othr_to_ecef(&m, &reg, 10_000.0).expect("othr");
        assert_eq!(det.len(), 3);
        // Due north ~2000 km from (0,0) — latitude ~18°, longitude ~0°
        // Position magnitude should be ~ WGS84_A.
        let r = (det[0] * det[0] + det[1] * det[1] + det[2] * det[2]).sqrt();
        assert!((r - 6_378_137.0).abs() < 30_000.0, "ECEF magnitude: {r}");
    }

    #[test]
    fn to_enu_at_point_converts() {
        let mut track = EcefTrack::from_position(
            &DVector::from_column_slice(&[6_378_137.0, 0.0, 0.0]),
            TargetClass::Aircraft,
        );
        track.state[0] = 6_378_137.0;
        track.state[2] = 0.0;
        track.state[4] = 0.0;
        // Reference = same point: ENU must be ~0.
        let enu = track.to_enu(0.0, 0.0, 0.0);
        assert!(enu[0].abs() < 1e-6);
        assert!(enu[1].abs() < 1e-6);
        assert!(enu[2].abs() < 1e-6);
    }

    #[test]
    fn othr_jacobian_ecef_has_correct_shape() {
        let tx_lat = 0.0;
        let tx_lon = 0.0;
        let tx_ecef_v = wgs84_to_ecef(tx_lat, tx_lon, 0.0);
        let tx_ecef = [tx_ecef_v.x, tx_ecef_v.y, tx_ecef_v.z];
        // State ~1000 km "east" of transmitter, moving east at 200 m/s.
        let state = DVector::from_column_slice(&[
            tx_ecef[0],
            0.0,
            tx_ecef[1] + 1_000_000.0,
            200.0,
            tx_ecef[2],
            0.0,
        ]);
        let j = othr_observation_jacobian_ecef(&state, &tx_ecef, tx_lat, tx_lon);
        assert_eq!(j.nrows(), 3);
        assert_eq!(j.ncols(), 6);
    }
}
