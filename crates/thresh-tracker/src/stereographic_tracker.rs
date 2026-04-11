//! Multi-object tracker that operates in a local conformal stereographic
//! projection of the Earth's surface.
//!
//! For OTHR-class sensors, targets can be thousands of kilometres from the
//! transmitter. Flat ENU frames become increasingly distorted at those ranges
//! because the local tangent plane and the curved Earth diverge. A
//! stereographic projection centred at the sensor (or at the centroid of a
//! multi-sensor coverage region) keeps angle relationships exact and distorts
//! distances only as a smooth function of the radius from the centre. For a
//! 3000 km OTHR footprint the scale factor grows to only a few percent, which
//! is much better than the multi-percent-of-altitude errors produced by ENU.
//!
//! The tracker here is intentionally a self-contained variant of
//! [`crate::tracker::MultiObjectTracker`]: state is `[x, vx, y, vy, alt, valt]`
//! with the horizontal components expressed in the stereographic plane and the
//! vertical channel left untouched.

use nalgebra::{DMatrix, DVector};
use thresh_association::hungarian::hungarian_assignment;

use crate::cost_matrix::{alive_indices, build_track_cost_matrix, predict_all};
use thresh_core::measurement::Measurement;
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct};
use thresh_core::track::{TargetClass, TrackId, TrackState};
use thresh_filter::kf::KalmanFilter;
use thresh_filter::models::cv::ConstantVelocity;
use thresh_filter::traits::{LinearModel, MotionModel};

/// Mean Earth radius (metres) used for the spherical stereographic projection.
pub const R_EARTH_M: f64 = 6_371_000.0;

// ── Stereographic projection (Task 8.D.1 / 8.D.2) ─────────────────────────

/// Conformal (spherical) stereographic projection from geodetic `(lat, lon)`
/// to a 2D plane centred at `(center_lat_rad, center_lon_rad)`.
///
/// Returned `(x, y)` are in metres, with `x` east-positive and `y`
/// north-positive at the projection centre. The mapping is angle-preserving,
/// and distances only distort through the smooth scale factor
/// `k = 2 / (1 + cos(c))` where `c` is the angular distance from the centre.
/// For a 3000 km arc that is roughly 5% — small enough that a Kalman filter
/// running in the projected plane can still use linear motion models without
/// the large tangent-plane errors you get from ENU.
pub fn stereographic_project(
    lat_rad: f64,
    lon_rad: f64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> (f64, f64) {
    let sin_lat0 = center_lat_rad.sin();
    let cos_lat0 = center_lat_rad.cos();
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let d_lon = lon_rad - center_lon_rad;
    let cos_dlon = d_lon.cos();
    let sin_dlon = d_lon.sin();

    let denom = 1.0 + sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon;
    // Guard against the antipodal singularity.
    let k = 2.0 * R_EARTH_M / denom.max(1e-12);

    let x = k * cos_lat * sin_dlon;
    let y = k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon);
    (x, y)
}

/// Inverse of [`stereographic_project`]. Returns `(lat_rad, lon_rad)`.
pub fn stereographic_inverse(
    x_m: f64,
    y_m: f64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> (f64, f64) {
    let rho = (x_m * x_m + y_m * y_m).sqrt();
    if rho < 1e-9 {
        return (center_lat_rad, center_lon_rad);
    }
    let c = 2.0 * (rho / (2.0 * R_EARTH_M)).atan();
    let sin_c = c.sin();
    let cos_c = c.cos();
    let sin_lat0 = center_lat_rad.sin();
    let cos_lat0 = center_lat_rad.cos();

    let lat = (cos_c * sin_lat0 + y_m * sin_c * cos_lat0 / rho).asin();
    let lon = center_lon_rad + (x_m * sin_c).atan2(rho * cos_lat0 * cos_c - y_m * sin_lat0 * sin_c);
    (lat, lon)
}

// ── Center selection helper (Task 8.D.5) ──────────────────────────────────

/// Recommended projection centre given one or more transmitter locations.
///
/// * For a single transmitter, the centre is simply the transmitter itself.
/// * For multiple transmitters, we return the spherical centroid: convert
///   each `(lat, lon)` to a unit vector, average, and convert back. This is
///   robust to wrap-around at the antimeridian.
///
/// Returns `(center_lat_rad, center_lon_rad)`. Panics if `transmitters` is
/// empty, matching the precondition that a tracker must have at least one
/// sensor to project around.
pub fn recommended_center(transmitters: &[(f64, f64)]) -> (f64, f64) {
    assert!(
        !transmitters.is_empty(),
        "recommended_center requires at least one transmitter"
    );
    if transmitters.len() == 1 {
        return transmitters[0];
    }

    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sz = 0.0_f64;
    for &(lat, lon) in transmitters {
        let cl = lat.cos();
        sx += cl * lon.cos();
        sy += cl * lon.sin();
        sz += lat.sin();
    }
    let n = transmitters.len() as f64;
    sx /= n;
    sy /= n;
    sz /= n;
    let hyp = (sx * sx + sy * sy).sqrt();
    let lat = sz.atan2(hyp);
    let lon = sy.atan2(sx);
    (lat, lon)
}

// ── OTHR → stereographic detection (Task 8.D.4) ───────────────────────────

/// Convert an OTHR measurement to a stereographic-plane detection vector
/// `[x, y, alt]`.
///
/// The ground range and azimuth are propagated from the transmitter location
/// via Vincenty's direct formula (ellipsoidal), the resulting geodetic point
/// is projected onto the stereographic plane centred at
/// `(center_lat_rad, center_lon_rad)`, and `estimated_alt_m` is appended as
/// the vertical channel. Returns `None` for any non-OTHR measurement variant.
pub fn othr_to_stereographic(
    measurement: &Measurement,
    registration: &OthrSensorRegistration,
    estimated_alt_m: f64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> Option<DVector<f64>> {
    match measurement {
        Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            let (lat, lon) = vincenty_direct(
                registration.transmitter_lat_rad,
                registration.transmitter_lon_rad,
                *azimuth_rad,
                *ground_range_m,
            );
            let (x, y) = stereographic_project(lat, lon, center_lat_rad, center_lon_rad);
            Some(DVector::from_column_slice(&[x, y, estimated_alt_m]))
        }
        _ => None,
    }
}

// ── Stereographic tracker (Task 8.D.3) ────────────────────────────────────

/// Minimum hits for a tentative track to be confirmed.
const CONFIRM_HITS: usize = 3;
/// Maximum consecutive misses before an alive track is deleted.
const MAX_MISSES: usize = 5;

/// A single track maintained by [`MultiObjectTrackerStereographic`].
///
/// State layout: `[x, vx, y, vy, alt, valt]`, where `(x, y)` are
/// stereographic-plane coordinates in metres.
#[derive(Debug, Clone)]
pub struct StereoTrack {
    /// Unique track identifier.
    pub id: TrackId,
    /// 6-D state vector `[x, vx, y, vy, alt, valt]`.
    pub state: DVector<f64>,
    /// 6x6 state covariance.
    pub covariance: DMatrix<f64>,
    /// Lifecycle state.
    pub lifecycle: TrackState,
    /// Classification label.
    pub class: TargetClass,
    /// Total hits (associated measurements) over the track's lifetime.
    pub hits: usize,
    /// Number of consecutive missed updates.
    pub misses: usize,
}

impl StereoTrack {
    fn is_alive(&self) -> bool {
        self.lifecycle != TrackState::Deleted
    }
}

impl crate::cost_matrix::LinearTrack for StereoTrack {
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

/// Multi-object tracker operating in a local stereographic projection.
pub struct MultiObjectTrackerStereographic {
    /// Active tracks.
    pub tracks: Vec<StereoTrack>,
    /// Projection centre latitude (radians).
    pub center_lat_rad: f64,
    /// Projection centre longitude (radians).
    pub center_lon_rad: f64,
    /// Mahalanobis-squared gate threshold.
    pub gate_threshold: f64,
    /// Per-axis process-noise standard deviation (m/s² for CV channels,
    /// m/s² for the altitude channel).
    pub process_noise_sigma: f64,
}

impl MultiObjectTrackerStereographic {
    /// Create a new tracker.
    pub fn new(center_lat_rad: f64, center_lon_rad: f64, process_noise: f64, gate: f64) -> Self {
        Self {
            tracks: Vec::new(),
            center_lat_rad,
            center_lon_rad,
            gate_threshold: gate,
            process_noise_sigma: process_noise,
        }
    }

    /// Observation matrix mapping state `[x, vx, y, vy, alt, valt]` to a
    /// position detection `[x, y, alt]`.
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

    /// Default measurement-noise matrix (20 km horizontal, 1 km vertical),
    /// matching the `new_cv_position(20_000.0, …)` preset of the ENU tracker
    /// so the two variants can be compared apples-to-apples.
    fn default_measurement_noise() -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_column_slice(&[
            20_000.0 * 20_000.0,
            20_000.0 * 20_000.0,
            1_000.0 * 1_000.0,
        ]))
    }

    /// Build a 6x6 transition matrix combining a horizontal CV model with an
    /// independent vertical CV channel.
    fn transition(&self, dt: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let cv = ConstantVelocity::new(self.process_noise_sigma);
        let f_horiz = cv.transition_matrix(dt); // 6x6 (x,vx,y,vy,z,vz)
        let q_horiz = cv.process_noise(dt);
        // The ConstantVelocity model already provides [x,vx,y,vy,z,vz] which
        // matches our state layout directly once we reinterpret the z/vz
        // channel as altitude.
        (f_horiz, q_horiz)
    }

    /// Run one predict → associate → update → lifecycle cycle.
    pub fn step(&mut self, detections: &[DVector<f64>], dt: f64) {
        // 1. Predict all alive tracks
        let (f, q) = self.transition(dt);
        predict_all(&mut self.tracks, &f, &q);

        // 2. Build Mahalanobis cost matrix
        let alive = alive_indices(&self.tracks);
        let h = Self::observation_matrix();
        let r = Self::default_measurement_noise();
        let cost_matrix = build_track_cost_matrix(
            &self.tracks,
            &alive,
            &h,
            &r,
            detections,
            self.gate_threshold,
        );

        // 3. Hungarian assignment.
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; detections.len()];

        // 4. Apply KF updates for matched pairs.
        for &(ai, dj) in &result.matches {
            associated_tracks[ai] = true;
            associated_dets[dj] = true;
            let ti = alive[ai];

            let mut kf = KalmanFilter::new(
                self.tracks[ti].state.clone(),
                self.tracks[ti].covariance.clone(),
            );
            kf.update(&detections[dj], &h, &r);
            self.tracks[ti].state = kf.x;
            self.tracks[ti].covariance = kf.p;
            self.tracks[ti].hits += 1;
            self.tracks[ti].misses = 0;

            match self.tracks[ti].lifecycle {
                TrackState::Tentative if self.tracks[ti].hits >= CONFIRM_HITS => {
                    self.tracks[ti].lifecycle = TrackState::Confirmed;
                }
                TrackState::Coasting => {
                    self.tracks[ti].lifecycle = TrackState::Confirmed;
                }
                _ => {}
            }
        }

        // 5. Lifecycle bookkeeping for unassociated tracks.
        for (ai, &ti) in alive.iter().enumerate() {
            if !associated_tracks[ai] {
                self.tracks[ti].misses += 1;
                if self.tracks[ti].misses >= MAX_MISSES {
                    self.tracks[ti].lifecycle = TrackState::Deleted;
                } else if self.tracks[ti].lifecycle == TrackState::Confirmed {
                    self.tracks[ti].lifecycle = TrackState::Coasting;
                }
            }
        }

        // 6. Birth new tracks from unassociated detections.
        for (dj, det) in detections.iter().enumerate() {
            if !associated_dets[dj] {
                self.birth_track(det);
            }
        }

        // 7. Drop deleted tracks.
        self.tracks.retain(|t| t.lifecycle != TrackState::Deleted);
    }

    fn birth_track(&mut self, detection: &DVector<f64>) {
        let mut state = DVector::zeros(6);
        state[0] = detection[0]; // x
        state[2] = detection[1]; // y
        state[4] = detection[2]; // alt
        // Velocities default to zero; give them a wide initial covariance.
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[
            1.0e8, // x position (10 km)
            1.0e4, // vx
            1.0e8, // y position
            1.0e4, // vy
            1.0e6, // alt (1 km)
            1.0e2, // valt
        ]));
        self.tracks.push(StereoTrack {
            id: TrackId::new(),
            state,
            covariance: cov,
            lifecycle: TrackState::Tentative,
            class: TargetClass::Unknown,
            hits: 1,
            misses: 0,
        });
    }

    /// Number of alive tracks (tentative + confirmed + coasting).
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_identity_at_center() {
        let lat0 = 0.3_f64;
        let lon0 = -1.1_f64;
        let (x, y) = stereographic_project(lat0, lon0, lat0, lon0);
        assert!(x.abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn inverse_roundtrip_small() {
        let lat0 = 0.3_f64;
        let lon0 = -1.1_f64;
        let lat = lat0 + 0.01;
        let lon = lon0 + 0.02;
        let (x, y) = stereographic_project(lat, lon, lat0, lon0);
        let (lat_b, lon_b) = stereographic_inverse(x, y, lat0, lon0);
        assert!((lat - lat_b).abs() < 1e-12);
        assert!((lon - lon_b).abs() < 1e-12);
    }

    #[test]
    fn recommended_center_single() {
        let (lat, lon) = recommended_center(&[(0.5, -1.2)]);
        assert_eq!(lat, 0.5);
        assert_eq!(lon, -1.2);
    }

    #[test]
    fn recommended_center_centroid() {
        let (lat, lon) = recommended_center(&[(0.0, -0.01), (0.0, 0.01)]);
        assert!(lat.abs() < 1e-12);
        assert!(lon.abs() < 1e-6);
    }
}
