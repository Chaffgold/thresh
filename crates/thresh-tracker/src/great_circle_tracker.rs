//! Great-circle motion model tracker for long-range OTHR targets.
//!
//! This tracker keeps its state in geodetic coordinates
//! `[lat, lon, alt, ground_speed, heading, climb_rate]` and advances lat/lon
//! along the current heading with Vincenty's direct formula, so long-duration
//! constant-heading flight does not accumulate flat-Earth error.
//!
//! The tracker is intentionally isolated from the Cartesian
//! [`crate::tracker::MultiObjectTracker`]; it consumes
//! [`Measurement::Othr`](thresh_core::measurement::Measurement) directly and
//! runs an EKF update with a Vincenty-based observation model.

use std::f64::consts::{PI, TAU};

use nalgebra::{DMatrix, DVector};

use thresh_association::gating::mahalanobis_squared;
use thresh_association::hungarian::hungarian_assignment;
use thresh_core::measurement::Measurement;
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct, vincenty_inverse};
use thresh_core::track::{TargetClass, TrackId, TrackState};

// ── State ──────────────────────────────────────────────────────────────────

/// Geodetic state for great-circle tracking.
///
/// State vector layout (length 6):
/// `[lat_rad, lon_rad, alt_m, ground_speed_m_s, heading_rad, climb_rate_m_s]`.
///
/// * `heading_rad` is measured clockwise from north (0 = north).
/// * `ground_speed_m_s` is the magnitude of horizontal velocity.
/// * `climb_rate_m_s` is the time derivative of altitude.
#[derive(Debug, Clone, Copy)]
pub struct GreatCircleState {
    /// Geodetic latitude (radians).
    pub lat_rad: f64,
    /// Geodetic longitude (radians).
    pub lon_rad: f64,
    /// Altitude above the WGS84 ellipsoid (meters).
    pub alt_m: f64,
    /// Horizontal ground speed (m/s).
    pub ground_speed_m_s: f64,
    /// Heading, clockwise from north (radians).
    pub heading_rad: f64,
    /// Climb rate (m/s).
    pub climb_rate_m_s: f64,
}

impl GreatCircleState {
    /// Convert to a 6-dimensional column vector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_column_slice(&[
            self.lat_rad,
            self.lon_rad,
            self.alt_m,
            self.ground_speed_m_s,
            self.heading_rad,
            self.climb_rate_m_s,
        ])
    }

    /// Build from a 6-dimensional column vector.
    pub fn from_vector(v: &DVector<f64>) -> Self {
        assert_eq!(v.len(), 6, "GreatCircleState expects a 6-dim vector");
        Self {
            lat_rad: v[0],
            lon_rad: v[1],
            alt_m: v[2],
            ground_speed_m_s: v[3],
            heading_rad: v[4],
            climb_rate_m_s: v[5],
        }
    }
}

/// Wrap an angle to the range `[-pi, pi]`.
fn wrap_angle(a: f64) -> f64 {
    let mut x = (a + PI) % TAU;
    if x < 0.0 {
        x += TAU;
    }
    x - PI
}

// ── Motion model ───────────────────────────────────────────────────────────

/// Constant-heading great-circle motion model.
#[derive(Debug, Clone, Copy)]
pub struct GreatCircleMotionModel {
    /// 1-sigma process noise on heading (rad / sqrt(s)).
    pub heading_noise_rad: f64,
    /// 1-sigma process noise on ground speed (m/s / sqrt(s)).
    pub speed_noise_m_s: f64,
}

impl Default for GreatCircleMotionModel {
    fn default() -> Self {
        Self {
            heading_noise_rad: 0.01_f64.to_radians(),
            speed_noise_m_s: 1.0,
        }
    }
}

impl GreatCircleMotionModel {
    /// Predict the next state using Vincenty's direct formula.
    ///
    /// The target is advanced along its current heading by `ground_speed * dt`.
    /// Heading, speed and climb rate are held constant (constant-heading model);
    /// altitude increases by `climb_rate * dt`.
    pub fn predict(&self, state: &GreatCircleState, dt: f64) -> GreatCircleState {
        let distance = state.ground_speed_m_s * dt;
        let (lat2, lon2) = if distance.abs() < 1e-9 {
            (state.lat_rad, state.lon_rad)
        } else {
            vincenty_direct(state.lat_rad, state.lon_rad, state.heading_rad, distance)
        };

        GreatCircleState {
            lat_rad: lat2,
            lon_rad: wrap_angle(lon2),
            alt_m: state.alt_m + state.climb_rate_m_s * dt,
            ground_speed_m_s: state.ground_speed_m_s,
            heading_rad: wrap_angle(state.heading_rad),
            climb_rate_m_s: state.climb_rate_m_s,
        }
    }

    /// Predict in vector form.
    pub fn predict_vec(&self, state: &DVector<f64>, dt: f64) -> DVector<f64> {
        self.predict(&GreatCircleState::from_vector(state), dt)
            .to_vector()
    }

    /// Compute the state-transition Jacobian via central finite differences.
    pub fn jacobian(&self, state: &GreatCircleState, dt: f64) -> DMatrix<f64> {
        let x = state.to_vector();
        let n = x.len();
        let mut j = DMatrix::<f64>::zeros(n, n);
        // Perturbations tuned per-component (angles use a small radian step,
        // distances use meter-scale steps). These scales keep the finite
        // difference well inside the linear regime of Vincenty's direct.
        let eps = [1e-7, 1e-7, 1.0, 1e-3, 1e-7, 1e-3];

        for i in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps[i];
            xm[i] -= eps[i];
            let fp = self.predict_vec(&xp, dt);
            let fm = self.predict_vec(&xm, dt);
            let mut col = (fp - fm) / (2.0 * eps[i]);
            // Angular outputs (lon, heading) can straddle the ±π seam which
            // corrupts the finite difference; collapse those back to `[-π, π]`.
            col[1] = wrap_angle_delta(col[1]);
            col[4] = wrap_angle_delta(col[4]);
            for r in 0..n {
                j[(r, i)] = col[r];
            }
        }

        j
    }

    /// Process noise covariance matrix for a `dt` step.
    pub fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        let mut q = DMatrix::<f64>::zeros(6, 6);
        // Position uncertainty grows with heading/speed noise projected over dt.
        // We use a simple diagonal block that is positive-definite and small
        // enough that the filter is driven by measurements, not process noise.
        let horiz_pos_var = (self.speed_noise_m_s * dt).powi(2) * 1e-12; // rad^2
        q[(0, 0)] = horiz_pos_var;
        q[(1, 1)] = horiz_pos_var;
        q[(2, 2)] = (self.speed_noise_m_s * dt).powi(2); // alt
        q[(3, 3)] = self.speed_noise_m_s.powi(2) * dt;
        q[(4, 4)] = self.heading_noise_rad.powi(2) * dt;
        q[(5, 5)] = (self.speed_noise_m_s * dt).powi(2) * 0.01;
        q
    }
}

/// Wrap a (possibly large) finite-difference result that represents an angular
/// derivative into `[-pi, pi]` to avoid 2π wraparound artifacts.
fn wrap_angle_delta(a: f64) -> f64 {
    // For derivatives we expect values on the order of unity; large values come
    // from a wrap discontinuity in the output. Collapse those back into
    // `[-pi, pi]` so the Jacobian entry stays finite.
    if a.abs() > PI {
        let mut x = (a + PI) % TAU;
        if x < 0.0 {
            x += TAU;
        }
        x - PI
    } else {
        a
    }
}

// ── Observation model ──────────────────────────────────────────────────────

/// Compute the predicted OTHR observation `[ground_range, azimuth, doppler]`
/// from a great-circle state, given the transmitter's geodetic position.
///
/// The ground range and azimuth are obtained from Vincenty's inverse formula
/// on the WGS84 ellipsoid. The Doppler is the ground-speed component projected
/// along the bearing from the target toward the transmitter (positive =
/// approaching the transmitter).
pub fn predict_othr_observation(
    state: &GreatCircleState,
    transmitter_lat_rad: f64,
    transmitter_lon_rad: f64,
) -> [f64; 3] {
    // Azimuth and range *from the transmitter* (this is the observable).
    let (ground_range, az_from_tx) = vincenty_inverse(
        transmitter_lat_rad,
        transmitter_lon_rad,
        state.lat_rad,
        state.lon_rad,
    );

    // Bearing *from the target* back toward the transmitter, used to resolve
    // the radial velocity component.
    let (_, az_to_tx) = vincenty_inverse(
        state.lat_rad,
        state.lon_rad,
        transmitter_lat_rad,
        transmitter_lon_rad,
    );

    // Velocity in local east-north components.
    let v_east = state.ground_speed_m_s * state.heading_rad.sin();
    let v_north = state.ground_speed_m_s * state.heading_rad.cos();
    // Unit vector from target toward transmitter.
    let u_east = az_to_tx.sin();
    let u_north = az_to_tx.cos();
    // Radial velocity (positive = approaching transmitter).
    let doppler = v_east * u_east + v_north * u_north;

    [ground_range, az_from_tx, doppler]
}

/// Compute the OTHR observation Jacobian at a great-circle state by central
/// finite differences. Maps a 6-dim state to a 3-dim observation.
fn othr_observation_jacobian(
    state: &GreatCircleState,
    transmitter_lat_rad: f64,
    transmitter_lon_rad: f64,
) -> DMatrix<f64> {
    let x = state.to_vector();
    let eps = [1e-7, 1e-7, 1.0, 1e-3, 1e-7, 1e-3];
    let mut h = DMatrix::<f64>::zeros(3, 6);

    for i in 0..6 {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps[i];
        xm[i] -= eps[i];
        let sp = GreatCircleState::from_vector(&xp);
        let sm = GreatCircleState::from_vector(&xm);
        let zp = predict_othr_observation(&sp, transmitter_lat_rad, transmitter_lon_rad);
        let zm = predict_othr_observation(&sm, transmitter_lat_rad, transmitter_lon_rad);

        // Handle azimuth wraparound in the finite difference.
        let daz = wrap_angle_delta(zp[1] - zm[1]);
        h[(0, i)] = (zp[0] - zm[0]) / (2.0 * eps[i]);
        h[(1, i)] = daz / (2.0 * eps[i]);
        h[(2, i)] = (zp[2] - zm[2]) / (2.0 * eps[i]);
    }

    h
}

// ── Track container ────────────────────────────────────────────────────────

/// A single great-circle track.
#[derive(Debug, Clone)]
pub struct GreatCircleTrack {
    /// Globally unique track identifier.
    pub id: TrackId,
    /// State vector
    /// `[lat_rad, lon_rad, alt_m, ground_speed, heading, climb_rate]`.
    pub state: DVector<f64>,
    /// State covariance.
    pub covariance: DMatrix<f64>,
    /// Lifecycle state.
    pub lifecycle: TrackState,
    /// Target classification.
    pub class: TargetClass,
    /// Total associated hits since creation.
    pub hits: usize,
    /// Consecutive misses.
    pub misses: usize,
}

impl GreatCircleTrack {
    /// Return this track's current state as a [`GreatCircleState`].
    pub fn as_state(&self) -> GreatCircleState {
        GreatCircleState::from_vector(&self.state)
    }
}

// ── Tracker ────────────────────────────────────────────────────────────────

/// Great-circle multi-object tracker for OTHR measurements.
pub struct MultiObjectTrackerGreatCircle {
    /// Active tracks.
    pub tracks: Vec<GreatCircleTrack>,
    /// Mahalanobis gate threshold (chi-squared, dim 3).
    pub gate_threshold: f64,
    /// Motion model shared across all tracks.
    pub motion_model: GreatCircleMotionModel,
    /// OTHR sensor registration (transmitter position and frequency).
    pub registration: OthrSensorRegistration,
    /// Measurement noise covariance for OTHR `[range, azimuth, doppler]`.
    pub measurement_noise: DMatrix<f64>,
    /// Assumed target altitude for initialization (m).
    pub assumed_alt_m: f64,
    /// Number of hits required to confirm a tentative track.
    pub confirmation_hits: usize,
    /// Number of consecutive misses before deletion.
    pub deletion_misses: usize,
}

impl MultiObjectTrackerGreatCircle {
    /// Create a new tracker with sensible OTHR defaults.
    pub fn new(
        registration: OthrSensorRegistration,
        measurement_noise: DMatrix<f64>,
        gate_threshold: f64,
    ) -> Self {
        Self {
            tracks: Vec::new(),
            gate_threshold,
            motion_model: GreatCircleMotionModel::default(),
            registration,
            measurement_noise,
            assumed_alt_m: 10_000.0,
            confirmation_hits: 3,
            deletion_misses: 5,
        }
    }

    /// Number of alive tracks.
    pub fn alive_count(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.lifecycle != TrackState::Deleted)
            .count()
    }

    /// Number of confirmed tracks.
    pub fn confirmed_count(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.lifecycle == TrackState::Confirmed)
            .count()
    }

    /// Run one tracking cycle on a batch of OTHR measurements.
    pub fn step(&mut self, measurements: &[Measurement], dt: f64) {
        // 1. Predict each track with Vincenty direct.
        for track in self.tracks.iter_mut() {
            if track.lifecycle == TrackState::Deleted {
                continue;
            }
            let state = GreatCircleState::from_vector(&track.state);
            let predicted = self.motion_model.predict(&state, dt);
            let f_jac = self.motion_model.jacobian(&state, dt);
            let q = self.motion_model.process_noise(dt);
            track.state = predicted.to_vector();
            track.covariance = &f_jac * &track.covariance * f_jac.transpose() + q;
        }

        // Collect alive track indices.
        let alive: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.lifecycle != TrackState::Deleted)
            .map(|(i, _)| i)
            .collect();

        // 2. Extract OTHR measurement vectors.
        let othr_meas: Vec<[f64; 3]> = measurements
            .iter()
            .filter_map(|m| match m {
                Measurement::Othr {
                    ground_range_m,
                    azimuth_rad,
                    doppler_m_s,
                    ..
                } => Some([*ground_range_m, *azimuth_rad, *doppler_m_s]),
                _ => None,
            })
            .collect();

        // 3. Build cost matrix via Mahalanobis distance in measurement space.
        let mut cost_matrix = vec![vec![self.gate_threshold; othr_meas.len()]; alive.len()];
        let mut per_track_cache: Vec<(DVector<f64>, DMatrix<f64>, DMatrix<f64>)> =
            Vec::with_capacity(alive.len());
        for &ti in alive.iter() {
            let track = &self.tracks[ti];
            let state = GreatCircleState::from_vector(&track.state);
            let z_hat = predict_othr_observation(
                &state,
                self.registration.transmitter_lat_rad,
                self.registration.transmitter_lon_rad,
            );
            let z_hat_v = DVector::from_column_slice(&z_hat);
            let h_jac = othr_observation_jacobian(
                &state,
                self.registration.transmitter_lat_rad,
                self.registration.transmitter_lon_rad,
            );
            let s = &h_jac * &track.covariance * h_jac.transpose() + &self.measurement_noise;
            per_track_cache.push((z_hat_v, h_jac, s));
        }

        for (ai, &_ti) in alive.iter().enumerate() {
            let (z_hat_v, _, s) = &per_track_cache[ai];
            let zero = DVector::<f64>::zeros(3);
            for (dj, m) in othr_meas.iter().enumerate() {
                let mut diff = DVector::from_column_slice(m) - z_hat_v;
                // Azimuth residual wrap.
                diff[1] = wrap_angle(diff[1]);
                // Mahalanobis on the (wrapped) innovation: pass innovation as
                // the "observation" and zero as the "prediction".
                let d2 = mahalanobis_squared(&diff, &zero, s);
                if d2 < self.gate_threshold {
                    cost_matrix[ai][dj] = d2;
                }
            }
        }

        // 4. Hungarian assignment.
        let result = hungarian_assignment(&cost_matrix, self.gate_threshold);

        let mut associated_tracks = vec![false; alive.len()];
        let mut associated_dets = vec![false; othr_meas.len()];

        for &(ai, dj) in &result.matches {
            if ai >= alive.len() || dj >= othr_meas.len() {
                continue;
            }
            if cost_matrix[ai][dj] >= self.gate_threshold {
                continue;
            }
            associated_tracks[ai] = true;
            associated_dets[dj] = true;

            let ti = alive[ai];
            let (z_hat_v, h_jac, s) = &per_track_cache[ai];

            // EKF update with azimuth wraparound on the innovation.
            let mut innovation = DVector::from_column_slice(&othr_meas[dj]) - z_hat_v;
            innovation[1] = wrap_angle(innovation[1]);

            let s_inv = s
                .clone()
                .try_inverse()
                .expect("OTHR innovation covariance singular");
            let track = &mut self.tracks[ti];
            let k = &track.covariance * h_jac.transpose() * s_inv;
            let new_state = &track.state + &k * &innovation;
            let n = track.state.len();
            let i_kh = DMatrix::<f64>::identity(n, n) - &k * h_jac;
            let new_cov = &i_kh * &track.covariance * i_kh.transpose()
                + &k * &self.measurement_noise * k.transpose();

            track.state = new_state;
            // Wrap lat/lon/heading back into canonical ranges.
            track.state[1] = wrap_angle(track.state[1]);
            track.state[4] = wrap_angle(track.state[4]);
            track.covariance = new_cov;
            track.hits += 1;
            track.misses = 0;
        }

        // 5. Lifecycle: confirm / coast / delete.
        for (ai, &ti) in alive.iter().enumerate() {
            let was_associated = associated_tracks[ai];
            let track = &mut self.tracks[ti];
            if was_associated {
                let promote_tentative = track.lifecycle == TrackState::Tentative
                    && track.hits >= self.confirmation_hits;
                if promote_tentative || track.lifecycle == TrackState::Coasting {
                    track.lifecycle = TrackState::Confirmed;
                }
            } else {
                track.misses += 1;
                if track.lifecycle == TrackState::Confirmed {
                    track.lifecycle = TrackState::Coasting;
                }
                if track.misses >= self.deletion_misses {
                    track.lifecycle = TrackState::Deleted;
                }
            }
        }

        // 6. Birth new tracks from unassigned measurements.
        for (dj, associated) in associated_dets.iter().enumerate() {
            if !*associated {
                let raw = &othr_meas[dj];
                let m = Measurement::Othr {
                    ground_range_m: raw[0],
                    azimuth_rad: raw[1],
                    doppler_m_s: raw[2],
                    propagation_mode: thresh_core::measurement::PropagationMode::FLayer,
                    time: 0.0,
                    sensor_id: 0,
                };
                self.init_from_othr(&m, &self.registration.clone(), self.assumed_alt_m);
            }
        }

        // 7. Remove deleted tracks.
        self.tracks.retain(|t| t.lifecycle != TrackState::Deleted);
    }

    /// Initialize a new track from a single OTHR detection.
    ///
    /// The target geodetic position is obtained via Vincenty's direct formula
    /// from the transmitter. Ground speed is initialized to the absolute
    /// Doppler (a coarse but unbiased guess), heading is initialized to the
    /// outbound bearing from the transmitter, and altitude is set to the
    /// supplied assumed value.
    pub fn init_from_othr(
        &mut self,
        measurement: &Measurement,
        registration: &OthrSensorRegistration,
        assumed_alt_m: f64,
    ) {
        let (ground_range, azimuth, doppler) = match measurement {
            Measurement::Othr {
                ground_range_m,
                azimuth_rad,
                doppler_m_s,
                ..
            } => (*ground_range_m, *azimuth_rad, *doppler_m_s),
            _ => return,
        };

        let (lat, lon) = vincenty_direct(
            registration.transmitter_lat_rad,
            registration.transmitter_lon_rad,
            azimuth,
            ground_range.max(0.0),
        );

        // Initial heading: outbound bearing from the transmitter is a decent
        // first guess; the filter can slew to the true value over a few hits.
        let state = GreatCircleState {
            lat_rad: lat,
            lon_rad: wrap_angle(lon),
            alt_m: assumed_alt_m,
            ground_speed_m_s: doppler.abs(),
            heading_rad: wrap_angle(azimuth),
            climb_rate_m_s: 0.0,
        };

        // Large initial covariance: OTHR is coarse and we have no velocity
        // information from a single detection.
        let diag = DVector::from_column_slice(&[
            (1.0_f64.to_radians()).powi(2),  // lat: ~1°
            (1.0_f64.to_radians()).powi(2),  // lon: ~1°
            1_000_000.0,                     // alt: 1 km std
            (200.0_f64).powi(2),             // speed: 200 m/s std
            (30.0_f64.to_radians()).powi(2), // heading: 30°
            (5.0_f64).powi(2),               // climb: 5 m/s std
        ]);
        let covariance = DMatrix::from_diagonal(&diag);

        self.tracks.push(GreatCircleTrack {
            id: TrackId::new(),
            state: state.to_vector(),
            covariance,
            lifecycle: TrackState::Tentative,
            class: TargetClass::Aircraft,
            hits: 1,
            misses: 0,
        });
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_reg() -> OthrSensorRegistration {
        OthrSensorRegistration {
            transmitter_lat_rad: 0.0,
            transmitter_lon_rad: 0.0,
            transmitter_alt_m: 0.0,
            operating_freq_mhz: 15.0,
        }
    }

    #[test]
    fn state_vector_roundtrip() {
        let s = GreatCircleState {
            lat_rad: 0.5,
            lon_rad: -1.2,
            alt_m: 10_000.0,
            ground_speed_m_s: 250.0,
            heading_rad: 1.0,
            climb_rate_m_s: 2.0,
        };
        let v = s.to_vector();
        let s2 = GreatCircleState::from_vector(&v);
        assert_eq!(s.lat_rad, s2.lat_rad);
        assert_eq!(s.lon_rad, s2.lon_rad);
        assert_eq!(s.alt_m, s2.alt_m);
        assert_eq!(s.ground_speed_m_s, s2.ground_speed_m_s);
        assert_eq!(s.heading_rad, s2.heading_rad);
        assert_eq!(s.climb_rate_m_s, s2.climb_rate_m_s);
    }

    #[test]
    fn predict_advances_eastward_on_equator() {
        let model = GreatCircleMotionModel::default();
        let state = GreatCircleState {
            lat_rad: 0.0,
            lon_rad: 0.0,
            alt_m: 10_000.0,
            ground_speed_m_s: 250.0,
            heading_rad: std::f64::consts::FRAC_PI_2,
            climb_rate_m_s: 0.0,
        };
        let next = model.predict(&state, 60.0);
        // 15 km due east: longitude should increase, latitude near zero.
        assert!(next.lon_rad > 0.0);
        assert!(next.lat_rad.abs() < 1e-6);
    }

    #[test]
    fn jacobian_is_finite_and_near_identity_for_small_dt() {
        let model = GreatCircleMotionModel::default();
        let state = GreatCircleState {
            lat_rad: 0.5,
            lon_rad: -1.0,
            alt_m: 10_000.0,
            ground_speed_m_s: 250.0,
            heading_rad: 1.0,
            climb_rate_m_s: 0.0,
        };
        let j = model.jacobian(&state, 0.0);
        assert_eq!(j.nrows(), 6);
        assert_eq!(j.ncols(), 6);
        for i in 0..6 {
            for k in 0..6 {
                assert!(j[(i, k)].is_finite(), "jacobian entry {i},{k} not finite");
            }
            assert!(
                (j[(i, i)] - 1.0).abs() < 1e-3,
                "diagonal should be ~1 at dt=0, got {}",
                j[(i, i)]
            );
        }
    }

    #[test]
    fn process_noise_is_positive_definite() {
        let model = GreatCircleMotionModel::default();
        let q = model.process_noise(1.0);
        assert!(q.clone().cholesky().is_some());
    }

    #[test]
    fn predict_observation_matches_initialization() {
        let reg = test_reg();
        // 2000 km north of the transmitter.
        let (lat, lon) = vincenty_direct(0.0, 0.0, 0.0, 2_000_000.0);
        let state = GreatCircleState {
            lat_rad: lat,
            lon_rad: lon,
            alt_m: 10_000.0,
            ground_speed_m_s: 200.0,
            heading_rad: 0.0, // heading north = moving away from transmitter
            climb_rate_m_s: 0.0,
        };
        let z = predict_othr_observation(&state, reg.transmitter_lat_rad, reg.transmitter_lon_rad);
        assert!(
            (z[0] - 2_000_000.0).abs() < 0.1,
            "ground range mismatch: {}",
            z[0]
        );
        // Azimuth from transmitter to target is due north.
        assert!(z[1].abs() < 1e-6 || (z[1] - TAU).abs() < 1e-6);
        // Doppler: moving north, transmitter is south => receding => negative.
        assert!(z[2] < 0.0, "doppler should be negative, got {}", z[2]);
        assert!((z[2] + 200.0).abs() < 1e-6);
    }

    #[test]
    fn init_from_othr_creates_track_at_expected_position() {
        let reg = test_reg();
        let mut tracker = MultiObjectTrackerGreatCircle::new(
            reg.clone(),
            DMatrix::from_diagonal(&DVector::from_column_slice(&[
                (20_000.0_f64).powi(2),
                (0.017_f64).powi(2),
                (2.0_f64).powi(2),
            ])),
            50.0,
        );
        let m = Measurement::Othr {
            ground_range_m: 2_000_000.0,
            azimuth_rad: 0.0,
            doppler_m_s: 250.0,
            propagation_mode: thresh_core::measurement::PropagationMode::FLayer,
            time: 0.0,
            sensor_id: 0,
        };
        tracker.init_from_othr(&m, &reg, 10_000.0);
        assert_eq!(tracker.tracks.len(), 1);
        let s = tracker.tracks[0].as_state();
        let (lat_true, _lon_true) = vincenty_direct(0.0, 0.0, 0.0, 2_000_000.0);
        assert!((s.lat_rad - lat_true).abs() < 1e-9);
        assert_eq!(s.alt_m, 10_000.0);
        assert_eq!(s.ground_speed_m_s, 250.0);
    }
}
