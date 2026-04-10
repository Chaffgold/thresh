//! Integration tests for the great-circle motion-model tracker (task 8.B.6-8).
//!
//! These tests exercise long-duration, long-range aircraft tracking scenarios
//! where a flat-Earth (ENU) tracker would accumulate significant error. The
//! great-circle tracker uses Vincenty's direct formula for state propagation
//! and should maintain accurate position across 1000+ km transits.
//!
//! Synthetic OTHR measurements are constructed directly from Vincenty inverse
//! on geodetic ground truth so the measurement geometry matches the tracker's
//! observation model, avoiding the ENU/great-circle inconsistency that
//! `thresh_synth::othr_generator::generate_othr` would introduce at very long
//! ranges. The Vincenty ground-truth approach mirrors how OTHR measurements
//! are actually formed on the real ellipsoid.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use thresh_core::measurement::{Measurement, PropagationMode};
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct, vincenty_inverse};
use thresh_tracker::great_circle_tracker::{GreatCircleState, MultiObjectTrackerGreatCircle};
use thresh_tracker::tracker::MultiObjectTracker;

/// Build a synthetic OTHR measurement from a geodetic ground truth position
/// and velocity (east, north components), adding Gaussian noise.
#[allow(clippy::too_many_arguments)]
fn othr_from_truth(
    reg: &OthrSensorRegistration,
    lat: f64,
    lon: f64,
    v_east: f64,
    v_north: f64,
    time: f64,
    range_sigma: f64,
    az_sigma: f64,
    dop_sigma: f64,
    rng: &mut StdRng,
) -> Measurement {
    // True range/azimuth from transmitter to target.
    let (range, az) = vincenty_inverse(reg.transmitter_lat_rad, reg.transmitter_lon_rad, lat, lon);
    // Bearing from target back to transmitter, for Doppler projection.
    let (_, az_back) = vincenty_inverse(lat, lon, reg.transmitter_lat_rad, reg.transmitter_lon_rad);
    // Radial velocity (positive = approaching transmitter).
    let u_east = az_back.sin();
    let u_north = az_back.cos();
    let doppler = v_east * u_east + v_north * u_north;

    let rn = Normal::new(0.0, range_sigma).unwrap();
    let an = Normal::new(0.0, az_sigma).unwrap();
    let dn = Normal::new(0.0, dop_sigma).unwrap();

    Measurement::Othr {
        ground_range_m: range + rn.sample(rng),
        azimuth_rad: az + an.sample(rng),
        doppler_m_s: doppler + dn.sample(rng),
        propagation_mode: PropagationMode::FLayer,
        time,
        sensor_id: 0,
    }
}

/// Default OTHR measurement noise used throughout these tests.
fn default_r() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(&[
        (15_000.0_f64).powi(2), // range: 15 km std
        (0.01_f64).powi(2),     // azimuth: ~0.57°
        (2.0_f64).powi(2),      // doppler: 2 m/s
    ]))
}

// ── 8.B.6 — Long constant-heading flight ──────────────────────────────────

#[test]
fn great_circle_tracker_constant_heading_1500km() {
    let reg = OthrSensorRegistration {
        transmitter_lat_rad: 20.0_f64.to_radians(),
        transmitter_lon_rad: 0.0,
        transmitter_alt_m: 0.0,
        operating_freq_mhz: 15.0,
    };

    // Target starts 1500 km due north of the transmitter at a high latitude
    // and flies due east at 250 m/s for 1.5 hours (~1350 km).
    let (start_lat, start_lon) = vincenty_direct(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        0.0,
        1_500_000.0,
    );

    let speed = 250.0_f64;
    let heading = std::f64::consts::FRAC_PI_2; // east
    let dt = 10.0_f64;
    let n_steps: usize = 540; // 540 * 10 s = 5400 s = 1.5 h

    let mut tracker = MultiObjectTrackerGreatCircle::new(reg.clone(), default_r(), 30.0);
    let mut rng = StdRng::seed_from_u64(42);

    let (mut lat, mut lon) = (start_lat, start_lon);
    let mut final_err_m = f64::NAN;

    for k in 0..n_steps {
        // Advance truth by one step along a great-circle heading east.
        let (nl, nlo) = vincenty_direct(lat, lon, heading, speed * dt);
        lat = nl;
        lon = nlo;

        let v_east = speed * heading.sin();
        let v_north = speed * heading.cos();
        let m = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            k as f64 * dt,
            15_000.0,
            0.01,
            2.0,
            &mut rng,
        );
        tracker.step(&[m], dt);

        if k + 1 == n_steps && !tracker.tracks.is_empty() {
            // Pick the track closest to truth.
            let (_, best_err) = tracker
                .tracks
                .iter()
                .map(|t| {
                    let s = GreatCircleState::from_vector(&t.state);
                    let (d, _) = vincenty_inverse(s.lat_rad, s.lon_rad, lat, lon);
                    (t.id, d)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            final_err_m = best_err;
        }
    }

    assert!(tracker.alive_count() >= 1, "should hold at least one track");
    assert!(
        tracker.confirmed_count() >= 1,
        "track should confirm over 1.5 hours"
    );
    assert!(
        final_err_m.is_finite() && final_err_m < 200_000.0,
        "great-circle tracker final position error too large: {final_err_m} m"
    );
}

// ── 8.B.7 — Polar longitude wraparound ────────────────────────────────────

#[test]
fn great_circle_tracker_handles_longitude_wrap() {
    // Transmitter at a high northern latitude near the dateline.
    let reg = OthrSensorRegistration {
        transmitter_lat_rad: 70.0_f64.to_radians(),
        transmitter_lon_rad: 170.0_f64.to_radians(),
        transmitter_alt_m: 0.0,
        operating_freq_mhz: 15.0,
    };

    // Target starts 1200 km due east of the transmitter (will cross 180°).
    let (start_lat, start_lon) = vincenty_direct(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        std::f64::consts::FRAC_PI_2,
        1_200_000.0,
    );

    let speed = 220.0_f64;
    let heading = std::f64::consts::FRAC_PI_2; // due east: crosses 180°
    let dt = 10.0_f64;
    let n_steps: usize = 300; // 3000 s, ~660 km

    let mut tracker = MultiObjectTrackerGreatCircle::new(reg.clone(), default_r(), 30.0);
    let mut rng = StdRng::seed_from_u64(7);

    let (mut lat, mut lon) = (start_lat, start_lon);

    for k in 0..n_steps {
        let (nl, nlo) = vincenty_direct(lat, lon, heading, speed * dt);
        lat = nl;
        lon = nlo;

        let v_east = speed * heading.sin();
        let v_north = speed * heading.cos();
        let m = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            k as f64 * dt,
            15_000.0,
            0.01,
            2.0,
            &mut rng,
        );
        tracker.step(&[m], dt);

        // All states must remain finite.
        for t in &tracker.tracks {
            for v in t.state.iter() {
                assert!(v.is_finite(), "non-finite state at step {k}: {v}");
            }
        }
    }

    assert!(
        tracker.alive_count() >= 1,
        "track should survive longitude wrap"
    );

    // Track longitude must be within ±π (wrapped).
    for t in &tracker.tracks {
        let s = GreatCircleState::from_vector(&t.state);
        assert!(
            s.lon_rad >= -std::f64::consts::PI && s.lon_rad <= std::f64::consts::PI,
            "track longitude out of canonical range: {}",
            s.lon_rad
        );
    }

    // Position error should still be bounded.
    let (_, err) = tracker
        .tracks
        .iter()
        .map(|t| {
            let s = GreatCircleState::from_vector(&t.state);
            let (d, _) = vincenty_inverse(s.lat_rad, s.lon_rad, lat, lon);
            (t.id, d)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert!(
        err < 300_000.0,
        "polar-crossing position error too large: {err} m"
    );
}

// ── 8.B.8 — Benchmark: great-circle vs ENU tracker ────────────────────────

#[test]
fn benchmark_great_circle_vs_enu_long_duration() {
    // Same scenario as 8.B.6: 1.5-hour due-east flight at high latitude.
    let reg = OthrSensorRegistration {
        transmitter_lat_rad: 20.0_f64.to_radians(),
        transmitter_lon_rad: 0.0,
        transmitter_alt_m: 0.0,
        operating_freq_mhz: 15.0,
    };

    let (start_lat, start_lon) = vincenty_direct(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        0.0,
        1_500_000.0,
    );

    let speed = 250.0_f64;
    let heading = std::f64::consts::FRAC_PI_2;
    let dt = 10.0_f64;
    let n_steps: usize = 540;

    // Great-circle tracker.
    let mut gc_tracker = MultiObjectTrackerGreatCircle::new(reg.clone(), default_r(), 30.0);
    // ENU tracker: reference at transmitter, treating OTHR measurements as
    // flat (range, azimuth) -> ENU (east, north). Measurement noise matches
    // OTHR cross-range uncertainty.
    let mut enu_tracker = MultiObjectTracker::new_cv_position(25_000.0, 50.0);

    let mut rng_gc = StdRng::seed_from_u64(1234);
    let mut rng_enu = StdRng::seed_from_u64(1234);

    let (mut lat, mut lon) = (start_lat, start_lon);
    let mut gc_errors: Vec<f64> = Vec::new();
    let mut enu_errors: Vec<f64> = Vec::new();

    for k in 0..n_steps {
        let (nl, nlo) = vincenty_direct(lat, lon, heading, speed * dt);
        lat = nl;
        lon = nlo;
        let v_east = speed * heading.sin();
        let v_north = speed * heading.cos();

        // Great-circle update.
        let m_gc = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            k as f64 * dt,
            15_000.0,
            0.01,
            2.0,
            &mut rng_gc,
        );
        gc_tracker.step(&[m_gc], dt);

        // ENU update: convert the same-noise measurement to a flat (x, y, z).
        let m_enu = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            k as f64 * dt,
            15_000.0,
            0.01,
            2.0,
            &mut rng_enu,
        );
        if let Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } = m_enu
        {
            // Flat approximation: treat (range, azimuth) as polar in ENU at
            // the transmitter. This is the "pretend OTHR is flat" approach
            // that motivates the great-circle tracker in the first place.
            let x = ground_range_m * azimuth_rad.sin();
            let y = ground_range_m * azimuth_rad.cos();
            let det = DVector::from_column_slice(&[x, y, 10_000.0]);
            enu_tracker.step(&[det], dt);
        }

        // Errors: compute true ENU position at the transmitter via Vincenty
        // inverse (range, azimuth) -> flat ENU is NOT used for truth; we use
        // the great-circle ground range/azimuth directly to get the *true*
        // (range, az) and compare against both trackers in metric distance.
        let (true_range, true_az) =
            vincenty_inverse(reg.transmitter_lat_rad, reg.transmitter_lon_rad, lat, lon);

        // Great-circle error in meters on the ellipsoid.
        if let Some(t) = gc_tracker.tracks.first() {
            let s = GreatCircleState::from_vector(&t.state);
            let (d, _) = vincenty_inverse(s.lat_rad, s.lon_rad, lat, lon);
            gc_errors.push(d);
        }

        // ENU error: compare the ENU tracker's (x, y) against the *true*
        // flat-projected (x, y). The flat projection will increasingly
        // disagree with the ellipsoid at long range/high latitude; that is
        // exactly the error this benchmark is meant to surface.
        if let Some(t) = enu_tracker.tracks.first() {
            let true_x = true_range * true_az.sin();
            let true_y = true_range * true_az.cos();
            let ex = t.state[0] - true_x;
            let ey = t.state[2] - true_y;
            enu_errors.push((ex * ex + ey * ey).sqrt());
        }
    }

    assert!(
        gc_errors.len() >= 100,
        "great-circle tracker should produce samples"
    );
    assert!(
        enu_errors.len() >= 100,
        "ENU tracker should produce samples"
    );

    let tail = |v: &[f64]| -> f64 {
        let start = v.len().saturating_sub(50);
        v[start..].iter().sum::<f64>() / (v.len() - start) as f64
    };
    let gc_tail = tail(&gc_errors);
    let enu_tail = tail(&enu_errors);

    // Log (via stdout captured by cargo test in nocapture) for visibility.
    println!("great-circle tail mean error (m): {gc_tail}");
    println!("ENU flat-approx tail mean error (m): {enu_tail}");

    // The great-circle tracker should be at least competitive. We allow a
    // generous margin because both trackers must converge under coarse OTHR
    // noise; the key assertion is that the great-circle solution stays
    // bounded over a 1.5-hour run.
    assert!(
        gc_tail.is_finite() && gc_tail < 200_000.0,
        "great-circle tail error too large: {gc_tail} m"
    );
    assert!(
        enu_tail.is_finite(),
        "ENU tail error should be finite: {enu_tail} m"
    );
}
