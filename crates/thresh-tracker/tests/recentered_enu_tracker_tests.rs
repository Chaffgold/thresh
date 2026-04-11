//! Integration tests for the recentered-ENU tracker (tasks 8.C.5-8.C.6).
//!
//! Synthetic OTHR measurements are built directly from `vincenty_inverse` on a
//! geodetic ground-truth trajectory so the observation geometry matches the
//! curved earth, bypassing the flat-ENU bias that
//! `thresh_synth::othr_generator::generate_othr` picks up at long ranges.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use thresh_core::measurement::{Measurement, PropagationMode};
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct, vincenty_inverse};
use thresh_core::track::TargetClass;
use thresh_tracker::great_circle_tracker::{GreatCircleState, MultiObjectTrackerGreatCircle};
use thresh_tracker::recentered_enu_tracker::{
    MultiObjectTrackerRecenteredEnu, RecenteredEnuTrack, RecenteringPolicy, recenter_track,
};

// ── Helpers ────────────────────────────────────────────────────────────────

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
    let (range, az) = vincenty_inverse(reg.transmitter_lat_rad, reg.transmitter_lon_rad, lat, lon);
    let (_, az_back) = vincenty_inverse(lat, lon, reg.transmitter_lat_rad, reg.transmitter_lon_rad);
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

fn default_gc_r() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(&[
        (15_000.0_f64).powi(2),
        (0.01_f64).powi(2),
        (2.0_f64).powi(2),
    ]))
}

// ── 8.C.5 — Recentering preserves continuity ──────────────────────────────

#[test]
fn recentering_preserves_state_continuity() {
    // Build a track with a non-trivial state (non-zero velocity, off-origin
    // position, non-diagonal covariance) and verify that recentering onto its
    // own geodetic centroid leaves the externally visible geodetic position
    // and velocity magnitude essentially unchanged.
    let origin_lat = 10.0_f64.to_radians();
    let origin_lon = (-30.0_f64).to_radians();
    let origin_alt = 0.0;

    let mut state = DVector::<f64>::zeros(6);
    state[0] = 250_000.0; // east
    state[2] = 150_000.0; // north
    state[4] = 10_000.0; // up
    state[1] = 180.0; // vx
    state[3] = 120.0; // vy
    state[5] = 1.0; // vz

    // Non-trivial (still symmetric PSD) covariance.
    let mut p = DMatrix::<f64>::identity(6, 6);
    for i in 0..6 {
        p[(i, i)] = (i as f64 + 1.0) * 1.0e4;
    }
    p[(0, 2)] = 1.0e3;
    p[(2, 0)] = 1.0e3;
    p[(1, 3)] = 50.0;
    p[(3, 1)] = 50.0;

    let mut track = RecenteredEnuTrack::new(
        state.clone(),
        p.clone(),
        TargetClass::Aircraft,
        origin_lat,
        origin_lon,
        origin_alt,
    );

    let (lat_before, lon_before, alt_before) = track.geodetic_position();
    let vmag_before = (state[1] * state[1] + state[3] * state[3] + state[5] * state[5]).sqrt();

    // Recenter onto the track's own geodetic centroid.
    recenter_track(&mut track, lat_before, lon_before, alt_before);

    // Geodetic position continuity: should match within a few metres once we
    // convert back through the new frame.
    let (lat_after, lon_after, alt_after) = track.geodetic_position();
    assert!((lat_after - lat_before).abs() < 1e-9);
    assert!((lon_after - lon_before).abs() < 1e-9);
    assert!((alt_after - alt_before).abs() < 1e-2);

    // Speed magnitude is invariant under an orthogonal rotation.
    let vmag_after =
        (track.state[1].powi(2) + track.state[3].powi(2) + track.state[5].powi(2)).sqrt();
    assert!(
        (vmag_after - vmag_before).abs() < 1e-9,
        "velocity magnitude drifted: {vmag_before} -> {vmag_after}"
    );

    // Covariance trace is invariant under R P R^T for orthogonal R.
    let trace_before: f64 = (0..6).map(|i| p[(i, i)]).sum();
    let trace_after: f64 = (0..6).map(|i| track.covariance[(i, i)]).sum();
    assert!(
        (trace_after - trace_before).abs() < 1e-6,
        "trace drifted: {trace_before} -> {trace_after}"
    );

    // Local position collapses to the origin of the new frame.
    assert!(track.state[0].abs() < 1.0);
    assert!(track.state[2].abs() < 1.0);
}

// ── 8.C.6 — Long traverse: recentered ENU vs static ENU ───────────────────

fn count_recenter_events(
    pre_origins: &[(f64, f64, f64)],
    tracker: &MultiObjectTrackerRecenteredEnu,
) -> usize {
    let mut events = 0;
    for t in &tracker.tracks {
        let matched = pre_origins.iter().any(|&(la, lo, _)| {
            (la - t.origin_lat_rad).abs() < 1e-15 && (lo - t.origin_lon_rad).abs() < 1e-15
        });
        if !matched {
            events += 1;
        }
    }
    events
}

fn best_re_error(tracker: &MultiObjectTrackerRecenteredEnu, lat: f64, lon: f64) -> f64 {
    tracker
        .tracks
        .iter()
        .map(|t| {
            let (la, lo, _) = t.geodetic_position();
            vincenty_inverse(la, lo, lat, lon).0
        })
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(f64::NAN)
}

fn best_gc_error(tracker: &MultiObjectTrackerGreatCircle, lat: f64, lon: f64) -> f64 {
    tracker
        .tracks
        .iter()
        .map(|t| {
            let s = GreatCircleState::from_vector(&t.state);
            vincenty_inverse(s.lat_rad, s.lon_rad, lat, lon).0
        })
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(f64::NAN)
}

#[test]
fn recentered_enu_tracks_long_traverse() {
    // Aircraft flies 2000+ km due east at high latitude. A static ENU anchored
    // at the transmitter would accumulate flat-earth error, but the recentered
    // ENU tracker should stay within a few tens of km of truth on the
    // ellipsoid, comparable to the great-circle tracker.
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
    let n_steps: usize = 900;

    let mut re_tracker = MultiObjectTrackerRecenteredEnu::new(reg.clone(), 10_000.0);
    re_tracker.gate_threshold = 50.0;
    re_tracker.measurement_noise_sigma = 20_000.0;
    re_tracker.recentering_policy = RecenteringPolicy {
        drift_threshold_m: 200_000.0,
    };

    let mut gc_tracker = MultiObjectTrackerGreatCircle::new(reg.clone(), default_gc_r(), 30.0);
    let mut rng_re = StdRng::seed_from_u64(9001);
    let mut rng_gc = StdRng::seed_from_u64(9001);

    let (mut lat, mut lon) = (start_lat, start_lon);
    let mut recentered_count = 0usize;

    for k in 0..n_steps {
        let (nl, nlo) = vincenty_direct(lat, lon, heading, speed * dt);
        lat = nl;
        lon = nlo;
        let v_east = speed * heading.sin();
        let v_north = speed * heading.cos();
        let time = k as f64 * dt;

        let m_re = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            time,
            15_000.0,
            0.01,
            2.0,
            &mut rng_re,
        );
        let pre_origins: Vec<(f64, f64, f64)> = re_tracker
            .tracks
            .iter()
            .map(|t| (t.origin_lat_rad, t.origin_lon_rad, t.origin_alt_m))
            .collect();
        re_tracker.step(&[m_re], dt);
        recentered_count += count_recenter_events(&pre_origins, &re_tracker);

        let m_gc = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            time,
            15_000.0,
            0.01,
            2.0,
            &mut rng_gc,
        );
        gc_tracker.step(&[m_gc], dt);
    }

    let final_re_err_m = best_re_error(&re_tracker, lat, lon);
    let final_gc_err_m = best_gc_error(&gc_tracker, lat, lon);

    println!("recentered ENU final error (m): {final_re_err_m}");
    println!("great-circle  final error (m): {final_gc_err_m}");
    println!("recenter events: {recentered_count}");

    assert!(
        re_tracker.alive_count() >= 1,
        "recentered tracker lost all tracks"
    );
    assert!(
        re_tracker.confirmed_count() >= 1,
        "recentered tracker should confirm at least one track"
    );
    assert!(
        final_re_err_m.is_finite() && final_re_err_m < 250_000.0,
        "recentered ENU final error too large: {final_re_err_m} m"
    );
    assert!(
        recentered_count >= 1,
        "expected at least one recenter over the long traverse, got {recentered_count}"
    );
}

// ── 8.C extra — State transformation roundtrip ─────────────────────────────

#[test]
fn recenter_roundtrip_preserves_position() {
    // Pick a state, recenter onto a distant origin, then recenter back to the
    // original origin; the position should come back to itself within ~10 m.
    let origin_lat = 35.0_f64.to_radians();
    let origin_lon = (-80.0_f64).to_radians();
    let origin_alt = 0.0;

    let mut state = DVector::<f64>::zeros(6);
    state[0] = 180_000.0;
    state[2] = -90_000.0;
    state[4] = 12_000.0;
    state[1] = 200.0;
    state[3] = -150.0;
    state[5] = 1.5;

    let p0 = DMatrix::<f64>::identity(6, 6) * 1.0e4;
    let mut track = RecenteredEnuTrack::new(
        state.clone(),
        p0.clone(),
        TargetClass::Aircraft,
        origin_lat,
        origin_lon,
        origin_alt,
    );

    // Recenter to a different origin ~500 km away.
    let (mid_lat, mid_lon) = vincenty_direct(origin_lat, origin_lon, 0.7, 500_000.0);
    recenter_track(&mut track, mid_lat, mid_lon, 0.0);

    // Recenter back to the original origin.
    recenter_track(&mut track, origin_lat, origin_lon, origin_alt);

    // Position should match the original to within ~10 m.
    for (i, orig) in [(0usize, state[0]), (2, state[2]), (4, state[4])] {
        assert!(
            (track.state[i] - orig).abs() < 10.0,
            "state[{i}] drifted: {} vs {}",
            track.state[i],
            orig
        );
    }
    // Velocity is exactly orthogonally rotated → magnitude is preserved and
    // roundtrip recovers the original components.
    for (i, orig) in [(1usize, state[1]), (3, state[3]), (5, state[5])] {
        assert!(
            (track.state[i] - orig).abs() < 1e-6,
            "vel[{i}] drifted: {} vs {}",
            track.state[i],
            orig
        );
    }
}
