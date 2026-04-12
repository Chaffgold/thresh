//! End-to-end comparison of the tracker variants on a single OTHR scenario
//! (task 8.E.4).
//!
//! The scenario is a civil-aircraft-class target flying due east at 250 m/s
//! for 30 minutes (~450 km traverse) starting 2000 km north of a single
//! OTHR transmitter. Synthetic OTHR measurements are generated via Vincenty
//! inverse on the ellipsoid so the measurement geometry is consistent with
//! the curved Earth regardless of which tracker consumes them.
//!
//! The test runs the same measurement stream through every available
//! tracker variant and prints a comparison table of final position error
//! against ground truth. The hard assertion is only that each variant
//! produces a track and lands within a loose error bound (100 km) so none
//! of the variants silently blow up.
//!
//! The ECEF tracker (`MultiObjectTrackerEcef`) is listed in the
//! [`thresh_tracker::tracker_variant::TrackerVariant`] enum for
//! forward compatibility, but its implementation lives on a future task
//! (8.A) and is not yet part of the crate. The comparison table marks it
//! as "not yet implemented" to keep this test honest without holding up
//! the selection/documentation work in task 8.E.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use thresh_core::measurement::{Measurement, PropagationMode};
use thresh_core::othr::{OthrSensorRegistration, vincenty_direct, vincenty_inverse};
use thresh_tracker::great_circle_tracker::{GreatCircleState, MultiObjectTrackerGreatCircle};
use thresh_tracker::stereographic_tracker::{
    MultiObjectTrackerStereographic, othr_to_stereographic, stereographic_inverse,
};
use thresh_tracker::tracker::MultiObjectTracker;

/// Build a synthetic OTHR measurement from a geodetic ground-truth position
/// and east/north velocity components, adding Gaussian noise on range,
/// azimuth and Doppler.
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

/// OTHR measurement noise used consistently across all variants.
fn default_r() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(&[
        (15_000.0_f64).powi(2), // ground range: 15 km std
        (0.01_f64).powi(2),     // azimuth: ~0.57°
        (2.0_f64).powi(2),      // doppler: 2 m/s
    ]))
}

/// Great-circle distance (meters) between two geodetic points via Vincenty
/// inverse; a thin wrapper so the main test body stays readable.
fn ellipsoidal_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (d, _) = vincenty_inverse(lat1, lon1, lat2, lon2);
    d
}

#[test]
fn tracker_variant_comparison_on_othr_aircraft_scenario() {
    // ── Scenario ────────────────────────────────────────────────────────
    // OTHR transmitter at 20°N / 0°E, target begins 2000 km due north
    // and flies east at 250 m/s for 30 minutes (~450 km traverse).
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
        2_000_000.0,
    );

    let speed = 250.0_f64;
    let heading = std::f64::consts::FRAC_PI_2; // due east
    let dt = 10.0_f64;
    let n_steps: usize = 180; // 180 * 10 s = 1800 s = 30 min
    let assumed_alt_m = 10_000.0_f64;

    // ── Trackers ────────────────────────────────────────────────────────
    let mut enu_tracker = MultiObjectTracker::new_cv_position(25_000.0, 50.0);
    let mut gc_tracker = MultiObjectTrackerGreatCircle::new(reg.clone(), default_r(), 30.0);
    let mut stereo_tracker = MultiObjectTrackerStereographic::new(
        reg.transmitter_lat_rad,
        reg.transmitter_lon_rad,
        5.0,
        50.0,
    );

    // Independent RNGs (same seed) so each variant sees the same noise
    // realisation — this makes the cross-variant error comparison fair.
    let mut rng_enu = StdRng::seed_from_u64(20260409);
    let mut rng_gc = StdRng::seed_from_u64(20260409);
    let mut rng_stereo = StdRng::seed_from_u64(20260409);

    let (mut lat, mut lon) = (start_lat, start_lon);
    let (mut last_lat, mut last_lon) = (lat, lon);

    for k in 0..n_steps {
        // Advance truth by one step along a great-circle heading east.
        let (nl, nlo) = vincenty_direct(lat, lon, heading, speed * dt);
        lat = nl;
        lon = nlo;
        last_lat = lat;
        last_lon = lon;

        let v_east = speed * heading.sin();
        let v_north = speed * heading.cos();
        let time = k as f64 * dt;

        // ENU: treat (range, az) as polar in a flat tangent plane at the
        // transmitter. This is the "pretend OTHR is flat" approach that
        // the other variants exist to avoid.
        let m_enu = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            time,
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
            let x = ground_range_m * azimuth_rad.sin();
            let y = ground_range_m * azimuth_rad.cos();
            let det = DVector::from_column_slice(&[x, y, assumed_alt_m]);
            enu_tracker.step(&[det], dt);
        }

        // Great-circle: consumes the OTHR measurement directly.
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

        // Stereographic: convert OTHR detection → (x, y, alt) in the
        // conformal plane centred at the transmitter, then feed the linear
        // Cartesian tracker.
        let m_stereo = othr_from_truth(
            &reg,
            lat,
            lon,
            v_east,
            v_north,
            time,
            15_000.0,
            0.01,
            2.0,
            &mut rng_stereo,
        );
        let det = othr_to_stereographic(
            &m_stereo,
            &reg,
            assumed_alt_m,
            reg.transmitter_lat_rad,
            reg.transmitter_lon_rad,
        )
        .expect("stereographic projection should accept an OTHR measurement");
        stereo_tracker.step(&[det], dt);
    }

    // ── Final error per variant ─────────────────────────────────────────

    // ENU: convert tracker (x, y) back to (lat, lon) by inverting the flat
    // polar approximation (the same approximation used to feed the tracker)
    // and then measuring ellipsoidal distance to ground truth.
    let enu_err_m = enu_tracker
        .tracks
        .iter()
        .map(|t| {
            let x = t.state[0];
            let y = t.state[2];
            let range = (x * x + y * y).sqrt();
            let az = x.atan2(y);
            let (tlat, tlon) =
                vincenty_direct(reg.transmitter_lat_rad, reg.transmitter_lon_rad, az, range);
            ellipsoidal_distance(tlat, tlon, last_lat, last_lon)
        })
        .fold(f64::INFINITY, f64::min);

    // Great-circle: tracker state is already geodetic.
    let gc_err_m = gc_tracker
        .tracks
        .iter()
        .map(|t| {
            let s = GreatCircleState::from_vector(&t.state);
            ellipsoidal_distance(s.lat_rad, s.lon_rad, last_lat, last_lon)
        })
        .fold(f64::INFINITY, f64::min);

    // Stereographic: invert the conformal projection to recover (lat, lon).
    let stereo_err_m = stereo_tracker
        .tracks
        .iter()
        .map(|t| {
            let (tlat, tlon) = stereographic_inverse(
                t.state[0],
                t.state[2],
                reg.transmitter_lat_rad,
                reg.transmitter_lon_rad,
            );
            ellipsoidal_distance(tlat, tlon, last_lat, last_lon)
        })
        .fold(f64::INFINITY, f64::min);

    // ── Comparison table ────────────────────────────────────────────────
    eprintln!();
    eprintln!("Tracker variant comparison on 30-min OTHR aircraft scenario");
    eprintln!("  - transmitter: 20°N, 0°E");
    eprintln!("  - target: 2000 km due north, flying east at 250 m/s");
    eprintln!("  - duration: 30 min, dt = 10 s, {n_steps} steps");
    eprintln!();
    eprintln!(
        "  {:<16} {:>18}  {:>12}",
        "variant", "final err (km)", "status"
    );
    eprintln!(
        "  {:<16} {:>18}  {:>12}",
        "-------", "--------------", "------"
    );
    eprintln!(
        "  {:<16} {:>18.2}  {:>12}",
        "ENU (flat)",
        enu_err_m / 1000.0,
        "ok"
    );
    eprintln!("  {:<16} {:>18}  {:>12}", "ECEF", "n/a", "pending 8.A");
    eprintln!(
        "  {:<16} {:>18.2}  {:>12}",
        "Great-Circle",
        gc_err_m / 1000.0,
        "ok"
    );
    eprintln!(
        "  {:<16} {:>18.2}  {:>12}",
        "Stereographic",
        stereo_err_m / 1000.0,
        "ok"
    );
    eprintln!();

    // ── Sanity assertions (loose 100 km bound) ──────────────────────────
    assert!(enu_tracker.alive_count() >= 1, "ENU should hold a track");
    assert!(
        gc_tracker.alive_count() >= 1,
        "great-circle should hold a track"
    );
    assert!(
        stereo_tracker.alive_count() >= 1,
        "stereographic should hold a track"
    );

    let bound = 100_000.0_f64;
    assert!(
        enu_err_m.is_finite() && enu_err_m < bound,
        "ENU final error should be below {bound} m, got {enu_err_m} m"
    );
    assert!(
        gc_err_m.is_finite() && gc_err_m < bound,
        "great-circle final error should be below {bound} m, got {gc_err_m} m"
    );
    assert!(
        stereo_err_m.is_finite() && stereo_err_m < bound,
        "stereographic final error should be below {bound} m, got {stereo_err_m} m"
    );
}
