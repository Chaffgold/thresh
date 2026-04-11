#![cfg(feature = "jsbsim")]
//! JSBSim integration tests.
//!
//! These tests require the Python `jsbsim` package and the JSBSim aircraft
//! data files to be installed; they are therefore marked `#[ignore]` so CI
//! can build them but skip execution by default.

use thresh_synth::jsbsim::*;

#[test]
#[ignore]
fn f16_initialization() {
    let ic = InitialConditions {
        lat_deg: 40.0,
        lon_deg: -75.0,
        altitude_ft: 10_000.0,
        speed_kts: 400.0,
        heading_deg: 90.0,
    };
    let _bridge = JsbSimBridge::new(AircraftModel::F16, &ic).expect("jsbsim init");
}

#[test]
#[ignore]
fn f16_level_turn_bank_angle_4g() {
    // Task 4.8 — launch the F-16, fly a 4 g level turn and verify bank angle.
    // For a coordinated level turn: phi = acos(1/n). For n=4: phi ≈ 75.52°.
    let ic = InitialConditions {
        lat_deg: 40.0,
        lon_deg: -75.0,
        altitude_ft: 15_000.0,
        speed_kts: 450.0,
        heading_deg: 0.0,
    };
    let bridge = JsbSimBridge::new(AircraftModel::F16, &ic).expect("jsbsim init");
    let wps = Maneuvers::level_turn_g(0.0, 30.0, 4.0, 0.0);
    bridge.set_autopilot_waypoints(wps).expect("ap schedule");

    let states = bridge.run(30.0, 0.01, 1.0).expect("run");
    assert!(!states.is_empty());

    // Steady state: sample the final third of the run.
    let tail = &states[(states.len() * 2 / 3)..];
    let expected_bank = (1.0_f64 / 4.0).acos();
    let avg_bank = tail.iter().map(|s| s.euler_rad[0].abs()).sum::<f64>() / tail.len() as f64;
    // 15° tolerance — the JSBSim autopilot will not hit the theoretical angle
    // exactly but should be within a reasonable band.
    assert!((avg_bank - expected_bank).abs() < 15.0_f64.to_radians());
}

#[test]
#[ignore]
fn b737_climb_performance() {
    // Task 4.9 — 737 climbs at a published rate (~2 500 fpm near sea level,
    // per 737-800 flight manual figures). Verify within 10 %.
    let ic = InitialConditions {
        lat_deg: 40.0,
        lon_deg: -75.0,
        altitude_ft: 2_000.0,
        speed_kts: 280.0,
        heading_deg: 0.0,
    };
    let bridge = JsbSimBridge::new(AircraftModel::B737, &ic).expect("jsbsim init");
    let wps = Maneuvers::climb_descent(0.0, 60.0, 2_500.0, 2_000.0);
    bridge.set_autopilot_waypoints(wps).expect("ap schedule");

    let states = bridge.run(60.0, 0.05, 5.0).expect("run");
    assert!(states.len() >= 2);
    let first_alt = states.first().unwrap().altitude_ft;
    let last_alt = states.last().unwrap().altitude_ft;
    let duration = states.last().unwrap().time_s - states.first().unwrap().time_s;
    let observed_fpm = (last_alt - first_alt) / (duration / 60.0);
    let expected_fpm = 2_500.0;
    let err = (observed_fpm - expected_fpm).abs() / expected_fpm;
    assert!(
        err < 0.10,
        "observed {observed_fpm} fpm vs {expected_fpm} fpm"
    );
}
