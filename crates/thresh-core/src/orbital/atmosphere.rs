//! Piecewise-exponential atmosphere model and ballistic-coefficient drag.
//!
//! The density table and lookup were moved verbatim from
//! `thresh-synth/src/orbital.rs` (design Decision 1 of the
//! `orbital-ballistic-filter-models` change) so that filter prediction and
//! synthetic truth generation share one atmosphere. The model is
//! Earth-specific (Vallado Table 8-4 lineage).

use nalgebra::Vector3;

use super::gravity::GravityModel;
use crate::eci::EARTH_ROTATION_RATE;

/// (base altitude km, nominal density kg/m³, scale height km)
pub const ATMOSPHERE_TABLE: &[(f64, f64, f64)] = &[
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Compute atmospheric density (kg/m³) at a given geometric altitude (m)
/// using a piecewise-exponential model.
pub fn atmosphere_density(alt_m: f64) -> f64 {
    let alt_km = alt_m / 1000.0;
    if alt_km < 0.0 {
        return ATMOSPHERE_TABLE[0].1;
    }
    if alt_km > 1000.0 {
        return 0.0;
    }
    // Find the bracket
    let mut idx = 0;
    for (i, &(h, _, _)) in ATMOSPHERE_TABLE.iter().enumerate() {
        if h <= alt_km {
            idx = i;
        } else {
            break;
        }
    }
    let (h0, rho0, scale_h) = ATMOSPHERE_TABLE[idx];
    rho0 * (-((alt_km - h0) / scale_h)).exp()
}

/// Compute atmospheric drag acceleration for an ECI state.
///
/// Parameterized on the **inverse ballistic coefficient**
/// `inv_beta = C_d·A/m = 1/β` (m²/kg). Uses the co-rotating atmosphere
/// relative velocity `v_rel = v − ω_⊕ × r` (with
/// [`EARTH_ROTATION_RATE`]), so Coriolis/centrifugal drag effects are
/// captured without an ECEF state. Altitude for the density lookup is
/// geometric, `‖r‖ − equatorial_radius`, using
/// [`GravityModel::EARTH_WGS84`] — the atmosphere model is Earth-specific.
///
/// Returns zero outside the 0–1000 km altitude band of [`ATMOSPHERE_TABLE`]
/// and for negligible relative speed.
pub fn drag_acceleration(pos: &Vector3<f64>, vel: &Vector3<f64>, inv_beta: f64) -> Vector3<f64> {
    let r = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
    let alt = r - GravityModel::EARTH_WGS84.equatorial_radius;
    if !(0.0..=1_000_000.0).contains(&alt) {
        return Vector3::zeros();
    }

    // Approximate atmospheric co-rotation velocity
    let v_atm = Vector3::new(
        -EARTH_ROTATION_RATE * pos.y,
        EARTH_ROTATION_RATE * pos.x,
        0.0,
    );
    let v_rel = Vector3::new(vel.x - v_atm.x, vel.y - v_atm.y, vel.z - v_atm.z);
    let v_rel_mag = (v_rel.x * v_rel.x + v_rel.y * v_rel.y + v_rel.z * v_rel.z).sqrt();

    if v_rel_mag < 1e-10 {
        return Vector3::zeros();
    }

    let rho = atmosphere_density(alt);
    let drag_factor = -0.5 * inv_beta * rho * v_rel_mag;

    Vector3::new(
        drag_factor * v_rel.x,
        drag_factor * v_rel.y,
        drag_factor * v_rel.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Density lookup ───────────────────────────────────────────────────

    #[test]
    fn density_decreases_with_altitude() {
        let rho_200 = atmosphere_density(200_000.0);
        let rho_400 = atmosphere_density(400_000.0);
        let rho_800 = atmosphere_density(800_000.0);
        assert!(rho_200 > rho_400);
        assert!(rho_400 > rho_800);
        assert!(rho_200 > 0.0);
    }

    #[test]
    fn density_clamps_to_sea_level_below_zero_altitude() {
        assert_eq!(atmosphere_density(-100.0), ATMOSPHERE_TABLE[0].1);
    }

    #[test]
    fn density_is_zero_above_table_ceiling() {
        assert_eq!(atmosphere_density(1_000_001.0), 0.0);
    }

    #[test]
    fn density_equals_table_value_at_bracket_base() {
        // At exactly 100 km the exponential factor is exp(0) = 1, so the
        // lookup returns the table row's nominal density.
        assert_eq!(atmosphere_density(100_000.0), 5.297e-7);
    }

    // ── Drag acceleration ────────────────────────────────────────────────

    /// LEO-ish state inside the sensible atmosphere (~300 km altitude).
    fn leo_state() -> (Vector3<f64>, Vector3<f64>) {
        let r = GravityModel::EARTH_WGS84.equatorial_radius + 300_000.0;
        (Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, 7_700.0, 0.0))
    }

    #[test]
    fn drag_opposes_corotating_relative_velocity() {
        let (pos, vel) = leo_state();
        let inv_beta = 2.2 * 20.0 / 500.0;
        let acc = drag_acceleration(&pos, &vel, inv_beta);

        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;

        assert!(acc.norm() > 0.0, "drag should be nonzero at 300 km");
        assert!(acc.dot(&v_rel) < 0.0, "drag should oppose v_rel");
        // Anti-parallel: the cross product with v_rel vanishes.
        assert!(acc.cross(&v_rel).norm() < 1e-12 * acc.norm() * v_rel.norm());
    }

    #[test]
    fn drag_magnitude_matches_half_rho_v2_over_beta() {
        let (pos, vel) = leo_state();
        let inv_beta = 2.2 * 20.0 / 500.0;
        let acc = drag_acceleration(&pos, &vel, inv_beta);

        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let v_rel = vel - v_atm;
        let rho = atmosphere_density(pos.norm() - GravityModel::EARTH_WGS84.equatorial_radius);
        let expected = 0.5 * rho * v_rel.norm_squared() * inv_beta;

        assert!((acc.norm() - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn drag_scales_linearly_with_inv_beta() {
        let (pos, vel) = leo_state();
        let a1 = drag_acceleration(&pos, &vel, 0.01);
        let a2 = drag_acceleration(&pos, &vel, 0.02);
        assert!((a2.norm() - 2.0 * a1.norm()).abs() < 1e-12 * a1.norm());
    }

    #[test]
    fn drag_vanishes_above_atmosphere_ceiling() {
        let r = GravityModel::EARTH_WGS84.equatorial_radius + 1_500_000.0;
        let acc = drag_acceleration(
            &Vector3::new(r, 0.0, 0.0),
            &Vector3::new(0.0, 7_000.0, 0.0),
            0.1,
        );
        assert_eq!(acc, Vector3::zeros());
    }

    #[test]
    fn drag_vanishes_when_moving_with_the_atmosphere() {
        let (pos, _) = leo_state();
        // Velocity exactly equal to the atmospheric co-rotation velocity.
        let v_atm = Vector3::new(
            -EARTH_ROTATION_RATE * pos.y,
            EARTH_ROTATION_RATE * pos.x,
            0.0,
        );
        let acc = drag_acceleration(&pos, &v_atm, 0.1);
        assert_eq!(acc, Vector3::zeros());
    }
}
