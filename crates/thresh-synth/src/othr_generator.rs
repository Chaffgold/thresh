//! Synthetic OTHR (Over-The-Horizon Radar) measurement generator.
//!
//! Generates realistic OTHR measurements from target waypoints, including
//! skip zone blanking, Doppler-based detection probability, diurnal coverage
//! variation, and configurable measurement noise.

use crate::trajectory::Waypoint;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use thresh_core::measurement::{Measurement, PropagationMode};

/// Configuration for a synthetic OTHR sensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OthrConfig {
    /// Sensor identifier.
    pub sensor_id: u32,
    /// Transmitter latitude (radians).
    pub transmitter_lat_rad: f64,
    /// Transmitter longitude (radians).
    pub transmitter_lon_rad: f64,
    /// Transmitter altitude above WGS84 ellipsoid (meters).
    pub transmitter_alt_m: f64,
    /// Operating frequency (MHz).
    pub freq_mhz: f64,
    /// Bandwidth (MHz).
    pub bandwidth_mhz: f64,
    /// Pulse repetition frequency (Hz).
    pub prf_hz: f64,
    /// Coherent integration time (seconds).
    pub integration_time_s: f64,
    /// Range measurement noise standard deviation (meters). Typical: 15,000-25,000 m.
    pub range_sigma_m: f64,
    /// Azimuth measurement noise standard deviation (radians). Typical: ~0.017 rad (1 deg).
    pub azimuth_sigma_rad: f64,
    /// Doppler measurement noise standard deviation (m/s). Typical: ~1.0 m/s.
    pub doppler_sigma_m_s: f64,
    /// Minimum ground range — skip zone (meters). Typical: ~1,000,000 m.
    pub min_range_m: f64,
    /// Maximum ground range (meters). Typical: ~3,500,000 m.
    pub max_range_m: f64,
    /// Base detection probability (before Doppler discrimination).
    pub base_p_detection: f64,
    /// Minimum Doppler separation from clutter for full detection (m/s). Typical: ~5 m/s.
    pub min_doppler_clutter_m_s: f64,
}

impl OthrConfig {
    /// ROTHR-class preset: 5-28 MHz, 1000-3500 km range.
    pub fn rothr() -> Self {
        Self {
            sensor_id: 0,
            transmitter_lat_rad: 0.3176, // ~18.2 deg N (Virginia Capes area)
            transmitter_lon_rad: -1.147, // ~-65.7 deg W
            transmitter_alt_m: 0.0,
            freq_mhz: 15.0,
            bandwidth_mhz: 0.025,
            prf_hz: 50.0,
            integration_time_s: 10.0,
            range_sigma_m: 20_000.0,
            azimuth_sigma_rad: 0.017, // ~1 deg
            doppler_sigma_m_s: 1.0,
            min_range_m: 1_000_000.0,
            max_range_m: 3_500_000.0,
            base_p_detection: 0.8,
            min_doppler_clutter_m_s: 5.0,
        }
    }

    /// JORN-class preset: 6-30 MHz, similar range.
    pub fn jorn() -> Self {
        Self {
            sensor_id: 0,
            transmitter_lat_rad: -0.395, // ~-22.6 deg S (Alice Springs area)
            transmitter_lon_rad: 2.347,  // ~134.4 deg E
            transmitter_alt_m: 0.0,
            freq_mhz: 18.0,
            bandwidth_mhz: 0.030,
            prf_hz: 55.0,
            integration_time_s: 12.0,
            range_sigma_m: 18_000.0,
            azimuth_sigma_rad: 0.015, // slightly better than ROTHR
            doppler_sigma_m_s: 0.8,
            min_range_m: 1_000_000.0,
            max_range_m: 3_500_000.0,
            base_p_detection: 0.85,
            min_doppler_clutter_m_s: 4.5,
        }
    }
}

/// Detection probability based on target Doppler separation from clutter.
///
/// Targets with |doppler| < `min_doppler_clutter_m_s` are attenuated.
/// P_d ramps linearly from ~0.1 at zero Doppler to `base_p_detection`
/// at `min_doppler_clutter_m_s`.
pub fn doppler_detection_probability(radial_velocity_m_s: f64, config: &OthrConfig) -> f64 {
    let abs_doppler = radial_velocity_m_s.abs();
    let min_clutter = config.min_doppler_clutter_m_s;

    if abs_doppler >= min_clutter {
        config.base_p_detection
    } else {
        // Linear ramp from 0.1 at zero to base_p_detection at min_clutter
        let fraction = abs_doppler / min_clutter;
        0.1 + fraction * (config.base_p_detection - 0.1)
    }
}

/// Adjust detection probability based on time of day (simplified diurnal model).
///
/// Returns a multiplier in [0.6, 1.0] that peaks at local noon (12.0 hours).
/// Better propagation during daytime due to higher ionization.
pub fn diurnal_factor(solar_local_time_hours: f64) -> f64 {
    // Cosine model: peak at noon, trough at midnight
    let hour_angle = (solar_local_time_hours - 12.0) * std::f64::consts::PI / 12.0;
    // cos ranges from -1 (midnight) to 1 (noon)
    // Map to [0.6, 1.0]: factor = 0.8 + 0.2 * cos(angle)
    0.8 + 0.2 * hour_angle.cos()
}

/// Generate an OTHR measurement from a target waypoint.
///
/// Returns `None` if the target is outside coverage (skip zone, max range,
/// or below detection threshold after Doppler discrimination and diurnal effects).
///
/// The waypoint position is treated as ENU/Cartesian (meters) relative to the
/// transmitter. Ground range is computed as `sqrt(x^2 + y^2)` and azimuth as
/// `atan2(east, north)`.
pub fn generate_othr<R: Rng>(
    waypoint: &Waypoint,
    config: &OthrConfig,
    solar_local_time_hours: f64,
    rng: &mut R,
) -> Option<Measurement> {
    let east = waypoint.position[0];
    let north = waypoint.position[1];

    // Step 1: Ground range from transmitter to target
    let ground_range = (east * east + north * north).sqrt();

    // Step 2: Skip zone and max range check
    if ground_range < config.min_range_m || ground_range > config.max_range_m {
        return None;
    }

    // Step 3: Azimuth from transmitter to target (clockwise from north)
    let azimuth = east.atan2(north);

    // Step 4: Radial velocity (Doppler) — component of velocity along the
    // line-of-sight direction (transmitter-to-target bearing on the ground plane)
    let range_inv = 1.0 / ground_range;
    let unit_east = east * range_inv;
    let unit_north = north * range_inv;
    let radial_velocity = waypoint.velocity[0] * unit_east + waypoint.velocity[1] * unit_north;

    // Step 5: Detection probability = Doppler discrimination * diurnal factor
    let p_doppler = doppler_detection_probability(radial_velocity, config);
    let p_diurnal = diurnal_factor(solar_local_time_hours);
    let p_detection = p_doppler * p_diurnal;

    if rng.random::<f64>() > p_detection {
        return None;
    }

    // Step 6: Add measurement noise
    let range_noise = Normal::new(0.0, config.range_sigma_m).ok()?;
    let az_noise = Normal::new(0.0, config.azimuth_sigma_rad).ok()?;
    let doppler_noise = Normal::new(0.0, config.doppler_sigma_m_s).ok()?;

    let noisy_range = ground_range + range_noise.sample(rng);
    let noisy_azimuth = azimuth + az_noise.sample(rng);
    let noisy_doppler = radial_velocity + doppler_noise.sample(rng);

    // Step 7: Build measurement
    Some(Measurement::Othr {
        ground_range_m: noisy_range,
        azimuth_rad: noisy_azimuth,
        doppler_m_s: noisy_doppler,
        propagation_mode: PropagationMode::FLayer,
        time: waypoint.time,
        sensor_id: config.sensor_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn default_config() -> OthrConfig {
        OthrConfig {
            sensor_id: 1,
            transmitter_lat_rad: 0.0,
            transmitter_lon_rad: 0.0,
            transmitter_alt_m: 0.0,
            freq_mhz: 15.0,
            bandwidth_mhz: 0.025,
            prf_hz: 50.0,
            integration_time_s: 10.0,
            range_sigma_m: 20_000.0,
            azimuth_sigma_rad: 0.017,
            doppler_sigma_m_s: 1.0,
            min_range_m: 1_000_000.0,
            max_range_m: 3_500_000.0,
            base_p_detection: 0.8,
            min_doppler_clutter_m_s: 5.0,
        }
    }

    fn make_waypoint(east: f64, north: f64, ve: f64, vn: f64) -> Waypoint {
        Waypoint {
            time: 100.0,
            position: [east, north, 10_000.0],
            velocity: [ve, vn, 0.0],
        }
    }

    // ── Task 6.7/6.8: Measurements within expected bounds ─────────────

    #[test]
    fn measurement_within_expected_bounds() {
        let config = default_config();
        let wp = make_waypoint(0.0, 2_000_000.0, 100.0, -50.0);
        let mut rng = StdRng::seed_from_u64(42);

        let mut measurements = Vec::new();
        for _ in 0..500 {
            if let Some(m) = generate_othr(&wp, &config, 12.0, &mut rng) {
                measurements.push(m);
            }
        }

        assert!(
            !measurements.is_empty(),
            "should produce at least some measurements"
        );

        for m in &measurements {
            if let Measurement::Othr {
                ground_range_m,
                azimuth_rad,
                ..
            } = m
            {
                // Range should be within several sigma of true range
                assert!(
                    (*ground_range_m - 2_000_000.0).abs() < 5.0 * config.range_sigma_m,
                    "range out of bounds: {ground_range_m}"
                );
                // Azimuth should be within several sigma of true azimuth (0 rad = due north)
                assert!(
                    (*azimuth_rad - 0.0).abs() < 5.0 * config.azimuth_sigma_rad,
                    "azimuth out of bounds: {azimuth_rad}"
                );
            }
        }
    }

    // ── Skip zone prevents close-range detections ─────────────────────

    #[test]
    fn skip_zone_prevents_close_range() {
        let config = default_config();
        // Target at 500 km — inside skip zone
        let wp = make_waypoint(0.0, 500_000.0, 100.0, 0.0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            assert!(
                generate_othr(&wp, &config, 12.0, &mut rng).is_none(),
                "skip zone should block close targets"
            );
        }
    }

    // ── Max range prevents far detections ─────────────────────────────

    #[test]
    fn max_range_prevents_far_detections() {
        let config = default_config();
        // Target at 4000 km — beyond max range
        let wp = make_waypoint(0.0, 4_000_000.0, 100.0, 0.0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            assert!(
                generate_othr(&wp, &config, 12.0, &mut rng).is_none(),
                "max range should block far targets"
            );
        }
    }

    // ── Doppler discrimination ────────────────────────────────────────

    #[test]
    fn doppler_discrimination_high_vs_low_velocity() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let n_trials = 2000;

        // High-velocity target (radial velocity well above clutter threshold)
        let wp_fast = make_waypoint(0.0, 2_000_000.0, 0.0, -200.0);
        let fast_detections = (0..n_trials)
            .filter(|_| generate_othr(&wp_fast, &config, 12.0, &mut rng).is_some())
            .count();

        // Low-velocity target (radial velocity near zero — in clutter)
        let wp_slow = make_waypoint(0.0, 2_000_000.0, 0.1, 0.0);
        let slow_detections = (0..n_trials)
            .filter(|_| generate_othr(&wp_slow, &config, 12.0, &mut rng).is_some())
            .count();

        assert!(
            fast_detections > slow_detections,
            "fast targets ({fast_detections}) should be detected more often than slow ({slow_detections})"
        );
    }

    #[test]
    fn doppler_detection_probability_values() {
        let config = default_config();

        // At zero Doppler, probability should be 0.1
        let p0 = doppler_detection_probability(0.0, &config);
        assert!(
            (p0 - 0.1).abs() < 1e-10,
            "p_d at zero doppler: {p0}, expected 0.1"
        );

        // At min_clutter threshold, probability should be base_p_detection
        let p_max = doppler_detection_probability(config.min_doppler_clutter_m_s, &config);
        assert!(
            (p_max - config.base_p_detection).abs() < 1e-10,
            "p_d at clutter threshold: {p_max}, expected {}",
            config.base_p_detection
        );

        // Above threshold, should also be base_p_detection
        let p_above = doppler_detection_probability(100.0, &config);
        assert!(
            (p_above - config.base_p_detection).abs() < 1e-10,
            "p_d above threshold: {p_above}"
        );

        // Midpoint should be midway
        let p_mid = doppler_detection_probability(config.min_doppler_clutter_m_s / 2.0, &config);
        let expected_mid = 0.1 + 0.5 * (config.base_p_detection - 0.1);
        assert!(
            (p_mid - expected_mid).abs() < 1e-10,
            "p_d at midpoint: {p_mid}, expected {expected_mid}"
        );
    }

    // ── Noise statistics match configured sigmas ──────────────────────

    #[test]
    fn noise_statistics_match_sigmas() {
        let config = default_config();
        // Place target due north at 2000 km, moving radially at -100 m/s
        let wp = make_waypoint(0.0, 2_000_000.0, 0.0, -100.0);
        let mut rng = StdRng::seed_from_u64(123);

        let true_range = 2_000_000.0;
        let true_azimuth = 0.0;
        let true_doppler = -100.0; // all velocity is radial (northward component)

        let mut range_errors = Vec::new();
        let mut az_errors = Vec::new();
        let mut doppler_errors = Vec::new();

        for _ in 0..10_000 {
            if let Some(Measurement::Othr {
                ground_range_m,
                azimuth_rad,
                doppler_m_s,
                ..
            }) = generate_othr(&wp, &config, 12.0, &mut rng)
            {
                range_errors.push(ground_range_m - true_range);
                az_errors.push(azimuth_rad - true_azimuth);
                doppler_errors.push(doppler_m_s - true_doppler);
            }
        }

        assert!(
            range_errors.len() > 100,
            "need enough samples: got {}",
            range_errors.len()
        );

        let range_std = std_dev(&range_errors);
        let az_std = std_dev(&az_errors);
        let doppler_std = std_dev(&doppler_errors);

        // Allow 20% tolerance for statistical estimation
        assert!(
            (range_std - config.range_sigma_m).abs() / config.range_sigma_m < 0.2,
            "range std {range_std} not close to configured {}",
            config.range_sigma_m
        );
        assert!(
            (az_std - config.azimuth_sigma_rad).abs() / config.azimuth_sigma_rad < 0.2,
            "azimuth std {az_std} not close to configured {}",
            config.azimuth_sigma_rad
        );
        assert!(
            (doppler_std - config.doppler_sigma_m_s).abs() / config.doppler_sigma_m_s < 0.2,
            "doppler std {doppler_std} not close to configured {}",
            config.doppler_sigma_m_s
        );
    }

    // ── Diurnal factor ────────────────────────────────────────────────

    #[test]
    fn diurnal_factor_range() {
        // Should be in [0.6, 1.0]
        for hour in 0..24 {
            let f = diurnal_factor(hour as f64);
            assert!(
                (0.6..=1.0).contains(&f),
                "diurnal_factor({hour}) = {f}, out of [0.6, 1.0]"
            );
        }
    }

    #[test]
    fn diurnal_factor_peaks_at_noon() {
        let noon = diurnal_factor(12.0);
        let midnight = diurnal_factor(0.0);
        assert!(
            (noon - 1.0).abs() < 1e-10,
            "noon factor should be 1.0: {noon}"
        );
        assert!(
            (midnight - 0.6).abs() < 1e-10,
            "midnight factor should be 0.6: {midnight}"
        );
        assert!(noon > midnight);
    }

    // ── Preset configs have valid parameters ──────────────────────────

    #[test]
    fn rothr_preset_valid() {
        let c = OthrConfig::rothr();
        assert!(c.freq_mhz > 0.0);
        assert!(c.min_range_m > 0.0);
        assert!(c.max_range_m > c.min_range_m);
        assert!(c.range_sigma_m > 0.0);
        assert!(c.azimuth_sigma_rad > 0.0);
        assert!(c.doppler_sigma_m_s > 0.0);
        assert!(c.base_p_detection > 0.0 && c.base_p_detection <= 1.0);
        assert!(c.min_doppler_clutter_m_s > 0.0);
        assert!(c.prf_hz > 0.0);
        assert!(c.integration_time_s > 0.0);
    }

    #[test]
    fn jorn_preset_valid() {
        let c = OthrConfig::jorn();
        assert!(c.freq_mhz > 0.0);
        assert!(c.min_range_m > 0.0);
        assert!(c.max_range_m > c.min_range_m);
        assert!(c.range_sigma_m > 0.0);
        assert!(c.azimuth_sigma_rad > 0.0);
        assert!(c.doppler_sigma_m_s > 0.0);
        assert!(c.base_p_detection > 0.0 && c.base_p_detection <= 1.0);
        assert!(c.min_doppler_clutter_m_s > 0.0);
        assert!(c.prf_hz > 0.0);
        assert!(c.integration_time_s > 0.0);
    }

    // ── Serialization roundtrip ───────────────────────────────────────

    #[test]
    fn config_serialization_roundtrip() {
        let c = OthrConfig::rothr();
        let json = serde_json::to_string(&c).expect("serialize");
        let c2: OthrConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(c.sensor_id, c2.sensor_id);
        assert_eq!(c.freq_mhz, c2.freq_mhz);
        assert_eq!(c.range_sigma_m, c2.range_sigma_m);
    }

    // ── Propagation mode is FLayer by default ─────────────────────────

    #[test]
    fn default_propagation_mode_is_f_layer() {
        let config = default_config();
        let wp = make_waypoint(0.0, 2_000_000.0, 0.0, -100.0);
        let mut rng = StdRng::seed_from_u64(99);

        let m = generate_othr(&wp, &config, 12.0, &mut rng).expect("should detect");
        if let Measurement::Othr {
            propagation_mode, ..
        } = m
        {
            assert_eq!(propagation_mode, PropagationMode::FLayer);
        } else {
            panic!("expected OTHR measurement");
        }
    }

    // ── Helper: standard deviation ────────────────────────────────────

    fn std_dev(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }
}
