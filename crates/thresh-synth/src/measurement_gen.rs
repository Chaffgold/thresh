//! Measurement generators for radar, EO/IR, and ADS-B sensors.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use thresh_core::measurement::Measurement;

use crate::trajectory::Waypoint;

/// Boltzmann constant (J/K).
const BOLTZMANN_K: f64 = 1.380_649e-23;

/// Standard reference temperature (K).
const T_REF: f64 = 290.0;

/// Radar equation parameters for RCS-dependent detection probability.
#[derive(Debug, Clone)]
pub struct RadarEquationConfig {
    /// Peak transmit power (watts).
    pub peak_power_w: f64,
    /// Antenna gain (dB).
    pub antenna_gain_db: f64,
    /// Wavelength (meters).
    pub wavelength_m: f64,
    /// Receiver bandwidth (Hz).
    pub bandwidth_hz: f64,
    /// Receiver noise figure (dB).
    pub noise_figure_db: f64,
    /// Total system losses (dB).
    pub system_losses_db: f64,
    /// Number of coherently integrated pulses.
    pub n_pulses: u32,
    /// False alarm probability.
    pub pfa: f64,
}

impl RadarEquationConfig {
    /// Typical X-band surveillance radar preset.
    pub fn x_band_surveillance() -> Self {
        Self {
            peak_power_w: 100_000.0, // 100 kW
            antenna_gain_db: 34.0,   // ~34 dBi
            wavelength_m: 0.03,      // X-band ~10 GHz
            bandwidth_hz: 1.0e6,     // 1 MHz
            noise_figure_db: 3.0,    // 3 dB
            system_losses_db: 4.0,   // 4 dB total
            n_pulses: 16,            // 16 pulses integrated
            pfa: 1e-6,               // 10^-6 false alarm rate
        }
    }
}

/// Convert decibels to linear scale.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Compute single-pulse SNR (in linear) from the radar equation.
///
/// SNR = (Pt * G² * λ² * σ) / ((4π)³ * R⁴ * k * T_sys * B * L)
///
/// With `n_pulses` coherent integration the effective SNR is multiplied by `n_pulses`.
pub fn compute_snr(range_m: f64, rcs_m2: f64, config: &RadarEquationConfig) -> f64 {
    let gain_linear = db_to_linear(config.antenna_gain_db);
    let nf_linear = db_to_linear(config.noise_figure_db);
    let loss_linear = db_to_linear(config.system_losses_db);

    // System noise temperature
    let t_sys = (nf_linear - 1.0) * T_REF;
    // Clamp to avoid division by zero for very low noise figures
    let t_sys = t_sys.max(1.0);

    let lambda_sq = config.wavelength_m * config.wavelength_m;
    let four_pi_cubed = (4.0 * std::f64::consts::PI).powi(3);
    let r4 = range_m.powi(4);

    let numerator = config.peak_power_w * gain_linear * gain_linear * lambda_sq * rcs_m2;
    let denominator = four_pi_cubed * r4 * BOLTZMANN_K * t_sys * config.bandwidth_hz * loss_linear;

    let snr_single = numerator / denominator;
    // Coherent integration gain
    snr_single * f64::from(config.n_pulses)
}

/// Albersheim's approximation: compute detection probability from SNR (dB) and P_fa.
///
/// Reference: Albersheim (1981). Valid for 10^-7 < P_fa < 10^-3 and single-pulse.
pub fn albersheim_pd(snr_db: f64, pfa: f64) -> f64 {
    // A = ln(0.62 / P_fa)
    let a = (0.62 / pfa).ln();

    // Convert SNR from dB to linear
    let snr_lin = 10.0_f64.powf(snr_db / 10.0);

    // B = log(A) approximation constant
    // Albersheim: Z = SNR_linear
    // P_d = (1 + exp(-Z))^-1 where Z is mapped through the approximation
    // Simplified Albersheim: P_d ≈ 1/(1 + exp(-x))
    //   where x = sqrt(2 * snr_lin) - sqrt(2 * a)
    // This is the logistic (sigmoid) form of Albersheim's result.

    let x = (2.0 * snr_lin).sqrt() - (2.0 * a).sqrt();

    // Sigmoid
    let pd = 1.0 / (1.0 + (-x).exp());
    pd.clamp(0.0, 1.0)
}

/// Compute detection probability: use radar equation if configured, otherwise fixed.
pub fn detection_probability(range_m: f64, rcs_m2: Option<f64>, config: &RadarConfig) -> f64 {
    match (&config.radar_equation, rcs_m2) {
        (Some(req), Some(rcs)) => {
            let snr_linear = compute_snr(range_m, rcs, req);
            let snr_db = 10.0 * snr_linear.log10();
            albersheim_pd(snr_db, req.pfa)
        }
        (Some(req), None) => {
            // Use a default RCS of 1.0 m² when not specified
            let snr_linear = compute_snr(range_m, 1.0, req);
            let snr_db = 10.0 * snr_linear.log10();
            albersheim_pd(snr_db, req.pfa)
        }
        _ => config.p_detection,
    }
}

/// Radar measurement generator configuration.
#[derive(Debug, Clone)]
pub struct RadarConfig {
    pub sensor_id: u32,
    /// Range noise standard deviation (meters).
    pub range_sigma: f64,
    /// Azimuth noise standard deviation (radians).
    pub azimuth_sigma: f64,
    /// Elevation noise standard deviation (radians).
    pub elevation_sigma: f64,
    /// Probability of detection [0, 1]. Used when `radar_equation` is `None`.
    pub p_detection: f64,
    /// Mean number of clutter returns per scan.
    pub clutter_rate: f64,
    /// Maximum range (meters).
    pub max_range: f64,
    /// Optional radar equation config for RCS-dependent P_d.
    pub radar_equation: Option<RadarEquationConfig>,
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            sensor_id: 0,
            range_sigma: 10.0,
            azimuth_sigma: 0.001,
            elevation_sigma: 0.001,
            p_detection: 0.9,
            clutter_rate: 2.0,
            max_range: 200_000.0,
            radar_equation: None,
        }
    }
}

/// Generate a radar measurement from a waypoint (or None if not detected).
///
/// When `rcs_m2` is `Some(...)` and the config has a `radar_equation`, detection
/// probability is computed from the radar equation. Otherwise the fixed
/// `config.p_detection` is used.
pub fn generate_radar<R: Rng>(
    waypoint: &Waypoint,
    config: &RadarConfig,
    rng: &mut R,
) -> Option<Measurement> {
    generate_radar_with_rcs(waypoint, config, None, rng)
}

/// Generate a radar measurement with an explicit target RCS.
pub fn generate_radar_with_rcs<R: Rng>(
    waypoint: &Waypoint,
    config: &RadarConfig,
    rcs_m2: Option<f64>,
    rng: &mut R,
) -> Option<Measurement> {
    let pos = &waypoint.position;
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

    if range > config.max_range || range < 1e-6 {
        return None;
    }

    // Detection probability — RCS-dependent or fixed
    let pd = detection_probability(range, rcs_m2, config);
    if rng.random::<f64>() > pd {
        return None;
    }

    let azimuth = pos[1].atan2(pos[0]);
    let elevation = (pos[2] / range).asin();

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let range_n: f64 = std_normal.sample(rng);
    let az_n: f64 = std_normal.sample(rng);
    let el_n: f64 = std_normal.sample(rng);

    Some(Measurement::Radar {
        range: range + range_n * config.range_sigma,
        azimuth: azimuth + az_n * config.azimuth_sigma,
        elevation: elevation + el_n * config.elevation_sigma,
        range_rate: None,
        time: waypoint.time,
        sensor_id: config.sensor_id,
    })
}

/// EO/IR measurement generator configuration.
#[derive(Debug, Clone)]
pub struct EoIrConfig {
    pub sensor_id: u32,
    /// Angular noise standard deviation (radians).
    pub angular_sigma: f64,
    /// Probability of detection.
    pub p_detection: f64,
    /// Field of view half-angle (radians).
    pub fov_half_angle: f64,
}

impl Default for EoIrConfig {
    fn default() -> Self {
        Self {
            sensor_id: 1,
            angular_sigma: 0.0005,
            p_detection: 0.85,
            fov_half_angle: 0.5,
        }
    }
}

/// Generate an EO/IR bearing-only measurement.
pub fn generate_eoir<R: Rng>(
    waypoint: &Waypoint,
    config: &EoIrConfig,
    rng: &mut R,
) -> Option<Measurement> {
    let pos = &waypoint.position;
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    if range < 1e-6 {
        return None;
    }

    let azimuth = pos[1].atan2(pos[0]);
    let elevation = (pos[2] / range).asin();

    // FOV check
    if azimuth.abs() > config.fov_half_angle || elevation.abs() > config.fov_half_angle {
        return None;
    }

    if rng.random::<f64>() > config.p_detection {
        return None;
    }

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let az_n: f64 = std_normal.sample(rng);
    let el_n: f64 = std_normal.sample(rng);

    Some(Measurement::EoIr {
        azimuth: azimuth + az_n * config.angular_sigma,
        elevation: elevation + el_n * config.angular_sigma,
        time: waypoint.time,
        sensor_id: config.sensor_id,
    })
}

/// ADS-B message generator configuration.
#[derive(Debug, Clone)]
pub struct AdsBConfig {
    /// Position noise standard deviation (meters).
    pub position_sigma: f64,
    /// Message dropout probability.
    pub dropout_rate: f64,
}

impl Default for AdsBConfig {
    fn default() -> Self {
        Self {
            position_sigma: 5.0,
            dropout_rate: 0.05,
        }
    }
}

/// Generate an ADS-B position report (1 Hz).
pub fn generate_adsb<R: Rng>(
    waypoint: &Waypoint,
    config: &AdsBConfig,
    rng: &mut R,
) -> Option<Measurement> {
    if rng.random::<f64>() < config.dropout_rate {
        return None;
    }

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let px: f64 = std_normal.sample(rng);
    let py: f64 = std_normal.sample(rng);
    let pz: f64 = std_normal.sample(rng);

    Some(Measurement::AdsB {
        lat: waypoint.position[0] + px * config.position_sigma,
        lon: waypoint.position[1] + py * config.position_sigma,
        alt: waypoint.position[2] + pz * config.position_sigma,
        velocity: Some(waypoint.velocity),
        time: waypoint.time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Segment, SegmentType, Trajectory};

    fn make_test_trajectory() -> Vec<Waypoint> {
        Trajectory {
            target_id: 0,
            initial_position: [10000.0, 5000.0, 3000.0],
            initial_velocity: [250.0, 0.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: 10.0,
            }],
            dt: 1.0,
        }
        .generate()
    }

    #[test]
    fn radar_noise_statistics() {
        let config = RadarConfig {
            p_detection: 1.0, // always detect
            clutter_rate: 0.0,
            ..Default::default()
        };
        let wps = make_test_trajectory();
        let wp = &wps[5]; // middle waypoint

        let mut rng = rand::rng();
        let mut ranges = Vec::new();

        for _ in 0..10_000 {
            if let Some(Measurement::Radar { range, .. }) = generate_radar(wp, &config, &mut rng) {
                let true_range =
                    (wp.position[0].powi(2) + wp.position[1].powi(2) + wp.position[2].powi(2))
                        .sqrt();
                ranges.push(range - true_range);
            }
        }

        // Check noise is roughly Gaussian with correct sigma
        let mean: f64 = ranges.iter().sum::<f64>() / ranges.len() as f64;
        let variance: f64 =
            ranges.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ranges.len() as f64;
        let std = variance.sqrt();

        assert!(mean.abs() < 1.0, "Mean bias too large: {mean}");
        assert!(
            (std - config.range_sigma).abs() < 1.0,
            "Std {std} far from configured {}",
            config.range_sigma
        );
    }

    #[test]
    fn pd_decreases_with_range() {
        let req = RadarEquationConfig::x_band_surveillance();
        let config = RadarConfig {
            radar_equation: Some(req),
            ..Default::default()
        };
        let rcs = Some(1.0); // 1 m²

        let ranges = [10_000.0, 50_000.0, 100_000.0, 150_000.0, 200_000.0];
        let pds: Vec<f64> = ranges
            .iter()
            .map(|&r| detection_probability(r, rcs, &config))
            .collect();

        for i in 1..pds.len() {
            assert!(
                pds[i] <= pds[i - 1] + 1e-12,
                "P_d should decrease with range: P_d({})={} > P_d({})={}",
                ranges[i],
                pds[i],
                ranges[i - 1],
                pds[i - 1],
            );
        }
    }

    #[test]
    fn pd_increases_with_rcs() {
        let req = RadarEquationConfig::x_band_surveillance();
        let config = RadarConfig {
            radar_equation: Some(req),
            ..Default::default()
        };
        let range = 100_000.0;

        let rcs_values = [0.01, 0.1, 1.0, 10.0, 100.0];
        let pds: Vec<f64> = rcs_values
            .iter()
            .map(|&rcs| detection_probability(range, Some(rcs), &config))
            .collect();

        for i in 1..pds.len() {
            assert!(
                pds[i] >= pds[i - 1] - 1e-12,
                "P_d should increase with RCS: P_d(rcs={})={} < P_d(rcs={})={}",
                rcs_values[i],
                pds[i],
                rcs_values[i - 1],
                pds[i - 1],
            );
        }
    }

    #[test]
    fn pd_near_one_at_close_range() {
        let req = RadarEquationConfig::x_band_surveillance();
        let config = RadarConfig {
            radar_equation: Some(req),
            ..Default::default()
        };

        let pd = detection_probability(1_000.0, Some(1.0), &config);
        assert!(pd > 0.99, "P_d at close range should be ~1.0, got {pd}");
    }

    #[test]
    fn pd_near_zero_at_extreme_range() {
        let req = RadarEquationConfig::x_band_surveillance();
        let config = RadarConfig {
            radar_equation: Some(req),
            max_range: 1_000_000.0, // extend max range for test
            ..Default::default()
        };

        let pd = detection_probability(800_000.0, Some(0.01), &config);
        assert!(
            pd < 0.1,
            "P_d at extreme range with small RCS should be ~0, got {pd}"
        );
    }

    #[test]
    fn fixed_pd_when_no_radar_equation() {
        let config = RadarConfig::default();
        // Without radar_equation, detection_probability returns fixed p_detection
        let pd = detection_probability(50_000.0, Some(10.0), &config);
        assert!(
            (pd - config.p_detection).abs() < f64::EPSILON,
            "Without radar_equation, should use fixed p_detection"
        );
    }

    #[test]
    fn generate_radar_backward_compatible() {
        // The original generate_radar API (no RCS) still works
        let config = RadarConfig {
            p_detection: 1.0,
            ..Default::default()
        };
        let wp = Waypoint {
            time: 0.0,
            position: [10_000.0, 0.0, 0.0],
            velocity: [0.0; 3],
        };
        let mut rng = rand::rng();
        let result = generate_radar(&wp, &config, &mut rng);
        assert!(result.is_some(), "Should detect at p_detection=1.0");
    }
}
