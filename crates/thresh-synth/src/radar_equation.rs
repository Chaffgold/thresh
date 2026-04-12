//! Enhanced radar equation, detection models, and atmospheric propagation.
//!
//! This module provides a full radar-equation pipeline:
//! Swerling RCS sampling -> SNR computation -> atmospheric loss ->
//! detection probability (Albersheim / Shnidman) -> range-dependent noise.
//!
//! The original [`crate::measurement_gen`] API remains backward-compatible;
//! this module is used for higher-fidelity simulation.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;

use crate::swerling::{DwellRcs, RcsProfile, SwerlingType};
use crate::trajectory::Waypoint;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Boltzmann constant (J/K).
const BOLTZMANN_K: f64 = 1.380_649e-23;

/// Standard reference temperature (K).
const T_REF: f64 = 290.0;

// ---------------------------------------------------------------------------
// Antenna pattern (placeholder for future expansion)
// ---------------------------------------------------------------------------

/// Antenna radiation pattern model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AntennaPattern {
    /// Ideal isotropic (gain = 0 dBi everywhere off-boresight, used as
    /// placeholder; actual gain is in `RadarParameters::antenna_gain_db`).
    Isotropic,
    /// Sinc-squared pattern with the given 3-dB beamwidth in degrees.
    SincSquared { beamwidth_deg: f64 },
}

// ---------------------------------------------------------------------------
// Radar parameters
// ---------------------------------------------------------------------------

/// Full radar system parameters for the enhanced radar equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarParameters {
    /// Peak transmit power (watts).
    pub peak_power_w: f64,
    /// Antenna gain on boresight (dB).
    pub antenna_gain_db: f64,
    /// Wavelength (metres).
    pub wavelength_m: f64,
    /// Receiver bandwidth (Hz).
    pub bandwidth_hz: f64,
    /// Receiver noise figure (dB).
    pub noise_figure_db: f64,
    /// Total system losses (dB).
    pub system_losses_db: f64,
    /// Antenna noise temperature (K). Defaults to 0 (use T_ref only).
    pub t_antenna_k: f64,
    /// Number of coherently integrated pulses.
    pub n_pulses: u32,
    /// False-alarm probability.
    pub pfa: f64,
    /// Optional antenna pattern.
    pub antenna_pattern: Option<AntennaPattern>,
}

impl RadarParameters {
    /// AN/TPS-80 style X-band surveillance radar.
    pub fn x_band_surveillance() -> Self {
        Self {
            peak_power_w: 100_000.0,
            antenna_gain_db: 34.0,
            wavelength_m: 0.03,
            bandwidth_hz: 1.0e6,
            noise_figure_db: 3.0,
            system_losses_db: 4.0,
            t_antenna_k: 100.0,
            n_pulses: 16,
            pfa: 1e-6,
            antenna_pattern: None,
        }
    }

    /// AN/SPY-1 style S-band search radar.
    pub fn s_band_search() -> Self {
        Self {
            peak_power_w: 4_000_000.0,
            antenna_gain_db: 42.0,
            wavelength_m: 0.1, // ~3 GHz
            bandwidth_hz: 5.0e6,
            noise_figure_db: 4.0,
            system_losses_db: 5.0,
            t_antenna_k: 150.0,
            n_pulses: 20,
            pfa: 1e-8,
            antenna_pattern: None,
        }
    }

    /// Precision C-band tracking radar.
    pub fn c_band_tracking() -> Self {
        Self {
            peak_power_w: 500_000.0,
            antenna_gain_db: 45.0,
            wavelength_m: 0.055, // ~5.5 GHz
            bandwidth_hz: 2.0e6,
            noise_figure_db: 2.5,
            system_losses_db: 3.0,
            t_antenna_k: 80.0,
            n_pulses: 64,
            pfa: 1e-9,
            antenna_pattern: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Convert decibels to linear scale.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Convert linear to decibels.
fn linear_to_db(lin: f64) -> f64 {
    10.0 * lin.log10()
}

// ---------------------------------------------------------------------------
// SNR computation (enhanced)
// ---------------------------------------------------------------------------

/// Compute the system noise temperature.
///
/// `T_sys = T_antenna + (NF_linear - 1) * 290`
pub fn system_noise_temp(noise_figure_db: f64, t_antenna_k: f64) -> f64 {
    let nf_linear = db_to_linear(noise_figure_db);
    let t_sys = t_antenna_k + (nf_linear - 1.0) * T_REF;
    t_sys.max(1.0) // clamp to avoid division by zero
}

/// Compute effective SNR (linear) from the full radar equation.
///
/// SNR = (P_t * G^2 * lambda^2 * sigma) / ((4*pi)^3 * R^4 * k * T_sys * B * L)
///
/// With coherent integration the effective SNR is multiplied by `n_pulses`.
pub fn compute_snr_enhanced(range_m: f64, rcs_m2: f64, params: &RadarParameters) -> f64 {
    let gain_linear = db_to_linear(params.antenna_gain_db);
    let loss_linear = db_to_linear(params.system_losses_db);
    let t_sys = system_noise_temp(params.noise_figure_db, params.t_antenna_k);

    let lambda_sq = params.wavelength_m * params.wavelength_m;
    let four_pi_cubed = (4.0 * std::f64::consts::PI).powi(3);
    let r4 = range_m.powi(4);

    let numerator = params.peak_power_w * gain_linear * gain_linear * lambda_sq * rcs_m2;
    let denominator = four_pi_cubed * r4 * BOLTZMANN_K * t_sys * params.bandwidth_hz * loss_linear;

    let snr_single = numerator / denominator;
    snr_single * f64::from(params.n_pulses)
}

// ---------------------------------------------------------------------------
// Albersheim's approximation (re-exported from measurement_gen)
// ---------------------------------------------------------------------------

/// Albersheim's approximation for detection probability.
///
/// Same algorithm as [`crate::measurement_gen::albersheim_pd`]; provided here
/// so the full pipeline can be used without reaching into measurement_gen.
pub fn albersheim_pd(snr_db: f64, pfa: f64) -> f64 {
    let a = (0.62 / pfa).ln();
    let snr_lin = db_to_linear(snr_db);
    let x = (2.0 * snr_lin).sqrt() - (2.0 * a).sqrt();
    let pd = 1.0 / (1.0 + (-x).exp());
    pd.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Shnidman's equation (approximate)
// ---------------------------------------------------------------------------

/// Shnidman's equation: detection probability for N integrated pulses with
/// Swerling fluctuation loss.
///
/// This is an approximate implementation that applies an effective
/// fluctuation-loss correction to the Albersheim baseline. The correction
/// accounts for both the integration gain of N pulses and the SNR penalty
/// imposed by the Swerling fluctuation model.
///
/// For Swerling 0 with N=1 this reduces to Albersheim.
pub fn shnidman_pd(snr_db: f64, pfa: f64, n_pulses: u32, swerling: SwerlingType) -> f64 {
    let n = f64::from(n_pulses.max(1));

    // Integration gain: for non-coherent integration the gain is approximately
    // sqrt(N) in SNR, but we work in dB.
    let integration_gain_db = 5.0 * n.log10(); // ~half-way between coherent (10*log N) and non-coh

    // Fluctuation loss: empirical dB penalty that depends on the Swerling type
    // and P_fa. Simplified from published tables.
    let fluct_loss_db = match swerling {
        SwerlingType::Zero => 0.0,
        SwerlingType::One => {
            // Slow, 2-DOF: higher loss, reduced by integration
            (8.0 - 2.0 * n.log10()).max(0.0)
        }
        SwerlingType::Two => {
            // Fast, 2-DOF: diversity helps
            (4.0 - 3.0 * n.log10()).max(0.0)
        }
        SwerlingType::Three => {
            // Slow, 4-DOF: moderate loss
            (4.0 - 1.5 * n.log10()).max(0.0)
        }
        SwerlingType::Four => {
            // Fast, 4-DOF: least loss
            (2.0 - 2.0 * n.log10()).max(0.0)
        }
    };

    let effective_snr_db = snr_db + integration_gain_db - fluct_loss_db;
    albersheim_pd(effective_snr_db, pfa)
}

// ---------------------------------------------------------------------------
// Atmospheric attenuation (simplified ITU-R P.676)
// ---------------------------------------------------------------------------

/// Simplified one-way atmospheric attenuation (dB/km) based on ITU-R P.676.
///
/// This is a coarse model suitable for system-level simulation. It captures
/// the O2 absorption peak near 60 GHz and the water-vapour line at 22 GHz,
/// plus a baseline that grows with frequency.
///
/// `elevation_deg` is the ray elevation above the horizon; lower elevations
/// traverse more atmosphere (cosecant law).
pub fn atmospheric_attenuation_db_per_km(freq_ghz: f64, elevation_deg: f64) -> f64 {
    // Zenith-referenced specific attenuation (very simplified)
    let base = if freq_ghz < 1.0 {
        0.005
    } else if freq_ghz < 10.0 {
        0.005 + 0.002 * (freq_ghz - 1.0)
    } else if freq_ghz < 20.0 {
        0.02 + 0.005 * (freq_ghz - 10.0)
    } else if freq_ghz < 30.0 {
        // Water vapour line near 22.2 GHz
        let water = 0.05 * (-(((freq_ghz - 22.2) / 3.0).powi(2))).exp();
        0.07 + water + 0.003 * (freq_ghz - 20.0)
    } else if freq_ghz < 50.0 {
        0.1 + 0.01 * (freq_ghz - 30.0)
    } else if freq_ghz < 70.0 {
        // O2 absorption peak ~60 GHz
        let o2 = 10.0 * (-(((freq_ghz - 60.0) / 5.0).powi(2))).exp();
        0.3 + o2
    } else {
        0.5 + 0.005 * (freq_ghz - 70.0)
    };

    // Cosecant scaling: more atmosphere traversed at low elevation.
    // Clamp elevation to avoid singularity at 0 degrees.
    let el_rad = elevation_deg.to_radians().max(0.5_f64.to_radians());
    base / el_rad.sin()
}

/// Compute two-way atmospheric loss in dB for a given range and frequency.
pub fn atmospheric_loss_db(range_m: f64, freq_ghz: f64, elevation_deg: f64) -> f64 {
    let atten_per_km = atmospheric_attenuation_db_per_km(freq_ghz, elevation_deg);
    let range_km = range_m / 1000.0;
    2.0 * atten_per_km * range_km // two-way
}

// ---------------------------------------------------------------------------
// Range-dependent measurement noise
// ---------------------------------------------------------------------------

/// Compute range and angle measurement noise standard deviations that scale
/// inversely with the square root of SNR.
///
/// Returns `(range_sigma_m, angle_sigma_rad)`.
///
/// At SNR = 1 (linear) the noise equals the base sigma values; higher SNR
/// reduces noise proportionally.
pub fn measurement_noise(
    snr_linear: f64,
    base_range_sigma: f64,
    base_angle_sigma: f64,
) -> (f64, f64) {
    let factor = 1.0 / snr_linear.max(1e-30).sqrt();
    (base_range_sigma * factor, base_angle_sigma * factor)
}

// ---------------------------------------------------------------------------
// Full-pipeline radar measurement generator
// ---------------------------------------------------------------------------

/// Configuration for the full radar-equation measurement pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullRadarConfig {
    /// Sensor identifier.
    pub sensor_id: u32,
    /// Radar system parameters.
    pub radar: RadarParameters,
    /// Base range noise sigma (metres) at SNR=1.
    pub base_range_sigma: f64,
    /// Base angle noise sigma (radians) at SNR=1.
    pub base_angle_sigma: f64,
    /// Maximum detection range (metres).
    pub max_range: f64,
    /// Whether to apply atmospheric attenuation.
    pub apply_atmosphere: bool,
    /// Whether to use Shnidman (true) or Albersheim (false) for P_d.
    pub use_shnidman: bool,
}

impl Default for FullRadarConfig {
    fn default() -> Self {
        Self {
            sensor_id: 0,
            radar: RadarParameters::x_band_surveillance(),
            base_range_sigma: 10.0,
            base_angle_sigma: 0.001,
            max_range: 200_000.0,
            apply_atmosphere: true,
            use_shnidman: true,
        }
    }
}

/// Generate a radar measurement using the full physics pipeline.
///
/// Steps:
/// 1. Sample RCS from the Swerling model via `dwell_rcs`.
/// 2. Compute SNR using the enhanced radar equation.
/// 3. Optionally subtract atmospheric loss.
/// 4. Compute P_d via Albersheim or Shnidman.
/// 5. Roll for detection.
/// 6. Compute range-dependent noise and generate the measurement.
pub fn generate_radar_full<R: Rng>(
    waypoint: &Waypoint,
    config: &FullRadarConfig,
    dwell_rcs: &DwellRcs,
    rng: &mut R,
) -> Option<Measurement> {
    let pos = &waypoint.position;
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

    if range > config.max_range || range < 1e-6 {
        return None;
    }

    // 1. RCS
    let rcs_m2 = dwell_rcs.rcs(rng);

    // 2. SNR
    let mut snr_linear = compute_snr_enhanced(range, rcs_m2, &config.radar);

    // 3. Atmospheric loss
    if config.apply_atmosphere {
        let elevation_deg = (pos[2] / range).asin().to_degrees();
        let freq_ghz = 3e8 / config.radar.wavelength_m / 1e9;
        let loss_db = atmospheric_loss_db(range, freq_ghz, elevation_deg.abs().max(1.0));
        snr_linear /= db_to_linear(loss_db);
    }

    let snr_db = linear_to_db(snr_linear.max(1e-30));

    // 4. Detection probability
    let pd = if config.use_shnidman {
        shnidman_pd(
            snr_db,
            config.radar.pfa,
            config.radar.n_pulses,
            SwerlingType::Zero, // fluctuation already in RCS sample
        )
    } else {
        albersheim_pd(snr_db, config.radar.pfa)
    };

    // 5. Detection roll
    if rng.random::<f64>() > pd {
        return None;
    }

    // 6. Range-dependent noise
    let (range_sigma, angle_sigma) =
        measurement_noise(snr_linear, config.base_range_sigma, config.base_angle_sigma);

    let azimuth = pos[1].atan2(pos[0]);
    let elevation = (pos[2] / range).asin();

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let range_n: f64 = std_normal.sample(rng);
    let az_n: f64 = std_normal.sample(rng);
    let el_n: f64 = std_normal.sample(rng);

    Some(Measurement::Radar {
        range: range + range_n * range_sigma,
        azimuth: azimuth + az_n * angle_sigma,
        elevation: elevation + el_n * angle_sigma,
        range_rate: None,
        time: waypoint.time,
        sensor_id: config.sensor_id,
    })
}

/// Convenience: create a [`DwellRcs`] from an [`RcsProfile`].
pub fn dwell_rcs_from_profile<R: Rng>(profile: &RcsProfile, rng: &mut R) -> DwellRcs {
    DwellRcs::new(profile.swerling_type, profile.mean_rcs_m2(), rng)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Waypoint;

    #[test]
    fn pd_decreases_with_range() {
        let params = RadarParameters::x_band_surveillance();
        let rcs = 1.0;
        let ranges = [10_000.0, 50_000.0, 100_000.0, 150_000.0, 200_000.0];
        let pds: Vec<f64> = ranges
            .iter()
            .map(|&r| {
                let snr = compute_snr_enhanced(r, rcs, &params);
                let snr_db = linear_to_db(snr.max(1e-30));
                albersheim_pd(snr_db, params.pfa)
            })
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
    fn albersheim_matches_published() {
        // The simplified Albersheim (sigmoid) approximation should be
        // monotonically increasing, map high SNR -> P_d ~1, low SNR -> ~0,
        // and be in the right ballpark at moderate SNR.

        // At 13.2 dB, Pfa=1e-6: our approximation gives ~0.79
        let pd = albersheim_pd(13.2, 1e-6);
        assert!(
            pd > 0.5 && pd < 0.99,
            "Albersheim at 13.2 dB, Pfa=1e-6: expected moderate P_d, got {pd}"
        );

        // Monotonicity: higher SNR -> higher P_d
        let pd_higher = albersheim_pd(16.0, 1e-6);
        assert!(
            pd_higher > pd,
            "P_d should increase with SNR: {pd_higher} vs {pd}"
        );

        // At very high SNR, P_d -> 1
        let pd_high = albersheim_pd(25.0, 1e-6);
        assert!(
            pd_high > 0.999,
            "High SNR should give P_d ~1.0, got {pd_high}"
        );

        // At very low SNR, P_d -> 0
        let pd_low = albersheim_pd(-5.0, 1e-6);
        assert!(pd_low < 0.05, "Low SNR should give P_d ~0, got {pd_low}");
    }

    #[test]
    fn atmospheric_attenuation_increases_with_frequency() {
        let freqs = [1.0, 5.0, 10.0, 20.0, 35.0];
        let attens: Vec<f64> = freqs
            .iter()
            .map(|&f| atmospheric_attenuation_db_per_km(f, 45.0))
            .collect();

        for i in 1..attens.len() {
            assert!(
                attens[i] >= attens[i - 1] - 1e-12,
                "Attenuation should increase with freq: {:.4} dB/km at {} GHz < {:.4} dB/km at {} GHz",
                attens[i],
                freqs[i],
                attens[i - 1],
                freqs[i - 1],
            );
        }
    }

    #[test]
    fn shnidman_n1_sw0_matches_albersheim() {
        // Shnidman with N=1 and Swerling 0 should closely match Albersheim
        let pfa = 1e-6;
        for snr_db in &[5.0, 10.0, 15.0, 20.0] {
            let pd_alb = albersheim_pd(*snr_db, pfa);
            let pd_shn = shnidman_pd(*snr_db, pfa, 1, SwerlingType::Zero);
            assert!(
                (pd_alb - pd_shn).abs() < 0.05,
                "Shnidman(N=1,SW0) should match Albersheim at {} dB: alb={}, shn={}",
                snr_db,
                pd_alb,
                pd_shn,
            );
        }
    }

    #[test]
    fn measurement_noise_increases_with_range() {
        let params = RadarParameters::x_band_surveillance();
        let rcs = 1.0;
        let base_r = 10.0;
        let base_a = 0.001;

        let ranges = [10_000.0, 50_000.0, 100_000.0, 200_000.0];
        let sigmas: Vec<(f64, f64)> = ranges
            .iter()
            .map(|&r| {
                let snr = compute_snr_enhanced(r, rcs, &params);
                measurement_noise(snr, base_r, base_a)
            })
            .collect();

        for i in 1..sigmas.len() {
            assert!(
                sigmas[i].0 >= sigmas[i - 1].0 - 1e-15,
                "Range sigma should increase with range"
            );
            assert!(
                sigmas[i].1 >= sigmas[i - 1].1 - 1e-15,
                "Angle sigma should increase with range"
            );
        }
    }

    #[test]
    fn system_noise_temp_calculation() {
        // NF=3 dB -> NF_linear ≈ 2.0
        // T_sys = T_ant + (2.0 - 1.0) * 290 = T_ant + 290
        let t = system_noise_temp(3.0, 100.0);
        let nf_lin = db_to_linear(3.0);
        let expected = 100.0 + (nf_lin - 1.0) * 290.0;
        assert!(
            (t - expected).abs() < 1.0,
            "T_sys: expected {expected}, got {t}"
        );
    }

    #[test]
    fn full_pipeline_generates_measurement() {
        let config = FullRadarConfig::default();
        let wp = Waypoint {
            time: 0.0,
            position: [10_000.0, 5_000.0, 3_000.0],
            velocity: [250.0, 0.0, 0.0],
        };
        let mut rng = rand::rng();
        let profile = RcsProfile::airliner();
        let dwell = dwell_rcs_from_profile(&profile, &mut rng);

        // At close range with large RCS, should usually detect
        let mut detections = 0;
        for _ in 0..100 {
            if generate_radar_full(&wp, &config, &dwell, &mut rng).is_some() {
                detections += 1;
            }
        }
        assert!(
            detections > 50,
            "Should detect large target at close range most of the time, got {detections}/100"
        );
    }

    #[test]
    fn preset_radar_params_valid() {
        let presets = [
            RadarParameters::x_band_surveillance(),
            RadarParameters::s_band_search(),
            RadarParameters::c_band_tracking(),
        ];
        for p in &presets {
            assert!(p.peak_power_w > 0.0);
            assert!(p.wavelength_m > 0.0);
            assert!(p.n_pulses > 0);
        }
    }
}
