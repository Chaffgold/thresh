//! OTHR noise model: range-dependent measurement noise and ionospheric bias.

use serde::{Deserialize, Serialize};

/// OTHR measurement noise configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OthrNoiseConfig {
    /// Range standard deviation (m). Typical: 10,000-30,000 m.
    pub range_sigma_m: f64,
    /// Azimuth standard deviation (rad). Typical: 0.009-0.035 rad (0.5-2 deg).
    pub azimuth_sigma_rad: f64,
    /// Doppler velocity standard deviation (m/s). Typical: ~1 m/s.
    pub doppler_sigma_m_s: f64,
    /// Range bias uncertainty from ionospheric height error (m).
    pub range_bias_sigma_m: f64,
}

impl OthrNoiseConfig {
    /// Default configuration with typical OTHR noise values.
    pub fn default_config() -> Self {
        Self {
            range_sigma_m: 15_000.0,
            azimuth_sigma_rad: 0.017, // ~1 degree
            doppler_sigma_m_s: 1.0,
            range_bias_sigma_m: 5_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Task 4.2 — Range-dependent noise
// ---------------------------------------------------------------------------

/// Scale noise parameters by range and ionospheric instability.
///
/// Noise grows approximately as sqrt(range / reference_range) to account for
/// spreading loss and increased ionospheric path variability at longer ranges.
pub fn range_dependent_noise(base_config: &OthrNoiseConfig, range_km: f64) -> OthrNoiseConfig {
    let reference_range_km = 1000.0;
    let scale = (range_km / reference_range_km).sqrt().max(1.0);
    OthrNoiseConfig {
        range_sigma_m: base_config.range_sigma_m * scale,
        azimuth_sigma_rad: base_config.azimuth_sigma_rad * scale,
        doppler_sigma_m_s: base_config.doppler_sigma_m_s * scale,
        range_bias_sigma_m: base_config.range_bias_sigma_m * scale,
    }
}

// ---------------------------------------------------------------------------
// Task 4.3 — Ionospheric bias
// ---------------------------------------------------------------------------

/// Systematic ground-range error (m) caused by virtual-height uncertainty.
///
/// The ground range derived from group delay assumes a known virtual height.
/// An error `height_error_km` in that height shifts the inferred ground range.
///
/// For a flat-Earth approximation at moderate ranges:
///   delta_range ~ 2 * h_err / tan(elev)
/// where elev is estimated from range and a nominal virtual height.
pub fn ionospheric_bias_m(range_km: f64, height_error_km: f64) -> f64 {
    // Nominal virtual height (F-layer)
    let nominal_vh = 300.0;
    let half_range = range_km / 2.0;
    // Elevation angle from flat-Earth triangle
    let elev = (nominal_vh / half_range).atan();
    // Bias in km, converted to m
    let bias_km = 2.0 * height_error_km / elev.tan();
    bias_km * 1000.0
}

// =========================================================================
// Tests — Task 4.4
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn default_config_in_range() {
        let cfg = OthrNoiseConfig::default_config();
        assert!((10_000.0..=30_000.0).contains(&cfg.range_sigma_m));
        assert!((0.009..=0.035).contains(&cfg.azimuth_sigma_rad));
        assert!(cfg.doppler_sigma_m_s > 0.0);
    }

    #[test]
    fn range_dependent_noise_increases() {
        let base = OthrNoiseConfig::default_config();
        let near = range_dependent_noise(&base, 500.0);
        let far = range_dependent_noise(&base, 3000.0);
        assert!(
            far.range_sigma_m > near.range_sigma_m,
            "Noise should increase with range"
        );
        assert!(
            far.azimuth_sigma_rad > near.azimuth_sigma_rad,
            "Azimuth noise should increase with range"
        );
    }

    #[test]
    fn range_dependent_noise_at_reference() {
        let base = OthrNoiseConfig::default_config();
        let at_ref = range_dependent_noise(&base, 1000.0);
        assert!(
            (at_ref.range_sigma_m - base.range_sigma_m).abs() < 1e-6,
            "At reference range, noise should match base"
        );
    }

    #[test]
    fn ionospheric_bias_increases_with_height_error() {
        let b1 = ionospheric_bias_m(1500.0, 10.0).abs();
        let b2 = ionospheric_bias_m(1500.0, 30.0).abs();
        assert!(b2 > b1, "Bias should increase with height error");
    }

    #[test]
    fn noise_statistics_match_config() {
        let cfg = OthrNoiseConfig::default_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let range_dist = Normal::new(0.0, cfg.range_sigma_m).unwrap();
        let az_dist = Normal::new(0.0, cfg.azimuth_sigma_rad).unwrap();
        let dop_dist = Normal::new(0.0, cfg.doppler_sigma_m_s).unwrap();

        let n = 10_000;
        let mut range_samples = Vec::with_capacity(n);
        let mut az_samples = Vec::with_capacity(n);
        let mut dop_samples = Vec::with_capacity(n);

        for _ in 0..n {
            range_samples.push(range_dist.sample(&mut rng));
            az_samples.push(az_dist.sample(&mut rng));
            dop_samples.push(dop_dist.sample(&mut rng));
        }

        // Check that sample standard deviations are within 5% of configured values
        let range_std = sample_std(&range_samples);
        let az_std = sample_std(&az_samples);
        let dop_std = sample_std(&dop_samples);

        assert!(
            (range_std - cfg.range_sigma_m).abs() / cfg.range_sigma_m < 0.05,
            "Range noise std {range_std} should be near {:.0}",
            cfg.range_sigma_m
        );
        assert!(
            (az_std - cfg.azimuth_sigma_rad).abs() / cfg.azimuth_sigma_rad < 0.05,
            "Azimuth noise std {az_std} should be near {:.4}",
            cfg.azimuth_sigma_rad
        );
        assert!(
            (dop_std - cfg.doppler_sigma_m_s).abs() / cfg.doppler_sigma_m_s < 0.05,
            "Doppler noise std {dop_std} should be near {:.1}",
            cfg.doppler_sigma_m_s
        );
    }

    fn sample_std(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }
}
