//! EO/IR sensor physics: Planck radiance, atmospheric transmission, IR signatures,
//! and physics-based detection/measurement generation.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;

use crate::trajectory::Waypoint;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Planck constant (J·s).
const PLANCK_H: f64 = 6.626_070_15e-34;

/// Speed of light (m/s).
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;

/// Boltzmann constant (J/K).
const BOLTZMANN_K: f64 = 1.380_649e-23;

// ---------------------------------------------------------------------------
// Task 6.1 — Planck blackbody spectral radiance
// ---------------------------------------------------------------------------

/// Spectral radiance L(λ, T) in W/(m²·sr·m).
///
/// Planck's law: L = (2hc²/λ⁵) / (exp(hc/(λkT)) - 1)
pub fn planck_radiance(wavelength_m: f64, temperature_k: f64) -> f64 {
    let c1 = 2.0 * PLANCK_H * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    let c2 = PLANCK_H * SPEED_OF_LIGHT / (BOLTZMANN_K * temperature_k);
    let lambda5 = wavelength_m.powi(5);
    let exponent = c2 / wavelength_m;

    c1 / (lambda5 * (exponent.exp() - 1.0))
}

// ---------------------------------------------------------------------------
// Task 6.2 — Band-integrated radiance
// ---------------------------------------------------------------------------

/// MWIR band lower bound (3 μm).
pub const MWIR_MIN: f64 = 3.0e-6;
/// MWIR band upper bound (5 μm).
pub const MWIR_MAX: f64 = 5.0e-6;
/// LWIR band lower bound (8 μm).
pub const LWIR_MIN: f64 = 8.0e-6;
/// LWIR band upper bound (12 μm).
pub const LWIR_MAX: f64 = 12.0e-6;

/// Integrate Planck function over a wavelength band using the trapezoidal rule.
///
/// Returns band radiance in W/(m²·sr).
pub fn band_radiance(
    lambda_min_m: f64,
    lambda_max_m: f64,
    temperature_k: f64,
    n_steps: usize,
) -> f64 {
    assert!(n_steps >= 2, "n_steps must be >= 2");
    let dlambda = (lambda_max_m - lambda_min_m) / n_steps as f64;
    let mut sum = 0.5 * planck_radiance(lambda_min_m, temperature_k);
    for i in 1..n_steps {
        let lambda = lambda_min_m + i as f64 * dlambda;
        sum += planck_radiance(lambda, temperature_k);
    }
    sum += 0.5 * planck_radiance(lambda_max_m, temperature_k);
    sum * dlambda
}

// ---------------------------------------------------------------------------
// Task 6.3 — Target IR signatures
// ---------------------------------------------------------------------------

/// Infrared signature of a target, describing thermal emission areas and temperatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrSignature {
    /// Exhaust plume temperature (K).
    pub exhaust_temperature_k: f64,
    /// Airframe skin temperature (K).
    pub skin_temperature_k: f64,
    /// Projected exhaust area (m²).
    pub exhaust_area_m2: f64,
    /// Projected skin area (m²).
    pub skin_area_m2: f64,
}

impl IrSignature {
    /// Fighter in afterburner: ~1800 K exhaust, ~400 K skin.
    pub fn fighter_afterburner() -> Self {
        Self {
            exhaust_temperature_k: 1800.0,
            skin_temperature_k: 400.0,
            exhaust_area_m2: 0.5,
            skin_area_m2: 20.0,
        }
    }

    /// Fighter at military power: ~900 K exhaust, ~350 K skin.
    pub fn fighter_military() -> Self {
        Self {
            exhaust_temperature_k: 900.0,
            skin_temperature_k: 350.0,
            exhaust_area_m2: 0.3,
            skin_area_m2: 18.0,
        }
    }

    /// Commercial airliner: ~700 K exhaust, ~300 K skin.
    pub fn airliner() -> Self {
        Self {
            exhaust_temperature_k: 700.0,
            skin_temperature_k: 300.0,
            exhaust_area_m2: 0.8,
            skin_area_m2: 80.0,
        }
    }

    /// Electric UAV: no hot exhaust, ~310 K skin.
    pub fn uav_electric() -> Self {
        Self {
            exhaust_temperature_k: 310.0,
            skin_temperature_k: 310.0,
            exhaust_area_m2: 0.0,
            skin_area_m2: 2.0,
        }
    }

    /// Ballistic reentry vehicle: ~2000 K plasma-heated skin.
    pub fn ballistic_reentry() -> Self {
        Self {
            exhaust_temperature_k: 2000.0,
            skin_temperature_k: 2000.0,
            exhaust_area_m2: 0.0,
            skin_area_m2: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Task 6.4 — Atmospheric transmission
// ---------------------------------------------------------------------------

/// Spectral band for EO/IR sensors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpectralBand {
    /// Mid-wave infrared (3–5 μm).
    Mwir,
    /// Long-wave infrared (8–12 μm).
    Lwir,
    /// Visible (~0.4–0.7 μm).
    Visible,
}

/// Typical sea-level extinction coefficient (1/m) for each band.
fn extinction_coefficient(band: SpectralBand) -> f64 {
    match band {
        SpectralBand::Mwir => 0.2e-3,    // 0.2 / km → 0.0002 / m
        SpectralBand::Lwir => 0.5e-3,    // 0.5 / km
        SpectralBand::Visible => 0.1e-3, // 0.1 / km
    }
}

/// Beer-Lambert atmospheric transmission: τ = exp(−α · R).
pub fn atmospheric_transmission(range_m: f64, band: SpectralBand) -> f64 {
    let alpha = extinction_coefficient(band);
    (-alpha * range_m).exp()
}

// ---------------------------------------------------------------------------
// Task 6.5 — IR sensor parameters
// ---------------------------------------------------------------------------

/// Configuration for an infrared / electro-optical sensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrSensorConfig {
    /// Sensor identifier.
    pub sensor_id: u32,
    /// Noise-equivalent temperature difference (K).
    pub netd_k: f64,
    /// Instantaneous field of view (rad).
    pub ifov_rad: f64,
    /// Aperture diameter (m).
    pub aperture_diameter_m: f64,
    /// Spectral band.
    pub spectral_band: SpectralBand,
    /// Detector integration time (s).
    pub integration_time_s: f64,
    /// Half-angle of total sensor field of view (rad).
    pub fov_half_angle_rad: f64,
}

// ---------------------------------------------------------------------------
// Task 6.6 — Detection range / probability computation
// ---------------------------------------------------------------------------

/// Compute the total target radiant intensity in a given band (W/sr).
///
/// Sums contributions from exhaust and skin, subtracting background.
fn target_excess_intensity(
    target: &IrSignature,
    band: SpectralBand,
    background_temperature_k: f64,
) -> f64 {
    let (lmin, lmax) = band_limits(band);
    let n_steps = 200;

    let bg_radiance = band_radiance(lmin, lmax, background_temperature_k, n_steps);

    let exhaust_radiance = band_radiance(lmin, lmax, target.exhaust_temperature_k, n_steps);
    let skin_radiance = band_radiance(lmin, lmax, target.skin_temperature_k, n_steps);

    // Excess radiant intensity (W/sr) = ΔL · A  (Lambertian emitter → L × A / π, but
    // for detection we use contrast intensity which is ΔL · A assuming hemisphere).
    let exhaust_excess = (exhaust_radiance - bg_radiance).max(0.0) * target.exhaust_area_m2;
    let skin_excess = (skin_radiance - bg_radiance).max(0.0) * target.skin_area_m2;

    exhaust_excess + skin_excess
}

/// Return wavelength band limits for a [`SpectralBand`].
fn band_limits(band: SpectralBand) -> (f64, f64) {
    match band {
        SpectralBand::Mwir => (MWIR_MIN, MWIR_MAX),
        SpectralBand::Lwir => (LWIR_MIN, LWIR_MAX),
        SpectralBand::Visible => (0.4e-6, 0.7e-6),
    }
}

/// Compute detection probability from IR physics.
///
/// Pipeline: target excess intensity → atmospheric attenuation → irradiance at
/// aperture → signal-to-noise ratio (using NETD as noise floor) → P_d via
/// Albersheim-like sigmoid.
pub fn ir_detection_probability(
    target: &IrSignature,
    sensor: &IrSensorConfig,
    range_m: f64,
    background_temperature_k: f64,
) -> f64 {
    if range_m < 1e-6 {
        return 1.0;
    }

    // 1. Target excess intensity (W/sr) in the sensor's band
    let intensity = target_excess_intensity(target, sensor.spectral_band, background_temperature_k);
    if intensity <= 0.0 {
        return 0.0;
    }

    // 2. Atmospheric transmission
    let tau = atmospheric_transmission(range_m, sensor.spectral_band);

    // 3. Irradiance at sensor aperture (W/m²)
    let irradiance = intensity * tau / (range_m * range_m);

    // 4. Collected power (W) through aperture
    let aperture_area = std::f64::consts::FRAC_PI_4 * sensor.aperture_diameter_m.powi(2);
    let signal_power = irradiance * aperture_area;

    // 5. Noise-equivalent power (NEP): derived from NETD.
    //    NETD represents the temperature difference that yields SNR=1.
    //    NEP ≈ NETD × dL/dT × A_det × IFOV  (simplified).
    //    We use a simpler proxy: noise floor ~ NETD × background_dL × aperture × IFOV²
    let (lmin, lmax) = band_limits(sensor.spectral_band);
    let bg_radiance = band_radiance(lmin, lmax, background_temperature_k, 200);

    // dL/dT approximation: finite difference at 1 K
    let bg_radiance_p1 = band_radiance(lmin, lmax, background_temperature_k + 1.0, 200);
    let dl_dt = bg_radiance_p1 - bg_radiance;

    // NEP ~ NETD × (dL/dT) × aperture_area × IFOV²
    let nep = sensor.netd_k * dl_dt * aperture_area * sensor.ifov_rad * sensor.ifov_rad;

    if nep <= 0.0 {
        return 1.0;
    }

    // 6. SNR
    let snr = signal_power / nep;

    // 7. Albersheim-like sigmoid: P_d = sigmoid(x) where x = sqrt(2·SNR_lin) - threshold
    //    Using a detection threshold corresponding to ~P_fa = 1e-6
    let pfa = 1e-6_f64;
    let a = (0.62 / pfa).ln();
    let x = (2.0 * snr).sqrt() - (2.0 * a).sqrt();
    let pd = 1.0 / (1.0 + (-x).exp());
    pd.clamp(0.0, 1.0)
}

/// Compute angular noise standard deviation (rad) from SNR.
///
/// Higher SNR → lower angular noise; floor at IFOV/2.
fn angular_noise_from_snr(snr: f64, ifov_rad: f64) -> f64 {
    // Noise ≈ IFOV / SNR (centroiding limited by SNR), clamped to [IFOV/100, IFOV]
    if snr <= 0.0 {
        return ifov_rad;
    }
    let sigma = ifov_rad / snr.sqrt();
    sigma.clamp(ifov_rad * 0.01, ifov_rad)
}

// ---------------------------------------------------------------------------
// Task 6.7 — Physics-based EO/IR measurement generator
// ---------------------------------------------------------------------------

/// Generate an EO/IR bearing-only measurement using physics-based detection.
///
/// Returns `None` if the target is outside the sensor FOV or fails detection.
pub fn generate_eoir_physics<R: Rng>(
    waypoint: &Waypoint,
    target_signature: &IrSignature,
    sensor: &IrSensorConfig,
    background_temp_k: f64,
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
    if azimuth.abs() > sensor.fov_half_angle_rad || elevation.abs() > sensor.fov_half_angle_rad {
        return None;
    }

    // Physics-based detection probability
    let pd = ir_detection_probability(target_signature, sensor, range, background_temp_k);
    if rng.random::<f64>() > pd {
        return None;
    }

    // Angular noise depends on SNR: recompute a quick SNR for noise scaling
    let intensity =
        target_excess_intensity(target_signature, sensor.spectral_band, background_temp_k);
    let tau = atmospheric_transmission(range, sensor.spectral_band);
    let irradiance = intensity * tau / (range * range);
    let aperture_area = std::f64::consts::FRAC_PI_4 * sensor.aperture_diameter_m.powi(2);
    let signal_power = irradiance * aperture_area;

    let (lmin, lmax) = band_limits(sensor.spectral_band);
    let bg_radiance = band_radiance(lmin, lmax, background_temp_k, 200);
    let bg_radiance_p1 = band_radiance(lmin, lmax, background_temp_k + 1.0, 200);
    let dl_dt = bg_radiance_p1 - bg_radiance;
    let nep = sensor.netd_k * dl_dt * aperture_area * sensor.ifov_rad * sensor.ifov_rad;
    let snr = if nep > 0.0 {
        signal_power / nep
    } else {
        1000.0
    };

    let ang_sigma = angular_noise_from_snr(snr, sensor.ifov_rad);

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let az_n: f64 = std_normal.sample(rng);
    let el_n: f64 = std_normal.sample(rng);

    Some(Measurement::EoIr {
        azimuth: azimuth + az_n * ang_sigma,
        elevation: elevation + el_n * ang_sigma,
        time: waypoint.time,
        sensor_id: sensor.sensor_id,
    })
}

// ---------------------------------------------------------------------------
// Task 6.8 — Preset sensor configurations
// ---------------------------------------------------------------------------

impl IrSensorConfig {
    /// Wide-FOV MWIR search sensor (typical IRST).
    pub fn mwir_search() -> Self {
        Self {
            sensor_id: 10,
            netd_k: 0.025,
            ifov_rad: 0.5e-3,
            aperture_diameter_m: 0.15,
            spectral_band: SpectralBand::Mwir,
            integration_time_s: 5e-3,
            fov_half_angle_rad: 0.5,
        }
    }

    /// Narrow-FOV LWIR tracking sensor (precision).
    pub fn lwir_tracking() -> Self {
        Self {
            sensor_id: 11,
            netd_k: 0.020,
            ifov_rad: 0.1e-3,
            aperture_diameter_m: 0.20,
            spectral_band: SpectralBand::Lwir,
            integration_time_s: 10e-3,
            fov_half_angle_rad: 0.1,
        }
    }

    /// Visible-band imaging camera.
    pub fn visible_camera() -> Self {
        Self {
            sensor_id: 12,
            netd_k: 0.050,
            ifov_rad: 0.2e-3,
            aperture_diameter_m: 0.10,
            spectral_band: SpectralBand::Visible,
            integration_time_s: 1e-3,
            fov_half_angle_rad: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Tasks 6.9–6.11 — Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- 6.9: Planck / Wien's law ---

    #[test]
    fn wiens_law_peak() {
        // Wien's displacement law: λ_max * T ≈ 2898 μm·K
        let wien_b = 2898.0e-6; // m·K
        let temperature = 5778.0; // Sun's surface
        let expected_peak = wien_b / temperature;

        // Find peak by scanning
        let n = 10_000;
        let lambda_min = 0.1e-6;
        let lambda_max = 20.0e-6;
        let mut max_radiance = 0.0_f64;
        let mut peak_lambda = 0.0_f64;

        for i in 0..n {
            let lambda = lambda_min + (lambda_max - lambda_min) * i as f64 / n as f64;
            let l = planck_radiance(lambda, temperature);
            if l > max_radiance {
                max_radiance = l;
                peak_lambda = lambda;
            }
        }

        let ratio = peak_lambda * temperature;
        let error = (ratio - wien_b).abs() / wien_b;
        assert!(
            error < 0.01,
            "Wien's law: λ_max·T = {ratio:.6e}, expected ~{wien_b:.6e}, error = {error:.4}",
        );

        // Also check absolute peak is close to expected
        let peak_error = (peak_lambda - expected_peak).abs() / expected_peak;
        assert!(
            peak_error < 0.01,
            "Peak wavelength {peak_lambda:.4e} differs from expected {expected_peak:.4e}"
        );
    }

    #[test]
    fn band_radiance_increases_with_temperature() {
        let temps = [200.0, 300.0, 500.0, 1000.0, 2000.0];
        let n_steps = 200;

        for band in &[(MWIR_MIN, MWIR_MAX), (LWIR_MIN, LWIR_MAX)] {
            let radiances: Vec<f64> = temps
                .iter()
                .map(|&t| band_radiance(band.0, band.1, t, n_steps))
                .collect();

            for i in 1..radiances.len() {
                assert!(
                    radiances[i] > radiances[i - 1],
                    "Band radiance should increase with temperature: L({})={} <= L({})={}",
                    temps[i],
                    radiances[i],
                    temps[i - 1],
                    radiances[i - 1],
                );
            }
        }
    }

    // --- 6.10: P_d vs range ---

    #[test]
    fn pd_decreases_with_range() {
        let target = IrSignature::fighter_afterburner();
        let sensor = IrSensorConfig::mwir_search();
        let bg_temp = 280.0;

        let ranges = [1_000.0, 5_000.0, 20_000.0, 50_000.0, 100_000.0, 300_000.0];
        let pds: Vec<f64> = ranges
            .iter()
            .map(|&r| ir_detection_probability(&target, &sensor, r, bg_temp))
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
    fn pd_approaches_zero_beyond_horizon() {
        let target = IrSignature::fighter_military();
        let sensor = IrSensorConfig::mwir_search();
        let bg_temp = 280.0;

        let pd = ir_detection_probability(&target, &sensor, 500_000.0, bg_temp);
        assert!(pd < 0.1, "P_d at extreme range should approach 0, got {pd}");
    }

    #[test]
    fn pd_near_one_at_close_range() {
        let target = IrSignature::fighter_afterburner();
        let sensor = IrSensorConfig::mwir_search();
        let bg_temp = 280.0;

        let pd = ir_detection_probability(&target, &sensor, 500.0, bg_temp);
        assert!(pd > 0.95, "P_d at close range should be ~1, got {pd}");
    }

    // --- 6.11: MWIR vs LWIR for hot targets ---

    #[test]
    fn mwir_detects_afterburner_further_than_lwir() {
        let target = IrSignature::fighter_afterburner();
        let bg_temp = 280.0;

        let mwir_sensor = IrSensorConfig::mwir_search();
        let mut lwir_sensor = IrSensorConfig::lwir_tracking();
        // Give LWIR the same FOV / aperture so the comparison is about band physics
        lwir_sensor.aperture_diameter_m = mwir_sensor.aperture_diameter_m;
        lwir_sensor.ifov_rad = mwir_sensor.ifov_rad;
        lwir_sensor.netd_k = mwir_sensor.netd_k;
        lwir_sensor.fov_half_angle_rad = mwir_sensor.fov_half_angle_rad;

        // At a moderate range the MWIR sensor should have higher P_d for hot (1800K) targets
        let test_range = 80_000.0;
        let pd_mwir = ir_detection_probability(&target, &mwir_sensor, test_range, bg_temp);
        let pd_lwir = ir_detection_probability(&target, &lwir_sensor, test_range, bg_temp);

        assert!(
            pd_mwir > pd_lwir,
            "MWIR should outperform LWIR for hot targets: P_d_mwir={pd_mwir} vs P_d_lwir={pd_lwir}"
        );
    }

    #[test]
    fn atmospheric_transmission_decreases_with_range() {
        for band in &[
            SpectralBand::Mwir,
            SpectralBand::Lwir,
            SpectralBand::Visible,
        ] {
            let ranges = [0.0, 1_000.0, 10_000.0, 50_000.0, 100_000.0];
            let taus: Vec<f64> = ranges
                .iter()
                .map(|&r| atmospheric_transmission(r, *band))
                .collect();

            assert!(
                (taus[0] - 1.0).abs() < 1e-12,
                "Transmission at zero range should be 1.0"
            );
            for i in 1..taus.len() {
                assert!(
                    taus[i] < taus[i - 1],
                    "Transmission should decrease with range for {:?}",
                    band,
                );
            }
        }
    }

    #[test]
    fn preset_sensor_configs_valid() {
        let mwir = IrSensorConfig::mwir_search();
        assert_eq!(mwir.spectral_band, SpectralBand::Mwir);
        assert!(mwir.netd_k > 0.0);
        assert!(mwir.ifov_rad > 0.0);
        assert!(mwir.aperture_diameter_m > 0.0);
        assert!(mwir.fov_half_angle_rad > 0.0);

        let lwir = IrSensorConfig::lwir_tracking();
        assert_eq!(lwir.spectral_band, SpectralBand::Lwir);
        assert!(lwir.fov_half_angle_rad < mwir.fov_half_angle_rad);

        let vis = IrSensorConfig::visible_camera();
        assert_eq!(vis.spectral_band, SpectralBand::Visible);
    }

    #[test]
    fn preset_signatures_valid() {
        let afterburner = IrSignature::fighter_afterburner();
        let military = IrSignature::fighter_military();
        let airliner = IrSignature::airliner();
        let uav = IrSignature::uav_electric();
        let reentry = IrSignature::ballistic_reentry();

        // Afterburner exhaust hotter than military
        assert!(afterburner.exhaust_temperature_k > military.exhaust_temperature_k);
        // UAV has no exhaust area
        assert_eq!(uav.exhaust_area_m2, 0.0);
        // Reentry has very hot skin
        assert!(reentry.skin_temperature_k > 1500.0);
        // Airliner has largest skin area
        assert!(airliner.skin_area_m2 > afterburner.skin_area_m2);
    }

    #[test]
    fn generate_eoir_physics_produces_measurements() {
        let target = IrSignature::fighter_afterburner();
        let sensor = IrSensorConfig::mwir_search();
        let wp = Waypoint {
            time: 1.0,
            position: [5_000.0, 100.0, 200.0],
            velocity: [250.0, 0.0, 0.0],
        };
        let mut rng = rand::rng();

        let mut count = 0;
        for _ in 0..1000 {
            if generate_eoir_physics(&wp, &target, &sensor, 280.0, &mut rng).is_some() {
                count += 1;
            }
        }

        // At 5 km for afterburner, P_d should be very high
        assert!(
            count > 800,
            "Expected high detection rate at close range, got {count}/1000"
        );
    }

    #[test]
    fn generate_eoir_physics_outside_fov_returns_none() {
        let target = IrSignature::fighter_afterburner();
        let sensor = IrSensorConfig::mwir_search();
        // Place target far off-axis
        let wp = Waypoint {
            time: 1.0,
            position: [100.0, 10_000.0, 0.0],
            velocity: [0.0; 3],
        };
        let mut rng = rand::rng();

        for _ in 0..100 {
            assert!(
                generate_eoir_physics(&wp, &target, &sensor, 280.0, &mut rng).is_none(),
                "Should not detect target outside FOV"
            );
        }
    }
}
