//! Measurement generators for radar, EO/IR, and ADS-B sensors.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use thresh_core::measurement::Measurement;

use crate::trajectory::Waypoint;

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
    /// Probability of detection [0, 1].
    pub p_detection: f64,
    /// Mean number of clutter returns per scan.
    pub clutter_rate: f64,
    /// Maximum range (meters).
    pub max_range: f64,
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
        }
    }
}

/// Generate a radar measurement from a waypoint (or None if not detected).
pub fn generate_radar<R: Rng>(
    waypoint: &Waypoint,
    config: &RadarConfig,
    rng: &mut R,
) -> Option<Measurement> {
    let pos = &waypoint.position;
    let range = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

    if range > config.max_range || range < 1e-6 {
        return None;
    }

    // Detection probability
    if rng.random::<f64>() > config.p_detection {
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
}
