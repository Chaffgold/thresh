//! Measurement types for heterogeneous sensor data.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A measurement from a sensor with its observation model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Measurement {
    /// Radar measurement in spherical coordinates.
    Radar {
        /// Range in meters.
        range: f64,
        /// Azimuth in radians.
        azimuth: f64,
        /// Elevation in radians.
        elevation: f64,
        /// Range rate (Doppler) in m/s, if available.
        range_rate: Option<f64>,
        /// Timestamp in seconds.
        time: f64,
        /// Sensor ID.
        sensor_id: u32,
    },
    /// Electro-optical / infrared (bearing-only).
    EoIr {
        /// Azimuth in radians.
        azimuth: f64,
        /// Elevation in radians.
        elevation: f64,
        /// Timestamp in seconds.
        time: f64,
        /// Sensor ID.
        sensor_id: u32,
    },
    /// ADS-B cooperative surveillance.
    AdsB {
        /// Latitude in degrees.
        lat: f64,
        /// Longitude in degrees.
        lon: f64,
        /// Altitude in meters (MSL).
        alt: f64,
        /// Velocity vector [vx, vy, vz] in m/s, if available.
        velocity: Option<[f64; 3]>,
        /// Timestamp in seconds.
        time: f64,
    },
    /// Over-the-horizon radar measurement.
    Othr {
        /// Ground range along Earth's surface (meters).
        ground_range_m: f64,
        /// Azimuth from transmitter (radians, clockwise from north).
        azimuth_rad: f64,
        /// Doppler velocity (m/s, positive = approaching).
        doppler_m_s: f64,
        /// Ionospheric propagation mode.
        propagation_mode: PropagationMode,
        /// Measurement time.
        time: f64,
        /// Sensor identifier.
        sensor_id: u32,
    },
}

/// Ionospheric propagation mode for OTHR signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationMode {
    /// Single E-layer reflection.
    ELayer,
    /// Single F-layer reflection.
    FLayer,
    /// Multiple ionospheric hops.
    MultiHop(u8),
}

impl Measurement {
    /// Return the timestamp of this measurement.
    pub fn time(&self) -> f64 {
        match self {
            Measurement::Radar { time, .. } => *time,
            Measurement::EoIr { time, .. } => *time,
            Measurement::AdsB { time, .. } => *time,
            Measurement::Othr { time, .. } => *time,
        }
    }

    /// Return the measurement as a vector z.
    pub fn to_vector(&self) -> DVector<f64> {
        match self {
            Measurement::Radar {
                range,
                azimuth,
                elevation,
                range_rate,
                ..
            } => {
                if let Some(rr) = range_rate {
                    DVector::from_column_slice(&[*range, *azimuth, *elevation, *rr])
                } else {
                    DVector::from_column_slice(&[*range, *azimuth, *elevation])
                }
            }
            Measurement::EoIr {
                azimuth, elevation, ..
            } => DVector::from_column_slice(&[*azimuth, *elevation]),
            Measurement::AdsB {
                lat,
                lon,
                alt,
                velocity,
                ..
            } => {
                if let Some(v) = velocity {
                    DVector::from_column_slice(&[*lat, *lon, *alt, v[0], v[1], v[2]])
                } else {
                    DVector::from_column_slice(&[*lat, *lon, *alt])
                }
            }
            Measurement::Othr {
                ground_range_m,
                azimuth_rad,
                doppler_m_s,
                ..
            } => DVector::from_column_slice(&[*ground_range_m, *azimuth_rad, *doppler_m_s]),
        }
    }

    /// Return the measurement dimension.
    pub fn dim(&self) -> usize {
        match self {
            Measurement::Radar { range_rate, .. } => {
                if range_rate.is_some() {
                    4
                } else {
                    3
                }
            }
            Measurement::EoIr { .. } => 2,
            Measurement::AdsB { velocity, .. } => {
                if velocity.is_some() {
                    6
                } else {
                    3
                }
            }
            Measurement::Othr { .. } => 3,
        }
    }

    /// Return the default measurement noise covariance R for this sensor type.
    pub fn default_noise(&self) -> DMatrix<f64> {
        match self {
            Measurement::Radar { range_rate, .. } => {
                if range_rate.is_some() {
                    // [range, az, el, range_rate] noise
                    DMatrix::from_diagonal(&DVector::from_column_slice(&[
                        100.0,         // range: 10m std
                        0.001 * 0.001, // azimuth: 1 mrad std
                        0.001 * 0.001, // elevation: 1 mrad std
                        1.0,           // range_rate: 1 m/s std
                    ]))
                } else {
                    DMatrix::from_diagonal(&DVector::from_column_slice(&[
                        100.0,
                        0.001 * 0.001,
                        0.001 * 0.001,
                    ]))
                }
            }
            Measurement::EoIr { .. } => {
                // Bearing-only: ~0.5 mrad std each
                DMatrix::from_diagonal(&DVector::from_column_slice(&[
                    0.0005 * 0.0005,
                    0.0005 * 0.0005,
                ]))
            }
            Measurement::AdsB { velocity, .. } => {
                if velocity.is_some() {
                    DMatrix::from_diagonal(&DVector::from_column_slice(&[
                        25.0, 25.0, 100.0, // position: 5m/5m/10m std
                        1.0, 1.0, 1.0, // velocity: 1 m/s std each
                    ]))
                } else {
                    DMatrix::from_diagonal(&DVector::from_column_slice(&[25.0, 25.0, 100.0]))
                }
            }
            Measurement::Othr { .. } => {
                // [ground_range, azimuth, doppler] noise
                // ground_range: ~10 km std, azimuth: ~1° std, doppler: ~10 m/s std
                DMatrix::from_diagonal(&DVector::from_column_slice(&[
                    10_000.0 * 10_000.0,            // ground range: 10 km std
                    (1.0_f64.to_radians()).powi(2), // azimuth: 1° std
                    100.0,                          // doppler: 10 m/s std
                ]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radar_measurement_vector() {
        let m = Measurement::Radar {
            range: 10000.0,
            azimuth: 0.5,
            elevation: 0.1,
            range_rate: None,
            time: 1.0,
            sensor_id: 0,
        };
        let z = m.to_vector();
        assert_eq!(z.len(), 3);
        assert_eq!(z[0], 10000.0);
    }

    #[test]
    fn radar_with_doppler() {
        let m = Measurement::Radar {
            range: 5000.0,
            azimuth: 1.0,
            elevation: 0.2,
            range_rate: Some(-100.0),
            time: 2.0,
            sensor_id: 1,
        };
        assert_eq!(m.dim(), 4);
        let z = m.to_vector();
        assert_eq!(z[3], -100.0);
    }

    #[test]
    fn eoir_measurement_dim() {
        let m = Measurement::EoIr {
            azimuth: 0.3,
            elevation: 0.1,
            time: 0.0,
            sensor_id: 2,
        };
        assert_eq!(m.dim(), 2);
    }

    #[test]
    fn adsb_with_velocity() {
        let m = Measurement::AdsB {
            lat: 35.0,
            lon: -120.0,
            alt: 10000.0,
            velocity: Some([100.0, 50.0, -5.0]),
            time: 0.5,
        };
        assert_eq!(m.dim(), 6);
        let z = m.to_vector();
        assert_eq!(z[3], 100.0);
    }

    #[test]
    fn othr_measurement_vector() {
        let m = Measurement::Othr {
            ground_range_m: 2_000_000.0,
            azimuth_rad: 1.0,
            doppler_m_s: -50.0,
            propagation_mode: PropagationMode::FLayer,
            time: 10.0,
            sensor_id: 5,
        };
        assert_eq!(m.dim(), 3);
        let z = m.to_vector();
        assert_eq!(z[0], 2_000_000.0);
        assert_eq!(z[1], 1.0);
        assert_eq!(z[2], -50.0);
        assert_eq!(m.time(), 10.0);
    }

    #[test]
    fn othr_serialization_roundtrip() {
        let m = Measurement::Othr {
            ground_range_m: 1_500_000.0,
            azimuth_rad: 0.785,
            doppler_m_s: 25.0,
            propagation_mode: PropagationMode::ELayer,
            time: 5.0,
            sensor_id: 3,
        };
        let json = serde_json::to_string(&m).expect("serialize");
        let m2: Measurement = serde_json::from_str(&json).expect("deserialize");
        let z1 = m.to_vector();
        let z2 = m2.to_vector();
        assert_eq!(z1, z2);
        assert_eq!(m.time(), m2.time());
    }

    #[test]
    fn propagation_mode_serialization_roundtrip() {
        for mode in &[
            PropagationMode::ELayer,
            PropagationMode::FLayer,
            PropagationMode::MultiHop(2),
            PropagationMode::MultiHop(5),
        ] {
            let json = serde_json::to_string(mode).expect("serialize");
            let mode2: PropagationMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*mode, mode2);
        }
    }

    #[test]
    fn othr_noise_dimensions_match() {
        let m = Measurement::Othr {
            ground_range_m: 1_000_000.0,
            azimuth_rad: 0.0,
            doppler_m_s: 0.0,
            propagation_mode: PropagationMode::MultiHop(3),
            time: 0.0,
            sensor_id: 0,
        };
        let r = m.default_noise();
        assert_eq!(r.nrows(), m.dim());
        assert_eq!(r.ncols(), m.dim());
    }

    #[test]
    fn noise_dimensions_match() {
        let radar = Measurement::Radar {
            range: 0.0,
            azimuth: 0.0,
            elevation: 0.0,
            range_rate: Some(0.0),
            time: 0.0,
            sensor_id: 0,
        };
        let r = radar.default_noise();
        assert_eq!(r.nrows(), radar.dim());
        assert_eq!(r.ncols(), radar.dim());
    }
}
