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
}

impl Measurement {
    /// Return the timestamp of this measurement.
    pub fn time(&self) -> f64 {
        match self {
            Measurement::Radar { time, .. } => *time,
            Measurement::EoIr { time, .. } => *time,
            Measurement::AdsB { time, .. } => *time,
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
