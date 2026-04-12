//! Radar scene simulation bridge.
//!
//! Pure-Rust scene description types (transmitters, targets, clutter, CFAR
//! configuration) are always available. The actual PyO3 bridge to the Python
//! `radarsimpy` package is gated behind the `radar-scene` Cargo feature so
//! that this crate still builds without a Python installation.

use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;

#[cfg(feature = "radar-scene")]
use pyo3::prelude::*;

// ── Transmitter ─────────────────────────────────────────────────────────────

/// Transmitter configuration for a radar scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarTransmitter {
    /// Position in local ENU frame (metres).
    pub position: [f64; 3],
    /// Carrier frequency (GHz).
    pub frequency_ghz: f64,
    /// Pulse width (seconds).
    pub pulse_width_s: f64,
    /// Occupied bandwidth (Hz).
    pub bandwidth_hz: f64,
    /// Pulse repetition frequency (Hz).
    pub prf_hz: f64,
    /// Peak transmit power (watts).
    pub peak_power_w: f64,
    /// Antenna gain (dB).
    pub gain_db: f64,
    /// Number of coherently integrated pulses per CPI.
    ///
    /// Needs to be >1 to produce meaningful Doppler resolution in the
    /// range-Doppler processing downstream.
    #[serde(default = "default_n_pulses")]
    pub n_pulses: usize,
}

fn default_n_pulses() -> usize {
    64
}

// ── Target ──────────────────────────────────────────────────────────────────

/// A target in a radar scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneTarget {
    /// Position in local ENU frame (metres).
    pub position: [f64; 3],
    /// Velocity in local ENU frame (m/s).
    pub velocity: [f64; 3],
    /// Radar cross-section (m²).
    pub rcs_m2: f64,
}

// ── Clutter ─────────────────────────────────────────────────────────────────

/// Clutter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClutterModel {
    /// No clutter.
    None,
    /// Uniform surface clutter with reflectivity σ₀ (dBsm per m²).
    Surface {
        /// Reflectivity σ₀ in dBsm per m².
        sigma0_dbsm: f64,
    },
    /// Volume clutter (e.g., rain) with reflectivity per m³.
    Volume {
        /// Volumetric reflectivity in dBsm per m³.
        reflectivity_dbsm_m3: f64,
    },
}

impl ClutterModel {
    /// Typical sea clutter at X-band.
    pub fn sea_clutter_default() -> Self {
        ClutterModel::Surface { sigma0_dbsm: -30.0 }
    }

    /// Typical land clutter.
    pub fn land_clutter_default() -> Self {
        ClutterModel::Surface { sigma0_dbsm: -20.0 }
    }

    /// Typical moderate rain volume clutter.
    pub fn rain_clutter_default() -> Self {
        ClutterModel::Volume {
            reflectivity_dbsm_m3: -60.0,
        }
    }
}

// ── CFAR configuration ─────────────────────────────────────────────────────

/// CFAR detection algorithm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CfarAlgorithm {
    /// Cell-averaging CFAR.
    CaCfar,
    /// Ordered-statistic CFAR.
    OsCfar,
}

/// CFAR detector configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfarConfig {
    /// CFAR algorithm variant.
    pub algorithm: CfarAlgorithm,
    /// Number of training cells in the reference window.
    pub num_training_cells: usize,
    /// Number of guard cells surrounding the cell under test.
    pub num_guard_cells: usize,
    /// Probability of false alarm.
    pub pfa: f64,
}

impl Default for CfarConfig {
    fn default() -> Self {
        Self {
            algorithm: CfarAlgorithm::CaCfar,
            num_training_cells: 16,
            num_guard_cells: 2,
            pfa: 1e-6,
        }
    }
}

// ── Scene ──────────────────────────────────────────────────────────────────

/// Complete radar scene description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarScene {
    /// Transmitters (one or more for multi-radar scenes).
    pub transmitters: Vec<RadarTransmitter>,
    /// Targets observed by the scene.
    pub targets: Vec<SceneTarget>,
    /// Clutter model for the scene.
    pub clutter: ClutterModel,
    /// CFAR detector configuration.
    pub cfar: CfarConfig,
}

impl RadarScene {
    /// Append a transmitter to the scene.
    pub fn add_transmitter(&mut self, tx: RadarTransmitter) {
        self.transmitters.push(tx);
    }

    /// Append a target to the scene.
    pub fn add_target(&mut self, target: SceneTarget) {
        self.targets.push(target);
    }

    /// Build a monostatic X-band surveillance scene with one transmitter at
    /// the supplied ENU position and no targets or clutter.
    pub fn x_band_monostatic(tx_position: [f64; 3]) -> Self {
        Self {
            transmitters: vec![RadarTransmitter {
                position: tx_position,
                frequency_ghz: 10.0,
                pulse_width_s: 1e-6,
                bandwidth_hz: 1e6,
                prf_hz: 1000.0,
                peak_power_w: 1e5,
                gain_db: 34.0,
                n_pulses: default_n_pulses(),
            }],
            targets: Vec::new(),
            clutter: ClutterModel::None,
            cfar: CfarConfig::default(),
        }
    }
}

// ── Raw detection ──────────────────────────────────────────────────────────

/// Raw detection from the scene simulator (pre-tracker).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDetection {
    /// Index of the transmitter that produced this detection.
    pub transmitter_idx: usize,
    /// Range to the target (metres).
    pub range_m: f64,
    /// Azimuth (radians).
    pub azimuth_rad: f64,
    /// Elevation (radians).
    pub elevation_rad: f64,
    /// Doppler / line-of-sight range rate (m/s).
    pub doppler_m_s: f64,
    /// Signal-to-noise ratio (dB).
    pub snr_db: f64,
}

/// Convert a raw detection into a `Measurement::Radar`.
pub fn raw_to_measurement(raw: &RawDetection, time: f64, sensor_id: u32) -> Measurement {
    Measurement::Radar {
        range: raw.range_m,
        azimuth: raw.azimuth_rad,
        elevation: raw.elevation_rad,
        range_rate: Some(raw.doppler_m_s),
        time,
        sensor_id,
    }
}

// ── PyO3 bridge (feature gated) ─────────────────────────────────────────────

/// Bridge to the Python `radarsimpy` scene simulator.
#[cfg(feature = "radar-scene")]
pub struct RadarScenePyBridge;

#[cfg(feature = "radar-scene")]
impl RadarScenePyBridge {
    /// Simulate the scene via `radarsimpy` and return raw detections.
    ///
    /// This performs a one-shot simulation: it builds a `radarsimpy` radar
    /// (transmitter + receiver), a list of targets, runs the simulator,
    /// applies CFAR, and collects the resulting detections.
    pub fn simulate(scene: &RadarScene) -> PyResult<Vec<RawDetection>> {
        // Fail fast on misconfigured scenes.
        if scene.transmitters.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RadarScene::simulate requires at least one transmitter",
            ));
        }
        if !matches!(scene.clutter, ClutterModel::None) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "ClutterModel is not yet wired into RadarScenePyBridge::simulate",
            ));
        }

        Python::with_gil(|py| -> PyResult<Vec<RawDetection>> {
            // Import radarsimpy submodules lazily so that a failure here
            // surfaces as a PyErr rather than a link error.
            let rsp = py.import("radarsimpy")?;
            let simulator_mod = py.import("radarsimpy.simulator")?;
            let processing_mod = py.import("radarsimpy.processing")?;

            let transmitter_cls = rsp.getattr("Transmitter")?;
            let receiver_cls = rsp.getattr("Receiver")?;
            let radar_cls = rsp.getattr("Radar")?;
            let simulate_fn = simulator_mod.getattr("simc")?;

            let mut detections = Vec::new();

            for (tx_idx, tx) in scene.transmitters.iter().enumerate() {
                // Build transmitter. radarsimpy expects frequency in Hz.
                let f_hz = tx.frequency_ghz * 1e9;
                let tx_kwargs = pyo3::types::PyDict::new(py);
                tx_kwargs.set_item("f", f_hz)?;
                tx_kwargs.set_item("t", tx.pulse_width_s)?;
                tx_kwargs.set_item("tx_power", tx.peak_power_w)?;
                tx_kwargs.set_item("prp", 1.0 / tx.prf_hz)?;
                // Coherent processing interval length (pulses per CPI).
                // Range-Doppler needs >1 pulse to produce Doppler resolution.
                tx_kwargs.set_item("pulses", tx.n_pulses.max(1))?;
                let transmitter = transmitter_cls.call((), Some(&tx_kwargs))?;

                let rx_kwargs = pyo3::types::PyDict::new(py);
                rx_kwargs.set_item("fs", tx.bandwidth_hz)?;
                rx_kwargs.set_item("noise_figure", 3.0)?;
                rx_kwargs.set_item("rf_gain", tx.gain_db)?;
                let receiver = receiver_cls.call((), Some(&rx_kwargs))?;

                let radar_kwargs = pyo3::types::PyDict::new(py);
                radar_kwargs.set_item("transmitter", transmitter)?;
                radar_kwargs.set_item("receiver", receiver)?;
                radar_kwargs.set_item("location", tx.position.to_vec())?;
                let radar = radar_cls.call((), Some(&radar_kwargs))?;

                // Build target list as Python dicts.
                let targets_list = pyo3::types::PyList::empty(py);
                for tgt in &scene.targets {
                    let d = pyo3::types::PyDict::new(py);
                    d.set_item("location", tgt.position.to_vec())?;
                    d.set_item("speed", tgt.velocity.to_vec())?;
                    d.set_item("rcs", tgt.rcs_m2)?;
                    d.set_item("phase", 0.0)?;
                    targets_list.append(d)?;
                }

                // Run simulation → baseband data.
                let sim_kwargs = pyo3::types::PyDict::new(py);
                sim_kwargs.set_item("radar", &radar)?;
                sim_kwargs.set_item("targets", &targets_list)?;
                let sim_result = simulate_fn.call((), Some(&sim_kwargs))?;
                let baseband = sim_result.get_item("baseband")?;

                // Range-Doppler processing.
                let rdm = processing_mod.call_method1("range_doppler_fft", (baseband,))?;

                // CFAR detection.
                let cfar_name = match scene.cfar.algorithm {
                    CfarAlgorithm::CaCfar => "cfar_ca_2d",
                    CfarAlgorithm::OsCfar => "cfar_os_2d",
                };
                let cfar_kwargs = pyo3::types::PyDict::new(py);
                cfar_kwargs.set_item("guard", scene.cfar.num_guard_cells)?;
                cfar_kwargs.set_item("trailing", scene.cfar.num_training_cells)?;
                cfar_kwargs.set_item("pfa", scene.cfar.pfa)?;
                let cfar_peaks =
                    processing_mod.call_method(cfar_name, (rdm,), Some(&cfar_kwargs))?;

                // Each peak is expected to expose (range, azimuth, elevation,
                // doppler, snr_db). We tolerate any iterable of these tuples.
                let iter = cfar_peaks.try_iter()?;
                for item in iter {
                    let item = item?;
                    let range_m: f64 = item.get_item(0)?.extract()?;
                    let azimuth_rad: f64 = item.get_item(1)?.extract()?;
                    let elevation_rad: f64 = item.get_item(2)?.extract()?;
                    let doppler_m_s: f64 = item.get_item(3)?.extract()?;
                    let snr_db: f64 = item.get_item(4)?.extract()?;
                    detections.push(RawDetection {
                        transmitter_idx: tx_idx,
                        range_m,
                        azimuth_rad,
                        elevation_rad,
                        doppler_m_s,
                        snr_db,
                    });
                }
            }

            Ok(detections)
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tx() -> RadarTransmitter {
        RadarTransmitter {
            position: [0.0, 0.0, 0.0],
            frequency_ghz: 10.0,
            pulse_width_s: 1e-6,
            bandwidth_hz: 1e6,
            prf_hz: 1000.0,
            peak_power_w: 1e5,
            gain_db: 34.0,
            n_pulses: 64,
        }
    }

    #[test]
    fn single_target_scene_layout() {
        let mut scene = RadarScene {
            transmitters: vec![sample_tx()],
            targets: vec![SceneTarget {
                position: [10000.0, 0.0, 1000.0],
                velocity: [0.0, 0.0, 0.0],
                rcs_m2: 1.0,
            }],
            clutter: ClutterModel::None,
            cfar: CfarConfig::default(),
        };
        assert_eq!(scene.transmitters.len(), 1);
        scene.add_transmitter(RadarTransmitter {
            position: [5000.0, 0.0, 0.0],
            ..sample_tx()
        });
        assert_eq!(scene.transmitters.len(), 2);
    }

    #[test]
    fn raw_detection_to_measurement_preserves_fields() {
        let raw = RawDetection {
            transmitter_idx: 0,
            range_m: 5000.0,
            azimuth_rad: 0.5,
            elevation_rad: 0.1,
            doppler_m_s: 100.0,
            snr_db: 20.0,
        };
        let m = raw_to_measurement(&raw, 1.0, 42);
        match m {
            Measurement::Radar {
                range,
                azimuth,
                elevation,
                range_rate,
                time,
                sensor_id,
            } => {
                assert_eq!(range, 5000.0);
                assert_eq!(azimuth, 0.5);
                assert_eq!(elevation, 0.1);
                assert_eq!(range_rate, Some(100.0));
                assert_eq!(time, 1.0);
                assert_eq!(sensor_id, 42);
            }
            _ => panic!("expected Radar"),
        }
    }

    #[test]
    fn clutter_defaults_are_reasonable() {
        let sea = ClutterModel::sea_clutter_default();
        if let ClutterModel::Surface { sigma0_dbsm } = sea {
            assert!(sigma0_dbsm < 0.0);
        } else {
            panic!("expected Surface");
        }

        let land = ClutterModel::land_clutter_default();
        if let ClutterModel::Surface { sigma0_dbsm } = land {
            assert!(sigma0_dbsm < 0.0);
        } else {
            panic!("expected Surface");
        }

        let rain = ClutterModel::rain_clutter_default();
        if let ClutterModel::Volume {
            reflectivity_dbsm_m3,
        } = rain
        {
            assert!(reflectivity_dbsm_m3 < 0.0);
        } else {
            panic!("expected Volume");
        }
    }

    #[test]
    fn cfar_config_default_values() {
        let c = CfarConfig::default();
        assert!(matches!(c.algorithm, CfarAlgorithm::CaCfar));
        assert_eq!(c.num_training_cells, 16);
        assert_eq!(c.num_guard_cells, 2);
        assert!((c.pfa - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn scene_json_roundtrip() {
        let scene = RadarScene {
            transmitters: vec![sample_tx()],
            targets: vec![SceneTarget {
                position: [1000.0, 2000.0, 100.0],
                velocity: [10.0, 0.0, 0.0],
                rcs_m2: 5.0,
            }],
            clutter: ClutterModel::sea_clutter_default(),
            cfar: CfarConfig::default(),
        };
        let json = serde_json::to_string(&scene).unwrap();
        let parsed: RadarScene = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.transmitters.len(), 1);
        assert_eq!(parsed.targets.len(), 1);
    }

    #[test]
    fn x_band_monostatic_has_one_transmitter() {
        let scene = RadarScene::x_band_monostatic([100.0, 200.0, 0.0]);
        assert_eq!(scene.transmitters.len(), 1);
        assert!(scene.targets.is_empty());
        assert!(matches!(scene.clutter, ClutterModel::None));
        assert_eq!(scene.transmitters[0].position, [100.0, 200.0, 0.0]);
        assert!((scene.transmitters[0].frequency_ghz - 10.0).abs() < 1e-12);
    }

    #[test]
    fn add_target_appends_to_scene() {
        let mut scene = RadarScene::x_band_monostatic([0.0, 0.0, 0.0]);
        assert!(scene.targets.is_empty());
        scene.add_target(SceneTarget {
            position: [1000.0, 0.0, 100.0],
            velocity: [50.0, 0.0, 0.0],
            rcs_m2: 2.0,
        });
        assert_eq!(scene.targets.len(), 1);
        assert_eq!(scene.targets[0].position, [1000.0, 0.0, 100.0]);
        assert!((scene.targets[0].rcs_m2 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn default_n_pulses_is_nonzero() {
        // Ensures the serde default for n_pulses is compatible with
        // range-Doppler processing (needs > 1 pulse).
        assert!(default_n_pulses() > 1);
    }

    #[test]
    fn radar_transmitter_default_n_pulses_via_serde() {
        // A JSON payload that omits n_pulses should deserialise with the
        // default value from serde(default).
        let json = r#"{
            "position": [0.0, 0.0, 0.0],
            "frequency_ghz": 10.0,
            "pulse_width_s": 1e-6,
            "bandwidth_hz": 1e6,
            "prf_hz": 1000.0,
            "peak_power_w": 100000.0,
            "gain_db": 34.0
        }"#;
        let tx: RadarTransmitter = serde_json::from_str(json).unwrap();
        assert!(tx.n_pulses > 1);
    }

    #[test]
    fn cfar_algorithm_os_variant() {
        let config = CfarConfig {
            algorithm: CfarAlgorithm::OsCfar,
            num_training_cells: 32,
            num_guard_cells: 4,
            pfa: 1e-4,
        };
        assert!(matches!(config.algorithm, CfarAlgorithm::OsCfar));
        assert_eq!(config.num_training_cells, 32);
    }

    #[test]
    fn cfar_algorithm_serde_roundtrip() {
        let ca = CfarAlgorithm::CaCfar;
        let os = CfarAlgorithm::OsCfar;
        let ca_json = serde_json::to_string(&ca).unwrap();
        let os_json = serde_json::to_string(&os).unwrap();
        let ca_back: CfarAlgorithm = serde_json::from_str(&ca_json).unwrap();
        let os_back: CfarAlgorithm = serde_json::from_str(&os_json).unwrap();
        assert!(matches!(ca_back, CfarAlgorithm::CaCfar));
        assert!(matches!(os_back, CfarAlgorithm::OsCfar));
    }

    #[test]
    fn clutter_model_none_serde_roundtrip() {
        let none = ClutterModel::None;
        let json = serde_json::to_string(&none).unwrap();
        let back: ClutterModel = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ClutterModel::None));
    }

    #[test]
    fn ca_cfar_threshold_below_noise_floor() {
        // A simple CA-CFAR evaluation: given a cell under test and a window of
        // training cells, compute the threshold as alpha * mean(training). A
        // target whose signal is below threshold should not be detected.
        fn ca_cfar_threshold(training: &[f64], pfa: f64) -> f64 {
            let n = training.len() as f64;
            let mean: f64 = training.iter().sum::<f64>() / n;
            let alpha = n * (pfa.powf(-1.0 / n) - 1.0);
            alpha * mean
        }

        let training = vec![1.0_f64; 16];
        let threshold = ca_cfar_threshold(&training, 1e-6);
        assert!(threshold > 1.0, "threshold should exceed mean noise");

        let weak_target = 1.5_f64;
        assert!(
            weak_target < threshold,
            "weak target should be below threshold"
        );

        let strong_target = threshold * 2.0;
        assert!(
            strong_target > threshold,
            "strong target should exceed threshold"
        );
    }
}
