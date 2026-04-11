//! nuScenes dataset ingestion via a PyO3 bridge to the `nuscenes-devkit`.
//!
//! All functionality here is gated behind the `nuscenes` Cargo feature which
//! pulls in `pyo3`. When the feature is disabled, this module is absent and
//! downstream crates that do not need nuScenes ingestion can depend on
//! `thresh-data` without requiring a Python installation.
//!
//! The bridge follows the same pattern as `thresh-bridge`: every Python
//! interaction is wrapped in `Python::with_gil` and returns `PyResult<T>`.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use thresh_core::detection::BoundingBox3D;
use thresh_core::track::TargetClass;

use crate::dataset::{CoordinateFrame, Dataset, DatasetMetadata};
use crate::frame::{Frame, GroundTruthEntry, SensorInfo};

/// Handle to a loaded nuScenes dataset via the Python devkit.
pub struct NuScenesBridge {
    /// Python-side `NuScenes` object.
    nusc: Py<PyAny>,
    /// Version string (for example, `"v1.0-mini"`).
    pub version: String,
    /// Root directory of the dataset on disk.
    pub dataroot: String,
}

impl NuScenesBridge {
    /// Load a nuScenes dataset via the devkit.
    pub fn new(version: &str, dataroot: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let nuscenes_mod = py.import("nuscenes.nuscenes")?;
            let nusc_class = nuscenes_mod.getattr("NuScenes")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("version", version)?;
            kwargs.set_item("dataroot", dataroot)?;
            kwargs.set_item("verbose", false)?;
            let nusc = nusc_class.call((), Some(&kwargs))?.unbind();
            Ok(Self {
                nusc,
                version: version.to_string(),
                dataroot: dataroot.to_string(),
            })
        })
    }

    /// Get the number of scenes in the loaded split.
    pub fn scene_count(&self) -> PyResult<usize> {
        Python::with_gil(|py| {
            let scene = self.nusc.bind(py).getattr("scene")?;
            let list: &Bound<'_, PyList> = scene.downcast()?;
            Ok(list.len())
        })
    }

    /// Return all scene tokens in iteration order.
    pub fn scene_tokens(&self) -> PyResult<Vec<String>> {
        Python::with_gil(|py| {
            let scene = self.nusc.bind(py).getattr("scene")?;
            let list: &Bound<'_, PyList> = scene.downcast()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                let token: String = item.get_item("token")?.extract()?;
                out.push(token);
            }
            Ok(out)
        })
    }

    /// Look up a scene record by token.
    pub fn get_scene(&self, scene_token: &str) -> PyResult<NuScenesScene> {
        Python::with_gil(|py| {
            let rec = self
                .nusc
                .bind(py)
                .call_method1("get", ("scene", scene_token))?;
            Ok(NuScenesScene {
                token: rec.get_item("token")?.extract()?,
                name: rec.get_item("name")?.extract()?,
                nbr_samples: rec.get_item("nbr_samples")?.extract()?,
            })
        })
    }

    /// Walk the linked list of samples belonging to a scene.
    pub fn iter_samples(&self, scene_token: &str) -> PyResult<Vec<NuScenesSample>> {
        Python::with_gil(|py| {
            let nusc = self.nusc.bind(py);
            let scene = nusc.call_method1("get", ("scene", scene_token))?;
            let first_token: String = scene.get_item("first_sample_token")?.extract()?;
            let mut out = Vec::new();
            let mut token = first_token;
            while !token.is_empty() {
                let sample = nusc.call_method1("get", ("sample", token.as_str()))?;
                let prev: String = sample.get_item("prev")?.extract()?;
                let next: String = sample.get_item("next")?.extract()?;
                let timestamp_us: i64 = sample.get_item("timestamp")?.extract()?;
                let sample_token: String = sample.get_item("token")?.extract()?;
                out.push(NuScenesSample {
                    token: sample_token,
                    timestamp_us,
                    prev: if prev.is_empty() { None } else { Some(prev) },
                    next: if next.is_empty() {
                        None
                    } else {
                        Some(next.clone())
                    },
                });
                token = next;
            }
            Ok(out)
        })
    }

    /// Return all 3D annotation bounding boxes for a sample.
    pub fn sample_annotations(&self, sample_token: &str) -> PyResult<Vec<BoundingBox3D>> {
        Python::with_gil(|py| {
            let nusc = self.nusc.bind(py);
            let sample = nusc.call_method1("get", ("sample", sample_token))?;
            let anns = sample.get_item("anns")?;
            let anns_list: &Bound<'_, PyList> = anns.downcast()?;
            let mut out = Vec::with_capacity(anns_list.len());
            for tok in anns_list.iter() {
                let token: String = tok.extract()?;
                let ann = nusc.call_method1("get", ("sample_annotation", token.as_str()))?;
                out.push(annotation_to_box(&ann)?);
            }
            Ok(out)
        })
    }

    /// Collect per-instance tracks across every sample in a scene.
    ///
    /// Each returned [`InstanceTrack`] contains every annotation of a given
    /// instance across the scene, with a stable `target_id` derived from the
    /// instance token.
    pub fn scene_instance_tracks(&self, scene_token: &str) -> PyResult<Vec<InstanceTrack>> {
        use std::collections::HashMap;
        Python::with_gil(|py| {
            let nusc = self.nusc.bind(py);
            let samples = self.iter_samples(scene_token)?;
            let mut tracks: HashMap<String, InstanceTrack> = HashMap::new();
            for sample in &samples {
                let sample_rec = nusc.call_method1("get", ("sample", sample.token.as_str()))?;
                let anns = sample_rec.get_item("anns")?;
                let anns_list: &Bound<'_, PyList> = anns.downcast()?;
                for tok in anns_list.iter() {
                    let ann_token: String = tok.extract()?;
                    let ann =
                        nusc.call_method1("get", ("sample_annotation", ann_token.as_str()))?;
                    let instance_token: String = ann.get_item("instance_token")?.extract()?;
                    let bbox = annotation_to_box(&ann)?;
                    let entry =
                        tracks
                            .entry(instance_token.clone())
                            .or_insert_with(|| InstanceTrack {
                                target_id: hash_instance_token(&instance_token),
                                instance_token: instance_token.clone(),
                                samples: Vec::new(),
                            });
                    entry.samples.push((sample.token.clone(), bbox));
                }
            }
            let mut out: Vec<InstanceTrack> = tracks.into_values().collect();
            out.sort_by(|a, b| a.instance_token.cmp(&b.instance_token));
            Ok(out)
        })
    }

    /// Load the LiDAR point cloud associated with a sample.
    ///
    /// nuScenes stores LiDAR sweeps as a raw binary file with a packed
    /// `float32 [x, y, z, intensity, ring]` layout per point (20 bytes).
    pub fn load_lidar(&self, sample_token: &str) -> PyResult<Vec<LidarPoint>> {
        let rel = self.sample_data_filename(sample_token, "LIDAR_TOP")?;
        let path = PathBuf::from(&self.dataroot).join(rel);
        read_lidar_bin(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to read LiDAR bin {}: {e}",
                path.display()
            ))
        })
    }

    /// Load a radar sweep for a sample.
    ///
    /// Uses `nuscenes.utils.data_classes.RadarPointCloud.from_file` to decode
    /// the PCD file and then extracts the relevant channels (x, y, z, rcs,
    /// vx_comp, vy_comp).
    pub fn load_radar(&self, sample_token: &str) -> PyResult<Vec<RadarPoint>> {
        self.load_radar_channel(sample_token, "RADAR_FRONT")
    }

    /// Load radar data from a specific sensor channel (e.g. `RADAR_FRONT_LEFT`).
    pub fn load_radar_channel(
        &self,
        sample_token: &str,
        channel: &str,
    ) -> PyResult<Vec<RadarPoint>> {
        let rel = self.sample_data_filename(sample_token, channel)?;
        let path = PathBuf::from(&self.dataroot).join(rel);
        Python::with_gil(|py| {
            let module = py.import("nuscenes.utils.data_classes")?;
            let cls = module.getattr("RadarPointCloud")?;
            let cloud = cls.call_method1("from_file", (path.to_string_lossy().as_ref(),))?;
            let points = cloud.getattr("points")?; // numpy array, shape (18, N)
            let shape: (usize, usize) = points.getattr("shape")?.extract()?;
            let (_rows, cols) = shape;
            let tolist = points.call_method0("tolist")?;
            let rows: Vec<Vec<f64>> = tolist.extract()?;
            // nuScenes RadarPointCloud field order:
            // 0 x, 1 y, 2 z, 3 dyn_prop, 4 id, 5 rcs, 6 vx, 7 vy, 8 vx_comp, 9 vy_comp, ...
            let mut out = Vec::with_capacity(cols);
            let x_row = rows.first().map(|r| r.as_slice()).unwrap_or(&[]);
            for (i, &x) in x_row.iter().enumerate() {
                out.push(RadarPoint {
                    x: x as f32,
                    y: rows[1][i] as f32,
                    z: rows[2][i] as f32,
                    rcs_dbsm: rows[5][i] as f32,
                    vx: rows[8][i] as f32,
                    vy: rows[9][i] as f32,
                });
            }
            Ok(out)
        })
    }

    /// Load extrinsic (and optionally intrinsic) calibration for the sensor
    /// channel used in a sample.
    pub fn get_calibration(
        &self,
        sample_token: &str,
        sensor_channel: &str,
    ) -> PyResult<SensorCalibration> {
        Python::with_gil(|py| {
            let nusc = self.nusc.bind(py);
            let sample = nusc.call_method1("get", ("sample", sample_token))?;
            let data_map = sample.get_item("data")?;
            let sd_token: String = data_map.get_item(sensor_channel)?.extract()?;
            let sample_data = nusc.call_method1("get", ("sample_data", sd_token.as_str()))?;
            let cs_token: String = sample_data.get_item("calibrated_sensor_token")?.extract()?;
            let cs = nusc.call_method1("get", ("calibrated_sensor", cs_token.as_str()))?;
            let translation: Vec<f64> = cs.get_item("translation")?.extract()?;
            let rotation: Vec<f64> = cs.get_item("rotation")?.extract()?;
            let camera_intrinsic: Option<Vec<Vec<f64>>> =
                cs.get_item("camera_intrinsic")?.extract().ok();
            let intr = camera_intrinsic.and_then(|m| {
                if m.len() == 3 && m.iter().all(|r| r.len() == 3) {
                    Some([
                        [m[0][0], m[0][1], m[0][2]],
                        [m[1][0], m[1][1], m[1][2]],
                        [m[2][0], m[2][1], m[2][2]],
                    ])
                } else {
                    None
                }
            });
            Ok(SensorCalibration {
                translation: [translation[0], translation[1], translation[2]],
                rotation_quat: [rotation[0], rotation[1], rotation[2], rotation[3]],
                camera_intrinsic: intr,
            })
        })
    }

    /// Resolve the on-disk filename for a given sample/channel combo.
    fn sample_data_filename(&self, sample_token: &str, channel: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            let nusc = self.nusc.bind(py);
            let sample = nusc.call_method1("get", ("sample", sample_token))?;
            let data_map = sample.get_item("data")?;
            let sd_token: String = data_map.get_item(channel)?.extract()?;
            let sample_data = nusc.call_method1("get", ("sample_data", sd_token.as_str()))?;
            let filename: String = sample_data.get_item("filename")?.extract()?;
            Ok(filename)
        })
    }
}

/// A scene record returned by [`NuScenesBridge::get_scene`].
#[derive(Debug, Clone)]
pub struct NuScenesScene {
    /// Scene token (stable unique identifier).
    pub token: String,
    /// Human-readable name assigned by the devkit.
    pub name: String,
    /// Number of keyframe samples in the scene.
    pub nbr_samples: usize,
}

/// A sample (keyframe) inside a nuScenes scene.
#[derive(Debug, Clone)]
pub struct NuScenesSample {
    /// Sample token.
    pub token: String,
    /// Timestamp in microseconds since epoch.
    pub timestamp_us: i64,
    /// Token of the previous sample in the scene, if any.
    pub prev: Option<String>,
    /// Token of the next sample in the scene, if any.
    pub next: Option<String>,
}

/// An instance-level track: all annotations of one physical object across a
/// scene, tagged with a stable `target_id`.
#[derive(Debug, Clone)]
pub struct InstanceTrack {
    /// nuScenes instance token.
    pub instance_token: String,
    /// Stable integer ID derived from the instance token.
    pub target_id: u64,
    /// Per-sample annotation boxes for this instance.
    pub samples: Vec<(String, BoundingBox3D)>,
}

/// A single LiDAR return decoded from a nuScenes `.pcd.bin` sweep file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LidarPoint {
    /// Cartesian X in the sensor frame (metres).
    pub x: f32,
    /// Cartesian Y in the sensor frame (metres).
    pub y: f32,
    /// Cartesian Z in the sensor frame (metres).
    pub z: f32,
    /// Lidar return intensity (device-specific scale).
    pub intensity: f32,
    /// Ring index of the laser that produced this return.
    pub ring: u8,
}

/// A single radar point decoded from a nuScenes PCD file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarPoint {
    /// Cartesian X in the sensor frame (metres).
    pub x: f32,
    /// Cartesian Y in the sensor frame (metres).
    pub y: f32,
    /// Cartesian Z in the sensor frame (metres).
    pub z: f32,
    /// Radar cross-section in dBsm.
    pub rcs_dbsm: f32,
    /// Compensated velocity X (metres per second).
    pub vx: f32,
    /// Compensated velocity Y (metres per second).
    pub vy: f32,
}

/// Sensor calibration parameters (extrinsics plus optional camera intrinsics).
#[derive(Debug, Clone)]
pub struct SensorCalibration {
    /// Translation from ego frame to sensor frame, in metres.
    pub translation: [f64; 3],
    /// Rotation from ego frame to sensor frame as `(w, x, y, z)` quaternion.
    pub rotation_quat: [f64; 4],
    /// 3x3 camera intrinsic matrix (row-major), only populated for cameras.
    pub camera_intrinsic: Option<[[f64; 3]; 3]>,
}

/// Map a nuScenes category name to a thresh [`TargetClass`].
///
/// nuScenes is an autonomous-driving dataset, so its categories (car, truck,
/// pedestrian, etc.) do not correspond exactly to the aerospace-focused
/// classes thresh tracks. We perform a best-effort mapping that groups
/// wheeled road vehicles into [`TargetClass::Aircraft`] (the closest
/// rigid-body analog), small agents into [`TargetClass::Uav`], and leave
/// everything else as [`TargetClass::Unknown`].
pub fn map_category(nuscenes_category: &str) -> TargetClass {
    if nuscenes_category.starts_with("vehicle.car")
        || nuscenes_category.starts_with("vehicle.truck")
        || nuscenes_category.starts_with("vehicle.bus")
        || nuscenes_category.starts_with("vehicle.trailer")
        || nuscenes_category.starts_with("vehicle.construction")
        || nuscenes_category.starts_with("vehicle.emergency")
    {
        TargetClass::Aircraft
    } else if nuscenes_category.starts_with("vehicle.motorcycle")
        || nuscenes_category.starts_with("vehicle.bicycle")
        || nuscenes_category.starts_with("human.pedestrian")
    {
        TargetClass::Uav
    } else {
        TargetClass::Unknown
    }
}

/// A dataset adapter that exposes a single nuScenes scene through the
/// [`Dataset`] trait.
pub struct NuScenesDataset {
    #[allow(dead_code)]
    bridge: NuScenesBridge,
    scene_token: String,
    scene_name: String,
    cached_frames: Vec<Frame>,
}

impl NuScenesDataset {
    /// Load a nuScenes scene and eagerly materialise its frames.
    pub fn load(version: &str, dataroot: &str, scene_token: &str) -> PyResult<Self> {
        let bridge = NuScenesBridge::new(version, dataroot)?;
        let scene = bridge.get_scene(scene_token)?;
        let samples = bridge.iter_samples(scene_token)?;
        let tracks = bridge.scene_instance_tracks(scene_token)?;

        // Index instance tracks by sample for quick GT lookup.
        use std::collections::HashMap;
        let mut by_sample: HashMap<String, Vec<GroundTruthEntry>> = HashMap::new();
        for track in &tracks {
            for (sample_token, bbox) in &track.samples {
                by_sample
                    .entry(sample_token.clone())
                    .or_default()
                    .push(GroundTruthEntry {
                        target_id: track.target_id,
                        position: [bbox.x, bbox.y, bbox.z],
                        velocity: bbox.velocity.map(|[vx, vy]| [vx, vy, 0.0]),
                        class: Some(TargetClass::Unknown),
                    });
            }
        }

        let mut cached_frames = Vec::with_capacity(samples.len());
        for sample in &samples {
            let gt = by_sample.remove(&sample.token);
            cached_frames.push(Frame {
                timestamp: sample.timestamp_us as f64 * 1e-6,
                measurements: Vec::new(),
                ground_truth: gt,
                sensor_metadata: Some(SensorInfo {
                    sensor_id: 0,
                    sensor_type: "nuscenes".to_string(),
                }),
            });
        }

        Ok(Self {
            bridge,
            scene_token: scene.token,
            scene_name: scene.name,
            cached_frames,
        })
    }
}

impl Dataset for NuScenesDataset {
    fn metadata(&self) -> DatasetMetadata {
        let time_span = match (self.cached_frames.first(), self.cached_frames.last()) {
            (Some(f0), Some(fn_)) => Some((f0.timestamp, fn_.timestamp)),
            _ => None,
        };
        DatasetMetadata {
            name: format!("nuscenes:{}:{}", self.scene_name, self.scene_token),
            source: "nuscenes".to_string(),
            target_count: None,
            time_span,
            coordinate_frame: CoordinateFrame::Enu,
        }
    }

    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_> {
        Box::new(self.cached_frames.iter().cloned())
    }

    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>> {
        Some(Box::new(
            self.cached_frames
                .iter()
                .filter(|f| f.ground_truth.is_some())
                .cloned(),
        ))
    }
}

/// Convert a Python `sample_annotation` dict into a [`BoundingBox3D`].
fn annotation_to_box(ann: &Bound<'_, PyAny>) -> PyResult<BoundingBox3D> {
    let translation: Vec<f64> = ann.get_item("translation")?.extract()?;
    let size: Vec<f64> = ann.get_item("size")?.extract()?;
    let rotation: Vec<f64> = ann.get_item("rotation")?.extract()?;
    let category: String = ann.get_item("category_name")?.extract()?;
    // nuScenes size order is `[w, l, h]`.
    let width = size[0];
    let length = size[1];
    let height = size[2];
    // Rotation is a (w, x, y, z) quaternion; yaw is the rotation about Z.
    let (qw, qx, qy, qz) = (rotation[0], rotation[1], rotation[2], rotation[3]);
    let yaw = (2.0 * (qw * qz + qx * qy)).atan2(1.0 - 2.0 * (qy * qy + qz * qz));
    let class = map_category(&category);
    Ok(BoundingBox3D {
        x: translation[0],
        y: translation[1],
        z: translation[2],
        length,
        width,
        height,
        yaw,
        score: 1.0,
        class_id: class as u32,
        velocity: None,
    })
}

/// Hash a nuScenes instance token into a stable `u64` target identifier.
fn hash_instance_token(token: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

/// Parse a nuScenes LiDAR `.pcd.bin` file.
///
/// Each point is five packed `float32` values: `[x, y, z, intensity, ring]`.
/// The ring channel is actually stored as a float but represents an integer
/// laser index, so we round it to the nearest `u8`.
fn read_lidar_bin(path: &std::path::Path) -> std::io::Result<Vec<LidarPoint>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    parse_lidar_bytes(&buf)
}

/// Decode a LiDAR binary buffer into [`LidarPoint`]s.
pub fn parse_lidar_bytes(buf: &[u8]) -> std::io::Result<Vec<LidarPoint>> {
    const STRIDE: usize = 5 * 4;
    if !buf.len().is_multiple_of(STRIDE) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("lidar buffer length {} not multiple of 20", buf.len()),
        ));
    }
    let n = buf.len() / STRIDE;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * STRIDE;
        let f = |off: usize| {
            f32::from_le_bytes([
                buf[base + off],
                buf[base + off + 1],
                buf[base + off + 2],
                buf[base + off + 3],
            ])
        };
        out.push(LidarPoint {
            x: f(0),
            y: f(4),
            z: f(8),
            intensity: f(12),
            ring: f(16).round().clamp(0.0, 255.0) as u8,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_category_default_unknown() {
        assert_eq!(
            map_category("static_object.bicycle_rack"),
            TargetClass::Unknown
        );
        assert_eq!(map_category("movable_object.barrier"), TargetClass::Unknown);
    }

    #[test]
    fn map_category_vehicles_to_aircraft() {
        assert_eq!(map_category("vehicle.car"), TargetClass::Aircraft);
        assert_eq!(map_category("vehicle.truck"), TargetClass::Aircraft);
        assert_eq!(map_category("vehicle.bus.rigid"), TargetClass::Aircraft);
    }

    #[test]
    fn map_category_small_agents_to_uav() {
        assert_eq!(map_category("human.pedestrian.adult"), TargetClass::Uav);
        assert_eq!(map_category("vehicle.motorcycle"), TargetClass::Uav);
        assert_eq!(map_category("vehicle.bicycle"), TargetClass::Uav);
    }

    #[test]
    fn lidar_point_fields() {
        let p = LidarPoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 0.5,
            ring: 7,
        };
        assert_eq!(p.x, 1.0);
        assert_eq!(p.ring, 7);
    }

    #[test]
    fn parse_lidar_bytes_roundtrip() {
        // Build two synthetic points and verify round-trip parse.
        let mut buf = Vec::new();
        let rows: [(f32, f32, f32, f32, f32); 2] =
            [(1.0, 2.0, 3.0, 0.25, 4.0), (10.0, 20.0, 30.0, 0.9, 12.0)];
        for (x, y, z, i, r) in rows {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
            buf.extend_from_slice(&i.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
        }
        let pts = parse_lidar_bytes(&buf).expect("parse");
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].x, 1.0);
        assert_eq!(pts[0].ring, 4);
        assert_eq!(pts[1].intensity, 0.9);
        assert_eq!(pts[1].ring, 12);
    }

    #[test]
    fn parse_lidar_bytes_rejects_misaligned() {
        let buf = vec![0u8; 19];
        assert!(parse_lidar_bytes(&buf).is_err());
    }

    #[test]
    fn hash_instance_token_stable() {
        let a = hash_instance_token("abc123");
        let b = hash_instance_token("abc123");
        let c = hash_instance_token("abc124");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn radar_point_layout() {
        let p = RadarPoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            rcs_dbsm: -5.0,
            vx: 0.1,
            vy: -0.2,
        };
        assert_eq!(p.x, 1.0);
        assert_eq!(p.rcs_dbsm, -5.0);
    }
}
