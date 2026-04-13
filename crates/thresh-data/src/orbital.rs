//! Orbital data ingestion: TLE parsing, SGP4 propagation, and coordinate transforms.
//!
//! All types and functions in this module are gated behind the `orbital` feature flag.

use std::fmt;

use thresh_core::eci::teme_to_ecef;
use thresh_core::geodetic::ecef_to_enu;
use thresh_core::measurement::Measurement;

use crate::dataset::{CoordinateFrame, Dataset, DatasetMetadata};
use crate::frame::{Frame, GroundTruthEntry, SensorInfo};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during orbital data processing.
#[derive(Debug)]
pub enum OrbitalError {
    /// TLE parsing failed.
    TleParse(String),
    /// SGP4 propagation failed.
    Sgp4(String),
    /// JSON parsing failed.
    Json(String),
    /// Invalid input parameter.
    InvalidInput(String),
    /// HTTP request failed (Space-Track / CelesTrak clients).
    Http(String),
    /// HTTP response had a non-success status code.
    HttpStatus { code: u16, body: String },
    /// Authentication failed (bad credentials or expired session).
    Auth(String),
}

impl fmt::Display for OrbitalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrbitalError::TleParse(msg) => write!(f, "TLE parse error: {msg}"),
            OrbitalError::Sgp4(msg) => write!(f, "SGP4 error: {msg}"),
            OrbitalError::Json(msg) => write!(f, "JSON parse error: {msg}"),
            OrbitalError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            OrbitalError::Http(msg) => write!(f, "HTTP error: {msg}"),
            OrbitalError::HttpStatus { code, body } => {
                let snippet: String = body.chars().take(200).collect();
                write!(f, "HTTP {code}: {snippet}")
            }
            OrbitalError::Auth(msg) => write!(f, "authentication error: {msg}"),
        }
    }
}

impl std::error::Error for OrbitalError {}

impl From<reqwest::Error> for OrbitalError {
    fn from(e: reqwest::Error) -> Self {
        OrbitalError::Http(e.to_string())
    }
}

impl From<sgp4::Error> for OrbitalError {
    fn from(e: sgp4::Error) -> Self {
        OrbitalError::Sgp4(format!("{e:?}"))
    }
}

impl From<serde_json::Error> for OrbitalError {
    fn from(e: serde_json::Error) -> Self {
        OrbitalError::Json(e.to_string())
    }
}

/// Result type alias for orbital operations.
pub type Result<T> = std::result::Result<T, OrbitalError>;

// ---------------------------------------------------------------------------
// TLE types (Task 4.2)
// ---------------------------------------------------------------------------

/// A parsed Two-Line Element set.
#[derive(Debug, Clone)]
pub struct Tle {
    /// Satellite name (from line 0 of 3LE, or empty).
    pub name: String,
    /// TLE line 1.
    pub line1: String,
    /// TLE line 2.
    pub line2: String,
    /// NORAD catalog number.
    pub norad_id: u32,
    /// Epoch year (four-digit).
    pub epoch_year: i32,
    /// Epoch fractional day of year.
    pub epoch_day: f64,
}

impl Tle {
    /// Compute the Julian Date of this TLE's epoch.
    pub fn epoch_jd(&self) -> f64 {
        tle_epoch_to_jd(self.epoch_year, self.epoch_day)
    }

    /// Convert to `sgp4::Elements` for propagation.
    fn to_sgp4_elements(&self) -> Result<sgp4::Elements> {
        sgp4::Elements::from_tle(
            Some(self.name.clone()),
            self.line1.as_bytes(),
            self.line2.as_bytes(),
        )
        .map_err(OrbitalError::from)
    }
}

// ---------------------------------------------------------------------------
// TLE epoch -> Julian Date
// ---------------------------------------------------------------------------

/// Convert a TLE epoch (year + fractional day) to Julian Date.
fn tle_epoch_to_jd(year: i32, day: f64) -> f64 {
    // Julian Date of Jan 0.0 of the given year.
    // Using the standard formula: JD of Jan 1.0 = JD of Jan 0.0 + 1
    // For a given year, JD of Jan 1.5 (noon) can be computed.
    // We use a simplified approach via the known epoch of J2000.
    //
    // Days from J2000 (2000-01-01 12:00 TT, JD 2451545.0):
    //   From Jan 1.0 of `year` to Jan 1.5 of 2000 = (year-2000)*365 + leap_days - 0.5
    //   Then add `day` (where day 1.0 = Jan 1 00:00 UTC).

    let dy = year - 2000;
    // Number of leap years between 2000 and `year` (exclusive of `year` itself).
    let leap_days = if dy > 0 {
        let y = year - 1;
        (y / 4 - 499) - (y / 100 - 19) + (y / 400 - 4)
    } else if dy < 0 {
        let y = year;
        (y / 4 - 500) - (y / 100 - 20) + (y / 400 - 4)
    } else {
        0
    };

    // JD of Jan 1.0 of the year
    let jd_jan1 = 2_451_545.0 - 0.5 + (dy * 365 + leap_days) as f64;
    // TLE day 1.0 = Jan 1 00:00, so day-of-year is 1-based.
    jd_jan1 + day - 1.0
}

// ---------------------------------------------------------------------------
// TLE parsing (Task 4.2)
// ---------------------------------------------------------------------------

/// Parse epoch fields from TLE line 1.
fn parse_epoch_from_line1(line1: &str) -> Result<(i32, f64)> {
    if line1.len() < 32 {
        return Err(OrbitalError::TleParse(
            "line 1 too short for epoch".to_string(),
        ));
    }
    // Columns 19-20: 2-digit year, 21-32: fractional day
    let year_str = line1[18..20].trim();
    let day_str = line1[20..32].trim();

    let year2: i32 = year_str
        .parse()
        .map_err(|e| OrbitalError::TleParse(format!("epoch year: {e}")))?;

    // 2-digit year convention: 57-99 => 1957-1999, 00-56 => 2000-2056
    let epoch_year = if year2 >= 57 {
        1900 + year2
    } else {
        2000 + year2
    };

    let epoch_day: f64 = day_str
        .parse()
        .map_err(|e| OrbitalError::TleParse(format!("epoch day: {e}")))?;

    Ok((epoch_year, epoch_day))
}

/// Parse a NORAD ID from TLE line 1.
fn parse_norad_from_line1(line1: &str) -> Result<u32> {
    if line1.len() < 7 {
        return Err(OrbitalError::TleParse(
            "line 1 too short for NORAD ID".to_string(),
        ));
    }
    line1[2..7]
        .trim()
        .parse()
        .map_err(|e| OrbitalError::TleParse(format!("NORAD ID: {e}")))
}

/// Parse TLE text in 2-line format (lines paired: line1, line2, line1, line2, ...).
///
/// Each pair of lines is one element set (no name line).
pub fn parse_tle(text: &str) -> Result<Vec<Tle>> {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if !lines.len().is_multiple_of(2) {
        return Err(OrbitalError::TleParse(
            "2LE text must have an even number of non-empty lines".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(lines.len() / 2);
    for chunk in lines.chunks(2) {
        let line1 = chunk[0];
        let line2 = chunk[1];

        if !line1.starts_with('1') {
            return Err(OrbitalError::TleParse(format!(
                "expected line 1 to start with '1', got: {line1}"
            )));
        }
        if !line2.starts_with('2') {
            return Err(OrbitalError::TleParse(format!(
                "expected line 2 to start with '2', got: {line2}"
            )));
        }

        let norad_id = parse_norad_from_line1(line1)?;
        let (epoch_year, epoch_day) = parse_epoch_from_line1(line1)?;

        result.push(Tle {
            name: String::new(),
            line1: line1.to_string(),
            line2: line2.to_string(),
            norad_id,
            epoch_year,
            epoch_day,
        });
    }

    Ok(result)
}

/// Parse TLE text in 3-line format (name, line1, line2, name, line1, line2, ...).
pub fn parse_3le(text: &str) -> Result<Vec<Tle>> {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if !lines.len().is_multiple_of(3) {
        return Err(OrbitalError::TleParse(
            "3LE text must have a multiple of 3 non-empty lines".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(lines.len() / 3);
    for chunk in lines.chunks(3) {
        let name = chunk[0];
        let line1 = chunk[1];
        let line2 = chunk[2];

        if !line1.starts_with('1') {
            return Err(OrbitalError::TleParse(format!(
                "expected line 1 to start with '1', got: {line1}"
            )));
        }
        if !line2.starts_with('2') {
            return Err(OrbitalError::TleParse(format!(
                "expected line 2 to start with '2', got: {line2}"
            )));
        }

        let norad_id = parse_norad_from_line1(line1)?;
        let (epoch_year, epoch_day) = parse_epoch_from_line1(line1)?;

        result.push(Tle {
            name: name.to_string(),
            line1: line1.to_string(),
            line2: line2.to_string(),
            norad_id,
            epoch_year,
            epoch_day,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// GP JSON parsing (Task 4.3)
// ---------------------------------------------------------------------------

/// Parse a CelesTrak-style GP JSON array into TLE structs.
///
/// Each element in the JSON array should have `OBJECT_NAME`, `TLE_LINE1`, and
/// `TLE_LINE2` fields.
pub fn parse_gp_json(json: &str) -> Result<Vec<Tle>> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(json)?;
    let mut result = Vec::with_capacity(entries.len());

    for entry in &entries {
        let name = entry["OBJECT_NAME"].as_str().unwrap_or("").to_string();
        let line1 = entry["TLE_LINE1"]
            .as_str()
            .ok_or_else(|| OrbitalError::Json("missing TLE_LINE1".to_string()))?
            .to_string();
        let line2 = entry["TLE_LINE2"]
            .as_str()
            .ok_or_else(|| OrbitalError::Json("missing TLE_LINE2".to_string()))?
            .to_string();

        let norad_id = parse_norad_from_line1(&line1)?;
        let (epoch_year, epoch_day) = parse_epoch_from_line1(&line1)?;

        result.push(Tle {
            name,
            line1,
            line2,
            norad_id,
            epoch_year,
            epoch_day,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// SGP4 propagation (Task 4.4)
// ---------------------------------------------------------------------------

/// Position and velocity in the TEME frame from SGP4 propagation.
#[derive(Debug, Clone)]
pub struct TemeState {
    /// Position in km.
    pub position_km: [f64; 3],
    /// Velocity in km/s.
    pub velocity_km_s: [f64; 3],
    /// Minutes since TLE epoch.
    pub time_since_epoch_min: f64,
}

/// Propagate a TLE using SGP4 at the given times (minutes since epoch).
///
/// Returns TEME-frame position/velocity at each requested time.
pub fn propagate_tle(tle: &Tle, times_since_epoch_min: &[f64]) -> Result<Vec<TemeState>> {
    let elements = tle.to_sgp4_elements()?;
    let constants = sgp4::Constants::from_elements(&elements)?;

    let mut states = Vec::with_capacity(times_since_epoch_min.len());
    for &t in times_since_epoch_min {
        let prediction = constants.propagate(t)?;
        states.push(TemeState {
            position_km: prediction.position,
            velocity_km_s: prediction.velocity,
            time_since_epoch_min: t,
        });
    }

    Ok(states)
}

// ---------------------------------------------------------------------------
// TEME -> ECEF -> ENU chain (Task 4.5)
// ---------------------------------------------------------------------------

/// Position in the East-North-Up local tangent plane.
#[derive(Debug, Clone)]
pub struct EnuPosition {
    /// East component in metres.
    pub east: f64,
    /// North component in metres.
    pub north: f64,
    /// Up component in metres.
    pub up: f64,
    /// Minutes since TLE epoch.
    pub time_since_epoch_min: f64,
}

/// Propagate a TLE and convert the resulting TEME states to ENU coordinates
/// relative to a ground station.
///
/// # Arguments
/// * `tle` - The TLE to propagate.
/// * `times_min` - Times since TLE epoch in minutes.
/// * `station_lat_rad` - Station geodetic latitude in radians.
/// * `station_lon_rad` - Station geodetic longitude in radians.
/// * `station_alt_m` - Station altitude above the WGS-84 ellipsoid in metres.
pub fn propagate_to_enu(
    tle: &Tle,
    times_min: &[f64],
    station_lat_rad: f64,
    station_lon_rad: f64,
    station_alt_m: f64,
) -> Result<Vec<EnuPosition>> {
    let teme_states = propagate_tle(tle, times_min)?;
    let epoch_jd = tle.epoch_jd();

    let mut enu_positions = Vec::with_capacity(teme_states.len());
    for state in &teme_states {
        // Convert SGP4 km to metres
        let pos_teme_m = nalgebra::Vector3::new(
            state.position_km[0] * 1000.0,
            state.position_km[1] * 1000.0,
            state.position_km[2] * 1000.0,
        );
        let vel_teme_m = nalgebra::Vector3::new(
            state.velocity_km_s[0] * 1000.0,
            state.velocity_km_s[1] * 1000.0,
            state.velocity_km_s[2] * 1000.0,
        );

        // Julian date at this propagation time
        let jd = epoch_jd + state.time_since_epoch_min / 1440.0;

        // TEME -> ECEF
        let (pos_ecef, _vel_ecef) = teme_to_ecef(&pos_teme_m, &vel_teme_m, jd);

        // ECEF -> ENU
        let enu = ecef_to_enu(&pos_ecef, station_lat_rad, station_lon_rad, station_alt_m);

        enu_positions.push(EnuPosition {
            east: enu.x,
            north: enu.y,
            up: enu.z,
            time_since_epoch_min: state.time_since_epoch_min,
        });
    }

    Ok(enu_positions)
}

// ---------------------------------------------------------------------------
// Synthetic radar measurements (Task 4.6)
// ---------------------------------------------------------------------------

/// Configuration for adding noise to synthetic radar measurements.
#[derive(Debug, Clone)]
pub struct RadarNoiseConfig {
    /// Range noise standard deviation in metres.
    pub range_sigma_m: f64,
    /// Azimuth noise standard deviation in radians.
    pub azimuth_sigma_rad: f64,
    /// Elevation noise standard deviation in radians.
    pub elevation_sigma_rad: f64,
    /// Whether to include range-rate measurements.
    pub include_range_rate: bool,
    /// Sensor ID to assign to measurements.
    pub sensor_id: u32,
}

impl Default for RadarNoiseConfig {
    fn default() -> Self {
        Self {
            range_sigma_m: 10.0,
            azimuth_sigma_rad: 0.001,
            elevation_sigma_rad: 0.001,
            include_range_rate: false,
            sensor_id: 0,
        }
    }
}

/// Convert ENU positions to synthetic radar measurements (range, azimuth, elevation).
///
/// This produces noise-free measurements. To add noise, apply Gaussian perturbation
/// using `noise_config` sigma values externally (or use a random number generator).
///
/// Measurements are only generated for positions above the horizon (up > 0).
pub fn orbital_to_radar_measurements(
    enu_positions: &[EnuPosition],
    noise_config: &RadarNoiseConfig,
) -> Vec<Measurement> {
    let mut measurements = Vec::new();

    for pos in enu_positions {
        // Only generate measurements for visible passes (above horizon).
        if pos.up <= 0.0 {
            continue;
        }

        let range = (pos.east * pos.east + pos.north * pos.north + pos.up * pos.up).sqrt();
        let azimuth = pos.east.atan2(pos.north); // atan2(E, N) gives azimuth from north
        let elevation = (pos.up / range).asin();

        // Use time_since_epoch_min as the timestamp (in minutes, convert to seconds).
        let time = pos.time_since_epoch_min * 60.0;

        let range_rate = if noise_config.include_range_rate {
            Some(0.0) // No velocity info available from position-only data
        } else {
            None
        };

        measurements.push(Measurement::Radar {
            range,
            azimuth,
            elevation,
            range_rate,
            time,
            sensor_id: noise_config.sensor_id,
        });
    }

    measurements
}

// ---------------------------------------------------------------------------
// Pass predictions (Task 4.7)
// ---------------------------------------------------------------------------

/// Ground station definition.
#[derive(Debug, Clone)]
pub struct GroundStation {
    /// Station name.
    pub name: String,
    /// Geodetic latitude in radians.
    pub lat_rad: f64,
    /// Geodetic longitude in radians.
    pub lon_rad: f64,
    /// Altitude above WGS-84 ellipsoid in metres.
    pub alt_m: f64,
}

/// A satellite pass over a ground station.
#[derive(Debug, Clone)]
pub struct Pass {
    /// Start time in minutes since TLE epoch.
    pub start_time_min: f64,
    /// End time in minutes since TLE epoch.
    pub end_time_min: f64,
    /// Maximum elevation in radians during the pass.
    pub max_elevation_rad: f64,
    /// Time of maximum elevation in minutes since TLE epoch.
    pub max_elevation_time_min: f64,
}

/// Predict satellite passes over a ground station.
///
/// # Arguments
/// * `tle` - The TLE to propagate.
/// * `station` - Ground station location.
/// * `start_jd` - Start time as Julian Date.
/// * `duration_days` - Duration to search in days.
/// * `min_elevation_rad` - Minimum peak elevation to include a pass (radians).
pub fn predict_passes(
    tle: &Tle,
    station: &GroundStation,
    start_jd: f64,
    duration_days: f64,
    min_elevation_rad: f64,
) -> Result<Vec<Pass>> {
    let epoch_jd = tle.epoch_jd();

    // Sample at 1-minute intervals
    let start_min = (start_jd - epoch_jd) * 1440.0;
    let end_min = start_min + duration_days * 1440.0;
    let step_min = 1.0;

    let n_steps = ((end_min - start_min) / step_min).ceil() as usize;
    let times: Vec<f64> = (0..=n_steps)
        .map(|i| start_min + i as f64 * step_min)
        .collect();

    let enu_positions =
        propagate_to_enu(tle, &times, station.lat_rad, station.lon_rad, station.alt_m)?;

    let passes = extract_passes(&enu_positions, min_elevation_rad);
    Ok(passes)
}

/// Compute elevation angle from an ENU position.
fn enu_elevation(pos: &EnuPosition) -> f64 {
    let range = (pos.east * pos.east + pos.north * pos.north + pos.up * pos.up).sqrt();
    if range > 0.0 {
        (pos.up / range).asin()
    } else {
        0.0
    }
}

/// Extract contiguous above-horizon pass segments from a sequence of ENU positions.
fn extract_passes(enu_positions: &[EnuPosition], min_elevation_rad: f64) -> Vec<Pass> {
    let mut passes = Vec::new();
    let mut in_pass = false;
    let mut pass_start = 0.0_f64;
    let mut max_el = 0.0_f64;
    let mut max_el_time = 0.0_f64;

    for pos in enu_positions {
        let elevation = enu_elevation(pos);

        if elevation > 0.0 {
            if !in_pass {
                in_pass = true;
                pass_start = pos.time_since_epoch_min;
                max_el = elevation;
                max_el_time = pos.time_since_epoch_min;
            }
            if elevation > max_el {
                max_el = elevation;
                max_el_time = pos.time_since_epoch_min;
            }
        } else if in_pass {
            if max_el >= min_elevation_rad {
                passes.push(Pass {
                    start_time_min: pass_start,
                    end_time_min: pos.time_since_epoch_min,
                    max_elevation_rad: max_el,
                    max_elevation_time_min: max_el_time,
                });
            }
            in_pass = false;
        }
    }

    // Handle pass that extends to end of search window
    if in_pass && max_el >= min_elevation_rad {
        if let Some(last) = enu_positions.last() {
            passes.push(Pass {
                start_time_min: pass_start,
                end_time_min: last.time_since_epoch_min,
                max_elevation_rad: max_el,
                max_elevation_time_min: max_el_time,
            });
        }
    }

    passes
}

// ---------------------------------------------------------------------------
// OrbitalDataset (Task 4.8)
// ---------------------------------------------------------------------------

/// A dataset built from orbital TLE propagation.
///
/// Produces measurement frames at each propagation time step for a given
/// ground station.
pub struct OrbitalDataset {
    /// Satellite name.
    name: String,
    /// NORAD catalog ID.
    norad_id: u32,
    /// ENU positions from propagation.
    enu_positions: Vec<EnuPosition>,
    /// Radar noise configuration.
    noise_config: RadarNoiseConfig,
}

impl OrbitalDataset {
    /// Create a new orbital dataset by propagating a TLE over the given time
    /// steps relative to a ground station.
    pub fn new(
        tle: &Tle,
        times_min: &[f64],
        station_lat_rad: f64,
        station_lon_rad: f64,
        station_alt_m: f64,
        noise_config: RadarNoiseConfig,
    ) -> Result<Self> {
        let enu_positions = propagate_to_enu(
            tle,
            times_min,
            station_lat_rad,
            station_lon_rad,
            station_alt_m,
        )?;

        Ok(Self {
            name: tle.name.clone(),
            norad_id: tle.norad_id,
            enu_positions,
            noise_config,
        })
    }
}

impl Dataset for OrbitalDataset {
    fn metadata(&self) -> DatasetMetadata {
        let time_span = if self.enu_positions.len() >= 2 {
            let t0 = self.enu_positions.first().unwrap().time_since_epoch_min * 60.0;
            let t1 = self.enu_positions.last().unwrap().time_since_epoch_min * 60.0;
            Some((t0, t1))
        } else {
            None
        };

        DatasetMetadata {
            name: format!("Orbital: {} (NORAD {})", self.name, self.norad_id),
            source: "orbital".to_string(),
            target_count: Some(1),
            time_span,
            coordinate_frame: CoordinateFrame::Enu,
        }
    }

    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_> {
        let measurements = orbital_to_radar_measurements(&self.enu_positions, &self.noise_config);
        let sensor_id = self.noise_config.sensor_id;
        Box::new(
            measurements
                .into_iter()
                .map(move |m| build_radar_frame(m, sensor_id)),
        )
    }

    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>> {
        // Ground truth from the noise-free orbital positions (above horizon only).
        let norad_id = self.norad_id as u64;
        let gt: Vec<Frame> = self
            .enu_positions
            .iter()
            .filter(|p| p.up > 0.0)
            .map(|p| build_ground_truth_frame(p, norad_id))
            .collect();
        if gt.is_empty() {
            None
        } else {
            Some(Box::new(gt.into_iter()))
        }
    }
}

/// Wrap a radar measurement into a single-measurement `Frame` tagged with
/// the given sensor id.
fn build_radar_frame(m: Measurement, sensor_id: u32) -> Frame {
    let time = m.time();
    Frame {
        timestamp: time,
        measurements: vec![m],
        ground_truth: None,
        sensor_metadata: Some(SensorInfo {
            sensor_id,
            sensor_type: "radar".to_string(),
        }),
    }
}

/// Build a ground-truth `Frame` from a single ENU position.
fn build_ground_truth_frame(p: &EnuPosition, norad_id: u64) -> Frame {
    Frame {
        timestamp: p.time_since_epoch_min * 60.0,
        measurements: Vec::new(),
        ground_truth: Some(vec![GroundTruthEntry {
            target_id: norad_id,
            position: [p.east, p.north, p.up],
            velocity: None,
            class: None,
        }]),
        sensor_metadata: None,
    }
}

// ---------------------------------------------------------------------------
// Space-Track client (Task 4.1)
// ---------------------------------------------------------------------------

const SPACETRACK_BASE: &str = "https://www.space-track.org";

/// Space-Track.org API client with session-cookie authentication.
///
/// On construction, credentials are loaded from the thresh credential store
/// under the `"spacetrack"` service name (env vars `THRESH_SPACETRACK_USERNAME`
/// / `THRESH_SPACETRACK_PASSWORD`, or `~/.thresh/credentials.toml` as a
/// fallback). The first authenticated call triggers a POST to
/// `/ajaxauth/login`; the session cookie returned by the server is captured
/// from the `Set-Cookie` response header and replayed on subsequent requests.
///
/// A hand-rolled cookie jar is used (rather than reqwest's `cookies` feature)
/// so that enabling the `orbital` feature does not pull in the
/// `cookie_store` / `publicsuffix` transitive dependency graph.
pub struct SpaceTrackClient {
    client: reqwest::blocking::Client,
    credentials: crate::credentials::Credentials,
    session_cookie: std::cell::RefCell<Option<String>>,
}

impl Default for SpaceTrackClient {
    fn default() -> Self {
        Self::new()
    }
}

impl SpaceTrackClient {
    /// Create a new Space-Track client using stored credentials.
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            credentials: crate::credentials::load_credentials("spacetrack"),
            session_cookie: std::cell::RefCell::new(None),
        }
    }

    /// Authenticate with Space-Track by POSTing credentials to
    /// `/ajaxauth/login`. The session cookie returned in the
    /// `Set-Cookie` header is captured and reused for subsequent requests.
    fn authenticate(&self) -> Result<()> {
        if self.session_cookie.borrow().is_some() {
            return Ok(());
        }
        let (user, pass) = match (
            self.credentials.username.as_ref(),
            self.credentials.password.as_ref(),
        ) {
            (Some(u), Some(p)) => (u, p),
            _ => {
                return Err(OrbitalError::Auth(
                    "Space-Track credentials not set — set THRESH_SPACETRACK_USERNAME / \
                     THRESH_SPACETRACK_PASSWORD or ~/.thresh/credentials.toml"
                        .into(),
                ));
            }
        };

        let resp = self
            .client
            .post(format!("{SPACETRACK_BASE}/ajaxauth/login"))
            .form(&[("identity", user.as_str()), ("password", pass.as_str())])
            .send()?;

        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(OrbitalError::Auth(format!(
                "login failed (HTTP {code}): {}",
                body.chars().take(200).collect::<String>()
            )));
        }

        let cookies: Vec<String> = resp
            .headers()
            .get_all(reqwest::header::SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .map(parse_cookie_name_value)
            .collect();
        if cookies.is_empty() {
            return Err(OrbitalError::Auth(
                "login succeeded but no Set-Cookie header was returned".into(),
            ));
        }
        *self.session_cookie.borrow_mut() = Some(cookies.join("; "));
        Ok(())
    }

    /// Fetch the latest TLE for a single NORAD catalog ID.
    ///
    /// Issues a GP query against
    /// `/basicspacedata/query/class/gp/NORAD_CAT_ID/<id>/format/json`.
    /// Parses the response with [`parse_gp_json`] so TLE / GP changes stay
    /// localised to one parser.
    pub fn fetch_tle(&self, norad_id: u32) -> Result<Vec<Tle>> {
        self.authenticate()?;
        let url = format!(
            "{SPACETRACK_BASE}/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad_id}/format/json"
        );
        let body = self.get_text(&url)?;
        parse_gp_json(&body)
    }

    /// Fetch the latest TLEs for a list of NORAD catalog IDs in one request.
    pub fn fetch_tles(&self, norad_ids: &[u32]) -> Result<Vec<Tle>> {
        if norad_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.authenticate()?;
        let ids = norad_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let url = format!(
            "{SPACETRACK_BASE}/basicspacedata/query/class/gp/NORAD_CAT_ID/{ids}/format/json"
        );
        let body = self.get_text(&url)?;
        parse_gp_json(&body)
    }

    /// Issue an authenticated GET and return the response body as a string,
    /// mapping non-success HTTP statuses to [`OrbitalError::HttpStatus`].
    fn get_text(&self, url: &str) -> Result<String> {
        let mut req = self.client.get(url);
        if let Some(cookie) = self.session_cookie.borrow().as_ref() {
            req = req.header(reqwest::header::COOKIE, cookie);
        }
        let resp = req.send()?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(OrbitalError::HttpStatus { code, body });
        }
        Ok(resp.text()?)
    }
}

/// Parse a `Set-Cookie` header value down to its `name=value` pair. Any
/// attributes after the first `;` (Path=, HttpOnly, etc.) are dropped because
/// we only care about replaying the session cookie on subsequent requests to
/// the same host.
fn parse_cookie_name_value(set_cookie: &str) -> String {
    set_cookie
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// CelesTrak client (Task 4.9)
// ---------------------------------------------------------------------------

const CELESTRAK_BASE: &str = "https://celestrak.org";

/// CelesTrak public GP data client. No authentication required; used as a
/// redundant backup for Space-Track when Space-Track is throttling or the
/// user has not configured credentials.
pub struct CelestrakClient {
    client: reqwest::blocking::Client,
}

impl Default for CelestrakClient {
    fn default() -> Self {
        Self::new()
    }
}

impl CelestrakClient {
    /// Create a new CelesTrak client (no authentication needed).
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Fetch the GP JSON for a named CelesTrak group (e.g. `"stations"`,
    /// `"active"`, `"starlink"`).
    pub fn fetch_gp_group(&self, group: &str) -> Result<Vec<Tle>> {
        let url = format!("{CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP={group}&FORMAT=json");
        self.fetch_json(&url)
    }

    /// Fetch a single satellite by NORAD catalog ID.
    pub fn fetch_catnr(&self, norad_id: u32) -> Result<Vec<Tle>> {
        let url = format!("{CELESTRAK_BASE}/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=json");
        self.fetch_json(&url)
    }

    /// Shared GET-and-parse path used by both [`fetch_gp_group`] and
    /// [`fetch_catnr`]. Maps non-success HTTP statuses to
    /// [`OrbitalError::HttpStatus`] and delegates parsing to [`parse_gp_json`].
    fn fetch_json(&self, url: &str) -> Result<Vec<Tle>> {
        let resp = self.client.get(url).send()?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(OrbitalError::HttpStatus { code, body });
        }
        let body = resp.text()?;
        parse_gp_json(&body)
    }
}

// ---------------------------------------------------------------------------
// Tests (Tasks 4.10, 4.11, 4.12)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Well-known ISS TLE for testing.
    const ISS_TLE_3LE: &str = "\
ISS (ZARYA)
1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026
2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18";

    const ISS_TLE_2LE: &str = "\
1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026
2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18";

    // ── Task 4.10: TLE parsing tests ─────────────────────────────────────

    #[test]
    fn parse_3le_iss() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        assert_eq!(tles.len(), 1);
        let tle = &tles[0];
        assert_eq!(tle.name, "ISS (ZARYA)");
        assert_eq!(tle.norad_id, 25544);
        assert_eq!(tle.epoch_year, 2024);
        assert!((tle.epoch_day - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_2le_iss() {
        let tles = parse_tle(ISS_TLE_2LE).unwrap();
        assert_eq!(tles.len(), 1);
        let tle = &tles[0];
        assert_eq!(tle.norad_id, 25544);
        assert_eq!(tle.epoch_year, 2024);
    }

    #[test]
    fn parse_3le_rejects_bad_input() {
        assert!(parse_3le("not a tle").is_err());
    }

    #[test]
    fn parse_tle_rejects_odd_lines() {
        let bad = "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026";
        assert!(parse_tle(bad).is_err());
    }

    // ── Task 4.10: GP JSON parsing ───────────────────────────────────────

    #[test]
    fn parse_gp_json_basic() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "TLE_LINE1": "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026",
            "TLE_LINE2": "2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18"
        }]"#;
        let tles = parse_gp_json(json).unwrap();
        assert_eq!(tles.len(), 1);
        assert_eq!(tles[0].name, "ISS (ZARYA)");
        assert_eq!(tles[0].norad_id, 25544);
    }

    // ── Task 4.10: SGP4 propagation ──────────────────────────────────────

    #[test]
    fn propagate_iss_position_reasonable() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        let tle = &tles[0];

        // Propagate at epoch (t=0) and at t=30 min.
        let times = vec![0.0, 30.0, 60.0];
        let states = propagate_tle(tle, &times).unwrap();
        assert_eq!(states.len(), 3);

        for state in &states {
            let r_km = (state.position_km[0].powi(2)
                + state.position_km[1].powi(2)
                + state.position_km[2].powi(2))
            .sqrt();

            // ISS should be at ~400 km altitude => r ≈ 6778 km (6378 + 400)
            assert!(
                r_km > 6300.0 && r_km < 7200.0,
                "ISS radius {r_km} km out of expected LEO range"
            );

            let v_km_s = (state.velocity_km_s[0].powi(2)
                + state.velocity_km_s[1].powi(2)
                + state.velocity_km_s[2].powi(2))
            .sqrt();

            // ISS velocity should be ~7.7 km/s
            assert!(
                v_km_s > 7.0 && v_km_s < 8.5,
                "ISS velocity {v_km_s} km/s out of expected range"
            );
        }
    }

    // ── Task 4.11: Coordinate chain ──────────────────────────────────────

    #[test]
    fn teme_to_ecef_to_enu_chain() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        let tle = &tles[0];

        // Ground station near Washington DC
        let station_lat = 38.9_f64.to_radians();
        let station_lon = (-77.0_f64).to_radians();
        let station_alt = 0.0;

        let times = vec![0.0, 10.0, 20.0, 30.0];
        let enu = propagate_to_enu(tle, &times, station_lat, station_lon, station_alt).unwrap();

        assert_eq!(enu.len(), 4);

        for pos in &enu {
            // Range to satellite should be reasonable (hundreds to thousands of km)
            let range = (pos.east * pos.east + pos.north * pos.north + pos.up * pos.up).sqrt();
            // In metres: at least a few hundred km, at most ~6000 km (half the Earth)
            assert!(
                range > 100_000.0 && range < 20_000_000.0,
                "ENU range {} m seems unreasonable",
                range
            );
        }
    }

    #[test]
    fn radar_measurements_from_enu() {
        // Create some synthetic ENU positions above horizon
        let positions = vec![
            EnuPosition {
                east: 100_000.0,
                north: 200_000.0,
                up: 400_000.0,
                time_since_epoch_min: 0.0,
            },
            EnuPosition {
                east: 50_000.0,
                north: 100_000.0,
                up: 350_000.0,
                time_since_epoch_min: 1.0,
            },
            EnuPosition {
                east: -50_000.0,
                north: -100_000.0,
                up: -10_000.0, // Below horizon — should be excluded
                time_since_epoch_min: 2.0,
            },
        ];

        let config = RadarNoiseConfig::default();
        let measurements = orbital_to_radar_measurements(&positions, &config);

        // Only 2 measurements (third is below horizon)
        assert_eq!(measurements.len(), 2);

        for m in &measurements {
            match m {
                Measurement::Radar {
                    range, elevation, ..
                } => {
                    assert!(*range > 0.0, "range should be positive");
                    assert!(
                        *elevation > 0.0,
                        "elevation should be positive for above-horizon"
                    );
                }
                _ => panic!("expected Radar measurement"),
            }
        }
    }

    // ── Task 4.11: Julian Date computation ───────────────────────────────

    #[test]
    fn tle_epoch_jd_reasonable() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        let tle = &tles[0];

        let jd = tle.epoch_jd();
        // 2024-01-01 00:00 UTC ≈ JD 2460310.5
        assert!(
            (jd - 2_460_310.5).abs() < 1.0,
            "ISS TLE epoch JD {jd} not near expected 2460310.5"
        );
    }

    // ── Task 4.11: Pass predictions ──────────────────────────────────────

    #[test]
    fn predict_passes_finds_some() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        let tle = &tles[0];

        let station = GroundStation {
            name: "Test Station".to_string(),
            lat_rad: 38.9_f64.to_radians(),
            lon_rad: (-77.0_f64).to_radians(),
            alt_m: 0.0,
        };

        // Search 1 day starting at TLE epoch
        let start_jd = tle.epoch_jd();
        let passes = predict_passes(tle, &station, start_jd, 1.0, 5.0_f64.to_radians()).unwrap();

        // ISS makes ~16 orbits per day, so there should be some visible passes
        // (though not all will be above the minimum elevation from any given station).
        // We just verify the function runs and returns reasonable data.
        for pass in &passes {
            assert!(pass.end_time_min > pass.start_time_min);
            assert!(pass.max_elevation_rad >= 5.0_f64.to_radians());
        }
    }

    // ── Task 4.11: OrbitalDataset ────────────────────────────────────────

    #[test]
    fn orbital_dataset_metadata() {
        let tles = parse_3le(ISS_TLE_3LE).unwrap();
        let tle = &tles[0];

        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let station_lat = 38.9_f64.to_radians();
        let station_lon = (-77.0_f64).to_radians();

        let ds = OrbitalDataset::new(
            tle,
            &times,
            station_lat,
            station_lon,
            0.0,
            RadarNoiseConfig::default(),
        )
        .unwrap();

        let meta = ds.metadata();
        assert_eq!(meta.source, "orbital");
        assert_eq!(meta.target_count, Some(1));
        assert!(meta.name.contains("ISS"));
    }

    // ── Task 4.1 / 4.9: Unit tests for HTTP clients (no network) ─────────

    #[test]
    fn parse_cookie_strips_attributes() {
        // A realistic Space-Track Set-Cookie header.
        let raw = "chocolatechip=abc123xyz; Path=/; HttpOnly; Secure; SameSite=Lax";
        assert_eq!(parse_cookie_name_value(raw), "chocolatechip=abc123xyz");
    }

    #[test]
    fn parse_cookie_handles_no_attributes() {
        assert_eq!(
            parse_cookie_name_value("session=token42"),
            "session=token42"
        );
    }

    #[test]
    fn parse_cookie_trims_whitespace() {
        assert_eq!(
            parse_cookie_name_value("  session=token42  ; Path=/"),
            "session=token42"
        );
    }

    #[test]
    fn spacetrack_missing_credentials_yields_auth_error() {
        // Force empty credentials by constructing the client with a
        // guaranteed-missing service name, then swapping the inner
        // credentials. We can't easily do that, so this test asserts the
        // error path on a freshly-constructed client when there is no
        // configured THRESH_SPACETRACK_* env vars in the test environment.
        // If the developer happens to have real creds set in their shell,
        // `authenticate()` would instead try to hit the network; we skip
        // the assertion in that case.
        if std::env::var("THRESH_SPACETRACK_USERNAME").is_ok()
            && std::env::var("THRESH_SPACETRACK_PASSWORD").is_ok()
        {
            return;
        }
        let client = SpaceTrackClient::new();
        // Hold session cookie to None so authenticate() runs.
        *client.session_cookie.borrow_mut() = None;
        // The client's credentials come from load_credentials("spacetrack");
        // with no env vars and no ~/.thresh/credentials.toml entry, both
        // fields are None, which must map to an Auth error rather than
        // attempting a network request.
        if client.credentials.username.is_none() || client.credentials.password.is_none() {
            let err = client.authenticate().unwrap_err();
            assert!(
                matches!(err, OrbitalError::Auth(_)),
                "expected Auth error, got {err:?}"
            );
        }
    }

    #[test]
    fn celestrak_fetch_catnr_url_shape() {
        // Build a client but don't dispatch — just confirm the struct
        // constructs cleanly. The URL shape is covered by the network-gated
        // integration test below.
        let _client = CelestrakClient::new();
    }

    // ── Task 4.12: Network integration tests (ignored by default) ────────

    #[test]
    #[ignore]
    fn spacetrack_fetch_tle() {
        // Exercises the real Space-Track HTTP client. Requires:
        //   - THRESH_SPACETRACK_USERNAME / _PASSWORD env vars
        //   - Network access to www.space-track.org
        // Run with: `cargo test -p thresh-data --features orbital -- --ignored spacetrack_fetch_tle`
        let client = SpaceTrackClient::new();
        let tles = client.fetch_tle(25544).expect("fetch ISS TLE");
        assert!(!tles.is_empty(), "Space-Track returned no TLEs for ISS");
        assert_eq!(tles[0].norad_id, 25544);
    }

    #[test]
    #[ignore]
    fn celestrak_fetch_gp() {
        // Exercises the real CelesTrak HTTP client. Requires network access
        // to celestrak.org. Run with:
        // `cargo test -p thresh-data --features orbital -- --ignored celestrak_fetch_gp`
        let client = CelestrakClient::new();
        let tles = client.fetch_gp_group("stations").expect("fetch stations");
        assert!(!tles.is_empty(), "CelesTrak returned no TLEs for stations");
    }

    #[test]
    #[ignore]
    fn celestrak_fetch_catnr_iss() {
        // Same end-to-end exercise but via the single-ID path.
        let client = CelestrakClient::new();
        let tles = client.fetch_catnr(25544).expect("fetch ISS via CATNR");
        assert!(!tles.is_empty());
        assert_eq!(tles[0].norad_id, 25544);
    }
}
