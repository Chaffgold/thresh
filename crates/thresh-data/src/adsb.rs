//! ADS-B data ingestion from OpenSky Network and SBS BaseStation format.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::Path;
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thresh_core::measurement::Measurement;

use crate::credentials::load_credentials;
use crate::dataset::{CoordinateFrame, Dataset, DatasetMetadata};
use crate::frame::{Frame, GroundTruthEntry};

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors that can occur during ADS-B data ingestion.
#[derive(Debug)]
pub enum AdsBError {
    /// HTTP request failed.
    Http(reqwest::Error),
    /// I/O error (cache, file reads, etc.).
    Io(io::Error),
    /// JSON deserialization error.
    Json(serde_json::Error),
    /// CSV parsing error.
    Csv(csv::Error),
    /// Rate-limited by the API.
    RateLimited,
    /// Unexpected API response structure.
    BadResponse(String),
}

impl fmt::Display for AdsBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdsBError::Http(e) => write!(f, "HTTP error: {e}"),
            AdsBError::Io(e) => write!(f, "I/O error: {e}"),
            AdsBError::Json(e) => write!(f, "JSON error: {e}"),
            AdsBError::Csv(e) => write!(f, "CSV error: {e}"),
            AdsBError::RateLimited => write!(f, "rate limited by OpenSky API"),
            AdsBError::BadResponse(msg) => write!(f, "bad API response: {msg}"),
        }
    }
}

impl std::error::Error for AdsBError {}

impl From<reqwest::Error> for AdsBError {
    fn from(e: reqwest::Error) -> Self {
        AdsBError::Http(e)
    }
}

impl From<io::Error> for AdsBError {
    fn from(e: io::Error) -> Self {
        AdsBError::Io(e)
    }
}

impl From<serde_json::Error> for AdsBError {
    fn from(e: serde_json::Error) -> Self {
        AdsBError::Json(e)
    }
}

impl From<csv::Error> for AdsBError {
    fn from(e: csv::Error) -> Self {
        AdsBError::Csv(e)
    }
}

/// Result alias for ADS-B operations.
pub type Result<T> = std::result::Result<T, AdsBError>;

// ── Bounding box ───────────────────────────────────────────────────────────

/// Geographic bounding box for spatial queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum latitude in degrees.
    pub lat_min: f64,
    /// Maximum latitude in degrees.
    pub lat_max: f64,
    /// Minimum longitude in degrees.
    pub lon_min: f64,
    /// Maximum longitude in degrees.
    pub lon_max: f64,
}

// ── State vector ───────────────────────────────────────────────────────────

/// A single aircraft state from the OpenSky Network REST API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVector {
    /// ICAO24 transponder address (hex).
    pub icao24: String,
    /// Callsign (trimmed), if available.
    pub callsign: Option<String>,
    /// Country of registration.
    pub origin_country: String,
    /// Unix timestamp of last position update.
    pub time_position: Option<f64>,
    /// Unix timestamp of last contact.
    pub last_contact: f64,
    /// Longitude in degrees.
    pub longitude: Option<f64>,
    /// Latitude in degrees.
    pub latitude: Option<f64>,
    /// Barometric altitude in meters.
    pub baro_altitude: Option<f64>,
    /// Whether the aircraft is on the ground.
    pub on_ground: bool,
    /// Ground speed in m/s.
    pub velocity: Option<f64>,
    /// True track angle in degrees (clockwise from north).
    pub true_track: Option<f64>,
    /// Vertical rate in m/s.
    pub vertical_rate: Option<f64>,
    /// Geometric altitude in meters.
    pub geo_altitude: Option<f64>,
}

// ── Track point ────────────────────────────────────────────────────────────

/// A single waypoint from the OpenSky track API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackPoint {
    /// Unix timestamp.
    pub time: f64,
    /// Latitude in degrees.
    pub latitude: Option<f64>,
    /// Longitude in degrees.
    pub longitude: Option<f64>,
    /// Barometric altitude in meters.
    pub baro_altitude: Option<f64>,
    /// True track angle in degrees.
    pub true_track: Option<f64>,
    /// Whether the aircraft is on the ground.
    pub on_ground: bool,
}

// ── SBS message ────────────────────────────────────────────────────────────

/// A parsed SBS BaseStation message (MSG type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbsMessage {
    /// Transmission type (1-8).
    pub transmission_type: u8,
    /// ICAO24 transponder address (hex).
    pub icao24: String,
    /// Callsign, if available.
    pub callsign: Option<String>,
    /// Altitude in feet, if available.
    pub altitude: Option<f64>,
    /// Ground speed in knots, if available.
    pub ground_speed: Option<f64>,
    /// Track angle in degrees, if available.
    pub track: Option<f64>,
    /// Latitude in degrees, if available.
    pub latitude: Option<f64>,
    /// Longitude in degrees, if available.
    pub longitude: Option<f64>,
    /// Vertical rate in ft/min, if available.
    pub vertical_rate: Option<f64>,
    /// Timestamp as combined date+time string.
    pub timestamp: Option<String>,
}

// ── Ground truth trajectory ────────────────────────────────────────────────

/// A ground-truth trajectory for one aircraft, derived from ADS-B state
/// vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthTrajectory {
    /// ICAO24 transponder address.
    pub icao24: String,
    /// Start time (Unix timestamp) of the trajectory.
    pub start_time: f64,
    /// Time-ordered list of ground-truth entries at 1-second intervals.
    pub entries: Vec<GroundTruthEntry>,
}

// ── Rate limiter ───────────────────────────────────────────────────────────

/// Simple rate limiter with exponential backoff.
struct RateLimiter {
    min_interval: Duration,
    last_request: Option<std::time::Instant>,
    backoff: Duration,
    max_backoff: Duration,
}

impl RateLimiter {
    fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_request: None,
            backoff: min_interval,
            max_backoff: Duration::from_secs(60),
        }
    }

    fn wait(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed();
            if elapsed < self.backoff {
                thread::sleep(self.backoff - elapsed);
            }
        }
        self.last_request = Some(std::time::Instant::now());
    }

    fn success(&mut self) {
        self.backoff = self.min_interval;
    }

    fn failure(&mut self) {
        self.backoff = (self.backoff * 2).min(self.max_backoff);
    }
}

// ── OpenSky client ─────────────────────────────────────────────────────────

const OPENSKY_BASE: &str = "https://opensky-network.org/api";

/// Client for the OpenSky Network REST API.
pub struct OpenSkyClient {
    client: reqwest::blocking::Client,
    credentials: Option<(String, String)>,
    rate_limiter: RateLimiter,
}

impl OpenSkyClient {
    /// Create a new client, loading credentials from the thresh credential
    /// store.
    pub fn new() -> Self {
        let creds = load_credentials("opensky");
        let credentials = match (creds.username, creds.password) {
            (Some(u), Some(p)) => Some((u, p)),
            _ => None,
        };

        // Authenticated users get 4 req/s, anonymous get 0.1 req/s.
        let interval = if credentials.is_some() {
            Duration::from_millis(250)
        } else {
            Duration::from_secs(10)
        };

        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            credentials,
            rate_limiter: RateLimiter::new(interval),
        }
    }

    /// Fetch current aircraft states, optionally within a bounding box.
    pub fn fetch_states(
        &mut self,
        time: i64,
        bbox: Option<BoundingBox>,
    ) -> Result<Vec<StateVector>> {
        // Content-hashed cache key: hashes (time, bbox-or-none) into a
        // 16-char hex filename so cache entries don't collide when a new
        // query parameter is added later, and we don't have to worry about
        // formatting floats consistently across callers.
        let time_part = format!("time={time}");
        let bbox_part = match &bbox {
            Some(b) => format!(
                "bbox={},{},{},{}",
                b.lat_min.to_bits(),
                b.lat_max.to_bits(),
                b.lon_min.to_bits(),
                b.lon_max.to_bits(),
            ),
            None => "bbox=none".to_string(),
        };
        let hash = crate::cache::content_hash_key(
            "opensky/states",
            &[time_part.as_str(), bbox_part.as_str()],
        );
        let cache_key = format!("{hash}.json");

        // Check cache first.
        if crate::cache::is_cached("opensky", "states", &cache_key)
            && let Ok(path) = crate::cache::cache_path("opensky", "states", &cache_key)
            && let Ok(data) = std::fs::read_to_string(&path)
            && let Ok(states) = serde_json::from_str::<Vec<StateVector>>(&data)
        {
            return Ok(states);
        }

        let mut url = format!("{OPENSKY_BASE}/states/all?time={time}");
        if let Some(b) = &bbox {
            url.push_str(&format!(
                "&lamin={}&lamax={}&lomin={}&lomax={}",
                b.lat_min, b.lat_max, b.lon_min, b.lon_max
            ));
        }

        let body = self.get_with_retry(&url)?;
        let states = parse_states_response(&body)?;

        // Cache the result.
        if let Ok(dir) = crate::cache::cache_dir("opensky", "states") {
            let _ = std::fs::write(dir.join(&cache_key), serde_json::to_string(&states)?);
        }

        Ok(states)
    }

    /// Fetch the track (waypoints) for a specific aircraft.
    pub fn fetch_track(&mut self, icao24: &str) -> Result<Vec<TrackPoint>> {
        // Content-hashed cache key keyed on (icao24). Using
        // `content_hash_key` rather than the raw ICAO24 means a future
        // change that adds a time window to this endpoint's cache key
        // won't retroactively collide with existing cached tracks.
        let icao_part = format!("icao24={icao24}");
        let hash = crate::cache::content_hash_key("opensky/tracks", &[icao_part.as_str()]);
        let cache_key = format!("{hash}.json");

        if crate::cache::is_cached("opensky", "tracks", &cache_key)
            && let Ok(path) = crate::cache::cache_path("opensky", "tracks", &cache_key)
            && let Ok(data) = std::fs::read_to_string(&path)
            && let Ok(track) = serde_json::from_str::<Vec<TrackPoint>>(&data)
        {
            return Ok(track);
        }

        let url = format!("{OPENSKY_BASE}/tracks/all?icao24={icao24}&time=0");
        let body = self.get_with_retry(&url)?;
        let track = parse_track_response(&body)?;

        if let Ok(dir) = crate::cache::cache_dir("opensky", "tracks") {
            let _ = std::fs::write(dir.join(&cache_key), serde_json::to_string(&track)?);
        }

        Ok(track)
    }

    /// Perform a GET request with rate limiting and exponential backoff on
    /// failure.
    fn get_with_retry(&mut self, url: &str) -> Result<String> {
        const MAX_RETRIES: u32 = 3;

        for attempt in 0..MAX_RETRIES {
            self.rate_limiter.wait();

            let mut req = self.client.get(url);
            if let Some((ref user, ref pass)) = self.credentials {
                req = req.basic_auth(user, Some(pass));
            }

            match req.send() {
                Ok(resp) => {
                    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        self.rate_limiter.failure();
                        if attempt == MAX_RETRIES - 1 {
                            return Err(AdsBError::RateLimited);
                        }
                        continue;
                    }
                    let resp = resp.error_for_status()?;
                    self.rate_limiter.success();
                    return Ok(resp.text()?);
                }
                Err(e) => {
                    self.rate_limiter.failure();
                    if attempt == MAX_RETRIES - 1 {
                        return Err(AdsBError::Http(e));
                    }
                }
            }
        }
        Err(AdsBError::RateLimited)
    }
}

impl Default for OpenSkyClient {
    fn default() -> Self {
        Self::new()
    }
}

// ── JSON parsing ───────────────────────────────────────────────────────────

/// Raw response from the OpenSky states endpoint.
#[derive(Deserialize)]
struct StatesResponse {
    #[allow(dead_code)]
    time: i64,
    states: Option<Vec<Vec<serde_json::Value>>>,
}

fn parse_states_response(body: &str) -> Result<Vec<StateVector>> {
    let resp: StatesResponse = serde_json::from_str(body)?;
    let rows = resp.states.unwrap_or_default();

    let mut states = Vec::with_capacity(rows.len());
    for row in &rows {
        if row.len() < 13 {
            continue;
        }
        let sv = StateVector {
            icao24: row[0].as_str().unwrap_or_default().to_string(),
            callsign: row[1]
                .as_str()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            origin_country: row[2].as_str().unwrap_or_default().to_string(),
            time_position: row[3].as_f64(),
            last_contact: row[4].as_f64().unwrap_or(0.0),
            longitude: row[5].as_f64(),
            latitude: row[6].as_f64(),
            baro_altitude: row[7].as_f64(),
            on_ground: row[8].as_bool().unwrap_or(false),
            velocity: row[9].as_f64(),
            true_track: row[10].as_f64(),
            vertical_rate: row[11].as_f64(),
            geo_altitude: if row.len() > 13 {
                row[13].as_f64()
            } else {
                None
            },
        };
        states.push(sv);
    }
    Ok(states)
}

/// Raw response from the OpenSky tracks endpoint.
#[derive(Deserialize)]
struct TrackResponse {
    #[allow(dead_code)]
    icao24: String,
    path: Option<Vec<Vec<serde_json::Value>>>,
}

fn parse_track_response(body: &str) -> Result<Vec<TrackPoint>> {
    let resp: TrackResponse = serde_json::from_str(body)?;
    let rows = resp.path.unwrap_or_default();

    let mut points = Vec::with_capacity(rows.len());
    for row in &rows {
        if row.len() < 6 {
            continue;
        }
        points.push(TrackPoint {
            time: row[0].as_f64().unwrap_or(0.0),
            latitude: row[1].as_f64(),
            longitude: row[2].as_f64(),
            baro_altitude: row[3].as_f64(),
            true_track: row[4].as_f64(),
            on_ground: row[5].as_bool().unwrap_or(false),
        });
    }
    Ok(points)
}

// ── SBS parsing ────────────────────────────────────────────────────────────

/// Parse a single SBS BaseStation CSV line into an `SbsMessage`.
///
/// SBS format: MSG,type,session,aircraft,icao24,flight,
///             date_gen,time_gen,date_log,time_log,
///             callsign,alt,speed,track,lat,lon,vert_rate,...
pub fn parse_sbs_line(line: &str) -> Option<SbsMessage> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 11 {
        return None;
    }
    if fields[0] != "MSG" {
        return None;
    }

    let transmission_type: u8 = fields[1].trim().parse().ok()?;
    let icao24 = fields[4].trim().to_string();
    if icao24.is_empty() {
        return None;
    }

    let callsign = {
        let s = fields.get(10).map(|s| s.trim().to_string());
        s.filter(|s| !s.is_empty())
    };

    let altitude = fields.get(11).and_then(|s| s.trim().parse::<f64>().ok());
    let ground_speed = fields.get(12).and_then(|s| s.trim().parse::<f64>().ok());
    let track = fields.get(13).and_then(|s| s.trim().parse::<f64>().ok());
    let latitude = fields.get(14).and_then(|s| s.trim().parse::<f64>().ok());
    let longitude = fields.get(15).and_then(|s| s.trim().parse::<f64>().ok());
    let vertical_rate = fields.get(16).and_then(|s| s.trim().parse::<f64>().ok());

    // Combine date and time fields.
    let timestamp = match (fields.get(6), fields.get(7)) {
        (Some(date), Some(time)) => {
            let d = date.trim();
            let t = time.trim();
            if d.is_empty() && t.is_empty() {
                None
            } else {
                Some(format!("{d} {t}"))
            }
        }
        _ => None,
    };

    Some(SbsMessage {
        transmission_type,
        icao24,
        callsign,
        altitude,
        ground_speed,
        track,
        latitude,
        longitude,
        vertical_rate,
        timestamp,
    })
}

/// Parse an entire SBS BaseStation file.
pub fn parse_sbs_file(path: &Path) -> Result<Vec<SbsMessage>> {
    let content = std::fs::read_to_string(path)?;
    let mut messages = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(msg) = parse_sbs_line(line) {
            messages.push(msg);
        }
    }
    Ok(messages)
}

// ── Measurement conversion ─────────────────────────────────────────────────

/// Convert an OpenSky `StateVector` to a `Measurement::AdsB`.
///
/// Returns `None` if latitude, longitude, or altitude are missing.
pub fn state_to_measurement(sv: &StateVector) -> Option<Measurement> {
    let lat = sv.latitude?;
    let lon = sv.longitude?;
    let alt = sv.baro_altitude.or(sv.geo_altitude)?;
    let time = sv.time_position.unwrap_or(sv.last_contact);

    // Compute velocity vector from ground speed, track, and vertical rate.
    let velocity = match (sv.velocity, sv.true_track) {
        (Some(speed), Some(track_deg)) => {
            let track_rad = track_deg.to_radians();
            let vx = speed * track_rad.sin(); // east
            let vy = speed * track_rad.cos(); // north
            let vz = sv.vertical_rate.unwrap_or(0.0);
            Some([vx, vy, vz])
        }
        _ => None,
    };

    Some(Measurement::AdsB {
        lat,
        lon,
        alt,
        velocity,
        time,
    })
}

/// Convert an `SbsMessage` to a `Measurement::AdsB`.
///
/// Returns `None` if position data is missing.
pub fn sbs_to_measurement(msg: &SbsMessage) -> Option<Measurement> {
    let lat = msg.latitude?;
    let lon = msg.longitude?;
    // SBS altitude is in feet; convert to meters.
    let alt = msg.altitude.map(|a| a * 0.3048)?;

    // We don't have a proper Unix timestamp from SBS; use 0.0 as placeholder.
    let time = 0.0;

    let velocity = match (msg.ground_speed, msg.track) {
        (Some(speed_knots), Some(track_deg)) => {
            let speed_ms = speed_knots * 0.514444; // knots to m/s
            let track_rad = track_deg.to_radians();
            let vx = speed_ms * track_rad.sin();
            let vy = speed_ms * track_rad.cos();
            let vz = msg
                .vertical_rate
                .map(|vr| vr * 0.00508) // ft/min to m/s
                .unwrap_or(0.0);
            Some([vx, vy, vz])
        }
        _ => None,
    };

    Some(Measurement::AdsB {
        lat,
        lon,
        alt,
        velocity,
        time,
    })
}

// ── Ground truth extraction ────────────────────────────────────────────────

/// Extract ground-truth trajectories from a set of state vectors.
///
/// Groups by ICAO24, sorts by time, and interpolates to a regular 1-second
/// grid using linear interpolation.
pub fn extract_ground_truth(states: &[StateVector]) -> Vec<GroundTruthTrajectory> {
    let grouped = group_states_by_icao24(states);

    let mut trajectories = Vec::new();

    for (icao24, mut svs) in grouped {
        // Sort by time.
        svs.sort_by(|a, b| {
            let ta = a.time_position.unwrap_or(a.last_contact);
            let tb = b.time_position.unwrap_or(b.last_contact);
            ta.total_cmp(&tb)
        });

        if svs.len() < 2 {
            if let Some(traj) = short_trajectory_to_ground_truth(icao24, &svs) {
                trajectories.push(traj);
            }
            continue;
        }

        if let Some(traj) = interpolate_trajectory_to_grid(icao24, &svs) {
            trajectories.push(traj);
        }
    }

    trajectories
}

/// Group state vectors by ICAO24, keeping only those with a lat/lon fix.
fn group_states_by_icao24(states: &[StateVector]) -> HashMap<String, Vec<&StateVector>> {
    let mut grouped: HashMap<String, Vec<&StateVector>> = HashMap::new();
    for sv in states {
        if sv.latitude.is_some() && sv.longitude.is_some() {
            grouped.entry(sv.icao24.clone()).or_default().push(sv);
        }
    }
    grouped
}

/// Build a trajectory containing a single entry from a single-sample group.
fn short_trajectory_to_ground_truth(
    icao24: String,
    svs: &[&StateVector],
) -> Option<GroundTruthTrajectory> {
    let sv = svs.first()?;
    let (lat, lon) = match (sv.latitude, sv.longitude) {
        (Some(lat), Some(lon)) => (lat, lon),
        _ => return None,
    };
    let alt = sv.baro_altitude.or(sv.geo_altitude).unwrap_or(0.0);
    let velocity = match (sv.velocity, sv.true_track) {
        (Some(speed), Some(track_deg)) => {
            let track_rad = track_deg.to_radians();
            Some([
                speed * track_rad.sin(),
                speed * track_rad.cos(),
                sv.vertical_rate.unwrap_or(0.0),
            ])
        }
        _ => None,
    };
    let start_time = sv.time_position.unwrap_or(sv.last_contact);
    let target_id = u64::from_str_radix(&sv.icao24, 16).unwrap_or(0);
    Some(GroundTruthTrajectory {
        icao24,
        start_time,
        entries: vec![GroundTruthEntry {
            target_id,
            position: [lat, lon, alt],
            velocity,
            class: Some(thresh_core::track::TargetClass::Aircraft),
        }],
    })
}

/// Build an entry by linearly interpolating between two bracketing state vectors.
fn interpolate_entry(
    target_id: u64,
    sv0: &StateVector,
    sv1: &StateVector,
    alpha: f64,
) -> Option<GroundTruthEntry> {
    let lat = lerp_opt(sv0.latitude, sv1.latitude, alpha)?;
    let lon = lerp_opt(sv0.longitude, sv1.longitude, alpha)?;
    let alt = lerp_opt(
        sv0.baro_altitude.or(sv0.geo_altitude),
        sv1.baro_altitude.or(sv1.geo_altitude),
        alpha,
    )
    .unwrap_or(0.0);

    let velocity = match (sv0.velocity, sv0.true_track, sv1.velocity, sv1.true_track) {
        (Some(s0), Some(t0_deg), Some(s1), Some(t1_deg)) => {
            let s = s0 + alpha * (s1 - s0);
            let tr = t0_deg.to_radians() + alpha * (t1_deg.to_radians() - t0_deg.to_radians());
            let vr0 = sv0.vertical_rate.unwrap_or(0.0);
            let vr1 = sv1.vertical_rate.unwrap_or(0.0);
            let vr = vr0 + alpha * (vr1 - vr0);
            Some([s * tr.sin(), s * tr.cos(), vr])
        }
        _ => None,
    };

    Some(GroundTruthEntry {
        target_id,
        position: [lat, lon, alt],
        velocity,
        class: Some(thresh_core::track::TargetClass::Aircraft),
    })
}

/// Interpolate a multi-sample sorted trajectory to a 1-second grid.
fn interpolate_trajectory_to_grid(
    icao24: String,
    svs: &[&StateVector],
) -> Option<GroundTruthTrajectory> {
    let first = svs.first()?;
    let last = svs.last()?;
    let t_start = first.time_position.unwrap_or(first.last_contact);
    let t_end = last.time_position.unwrap_or(last.last_contact);

    if (t_end - t_start) < 1.0 {
        return None;
    }

    let target_id = u64::from_str_radix(&icao24, 16).unwrap_or(0);
    let mut entries = Vec::new();

    let mut t = t_start;
    let mut idx = 0;
    while t <= t_end {
        // Advance index to bracket t.
        while idx + 1 < svs.len() {
            let t_next = svs[idx + 1]
                .time_position
                .unwrap_or(svs[idx + 1].last_contact);
            if t_next >= t {
                break;
            }
            idx += 1;
        }

        if idx + 1 >= svs.len() {
            break;
        }

        let sv0 = svs[idx];
        let sv1 = svs[idx + 1];
        let t0 = sv0.time_position.unwrap_or(sv0.last_contact);
        let t1 = sv1.time_position.unwrap_or(sv1.last_contact);

        if (t1 - t0).abs() < 1e-9 {
            t += 1.0;
            continue;
        }

        let alpha = (t - t0) / (t1 - t0);
        if let Some(entry) = interpolate_entry(target_id, sv0, sv1, alpha) {
            entries.push(entry);
        }

        t += 1.0;
    }

    if entries.is_empty() {
        None
    } else {
        Some(GroundTruthTrajectory {
            icao24,
            start_time: t_start,
            entries,
        })
    }
}

/// Linearly interpolate between two optional values.
fn lerp_opt(a: Option<f64>, b: Option<f64>, alpha: f64) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + alpha * (b - a)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

// ── AdsBDataset ────────────────────────────────────────────────────────────

/// A dataset backed by ADS-B state vectors from the OpenSky Network.
pub struct AdsBDataset {
    /// Name of this dataset.
    name: String,
    /// State vectors, sorted by time.
    states: Vec<StateVector>,
    /// Ground-truth trajectories derived from the states.
    ground_truth_trajectories: Vec<GroundTruthTrajectory>,
}

impl AdsBDataset {
    /// Create a dataset from a collection of state vectors.
    pub fn from_states(name: impl Into<String>, mut states: Vec<StateVector>) -> Self {
        states.sort_by(|a, b| {
            let ta = a.time_position.unwrap_or(a.last_contact);
            let tb = b.time_position.unwrap_or(b.last_contact);
            ta.total_cmp(&tb)
        });
        let ground_truth_trajectories = extract_ground_truth(&states);
        Self {
            name: name.into(),
            states,
            ground_truth_trajectories,
        }
    }

    /// Create a dataset by fetching live data from the OpenSky API.
    pub fn fetch_live(
        client: &mut OpenSkyClient,
        name: impl Into<String>,
        time: i64,
        bbox: Option<BoundingBox>,
    ) -> Result<Self> {
        let states = client.fetch_states(time, bbox)?;
        Ok(Self::from_states(name, states))
    }

    /// Return a reference to the raw state vectors.
    pub fn states(&self) -> &[StateVector] {
        &self.states
    }
}

impl Dataset for AdsBDataset {
    fn metadata(&self) -> DatasetMetadata {
        let times: Vec<f64> = self
            .states
            .iter()
            .map(|s| s.time_position.unwrap_or(s.last_contact))
            .collect();
        let time_span = if times.is_empty() {
            None
        } else {
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((min, max))
        };

        let target_count = {
            let mut ids: Vec<&str> = self.states.iter().map(|s| s.icao24.as_str()).collect();
            ids.sort_unstable();
            ids.dedup();
            Some(ids.len())
        };

        DatasetMetadata {
            name: self.name.clone(),
            source: "opensky".to_string(),
            target_count,
            time_span,
            coordinate_frame: CoordinateFrame::Wgs84,
        }
    }

    fn frames(&self) -> Box<dyn Iterator<Item = Frame> + '_> {
        // Group measurements by integer second.
        let mut time_groups: HashMap<i64, Vec<Measurement>> = HashMap::new();
        for sv in &self.states {
            if let Some(m) = state_to_measurement(sv) {
                let t = m.time() as i64;
                time_groups.entry(t).or_default().push(m);
            }
        }

        let mut frames: Vec<(i64, Vec<Measurement>)> = time_groups.into_iter().collect();
        frames.sort_by_key(|(t, _)| *t);

        Box::new(frames.into_iter().map(|(t, measurements)| Frame {
            timestamp: t as f64,
            measurements,
            ground_truth: None,
            sensor_metadata: None,
        }))
    }

    fn ground_truth(&self) -> Option<Box<dyn Iterator<Item = Frame> + '_>> {
        if self.ground_truth_trajectories.is_empty() {
            return None;
        }

        // Flatten all entries into frames using real timestamps.
        // Each trajectory's entries are at 1-second intervals from start_time.
        let mut all_entries: Vec<(f64, GroundTruthEntry)> = Vec::new();
        for traj in &self.ground_truth_trajectories {
            for (i, entry) in traj.entries.iter().enumerate() {
                all_entries.push((traj.start_time + i as f64, entry.clone()));
            }
        }

        Some(Box::new(all_entries.into_iter().map(|(t, entry)| Frame {
            timestamp: t,
            measurements: Vec::new(),
            ground_truth: Some(vec![entry]),
            sensor_metadata: None,
        })))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sbs_msg3_position() {
        let line = "MSG,3,111,11111,A00001,111111,2024/01/01,12:00:00.000,2024/01/01,12:00:00.000,,35000,,,40.6413,-73.7781,,,,,,0";
        let msg = parse_sbs_line(line).expect("should parse valid SBS line");
        assert_eq!(msg.transmission_type, 3);
        assert_eq!(msg.icao24, "A00001");
        assert!((msg.latitude.unwrap() - 40.6413).abs() < 1e-4);
        assert!((msg.longitude.unwrap() - -73.7781).abs() < 1e-4);
        assert!((msg.altitude.unwrap() - 35000.0).abs() < 1e-1);
    }

    #[test]
    fn parse_sbs_msg1_id() {
        let line = "MSG,1,111,11111,A00002,111111,2024/01/01,12:00:00.000,2024/01/01,12:00:00.000,UAL123,,,,,,,,,,0";
        let msg = parse_sbs_line(line).expect("should parse MSG type 1");
        assert_eq!(msg.transmission_type, 1);
        assert_eq!(msg.icao24, "A00002");
        assert_eq!(msg.callsign.as_deref(), Some("UAL123"));
        assert!(msg.latitude.is_none());
    }

    #[test]
    fn parse_sbs_rejects_non_msg() {
        assert!(parse_sbs_line("SEL,,,,,,,,,,,,,,,,,,").is_none());
        assert!(parse_sbs_line("").is_none());
        assert!(parse_sbs_line("MSG").is_none()); // too few fields
    }

    #[test]
    fn parse_sbs_msg4_velocity() {
        let line = "MSG,4,111,11111,A00003,111111,2024/01/01,12:00:00.000,2024/01/01,12:00:00.000,,35000,450.0,180.0,,,500,,,,0";
        let msg = parse_sbs_line(line).expect("should parse MSG type 4");
        assert_eq!(msg.transmission_type, 4);
        assert!((msg.ground_speed.unwrap() - 450.0).abs() < 1e-1);
        assert!((msg.track.unwrap() - 180.0).abs() < 1e-1);
        assert!((msg.vertical_rate.unwrap() - 500.0).abs() < 1e-1);
    }

    #[test]
    fn state_to_measurement_with_position_only() {
        let sv = StateVector {
            icao24: "abc123".into(),
            callsign: None,
            origin_country: "US".into(),
            time_position: Some(1700000000.0),
            last_contact: 1700000000.0,
            longitude: Some(-73.7781),
            latitude: Some(40.6413),
            baro_altitude: Some(10668.0),
            on_ground: false,
            velocity: None,
            true_track: None,
            vertical_rate: None,
            geo_altitude: None,
        };

        let m = state_to_measurement(&sv).expect("should convert to measurement");
        match m {
            Measurement::AdsB {
                lat,
                lon,
                alt,
                velocity,
                time,
            } => {
                assert!((lat - 40.6413).abs() < 1e-4);
                assert!((lon - -73.7781).abs() < 1e-4);
                assert!((alt - 10668.0).abs() < 1e-1);
                assert!(velocity.is_none());
                assert!((time - 1700000000.0).abs() < 1e-1);
            }
            _ => panic!("expected AdsB measurement"),
        }
    }

    #[test]
    fn state_to_measurement_with_velocity() {
        let sv = StateVector {
            icao24: "abc123".into(),
            callsign: Some("TEST123".into()),
            origin_country: "US".into(),
            time_position: Some(1700000000.0),
            last_contact: 1700000000.0,
            longitude: Some(-73.7781),
            latitude: Some(40.6413),
            baro_altitude: Some(10668.0),
            on_ground: false,
            velocity: Some(250.0),
            true_track: Some(90.0),
            vertical_rate: Some(-5.0),
            geo_altitude: Some(10700.0),
        };

        let m = state_to_measurement(&sv).expect("should convert to measurement");
        match m {
            Measurement::AdsB { velocity, .. } => {
                let v = velocity.expect("should have velocity");
                // Track 90 deg => vx = 250, vy ~= 0
                assert!((v[0] - 250.0).abs() < 1e-1);
                assert!(v[1].abs() < 1e-1);
                assert!((v[2] - -5.0).abs() < 1e-1);
            }
            _ => panic!("expected AdsB measurement"),
        }
    }

    #[test]
    fn state_to_measurement_missing_position() {
        let sv = StateVector {
            icao24: "abc123".into(),
            callsign: None,
            origin_country: "US".into(),
            time_position: None,
            last_contact: 1700000000.0,
            longitude: None,
            latitude: None,
            baro_altitude: None,
            on_ground: true,
            velocity: None,
            true_track: None,
            vertical_rate: None,
            geo_altitude: None,
        };
        assert!(state_to_measurement(&sv).is_none());
    }

    #[test]
    fn ground_truth_extraction_groups_by_icao() {
        let states = vec![
            make_test_sv("aaa111", 1000.0, 40.0, -74.0, 10000.0),
            make_test_sv("aaa111", 1001.0, 40.001, -74.001, 10010.0),
            make_test_sv("aaa111", 1002.0, 40.002, -74.002, 10020.0),
            make_test_sv("bbb222", 1000.0, 35.0, -118.0, 8000.0),
            make_test_sv("bbb222", 1001.0, 35.001, -118.001, 8010.0),
        ];

        let trajectories = extract_ground_truth(&states);
        assert_eq!(trajectories.len(), 2);

        // Both trajectories should have entries.
        for traj in &trajectories {
            assert!(!traj.entries.is_empty());
        }
    }

    #[test]
    fn ground_truth_interpolation() {
        let states = vec![
            make_test_sv("abc123", 1000.0, 40.0, -74.0, 10000.0),
            make_test_sv("abc123", 1002.0, 40.002, -74.002, 10020.0),
        ];

        let trajectories = extract_ground_truth(&states);
        assert_eq!(trajectories.len(), 1);

        let traj = &trajectories[0];
        // Should have 3 entries (t=1000, t=1001, t=1002).
        assert_eq!(traj.entries.len(), 3);

        // Midpoint should be interpolated.
        let mid = &traj.entries[1];
        assert!((mid.position[0] - 40.001).abs() < 1e-6);
        assert!((mid.position[1] - -74.001).abs() < 1e-6);
        assert!((mid.position[2] - 10010.0).abs() < 1e-1);
    }

    #[test]
    fn parse_opensky_states_json() {
        let json = r#"{"time":1700000000,"states":[["abc123","TEST   ","United States",1700000000,1700000000,-73.7781,40.6413,10668.0,false,250.0,90.0,-5.0,null,10700.0,null,false,0]]}"#;
        let states = parse_states_response(json).expect("should parse states JSON");
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].icao24, "abc123");
        assert_eq!(states[0].callsign.as_deref(), Some("TEST"));
        assert!((states[0].latitude.unwrap() - 40.6413).abs() < 1e-4);
    }

    #[test]
    fn parse_opensky_states_empty() {
        let json = r#"{"time":1700000000,"states":null}"#;
        let states = parse_states_response(json).expect("should parse null states");
        assert!(states.is_empty());
    }

    #[test]
    fn parse_opensky_track_json() {
        let json = r#"{"icao24":"abc123","callsign":"TEST","startTime":0,"endTime":0,"path":[[1700000000,40.6413,-73.7781,10668.0,90.0,false]]}"#;
        let track = parse_track_response(json).expect("should parse track JSON");
        assert_eq!(track.len(), 1);
        assert!((track[0].time - 1700000000.0).abs() < 1.0);
        assert!((track[0].latitude.unwrap() - 40.6413).abs() < 1e-4);
    }

    #[test]
    fn adsb_dataset_metadata() {
        let states = vec![
            make_test_sv("aaa111", 1000.0, 40.0, -74.0, 10000.0),
            make_test_sv("bbb222", 1001.0, 35.0, -118.0, 8000.0),
        ];
        let ds = AdsBDataset::from_states("test", states);
        let meta = ds.metadata();
        assert_eq!(meta.name, "test");
        assert_eq!(meta.source, "opensky");
        assert_eq!(meta.target_count, Some(2));
    }

    #[test]
    fn adsb_dataset_frames() {
        let states = vec![
            make_test_sv("aaa111", 1000.0, 40.0, -74.0, 10000.0),
            make_test_sv("bbb222", 1000.0, 35.0, -118.0, 8000.0),
            make_test_sv("aaa111", 1001.0, 40.001, -74.001, 10010.0),
        ];
        let ds = AdsBDataset::from_states("test", states);
        let frames: Vec<Frame> = ds.frames().collect();
        assert_eq!(frames.len(), 2); // two distinct time buckets
    }

    #[test]
    fn sbs_to_measurement_conversion() {
        let msg = SbsMessage {
            transmission_type: 3,
            icao24: "A00001".into(),
            callsign: None,
            altitude: Some(35000.0),   // feet
            ground_speed: Some(450.0), // knots
            track: Some(180.0),
            latitude: Some(40.6413),
            longitude: Some(-73.7781),
            vertical_rate: Some(0.0),
            timestamp: None,
        };
        let m = sbs_to_measurement(&msg).expect("should convert SBS to measurement");
        match m {
            Measurement::AdsB { lat, lon, alt, .. } => {
                assert!((lat - 40.6413).abs() < 1e-4);
                assert!((lon - -73.7781).abs() < 1e-4);
                // 35000 ft * 0.3048 = 10668 m
                assert!((alt - 10668.0).abs() < 1.0);
            }
            _ => panic!("expected AdsB measurement"),
        }
    }

    // ── Network integration test ───────────────────────────────────────

    #[test]
    #[ignore]
    fn integration_fetch_live_states() {
        let mut client = OpenSkyClient::new();
        let bbox = BoundingBox {
            lat_min: 45.0,
            lat_max: 47.0,
            lon_min: 5.0,
            lon_max: 8.0,
        };
        let states = client
            .fetch_states(0, Some(bbox))
            .expect("should fetch states");
        // We can't guarantee results, but the call should succeed.
        println!("Fetched {} state vectors", states.len());
    }

    // ── Helpers ────────────────────────────────────────────────────────

    fn make_test_sv(icao24: &str, time: f64, lat: f64, lon: f64, alt: f64) -> StateVector {
        StateVector {
            icao24: icao24.into(),
            callsign: None,
            origin_country: "US".into(),
            time_position: Some(time),
            last_contact: time,
            longitude: Some(lon),
            latitude: Some(lat),
            baro_altitude: Some(alt),
            on_ground: false,
            velocity: None,
            true_track: None,
            vertical_rate: None,
            geo_altitude: None,
        }
    }
}
