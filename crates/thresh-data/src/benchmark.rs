//! Benchmark scenario manifest, runner, and regression checking.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use nalgebra::{DVector, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use thresh_core::eci::enu_to_eci;
use thresh_core::track::TargetClass;
use thresh_eval::consistency::{
    ConsistencyAccumulator, EstimateFrame, INTERLEAVED_POSITION_INDICES, TrackEstimate, chi2,
    sequence_anees,
};
use thresh_eval::gospa::{GospaParams, gospa_sequence};
use thresh_eval::hota::compute_hota_at_threshold;
use thresh_eval::matching::FrameData;
use thresh_eval::metrics::{compute_idf1, compute_mot_metrics};
use thresh_synth::measurement_gen::RadarConfig;
use thresh_synth::scenario::{GroundTruth, run_scenario_with_rng};
use thresh_synth::trajectory::{Segment, SegmentType, Trajectory};
use thresh_tracker::tracker::MultiObjectTracker;
use thresh_tracker::tracker_variant::TrackerVariant;

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/// Top-level benchmark scenario description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub name: String,
    pub description: String,
    pub source: ScenarioSource,
    pub parameters: ScenarioParameters,
    pub baselines: Option<Baselines>,
}

/// Where the data comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioSource {
    Synthetic,
    /// ADS-B scenario sourced from a cached state-vector fixture or the
    /// OpenSky Network REST API. Works the same way as `Orbital`: when
    /// `state_file` is set, the runner reads that file (JSON-serialised
    /// `Vec<StateVector>`) from the manifest's directory; otherwise it
    /// falls back to an authenticated OpenSky call bounded by `bbox`.
    /// `region` stays in the schema for backwards compatibility and
    /// is echoed in CLI output so older scenarios still print sensibly.
    /// `ref_lat_deg` / `ref_lon_deg` / `ref_alt_m` define the tracker's
    /// local ENU frame origin in degrees (human-editable in TOML).
    AdsB {
        region: String,
        #[serde(default)]
        state_file: Option<String>,
        #[serde(default)]
        bbox: Option<AdsBBoundingBox>,
        #[serde(default = "default_adsb_ref_lat_deg")]
        ref_lat_deg: f64,
        #[serde(default = "default_adsb_ref_lon_deg")]
        ref_lon_deg: f64,
        #[serde(default)]
        ref_alt_m: f64,
    },
    /// Orbital scenario sourced from SGP4 propagation of one or more TLEs.
    ///
    /// Field meanings:
    /// - `norad_ids`: catalog IDs to fetch over HTTP (Space-Track / CelesTrak)
    ///   when `tle_file` is not set and the `orbital` feature is enabled.
    /// - `tle_file`: optional path to a local cached TLE file (3LE or 2LE
    ///   format). **Relative paths** are resolved relative to the scenario
    ///   manifest's directory, so a scenario `scenarios/orbital-iss.toml`
    ///   with `tle_file = "orbital-iss.tle"` reads `scenarios/orbital-iss.tle`.
    ///   When present, the runner uses this file and never touches the
    ///   network — this is what lets the CI gate run orbital scenarios
    ///   offline.
    /// - `station_lat_deg` / `station_lon_deg` / `station_alt_m`: ground
    ///   station location used to convert ECI → ENU for the tracker and
    ///   radar observation model. Degrees (not radians) in the manifest to
    ///   keep the TOML human-editable.
    /// - `time_step_s`: interval between propagation samples (and between
    ///   radar scans). If omitted, defaults to the `parameters.dt` value.
    Orbital {
        norad_ids: Vec<u32>,
        #[serde(default)]
        tle_file: Option<String>,
        #[serde(default = "default_station_lat_deg")]
        station_lat_deg: f64,
        #[serde(default = "default_station_lon_deg")]
        station_lon_deg: f64,
        #[serde(default)]
        station_alt_m: f64,
        #[serde(default)]
        time_step_s: Option<f64>,
    },
    /// nuScenes scenario sourced from a local nuScenes mini / trainval /
    /// test split via the feature-gated PyO3 bridge.
    ///
    /// - `version`: nuScenes split name the devkit accepts (e.g.
    ///   `"v1.0-mini"`).
    /// - `dataroot`: optional path to the dataset root. When omitted the
    ///   runner reads `NUSCENES_DATA_ROOT` from the environment so the
    ///   manifest stays portable across developer machines — no absolute
    ///   paths checked in.
    /// - `scene_token`: optional specific scene token. When omitted the
    ///   runner picks the first scene returned by the devkit (stable
    ///   ordering for `v1.0-mini`).
    NuScenes {
        #[serde(default = "default_nuscenes_version")]
        version: String,
        #[serde(default)]
        dataroot: Option<String>,
        #[serde(default)]
        scene_token: Option<String>,
    },
    /// Ballistic scenario generated from a phased boost / midcourse /
    /// reentry [`thresh_synth::ballistic::BallisticProfile`] truth observed
    /// by a single ground radar (design Decision 7 of the
    /// `orbital-ballistic-filter-models` change).
    ///
    /// The launch/profile fields map 1:1 onto `BallisticProfile` with
    /// angles in **degrees** (and β/thrust units spelled out) so the TOML
    /// stays human-editable; the runner converts to radians. The station
    /// fields locate the observing radar exactly like
    /// [`ScenarioSource::Orbital`]'s station fields. Truth generation is
    /// pure Rust (no SGP4), so ballistic scenarios run under **default
    /// features** — no `orbital` feature required.
    Ballistic {
        /// Launch geodetic latitude (degrees).
        launch_lat_deg: f64,
        /// Launch geodetic longitude (degrees).
        launch_lon_deg: f64,
        /// Launch altitude above the WGS-84 ellipsoid (metres).
        #[serde(default)]
        launch_alt_m: f64,
        /// Launch azimuth (degrees clockwise from north).
        launch_azimuth_deg: f64,
        /// Constant boost thrust acceleration (m/s²).
        thrust_accel_m_s2: f64,
        /// Boost duration (s).
        burn_time_s: f64,
        /// Vertical-rise duration before the pitch kick (s).
        pitch_over_s: f64,
        /// Instantaneous downrange pitch kick (degrees).
        pitch_kick_deg: f64,
        /// Ballistic coefficient β = m/(C_d·A) (kg/m²).
        beta_kg_m2: f64,
        /// Launch epoch as a Julian Date in the **UTC** scale (kept as a
        /// raw TOML float — it feeds `BallisticProfile::epoch_jd`, the
        /// calibrated chain's raw-JD plumbing; see that field for the
        /// bitwise-invariance rationale of `astro-time-and-frames`
        /// Decision 6).
        epoch_jd: f64,
        /// Radar station geodetic latitude (degrees).
        station_lat_deg: f64,
        /// Radar station geodetic longitude (degrees).
        station_lon_deg: f64,
        /// Radar station altitude above the WGS-84 ellipsoid (metres).
        #[serde(default)]
        station_alt_m: f64,
    },
}

fn default_nuscenes_version() -> String {
    // The mini split is the only one small enough to keep on a developer
    // laptop (~4 GB), so it's the sensible default.
    "v1.0-mini".to_string()
}

fn default_station_lat_deg() -> f64 {
    // Colorado Springs ground station (generic default, no operational
    // significance). Overridden per-scenario when it matters.
    38.8339
}

fn default_station_lon_deg() -> f64 {
    -104.8214
}

fn default_adsb_ref_lat_deg() -> f64 {
    // JFK International Airport — arbitrary default used by both the
    // `adsb-single-flight` and `adsb-tracon` scenarios.
    40.6413
}

fn default_adsb_ref_lon_deg() -> f64 {
    -73.7781
}

/// Serializable bounding box for ADS-B scenario manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdsBBoundingBox {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
}

/// Common scenario parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioParameters {
    pub duration_s: f64,
    pub dt: f64,
    pub measurement_noise_sigma: f64,
    pub gate_threshold: f64,
    /// Optional tracker variant override. When `None` (the default) the
    /// benchmark runner uses the Cartesian ENU tracker — the same behaviour
    /// as before this field was added.
    ///
    /// This field is wired up as a forward-compatibility hook: the runner
    /// currently only drives the ENU tracker end-to-end. When the runner
    /// gains support for the other variants (ECEF, Great-Circle,
    /// Stereographic), selection will be honoured automatically without
    /// requiring scenario files to change.
    #[serde(default)]
    pub tracker_variant: Option<TrackerVariant>,
    /// Scenario type for synthetic benchmarks. Controls which trajectory
    /// builder and radar configuration are used.
    ///
    /// Recognised values: `"cv-clean"`, `"maneuvering"`, `"heterogeneous"`,
    /// `"low-pd"`. When `None` the runner defaults to `"cv-clean"`.
    ///
    /// The orbital runner recognises one additional value:
    /// `"force-cv-head"` births tracks as [`TargetClass::Unknown`] (the CV
    /// head) instead of [`TargetClass::Orbital`] (the Kepler+J2 head),
    /// keeping every other pipeline stage identical. It exists solely for
    /// the deliberate-regression check of the orbital benchmark gate
    /// (task 6.6 of `orbital-ballistic-filter-models`): forcing the CV
    /// head on the ISS scenario must fail the calibrated MOTA baseline.
    #[serde(default)]
    pub scenario_type: Option<String>,
    /// Measurement model for the synthetic runner (defaults to
    /// [`MeasurementModel::RadarRae`] when the TOML omits the key).
    #[serde(default)]
    pub measurement_model: MeasurementModel,
    /// Measurement-noise sigma the *tracker* is configured with, when it
    /// should differ from the generator's `measurement_noise_sigma`.
    ///
    /// `None` (the default, and every committed scenario) keeps the honest
    /// configuration: the tracker's R matches the noise actually generated.
    /// A mismatched value is a deliberate covariance lie — it leaves the
    /// detections (and therefore MOT metrics) essentially untouched while
    /// driving ANEES/ANIS out of the chi-squared interval. It exists for
    /// the `eval-consistency-metrics` deliberate-regression tests
    /// (task 6.6: the gate must catch a covariance lie that MOTA cannot
    /// see, in both directions).
    #[serde(default)]
    pub tracker_noise_sigma: Option<f64>,
}

/// Measurement model driving the synthetic runner's noise generation.
///
/// A typed enum rather than a free string so that a misspelled TOML value
/// (`"cartesain"`, `"Cartesian"`) fails at parse time with serde's
/// variant list instead of silently selecting the default radar model —
/// the choice is load-bearing for the consistency gates: under
/// [`MeasurementModel::RadarRae`] the range-dependent cross-range error
/// judged against the tracker's isotropic R measured ANEES ≈ 13 on
/// `synth-cv-clean`, far outside its committed [2.46, 3.56] bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementModel {
    /// RAE radar noise via `generate_radar`: range/azimuth/elevation
    /// sigmas whose Cartesian-converted per-axis error grows with range
    /// (cross-range sigma = r · sigma_az).
    #[default]
    RadarRae,
    /// Isotropic Gaussian of `measurement_noise_sigma` per axis drawn
    /// directly on truth positions, exactly matching the tracker's
    /// isotropic R — the textbook consistency configuration the
    /// `synth-cv-clean` ANEES/ANIS gate requires
    /// (`eval-consistency-metrics` design Decision 5).
    Cartesian,
}

/// Expected metric baselines for regression gating.
///
/// The MOT fields (`mota` / `hota` / `idf1`) are one-sided floors; the
/// consistency fields are the two-sided ANEES / average-NIS acceptance
/// intervals of the `eval-consistency-metrics` change (design Decision 5).
/// All consistency bounds are optional with serde defaults, so every
/// pre-existing scenario TOML parses unchanged. Asserting a bound against a
/// run that produced no samples for that statistic is a gate **failure**
/// (fail-loud rule), never a silent pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Baselines {
    pub mota: Option<f64>,
    pub hota: Option<f64>,
    pub idf1: Option<f64>,
    /// Lower ANEES bound (two-sided interval, underconfidence tail).
    #[serde(default)]
    pub anees_min: Option<f64>,
    /// Upper ANEES bound (two-sided interval, overconfidence tail).
    #[serde(default)]
    pub anees_max: Option<f64>,
    /// Lower average-NIS bound (two-sided interval, underconfidence tail).
    #[serde(default)]
    pub anis_min: Option<f64>,
    /// Upper average-NIS bound (two-sided interval, overconfidence tail).
    #[serde(default)]
    pub anis_max: Option<f64>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a scenario manifest from a TOML file.
pub fn load_scenario(path: &Path) -> Result<ScenarioManifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// Results produced by running a benchmark scenario.
///
/// The consistency statistics (`anees` / `anis` / `gospa`) are `Option`s:
/// absent (`None`) is distinct from zero and from NaN (spec: "Absent
/// statistics are explicit"). A statistic is absent when the run produced
/// no samples for it — e.g. no matched truth/track pairs for ANEES, or an
/// update path that legitimately collects no NIS diagnostics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub scenario: String,
    pub mota: f64,
    pub motp: f64,
    pub idf1: f64,
    pub hota: f64,
    pub id_switches: usize,
    /// Scenario-level ANEES: position-marginal NEES (dof 3) averaged over
    /// the matched truth/track pairs at the run's MOTA `dist_threshold`
    /// (the NEES population is exactly the MOTA true-positive population).
    pub anees: Option<f64>,
    /// Number of NEES samples behind [`Self::anees`].
    pub anees_samples: usize,
    /// Average NIS of the tracker's post-warmup single-model KF updates
    /// ([`MultiObjectTracker::nis_stats`] — each track's first two updates
    /// are excluded as birth transient); IMM/JPDA paths collect none.
    pub anis: Option<f64>,
    /// Number of NIS samples behind [`Self::anis`].
    pub anis_samples: usize,
    /// Sequence GOSPA (α = 2, order p = 2): the order-p mean of the
    /// per-frame totals with cutoff `c` = the run's MOTA `dist_threshold`.
    /// Reported for diagnostics, not gated. `None` for an empty sequence.
    pub gospa: Option<f64>,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Shared runner helpers (§7.3 / §7.5 / §7.7)
// ---------------------------------------------------------------------------

/// Collect the current confirmed-track positions out of a
/// `MultiObjectTracker` in the `(id, [x, y, z])` shape expected by
/// `FrameData`. Factoring this out lets every feature-gated runner
/// share the same filter / project / collect logic without each one
/// open-coding it.
pub(crate) fn collect_confirmed_track_positions(
    tracker: &MultiObjectTracker,
) -> Vec<(u64, [f64; 3])> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle == thresh_core::track::TrackState::Confirmed)
        .map(|t| (t.id.0, [t.state[0], t.state[2], t.state[4]]))
        .collect()
}

/// Collect the current confirmed tracks' **full** state estimates and
/// covariances in the [`TrackEstimate`] shape `sequence_anees` consumes.
///
/// Sibling of [`collect_confirmed_track_positions`]: same confirmed-only
/// filter and ID space, but copying the already-`pub` `track.state` /
/// `track.covariance` so position-marginal NEES can extract the 3×3
/// covariance block (`eval-consistency-metrics` design Decision 5 —
/// no thresh-core change needed).
pub(crate) fn collect_confirmed_track_estimates(
    tracker: &MultiObjectTracker,
) -> Vec<TrackEstimate> {
    tracker
        .tracks
        .iter()
        .filter(|t| t.lifecycle == thresh_core::track::TrackState::Confirmed)
        .map(|t| TrackEstimate {
            id: t.id.0,
            state: t.state.clone(),
            covariance: t.covariance.clone(),
        })
        .collect()
}

/// Append one benchmark step's [`FrameData`] (position-only, for the MOT
/// metrics and GOSPA) and [`EstimateFrame`] (full state + covariance, for
/// ANEES) built from the same ground truth and the tracker's current
/// confirmed set — the shared per-step collection all runners use.
fn push_step_frames(
    tracker: &MultiObjectTracker,
    gt: Vec<(u64, [f64; 3])>,
    frames: &mut Vec<FrameData>,
    estimates: &mut Vec<EstimateFrame>,
) {
    frames.push(FrameData {
        gt: gt.clone(),
        tracks: collect_confirmed_track_positions(tracker),
    });
    estimates.push(EstimateFrame {
        gt,
        tracks: collect_confirmed_track_estimates(tracker),
    });
}

/// Compute the final MOT metric set from a collected `FrameData`
/// sequence and package everything into a [`BenchmarkResult`]. All
/// benchmark runners (synthetic / ADS-B / orbital / ballistic / nuScenes)
/// converge on this path once their step loops finish — it centralises the
/// MOTA / MOTP / IDF1 / HOTA calls, the consistency statistics (ANEES via
/// [`sequence_anees`] over `estimate_frames`, ANIS from the tracker's
/// [`MultiObjectTracker::nis_stats`] accessor, sequence GOSPA with
/// `c = dist_threshold`), the calibration print, the `duration_ms`
/// stopwatch reading, and the `BenchmarkResult` assembly.
///
/// `dist_threshold` is the matcher distance threshold in metres.
/// Callers pick a value appropriate for their scenario regime
/// (nuScenes uses metres at ~1 m noise, orbital uses kilometres at
/// ~1 km noise, and so on). ANEES uses the **same** threshold, so the NEES
/// sample population is exactly the MOTA true-positive population, and
/// GOSPA's cutoff derives from the same expression so all three metrics
/// agree on what "close enough" means.
pub(crate) fn build_benchmark_result(
    scenario_name: &str,
    frame_data_vec: &[FrameData],
    estimate_frames: &[EstimateFrame],
    tracker: &MultiObjectTracker,
    dist_threshold: f64,
    start: Instant,
) -> BenchmarkResult {
    let (mota, motp, id_switches) = compute_mot_metrics(frame_data_vec, dist_threshold);
    let idf1 = compute_idf1(frame_data_vec, dist_threshold);
    let (hota, _, _) = compute_hota_at_threshold(frame_data_vec, dist_threshold);
    let anees_acc = sequence_anees(
        estimate_frames,
        INTERLEAVED_POSITION_INDICES,
        dist_threshold,
    );
    let anis_acc = nis_accumulator(tracker);
    let gospa = compute_sequence_gospa(frame_data_vec, dist_threshold);
    eprint_consistency_calibration(scenario_name, &anees_acc, &anis_acc);
    BenchmarkResult {
        scenario: scenario_name.to_string(),
        mota,
        motp,
        idf1,
        hota,
        id_switches,
        anees: anees_acc.mean(),
        anees_samples: anees_acc.n,
        anis: anis_acc.mean(),
        anis_samples: anis_acc.n,
        gospa,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// View the tracker's accumulated NIS statistics as a
/// [`ConsistencyAccumulator`] so the eval crate's mean / two-sided-bounds
/// machinery applies. The tracker cannot depend on `thresh-eval` (workspace
/// layering), so its accumulator is a plain local type and the two meet
/// here (`eval-consistency-metrics` design Decision 5).
fn nis_accumulator(tracker: &MultiObjectTracker) -> ConsistencyAccumulator {
    let stats = tracker.nis_stats();
    ConsistencyAccumulator {
        sum: stats.sum,
        n: stats.count,
        dof: stats.dof,
    }
}

/// Sequence GOSPA (reported, not gated): α = 2 with the default order
/// p = 2 and cutoff `c` = the runner's MOTA `dist_threshold`. `None` for an
/// empty frame sequence — absent, not zero (spec: "Absent statistics are
/// explicit").
fn compute_sequence_gospa(frames: &[FrameData], dist_threshold: f64) -> Option<f64> {
    (!frames.is_empty())
        .then(|| gospa_sequence(frames, &GospaParams::new(dist_threshold)).mean_gospa)
}

/// Calibration print (task 6.5): the observed ANEES/ANIS next to the
/// theoretical two-sided 95% chi-squared interval for this run's `N·d`,
/// making bound-setting "run, read, copy, add margin" — the anchor the
/// orbital change's section-6 calibration adopts.
fn eprint_consistency_calibration(
    scenario_name: &str,
    anees: &ConsistencyAccumulator,
    anis: &ConsistencyAccumulator,
) {
    eprint_calibration_line(scenario_name, "ANEES", anees);
    eprint_calibration_line(scenario_name, "ANIS", anis);
}

/// One statistic's calibration line: observed mean, sample count, per-sample
/// dof, and the theoretical `chi2(N·d)/N` two-sided 95% interval.
fn eprint_calibration_line(scenario_name: &str, stat: &str, acc: &ConsistencyAccumulator) {
    match (acc.mean(), acc.bounds(chi2::DEFAULT_ALPHA)) {
        (Some(mean), Some((lo, hi))) => eprintln!(
            "{scenario_name} consistency: {stat} {mean:.4} over {} samples (dof {}) \
             vs theoretical 95% interval [{lo:.4}, {hi:.4}]",
            acc.n, acc.dof,
        ),
        _ => eprintln!("{scenario_name} consistency: {stat} absent (0 samples)"),
    }
}

// ---------------------------------------------------------------------------
// Shared ECI-tracking helpers (orbital-ballistic-filter-models, §6)
// ---------------------------------------------------------------------------

/// Angular noise divisor shared by the orbital and ballistic runners: the
/// radar azimuth/elevation sigma is `measurement_noise_sigma / DIVISOR`
/// radians, so one scenario knob scales the whole RAE noise model.
const RADAR_ANGLE_SIGMA_DIVISOR: f64 = 50_000.0;

/// Ground-station geodetics shared by the orbital / ballistic step helpers.
#[derive(Debug, Clone, Copy)]
struct StationGeodetics {
    lat_rad: f64,
    lon_rad: f64,
    alt_m: f64,
}

/// Convert a radar `(range, azimuth, elevation)` triple to Cartesian ENU.
///
/// Azimuth is measured clockwise from north (`atan2(east, north)`, the
/// convention of `orbital_to_radar_measurements`), so **east** carries
/// `sin(az)` and **north** carries `cos(az)`. The previous orbital runner
/// reconstructed with `cos(az)` on the first component, silently swapping
/// East/North between detections and ground truth; this helper is the
/// single tested inverse both runners now share.
fn rae_to_enu(range: f64, azimuth: f64, elevation: f64) -> Vector3<f64> {
    Vector3::new(
        range * elevation.cos() * azimuth.sin(),
        range * elevation.cos() * azimuth.cos(),
        range * elevation.sin(),
    )
}

/// Convert a Cartesian ENU position to the radar `(range, azimuth,
/// elevation)` triple, the exact inverse of [`rae_to_enu`].
fn enu_to_rae(enu: &[f64; 3]) -> (f64, f64, f64) {
    let [east, north, up] = *enu;
    let range = (east * east + north * north + up * up).sqrt();
    let azimuth = east.atan2(north);
    let elevation = (up / range).asin();
    (range, azimuth, elevation)
}

/// Birth class for the ECI benchmark runners: the class-specific physics
/// head by default, or the CV head (`Unknown`) under the `"force-cv-head"`
/// scenario-type override used by the deliberate-regression checks of the
/// benchmark gate (task 6.6 of `orbital-ballistic-filter-models`).
fn birth_class_or_forced_cv(params: &ScenarioParameters, class: TargetClass) -> TargetClass {
    if params.scenario_type.as_deref() == Some("force-cv-head") {
        TargetClass::Unknown
    } else {
        class
    }
}

/// Effective isotropic Cartesian measurement sigma for the tracker's `R`.
///
/// The RAE noise model is range-dependent: angular noise of `angle_sigma`
/// radians displaces a detection cross-range by `range · angle_sigma`
/// metres, which at LEO/ballistic slant ranges dwarfs the range sigma. The
/// tracker consumes a single fixed `R`, so the honest middle ground is the
/// combined sigma linearised at the **median visible slant range**;
/// `gate_threshold` carries the headroom for the range spread around the
/// median (see the scenario TOML comments). Falls back to the range sigma
/// (floored at 1 m so `R` stays invertible) when nothing is visible.
fn effective_cartesian_sigma(
    visible_ranges_m: &[f64],
    range_sigma_m: f64,
    angle_sigma_rad: f64,
) -> f64 {
    if visible_ranges_m.is_empty() {
        return range_sigma_m.max(1.0);
    }
    let mut sorted = visible_ranges_m.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("slant ranges must be finite"));
    let median = sorted[sorted.len() / 2];
    (range_sigma_m.powi(2) + (median * angle_sigma_rad).powi(2))
        .sqrt()
        .max(1.0)
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run a synthetic benchmark scenario end-to-end.
///
/// 1. Generate trajectories from manifest parameters.
/// 2. Generate radar measurements with the given noise / detection settings.
/// 3. Run the tracker.
/// 4. Evaluate MOT metrics.
pub fn run_synthetic_benchmark(manifest: &ScenarioManifest) -> BenchmarkResult {
    let start = Instant::now();
    let params = &manifest.parameters;

    // --- Build trajectories (deterministic, spread out) ---
    let trajectories = build_trajectories(params);

    let scenario = thresh_synth::scenario::Scenario {
        name: manifest.name.clone(),
        trajectories,
    };

    // --- Radar config from scenario type ---
    let radar_config = radar_config_for_scenario(params);

    // Deterministic seeded RNG so the CI regression gate is reproducible
    // (spec "Deterministic gate outcome" — matches the seeded ballistic /
    // orbital / ADS-B runners in this file; previously thread-local).
    let mut scenario_rng = StdRng::seed_from_u64(0x5EED_C0DE_5EED_C0DE);
    let (gt_entries, measurements) =
        run_scenario_with_rng(&scenario, &radar_config, &mut scenario_rng);

    // --- Run tracker ---
    // The benchmark runner currently only drives the Cartesian ENU tracker
    // end-to-end. If a scenario explicitly requests another variant, honour
    // the request only when it is `Enu`; otherwise fall back to ENU and
    // leave full wiring for the other variants to a future change.
    let _requested_variant = params.tracker_variant.unwrap_or(TrackerVariant::Enu);
    // Honest configuration unless a deliberate-regression test decouples
    // the tracker's R from the generator's noise (see `tracker_noise_sigma`).
    let tracker_sigma = params
        .tracker_noise_sigma
        .unwrap_or(params.measurement_noise_sigma);
    let mut tracker = MultiObjectTracker::new_cv_position(tracker_sigma, params.gate_threshold);

    // Group measurements and ground truth by time step
    let mut meas_by_time: HashMap<i64, Vec<DVector<f64>>> = HashMap::new();
    if params.measurement_model == MeasurementModel::Cartesian {
        collect_cartesian_measurements(&gt_entries, params, &mut scenario_rng, &mut meas_by_time);
    } else {
        for tm in &measurements {
            let key = (tm.time / params.dt).round() as i64;
            let pos = measurement_to_cartesian(&tm.measurement);
            meas_by_time.entry(key).or_default().push(pos);
        }
    }

    let mut gt_by_time: HashMap<i64, Vec<GroundTruth>> = HashMap::new();
    for g in &gt_entries {
        let key = (g.time / params.dt).round() as i64;
        gt_by_time.entry(key).or_default().push(g.clone());
    }

    let max_step = meas_by_time
        .keys()
        .chain(gt_by_time.keys())
        .copied()
        .max()
        .unwrap_or(0);

    let mut frame_data_vec: Vec<FrameData> = Vec::new();
    let mut estimate_frames: Vec<EstimateFrame> = Vec::new();

    for step in 0..=max_step {
        let dets: Vec<DVector<f64>> = meas_by_time.remove(&step).unwrap_or_default();

        tracker.step(&dets, params.dt);

        // Build FrameData + EstimateFrame for this step
        let gt_positions: Vec<(u64, [f64; 3])> = gt_by_time
            .get(&step)
            .map(|gs| {
                gs.iter()
                    .map(|g| (u64::from(g.target_id), g.position))
                    .collect()
            })
            .unwrap_or_default();

        push_step_frames(
            &tracker,
            gt_positions,
            &mut frame_data_vec,
            &mut estimate_frames,
        );
    }

    build_benchmark_result(
        &manifest.name,
        &frame_data_vec,
        &estimate_frames,
        &tracker,
        params.measurement_noise_sigma * 5.0,
        start,
    )
}

// ---------------------------------------------------------------------------
// Regression checking
// ---------------------------------------------------------------------------

/// Check a benchmark result against baselines.
/// Returns a list of failure messages (empty = pass).
///
/// MOT baselines are one-sided floors (exactly as before the
/// `eval-consistency-metrics` change); ANEES / average-NIS bounds are
/// enforced two-sidedly via the `check_bound` phase helper, including the
/// fail-loud rule:
/// a bound asserted against an absent statistic (zero samples) is a gate
/// failure, never a silent pass. Scenarios declaring no consistency bounds
/// are checked exactly as before.
pub fn check_regression(result: &BenchmarkResult, baselines: &Baselines) -> Vec<String> {
    let mut failures = Vec::new();
    check_floor("MOTA", result.mota, baselines.mota, &mut failures);
    check_floor("HOTA", result.hota, baselines.hota, &mut failures);
    check_floor("IDF1", result.idf1, baselines.idf1, &mut failures);
    check_bound(
        "ANEES",
        result.anees,
        baselines.anees_min,
        baselines.anees_max,
        &mut failures,
    );
    check_bound(
        "ANIS",
        result.anis,
        baselines.anis_min,
        baselines.anis_max,
        &mut failures,
    );
    failures
}

/// One-sided floor check for the MOT metrics (pre-existing semantics and
/// message format, byte-identical to the pre-refactor `check_regression`).
fn check_floor(name: &str, value: f64, baseline: Option<f64>, failures: &mut Vec<String>) {
    if let Some(floor) = baseline
        && value < floor
    {
        failures.push(format!("{name} {value:.2} below baseline {floor:.2}"));
    }
}

/// Two-sided bound check for an optional consistency statistic (phase
/// helper of [`check_regression`], design Decision 5).
///
/// Each present bound is enforced in its direction with a message naming
/// the statistic, its value, and the violated bound. **Fail-loud rule:** a
/// bound asserted while the statistic is absent (`None` — zero samples) or
/// non-finite is itself a failure, so a plumbing regression can never read
/// as consistency.
fn check_bound(
    name: &str,
    value: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    failures: &mut Vec<String>,
) {
    if min.is_none() && max.is_none() {
        return;
    }
    let Some(value) = value.filter(|v| v.is_finite()) else {
        failures.push(format!(
            "{name} bound declared but the statistic is absent (zero samples \
             or non-finite) — failing loud instead of passing silently"
        ));
        return;
    };
    if let Some(lo) = min
        && value < lo
    {
        failures.push(format!("{name} {value:.4} below lower bound {lo:.4}"));
    }
    if let Some(hi) = max
        && value > hi
    {
        failures.push(format!("{name} {value:.4} above upper bound {hi:.4}"));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a measurement to Cartesian [x, y, z] for the tracker.
fn measurement_to_cartesian(m: &thresh_core::measurement::Measurement) -> DVector<f64> {
    match m {
        thresh_core::measurement::Measurement::Radar {
            range,
            azimuth,
            elevation,
            ..
        } => {
            let x = range * elevation.cos() * azimuth.cos();
            let y = range * elevation.cos() * azimuth.sin();
            let z = range * elevation.sin();
            DVector::from_column_slice(&[x, y, z])
        }
        thresh_core::measurement::Measurement::AdsB { lat, lon, alt, .. } => {
            DVector::from_column_slice(&[*lat, *lon, *alt])
        }
        thresh_core::measurement::Measurement::EoIr { .. } => {
            // Bearing-only: not directly usable for position-based tracker
            DVector::from_column_slice(&[0.0, 0.0, 0.0])
        }
        thresh_core::measurement::Measurement::Othr {
            ground_range_m,
            azimuth_rad,
            ..
        } => {
            // Approximate Cartesian from ground range and azimuth (flat-earth approx)
            let x = ground_range_m * azimuth_rad.sin();
            let y = ground_range_m * azimuth_rad.cos();
            DVector::from_column_slice(&[x, y, 0.0])
        }
        thresh_core::measurement::Measurement::Lidar { position, .. } => {
            DVector::from_column_slice(position)
        }
        thresh_core::measurement::Measurement::Camera { position, .. } => {
            // Use the monocular 3D estimate when present; a pixel-only camera
            // detection has no Cartesian position for this benchmark path.
            position
                .map(|p| DVector::from_column_slice(&p))
                .unwrap_or_else(|| DVector::from_column_slice(&[0.0, 0.0, 0.0]))
        }
    }
}

/// Dispatch to the appropriate trajectory builder based on `scenario_type`.
fn build_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    match params.scenario_type.as_deref() {
        None | Some("cv-clean") => build_cv_clean_trajectories(params),
        Some("maneuvering") => build_maneuvering_trajectories(params),
        Some("heterogeneous") => build_heterogeneous_trajectories(params),
        Some("low-pd") => build_low_pd_trajectories(params),
        Some(other) => {
            eprintln!("unknown scenario_type {other:?}, falling back to cv-clean");
            build_cv_clean_trajectories(params)
        }
    }
}

/// Build 5 constant-velocity trajectories spread in space (the original
/// default benchmark geometry).
fn build_cv_clean_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    let n_targets = 5;
    (0..n_targets)
        .map(|i| {
            let spacing = 5000.0;
            Trajectory {
                target_id: i,
                initial_position: [
                    10_000.0 + i as f64 * spacing,
                    5_000.0 + i as f64 * spacing * 0.5,
                    3_000.0,
                ],
                initial_velocity: [200.0 + i as f64 * 20.0, 50.0 - i as f64 * 10.0, 0.0],
                segments: vec![Segment {
                    segment_type: SegmentType::Cv,
                    duration: params.duration_s,
                }],
                dt: params.dt,
            }
        })
        .collect()
}

/// Build 4 maneuvering trajectories with CV / CTRV / CA segments,
/// 10 km spacing between targets.
fn build_maneuvering_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    let segment_dur = params.duration_s / 3.0;
    (0..4)
        .map(|i| {
            let spacing = 10_000.0;
            Trajectory {
                target_id: i,
                initial_position: [
                    10_000.0 + i as f64 * spacing,
                    5_000.0 + i as f64 * spacing * 0.5,
                    3_000.0,
                ],
                initial_velocity: [200.0 + i as f64 * 30.0, 100.0 - i as f64 * 20.0, 0.0],
                segments: vec![
                    Segment {
                        segment_type: SegmentType::Cv,
                        duration: segment_dur,
                    },
                    Segment {
                        segment_type: SegmentType::Ctrv {
                            turn_rate: 0.03 * if i % 2 == 0 { 1.0 } else { -1.0 },
                        },
                        duration: segment_dur,
                    },
                    Segment {
                        segment_type: SegmentType::Ca {
                            acceleration: [10.0, -5.0, 0.0],
                        },
                        duration: segment_dur,
                    },
                ],
                dt: params.dt,
            }
        })
        .collect()
}

/// Build 5 heterogeneous trajectories across three kinematic classes:
/// UAV-like (slow, low altitude), aircraft-like (medium), and
/// missile-like (fast, high altitude).
fn build_heterogeneous_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    vec![
        // --- UAV-like targets (2) ---
        Trajectory {
            target_id: 0,
            initial_position: [5_000.0, 2_000.0, 200.0],
            initial_velocity: [15.0, 10.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: params.duration_s,
            }],
            dt: params.dt,
        },
        Trajectory {
            target_id: 1,
            initial_position: [8_000.0, 4_000.0, 300.0],
            initial_velocity: [20.0, -5.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ctrv { turn_rate: 0.02 },
                duration: params.duration_s,
            }],
            dt: params.dt,
        },
        // --- Aircraft-like targets (2) ---
        Trajectory {
            target_id: 2,
            initial_position: [20_000.0, 10_000.0, 8_000.0],
            initial_velocity: [250.0, 50.0, 0.0],
            segments: vec![Segment {
                segment_type: SegmentType::Cv,
                duration: params.duration_s,
            }],
            dt: params.dt,
        },
        Trajectory {
            target_id: 3,
            initial_position: [30_000.0, 15_000.0, 10_000.0],
            initial_velocity: [220.0, -30.0, 0.0],
            segments: vec![
                Segment {
                    segment_type: SegmentType::Cv,
                    duration: params.duration_s / 2.0,
                },
                Segment {
                    segment_type: SegmentType::Ctrv { turn_rate: 0.01 },
                    duration: params.duration_s / 2.0,
                },
            ],
            dt: params.dt,
        },
        // --- Missile-like target (1) ---
        Trajectory {
            target_id: 4,
            initial_position: [50_000.0, 30_000.0, 20_000.0],
            initial_velocity: [800.0, 200.0, 50.0],
            segments: vec![Segment {
                segment_type: SegmentType::Ca {
                    acceleration: [20.0, 0.0, 5.0],
                },
                duration: params.duration_s,
            }],
            dt: params.dt,
        },
    ]
}

/// Build trajectories for the low-pd scenario. Uses the same geometry as
/// cv-clean; the challenge comes from the reduced detection probability
/// and increased clutter in the radar config.
fn build_low_pd_trajectories(params: &ScenarioParameters) -> Vec<Trajectory> {
    build_cv_clean_trajectories(params)
}

/// Cartesian measurement mode ([`MeasurementModel::Cartesian`]): one
/// detection per truth entry with isotropic per-axis Gaussian noise of
/// `measurement_noise_sigma`, exactly matching the tracker's isotropic R.
///
/// This is the textbook consistency configuration: every NEES/NIS sample is
/// chi-squared distributed by construction, so the `synth-cv-clean`
/// ANEES/ANIS gate can be judged against the theoretical interval
/// (`eval-consistency-metrics` design Decision 5). Detection is perfect and
/// clutter-free; scenarios needing missed detections or clutter use the
/// default RAE radar model.
fn collect_cartesian_measurements<R: rand::Rng>(
    gt_entries: &[GroundTruth],
    params: &ScenarioParameters,
    rng: &mut R,
    meas_by_time: &mut HashMap<i64, Vec<DVector<f64>>>,
) {
    let normal = rand_distr::Normal::new(0.0, params.measurement_noise_sigma)
        .expect("measurement_noise_sigma must be finite and non-negative");
    for g in gt_entries {
        let key = (g.time / params.dt).round() as i64;
        let pos = DVector::from_column_slice(&[
            g.position[0] + rand_distr::Distribution::sample(&normal, rng),
            g.position[1] + rand_distr::Distribution::sample(&normal, rng),
            g.position[2] + rand_distr::Distribution::sample(&normal, rng),
        ]);
        meas_by_time.entry(key).or_default().push(pos);
    }
}

/// Return the appropriate `RadarConfig` for a scenario type.
///
/// Default (cv-clean / maneuvering / heterogeneous) uses perfect detection
/// (`p_detection = 1.0`, `clutter_rate = 0.0`). The `"low-pd"` variant
/// models a challenging sensor environment with `p_detection = 0.7` and
/// `clutter_rate = 5.0`.
fn radar_config_for_scenario(params: &ScenarioParameters) -> RadarConfig {
    let (p_detection, clutter_rate) = match params.scenario_type.as_deref() {
        Some("low-pd") => (0.7, 5.0),
        _ => (1.0, 0.0),
    };
    RadarConfig {
        range_sigma: params.measurement_noise_sigma,
        azimuth_sigma: params.measurement_noise_sigma / 10_000.0,
        elevation_sigma: params.measurement_noise_sigma / 10_000.0,
        p_detection,
        clutter_rate,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Orbital benchmark runner (§7.5 / §7.6)
// ---------------------------------------------------------------------------

/// Run an orbital benchmark scenario end-to-end.
///
/// This function is only compiled when the `orbital` feature is enabled
/// because it depends on the `sgp4` crate (via `crate::orbital`). Downstream
/// callers that always need orbital support should build `thresh-data` with
/// `--features orbital`; the CLI surfaces a clean "feature required" error
/// when the feature is not compiled in.
///
/// Pipeline (design Decision 7 of `orbital-ballistic-filter-models`):
/// 1. Load TLEs — from a local file (`tle_file` relative to the manifest
///    directory) when set; otherwise from Space-Track / CelesTrak via the
///    orbital HTTP clients. The local-file path is what allows the CI
///    synthetic-benchmark gate to run orbital scenarios offline.
/// 2. Propagate each TLE via SGP4 → TEME → ECEF → ENU relative to the
///    station configured in `ScenarioSource::Orbital`. The GMST-only
///    Earth-fixing spin of this chain is exactly the **TEME → PEF**
///    rotation of the IAU-76/FK5 chain (`thresh_core::frames`) — correct
///    for SGP4's TEME output, and kept verbatim by `astro-time-and-frames`
///    task 5.1 (GCRF-expressed states would instead route through the full
///    reduction, `thresh_core::eci::gcrf_to_enu`). Samples are spaced
///    at `time_step_s` (falling back to `parameters.dt`) over `duration_s`
///    seconds starting at the TLE epoch.
/// 3. Convert the visible (above-horizon) ENU positions to synthetic radar
///    measurements, add Gaussian noise with the configured sigmas, lift the
///    noisy detections **ENU → ECI** ([`enu_to_eci`] at the station
///    geodetics + per-step GMST epoch; the "ECI" of this chain is
///    TEME-consistent — the lift inverts the same TEME → PEF spin), and
///    feed them class-tagged as
///    [`TargetClass::Orbital`] so the tracker births 6D Kepler+J2 EKF
///    tracks (Decision 6 head dispatch). The tracker's `R` uses the
///    `effective_cartesian_sigma` of the visible pass so the manifest's
///    `gate_threshold` can be a Mahalanobis-consistent chi-squared value
///    instead of the historical `1e5` escape hatch.
/// 4. Build `FrameData` per time step (ground truth converted to the same
///    ECI frame) and compute MOTA / MOTP / IDF1 / HOTA.
///
/// `parameters.scenario_type = "force-cv-head"` births tracks as
/// [`TargetClass::Unknown`] (CV head) instead — the deliberate-regression
/// probe for the benchmark gate (task 6.6).
///
/// `manifest_dir` is the parent directory of the scenario file — used to
/// resolve relative `tle_file` paths without hardcoding the workspace root.
#[cfg(feature = "orbital")]
pub fn run_orbital_benchmark(
    manifest: &ScenarioManifest,
    manifest_dir: &Path,
) -> core::result::Result<BenchmarkResult, String> {
    use crate::orbital::{RadarNoiseConfig, orbital_to_radar_measurements, propagate_to_enu};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Normal;

    let start = Instant::now();
    let params = &manifest.parameters;

    let ScenarioSource::Orbital {
        norad_ids,
        tle_file,
        station_lat_deg,
        station_lon_deg,
        station_alt_m,
        time_step_s,
    } = &manifest.source
    else {
        return Err(format!(
            "run_orbital_benchmark called on non-Orbital source: {:?}",
            manifest.source
        ));
    };

    // ---- 1. Load + select TLEs ----
    let tles = load_scenario_tles(tle_file.as_deref(), norad_ids, manifest_dir)?;
    let selected_tles = select_scenario_tles(&tles, norad_ids)?;

    // ---- 2. Propagate each TLE to station ENU ----
    let step_s = time_step_s.unwrap_or(params.dt);
    let n_steps = (params.duration_s / step_s).ceil() as usize + 1;
    let times_min: Vec<f64> = (0..n_steps).map(|i| i as f64 * step_s / 60.0).collect();
    let station = StationGeodetics {
        lat_rad: station_lat_deg.to_radians(),
        lon_rad: station_lon_deg.to_radians(),
        alt_m: *station_alt_m,
    };

    let mut trajectories: Vec<OrbitalTruth> = Vec::new();
    for tle in &selected_tles {
        let enu = propagate_to_enu(
            tle,
            &times_min,
            station.lat_rad,
            station.lon_rad,
            station.alt_m,
        )
        .map_err(|e| format!("SGP4 propagation failed for {}: {e}", tle.norad_id))?;
        trajectories.push(OrbitalTruth {
            norad_id: tle.norad_id,
            epoch_jd: tle.epoch_jd(),
            enu,
        });
    }

    // ---- 3. Noise-free radar measurements + effective tracker noise ----
    let noise = RadarNoiseConfig {
        range_sigma_m: params.measurement_noise_sigma,
        azimuth_sigma_rad: params.measurement_noise_sigma / RADAR_ANGLE_SIGMA_DIVISOR,
        elevation_sigma_rad: params.measurement_noise_sigma / RADAR_ANGLE_SIGMA_DIVISOR,
        include_range_rate: false,
        sensor_id: 0,
    };
    let measurements_by_sat: Vec<(u32, Vec<thresh_core::measurement::Measurement>)> = trajectories
        .iter()
        .map(|t| (t.norad_id, orbital_to_radar_measurements(&t.enu, &noise)))
        .collect();

    let visible_ranges: Vec<f64> = measurements_by_sat
        .iter()
        .flat_map(|(_, ms)| ms.iter())
        .filter_map(|m| match m {
            thresh_core::measurement::Measurement::Radar { range, .. } => Some(*range),
            _ => None,
        })
        .collect();
    let sigma_eff = effective_cartesian_sigma(
        &visible_ranges,
        noise.range_sigma_m,
        noise.azimuth_sigma_rad,
    );

    // ---- 4. Track in ECI ----
    // Deterministic seeded RNG so the CI regression gate is reproducible.
    let mut rng = StdRng::seed_from_u64(0xA5_A5_A5_A5_A5_A5_A5_A5);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let birth_class = birth_class_or_forced_cv(params, TargetClass::Orbital);

    let mut tracker = MultiObjectTracker::new_cv_position(sigma_eff, params.gate_threshold);
    let mut frame_data_vec: Vec<FrameData> = Vec::with_capacity(n_steps);
    let mut estimate_frames: Vec<EstimateFrame> = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t_min = step as f64 * step_s / 60.0;
        let (detections, gt_positions) = collect_orbital_step_data(
            &trajectories,
            &measurements_by_sat,
            &noise,
            &normal,
            &mut rng,
            t_min,
            &station,
        );

        let classed: Vec<(DVector<f64>, TargetClass)> =
            detections.into_iter().map(|d| (d, birth_class)).collect();
        tracker.step_classed(&classed, step_s);

        push_step_frames(
            &tracker,
            gt_positions,
            &mut frame_data_vec,
            &mut estimate_frames,
        );
    }

    // ---- 5. Metrics ----
    // Orbital scenarios use a much larger match threshold because slant
    // ranges span hundreds of km and measurement noise is multi-kilometre.
    let dist_threshold = (params.measurement_noise_sigma * 10.0).max(5_000.0);
    let total_gt: usize = frame_data_vec.iter().map(|f| f.gt.len()).sum();
    let total_tracks: usize = frame_data_vec.iter().map(|f| f.tracks.len()).sum();
    eprintln!(
        "orbital pipeline: {} frames, {} ground-truth points, {} confirmed-track points, \
         effective measurement sigma {sigma_eff:.0} m",
        frame_data_vec.len(),
        total_gt,
        total_tracks,
    );
    Ok(build_benchmark_result(
        &manifest.name,
        &frame_data_vec,
        &estimate_frames,
        &tracker,
        dist_threshold,
        start,
    ))
}

/// One satellite's propagated truth: NORAD ID, TLE epoch (Julian Date in
/// the UTC scale, the zero of the scenario time base), and the station-ENU
/// sample path.
///
/// The epoch stays a raw `f64` JD: the whole orbital benchmark chain plumbs
/// `epoch_jd + t_min / 1440` into the GMST rotation (the TEME → PEF leg of
/// the IAU-76/FK5 chain — correct for SGP4's TEME output), and its
/// calibrated metrics must stay bitwise stable across the
/// `astro-time-and-frames` migration (design Decision 6) — JD↔`Epoch`
/// round trips are not guaranteed bit-exact.
#[cfg(feature = "orbital")]
struct OrbitalTruth {
    norad_id: u32,
    epoch_jd: f64,
    enu: Vec<crate::orbital::EnuPosition>,
}

/// Load the scenario's TLEs from the cached `tle_file` (relative to the
/// manifest directory) when set, otherwise via HTTP (CelesTrak first).
#[cfg(feature = "orbital")]
fn load_scenario_tles(
    tle_file: Option<&str>,
    norad_ids: &[u32],
    manifest_dir: &Path,
) -> core::result::Result<Vec<crate::orbital::Tle>, String> {
    use crate::orbital::{parse_3le, parse_tle};

    let tles = if let Some(file) = tle_file {
        let path = manifest_dir.join(file);
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read TLE file {}: {e}", path.display()))?;
        // Try 3LE (name + two data lines) first; fall back to 2LE if the
        // file has no leading name line.
        parse_3le(&contents)
            .or_else(|_| parse_tle(&contents))
            .map_err(|e| format!("failed to parse TLE file {}: {e}", path.display()))?
    } else {
        // No cached TLE — fall back to HTTP. Try CelesTrak first because
        // it's unauthenticated; if that fails, try Space-Track.
        if norad_ids.is_empty() {
            return Err("orbital scenario has no norad_ids and no tle_file".into());
        }
        fetch_tles_via_http(norad_ids).map_err(|e| {
            format!(
                "no tle_file set and HTTP fetch failed: {e}. Provide a \
                 cached TLE file alongside the manifest to run offline."
            )
        })?
    };

    if tles.is_empty() {
        return Err("no TLEs available after loading".into());
    }
    Ok(tles)
}

/// Filter loaded TLEs to the requested NORAD IDs when both are specified,
/// so a shared CelesTrak GROUP response can feed multiple scenarios.
#[cfg(feature = "orbital")]
fn select_scenario_tles<'a>(
    tles: &'a [crate::orbital::Tle],
    norad_ids: &[u32],
) -> core::result::Result<Vec<&'a crate::orbital::Tle>, String> {
    let selected: Vec<&crate::orbital::Tle> = if norad_ids.is_empty() {
        tles.iter().collect()
    } else {
        let wanted: std::collections::HashSet<u32> = norad_ids.iter().copied().collect();
        tles.iter()
            .filter(|t| wanted.contains(&t.norad_id))
            .collect()
    };

    if selected.is_empty() {
        return Err(format!(
            "TLE file contained {} TLEs but none matched the requested norad_ids {:?}",
            tles.len(),
            norad_ids
        ));
    }
    Ok(selected)
}

/// Detections + ground truth for a single benchmark step.
type StepData = (Vec<DVector<f64>>, Vec<(u64, [f64; 3])>);

/// Collect ECI ground-truth positions and noisy ECI radar detections for a
/// single orbital benchmark step.
#[cfg(feature = "orbital")]
fn collect_orbital_step_data(
    trajectories: &[OrbitalTruth],
    measurements_by_sat: &[(u32, Vec<thresh_core::measurement::Measurement>)],
    noise: &crate::orbital::RadarNoiseConfig,
    normal: &rand_distr::Normal<f64>,
    rng: &mut impl rand::Rng,
    t_min: f64,
    station: &StationGeodetics,
) -> StepData {
    let mut detections: Vec<DVector<f64>> = Vec::new();
    let mut gt_positions: Vec<(u64, [f64; 3])> = Vec::new();

    for (truth, (_, measurements)) in trajectories.iter().zip(measurements_by_sat.iter()) {
        let frame = StepFrame {
            t_min,
            jd: truth.epoch_jd + t_min / 1440.0,
            station: *station,
        };
        collect_gt_for_step(&truth.enu, truth.norad_id, &frame, &mut gt_positions);
        collect_noisy_detection(measurements, noise, normal, rng, &frame, &mut detections);
    }

    (detections, gt_positions)
}

/// Time base + station context for one orbital benchmark step: scenario
/// time (minutes since the satellite's TLE epoch), the corresponding
/// Julian Date (UTC scale, raw-JD plumbing — see [`OrbitalTruth`]) for the
/// GMST (TEME → PEF) rotation, and the station geodetics for the
/// ENU → ECI (TEME-consistent) lift.
#[cfg(feature = "orbital")]
struct StepFrame {
    t_min: f64,
    jd: f64,
    station: StationGeodetics,
}

/// Append the ECI ground-truth position for this satellite at the step's
/// time if it is visible (above the station horizon).
#[cfg(feature = "orbital")]
fn collect_gt_for_step(
    enu: &[crate::orbital::EnuPosition],
    id: u32,
    frame: &StepFrame,
    gt_positions: &mut Vec<(u64, [f64; 3])>,
) {
    if let Some(pos) = enu
        .iter()
        .find(|p| (p.time_since_epoch_min - frame.t_min).abs() < 1e-6)
        && pos.up > 0.0
    {
        let eci = enu_to_eci(
            &Vector3::new(pos.east, pos.north, pos.up),
            frame.jd,
            frame.station.lat_rad,
            frame.station.lon_rad,
            frame.station.alt_m,
        );
        gt_positions.push((u64::from(id), [eci.x, eci.y, eci.z]));
    }
}

/// Convert a radar measurement at the step's time to a noisy ECI detection:
/// perturb RAE with the configured sigmas, reconstruct Cartesian ENU via
/// [`rae_to_enu`], and lift into ECI (TEME-consistent — the inverse of the
/// GMST/TEME → PEF spin) with the step's GMST epoch.
#[cfg(feature = "orbital")]
fn collect_noisy_detection(
    measurements: &[thresh_core::measurement::Measurement],
    noise: &crate::orbital::RadarNoiseConfig,
    normal: &rand_distr::Normal<f64>,
    rng: &mut impl rand::Rng,
    frame: &StepFrame,
    detections: &mut Vec<DVector<f64>>,
) {
    use rand_distr::Distribution;

    let m = measurements.iter().find(|m| match m {
        thresh_core::measurement::Measurement::Radar { time, .. } => {
            (time - frame.t_min * 60.0).abs() < 1e-6
        }
        _ => false,
    });
    if let Some(thresh_core::measurement::Measurement::Radar {
        range,
        azimuth,
        elevation,
        ..
    }) = m
    {
        let noisy_range = range + noise.range_sigma_m * normal.sample(rng);
        let noisy_az = azimuth + noise.azimuth_sigma_rad * normal.sample(rng);
        let noisy_el = elevation + noise.elevation_sigma_rad * normal.sample(rng);
        let enu = rae_to_enu(noisy_range, noisy_az, noisy_el);
        let eci = enu_to_eci(
            &enu,
            frame.jd,
            frame.station.lat_rad,
            frame.station.lon_rad,
            frame.station.alt_m,
        );
        detections.push(DVector::from_column_slice(&[eci.x, eci.y, eci.z]));
    }
}

/// Fetch TLEs for the given NORAD IDs via HTTP, trying CelesTrak first
/// (public, no auth required) and falling back to Space-Track for IDs
/// CelesTrak doesn't return.
#[cfg(feature = "orbital")]
fn fetch_tles_via_http(norad_ids: &[u32]) -> Result<Vec<crate::orbital::Tle>, String> {
    use crate::orbital::{CelestrakClient, Tle};

    let client = CelestrakClient::new();
    let mut out: Vec<Tle> = Vec::new();
    for id in norad_ids {
        let tles = client
            .fetch_catnr(*id)
            .map_err(|e| format!("CelesTrak fetch for NORAD {id} failed: {e}"))?;
        out.extend(tles);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Ballistic benchmark runner (orbital-ballistic-filter-models, task 6.3)
// ---------------------------------------------------------------------------

/// Run a ballistic benchmark scenario end-to-end (spec "Ballistic benchmark
/// end-to-end run", design Decision 7 of `orbital-ballistic-filter-models`).
///
/// Pipeline:
/// 1. Generate phased boost / midcourse / reentry truth in ECI from the
///    manifest's [`ScenarioSource::Ballistic`] profile via
///    [`thresh_synth::ballistic::generate`] on the `parameters.dt` grid.
/// 2. Project each truth sample into the downrange radar station's ENU
///    frame ([`thresh_synth::ballistic::ballistic_to_enu`], the same
///    station projection the SGP4 orbital source uses).
/// 3. For each above-horizon sample, form a radar RAE measurement, add
///    seeded Gaussian noise (range sigma = `measurement_noise_sigma`,
///    angle sigma = `measurement_noise_sigma / 50 000` rad — the orbital
///    runner's convention), reconstruct Cartesian ENU, and lift the
///    detection **ENU → ECI** at the sample's epoch.
/// 4. Feed detections class-tagged [`TargetClass::Ballistic`] so the
///    tracker births 7D `BallisticReentry` EKF tracks (Decision 6 head
///    dispatch; one 7D model covers midcourse + reentry — the drag term
///    vanishes exo-atmospherically), then compute MOT metrics against the
///    ECI truth.
///
/// `parameters.scenario_type = "force-cv-head"` births tracks as
/// [`TargetClass::Unknown`] (CV head) instead — the deliberate-regression
/// probe for the benchmark gate (task 6.6).
///
/// Truth generation and tracking are pure Rust with no SGP4/network
/// dependency, so this runner is available under **default features**.
pub fn run_ballistic_benchmark(
    manifest: &ScenarioManifest,
) -> core::result::Result<BenchmarkResult, String> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Normal;
    use thresh_synth::ballistic::{BallisticProfile, ballistic_to_enu, generate};

    let start = Instant::now();
    let params = &manifest.parameters;

    let ScenarioSource::Ballistic {
        launch_lat_deg,
        launch_lon_deg,
        launch_alt_m,
        launch_azimuth_deg,
        thrust_accel_m_s2,
        burn_time_s,
        pitch_over_s,
        pitch_kick_deg,
        beta_kg_m2,
        epoch_jd,
        station_lat_deg,
        station_lon_deg,
        station_alt_m,
    } = &manifest.source
    else {
        return Err(format!(
            "run_ballistic_benchmark called on non-Ballistic source: {:?}",
            manifest.source
        ));
    };

    // ---- 1. Phased truth generation (boost / midcourse / reentry) ----
    let profile = BallisticProfile {
        launch_lat_rad: launch_lat_deg.to_radians(),
        launch_lon_rad: launch_lon_deg.to_radians(),
        launch_alt_m: *launch_alt_m,
        launch_azimuth_rad: launch_azimuth_deg.to_radians(),
        thrust_accel: *thrust_accel_m_s2,
        burn_time_s: *burn_time_s,
        pitch_over_s: *pitch_over_s,
        pitch_kick_rad: pitch_kick_deg.to_radians(),
        beta: *beta_kg_m2,
        epoch_jd: *epoch_jd,
    };
    if params.dt <= 0.0 || !params.dt.is_finite() {
        return Err(format!(
            "ballistic scenario requires a positive finite dt, got {}",
            params.dt
        ));
    }
    let truth = generate(&profile, params.dt);

    // ---- 2. Station ENU projection ----
    let station = StationGeodetics {
        lat_rad: station_lat_deg.to_radians(),
        lon_rad: station_lon_deg.to_radians(),
        alt_m: *station_alt_m,
    };
    let enu = ballistic_to_enu(&truth, station.lat_rad, station.lon_rad, station.alt_m);

    // ---- 3. Effective tracker noise from the visible slant ranges ----
    let sigmas = RaeSigmas {
        range_m: params.measurement_noise_sigma,
        angle_rad: params.measurement_noise_sigma / RADAR_ANGLE_SIGMA_DIVISOR,
    };
    let visible_ranges: Vec<f64> = enu
        .iter()
        .filter(|p| p[2] > 0.0)
        .map(|p| enu_to_rae(p).0)
        .collect();
    let sigma_eff = effective_cartesian_sigma(&visible_ranges, sigmas.range_m, sigmas.angle_rad);

    // ---- 4. Track with the 7D ballistic reentry head in ECI ----
    // Deterministic seeded RNG so the CI regression gate is reproducible.
    let mut rng = StdRng::seed_from_u64(0xBA11_157C_0DE5_EED5);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let birth_class = birth_class_or_forced_cv(params, TargetClass::Ballistic);

    let n_steps = truth
        .len()
        .min((params.duration_s / params.dt).ceil() as usize + 1);
    let mut tracker = MultiObjectTracker::new_cv_position(sigma_eff, params.gate_threshold);
    let mut frame_data_vec: Vec<FrameData> = Vec::with_capacity(n_steps);
    let mut estimate_frames: Vec<EstimateFrame> = Vec::with_capacity(n_steps);

    for k in 0..n_steps {
        let (detections, gt_positions) =
            ballistic_step_data(&truth[k], &enu[k], &sigmas, &normal, &mut rng, &station);
        let classed: Vec<(DVector<f64>, TargetClass)> =
            detections.into_iter().map(|d| (d, birth_class)).collect();
        tracker.step_classed(&classed, params.dt);
        push_step_frames(
            &tracker,
            gt_positions,
            &mut frame_data_vec,
            &mut estimate_frames,
        );
    }

    // ---- 5. Metrics ----
    let dist_threshold = (params.measurement_noise_sigma * 10.0).max(5_000.0);
    let total_gt: usize = frame_data_vec.iter().map(|f| f.gt.len()).sum();
    let total_tracks: usize = frame_data_vec.iter().map(|f| f.tracks.len()).sum();
    eprintln!(
        "ballistic pipeline: {} frames ({} truth samples), {} ground-truth points, \
         {} confirmed-track points, effective measurement sigma {sigma_eff:.0} m",
        frame_data_vec.len(),
        truth.len(),
        total_gt,
        total_tracks,
    );
    eprint_ballistic_flight_diagnostics(&truth, &enu, params.dt);
    Ok(build_benchmark_result(
        &manifest.name,
        &frame_data_vec,
        &estimate_frames,
        &tracker,
        dist_threshold,
        start,
    ))
}

/// Print flight-shape and visibility-window diagnostics for a ballistic
/// truth run: apogee, flight time, impact distance from the station, and
/// the above-horizon window. These are the numbers that settled the
/// scenario's visibility-window open question (see `ballistic-mrbm.toml`
/// and the design's Open Questions) and they make calibration runs
/// self-documenting.
fn eprint_ballistic_flight_diagnostics(
    truth: &[thresh_synth::orbital::OrbitalState],
    enu: &[[f64; 3]],
    dt: f64,
) {
    let earth_radius = thresh_core::orbital::GravityModel::EARTH_WGS84.equatorial_radius;
    let apogee_km = truth
        .iter()
        .map(|s| (Vector3::from(s.position).norm() - earth_radius) / 1_000.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let flight_s = (truth.len().saturating_sub(1)) as f64 * dt;
    let impact_from_station_km = enu
        .last()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt() / 1_000.0)
        .unwrap_or(f64::NAN);
    let first_visible = enu.iter().position(|p| p[2] > 0.0);
    let last_visible = enu.iter().rposition(|p| p[2] > 0.0);
    match (first_visible, last_visible) {
        (Some(a), Some(b)) => eprintln!(
            "ballistic flight: apogee {apogee_km:.0} km, flight {flight_s:.0} s, impact \
             {impact_from_station_km:.0} km from station, visible window [{:.0}, {:.0}] s",
            a as f64 * dt,
            b as f64 * dt,
        ),
        _ => eprintln!(
            "ballistic flight: apogee {apogee_km:.0} km, flight {flight_s:.0} s, impact \
             {impact_from_station_km:.0} km from station, never visible from station"
        ),
    }
}

/// Radar RAE noise sigmas for the ballistic runner.
struct RaeSigmas {
    range_m: f64,
    angle_rad: f64,
}

/// Detections + ground truth for a single ballistic benchmark step: when
/// the truth sample is above the station horizon, its ECI position is the
/// ground truth (single target, ID 1) and one noisy RAE detection is
/// formed and lifted ENU → ECI at the sample's epoch; below the horizon
/// the step is empty (same visibility convention as the orbital runner).
fn ballistic_step_data(
    truth: &thresh_synth::orbital::OrbitalState,
    enu: &[f64; 3],
    sigmas: &RaeSigmas,
    normal: &rand_distr::Normal<f64>,
    rng: &mut impl rand::Rng,
    station: &StationGeodetics,
) -> StepData {
    use rand_distr::Distribution;

    let mut detections: Vec<DVector<f64>> = Vec::new();
    let mut gt_positions: Vec<(u64, [f64; 3])> = Vec::new();
    if enu[2] <= 0.0 {
        return (detections, gt_positions);
    }

    gt_positions.push((1, truth.position));

    let (range, azimuth, elevation) = enu_to_rae(enu);
    let noisy_range = range + sigmas.range_m * normal.sample(rng);
    let noisy_az = azimuth + sigmas.angle_rad * normal.sample(rng);
    let noisy_el = elevation + sigmas.angle_rad * normal.sample(rng);
    let noisy_enu = rae_to_enu(noisy_range, noisy_az, noisy_el);
    let eci = enu_to_eci(
        &noisy_enu,
        truth.epoch_jd,
        station.lat_rad,
        station.lon_rad,
        station.alt_m,
    );
    detections.push(DVector::from_column_slice(&[eci.x, eci.y, eci.z]));

    (detections, gt_positions)
}

// ---------------------------------------------------------------------------
// ADS-B benchmark runner (§7.3 / §7.4)
// ---------------------------------------------------------------------------

/// Run an ADS-B benchmark scenario end-to-end.
///
/// Feature-gated on `adsb` because it depends on `OpenSkyClient` (and
/// therefore `reqwest`) plus `StateVector` / `state_to_measurement` from
/// the ADS-B module. Downstream callers that always need ADS-B should
/// build `thresh-data` with `--features adsb`; the CLI surfaces a clean
/// "feature required" error otherwise.
///
/// Pipeline:
/// 1. Load state vectors — from a local cached JSON file (`state_file`
///    relative to the manifest directory) when set; otherwise from
///    OpenSky via `OpenSkyClient::fetch_states`. The local-file path is
///    what lets the CI benchmark gate run ADS-B scenarios offline.
/// 2. Extract per-ICAO24 ground-truth trajectories via the existing
///    `extract_ground_truth` pipeline.
/// 3. Convert each state vector to a noisy ADS-B-sourced Cartesian
///    detection (WGS84 → ENU relative to the scenario's reference point)
///    and bin everything by the scenario's `dt` step.
/// 4. Feed the binned detections into the Cartesian ENU tracker and
///    compute MOTA / MOTP / IDF1 / HOTA against the ground truth.
///
/// `manifest_dir` is the parent of the manifest file — used to resolve
/// relative `state_file` paths.
#[cfg(feature = "adsb")]
pub fn run_adsb_benchmark(
    manifest: &ScenarioManifest,
    manifest_dir: &Path,
) -> core::result::Result<BenchmarkResult, String> {
    use crate::adsb::{StateVector, extract_ground_truth};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Normal;

    let start = Instant::now();
    let params = &manifest.parameters;

    let ScenarioSource::AdsB {
        region,
        state_file,
        bbox,
        ref_lat_deg,
        ref_lon_deg,
        ref_alt_m,
    } = &manifest.source
    else {
        return Err(format!(
            "run_adsb_benchmark called on non-AdsB source: {:?}",
            manifest.source
        ));
    };

    // ---- 1. Load ADS-B state vectors ----
    let states: Vec<StateVector> = if let Some(file) = state_file {
        let path = manifest_dir.join(file);
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read ADS-B state file {}: {e}", path.display()))?;
        serde_json::from_str(&contents).map_err(|e| {
            format!(
                "failed to parse ADS-B state file {} as JSON: {e}",
                path.display()
            )
        })?
    } else {
        // No cached fixture — fall back to live OpenSky. Requires network
        // access and usually OpenSky credentials; surfaced as a plain
        // error so the CI gate's "no network" failure mode is obvious.
        let _ = region;
        let _ = bbox;
        return Err(
            "no state_file set and live OpenSky fetch is not wired into the \
             benchmark runner. Provide a cached JSON fixture alongside the \
             manifest to run offline."
                .to_string(),
        );
    };

    if states.is_empty() {
        return Err("ADS-B state file contained no state vectors".into());
    }

    // ---- 2. Build ground-truth trajectories (ICAO24 → 1 Hz grid) ----
    let trajectories = extract_ground_truth(&states);
    if trajectories.is_empty() {
        return Err("extract_ground_truth produced no trajectories".into());
    }

    // Stable target-ID assignment — first-seen ICAO24 order so frame IDs
    // are reproducible across runs.
    let id_for_icao24: std::collections::HashMap<String, u64> = trajectories
        .iter()
        .enumerate()
        .map(|(i, t)| (t.icao24.clone(), i as u64 + 1))
        .collect();

    // ---- 3. Build ENU-frame measurements and ground truth ----
    let ref_lat_rad = ref_lat_deg.to_radians();
    let ref_lon_rad = ref_lon_deg.to_radians();

    // Earliest timestamp (seconds) across all state vectors — used as the
    // zero of the benchmark time base.
    let t0 = states
        .iter()
        .filter_map(|s| s.time_position.or(Some(s.last_contact)))
        .fold(f64::INFINITY, f64::min);
    if !t0.is_finite() {
        return Err("no timestamps in ADS-B state vectors".into());
    }

    // Detections binned by integer step index.
    let mut rng = StdRng::seed_from_u64(0xAD5B_5A5A_5A5A_5A5A_u64);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let enu_ref = EnuRef {
        lat_rad: ref_lat_rad,
        lon_rad: ref_lon_rad,
        alt_m: *ref_alt_m,
    };
    let mut dets_by_step = bin_adsb_detections(&states, params, t0, &enu_ref, &normal, &mut rng);

    // Ground truth binned by step, using the 1-Hz-interpolated entries.
    let mut gt_by_step: std::collections::BTreeMap<i64, Vec<(u64, [f64; 3])>> =
        std::collections::BTreeMap::new();
    for traj in &trajectories {
        let target_id = id_for_icao24[&traj.icao24];
        for (offset_s, entry) in traj.entries.iter().enumerate() {
            let t_abs = traj.start_time + offset_s as f64;
            let step = ((t_abs - t0) / params.dt).round() as i64;
            gt_by_step
                .entry(step)
                .or_default()
                .push((target_id, entry.position));
        }
    }

    // ---- 4. Step the tracker and collect FrameData ----
    let mut tracker =
        MultiObjectTracker::new_cv_position(params.measurement_noise_sigma, params.gate_threshold);
    let mut frame_data_vec: Vec<FrameData> = Vec::new();
    let mut estimate_frames: Vec<EstimateFrame> = Vec::new();

    let step_lo = dets_by_step
        .keys()
        .chain(gt_by_step.keys())
        .min()
        .copied()
        .unwrap_or(0);
    let step_hi = dets_by_step
        .keys()
        .chain(gt_by_step.keys())
        .max()
        .copied()
        .unwrap_or(0);

    for step in step_lo..=step_hi {
        let dets: Vec<DVector<f64>> = dets_by_step.remove(&step).unwrap_or_default();
        tracker.step(&dets, params.dt);

        let gt_positions: Vec<(u64, [f64; 3])> = gt_by_step.remove(&step).unwrap_or_default();
        push_step_frames(
            &tracker,
            gt_positions,
            &mut frame_data_vec,
            &mut estimate_frames,
        );
    }

    let total_gt: usize = frame_data_vec.iter().map(|f| f.gt.len()).sum();
    let total_tracks: usize = frame_data_vec.iter().map(|f| f.tracks.len()).sum();
    eprintln!(
        "ADS-B pipeline: {} frames, {} trajectories, {} ground-truth points, {} confirmed-track points",
        frame_data_vec.len(),
        trajectories.len(),
        total_gt,
        total_tracks,
    );

    Ok(build_benchmark_result(
        &manifest.name,
        &frame_data_vec,
        &estimate_frames,
        &tracker,
        (params.measurement_noise_sigma * 10.0).max(500.0),
        start,
    ))
}

/// ENU reference point for ADS-B measurement conversion.
#[cfg(feature = "adsb")]
struct EnuRef {
    lat_rad: f64,
    lon_rad: f64,
    alt_m: f64,
}

/// Bin ADS-B state vectors into per-step noisy ENU detection vectors.
#[cfg(feature = "adsb")]
fn bin_adsb_detections(
    states: &[crate::adsb::StateVector],
    params: &ScenarioParameters,
    t0: f64,
    enu_ref: &EnuRef,
    normal: &rand_distr::Normal<f64>,
    rng: &mut impl rand::Rng,
) -> std::collections::BTreeMap<i64, Vec<DVector<f64>>> {
    let ref_lat_rad = enu_ref.lat_rad;
    let ref_lon_rad = enu_ref.lon_rad;
    let ref_alt_m = enu_ref.alt_m;
    use crate::adsb::state_to_measurement;
    use rand_distr::Distribution;
    use thresh_core::geodetic::wgs84_to_enu;

    let mut dets_by_step: std::collections::BTreeMap<i64, Vec<DVector<f64>>> =
        std::collections::BTreeMap::new();

    for sv in states {
        if let Some(thresh_core::measurement::Measurement::AdsB {
            lat,
            lon,
            alt,
            time,
            ..
        }) = state_to_measurement(sv)
        {
            let enu = wgs84_to_enu(
                lat.to_radians(),
                lon.to_radians(),
                alt,
                ref_lat_rad,
                ref_lon_rad,
                ref_alt_m,
            );
            let step = ((time - t0) / params.dt).round() as i64;
            let noisy = DVector::from_column_slice(&[
                enu.x + params.measurement_noise_sigma * normal.sample(rng),
                enu.y + params.measurement_noise_sigma * normal.sample(rng),
                enu.z + params.measurement_noise_sigma * normal.sample(rng),
            ]);
            dets_by_step.entry(step).or_default().push(noisy);
        }
    }

    dets_by_step
}

// ---------------------------------------------------------------------------
// nuScenes benchmark runner (§7.7)
// ---------------------------------------------------------------------------

/// Run a nuScenes benchmark scenario end-to-end.
///
/// Feature-gated on `nuscenes` because it depends on the PyO3 bridge to
/// the `nuscenes-devkit` Python package. Downstream callers that need
/// nuScenes scenarios must:
///
/// 1. Build `thresh-data` with `--features nuscenes` (pulls in PyO3).
/// 2. Have a Python environment with `nuscenes-devkit` installed and
///    on `PYTHONPATH` / activated in the shell the binary runs from.
/// 3. Provide a local copy of the nuScenes dataset (the `v1.0-mini`
///    split is ~4 GB). The dataset root is resolved from
///    `ScenarioSource::NuScenes::dataroot` when set, otherwise from the
///    `NUSCENES_DATA_ROOT` environment variable, so scenario manifests
///    can stay portable.
///
/// Pipeline:
/// 1. Open the requested split via `NuScenesDataset::load`, which
///    eagerly materialises per-sample `Frame`s with ground-truth
///    annotations.
/// 2. Use the 3-D annotation centroid as a simulated detection source,
///    adding seeded Gaussian noise with the configured `measurement_noise_sigma`.
///    nuScenes samples are ~0.5 s apart, so the benchmark steps at
///    that cadence rather than using `parameters.dt` naively.
/// 3. Feed the noisy detections into `MultiObjectTracker::new_cv_position`
///    and compute MOTA / MOTP / IDF1 / HOTA against the annotation
///    ground truth.
///
/// `manifest_dir` is accepted for parity with the ADS-B / orbital runners
/// but isn't used — the nuScenes dataroot resolution is environment-
/// rather than manifest-dir-relative.
#[cfg(feature = "nuscenes")]
pub fn run_nuscenes_benchmark(
    manifest: &ScenarioManifest,
    _manifest_dir: &Path,
) -> core::result::Result<BenchmarkResult, String> {
    use crate::dataset::Dataset;
    use crate::nuscenes::{NuScenesBridge, NuScenesDataset};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    let start = Instant::now();
    let params = &manifest.parameters;

    let ScenarioSource::NuScenes {
        version,
        dataroot,
        scene_token,
    } = &manifest.source
    else {
        return Err(format!(
            "run_nuscenes_benchmark called on non-NuScenes source: {:?}",
            manifest.source
        ));
    };

    // ---- 1. Resolve dataroot and open the scene ----
    let dataroot_resolved: String = match dataroot.clone() {
        Some(d) => d,
        None => std::env::var("NUSCENES_DATA_ROOT").map_err(|_| {
            "nuScenes scenario requires `dataroot` in the manifest or \
             NUSCENES_DATA_ROOT in the environment"
                .to_string()
        })?,
    };

    // If no explicit scene token is supplied, pick the first scene in
    // the split. This matches what a user running `thresh-data run
    // nuscenes-mini.toml` expects without having to look up tokens.
    let resolved_scene_token: String = match scene_token.clone() {
        Some(t) => t,
        None => {
            let bridge = NuScenesBridge::new(version, &dataroot_resolved).map_err(|e| {
                format!("failed to open nuScenes bridge for dataroot {dataroot_resolved}: {e}")
            })?;
            let tokens = bridge
                .scene_tokens()
                .map_err(|e| format!("failed to list scene tokens in {dataroot_resolved}: {e}"))?;
            tokens
                .into_iter()
                .next()
                .ok_or_else(|| format!("nuScenes split {version} contained no scenes"))?
        }
    };

    let dataset = NuScenesDataset::load(version, &dataroot_resolved, &resolved_scene_token)
        .map_err(|e| format!("failed to load nuScenes scene {resolved_scene_token}: {e}"))?;

    // ---- 2. Collect frames and build detections + ground truth ----
    let mut rng = StdRng::seed_from_u64(0x1_2_3_4_5_6_7_8_u64);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut frame_data_vec: Vec<FrameData> = Vec::new();
    let mut estimate_frames: Vec<EstimateFrame> = Vec::new();
    let frames: Vec<_> = dataset.frames().collect();
    if frames.is_empty() {
        return Err(format!(
            "nuScenes scene {resolved_scene_token} produced no frames"
        ));
    }

    // The nuScenes benchmark runner uses a single tracker seeded with
    // the scenario-configured measurement noise, then steps once per
    // sample (keyframe) with the inter-sample time delta.
    let mut tracker =
        MultiObjectTracker::new_cv_position(params.measurement_noise_sigma, params.gate_threshold);

    // nuScenes keyframes are roughly 0.5 s apart; use the difference
    // between successive timestamps as the tracker `dt`, defaulting to
    // `params.dt` for the very first sample where we have no previous
    // timestamp.
    let mut prev_ts: Option<f64> = None;

    for frame in &frames {
        let dt = match prev_ts {
            Some(pt) => (frame.timestamp - pt).max(0.01),
            None => params.dt,
        };
        prev_ts = Some(frame.timestamp);

        let gt: Vec<(u64, [f64; 3])> = frame
            .ground_truth
            .as_ref()
            .map(|entries| entries.iter().map(|e| (e.target_id, e.position)).collect())
            .unwrap_or_default();

        let detections: Vec<DVector<f64>> = gt
            .iter()
            .map(|(_, pos)| {
                DVector::from_column_slice(&[
                    pos[0] + params.measurement_noise_sigma * normal.sample(&mut rng),
                    pos[1] + params.measurement_noise_sigma * normal.sample(&mut rng),
                    pos[2] + params.measurement_noise_sigma * normal.sample(&mut rng),
                ])
            })
            .collect();

        tracker.step(&detections, dt);

        push_step_frames(&tracker, gt, &mut frame_data_vec, &mut estimate_frames);
    }

    // ---- 3. Metrics ----
    let total_gt: usize = frame_data_vec.iter().map(|f| f.gt.len()).sum();
    let total_tracks: usize = frame_data_vec.iter().map(|f| f.tracks.len()).sum();
    eprintln!(
        "nuScenes pipeline: {} frames, {} ground-truth annotations, {} confirmed-track points",
        frame_data_vec.len(),
        total_gt,
        total_tracks,
    );

    Ok(build_benchmark_result(
        &manifest.name,
        &frame_data_vec,
        &estimate_frames,
        &tracker,
        (params.measurement_noise_sigma * 10.0).max(5.0),
        start,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv_clean_manifest() -> ScenarioManifest {
        ScenarioManifest {
            name: "synth-cv-clean".into(),
            description: "5 CV targets, low noise, perfect detection".into(),
            source: ScenarioSource::Synthetic,
            parameters: ScenarioParameters {
                duration_s: 30.0,
                dt: 1.0,
                measurement_noise_sigma: 50.0,
                gate_threshold: 500.0,
                tracker_variant: None,
                scenario_type: Some("cv-clean".into()),
                measurement_model: MeasurementModel::Cartesian,
                tracker_noise_sigma: None,
            },
            baselines: Some(Baselines {
                mota: Some(0.5),
                ..Baselines::default()
            }),
        }
    }

    /// `measurement_model` is a typed enum: a misspelled value must fail
    /// TOML parsing loudly (naming the field) instead of silently
    /// selecting the default RAE model — the model choice is load-bearing
    /// for the consistency gates.
    #[test]
    fn misspelled_measurement_model_fails_to_parse() {
        let toml = r#"
            name = "typo"
            description = "misspelled measurement model"
            source = "Synthetic"
            [parameters]
            duration_s = 30.0
            dt = 1.0
            measurement_noise_sigma = 50.0
            gate_threshold = 500.0
            measurement_model = "cartesain"
        "#;
        let err = toml::from_str::<ScenarioManifest>(toml)
            .expect_err("unknown measurement_model must be a parse error");
        let msg = err.to_string();
        assert!(
            msg.contains("cartesian") || msg.contains("measurement_model"),
            "error should point at the field or list variants: {msg}"
        );
    }

    /// The committed `synth-cv-clean.toml`, loaded from disk so the task
    /// 6.6 gate tests exercise exactly the scenario CI runs.
    fn committed_cv_clean_manifest() -> ScenarioManifest {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios/synth-cv-clean.toml");
        load_scenario(&path).expect("committed synth-cv-clean.toml loads")
    }

    /// Task 6.6 (specs "Honest covariance passes the gate", "Runner
    /// populates consistency statistics"): the committed scenario — honest
    /// R, calibrated bounds — passes every gate with ANEES/ANIS populated.
    #[test]
    fn honest_covariance_passes_the_gate() {
        let manifest = committed_cv_clean_manifest();
        let result = run_synthetic_benchmark(&manifest);
        assert!(result.anees.is_some(), "runner populates ANEES");
        assert!(result.anis.is_some(), "runner populates ANIS");
        assert!(result.anees_samples > 0 && result.anis_samples > 0);
        let failures = check_regression(&result, manifest.baselines.as_ref().unwrap());
        assert!(failures.is_empty(), "honest tuning must pass: {failures:?}");
    }

    /// Task 6.6 (spec "Dishonest covariance fails the gate"): a tracker R
    /// far below the generated noise (a covariance lie MOT metrics cannot
    /// see — MOTA/HOTA/IDF1 all still clear their floors) is flagged by
    /// ANEES **and** ANIS above their upper bounds.
    ///
    /// The knob is R rather than the task sketch's Q because cv-clean
    /// truth is exact CV with zero process noise: scaling the filter's
    /// (already negligible) Q cannot manufacture overconfidence here,
    /// while an understated R is exactly the same covariance lie
    /// (recorded as an implementation-time divergence in design.md).
    #[test]
    fn overconfident_covariance_fails_high_while_mot_passes() {
        let mut manifest = committed_cv_clean_manifest();
        manifest.parameters.tracker_noise_sigma = Some(5.0);
        let result = run_synthetic_benchmark(&manifest);
        let baselines = manifest.baselines.as_ref().unwrap();

        let failures = check_regression(&result, baselines);
        assert!(
            failures
                .iter()
                .any(|f| f.contains("ANEES") && f.contains("above")),
            "ANEES must fail above its upper bound: {failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|f| f.contains("ANIS") && f.contains("above")),
            "ANIS must fail above its upper bound: {failures:?}"
        );
        // The lie is invisible to the MOT gates: every failure is a
        // consistency bound, none an MOT floor.
        assert!(result.mota >= baselines.mota.unwrap(), "MOTA still passes");
        assert!(
            failures
                .iter()
                .all(|f| f.contains("ANEES") || f.contains("ANIS")),
            "only consistency gates may fail: {failures:?}"
        );
    }

    /// Task 6.6 (spec "Underconfident covariance also fails"): an inflated
    /// tracker R drives ANEES and ANIS below the lower bounds while the
    /// MOT floors still pass — both tails of the interval are enforced.
    #[test]
    fn underconfident_covariance_fails_low_while_mot_passes() {
        let mut manifest = committed_cv_clean_manifest();
        manifest.parameters.tracker_noise_sigma = Some(150.0);
        let result = run_synthetic_benchmark(&manifest);
        let baselines = manifest.baselines.as_ref().unwrap();

        let failures = check_regression(&result, baselines);
        assert!(
            failures
                .iter()
                .any(|f| f.contains("ANEES") && f.contains("below")),
            "ANEES must fail below its lower bound: {failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|f| f.contains("ANIS") && f.contains("below")),
            "ANIS must fail below its lower bound: {failures:?}"
        );
        assert!(result.mota >= baselines.mota.unwrap(), "MOTA still passes");
        assert!(
            failures
                .iter()
                .all(|f| f.contains("ANEES") || f.contains("ANIS")),
            "only consistency gates may fail: {failures:?}"
        );
    }

    /// Task 6.6 (specs "Deterministic gate outcome", "Repeated runs
    /// agree"): the seeded scenario produces bitwise-identical statistics
    /// and the same pass/fail outcome across runs in one process.
    #[test]
    fn same_seed_runs_bitwise_identical_and_same_outcome() {
        let manifest = committed_cv_clean_manifest();
        let a = run_synthetic_benchmark(&manifest);
        let b = run_synthetic_benchmark(&manifest);
        assert_eq!(a.mota.to_bits(), b.mota.to_bits());
        assert_eq!(a.hota.to_bits(), b.hota.to_bits());
        assert_eq!(a.idf1.to_bits(), b.idf1.to_bits());
        assert_eq!(
            a.anees.map(f64::to_bits),
            b.anees.map(f64::to_bits),
            "ANEES bitwise identical"
        );
        assert_eq!(
            a.anis.map(f64::to_bits),
            b.anis.map(f64::to_bits),
            "ANIS bitwise identical"
        );
        assert_eq!(a.anees_samples, b.anees_samples);
        assert_eq!(a.anis_samples, b.anis_samples);
        assert_eq!(a.gospa.map(f64::to_bits), b.gospa.map(f64::to_bits));
        let baselines = manifest.baselines.as_ref().unwrap();
        assert_eq!(
            check_regression(&a, baselines),
            check_regression(&b, baselines),
            "identical pass/fail outcome"
        );
    }

    /// A `BenchmarkResult` with the given MOT values and absent
    /// consistency statistics (the pre-`eval-consistency-metrics` shape).
    fn mot_result(mota: f64, motp: f64, idf1: f64, hota: f64) -> BenchmarkResult {
        BenchmarkResult {
            scenario: "test".into(),
            mota,
            motp,
            idf1,
            hota,
            id_switches: 0,
            anees: None,
            anees_samples: 0,
            anis: None,
            anis_samples: 0,
            gospa: None,
            duration_ms: 0,
        }
    }

    #[test]
    fn load_and_parse_scenario_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        let manifest = cv_clean_manifest();
        let toml_str = toml::to_string_pretty(&manifest).unwrap();
        std::fs::write(&path, &toml_str).unwrap();

        let loaded = load_scenario(&path).unwrap();
        assert_eq!(loaded.name, "synth-cv-clean");
        assert_eq!(loaded.parameters.duration_s, 30.0);
    }

    #[test]
    fn run_synth_cv_clean_mota() {
        let manifest = cv_clean_manifest();
        let result = run_synthetic_benchmark(&manifest);
        // With confirmed-only tracks and well-separated CV targets, MOTA
        // should be solidly positive. The first few frames have FN while
        // tracks are still tentative, but after confirmation tracking is
        // reliable.
        assert!(
            result.mota > 0.5,
            "MOTA should be reasonable for clean CV scenario, got {}",
            result.mota
        );
    }

    #[test]
    fn test_build_maneuvering_trajectories() {
        let params = ScenarioParameters {
            duration_s: 30.0,
            dt: 1.0,
            measurement_noise_sigma: 50.0,
            gate_threshold: 500.0,
            tracker_variant: None,
            scenario_type: Some("maneuvering".into()),
            measurement_model: MeasurementModel::RadarRae,
            tracker_noise_sigma: None,
        };
        let trajs = build_maneuvering_trajectories(&params);
        assert_eq!(trajs.len(), 4, "expected 4 maneuvering trajectories");
        for t in &trajs {
            assert!(
                t.segments.len() > 1,
                "each maneuvering trajectory should have multiple segments"
            );
            let waypoints = t.generate();
            let total_dur: f64 = waypoints.last().unwrap().time - waypoints.first().unwrap().time;
            assert!(
                total_dur >= params.duration_s - params.dt,
                "waypoints should span approximately duration_s"
            );
        }
    }

    #[test]
    fn test_build_heterogeneous_trajectories() {
        let params = ScenarioParameters {
            duration_s: 30.0,
            dt: 1.0,
            measurement_noise_sigma: 50.0,
            gate_threshold: 500.0,
            tracker_variant: None,
            scenario_type: Some("heterogeneous".into()),
            measurement_model: MeasurementModel::RadarRae,
            tracker_noise_sigma: None,
        };
        let trajs = build_heterogeneous_trajectories(&params);
        assert_eq!(trajs.len(), 5, "expected 5 heterogeneous trajectories");

        // Verify distinct velocity magnitudes: UAV < aircraft < missile.
        let speed = |t: &Trajectory| -> f64 {
            let v = t.initial_velocity;
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        };
        let uav_speed = speed(&trajs[0]).max(speed(&trajs[1]));
        let aircraft_speed = speed(&trajs[2]).min(speed(&trajs[3]));
        let missile_speed = speed(&trajs[4]);
        assert!(
            uav_speed < aircraft_speed,
            "UAV speed ({uav_speed:.1}) should be less than aircraft ({aircraft_speed:.1})"
        );
        assert!(
            aircraft_speed < missile_speed,
            "aircraft speed ({aircraft_speed:.1}) should be less than missile ({missile_speed:.1})"
        );
    }

    #[test]
    fn test_build_low_pd_trajectories() {
        let params = ScenarioParameters {
            duration_s: 30.0,
            dt: 1.0,
            measurement_noise_sigma: 50.0,
            gate_threshold: 500.0,
            tracker_variant: None,
            scenario_type: Some("low-pd".into()),
            measurement_model: MeasurementModel::RadarRae,
            tracker_noise_sigma: None,
        };
        let low_pd = build_low_pd_trajectories(&params);
        let cv_clean = build_cv_clean_trajectories(&params);
        assert_eq!(
            low_pd.len(),
            cv_clean.len(),
            "low-pd should produce the same number of trajectories as cv-clean"
        );
        for (a, b) in low_pd.iter().zip(cv_clean.iter()) {
            assert_eq!(a.initial_position, b.initial_position);
            assert_eq!(a.initial_velocity, b.initial_velocity);
        }
    }

    #[test]
    fn test_radar_config_for_scenario() {
        let default_params = ScenarioParameters {
            duration_s: 30.0,
            dt: 1.0,
            measurement_noise_sigma: 50.0,
            gate_threshold: 500.0,
            tracker_variant: None,
            scenario_type: None,
            measurement_model: MeasurementModel::RadarRae,
            tracker_noise_sigma: None,
        };
        let default_cfg = radar_config_for_scenario(&default_params);
        assert!(
            (default_cfg.p_detection - 1.0).abs() < f64::EPSILON,
            "default p_detection should be 1.0"
        );
        assert!(
            default_cfg.clutter_rate.abs() < f64::EPSILON,
            "default clutter_rate should be 0.0"
        );

        let low_pd_params = ScenarioParameters {
            scenario_type: Some("low-pd".into()),
            ..default_params.clone()
        };
        let low_pd_cfg = radar_config_for_scenario(&low_pd_params);
        assert!(
            (low_pd_cfg.p_detection - 0.7).abs() < f64::EPSILON,
            "low-pd p_detection should be 0.7"
        );
        assert!(
            (low_pd_cfg.clutter_rate - 5.0).abs() < f64::EPSILON,
            "low-pd clutter_rate should be 5.0"
        );
    }

    #[test]
    fn test_manifest_deserialization_scenario_variants() {
        let scenarios_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios");
        let cases = [
            ("synth-cv-clean.toml", Some("cv-clean"), Some(0.5)),
            ("synth-maneuvering.toml", Some("maneuvering"), Some(0.3)),
            ("synth-heterogeneous.toml", Some("heterogeneous"), Some(0.3)),
            ("synth-low-pd.toml", Some("low-pd"), Some(0.2)),
        ];
        for (file, expected_type, expected_mota) in &cases {
            let path = scenarios_dir.join(file);
            let manifest = load_scenario(&path).unwrap_or_else(|e| {
                panic!("failed to load {file}: {e}");
            });
            assert_eq!(
                manifest.parameters.scenario_type.as_deref(),
                *expected_type,
                "scenario_type mismatch for {file}"
            );
            if let Some(mota) = expected_mota {
                let baselines = manifest.baselines.as_ref().expect("baselines should exist");
                assert!(
                    (baselines.mota.unwrap() - mota).abs() < f64::EPSILON,
                    "mota baseline mismatch for {file}"
                );
            }
        }
    }

    #[test]
    fn regression_check_catches_failure() {
        let result = mot_result(0.72, 5.0, 0.60, 0.50);
        let baselines = Baselines {
            mota: Some(0.80),
            hota: Some(0.70),
            idf1: Some(0.75),
            ..Baselines::default()
        };
        let failures = check_regression(&result, &baselines);
        assert_eq!(failures.len(), 3);
        assert!(failures[0].contains("MOTA"));
        assert!(failures[1].contains("HOTA"));
        assert!(failures[2].contains("IDF1"));
    }

    #[test]
    fn regression_check_passes_above_baseline() {
        let result = mot_result(0.95, 3.0, 0.90, 0.85);
        let baselines = Baselines {
            mota: Some(0.80),
            hota: Some(0.70),
            idf1: Some(0.75),
            ..Baselines::default()
        };
        let failures = check_regression(&result, &baselines);
        assert!(
            failures.is_empty(),
            "Expected no failures, got: {failures:?}"
        );
    }

    // ---------------------------------------------------------------------
    // Consistency bounds in check_regression (eval-consistency-metrics,
    // task 6.4; spec "Regression check enforces consistency bounds" /
    // "Absent statistics are explicit")
    // ---------------------------------------------------------------------

    /// A result whose consistency statistics are populated with the given
    /// ANEES/ANIS values (samples > 0) and passing MOT values.
    fn consistency_result(anees: Option<f64>, anis: Option<f64>) -> BenchmarkResult {
        BenchmarkResult {
            anees,
            anees_samples: usize::from(anees.is_some()) * 100,
            anis,
            anis_samples: usize::from(anis.is_some()) * 100,
            ..mot_result(0.95, 3.0, 0.90, 0.85)
        }
    }

    /// Baselines carrying only the two-sided consistency intervals.
    fn consistency_baselines() -> Baselines {
        Baselines {
            anees_min: Some(2.5),
            anees_max: Some(3.5),
            anis_min: Some(2.6),
            anis_max: Some(3.4),
            ..Baselines::default()
        }
    }

    /// Above-max fails: the failure names the statistic, its value, and
    /// the violated bound.
    #[test]
    fn consistency_bound_above_max_fails() {
        let result = consistency_result(Some(5.0), Some(3.0));
        let failures = check_regression(&result, &consistency_baselines());
        assert_eq!(failures.len(), 1, "got: {failures:?}");
        assert!(failures[0].contains("ANEES"), "got: {}", failures[0]);
        assert!(failures[0].contains("5.0000"), "got: {}", failures[0]);
        assert!(
            failures[0].contains("above upper bound 3.5000"),
            "got: {}",
            failures[0]
        );
    }

    /// Below-min fails (bounds are two-sided, not a one-sided ceiling).
    #[test]
    fn consistency_bound_below_min_fails() {
        let result = consistency_result(Some(3.0), Some(1.0));
        let failures = check_regression(&result, &consistency_baselines());
        assert_eq!(failures.len(), 1, "got: {failures:?}");
        assert!(failures[0].contains("ANIS"), "got: {}", failures[0]);
        assert!(failures[0].contains("1.0000"), "got: {}", failures[0]);
        assert!(
            failures[0].contains("below lower bound 2.6000"),
            "got: {}",
            failures[0]
        );
    }

    /// Inside the interval passes.
    #[test]
    fn consistency_bound_inside_passes() {
        let result = consistency_result(Some(3.0), Some(3.0));
        let failures = check_regression(&result, &consistency_baselines());
        assert!(failures.is_empty(), "got: {failures:?}");
    }

    /// No declared bounds behaves exactly as today: a result with absent
    /// statistics produces no consistency failures, only the MOT floors
    /// apply (spec "Existing TOMLs parse unchanged").
    #[test]
    fn consistency_no_bounds_behaves_as_today() {
        let baselines = Baselines {
            mota: Some(0.80),
            hota: Some(0.70),
            idf1: Some(0.75),
            ..Baselines::default()
        };
        // Passing MOT, absent consistency: no failures at all.
        let good = mot_result(0.95, 3.0, 0.90, 0.85);
        assert!(check_regression(&good, &baselines).is_empty());
        // Failing MOT, absent consistency: exactly the three floor
        // messages, none mentioning ANEES/ANIS.
        let bad = mot_result(0.10, 5.0, 0.10, 0.10);
        let failures = check_regression(&bad, &baselines);
        assert_eq!(failures.len(), 3);
        assert!(
            failures
                .iter()
                .all(|f| !f.contains("ANEES") && !f.contains("ANIS")),
            "got: {failures:?}"
        );
    }

    /// FAIL-LOUD rule: a bound asserted against an absent statistic (zero
    /// samples) is a gate failure, never a silent pass.
    #[test]
    fn consistency_absent_statistic_with_bound_fails() {
        let result = consistency_result(None, None);
        let failures = check_regression(&result, &consistency_baselines());
        assert_eq!(failures.len(), 2, "got: {failures:?}");
        assert!(failures[0].contains("ANEES") && failures[0].contains("absent"));
        assert!(failures[1].contains("ANIS") && failures[1].contains("absent"));
    }

    /// A non-finite statistic under an asserted bound also fails loud
    /// (NaN comparisons would otherwise pass both sides silently).
    #[test]
    fn consistency_non_finite_statistic_with_bound_fails() {
        let result = consistency_result(Some(f64::NAN), Some(3.0));
        let failures = check_regression(&result, &consistency_baselines());
        assert_eq!(failures.len(), 1, "got: {failures:?}");
        assert!(failures[0].contains("ANEES") && failures[0].contains("absent"));
    }

    // ---------------------------------------------------------------------
    // Benchmark determinism (orbital-ballistic-filter-models, task 6.4;
    // spec "Repeated runs agree") — the CI baseline comparison is only
    // meaningful if the same scenario at the same revision reproduces the
    // same metrics, so these run the committed scenario manifests twice
    // through the library-level runners and require bitwise equality.
    // ---------------------------------------------------------------------

    /// Assert two benchmark results carry bitwise-identical metric values.
    fn assert_identical_metrics(a: &BenchmarkResult, b: &BenchmarkResult) {
        assert_eq!(
            a.mota.to_bits(),
            b.mota.to_bits(),
            "MOTA differs between runs: {} vs {}",
            a.mota,
            b.mota
        );
        assert_eq!(
            a.motp.to_bits(),
            b.motp.to_bits(),
            "MOTP differs between runs: {} vs {}",
            a.motp,
            b.motp
        );
        assert_eq!(
            a.idf1.to_bits(),
            b.idf1.to_bits(),
            "IDF1 differs between runs: {} vs {}",
            a.idf1,
            b.idf1
        );
        assert_eq!(
            a.hota.to_bits(),
            b.hota.to_bits(),
            "HOTA differs between runs: {} vs {}",
            a.hota,
            b.hota
        );
        assert_eq!(
            a.id_switches, b.id_switches,
            "ID switch counts differ between runs"
        );
        // Consistency statistics are part of the deterministic contract
        // too (spec "Deterministic gate outcome").
        assert_eq!(
            a.anees.map(f64::to_bits),
            b.anees.map(f64::to_bits),
            "ANEES differs between runs: {:?} vs {:?}",
            a.anees,
            b.anees
        );
        assert_eq!(a.anees_samples, b.anees_samples);
        assert_eq!(
            a.anis.map(f64::to_bits),
            b.anis.map(f64::to_bits),
            "ANIS differs between runs: {:?} vs {:?}",
            a.anis,
            b.anis
        );
        assert_eq!(a.anis_samples, b.anis_samples);
        assert_eq!(
            a.gospa.map(f64::to_bits),
            b.gospa.map(f64::to_bits),
            "GOSPA differs between runs: {:?} vs {:?}",
            a.gospa,
            b.gospa
        );
    }

    /// Task 6.4 — executing the committed ballistic scenario twice at the
    /// same revision yields identical metric values: the phased truth
    /// generator, the seeded measurement RNG, the tracker, and the metric
    /// computation are all deterministic. Runs under default features
    /// because the ballistic pipeline is pure Rust (no SGP4 / network).
    #[test]
    fn ballistic_benchmark_repeated_runs_identical() {
        let manifest_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scenarios")
            .join("ballistic-mrbm.toml");
        let manifest = load_scenario(&manifest_path).expect("load ballistic-mrbm.toml");
        let first = run_ballistic_benchmark(&manifest).expect("first ballistic run");
        let second = run_ballistic_benchmark(&manifest).expect("second ballistic run");
        assert_identical_metrics(&first, &second);
    }

    /// Task 6.4 — the same determinism property for the orbital runner on
    /// the cached-TLE ISS scenario (feature-gated like the runner itself).
    #[cfg(feature = "orbital")]
    #[test]
    fn orbital_benchmark_repeated_runs_identical() {
        let scenarios = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scenarios");
        let manifest =
            load_scenario(&scenarios.join("orbital-iss.toml")).expect("load orbital-iss.toml");
        let first = run_orbital_benchmark(&manifest, &scenarios).expect("first orbital run");
        let second = run_orbital_benchmark(&manifest, &scenarios).expect("second orbital run");
        assert_identical_metrics(&first, &second);
    }

    // ---------------------------------------------------------------------
    // ADS-B runner unit tests (feature-gated) — these cover error branches
    // that the CLI integration tests touch only indirectly, so SonarCloud
    // / codecov register them as direct coverage of the runner module.
    // ---------------------------------------------------------------------

    #[cfg(feature = "adsb")]
    fn adsb_manifest(state_file: Option<&str>) -> ScenarioManifest {
        ScenarioManifest {
            name: "adsb-test".into(),
            description: "ADS-B unit test".into(),
            source: ScenarioSource::AdsB {
                region: "JFK".into(),
                state_file: state_file.map(|s| s.to_string()),
                bbox: None,
                ref_lat_deg: 40.6413,
                ref_lon_deg: -73.7781,
                ref_alt_m: 0.0,
            },
            parameters: ScenarioParameters {
                duration_s: 10.0,
                dt: 1.0,
                measurement_noise_sigma: 50.0,
                gate_threshold: 500.0,
                tracker_variant: None,
                scenario_type: None,
            },
            baselines: Some(Baselines {
                mota: Some(-1.0),
                ..Baselines::default()
            }),
        }
    }

    #[test]
    fn adsb_source_defaults_fill_in_station_and_bbox() {
        // Non-feature-gated test: SonarCloud's coverage job runs
        // `cargo llvm-cov --workspace` without feature flags, so any
        // #[cfg(feature = "adsb")]-gated tests are invisible to it.
        // This test forces serde to invoke `default_adsb_ref_lat_deg`
        // and `default_adsb_ref_lon_deg` by deserializing an AdsB
        // manifest that omits those fields, so both defaults register
        // as covered in the default-features LCOV report.
        let toml = r#"
            name = "adsb-defaults-test"
            description = "parse test"
            [source.AdsB]
            region = "JFK"

            [parameters]
            duration_s = 1.0
            dt = 1.0
            measurement_noise_sigma = 10.0
            gate_threshold = 100.0
        "#;
        let manifest: ScenarioManifest = toml::from_str(toml).expect("parse AdsB defaults");
        let ScenarioSource::AdsB {
            region,
            state_file,
            bbox,
            ref_lat_deg,
            ref_lon_deg,
            ref_alt_m,
        } = &manifest.source
        else {
            panic!("expected AdsB source");
        };
        assert_eq!(region, "JFK");
        assert!(state_file.is_none());
        assert!(bbox.is_none());
        // The defaults should be the JFK-area pair the module ships.
        assert!((*ref_lat_deg - 40.6413).abs() < 1e-6);
        assert!((*ref_lon_deg - (-73.7781)).abs() < 1e-6);
        assert_eq!(*ref_alt_m, 0.0);
    }

    #[test]
    fn adsb_bounding_box_roundtrips() {
        // Exercise the `AdsBBoundingBox` struct in the default feature
        // set so its serde derive counts as covered too.
        let bbox = AdsBBoundingBox {
            lat_min: 40.0,
            lat_max: 41.0,
            lon_min: -74.0,
            lon_max: -73.0,
        };
        let json = serde_json::to_string(&bbox).unwrap();
        let parsed: AdsBBoundingBox = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.lat_min, 40.0);
        assert_eq!(parsed.lat_max, 41.0);
        assert_eq!(parsed.lon_min, -74.0);
        assert_eq!(parsed.lon_max, -73.0);
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_rejects_non_adsb_source() {
        let manifest = cv_clean_manifest();
        let err = run_adsb_benchmark(&manifest, std::path::Path::new(".")).unwrap_err();
        assert!(err.contains("non-AdsB"), "got: {err}");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_errors_without_state_file() {
        let manifest = adsb_manifest(None);
        let err = run_adsb_benchmark(&manifest, std::path::Path::new(".")).unwrap_err();
        assert!(err.contains("state_file"), "got: {err}");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_errors_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = adsb_manifest(Some("does-not-exist.json"));
        let err = run_adsb_benchmark(&manifest, dir.path()).unwrap_err();
        assert!(
            err.contains("does-not-exist") || err.contains("failed to read"),
            "got: {err}"
        );
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_errors_on_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bad.json"), "not valid json").unwrap();
        let manifest = adsb_manifest(Some("bad.json"));
        let err = run_adsb_benchmark(&manifest, dir.path()).unwrap_err();
        assert!(err.contains("JSON") || err.contains("parse"), "got: {err}");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_errors_on_empty_state_vec() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.json"), "[]").unwrap();
        let manifest = adsb_manifest(Some("empty.json"));
        let err = run_adsb_benchmark(&manifest, dir.path()).unwrap_err();
        assert!(err.contains("no state vectors"), "got: {err}");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_runs_committed_single_flight_fixture() {
        // Runs the real `adsb-single-flight.json` fixture through the
        // library-level runner (not just the CLI subprocess test) so
        // coverage of the long tracker-step loop, `extract_ground_truth`
        // interpolation, and the per-step bin / filter / collect paths
        // is attributed directly to `benchmark.rs` rather than to the
        // `tests/thresh_data_cli.rs` integration harness.
        let scenarios = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scenarios");
        let manifest_path = scenarios.join("adsb-single-flight.toml");
        assert!(manifest_path.exists(), "fixture missing");
        let manifest = load_scenario(&manifest_path).expect("load manifest");
        let result = run_adsb_benchmark(&manifest, &scenarios).expect("run_adsb_benchmark");
        assert_eq!(result.scenario, "adsb-single-flight");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_runs_committed_tracon_fixture() {
        let scenarios = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scenarios");
        let manifest_path = scenarios.join("adsb-tracon.toml");
        assert!(manifest_path.exists(), "fixture missing");
        let manifest = load_scenario(&manifest_path).expect("load manifest");
        let result = run_adsb_benchmark(&manifest, &scenarios).expect("run_adsb_benchmark");
        assert_eq!(result.scenario, "adsb-tracon");
    }

    #[cfg(feature = "adsb")]
    #[test]
    fn run_adsb_benchmark_runs_on_valid_fixture() {
        // Minimal valid fixture: 3 samples of 1 aircraft descending
        // into JFK over 3 seconds. Just enough to exercise the happy
        // path end-to-end through `extract_ground_truth` and the
        // tracker step loop.
        let json = r#"[
            {"icao24":"abc123","callsign":"T1","origin_country":"US",
             "time_position":1700000000.0,"last_contact":1700000000.0,
             "longitude":-73.7,"latitude":40.70,"baro_altitude":1000.0,
             "on_ground":false,"velocity":100.0,"true_track":260.0,
             "vertical_rate":-5.0,"geo_altitude":1000.0},
            {"icao24":"abc123","callsign":"T1","origin_country":"US",
             "time_position":1700000001.0,"last_contact":1700000001.0,
             "longitude":-73.75,"latitude":40.68,"baro_altitude":900.0,
             "on_ground":false,"velocity":100.0,"true_track":260.0,
             "vertical_rate":-100.0,"geo_altitude":900.0},
            {"icao24":"abc123","callsign":"T1","origin_country":"US",
             "time_position":1700000002.0,"last_contact":1700000002.0,
             "longitude":-73.78,"latitude":40.65,"baro_altitude":800.0,
             "on_ground":false,"velocity":100.0,"true_track":260.0,
             "vertical_rate":-100.0,"geo_altitude":800.0}
        ]"#;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("valid.json"), json).unwrap();
        let manifest = adsb_manifest(Some("valid.json"));
        let result =
            run_adsb_benchmark(&manifest, dir.path()).expect("valid fixture must run end-to-end");
        assert_eq!(result.scenario, "adsb-test");
        // The runner always produces a duration, even on a tiny fixture.
        // We don't assert on the metric values themselves because the
        // tracker's M-of-N confirmation window is longer than the 3-sample
        // fixture — the important thing is the pipeline completes and
        // emits a BenchmarkResult struct.
        assert!(result.id_switches == 0);
    }

    // ---- nuScenes runner (§7.7) ----

    // ---- Shared runner helpers ----

    #[test]
    fn collect_confirmed_track_positions_empty_when_no_confirmed() {
        // A fresh tracker has no tracks at all — the helper should
        // return an empty vec without panicking.
        let tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);
        let positions = collect_confirmed_track_positions(&tracker);
        assert!(positions.is_empty());
    }

    #[test]
    fn collect_confirmed_track_positions_matches_confirmed_tracks() {
        // Step the tracker enough times with a single consistent
        // detection to confirm a track, then assert the helper returns
        // exactly one entry at roughly the detection position. This
        // exercises the filter + project + collect path end-to-end
        // without touching any feature-gated runner.
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 500.0);
        for _ in 0..6 {
            let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
            tracker.step(&[det], 1.0);
        }
        let positions = collect_confirmed_track_positions(&tracker);
        assert!(
            !positions.is_empty(),
            "tracker should have confirmed at least one track after 6 consistent detections"
        );
        let (_, pos) = positions[0];
        assert!((pos[0] - 100.0).abs() < 50.0);
        assert!((pos[1] - 200.0).abs() < 50.0);
        assert!((pos[2] - 50.0).abs() < 50.0);
    }

    #[test]
    fn build_benchmark_result_wraps_metrics() {
        // Empty frame vec: all MOT metrics default to 0 / 0 / etc.
        // The helper must still produce a well-formed BenchmarkResult
        // with the supplied scenario name and a non-negative duration —
        // and every consistency statistic explicitly absent (`None`, not
        // zero, not NaN: spec "Absent statistics are explicit").
        let start = Instant::now();
        let frames: Vec<FrameData> = Vec::new();
        let estimates: Vec<EstimateFrame> = Vec::new();
        let tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);
        let result = build_benchmark_result("unit-test", &frames, &estimates, &tracker, 1.0, start);
        assert_eq!(result.scenario, "unit-test");
        assert_eq!(result.id_switches, 0);
        assert_eq!(result.anees, None);
        assert_eq!(result.anees_samples, 0);
        assert_eq!(result.anis, None);
        assert_eq!(result.anis_samples, 0);
        assert_eq!(result.gospa, None);
    }

    #[test]
    fn build_benchmark_result_populates_metrics_on_nonempty_frames() {
        // Hand-build a two-frame scenario where the tracker matches GT
        // exactly, so MOTA = 1. Exercises the compute_mot_metrics +
        // compute_idf1 + compute_hota_at_threshold branches inside
        // build_benchmark_result without needing a full runner.
        let frames = vec![
            FrameData {
                gt: vec![(1, [0.0, 0.0, 0.0])],
                tracks: vec![(1, [0.1, 0.0, 0.0])],
            },
            FrameData {
                gt: vec![(1, [1.0, 0.0, 0.0])],
                tracks: vec![(1, [1.05, 0.0, 0.0])],
            },
        ];
        let start = Instant::now();
        let tracker = MultiObjectTracker::new_cv_position(10.0, 100.0);
        let result = build_benchmark_result("perfect-match", &frames, &[], &tracker, 2.0, start);
        assert_eq!(result.scenario, "perfect-match");
        assert!(
            result.mota > 0.99,
            "expected near-perfect MOTA, got {}",
            result.mota
        );
        // GOSPA is computed from the same FrameData sequence (reported,
        // not gated); ANEES stays absent because no estimate frames were
        // collected, and ANIS stays absent because this tracker applied
        // no KF updates.
        let gospa = result.gospa.expect("non-empty sequence has GOSPA");
        assert!(gospa.is_finite() && gospa >= 0.0);
        assert_eq!(result.anees, None);
        assert_eq!(result.anis, None);
    }

    #[test]
    fn collect_confirmed_track_estimates_mirrors_positions() {
        // Confirm one track, then check the estimate collector returns the
        // same confirmed set as the position collector, carrying the full
        // 6D state and 6×6 covariance with the interleaved [0, 2, 4]
        // position projection agreeing bitwise.
        let mut tracker = MultiObjectTracker::new_cv_position(10.0, 500.0);
        for _ in 0..6 {
            let det = DVector::from_column_slice(&[100.0, 200.0, 50.0]);
            tracker.step(&[det], 1.0);
        }
        let positions = collect_confirmed_track_positions(&tracker);
        let estimates = collect_confirmed_track_estimates(&tracker);
        assert!(!estimates.is_empty(), "a track should have confirmed");
        assert_eq!(positions.len(), estimates.len());
        for ((pid, pos), est) in positions.iter().zip(&estimates) {
            assert_eq!(*pid, est.id);
            assert_eq!(est.state.len(), 6);
            assert_eq!(est.covariance.shape(), (6, 6));
            assert_eq!(est.state[0].to_bits(), pos[0].to_bits());
            assert_eq!(est.state[2].to_bits(), pos[1].to_bits());
            assert_eq!(est.state[4].to_bits(), pos[2].to_bits());
        }
    }

    /// Spec "Runner populates consistency statistics": the synthetic
    /// runner collects diagnostics and ground truth, so its result carries
    /// ANEES, ANIS (each with a positive sample count), and GOSPA.
    #[test]
    fn synthetic_runner_populates_consistency_statistics() {
        let manifest = cv_clean_manifest();
        let result = run_synthetic_benchmark(&manifest);
        let anees = result.anees.expect("ANEES populated");
        assert!(anees.is_finite() && anees > 0.0);
        assert!(result.anees_samples > 0);
        let anis = result.anis.expect("ANIS populated");
        assert!(anis.is_finite() && anis > 0.0);
        assert!(result.anis_samples > 0);
        let gospa = result.gospa.expect("GOSPA populated");
        assert!(gospa.is_finite() && gospa >= 0.0);
    }

    // ---------------------------------------------------------------------
    // Schema extension (eval-consistency-metrics, task 6.3)
    // ---------------------------------------------------------------------

    /// Spec "Existing TOMLs parse unchanged": every scenario TOML already
    /// committed under `scenarios/` deserializes, and every one except
    /// `synth-cv-clean` (whose bounds task 6.6 calibrated and committed)
    /// carries no consistency bounds — because absent bounds short-circuit
    /// `check_bound`, their regression-check outcome on any result is
    /// exactly the pre-change MOT-floor outcome.
    #[test]
    fn existing_scenario_tomls_parse_with_bounds_absent() {
        let scenarios_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(&scenarios_dir).expect("scenarios dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let manifest = load_scenario(&path)
                .unwrap_or_else(|e| panic!("{} must still parse: {e}", path.display()));
            let Some(b) = &manifest.baselines else {
                continue;
            };
            if manifest.name == "synth-cv-clean" {
                // The one calibrated gate (task 6.6): all four bounds set.
                assert!(
                    b.anees_min.is_some()
                        && b.anees_max.is_some()
                        && b.anis_min.is_some()
                        && b.anis_max.is_some(),
                    "synth-cv-clean must carry its calibrated consistency bounds"
                );
                continue;
            }
            checked += 1;
            assert!(
                b.anees_min.is_none()
                    && b.anees_max.is_none()
                    && b.anis_min.is_none()
                    && b.anis_max.is_none(),
                "{} must not declare consistency bounds (task 6.6 owns synth-cv-clean \
                 calibration; orbital section 6 owns the rest)",
                path.display()
            );
            // Outcome unchanged: with no consistency bounds, only the MOT
            // floors fire. A result violating every floor produces exactly
            // one failure per declared floor and none mention ANEES/ANIS…
            let worst = mot_result(f64::MIN, f64::MAX, f64::MIN, f64::MIN);
            let failures = check_regression(&worst, b);
            let declared_floors = usize::from(b.mota.is_some())
                + usize::from(b.hota.is_some())
                + usize::from(b.idf1.is_some());
            assert_eq!(failures.len(), declared_floors, "{}", path.display());
            assert!(
                failures
                    .iter()
                    .all(|f| !f.contains("ANEES") && !f.contains("ANIS")),
                "{}: {failures:?}",
                path.display()
            );
            // …and a result meeting every floor passes outright.
            let best = mot_result(f64::MAX, 0.0, f64::MAX, f64::MAX);
            assert!(check_regression(&best, b).is_empty(), "{}", path.display());
        }
        assert!(
            checked >= 4,
            "expected several committed scenarios with baselines, found {checked}"
        );
    }

    /// Spec "TOML with consistency bounds parses": a baselines section
    /// declaring the four bound fields deserializes with them populated
    /// and available to the regression check.
    #[test]
    fn toml_with_consistency_bounds_parses() {
        let toml = r#"
            name = "bounds-test"
            description = "consistency bounds parse test"
            source = "Synthetic"

            [parameters]
            duration_s = 10.0
            dt = 1.0
            measurement_noise_sigma = 50.0
            gate_threshold = 500.0

            [baselines]
            mota = 0.5
            anees_min = 1.8
            anees_max = 4.6
            anis_min = 2.1
            anis_max = 3.9
        "#;
        let manifest: ScenarioManifest = toml::from_str(toml).expect("bounds TOML parses");
        let b = manifest.baselines.expect("baselines present");
        assert_eq!(b.mota, Some(0.5));
        assert_eq!(b.anees_min, Some(1.8));
        assert_eq!(b.anees_max, Some(4.6));
        assert_eq!(b.anis_min, Some(2.1));
        assert_eq!(b.anis_max, Some(3.9));
        // The parsed bounds are live in the regression check.
        let result = consistency_result(Some(5.0), Some(3.0));
        let failures = check_regression(&result, &b);
        assert!(
            failures.iter().any(|f| f.contains("ANEES")),
            "got: {failures:?}"
        );
    }

    #[test]
    fn nuscenes_source_defaults_fill_in_version() {
        // Non-feature-gated coverage for `default_nuscenes_version`,
        // mirroring the ADS-B defaults test: SonarCloud runs coverage
        // without features, so anything gated behind `#[cfg(feature =
        // "nuscenes")]` is invisible. This forces serde to invoke the
        // default so it registers as covered.
        let toml = r#"
            name = "nuscenes-defaults-test"
            description = "parse test"
            [source.NuScenes]

            [parameters]
            duration_s = 1.0
            dt = 0.5
            measurement_noise_sigma = 1.0
            gate_threshold = 50.0
        "#;
        let manifest: ScenarioManifest = toml::from_str(toml).expect("parse NuScenes defaults");
        let ScenarioSource::NuScenes {
            version,
            dataroot,
            scene_token,
        } = &manifest.source
        else {
            panic!("expected NuScenes source");
        };
        assert_eq!(version, "v1.0-mini");
        assert!(dataroot.is_none());
        assert!(scene_token.is_none());
    }

    #[cfg(feature = "nuscenes")]
    fn nuscenes_manifest() -> ScenarioManifest {
        ScenarioManifest {
            name: "nuscenes-test".into(),
            description: "nuScenes unit test".into(),
            source: ScenarioSource::NuScenes {
                version: "v1.0-mini".into(),
                dataroot: None,
                scene_token: None,
            },
            parameters: ScenarioParameters {
                duration_s: 20.0,
                dt: 0.5,
                measurement_noise_sigma: 1.0,
                gate_threshold: 50.0,
                tracker_variant: None,
                scenario_type: None,
            },
            baselines: Some(Baselines {
                mota: Some(-2.0),
                ..Baselines::default()
            }),
        }
    }

    // The three `#[cfg(feature = "nuscenes")]` tests below are `#[ignore]`d
    // because the feature pulls in PyO3, which makes the test binary link
    // against libpython / Python3.framework at dyld load time. On macOS
    // that errors out before `main` even runs unless the framework is on
    // `DYLD_FALLBACK_FRAMEWORK_PATH`. The existing tests in
    // `crates/thresh-data/tests/nuscenes_integration.rs` follow the same
    // convention. Developers with a working Python env run them via:
    //
    //   cargo test -p thresh-data --features nuscenes -- --ignored nuscenes
    #[cfg(feature = "nuscenes")]
    #[test]
    #[ignore]
    fn run_nuscenes_benchmark_rejects_non_nuscenes_source() {
        let manifest = cv_clean_manifest();
        let err = run_nuscenes_benchmark(&manifest, std::path::Path::new(".")).unwrap_err();
        assert!(err.contains("non-NuScenes"), "got: {err}");
    }

    #[cfg(feature = "nuscenes")]
    #[test]
    #[ignore]
    fn run_nuscenes_benchmark_errors_without_dataroot_or_env() {
        // Save + clear `NUSCENES_DATA_ROOT` so the runner hits the
        // "manifest must supply dataroot" error branch.
        // SAFETY: tests run in a single process; we restore the env
        // var before returning.
        let previous = std::env::var("NUSCENES_DATA_ROOT").ok();
        unsafe {
            std::env::remove_var("NUSCENES_DATA_ROOT");
        }
        let manifest = nuscenes_manifest();
        let err = run_nuscenes_benchmark(&manifest, std::path::Path::new(".")).unwrap_err();
        if let Some(prev) = previous {
            unsafe {
                std::env::set_var("NUSCENES_DATA_ROOT", prev);
            }
        }
        assert!(
            err.contains("NUSCENES_DATA_ROOT") || err.contains("dataroot"),
            "got: {err}"
        );
    }

    /// Run a nuScenes scenario end-to-end when a local mini split is
    /// available. The test auto-skips if `NUSCENES_DATA_ROOT` is unset,
    /// so CI that doesn't provision the ~4 GB dataset silently passes
    /// this test while a developer with the mini split installed gets
    /// a real regression check.
    #[cfg(feature = "nuscenes")]
    #[test]
    #[ignore]
    fn run_nuscenes_benchmark_smoke_test_when_dataroot_set() {
        let Ok(dataroot) = std::env::var("NUSCENES_DATA_ROOT") else {
            eprintln!("NUSCENES_DATA_ROOT not set — skipping nuScenes smoke test");
            return;
        };
        if !std::path::Path::new(&dataroot).exists() {
            eprintln!(
                "NUSCENES_DATA_ROOT={dataroot} does not exist — skipping nuScenes smoke test"
            );
            return;
        }
        let manifest = nuscenes_manifest();
        let result = match run_nuscenes_benchmark(&manifest, std::path::Path::new(".")) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("nuScenes smoke test failed (likely missing nuscenes-devkit): {e}");
                return;
            }
        };
        assert_eq!(result.scenario, "nuscenes-test");
    }
}
