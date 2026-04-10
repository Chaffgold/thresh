//! Swerling RCS fluctuation models and target RCS profiles.
//!
//! Provides statistical RCS sampling for Swerling cases 0-IV and
//! aspect-dependent RCS lookup tables with bilinear interpolation.

use rand::Rng;
use rand_distr::{Distribution, Exp, Gamma};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Swerling type enum
// ---------------------------------------------------------------------------

/// Swerling fluctuation model type.
///
/// - **Zero**: deterministic (non-fluctuating) RCS.
/// - **One**: slow fluctuation (scan-to-scan), chi-squared 2 DOF.
/// - **Two**: fast fluctuation (pulse-to-pulse), chi-squared 2 DOF.
/// - **Three**: slow fluctuation, chi-squared 4 DOF.
/// - **Four**: fast fluctuation, chi-squared 4 DOF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwerlingType {
    Zero,
    One,
    Two,
    Three,
    Four,
}

impl SwerlingType {
    /// Returns `true` for slow-fluctuation models (I, III) where the RCS
    /// is constant across pulses within a single dwell.
    pub fn is_slow(&self) -> bool {
        matches!(self, SwerlingType::One | SwerlingType::Three)
    }
}

// ---------------------------------------------------------------------------
// Core sampling function
// ---------------------------------------------------------------------------

/// Sample an RCS value (m^2) from the given Swerling model.
///
/// - **Case 0**: returns `mean_rcs_m2` deterministically.
/// - **Cases I & II**: exponential (chi-squared 2 DOF), mean = `mean_rcs_m2`.
/// - **Cases III & IV**: Gamma(shape=2, scale=mean/2), i.e. chi-squared 4 DOF
///   scaled so the mean equals `mean_rcs_m2`.
///
/// For slow-fluctuation models (I, III), use [`DwellRcs`] to hold a single
/// sample constant across all pulses within a dwell.
pub fn sample_rcs<R: Rng>(swerling: SwerlingType, mean_rcs_m2: f64, rng: &mut R) -> f64 {
    match swerling {
        SwerlingType::Zero => mean_rcs_m2,
        SwerlingType::One | SwerlingType::Two => {
            // Exponential with mean = mean_rcs_m2
            // rand_distr::Exp parameterised by rate λ = 1/mean
            let dist = Exp::new(1.0 / mean_rcs_m2).expect("mean_rcs_m2 must be positive");
            dist.sample(rng)
        }
        SwerlingType::Three | SwerlingType::Four => {
            // Gamma(shape=2, scale=mean/2) so that E[X] = shape*scale = mean
            let dist = Gamma::new(2.0, mean_rcs_m2 / 2.0).expect("mean_rcs_m2 must be positive");
            dist.sample(rng)
        }
    }
}

// ---------------------------------------------------------------------------
// DwellRcs — holds RCS constant for slow-fluctuation models
// ---------------------------------------------------------------------------

/// Holds a single RCS sample for the duration of a dwell (slow fluctuation).
///
/// For Swerling I and III the RCS is drawn once per dwell and remains
/// constant across all pulses. Call [`DwellRcs::new_dwell`] at the start of
/// each new dwell to re-sample.
///
/// For Swerling 0, II, and IV this simply delegates to [`sample_rcs`] on
/// every call to [`DwellRcs::rcs`].
#[derive(Debug, Clone)]
pub struct DwellRcs {
    swerling: SwerlingType,
    mean_rcs_m2: f64,
    /// Cached value for slow-fluctuation models.
    cached: Option<f64>,
}

impl DwellRcs {
    /// Create a new `DwellRcs`. For slow-fluctuation types the first sample
    /// is drawn immediately.
    pub fn new<R: Rng>(swerling: SwerlingType, mean_rcs_m2: f64, rng: &mut R) -> Self {
        let cached = if swerling.is_slow() {
            Some(sample_rcs(swerling, mean_rcs_m2, rng))
        } else {
            None
        };
        Self {
            swerling,
            mean_rcs_m2,
            cached,
        }
    }

    /// Re-sample for a new dwell (only meaningful for slow-fluctuation types).
    pub fn new_dwell<R: Rng>(&mut self, rng: &mut R) {
        if self.swerling.is_slow() {
            self.cached = Some(sample_rcs(self.swerling, self.mean_rcs_m2, rng));
        }
    }

    /// Return the current RCS value (m^2).
    ///
    /// - Slow models: returns the cached dwell sample.
    /// - Fast models and Case 0: samples fresh (or returns deterministic).
    pub fn rcs<R: Rng>(&self, rng: &mut R) -> f64 {
        match self.cached {
            Some(v) => v,
            None => sample_rcs(self.swerling, self.mean_rcs_m2, rng),
        }
    }
}

// ---------------------------------------------------------------------------
// RCS lookup table (aspect-dependent)
// ---------------------------------------------------------------------------

/// Aspect-dependent RCS lookup table with bilinear interpolation.
///
/// The table stores RCS values in dBsm on a regular grid of azimuth and
/// elevation angles (degrees).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsLookupTable {
    /// Azimuth breakpoints in degrees (sorted, ascending).
    pub azimuth_deg: Vec<f64>,
    /// Elevation breakpoints in degrees (sorted, ascending).
    pub elevation_deg: Vec<f64>,
    /// RCS values in dBsm, indexed as `rcs_dbsm[az_idx][el_idx]`.
    pub rcs_dbsm: Vec<Vec<f64>>,
}

impl RcsLookupTable {
    /// Look up the RCS (in dBsm) at a given aspect angle using bilinear
    /// interpolation. Angles outside the table are clamped to the boundary.
    pub fn lookup(&self, azimuth_deg: f64, elevation_deg: f64) -> f64 {
        let (ai, at) = Self::interp_index(&self.azimuth_deg, azimuth_deg);
        let (ei, et) = Self::interp_index(&self.elevation_deg, elevation_deg);

        let ai1 = (ai + 1).min(self.azimuth_deg.len() - 1);
        let ei1 = (ei + 1).min(self.elevation_deg.len() - 1);

        let v00 = self.rcs_dbsm[ai][ei];
        let v01 = self.rcs_dbsm[ai][ei1];
        let v10 = self.rcs_dbsm[ai1][ei];
        let v11 = self.rcs_dbsm[ai1][ei1];

        let v0 = v00 + et * (v01 - v00);
        let v1 = v10 + et * (v11 - v10);
        v0 + at * (v1 - v0)
    }

    /// Find the lower index and fractional position for interpolation.
    fn interp_index(breakpoints: &[f64], value: f64) -> (usize, f64) {
        if breakpoints.len() < 2 {
            return (0, 0.0);
        }
        if value <= breakpoints[0] {
            return (0, 0.0);
        }
        let last = breakpoints.len() - 1;
        if value >= breakpoints[last] {
            return (last.saturating_sub(1), 1.0);
        }
        // Binary search for the interval
        let mut lo = 0;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if breakpoints[mid] <= value {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (value - breakpoints[lo]) / (breakpoints[hi] - breakpoints[lo]);
        (lo, t)
    }
}

// ---------------------------------------------------------------------------
// RCS profile
// ---------------------------------------------------------------------------

/// Complete RCS characterisation of a target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsProfile {
    /// Mean RCS in dBsm.
    pub mean_rcs_dbsm: f64,
    /// Swerling fluctuation model.
    pub swerling_type: SwerlingType,
    /// Optional aspect-dependent lookup table.
    pub aspect_table: Option<RcsLookupTable>,
}

impl RcsProfile {
    /// Convert `mean_rcs_dbsm` to linear m^2.
    pub fn mean_rcs_m2(&self) -> f64 {
        10.0_f64.powf(self.mean_rcs_dbsm / 10.0)
    }

    /// Fighter aircraft (~1 m^2 mean, Swerling I with aspect variation).
    pub fn fighter() -> Self {
        let azimuths = vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0];
        let elevations = vec![-10.0, 0.0, 10.0];
        // Nose ~-3 dBsm, beam ~+5 dBsm, tail ~0 dBsm
        let row = |nose: f64, beam: f64, tail: f64| -> Vec<f64> {
            vec![
                nose,
                (nose + beam) / 2.0,
                beam,
                (beam + tail) / 2.0,
                tail,
                (beam + tail) / 2.0,
                beam,
                (nose + beam) / 2.0,
                nose,
            ]
        };
        let rcs_dbsm = [
            row(-5.0, 3.0, -2.0), // look-down
            row(-3.0, 5.0, 0.0),  // level
            row(-5.0, 3.0, -2.0), // look-up
        ];
        // Transpose: table is [az_idx][el_idx]
        let mut table = vec![vec![0.0; elevations.len()]; azimuths.len()];
        for (ai, row_data) in azimuths.iter().enumerate() {
            let _ = row_data;
            for (ei, _) in elevations.iter().enumerate() {
                table[ai][ei] = rcs_dbsm[ei][ai];
            }
        }
        Self {
            mean_rcs_dbsm: 0.0, // 1 m^2
            swerling_type: SwerlingType::One,
            aspect_table: Some(RcsLookupTable {
                azimuth_deg: azimuths,
                elevation_deg: elevations,
                rcs_dbsm: table,
            }),
        }
    }

    /// Commercial airliner (~100 m^2 mean, Swerling I).
    pub fn airliner() -> Self {
        Self {
            mean_rcs_dbsm: 20.0, // 100 m^2
            swerling_type: SwerlingType::One,
            aspect_table: None,
        }
    }

    /// Cruise missile (~0.01-0.1 m^2, Swerling III).
    pub fn cruise_missile() -> Self {
        Self {
            mean_rcs_dbsm: -15.0, // ~0.032 m^2
            swerling_type: SwerlingType::Three,
            aspect_table: None,
        }
    }

    /// Small UAV (~0.01-1 m^2, Swerling I).
    pub fn uav() -> Self {
        Self {
            mean_rcs_dbsm: -10.0, // ~0.1 m^2
            swerling_type: SwerlingType::One,
            aspect_table: None,
        }
    }

    /// Satellite (~1-10 m^2, Swerling Zero / deterministic).
    pub fn satellite() -> Self {
        Self {
            mean_rcs_dbsm: 5.0, // ~3.16 m^2
            swerling_type: SwerlingType::Zero,
            aspect_table: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swerling_zero_deterministic() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let rcs = sample_rcs(SwerlingType::Zero, 5.0, &mut rng);
            assert!(
                (rcs - 5.0).abs() < f64::EPSILON,
                "Swerling 0 must return exact mean"
            );
        }
    }

    #[test]
    fn swerling_i_mean_and_variance() {
        let mut rng = rand::rng();
        let mean_rcs = 2.0;
        let n = 50_000;
        let samples: Vec<f64> = (0..n)
            .map(|_| sample_rcs(SwerlingType::One, mean_rcs, &mut rng))
            .collect();

        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        let sample_var = samples
            .iter()
            .map(|s| (s - sample_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Chi-squared 2 DOF (exponential): mean = mean_rcs, var = mean_rcs^2
        let expected_var = mean_rcs * mean_rcs;

        assert!(
            (sample_mean - mean_rcs).abs() < 0.1,
            "Swerling I mean: expected {mean_rcs}, got {sample_mean}"
        );
        assert!(
            (sample_var - expected_var).abs() / expected_var < 0.1,
            "Swerling I variance: expected {expected_var}, got {sample_var}"
        );
    }

    #[test]
    fn swerling_iii_mean_and_variance() {
        let mut rng = rand::rng();
        let mean_rcs = 3.0;
        let n = 50_000;
        let samples: Vec<f64> = (0..n)
            .map(|_| sample_rcs(SwerlingType::Three, mean_rcs, &mut rng))
            .collect();

        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        let sample_var = samples
            .iter()
            .map(|s| (s - sample_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Chi-squared 4 DOF scaled: mean = mean_rcs, var = mean_rcs^2 / 2
        let expected_var = mean_rcs * mean_rcs / 2.0;

        assert!(
            (sample_mean - mean_rcs).abs() < 0.1,
            "Swerling III mean: expected {mean_rcs}, got {sample_mean}"
        );
        assert!(
            (sample_var - expected_var).abs() / expected_var < 0.1,
            "Swerling III variance: expected {expected_var}, got {sample_var}"
        );
    }

    #[test]
    fn dwell_rcs_slow_constant_within_dwell() {
        let mut rng = rand::rng();
        let dwell = DwellRcs::new(SwerlingType::One, 5.0, &mut rng);
        let first = dwell.rcs(&mut rng);
        for _ in 0..100 {
            let v = dwell.rcs(&mut rng);
            assert!(
                (v - first).abs() < f64::EPSILON,
                "Slow model should be constant within dwell"
            );
        }
    }

    #[test]
    fn dwell_rcs_fast_varies() {
        let mut rng = rand::rng();
        let dwell = DwellRcs::new(SwerlingType::Two, 5.0, &mut rng);
        let samples: Vec<f64> = (0..100).map(|_| dwell.rcs(&mut rng)).collect();
        // Not all samples should be identical
        let all_same = samples
            .windows(2)
            .all(|w| (w[0] - w[1]).abs() < f64::EPSILON);
        assert!(!all_same, "Fast model (Swerling II) should vary per call");
    }

    #[test]
    fn lookup_table_at_grid_points() {
        let table = RcsLookupTable {
            azimuth_deg: vec![0.0, 90.0, 180.0],
            elevation_deg: vec![0.0, 45.0],
            rcs_dbsm: vec![
                vec![0.0, 5.0],   // az=0
                vec![10.0, 15.0], // az=90
                vec![0.0, 5.0],   // az=180
            ],
        };

        assert!((table.lookup(0.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((table.lookup(90.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((table.lookup(90.0, 45.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn lookup_table_interpolation() {
        let table = RcsLookupTable {
            azimuth_deg: vec![0.0, 90.0],
            elevation_deg: vec![0.0, 90.0],
            rcs_dbsm: vec![
                vec![0.0, 10.0],  // az=0
                vec![20.0, 30.0], // az=90
            ],
        };

        // Midpoint: az=45, el=45 -> average of all four corners
        let mid = table.lookup(45.0, 45.0);
        assert!(
            (mid - 15.0).abs() < 1e-10,
            "Bilinear midpoint should be 15.0, got {mid}"
        );

        // Along az edge: az=45, el=0 -> avg(0, 20) = 10
        let edge = table.lookup(45.0, 0.0);
        assert!(
            (edge - 10.0).abs() < 1e-10,
            "Edge interpolation should be 10.0, got {edge}"
        );
    }

    #[test]
    fn lookup_table_clamping() {
        let table = RcsLookupTable {
            azimuth_deg: vec![0.0, 90.0],
            elevation_deg: vec![0.0, 90.0],
            rcs_dbsm: vec![vec![0.0, 10.0], vec![20.0, 30.0]],
        };

        // Beyond boundaries should clamp
        let v = table.lookup(-10.0, -10.0);
        assert!((v - 0.0).abs() < 1e-10, "Clamped to corner (0,0)");

        let v = table.lookup(200.0, 200.0);
        assert!((v - 30.0).abs() < 1e-10, "Clamped to corner (90,90)");
    }

    #[test]
    fn preset_profiles_valid() {
        let profiles = [
            RcsProfile::fighter(),
            RcsProfile::airliner(),
            RcsProfile::cruise_missile(),
            RcsProfile::uav(),
            RcsProfile::satellite(),
        ];
        for p in &profiles {
            assert!(p.mean_rcs_m2() > 0.0, "Mean RCS must be positive");
        }
        // Fighter should have aspect table
        assert!(RcsProfile::fighter().aspect_table.is_some());
    }
}
