//! Golden-fixture schema for the SGP4 tier-3 validation vectors.
//!
//! Design Decision 8 of the `orbital-ballistic-filter-models` change layers
//! golden-vector validation in three tiers; this module carries the schema
//! for tier 3: TEME position/velocity samples produced by the AIAA-verified
//! `sgp4` crate and committed as JSON under `test-data/golden/orbital/`.
//!
//! Two consumers share these types so the schema cannot drift:
//!
//! - the `gen-sgp4-fixtures` binary (behind the `orbital` feature, run
//!   manually, never in CI) **serializes** fixtures, and
//! - the default-feature envelope test `tests/golden_orbital.rs`
//!   **deserializes** them — the test needs no `sgp4`, no network, and no
//!   non-default features, per the spec's "Deterministic CI execution"
//!   scenario.
//!
//! This module is deliberately not feature-gated: it is a pair of plain
//! serde structs plus a path helper, with no dependency beyond what the
//! default build already carries.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thresh_core::time::Epoch;

/// One SGP4-propagated TEME state sample at a fixed offset from the TLE
/// epoch.
///
/// Units are SI (metres, metres/second) even though the `sgp4` crate
/// reports km/km·s⁻¹ — the generator converts once at write time so the
/// fixtures match the metre-based state convention of `thresh-filter`'s
/// orbital models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sgp4Sample {
    /// Seconds since the TLE epoch (0, 60, 300, 600 in the committed set).
    pub offset_s: f64,
    /// TEME position in metres.
    pub position_m: [f64; 3],
    /// TEME velocity in metres per second.
    pub velocity_m_s: [f64; 3],
}

/// A committed SGP4 golden fixture: one LEO object propagated by the
/// AIAA-verified `sgp4` crate, sampled at t₀ and t₀ + {60, 300, 600} s.
///
/// The TLE lines are recorded verbatim so the fixture is self-describing;
/// `test-data/golden/orbital/PROVENANCE.md` records the same lines plus the
/// `sgp4` crate version and the regeneration command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sgp4Fixture {
    /// Human-readable object name (e.g. `"ISS (ZARYA)"`).
    pub object: String,
    /// NORAD catalog number.
    pub norad_id: u32,
    /// TLE line 1, verbatim.
    pub tle_line1: String,
    /// TLE line 2, verbatim.
    pub tle_line2: String,
    /// Julian Date of the TLE epoch **in the UTC scale** (offsets in
    /// `samples` count from here).
    ///
    /// Deliberately kept as the raw `f64` the committed fixtures carry —
    /// the `astro-time-and-frames` migration keeps every fixture under
    /// `test-data/golden/orbital/` byte-identical (no regeneration), and
    /// the loader converts on demand via [`Self::epoch`].
    pub epoch_jd: f64,
    /// Reference frame of the samples — always `"TEME"` (SGP4 output is
    /// TEME by definition).
    pub frame: String,
    /// State samples ordered by increasing `offset_s`, starting at 0.
    pub samples: Vec<Sgp4Sample>,
}

impl Sgp4Fixture {
    /// The TLE epoch as a time-scale-aware [`Epoch`], converted from the
    /// fixture's raw `epoch_jd` float (documented scale: **UTC**).
    pub fn epoch(&self) -> Epoch {
        Epoch::from_jde_utc(self.epoch_jd)
    }
}

/// Directory holding the committed SGP4 golden fixtures:
/// `test-data/golden/orbital/` at the workspace root.
///
/// Resolved `CARGO_MANIFEST_DIR`-relative per the repo precedent
/// (`thresh-tracker/src/tracker.rs` reaches `test-data/` the same way), so
/// it works from `cargo test` and `cargo run` regardless of the invocation
/// directory.
pub fn golden_orbital_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test-data")
        .join("golden")
        .join("orbital")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_round_trips_through_json() {
        let fixture = Sgp4Fixture {
            object: "TEST".to_string(),
            norad_id: 99999,
            tle_line1: "1 ...".to_string(),
            tle_line2: "2 ...".to_string(),
            epoch_jd: 2_460_310.5,
            frame: "TEME".to_string(),
            samples: vec![Sgp4Sample {
                offset_s: 0.0,
                position_m: [1.0, 2.0, 3.0],
                velocity_m_s: [4.0, 5.0, 6.0],
            }],
        };
        let json = serde_json::to_string_pretty(&fixture).expect("serialize");
        let back: Sgp4Fixture = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.norad_id, fixture.norad_id);
        assert_eq!(back.samples.len(), 1);
        assert_eq!(back.samples[0].position_m, fixture.samples[0].position_m);
    }

    #[test]
    fn golden_dir_points_into_test_data() {
        let dir = golden_orbital_dir();
        assert!(dir.ends_with("test-data/golden/orbital"));
    }

    /// The loader-side epoch conversion (astro-time-and-frames task 4.2):
    /// the committed fixtures keep their raw `epoch_jd` floats byte-for-byte
    /// and `epoch()` interprets them in the documented UTC scale.
    #[test]
    fn epoch_accessor_converts_the_utc_jd_float() {
        let epoch = Epoch::from_jde_utc(2_460_310.5);
        let fixture = Sgp4Fixture {
            object: "TEST".to_string(),
            norad_id: 99999,
            tle_line1: "1 ...".to_string(),
            tle_line2: "2 ...".to_string(),
            epoch_jd: 2_460_310.5,
            frame: "TEME".to_string(),
            samples: Vec::new(),
        };
        assert_eq!(fixture.epoch(), epoch);
        // JD 2460310.5 UTC = 2024-01-01T00:00:00 UTC (the ISS fixture TLE
        // epoch, day 24001.0 — consistent with the committed PROVENANCE).
        assert_eq!(fixture.epoch().to_string(), "2024-01-01T00:00:00 UTC");
    }
}
