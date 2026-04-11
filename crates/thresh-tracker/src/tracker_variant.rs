//! # Tracker Variant Selection Guide
//!
//! thresh provides four tracker variants, each optimized for different
//! scenario characteristics. Use this guide to select the right variant for
//! your application.
//!
//! ## ENU (default) — [`crate::tracker::MultiObjectTracker`]
//!
//! * **Coordinate frame:** Local East-North-Up tangent plane at a user-chosen
//!   reference point.
//! * **State vector:** `[x, vx, y, vy, z, vz]` (6-dim constant-velocity).
//! * **Strengths:** Simple linear Kalman filter, fast, well understood, small
//!   state vector, trivial observation models for Cartesian detections.
//! * **Weaknesses:** The tangent plane diverges from the curved Earth at
//!   long ranges. By ~1000 km the accumulated error exceeds typical OTHR
//!   measurement noise; by 3000 km the projection error is several percent
//!   of range.
//! * **Use when:** Targets stay within a few hundred kilometres of the
//!   reference point, tracks are short (minutes), and the sensor footprint
//!   is local (conventional radar, short-range OTHR).
//! * **Don't use when:** Targets traverse >500 km during a single track,
//!   the scenario includes ballistic/orbital dynamics, or you need a single
//!   global frame across multiple widely-separated sensors.
//! * **Error at OTHR ranges:** ~1 km planar error at 1000 km range, growing
//!   quadratically thereafter.
//!
//! ## ECEF — `MultiObjectTrackerEcef`
//!
//! * **Coordinate frame:** Earth-Centered Earth-Fixed (a single global
//!   frame). State is 6-dim CV in ECEF meters.
//! * **Strengths:** Exact for any Earth-referenced motion, no projection
//!   error, trivially composable across multiple sensors, appropriate for
//!   ballistic and orbital dynamics where great-circle assumptions break
//!   down.
//! * **Weaknesses:** Observation models for OTHR and azimuth/elevation
//!   sensors are nonlinear and require an EKF. Process noise must account
//!   for the rotating Earth (centrifugal/Coriolis) for very long dwells.
//! * **Use when:** Targets are ballistic or orbital, traverses are very
//!   long, multiple widely-separated sensors feed into one tracker, or the
//!   scenario spans hemispheres.
//! * **Don't use when:** A simple linear ENU tracker is sufficient for the
//!   scenario — ECEF adds complexity without accuracy benefit at short
//!   ranges.
//! * **Error at OTHR ranges:** Limited only by measurement noise; no
//!   geometric error floor.
//!
//! ## Great-Circle — [`crate::great_circle_tracker::MultiObjectTrackerGreatCircle`]
//!
//! * **Coordinate frame:** Geodetic state
//!   `[lat, lon, alt, ground_speed, heading, climb_rate]` with motion
//!   advanced along the WGS84 ellipsoid via Vincenty's direct formula.
//! * **Strengths:** Constant-heading aircraft flight produces no accumulated
//!   flat-Earth error, no matter how long the track runs; state matches the
//!   natural parameterisation of civil aviation (heading + ground speed).
//! * **Weaknesses:** Nonlinear EKF (Vincenty direct + inverse in the
//!   Jacobians), more expensive per step, assumes a single transmitter's
//!   frame for OTHR observations, and the heading-based model is a poor
//!   fit for maneuvering or ballistic targets.
//! * **Use when:** Long-duration aircraft tracks (>1000 km traverse, tens of
//!   minutes to hours), scenarios where you care about preserving great-
//!   circle paths and heading/speed semantics.
//! * **Don't use when:** Targets maneuver aggressively, are not aircraft,
//!   or you have multiple transmitters contributing observations.
//! * **Error at OTHR ranges:** Essentially the measurement noise floor for
//!   constant-heading flight; degrades for maneuvers.
//!
//! ## Stereographic — [`crate::stereographic_tracker::MultiObjectTrackerStereographic`]
//!
//! * **Coordinate frame:** Conformal stereographic projection of the Earth's
//!   surface centred on the sensor (or the centroid of a sensor network),
//!   with state `[x, vx, y, vy, alt, valt]` in projected metres.
//! * **Strengths:** Angle-preserving, smooth scale factor that distorts
//!   distances by only a few percent even at 3000 km. Linear Kalman filter
//!   (much simpler than ECEF or great-circle), yet much more accurate than
//!   ENU for area surveillance.
//! * **Weaknesses:** Still has some (smooth) scale distortion far from the
//!   centre, requires a sensible projection centre, and is tied to a
//!   particular plane orientation.
//! * **Use when:** Wide-area surveillance with a single sensor or a tight
//!   cluster of sensors, coverage radii up to a few thousand km, and you
//!   want a linear filter rather than a full EKF.
//! * **Don't use when:** Your sensors span hemispheres (use ECEF instead),
//!   or the scenario is small enough that ENU is sufficient.
//! * **Error at OTHR ranges:** Sub-percent planar scale error at 2000 km,
//!   a few percent at 3000 km — dominated by the smooth conformal scale
//!   factor rather than accumulated flat-Earth error.
//!
//! ## Quick decision table
//!
//! | Scenario characteristic                              | Recommended variant |
//! |------------------------------------------------------|---------------------|
//! | Short tracks, local sensor, < 500 km traverse        | ENU                 |
//! | Long aircraft flights, > 1000 km traverse            | Great-Circle        |
//! | Ballistic or orbital targets                         | ECEF                |
//! | Wide-area surveillance (> 2000 km radius coverage)   | Stereographic       |
//! | Multiple widely-separated sensors, global frame      | ECEF                |
//! | Single sensor, moderate range, linear filter desired | Stereographic       |
//!
//! The [`TrackerVariant::recommend`] helper encodes a simple version of this
//! table and is intended as a starting point rather than a definitive answer.

use serde::{Deserialize, Serialize};

/// Enumeration of the tracker variants shipped with thresh.
///
/// See the module-level documentation for a detailed comparison of each
/// variant's strengths, weaknesses, and intended use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackerVariant {
    /// Cartesian ENU (default; appropriate for short-range and short tracks).
    Enu,
    /// ECEF (single global frame; appropriate for ballistic/orbital and long traverses).
    Ecef,
    /// Geodetic + great-circle motion (appropriate for long-duration aircraft).
    GreatCircle,
    /// Local stereographic projection (appropriate for area surveillance).
    Stereographic,
}

/// Summary of scenario characteristics used by [`TrackerVariant::recommend`].
#[derive(Debug, Clone, Copy)]
pub struct TrackerScenario {
    /// Maximum target traverse distance during a single track (meters).
    pub max_traverse_m: f64,
    /// Maximum track duration (seconds).
    pub max_duration_s: f64,
    /// Whether targets include orbital/ballistic dynamics.
    pub orbital_or_ballistic: bool,
    /// Whether scenario covers a wide area (>2000 km radius).
    pub wide_area_surveillance: bool,
}

impl TrackerVariant {
    /// Recommend a variant based on scenario characteristics.
    ///
    /// This is an opinionated starting point that implements the following
    /// simple priority:
    ///
    /// 1. Orbital or ballistic dynamics → ECEF.
    /// 2. Traverse > 1000 km → Great-Circle.
    /// 3. Wide-area surveillance (>2000 km radius) → Stereographic.
    /// 4. Otherwise → ENU (the default linear tracker).
    ///
    /// More nuanced selection (e.g. multi-sensor networks, mixed target
    /// classes, runtime budgets) is beyond the scope of this helper — use
    /// the module-level decision table as a guide.
    pub fn recommend(scenario: &TrackerScenario) -> Self {
        if scenario.orbital_or_ballistic {
            Self::Ecef
        } else if scenario.max_traverse_m > 1_000_000.0 {
            Self::GreatCircle
        } else if scenario.wide_area_surveillance {
            Self::Stereographic
        } else {
            Self::Enu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_scenario() -> TrackerScenario {
        TrackerScenario {
            max_traverse_m: 100_000.0,
            max_duration_s: 60.0,
            orbital_or_ballistic: false,
            wide_area_surveillance: false,
        }
    }

    #[test]
    fn recommends_enu_for_short_local_track() {
        assert_eq!(
            TrackerVariant::recommend(&base_scenario()),
            TrackerVariant::Enu
        );
    }

    #[test]
    fn recommends_ecef_for_ballistic() {
        let s = TrackerScenario {
            orbital_or_ballistic: true,
            ..base_scenario()
        };
        assert_eq!(TrackerVariant::recommend(&s), TrackerVariant::Ecef);
    }

    #[test]
    fn recommends_great_circle_for_long_traverse() {
        let s = TrackerScenario {
            max_traverse_m: 1_500_000.0,
            ..base_scenario()
        };
        assert_eq!(TrackerVariant::recommend(&s), TrackerVariant::GreatCircle);
    }

    #[test]
    fn recommends_stereographic_for_wide_area() {
        let s = TrackerScenario {
            wide_area_surveillance: true,
            ..base_scenario()
        };
        assert_eq!(TrackerVariant::recommend(&s), TrackerVariant::Stereographic);
    }

    #[test]
    fn orbital_takes_priority_over_traverse() {
        let s = TrackerScenario {
            max_traverse_m: 2_000_000.0,
            orbital_or_ballistic: true,
            wide_area_surveillance: true,
            ..base_scenario()
        };
        assert_eq!(TrackerVariant::recommend(&s), TrackerVariant::Ecef);
    }

    #[test]
    fn variant_round_trips_through_serde() {
        for v in [
            TrackerVariant::Enu,
            TrackerVariant::Ecef,
            TrackerVariant::GreatCircle,
            TrackerVariant::Stereographic,
        ] {
            let s = serde_json::to_string(&v).unwrap();
            let back: TrackerVariant = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back);
        }
    }
}
