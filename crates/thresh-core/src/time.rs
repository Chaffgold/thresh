//! Time types: the tracker lane's relative [`Timestamp`] seconds and the
//! astro lane's absolute, time-scale-aware [`Epoch`].
//!
//! # Dependency license rationale
//!
//! [`Epoch`] is a newtype over [`hifitime::Epoch`] (pure Rust, MPL-2.0).
//! MPL-2.0 is a file-level copyleft: consuming the crate unmodified does not
//! encumber this Apache-2.0 workspace, so the dependency is acceptable.
//! nyx-space (same author, AGPL-3.0) remains rejected — AGPL's copyleft is
//! incompatible with this project's distribution terms. The newtype also
//! keeps `hifitime` out of thresh's public API, so the epoch backbone stays
//! swappable and the serialized form stays under thresh's control.

use std::fmt;
use std::ops::{Add, Sub};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use hifitime::TimeScale;

/// Timestamp in seconds (f64 for sub-second precision).
///
/// This is the tracker lane's *relative* time: bare seconds with no absolute
/// origin and no time scale. For absolute time — UTC/TAI/GPS/TT scales and
/// leap-second-correct arithmetic — use [`Epoch`] instead.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Timestamp(pub f64);

impl Timestamp {
    /// Time difference in seconds.
    pub fn dt(&self, other: &Timestamp) -> f64 {
        self.0 - other.0
    }
}

/// Absolute epoch with explicit time-scale semantics (UTC/TAI/GPS/TT).
///
/// Wraps [`hifitime::Epoch`], normalized to the **TAI scale internally**
/// (every constructor converts on entry). That invariant is what makes the
/// arithmetic leap-second correct: [`Sub`] between two epochs is an exact
/// difference on the continuous TAI timeline, so an interval that straddles a
/// leap-second insertion reports the true elapsed seconds (e.g. 2 s across
/// 2016-12-31T23:59:59 UTC → 2017-01-01T00:00:00 UTC), and `+ f64` steps by
/// true elapsed seconds — the same `dt` currency the tracker lane uses.
///
/// Every constructor and accessor states the time scale of its input/output;
/// no caller ever has to guess a scale.
///
/// # Precision
///
/// Internally hifitime stores integer nanoseconds, so construction from
/// calendar components or ISO-8601 strings and epoch differences are exact to
/// nanosecond resolution. Julian dates carried as `f64` are coarser: near
/// J2000-era dates one ulp of a JD is ~4.7e-10 day ≈ 40 µs, which bounds what
/// [`Epoch::to_jde_utc_days`]/[`Epoch::to_jde_tt_days`] can resolve.
///
/// # Example
///
/// ```
/// use thresh_core::time::Epoch;
///
/// // Scale-explicit construction: these UTC calendar components…
/// let e = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
/// // …step by true elapsed seconds and difference back exactly
/// // (exact for nanosecond-representable steps; see the `Add` impl).
/// let later = e + 60.0;
/// assert_eq!(later - e, 60.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(hifitime::Epoch);

impl Epoch {
    /// Builds an epoch from Gregorian calendar components **in the UTC
    /// scale** (`nanos` is the sub-second part in nanoseconds).
    ///
    /// # Panics
    ///
    /// Panics if the components do not form a valid Gregorian date
    /// (mirrors `hifitime::Epoch::from_gregorian_utc`).
    pub fn from_gregorian_utc(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        nanos: u32,
    ) -> Self {
        Self(
            hifitime::Epoch::from_gregorian_utc(year, month, day, hour, minute, second, nanos)
                .to_time_scale(TimeScale::TAI),
        )
    }

    /// Parses an ISO-8601 datetime string. The scale is taken from the
    /// trailing scale designator (`"2020-01-01T00:00:00 TAI"`, likewise
    /// `UTC`/`GPST`/`TT`); a `Z` suffix or **no designator means UTC**.
    ///
    /// # Errors
    ///
    /// Returns [`EpochParseError`] when the string is not a valid ISO-8601
    /// datetime with an optional scale designator.
    pub fn from_iso8601(s: &str) -> Result<Self, EpochParseError> {
        hifitime::Epoch::from_gregorian_str(s)
            .map(|e| Self(e.to_time_scale(TimeScale::TAI)))
            .map_err(|e| EpochParseError {
                input: s.to_string(),
                reason: e.to_string(),
            })
    }

    /// Builds an epoch from a Julian date (days) **in the UTC scale**.
    pub fn from_jde_utc(days: f64) -> Self {
        Self(hifitime::Epoch::from_jde_utc(days).to_time_scale(TimeScale::TAI))
    }

    /// Builds an epoch from a Julian date (days) **in the TT scale**.
    pub fn from_jde_tt(days: f64) -> Self {
        Self(
            hifitime::Epoch::from_jde_in_time_scale(days, TimeScale::TT)
                .to_time_scale(TimeScale::TAI),
        )
    }

    /// Julian date (days) **in the UTC scale**.
    ///
    /// `f64` quantization near J2000-era dates is one ulp ≈ 40 µs; see the
    /// type-level precision note.
    pub fn to_jde_utc_days(self) -> f64 {
        self.0.to_jde_utc_days()
    }

    /// Julian date (days) **in the TT scale** (TT = TAI + 32.184 s).
    ///
    /// `f64` quantization near J2000-era dates is one ulp ≈ 40 µs; see the
    /// type-level precision note.
    pub fn to_jde_tt_days(self) -> f64 {
        self.0.to_jde_tt_days()
    }
}

/// True elapsed time between two epochs, in **TAI seconds** (`self - rhs`).
///
/// Leap-second aware: the subtraction happens on the continuous TAI timeline
/// (both operands are stored in the TAI scale), so intervals spanning a
/// leap-second insertion report the physically elapsed seconds, not the UTC
/// wall-clock difference.
impl Sub for Epoch {
    type Output = f64;

    fn sub(self, rhs: Epoch) -> f64 {
        (self.0 - rhs.0).to_seconds()
    }
}

/// Steps an epoch forward by `seconds` of **true elapsed (TAI) time**
/// (negative values step backward).
///
/// The addition is performed in the internal TAI scale, which hifitime
/// quantizes to integer nanoseconds: `(e + dt) - e == dt` holds exactly
/// for any `dt` representable at nanosecond resolution (e.g. `60.0`,
/// `0.1`), while a non-representable `dt` (e.g. `1.0 / 3.0`) is rounded
/// to the nearest nanosecond on the way in — up to 0.5 ns per step.
/// Propagation loops that accumulate many steps of an irrational `dt`
/// should difference epochs rather than sum `dt`s. Same relative-seconds
/// `dt` convention as the tracker lane.
impl Add<f64> for Epoch {
    type Output = Epoch;

    fn add(self, seconds: f64) -> Epoch {
        Epoch(self.0 + seconds)
    }
}

/// Displays as the serde form: ISO-8601 Gregorian **in the UTC scale** with
/// an explicit ` UTC` scale designator.
impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.to_gregorian_str(TimeScale::UTC))
    }
}

/// Serializes as an ISO-8601 string **in the UTC scale** with an explicit
/// ` UTC` scale designator, e.g. `"2021-06-15T12:34:56.123456789 UTC"`
/// (nanosecond precision, so the round trip is exact).
impl Serialize for Epoch {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_gregorian_str(TimeScale::UTC))
    }
}

/// Deserializes from the string form accepted by [`Epoch::from_iso8601`]
/// (scale designator honored; bare or `Z`-suffixed strings are UTC).
impl<'de> Deserialize<'de> for Epoch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Epoch::from_iso8601(&s).map_err(serde::de::Error::custom)
    }
}

/// Error from [`Epoch::from_iso8601`]: the input string was not a valid
/// ISO-8601 datetime with an optional time-scale designator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochParseError {
    /// The string that failed to parse.
    pub input: String,
    /// Human-readable reason from the underlying parser.
    pub reason: String,
}

impl fmt::Display for EpochParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid ISO-8601 epoch {:?}: {}",
            self.input, self.reason
        )
    }
}

impl std::error::Error for EpochParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: "Scale conversions agree with published offsets".
    ///
    /// Published offsets at 2020-01-01: TAI−UTC = 37 s (IERS Bulletin C:
    /// UTC−TAI = −37 s from 2017-01-01 0h UTC until further notice; the same
    /// entry ships in hifitime 4.3.0's `epoch/leap_seconds.rs` table);
    /// GPS = TAI − 19 s, constant since the 1980-01-06 GPS epoch (encoded as
    /// hifitime's `SECONDS_GPS_TAI_OFFSET`) ⇒ GPS−UTC = 18 s;
    /// TT = TAI + 32.184 s by IAU definition.
    #[test]
    fn scale_offsets_at_2020_agree_with_published_values() {
        let utc = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);

        // Read the same epoch back in each scale (explicit "+offset" stamps).
        assert_eq!(
            utc.0.to_gregorian_str(TimeScale::TAI),
            "2020-01-01T00:00:37 TAI"
        );
        assert_eq!(
            utc.0.to_gregorian_str(TimeScale::GPST),
            "2020-01-01T00:00:18 GPST"
        );
        assert_eq!(
            utc.0.to_gregorian_str(TimeScale::TT),
            "2020-01-01T00:01:09.184000000 TT"
        );

        // Cross-scale construction lands on the identical instant (0 ns).
        let tai = Epoch::from_iso8601("2020-01-01T00:00:37 TAI").unwrap();
        let gps = Epoch::from_iso8601("2020-01-01T00:00:18 GPST").unwrap();
        let tt = Epoch::from_iso8601("2020-01-01T00:01:09.184 TT").unwrap();
        assert_eq!(tai - utc, 0.0, "TAI must lead UTC by exactly 37 s");
        assert_eq!(gps - utc, 0.0, "GPS must lead UTC by exactly 18 s");
        assert_eq!(tt - utc, 0.0, "TT must lead TAI by exactly 32.184 s");
    }

    /// Spec: "Julian-date round trip in a stated scale".
    ///
    /// JD 2458849.5 UTC = 2020-01-01T00:00:00 UTC: the Fliegel & Van Flandern
    /// (1968) day-count algorithm gives JDN 2458850 at 2020-01-01 noon, hence
    /// 2458849.5 at midnight (calendar↔JD day counting is scale-agnostic; the
    /// UTC tag states which scale both the input and the reading are in). The
    /// test also cross-checks against the UTC calendar constructor below.
    #[test]
    fn jd_utc_round_trip_and_tt_offset() {
        let jd_utc = 2458849.5;
        let e = Epoch::from_jde_utc(jd_utc);
        let sub_us_in_days = 1.0e-6 / 86400.0;
        assert!(
            (e.to_jde_utc_days() - jd_utc).abs() < sub_us_in_days,
            "UTC JD must round-trip to sub-microsecond"
        );
        // Consistency with the calendar constructor for the same instant.
        let from_calendar = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        assert!((e - from_calendar).abs() < 1.0e-6);

        // The TT reading of the same epoch is offset by TT−UTC =
        // 37 s + 32.184 s = 69.184 s on that date. f64 JD quantization is one
        // ulp ≈ 40 µs per accessor here, hence the 1e-4 s tolerance.
        let offset_s = (e.to_jde_tt_days() - e.to_jde_utc_days()) * 86400.0;
        assert!(
            (offset_s - 69.184).abs() < 1.0e-4,
            "TT JD reading must be offset by TT−UTC, got {offset_s} s"
        );

        // And the TT-scale JD constructor lands on the same instant.
        let e_tt = Epoch::from_jde_tt(e.to_jde_tt_days());
        assert!((e_tt - e).abs() < 1.0e-4);
    }

    /// Spec: "Interval across a leap second" — the 37th leap second was
    /// inserted at the end of 2016-12-31 (IERS Bulletin C), so the wall clock
    /// read 23:59:60 between these two stamps: true elapsed time is 2 s.
    #[test]
    fn interval_across_2016_leap_second_is_two_seconds() {
        let before = Epoch::from_gregorian_utc(2016, 12, 31, 23, 59, 59, 0);
        let after = Epoch::from_gregorian_utc(2017, 1, 1, 0, 0, 0, 0);
        assert_eq!(after - before, 2.0, "leap second must count as elapsed");
        assert_eq!(before - after, -2.0);
    }

    /// Spec: "Stepping an epoch by dt" — `(e + 60.0) − e == 60.0` exactly.
    #[test]
    fn stepping_by_dt_round_trips_exactly() {
        let e = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let stepped = e + 60.0;
        assert_eq!(stepped - e, 60.0);
    }

    /// Spec: "Serde round trip" — ISO-8601 UTC string with an explicit scale
    /// designator; nanosecond payload survives exactly.
    #[test]
    fn serde_round_trip_is_exact() {
        let e = Epoch::from_gregorian_utc(2021, 6, 15, 12, 34, 56, 123_456_789);
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#""2021-06-15T12:34:56.123456789 UTC""#);
        let back: Epoch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
        assert_eq!(back - e, 0.0);
    }

    /// Whole-second epochs serialize without a fractional part and still
    /// carry the explicit scale designator.
    #[test]
    fn serde_whole_second_form() {
        let e = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, r#""2020-01-01T00:00:00 UTC""#);
        let back: Epoch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    /// `from_iso8601` accepts scale designators, `Z`, and bare datetimes
    /// (bare = UTC), and rejects garbage.
    #[test]
    fn iso8601_parse_variants() {
        let base = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let utc = Epoch::from_iso8601("2020-01-01T00:00:00 UTC").unwrap();
        let zulu = Epoch::from_iso8601("2020-01-01T00:00:00Z").unwrap();
        let bare = Epoch::from_iso8601("2020-01-01T00:00:00").unwrap();
        assert_eq!(utc, base);
        assert_eq!(zulu, base);
        assert_eq!(bare, base);
        assert!(Epoch::from_iso8601("not a date").is_err());
    }

    /// Display matches the serde string form (scale-explicit UTC).
    #[test]
    fn display_matches_serde_form() {
        let e = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        assert_eq!(e.to_string(), "2020-01-01T00:00:00 UTC");
    }

    /// The relative-seconds `Timestamp` lane is untouched by `Epoch`.
    #[test]
    fn timestamp_dt_still_relative_seconds() {
        assert_eq!(Timestamp(12.5).dt(&Timestamp(10.0)), 2.5);
    }
}
