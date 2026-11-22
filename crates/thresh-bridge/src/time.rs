//! Checked translation from physical epochs to Python's civil UTC timestamp.

use thresh_core::time::Epoch;

use crate::error::{BridgeError, BridgeResult};

const NANOS_PER_SECOND: i128 = 1_000_000_000;
const SECONDS_PER_DAY: i128 = 86_400;

/// UTC components for a timezone-aware Python datetime, after ties-even rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtcDateTime {
    /// Gregorian year, in 1..=9999.
    pub year: i32,
    /// Month, in 1..=12.
    pub month: u8,
    /// Day of month, starting at 1.
    pub day: u8,
    /// Hour, in 0..=23.
    pub hour: u8,
    /// Minute, in 0..=59.
    pub minute: u8,
    /// Second, in 0..=59; leap seconds cannot be represented.
    pub second: u8,
    /// Rounded fraction, in 0..=999999.
    pub microsecond: u32,
}

/// Validated timestamp and lossless source metadata; independent of Python.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationTime {
    /// Rounded civil observation timestamp.
    pub calendar: UtcDateTime,
    /// Original relative seconds, never replaced by a rounded offset.
    pub relative_seconds: f64,
    /// Absolute origin rendered by core Epoch at nanosecond precision.
    pub origin_utc: String,
    /// Absolute observation instant before microsecond rounding.
    pub observation_utc: String,
    /// Separate arrival time, if supplied; never used to retime the observation.
    pub arrival_utc: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct Calendar {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    nanos: u32,
}

fn time_error(message: impl Into<String>) -> BridgeError {
    BridgeError::ConversionError(format!("observation time: {}", message.into()))
}

/// Validate without attaching to or importing Python.
pub(crate) fn observation_time(
    origin: Epoch,
    relative_seconds: f64,
    arrival: Option<Epoch>,
) -> BridgeResult<ObservationTime> {
    let (origin_calendar, origin_utc) = checked_calendar(origin)?;
    validate_offset_bounds(origin, relative_seconds)?;
    let observation = origin + relative_seconds;
    let (observation_calendar, observation_utc) = checked_calendar(observation)?;
    validate_physical_interval(origin, origin_calendar, observation, observation_calendar)?;
    let calendar = round_datetime(observation, observation_calendar)?;
    let arrival_utc = arrival
        .map(checked_calendar)
        .transpose()?
        .map(|(_, text)| text);
    Ok(ObservationTime {
        calendar,
        relative_seconds,
        origin_utc,
        observation_utc,
        arrival_utc,
    })
}

fn validate_offset_bounds(origin: Epoch, offset: f64) -> BridgeResult<()> {
    if !offset.is_finite() {
        return Err(time_error("relative seconds must be finite"));
    }
    // Bounds are computed before Epoch's unchecked Add<f64>. The complete
    // Python calendar range is small relative to hifitime's duration capacity.
    let lower = Epoch::from_gregorian_utc(1, 1, 1, 0, 0, 0, 0);
    let upper = Epoch::from_gregorian_utc(9999, 12, 31, 23, 59, 59, 999_999_999);
    if offset < lower - origin || offset > upper - origin {
        return Err(time_error("offset exceeds supported years 1 through 9999"));
    }
    Ok(())
}

fn checked_calendar(epoch: Epoch) -> BridgeResult<(Calendar, String)> {
    let text = epoch.to_string();
    let calendar = parse_calendar(&text)?;
    let reconstructed = Epoch::from_iso8601(&text)
        .map_err(|error| time_error(format!("invalid UTC calendar: {error}")))?;
    // Epoch's renderer can fold the inserted leap second onto an ordinary
    // second. String inspection alone therefore cannot establish validity.
    if reconstructed != epoch {
        return Err(time_error("UTC leap-second instant is not representable"));
    }
    Ok((calendar, text))
}

fn parse_calendar(text: &str) -> BridgeResult<Calendar> {
    let bare = text
        .strip_suffix(" UTC")
        .ok_or_else(|| time_error("expected an explicit UTC calendar"))?;
    let (date, time) = bare
        .split_once('T')
        .ok_or_else(|| time_error("missing calendar separator"))?;
    let mut dates = date.rsplitn(3, '-');
    let day = parse_component::<u8>(dates.next())?;
    let month = parse_component::<u8>(dates.next())?;
    let year = parse_component::<i32>(dates.next())?;
    let mut times = time.split(':');
    let hour = parse_component::<u8>(times.next())?;
    let minute = parse_component::<u8>(times.next())?;
    let second_text = times.next().ok_or_else(|| time_error("missing seconds"))?;
    let (second, nanos) = parse_second(second_text)?;
    let calendar = Calendar {
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanos,
    };
    validate_calendar(calendar)?;
    Ok(calendar)
}

fn parse_component<T: std::str::FromStr>(text: Option<&str>) -> BridgeResult<T> {
    text.and_then(|part| part.parse().ok())
        .ok_or_else(|| time_error("invalid calendar component"))
}

fn parse_second(text: &str) -> BridgeResult<(u8, u32)> {
    let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
    let second = parse_component::<u8>(Some(whole))?;
    if fraction.is_empty() {
        return Ok((second, 0));
    }
    if fraction.len() > 9 || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(time_error("calendar fraction exceeds nanosecond precision"));
    }
    let nanos = parse_component::<u32>(Some(fraction))? * 10u32.pow(9 - fraction.len() as u32);
    Ok((second, nanos))
}

fn validate_calendar(calendar: Calendar) -> BridgeResult<()> {
    if !(1..=9999).contains(&calendar.year)
        || !(1..=12).contains(&calendar.month)
        || calendar.day == 0
        || calendar.day > days_in_month(calendar.year, calendar.month)
        || calendar.hour > 23
        || calendar.minute > 59
        || calendar.second > 59
    {
        return Err(time_error(
            "calendar must be an ordinary UTC instant in years 1 through 9999",
        ));
    }
    Ok(())
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 31,
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Exact integer civil elapsed time, deliberately without leap-second insertion.
fn civil_nanos(calendar: Calendar) -> i128 {
    let prior_years = i128::from(calendar.year - 1);
    let prior_days = 365 * prior_years + prior_years / 4 - prior_years / 100 + prior_years / 400;
    let month_days: i128 = (1..calendar.month)
        .map(|month| i128::from(days_in_month(calendar.year, month)))
        .sum();
    let days = prior_days + month_days + i128::from(calendar.day - 1);
    let seconds = days * SECONDS_PER_DAY
        + i128::from(calendar.hour) * 3600
        + i128::from(calendar.minute) * 60
        + i128::from(calendar.second);
    seconds * NANOS_PER_SECOND + i128::from(calendar.nanos)
}

fn validate_physical_interval(
    origin: Epoch,
    origin_calendar: Calendar,
    observation: Epoch,
    observation_calendar: Calendar,
) -> BridgeResult<()> {
    let civil_elapsed = civil_nanos(observation_calendar) - civil_nanos(origin_calendar);
    let physical_elapsed = observation - origin;
    let civil_seconds = (civil_elapsed / NANOS_PER_SECOND) as f64
        + (civil_elapsed % NANOS_PER_SECOND) as f64 / NANOS_PER_SECOND as f64;
    // The public Epoch difference is f64. Across years 1..9999 its maximum
    // spacing is 61.1 microseconds; 0.25 ms safely covers the two rounded
    // readings but is far below even the smallest UTC leap discontinuity.
    if (physical_elapsed - civil_seconds).abs() > 0.000_25 {
        return Err(time_error(
            "origin-to-observation interval crosses a UTC leap discontinuity",
        ));
    }
    Ok(())
}

fn round_datetime(epoch: Epoch, mut calendar: Calendar) -> BridgeResult<UtcDateTime> {
    let micros = calendar.nanos / 1000;
    let remainder = calendar.nanos % 1000;
    let increment = remainder > 500 || (remainder == 500 && !micros.is_multiple_of(2));
    let mut microsecond = micros + u32::from(increment);
    if microsecond == 1_000_000 {
        carry_second(&mut calendar);
        microsecond = 0;
    }
    validate_calendar(calendar)?;
    let rounded = Epoch::from_gregorian_utc(
        calendar.year,
        calendar.month,
        calendar.day,
        calendar.hour,
        calendar.minute,
        calendar.second,
        microsecond * 1000,
    );
    if (rounded - epoch).abs() > 0.000_000_501 {
        return Err(time_error(
            "microsecond rounding crosses an unrepresentable UTC leap second",
        ));
    }
    Ok(UtcDateTime {
        year: calendar.year,
        month: calendar.month,
        day: calendar.day,
        hour: calendar.hour,
        minute: calendar.minute,
        second: calendar.second,
        microsecond,
    })
}

fn carry_second(calendar: &mut Calendar) {
    calendar.second += 1;
    if calendar.second < 60 {
        return;
    }
    calendar.second = 0;
    calendar.minute += 1;
    if calendar.minute < 60 {
        return;
    }
    calendar.minute = 0;
    calendar.hour += 1;
    if calendar.hour < 24 {
        return;
    }
    calendar.hour = 0;
    carry_day(calendar);
}

fn carry_day(calendar: &mut Calendar) {
    calendar.day += 1;
    if calendar.day <= days_in_month(calendar.year, calendar.month) {
        return;
    }
    calendar.day = 1;
    calendar.month += 1;
    if calendar.month <= 12 {
        return;
    }
    calendar.month = 1;
    calendar.year += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch(text: &str) -> Epoch {
        Epoch::from_iso8601(text).unwrap()
    }

    #[test]
    fn timestamps_preserve_offsets_order_and_arrival() {
        let origin = epoch("2020-01-01T00:00:00 UTC");
        let later = observation_time(origin, 1.25, None).unwrap();
        let delayed = observation_time(origin, -0.5, Some(origin + 10.0)).unwrap();
        let initial = observation_time(origin, 0.0, None).unwrap();
        assert_eq!(
            (later.calendar.second, later.calendar.microsecond),
            (1, 250_000)
        );
        assert_eq!(
            (
                delayed.calendar.year,
                delayed.calendar.day,
                delayed.calendar.second
            ),
            (2019, 31, 59)
        );
        assert_eq!(delayed.calendar.microsecond, 500_000);
        assert_eq!(delayed.relative_seconds, -0.5);
        assert_eq!(delayed.origin_utc, origin.to_string());
        assert_eq!(delayed.arrival_utc, Some((origin + 10.0).to_string()));
        assert_eq!(initial.calendar.second, 0);
    }

    #[test]
    fn timestamps_round_ties_even_and_keep_colliding_source_values() {
        let origin = epoch("2020-01-01T00:00:00 UTC");
        for (nanos, expected) in [(500, 0), (1500, 2), (2500, 2), (3500, 4)] {
            let instant = epoch(&format!("2020-01-01T00:00:00.{nanos:09} UTC"));
            assert_eq!(
                observation_time(instant, 0.0, None)
                    .unwrap()
                    .calendar
                    .microsecond,
                expected
            );
        }
        let a = observation_time(origin, 0.000_000_1, None).unwrap();
        let b = observation_time(origin, 0.000_000_2, None).unwrap();
        assert_eq!(a.calendar, b.calendar);
        assert_ne!(a.relative_seconds, b.relative_seconds);
        assert_ne!(a.observation_utc, b.observation_utc);
    }

    #[test]
    fn timestamps_round_checked_calendar_carries() {
        let cases = [
            ("2020-02-28T23:59:59.999999500 UTC", (2020, 2, 29)),
            ("2020-02-29T23:59:59.999999500 UTC", (2020, 3, 1)),
            ("2021-12-31T23:59:59.999999500 UTC", (2022, 1, 1)),
        ];
        for (input, expected) in cases {
            let rounded = observation_time(epoch(input), 0.0, None).unwrap().calendar;
            assert_eq!((rounded.year, rounded.month, rounded.day), expected);
            assert_eq!(
                (
                    rounded.hour,
                    rounded.minute,
                    rounded.second,
                    rounded.microsecond
                ),
                (0, 0, 0, 0)
            );
        }
    }

    #[test]
    fn timestamps_reject_nonfinite_overflow_and_outside_year_bounds() {
        let origin = epoch("2020-01-01T00:00:00 UTC");
        for offset in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            -f64::MAX,
        ] {
            assert!(observation_time(origin, offset, None).is_err());
        }
        for input in [
            "0000-12-31T00:00:00 UTC",
            "10000-01-01T00:00:00 UTC",
            "9999-12-31T23:59:59.999999500 UTC",
        ] {
            assert!(
                observation_time(epoch(input), 0.0, None).is_err(),
                "{input}"
            );
        }
        let earliest = epoch("0001-01-01T00:00:00 UTC");
        assert_eq!(
            observation_time(earliest, 0.0, None).unwrap().calendar.year,
            1
        );
        assert!(observation_time(earliest, -0.000_000_001, None).is_err());
        let latest = epoch("9999-12-31T23:59:59.999999 UTC");
        assert_eq!(
            observation_time(latest, 0.0, None).unwrap().calendar.year,
            9999
        );
        assert!(observation_time(latest, 1.0, None).is_err());
    }

    #[test]
    fn timestamps_reject_hidden_leap_origins_and_both_crossing_directions() {
        let before = epoch("2016-12-31T23:59:59 UTC");
        let after = epoch("2017-01-01T00:00:00 UTC");
        let leap = before + 1.0;
        for origin in [leap, leap + 0.25] {
            for offset in [0.0, 0.25, -0.25, 2.0] {
                assert!(observation_time(origin, offset, None).is_err());
            }
        }
        assert!(observation_time(before, 1.0, None).is_err());
        assert!(observation_time(before, 2.0, None).is_err());
        assert!(observation_time(after, -2.0, None).is_err());
        assert!(observation_time(before, 0.0, Some(leap)).is_err());
        assert!(observation_time(before, 0.999_999_5, None).is_err());
        assert!(observation_time(after, 0.0, Some(after + 1.0)).is_ok());
    }

    #[test]
    fn timestamps_accept_large_nonleap_intervals_without_float_cancellation() {
        let origin = epoch("3000-01-01T00:00:00.123456789 UTC");
        let result = observation_time(origin, 100_000_000_000.0, None).unwrap();
        assert_eq!(result.calendar.microsecond, 123_457);
    }
}
