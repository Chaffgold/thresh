# astro-time Specification (Delta)

## ADDED Requirements

### Requirement: Time-scale-aware epoch type
The system SHALL provide an absolute epoch type in `thresh-core` that carries its time scale explicitly, supporting at minimum UTC, TAI, GPS, and TT, with construction from UTC calendar components, ISO-8601 strings, and Julian dates of a stated scale, and accessors that return Julian dates in a stated scale. Constructing or reading an epoch SHALL never require the caller to guess the scale.

#### Scenario: Scale conversions agree with published offsets
- **WHEN** an epoch is constructed at 2020-01-01T00:00:00 UTC and read back in TAI, GPS, and TT
- **THEN** the TAI reading SHALL lead UTC by exactly 37 s, GPS SHALL lead UTC by exactly 18 s, and TT SHALL lead TAI by exactly 32.184 s

#### Scenario: Julian-date round trip in a stated scale
- **WHEN** an epoch is constructed from a Julian date in UTC and read back as a Julian date in UTC
- **THEN** the value SHALL round-trip within sub-microsecond tolerance, and reading the same epoch as a TT Julian date SHALL differ by the TT−UTC offset for that date

### Requirement: Leap-second correctness
Epoch arithmetic SHALL account for leap seconds: differences between epochs SHALL be true elapsed (TAI) seconds, including across leap-second insertions.

#### Scenario: Interval across a leap second
- **WHEN** the difference is taken between 2017-01-01T00:00:00 UTC and 2016-12-31T23:59:59 UTC (an interval containing the 2016-12-31 leap second)
- **THEN** the elapsed time SHALL be 2 seconds, not 1

### Requirement: Epoch serialization
The epoch type SHALL serialize through serde in a human-readable form that states its scale unambiguously and SHALL deserialize back to an identical epoch.

#### Scenario: Serde round trip
- **WHEN** an epoch is serialized to JSON and deserialized
- **THEN** the recovered epoch SHALL equal the original exactly, and the serialized form SHALL be a string a reader can interpret without consulting the code (e.g. ISO-8601 with an explicit scale designation)

### Requirement: Relative-time interoperability
The epoch type SHALL interoperate with the tracker lane's relative-seconds convention: subtracting two epochs SHALL yield elapsed seconds as `f64`, and adding a `f64` seconds offset to an epoch SHALL yield a new epoch, so propagation loops can step an absolute epoch with the same `dt` values trackers use.

#### Scenario: Stepping an epoch by dt
- **WHEN** an epoch is advanced by 60.0 seconds and the difference between the result and the original is taken
- **THEN** the difference SHALL be exactly 60.0 seconds
