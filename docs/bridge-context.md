# Observation context and timestamp contract

The boundary requires caller-owned time, geometry and uncertainty. No context
type implements `Default`: an absolute origin, frame origin, profile and full
covariance must be supplied. Missing required fields are compile-time errors;
conflicting or invalid supplied fields return `BridgeError::ConversionError`
from `DetectionContext::validate` before Python access.

The common frame is fixed WGS84 east/north/up, with geodetic origin latitude and
longitude in **radians** and height above the **ellipsoid**, not MSL. Its state
order is `[x,vx,y,vy,z,vz]` in metres and metres/second. Inertial, unspecified,
and moving-sensor frames are explicitly unsupported. Radar/EOIR registration
positions use that common frame; rotation maps sensor axes into common axes.
The rotation must be finite, proper and orthogonal within an absolute `1e-10`
element/determinant tolerance. Sensor IDs must match.

Source covariance is explicit and includes all correlations:

| Profile | Source covariance component order | Additional declaration |
|---|---|---|
| Radar3 | range, azimuth, elevation | Stationary sensor pose |
| Radar4 | range, azimuth, elevation, range rate | Pose and Doppler sign |
| Eoir2 | azimuth, elevation | Stationary sensor pose |
| Adsb3 | east, north, up position error at report | Geoid separation, metric-noise tag |
| Adsb6 | report-local ENU position error, then velocity error | Datum/noise and velocity frame |

Radar ranges are metres, angles radians and range rates metres/second. Finite
azimuth may wrap; elevation is restricted to `[-pi/2,pi/2]`. ADS-B reports keep
their core degree-valued latitude/longitude and MSL altitude at input. Geoid
separation is **ellipsoid height minus MSL height**, so ellipsoid height equals
MSL height plus separation. Velocity axes must be explicitly ECEF or ENU at the
report position. Position noise is always metric ENU at the report position,
never degree-valued latitude/longitude covariance. For Adsb6, velocity noise
uses the declared velocity axes, including position/velocity cross terms.
Callers must include applicable datum/registration uncertainty in their noise;
the bridge neither estimates it nor downloads a geoid model.

The covariance validator uses a scale-aware `1e-12 * max(1, max-row-sum(abs(R)))`
symmetry/eigenvalue tolerance with overflow-safe arithmetic. It does not repair,
symmetrize, jitter, or replace noise. A boundary-valid singular PSD covariance
does not imply that a later algorithm can invert it.

## All five profile examples

This example is compiled and executed as the `context` module's documentation
test with the `stonesoup` feature. Context validation itself never initializes
or imports Python. The measurements and noise below are entirely invented.

```rust
use nalgebra::{DMatrix, Vector3};
use thresh_bridge::context::{
    AdsbDatum, AdsbNoiseConvention, CommonFrame, DetectionContext, DopplerSign,
    EnuOrigin, ObservationProfile, SensorPose, VelocityFrame,
};
use thresh_core::{measurement::Measurement, sensor::SensorRegistration, time::Epoch};

let origin = Epoch::from_iso8601("2020-01-01T00:00:00 UTC")?;
let frame = CommonFrame::Wgs84Enu(EnuOrigin {
    latitude_rad: 0.0,
    longitude_rad: 0.0,
    ellipsoid_height_m: 0.0,
});
let sensor = SensorPose::Stationary(SensorRegistration::new(Vector3::zeros(), 7));
let datum = AdsbDatum {
    geoid_separation_m: 20.0,
    noise: AdsbNoiseConvention::MetricEnuAtReport,
};
let profiles = [
    ObservationProfile::Radar3 { sensor: sensor.clone() },
    ObservationProfile::Radar4 {
        sensor: sensor.clone(), doppler_sign: DopplerSign::PositiveAway,
    },
    ObservationProfile::Eoir2 { sensor },
    ObservationProfile::Adsb3 { datum },
    ObservationProfile::Adsb6 { datum, velocity_frame: VelocityFrame::Ecef },
];
let measurements = [
    Measurement::Radar { range: 100.0, azimuth: 0.2, elevation: 0.1,
        range_rate: None, time: 1.25, sensor_id: 7 },
    Measurement::Radar { range: 100.0, azimuth: 0.2, elevation: 0.1,
        range_rate: Some(-3.0), time: 1.25, sensor_id: 7 },
    Measurement::EoIr { azimuth: 0.2, elevation: 0.1, time: 1.25, sensor_id: 7 },
    Measurement::AdsB { lat: 0.0, lon: 0.0, alt: 100.0,
        velocity: None, time: 1.25 },
    Measurement::AdsB { lat: 0.0, lon: 0.0, alt: 100.0,
        velocity: Some([1.0, 2.0, 3.0]), time: 1.25 },
];
for (profile, measurement) in profiles.into_iter().zip(measurements) {
    let dimension = profile.dimension();
    let context = DetectionContext {
        origin,
        arrival: Some(origin + 10.0),
        frame,
        profile,
        covariance: DMatrix::identity(dimension, dimension),
    };
    let time = context.validate(&measurement)?;
    assert_eq!(time.calendar.second, 1);
    assert_eq!(time.calendar.microsecond, 250_000);
    assert_eq!(time.relative_seconds, 1.25);
    assert_eq!(time.arrival_utc, Some((origin + 10.0).to_string()));
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The required origin cannot be omitted:

```compile_fail
use nalgebra::DMatrix;
use thresh_bridge::context::*;
let context = DetectionContext {
    arrival: None,
    frame: CommonFrame::Unspecified,
    profile: ObservationProfile::Adsb3 {
        datum: AdsbDatum {
            geoid_separation_m: 0.0,
            noise: AdsbNoiseConvention::MetricEnuAtReport,
        },
    },
    covariance: DMatrix::identity(3, 3),
}; // Error: missing required field `origin`.
```

Likewise a velocity profile cannot omit its frame, nor a Doppler profile its sign:

```compile_fail
use thresh_bridge::context::*;
let profile = ObservationProfile::Adsb6 {
    datum: AdsbDatum {
        geoid_separation_m: 20.0,
        noise: AdsbNoiseConvention::MetricEnuAtReport,
    },
}; // Error: missing required field `velocity_frame`.
```

```compile_fail
use nalgebra::Vector3;
use thresh_bridge::context::*;
use thresh_core::sensor::SensorRegistration;
let profile = ObservationProfile::Radar4 {
    sensor: SensorPose::Stationary(SensorRegistration::new(Vector3::zeros(), 7)),
}; // Error: missing required field `doppler_sign`.
```

Explicit unsupported contexts return the documented category without Python:

```rust
use nalgebra::DMatrix;
use thresh_bridge::{context::*, error::BridgeError};
use thresh_core::{measurement::Measurement, time::Epoch};
let context = DetectionContext {
    origin: Epoch::from_iso8601("2020-01-01T00:00:00 UTC")?,
    arrival: None,
    frame: CommonFrame::Unspecified,
    profile: ObservationProfile::Adsb3 {
        datum: AdsbDatum {
            geoid_separation_m: 20.0,
            noise: AdsbNoiseConvention::MetricEnuAtReport,
        },
    },
    covariance: DMatrix::identity(3, 3),
};
let report = Measurement::AdsB {
    lat: 0.0, lon: 0.0, alt: 100.0, velocity: None, time: 0.0,
};
assert!(matches!(context.validate(&report), Err(BridgeError::ConversionError(_))));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Time and preserved metadata

Relative measurement seconds are physical elapsed time from `origin`. They are
not Unix seconds. Positive and negative finite offsets are supported, and
conversion is stateless: delayed observations are not sorted, rejected solely
for being late, or retimed to arrival. `ObservationTime` retains the original
relative float and nanosecond UTC strings for origin, unrounded observation and
optional arrival separately. The detection builder uses those values as metadata
while putting the actual rounded observation time in `Detection.timestamp`.

Python datetime is limited to years 1–9999 and microseconds. Rounding is nearest
microsecond with ties to even, including checked carries across calendar days.
Two observations may consequently have the same timestamp; original source
times remain distinct and no artificial increment is added. Nonfinite offsets,
out-of-range results and rounding beyond year 9999 are conversion errors.

UTC leap-second instants cannot be represented. Both origin and observation are
validated using a lossless Epoch UTC render/parse identity check, including the
case where core's renderer folds a leap instant into an ordinary second. The
origin-to-observation interval must not cross a UTC leap discontinuity in either
direction. Rounding into a leap second is rejected too. Optional arrival time
must itself be a representable ordinary UTC instant, but is not constrained to
the origin's side of a leap discontinuity. Core Epoch precision and physical
arithmetic are unchanged; these are limitations of the Python boundary only.
