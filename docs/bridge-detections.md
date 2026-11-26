# Context-aware Stone Soup detections

The boundary API is now `measurement_to_detection(&Measurement, &DetectionContext)`.
The old Python-token/context-free signature is removed: it could not determine
absolute time, sensor pose, noise, or ADS-B datum/velocity semantics. Initialize
Python in the host as described in [bridge-runtime.md](bridge-runtime.md), and
supply the units and explicit metadata documented in [bridge-context.md](bridge-context.md).
Structured Python errors also replace the old blanket `From<PyErr>` conversion.

The common state is `[x,vx,y,vy,z,vz]` in a fixed WGS84 local ENU frame. Successful
objects carry an actual timezone-aware UTC `timestamp`, a float64 column
`state_vector`, and a `measurement_model` with transformed `noise_covar`:

| Profile | Observation order | State mapping |
|---|---|---|
| Radar3 | elevation, azimuth, range | 0,2,4 |
| Radar4 | elevation, azimuth, range, positive-away range rate | position 0,2,4; velocity 1,3,5 |
| Eoir2 | elevation, azimuth | 0,2,4 |
| Adsb3 | east, north, up | 0,2,4 |
| Adsb6 | east, north, up, east-rate, north-rate, up-rate | 0,2,4,1,3,5 |

All five profiles can be built from wholly invented observations:

```rust
use nalgebra::{DMatrix, Vector3};
use pyo3::prelude::*;
use thresh_bridge::{context::*, convert, detection::measurement_to_detection};
use thresh_core::{measurement::Measurement, sensor::SensorRegistration, time::Epoch};

Python::initialize(); // application-owned startup, never done by the library
let sensor = SensorPose::Stationary(SensorRegistration::new(Vector3::zeros(), 7));
let datum = AdsbDatum {
    geoid_separation_m: 20.0,
    noise: AdsbNoiseConvention::MetricEnuAtReport,
};
let profiles = [
    ObservationProfile::Radar3 { sensor: sensor.clone() },
    ObservationProfile::Radar4 { sensor: sensor.clone(), doppler_sign: DopplerSign::PositiveAway },
    ObservationProfile::Eoir2 { sensor },
    ObservationProfile::Adsb3 { datum: datum.clone() },
    ObservationProfile::Adsb6 { datum, velocity_frame: VelocityFrame::Ecef },
];
let measurements = [
    Measurement::Radar { range: 100.0, azimuth: 0.2, elevation: 0.1,
        range_rate: None, time: 1.25, sensor_id: 7 },
    Measurement::Radar { range: 100.0, azimuth: 0.2, elevation: 0.1,
        range_rate: Some(-3.0), time: 1.25, sensor_id: 7 },
    Measurement::EoIr { azimuth: 0.2, elevation: 0.1, time: 1.25, sensor_id: 7 },
    Measurement::AdsB { lat: 0.0, lon: 0.0, alt: 100.0, velocity: None, time: 1.25 },
    Measurement::AdsB { lat: 0.0, lon: 0.0, alt: 100.0,
        velocity: Some([1.0, 2.0, 3.0]), time: 1.25 },
];
for (profile, measurement) in profiles.into_iter().zip(measurements) {
    let n = profile.dimension();
    let context = DetectionContext {
        origin: Epoch::from_iso8601("2020-01-01T00:00:00 UTC").unwrap(),
        arrival: None,
        frame: CommonFrame::Wgs84Enu(EnuOrigin {
            latitude_rad: 0.0, longitude_rad: 0.0, ellipsoid_height_m: 0.0,
        }),
        profile,
        covariance: DMatrix::identity(n, n), // explicit invented source noise
    };
    let detection = measurement_to_detection(&measurement, &context).unwrap();
    Python::attach(|py| {
        let detection = detection.bind(py);
        let vector = detection.getattr("state_vector").unwrap();
        assert_eq!(vector.getattr("shape").unwrap().extract::<(usize, usize)>().unwrap(), (n, 1));
        assert_eq!(vector.getattr("dtype").unwrap().str().unwrap().to_str().unwrap(), "float64");
        let model = detection.getattr("measurement_model").unwrap();
        assert_eq!(model.getattr("ndim_state").unwrap().extract::<usize>().unwrap(), 6);
        assert_eq!(model.getattr("ndim_meas").unwrap().extract::<usize>().unwrap(), n);
        let mapping = if n == 6 { vec![0,2,4,1,3,5] } else { vec![0,2,4] };
        assert_eq!(model.getattr("mapping").unwrap().extract::<Vec<usize>>().unwrap(), mapping);
        let covariance = model.getattr("noise_covar").unwrap();
        assert_eq!(convert::numpy_to_dmatrix(py, &covariance).unwrap(), DMatrix::identity(n, n));
        assert_eq!(detection.getattr("timestamp").unwrap().call_method0("isoformat").unwrap()
            .extract::<String>().unwrap(), "2020-01-01T00:00:01.250000+00:00");
    });
}
```

Metadata preserves original source fields, relative `time`, absolute `origin_utc`,
`observation_utc_unrounded`, and optional `arrival_utc`. ADS-B remains explicitly
tagged as a `cooperative_report`; `alt_msl` is the original height, not the
normalized ellipsoid value. Late calls are not reordered or retimed. Distinct
observations may round to the same microsecond without artificial time increments.

Unsupported variants and Rust-side validation errors take precedence over Python
availability. Finite inputs producing nonfinite derived values are rejected.
Native core measurement vectors/defaults and the training pipeline's ADS-B role
are unchanged. Passing these examples certifies only boundary construction.
Separate [linear-Gaussian JPDA/GM-PHD reference checks](bridge-references.md) do
not certify these nonlinear models' algorithm results. The bridge provides JPDA
and GM-PHD references only; MHT and IMM contracts are native (`mht-tracking`,
`imm-estimation`) and have interim gaps. The missing-shared-library deployment
gap remains unresolved.
