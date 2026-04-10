## Capability: Benchmark Scenarios

### Overview
Curated real-data benchmark scenarios with known characteristics and expected metric baselines, enabling regression testing and cross-domain tracker validation.

## ADDED Requirements

### Requirement: Scenario catalog
Named scenarios with documented characteristics (target count, dynamics, sensor modality, difficulty level). The system MUST support this.

#### Scenario: List available benchmark scenarios
- Given the benchmark scenario catalog is initialized
- When the catalog is queried
- Then it returns all named scenarios with their characteristics including target count, dynamics type, sensor modality, and difficulty level

### Requirement: Expected baselines
Each scenario has MOTA/HOTA target ranges based on the data difficulty. The system MUST support this.

#### Scenario: Retrieve metric baselines for a scenario
- Given a named benchmark scenario
- When its expected baselines are queried
- Then it returns MOTA and HOTA target ranges appropriate to the scenario difficulty

### Requirement: Automated download
`thresh-data fetch <scenario-name>` downloads and caches required data. The system MUST support this.

#### Scenario: Download a benchmark scenario by name
- Given a valid scenario name
- When `thresh-data fetch <scenario-name>` is executed
- Then the required data is downloaded and cached locally, ready for use

### Requirement: Benchmark runner
Execute tracker against a scenario, compute metrics, and compare to baselines. The system MUST support this.

#### Scenario: Run tracker against a benchmark and compare metrics
- Given a configured tracker and a benchmark scenario with expected baselines
- When the benchmark runner is invoked
- Then it executes the tracker, computes MOTA/HOTA metrics, and reports whether results meet the baseline thresholds

### Requirement: Regression detection
Flag when metrics drop below baseline thresholds. The system MUST support this.

#### Scenario: Detect metric regression
- Given a benchmark run that produces metrics below the baseline thresholds
- When regression detection is applied
- Then it flags the regression with the specific metrics and thresholds that were violated

### Scenario Catalog

**Aviation (ADS-B)**
| Scenario | Source | Targets | Duration | Difficulty | Notes |
|---|---|---|---|---|---|
| `adsb-single-flight` | OpenSky | 1 | 4 hrs | Easy | Transcontinental, clean trajectory |
| `adsb-approach` | OpenSky | 10-30 | 1 hr | Medium | Airport approach, merging tracks |
| `adsb-tracon` | OpenSky | 50+ | 1 hr | Hard | High-density terminal area |
| `adsb-oceanic` | OpenSky | 5-10 | 6 hrs | Medium | Sparse reports, long coast periods |

**Orbital (TLE/SGP4)**
| Scenario | Source | Targets | Duration | Difficulty | Notes |
|---|---|---|---|---|---|
| `orbital-iss` | space-track | 1 | 2 orbits | Easy | Well-known LEO, fast angular rate |
| `orbital-geo-cluster` | space-track | 5-10 | 24 hrs | Medium | GEO belt, slow drift, close proximity |
| `orbital-starlink-train` | space-track | 20-60 | 1 orbit | Hard | Similar orbits, association challenge |
| `orbital-conjunction` | space-track | 2 | 1 hr | Hard | Close approach, high relative velocity |

**Automotive (nuScenes)**
| Scenario | Source | Targets | Duration | Difficulty | Notes |
|---|---|---|---|---|---|
| `nuscenes-mini` | nuScenes mini | 10-30 | 20s | Medium | Multi-class urban, full multi-modal |
| `nuscenes-night` | nuScenes full | 5-15 | 20s | Hard | Degraded camera, tests LiDAR-only fallback |
| `nuscenes-rain` | nuScenes full | 10-20 | 20s | Hard | Degraded both modalities |

**Synthetic (thresh-synth baseline)**
| Scenario | Source | Targets | Duration | Difficulty | Notes |
|---|---|---|---|---|---|
| `synth-cv-clean` | thresh-synth | 10 | 60s | Easy | Perfect detection, CV motion |
| `synth-maneuvering` | thresh-synth | 5 | 30s | Medium | CTRV + CA segments |
| `synth-heterogeneous` | thresh-synth | 10 | 60s | Hard | Mixed aircraft + ballistic + UAV |
| `synth-low-pd` | thresh-synth | 10 | 60s | Hard | P_d = 0.6, heavy clutter |

### Metric Baselines (initial targets, refined after first runs)
- Easy: MOTA > 0.9, HOTA > 0.8
- Medium: MOTA > 0.7, HOTA > 0.6
- Hard: MOTA > 0.4, HOTA > 0.3
