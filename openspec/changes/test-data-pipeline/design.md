## Context

Thresh has a working synthetic data pipeline and tracker but needs real-world data validation. The three target domains (aviation, orbital, automotive) each have public data sources with different formats, coordinate systems, and access patterns. The design must handle this heterogeneity while presenting a uniform interface to the tracker and evaluation pipeline.

## Goals / Non-Goals

**Goals:**
- Ingest ADS-B data (OpenSky REST API, ADS-B Exchange SBS format)
- Ingest orbital TLEs (space-track.org) with SGP4 propagation
- Ingest nuScenes multi-modal data (via Python devkit bridge)
- Unified `Dataset` trait over all sources including synthetic
- Benchmark scenario catalog with regression testing
- Local caching to avoid redundant downloads

**Non-Goals:**
- Real-time streaming data ingestion (batch only for now)
- Custom radar data formats (no public radar datasets exist)
- Training data pipeline (this is evaluation/testing only)
- Hosting or redistributing any dataset

## Decisions

### 1. New `thresh-data` crate

**Decision:** Add a new `thresh-data` crate to the workspace for all data ingestion and dataset abstraction.

**Rationale:** Data ingestion has unique dependencies (HTTP client, CSV parsing, SGP4, optional nuScenes Python bridge) that don't belong in the core tracking crates. Feature flags isolate heavy dependencies.

### 2. Feature-gated data sources

**Decision:** Each data source is behind a Cargo feature:
- `adsb` — enables OpenSky/ADS-B Exchange ingestion (reqwest, csv)
- `orbital` — enables space-track.org ingestion (reqwest, sgp4)
- `nuscenes` — enables nuScenes ingestion (pyo3, requires Python devkit)
- Default: only the `Dataset` trait and synthetic adapter are available

**Rationale:** Users running only synthetic tests don't need HTTP clients or Python. CI can test without network access.

### 3. Coordinate normalization strategy

**Decision:** All data sources convert to ENU (East-North-Up) local tangent plane relative to a configurable reference point.

**Rationale:** ENU is the standard defense tracking frame, works for all domains. Aviation: reference = airport or scene center. Orbital: reference = ground station. Automotive: reference = ego vehicle start position. The ENU conversion happens at the data source level, before the `Dataset` trait.

### 4. Credential management

**Decision:** Read credentials from environment variables (`SPACETRACK_USER`, `SPACETRACK_PASS`, `OPENSKY_USER`, `OPENSKY_PASS`) or a `~/.thresh/credentials.toml` file. Never store credentials in the repo.

### 5. SGP4 in Rust (not Python)

**Decision:** Use the `sgp4` Rust crate for orbit propagation rather than bridging to Python.

**Rationale:** SGP4 is a well-defined algorithm with a mature Rust implementation. No need for Python overhead. The `sgp4` crate handles TEME→ECEF transforms.

### 6. Async HTTP with blocking adapter

**Decision:** Use `reqwest::blocking` for data downloads (not async). Provide progress reporting via callback.

**Rationale:** Data download is a batch operation, not latency-sensitive. Blocking API is simpler and avoids async runtime dependency in the library.

## Risks / Trade-offs

**[Risk] OpenSky API rate limits** → Mitigation: aggressive caching, download once per scenario, respect 429 backoff.

**[Risk] space-track.org availability** → Mitigation: cache TLEs locally, CelesTrak as backup for common objects.

**[Risk] nuScenes devkit Python version compatibility** → Mitigation: pin to known-good version, same PyO3 bridge pattern as Stone Soup.

**[Risk] Dataset size (nuScenes full = 350 GB)** → Mitigation: mini split (4 GB) sufficient for integration tests. Full split only for explicit benchmarking.

**[Trade-off] ENU vs ECEF** → ENU is local and intuitive but introduces errors for large areas. For orbital tracking over large arcs, ECEF may be needed. Design the coordinate transform as swappable.

## Open Questions

- Should the benchmark runner be a CLI binary or just a test harness?
- Should we support ASTERIX format (EUROCONTROL radar data) for future European radar data sources?
- What CI strategy for tests requiring network access? Skip by default, run in scheduled nightly CI?
