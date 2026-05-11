# thresh — Training Pipeline

Reproduction recipe for the trained ONNX checkpoints under `test-data/models/` that the inference pipeline consumes. This pipeline is tracked by the OpenSpec change `flight-data-training-pipeline`; see [`openspec/changes/flight-data-training-pipeline/`](openspec/changes/flight-data-training-pipeline/) for the full proposal, design, and tasks.

> **Status:** Phases 1–3 landed (toolchain + OpenSky + ADS-B Exchange acquisition). Training, export, and evaluation come in later phases — see the change's `tasks.md`. Until those land, this document covers Python environment bootstrap and acquisition usage only.

## Python tree layout

The training-side Python code lives under [`python/`](python/) — a separate concern from `crates/thresh-py` (which is the maturin-built Python binding for the Rust crates). The two have independent environments and `pyproject.toml` files.

```
python/
├── pyproject.toml         # training-side deps + ruff/pyright/pytest config
├── uv.lock                # pinned resolution (committed)
├── acquisition/           # OpenSky + ADSBx clients, canonical schema (Phases 2–3)
├── training/              # PyTorch training scripts (Phases 6–7)
├── export/                # ONNX export utilities (Phases 6–7)
├── eval/                  # MOT-metric evaluation harness (Phase 9)
└── tests/                 # smoke tests + per-module unit tests
```

## Bootstrap

Requires [`uv`](https://docs.astral.sh/uv/) (any recent version).

```sh
cd python
uv sync                          # install full env (training-side deps incl. torch)
uv sync --no-default-groups --group dev   # lightweight dev tooling only (ruff, pyright, pytest)
```

The `dev` group is what CI uses for the toolchain smoke test; the default groups + the `training` optional dependencies pull in `torch`, `onnx`, and `onnxruntime`, which are needed only when running the training and export scripts (Phases 6–7).

## Day-to-day commands

```sh
cd python

# Run all tests
uv run pytest

# Lint
uv run ruff check .

# Type-check
uv run pyright

# Auto-fix lint where possible
uv run ruff check . --fix
```

`pre-commit` is also wired to run `ruff` and `pyright` against `python/` on commit (see `.pre-commit-config.yaml`).

## Acquisition

### OpenSky Network (historical, redistributable)

The OpenSky public REST endpoint is rate-limited but credential-free. For larger pulls, register at <https://opensky-network.org/> and pass `(username, password)` as the `credentials` argument to `fetch_state_vectors`.

```python
from acquisition.opensky import fetch_state_vectors, BoundingBox, TimeRange

bbox = BoundingBox(lat_min=47.0, lat_max=48.0, lon_min=-123.0, lon_max=-122.0)  # KSEA-ish
time_range = TimeRange(start_s=..., end_s=...)
records = list(fetch_state_vectors(bbox, time_range))
```

For bulk historical use the published Zenodo trajectory dumps; SHA-256 verification is built in:

```python
from acquisition.opensky import load_zenodo_dump
records = list(load_zenodo_dump("opensky-traffic-2024.parquet", sha256="abc..."))
```

### ADS-B Exchange v2 (live, edge cases, military targets)

ADSBx requires an API key (RapidAPI marketplace listing or a direct ADSBx subscription). The acquisition layer supplies the key via the `x-rapidapi-key` header.

```sh
export ADSBX_API_KEY="your-key-here"
```

```python
import os
from pathlib import Path
from acquisition.adsbx_poller import run

result = run(
    airport_icao="KSEA",
    api_key=os.environ["ADSBX_API_KEY"],
    root=Path("./data"),
    rate_limit_hz=1.0,    # default; respect your plan's quota
    duration_s=3600.0,    # poll for one hour
)
print(f"Wrote {len(result.files_written)} parquet files; "
      f"{result.stats.records_appended} records, "
      f"{result.stats.records_deduplicated} duplicates dropped.")
```

Output layout: `<root>/airport=KSEA/source=adsbx/date=YYYY-MM-DD/trajectories.parquet` (one file per UTC day).

**Do not commit ADSBx output to git.** Per the ADSBx terms of service the raw feed is not redistributable; this repository ships only the acquisition recipe. See "License posture" below.

## License posture (preview)

OpenSky-derived data ships with the repository under the OpenSky Network terms with attribution. ADS-B Exchange data is **not** redistributed; an acquisition script is provided and reproduction requires a developer-supplied ADSBx API key. Full attribution strings will land in `LICENSING.md` in Phase 10.
