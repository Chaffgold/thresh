# thresh — Training Pipeline

Reproduction recipe for the trained ONNX checkpoints under `test-data/models/` that the inference pipeline consumes. This pipeline is tracked by the OpenSpec change `flight-data-training-pipeline`; see [`openspec/changes/flight-data-training-pipeline/`](openspec/changes/flight-data-training-pipeline/) for the full proposal, design, and tasks.

> **Status:** Phases 1–7 landed — acquisition (OpenSky + ADS-B Exchange), the trajectory-driven synth pairing, and the full training/export scaffolding for **both** learned components (Track B IMM mode classifier, Track A detector). The real GPU training runs (and the trained checkpoints that replace the random-weight stubs) are deferred — training is a non-CI goal (design Decision 7) and is gated on the full external dataset + GPU hardware. The end-to-end evaluation harness (Phase 9) is the remaining piece. The checked-in `test-data/models/*.onnx` are **random-weight stubs** for shape contracts; see `test-data/models/MODEL_CARD.md`.

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

## Synthetic dataset generation (Phase 4–5/7)

Both learned components train on **synthetic perception driven by real truth**: a real ADS-B trajectory is the truth target, and `thresh-synth`'s radar simulator turns it into sensor returns (design Decision 3). ADS-B is consumed as system-level truth, **never** as a measurement. The Rust side generates the training Parquet (behind the `training-export` Cargo feature on the umbrella `thresh` crate):

```sh
# Track B — IMM filter-state features paired with analytic mode labels:
cargo run -p thresh --features training-export --bin gen-imm-dataset -- \
  test-data/training/imm-classifier/imm-samples.parquet

# Track A — point-cloud / 3D-box / class detector snapshots:
cargo run -p thresh --features training-export --bin gen-detector-dataset -- \
  test-data/training/detector/detector-samples.parquet
```

The small checked-in samples under `test-data/training/` are produced by these binaries with default arguments (kept tiny via short scenes + Snappy compression). The full dataset over real acquisition trajectories is produced by the same binaries pointed at acquisition output.

## Track B — IMM mode classifier

Predicts `(p_CV, p_CA, p_CTRV, p_coord_turn)` from a window of classical-tracker filter-state projections (12-dim: 6D common state + 6 covariance-diagonal). The classifier is loaded at runtime behind the `thresh-filter` `learned-imm` feature to override the analytic IMM mode update.

```sh
cd python
uv run --extra training python -m training.train_imm \
  --data ../test-data/training/imm-classifier/imm-samples.parquet \
  --out best_imm.pt --epochs 30
uv run --extra training python -m export.export_imm \
  --checkpoint best_imm.pt --out ../test-data/models/imm_mode_classifier.onnx
```

`train_imm.py` uses a fixed seed and a trajectory-grouped split; `export_imm.py` exports `(batch, 10, 12) → (batch, 4)` and verifies under onnxruntime. **Exit criterion (task 6.8):** held-out accuracy ≥ 0.70 and downstream MOTA no worse than analytic — until met, the stub stays in place and `learned-imm` ships off by default.

## Track A — detector

A 3DETR-style point-cloud set predictor: `(1, 1000, 4)` point cloud → `boxes (1,100,7)` + `scores (1,100,1)` + `classes (1,100,1)`. Decoded Rust-side by `thresh_inference::detection::OnnxDetector`.

```sh
cd python
uv run --extra training python -m training.train_detector \
  --data ../test-data/training/detector/detector-samples.parquet \
  --out best_detector.pt --epochs 50
uv run --extra training python -m export.export_detector \
  --checkpoint best_detector.pt --out ../test-data/models/test_detector.onnx
```

`train_detector.py` uses a DETR set-prediction loss (Hungarian matcher + L1 box + 3D-IoU + class CE + objectness). **Exit criterion (task 7.9):** mAP@0.5 ≥ 0.30 on a held-out region and a downstream MOTA improvement — until met, the random-weight stub stays in place.

## Evaluation (Phase 9)

The end-to-end MOTA/MOTP/IDF1 A/B harness (`python/eval/run_tracker.py`, with `--learned-imm` / `--learned-detector` flags) is the remaining phase. Once it lands, the full reproduction is `uv sync` → acquire → `gen-*-dataset` → `train_*` → `export_*` → `run_tracker.py`.

## License posture

OpenSky-derived data ships with the repository under the OpenSky Network terms with attribution. ADS-B Exchange data is **not** redistributed; an acquisition script is provided and reproduction requires a developer-supplied ADSBx API key. See [`LICENSING.md`](LICENSING.md) for the full attribution and redistribution posture.
