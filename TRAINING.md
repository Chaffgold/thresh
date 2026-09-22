# thresh — Training Pipeline

Reproduction recipe for training and evaluating ONNX models; the checked-in `test-data/models/` artifacts are random-weight test stubs, not trained checkpoints. This pipeline is tracked by the OpenSpec change `flight-data-training-pipeline`; see [`openspec/changes/flight-data-training-pipeline/`](openspec/changes/flight-data-training-pipeline/) for the full proposal, design, and tasks.

> **Status (2026-09-22):** Acquisition clients, synthetic generation, and both training/export pipelines are implemented. The July experiments did not satisfy trained-model acceptance; no training run is currently implied. Further external-data work awaits confirmed access/use rights, and representative model acceptance plus documented distribution rights remain pending for both tracks. Synthetic-only correctness work includes the 13-feature elapsed-time IMM contract (Decision 30). Evaluation details and historical results are in [`docs/eval/flight-data-training-pipeline.md`](docs/eval/flight-data-training-pipeline.md). The checked-in models remain **random-weight stubs**; see `test-data/models/MODEL_CARD.md`.

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
uv sync --extra training         # full training/export env including torch
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

Real-data acquisition and training remain pending confirmation of the applicable
OpenSky access and use rights, including any required written permission. The
clients remain available for authorized local work; the checked-in acquisition
fixture is wholly synthetic and does not complete real-data validation.

### OpenSky Network (historical; verify use and distribution rights)

The OpenSky REST endpoint is usable anonymously but heavily quota-limited — nominally 400 credits/day, in practice ~55 bounding-box polls/day/IP. Anonymous access also **cannot** replay a historical window: the `time` parameter is rejected, so you only ever see the current snapshot.

OpenSky has **removed HTTP Basic authentication**. The API now accepts only OAuth2 bearer tokens minted through the `client_credentials` grant. Register at <https://opensky-network.org/>, create an API client under <https://opensky-network.org/my-opensky/api-client>, and supply the pair via environment variables (matching the `ADSBX_API_KEY` convention below):

```sh
export OPENSKY_CLIENT_ID="your-client-id"
export OPENSKY_CLIENT_SECRET="your-client-secret"
```

Alternatively, drop the `credentials.json` the web UI gives you at `~/.config/opensky/credentials.json` (mode `0600`); it is read unmodified. Environment variables win when both are present.

Authenticating raises the budget to 4,000 credits/day, or 8,000 for an active feeder (≥30% uptime) — the difference between a ~23-minute capture and a usable training set. Secrets are deliberately **not** accepted as CLI flags, since `argv` is readable by other users through `ps`.

```python
from acquisition.opensky import fetch_state_vectors, BoundingBox, TimeRange
from acquisition.opensky_auth import build_auth, load_credentials

bbox = BoundingBox(lat_min=47.0, lat_max=48.0, lon_min=-123.0, lon_max=-122.0)  # KSEA-ish
time_range = TimeRange(start_s=..., end_s=...)
auth = build_auth(load_credentials())  # None when no credentials are configured
records = list(fetch_state_vectors(bbox, time_range, auth=auth))
```

Tokens live 30 minutes; the auth flow caches one, refreshes a minute before expiry, and transparently re-mints and replays on a `401`, so long captures need no special handling.

For bulk historical use the published Zenodo trajectory dumps; check the exact dataset's license and any supplemental terms before use or redistribution. SHA-256 verification is built in:

```python
from acquisition.opensky import load_zenodo_dump
records = list(load_zenodo_dump("opensky-traffic-2024.parquet", sha256="abc..."))
```

For a live capture into a developer-local canonical-schema Parquet file, first
verify that the intended use is permitted under the terms in
[`LICENSING.md`](LICENSING.md), then pass an explicit output path:

```sh
uv run python -m acquisition.opensky_cli --bbox 49.0 51.0 7.0 10.0 --duration-s 300 \
  --out ../data/opensky-capture.parquet
```

Anonymous access always returns the *current* snapshot, so the CLI paces
itself against the wall clock (one poll per `--poll-interval-s`, default 10 s)
and dedups on `(icao24, timestamp_us)`.

The CLI defaults to `data/opensky-capture.parquet` at the repository root.
Do not commit or publish captures without documented distribution rights.
For offline schema checks, regenerate the synthetic CI fixture from `python/`
with `uv run python -m acquisition.synthetic_fixture`; its provenance is in
[`test-data/trajectories/README.md`](test-data/trajectories/README.md).

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

Both learned components train on **synthetic perception driven by truth trajectories**. The default commands below use wholly synthetic built-in trajectories, without provider access or data. Authorized real ADS-B trajectories can supply system-level truth, **never** direct measurements (design Decision 3). The Rust side generates the training Parquet (behind the `training-export` Cargo feature on the umbrella `thresh` crate):

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

Predicts `(p_CV, p_CA, p_CTRV, p_coord_turn)` from 13-dimensional projections: 6D common state, 6 covariance-diagonal entries, and elapsed seconds since the preceding recorded measurement update. History includes tentative post-birth updates but excludes birth initialization and prediction-only ticks. The first recorded update has elapsed time zero; missed measurements accumulate into the next update's interval. Sliding windows preserve each row's interval. Legacy 12-wide datasets/checkpoints must be regenerated/retrained, not assigned an assumed rate. The `learned-imm` feature overrides posterior mode probabilities, leaving analytic interaction/mixing unchanged.

```sh
cd python
uv run --extra training python -m training.train_imm \
  --data ../test-data/training/imm-classifier/imm-samples.parquet \
  --out best_imm.pt --epochs 30
uv run --extra training python -m export.export_imm \
  --checkpoint best_imm.pt --out ../data/imm_mode_classifier.onnx
```

`train_imm.py` uses a fixed seed and a trajectory-grouped split; `export_imm.py` exports `(batch, 10, 13) → (batch, 4)` and verifies under onnxruntime. Local candidates stay outside the tracked fixture path. **Exit criterion (task 6.8):** held-out accuracy ≥ 0.70, passing IMM tests, and downstream MOTA no worse than analytic, plus documented training/distribution rights. Until all gates pass, the stub stays and `learned-imm` ships off by default; an experimental label cannot waive a regression.

## Track A — detector

A 3DETR-style point-cloud set predictor: `(1, 1000, 4)` point cloud → `boxes (1,100,7)` + `scores (1,100,1)` + `classes (1,100,1)`. Decoded Rust-side by `thresh_inference::detection::OnnxDetector`.

```sh
cd python
uv run --extra training python -m training.train_detector \
  --data ../test-data/training/detector/detector-samples.parquet \
  --out best_detector.pt --epochs 50
uv run --extra training python -m export.export_detector \
  --checkpoint best_detector.pt --out ../data/test_detector.onnx
```

`train_detector.py` uses a DETR set-prediction loss (Hungarian matcher + L1 box + 3D-IoU + class CE + objectness). **Exit criterion (task 7.9, Decision 26):** micro distance-gated AP at {2.5, 5, 10} m strictly exceeds the classical baseline on the same held-out point clouds, matched-box class accuracy ≥ 0.50, and downstream MOTA beats the random stub. Documented training/distribution rights are also required; until then the random-weight stub stays.

## Evaluation (Phase 9)

The MOTA/MOTP/IDF1 evaluation wrapper (`python/eval/run_tracker.py`) drives the Rust-native `eval-tracker` binary (`thresh::eval_harness`); details and historical results live in [`docs/eval/flight-data-training-pipeline.md`](docs/eval/flight-data-training-pipeline.md). Offline reproduction uses synthetic `gen-*-dataset` output → `train_*` → `export_*` → `run_tracker.py`. External-data acquisition requires confirmed access/use rights. Representative trained-model acceptance and distribution-rights gates remain open for both tracks; passing synthetic contract tests does not close them.

## License posture

OpenSky API access and attribution do not establish permission to redistribute data. Verify and document the applicable dataset license or written authorization before sharing captures or derived datasets. The earlier API fixture is replaced with a wholly synthetic sample; no redistribution permission is known for the historical capture. ADS-B Exchange data is **not** redistributed; an acquisition script is provided and reproduction requires a developer-supplied ADSBx API key. See [`LICENSING.md`](LICENSING.md) for the full attribution and redistribution posture.

A trained checkpoint's release depends on its **data lineage** and documented rights. A checkpoint trained only on synthetic truth carries the code license. Before committing or publishing one trained with OpenSky-derived truth, document the terms or authorization permitting its intended training and distribution in [`test-data/models/MODEL_CARD.md`](test-data/models/MODEL_CARD.md). Meeting the evaluation thresholds does not satisfy this separate release requirement. See "Trained models" in `LICENSING.md`.
