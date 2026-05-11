# thresh — Training Pipeline

Reproduction recipe for the trained ONNX checkpoints under `test-data/models/` that the inference pipeline consumes. This pipeline is tracked by the OpenSpec change `flight-data-training-pipeline`; see [`openspec/changes/flight-data-training-pipeline/`](openspec/changes/flight-data-training-pipeline/) for the full proposal, design, and tasks.

> **Status:** Phase 1 (toolchain) only. Acquisition, training, and export are added in later phases — see the change's `tasks.md`. Until those land, this document covers only the Python environment bootstrap.

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

## License posture (preview)

OpenSky-derived data ships with the repository under the OpenSky Network terms with attribution. ADS-B Exchange data is **not** redistributed; an acquisition script is provided and reproduction requires a developer-supplied ADSBx API key. Full attribution strings will land in `LICENSING.md` in Phase 10.
