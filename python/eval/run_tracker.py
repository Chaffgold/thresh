"""Drive the thresh tracker evaluation and report MOT metrics (tasks 9.2/9.3).

Thin wrapper over the Rust ``eval-tracker`` binary: the eval engine is
Rust-native (design Decision 24 — ``thresh-py`` is CV-only and the Python synth
binding is deferred, so MOTA/MOTP/IDF1 are computed by `thresh::eval_harness`).
This script runs the binary with ``--json``, parses the per-scenario metrics,
and prints a summary.

Learned modes select the required Cargo features and require separate model
paths. They never silently run the analytic baseline. Relative model paths
are resolved from the caller's working directory before invoking Cargo.

    uv run python -m eval.run_tracker            # analytic baseline
    uv run python -m eval.run_tracker --learned-imm --imm-model /path/to/imm.onnx
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

# Repo root (python/eval/run_tracker.py → parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_eval_output(text: str) -> list[dict[str, Any]]:
    """Parse the ``eval-tracker --json`` output: one JSON object per line."""
    results: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("{"):
            results.append(json.loads(line))
    return results


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the thresh tracker eval (MOTA/MOTP/IDF1).")
    parser.add_argument("--learned-imm", action="store_true", help="Use the learned IMM classifier")
    parser.add_argument("--learned-detector", action="store_true", help="Use the learned detector")
    parser.add_argument("--imm-model", "--model", type=Path, help="IMM classifier ONNX path")
    parser.add_argument("--detector-model", type=Path, help="Candidate detector ONNX path")
    parser.add_argument(
        "--duration-seconds", type=float, help="Synthetic scenario duration (default 30)"
    )
    parser.add_argument(
        "--detector-baseline-model", type=Path,
        help="Compare a second detector on the identical point-cloud sequence",
    )
    return parser


def build_command(extra_args: list[str]) -> list[str]:
    """Validate mode selection and assemble explicit Cargo features/model paths."""
    args = argument_parser().parse_args(extra_args)
    features: list[str] = []
    forwarded: list[str] = []
    for mode, model, feature in [
        ("imm", args.imm_model, "learned-imm"),
        ("detector", args.detector_model, "onnx"),
    ]:
        enabled = bool(getattr(args, f"learned_{mode}"))
        if enabled != (model is not None):
            raise ValueError(f"--learned-{mode} and --{mode}-model must be provided together")
        if enabled:
            features.append(feature)
            forwarded.extend([f"--learned-{mode}", f"--{mode}-model", str(model.resolve())])
    if args.detector_baseline_model is not None:
        if not args.learned_detector:
            raise ValueError("--detector-baseline-model requires --learned-detector")
        forwarded.extend(["--detector-baseline-model", str(args.detector_baseline_model.resolve())])
    if args.duration_seconds is not None:
        if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
            raise ValueError("--duration-seconds requires a positive finite number")
        forwarded.extend(["--duration-seconds", str(args.duration_seconds)])
    cmd = ["cargo", "run", "-q", "-p", "thresh", "--bin", "eval-tracker"]
    if features:
        cmd.extend(["--features", ",".join(features)])
    return [*cmd, "--", "--json", *forwarded]


def run_eval(extra_args: list[str], *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Invoke the ``eval-tracker`` cargo binary and return parsed metrics."""
    cmd = build_command(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=repo_root)
    return parse_eval_output(proc.stdout)


def main() -> None:
    import sys

    try:
        results = run_eval(sys.argv[1:])
    except ValueError as error:
        argument_parser().error(str(error))
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or f"eval-tracker exited {error.returncode}"
        raise SystemExit(message) from error
    for r in results:
        print(
            f"{r['scenario']!s:<16} {r['pipeline']} MOTA={r['mota']:.3f}  "
            f"MOTP={r['motp']:.2f}m  IDF1={r['idf1']:.3f}  "
            f"IDSW={r['id_switches']}  frames={r['frames']}"
        )


if __name__ == "__main__":
    main()
