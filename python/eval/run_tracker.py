"""Drive the thresh tracker evaluation and report MOT metrics (tasks 9.2/9.3).

Thin wrapper over the Rust ``eval-tracker`` binary: the eval engine is
Rust-native (design Decision 24 — ``thresh-py`` is CV-only and the Python synth
binding is deferred, so MOTA/MOTP/IDF1 are computed by `thresh::eval_harness`).
This script runs the binary with ``--json``, parses the per-scenario metrics,
and prints a summary.

The ``--learned-imm`` / ``--learned-detector`` flags are passed through. Until
``MultiObjectTracker`` gains a learned-IMM path and trained checkpoints exist,
the binary falls back to the analytic baseline (see the Phase 9 eval report).

    uv run python -m eval.run_tracker            # analytic baseline
    uv run python -m eval.run_tracker --learned-imm
"""

from __future__ import annotations

import argparse
import json
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


def run_eval(extra_args: list[str], *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Invoke the ``eval-tracker`` cargo binary and return parsed metrics."""
    cmd = [
        "cargo", "run", "-q", "-p", "thresh", "--bin", "eval-tracker", "--", "--json", *extra_args
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=repo_root)
    return parse_eval_output(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thresh tracker eval (MOTA/MOTP/IDF1).")
    parser.add_argument("--learned-imm", action="store_true", help="A/B the learned IMM classifier")
    parser.add_argument(
        "--learned-detector", action="store_true", help="A/B the learned detector"
    )
    args = parser.parse_args()

    extra: list[str] = []
    if args.learned_imm:
        extra.append("--learned-imm")
    if args.learned_detector:
        extra.append("--learned-detector")

    for r in run_eval(extra):
        print(
            f"{r['scenario']!s:<16} MOTA={r['mota']:.3f}  "
            f"MOTP={r['motp']:.2f}m  IDF1={r['idf1']:.3f}  "
            f"IDSW={r['id_switches']}  frames={r['frames']}"
        )


if __name__ == "__main__":
    main()
