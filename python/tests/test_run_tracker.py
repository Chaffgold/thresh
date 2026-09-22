"""Model/feature forwarding without Cargo, inference, or external data."""
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.run_tracker import build_command, run_eval


def test_analytic_command_needs_no_features() -> None:
    assert build_command([]) == [
        "cargo", "run", "-q", "-p", "thresh", "--bin", "eval-tracker", "--", "--json"
    ]


@pytest.mark.parametrize(("mode", "feature"), [("imm", "learned-imm"), ("detector", "onnx")])
def test_individual_mode_forwards_feature_and_absolute_path(mode: str, feature: str) -> None:
    command = build_command([f"--learned-{mode}", f"--{mode}-model", "candidate.onnx"])
    assert command[7:] == [
        "--features", feature, "--", "--json", f"--learned-{mode}",
        f"--{mode}-model", str(Path("candidate.onnx").resolve()),
    ]


def test_combined_mode_and_baseline_forwarding() -> None:
    command = build_command([
        "--learned-imm", "--model", "imm.onnx", "--learned-detector",
        "--detector-model", "detector.onnx", "--detector-baseline-model", "stub.onnx",
    ])
    assert command[7:] == [
        "--features", "learned-imm,onnx", "--", "--json",
        "--learned-imm", "--imm-model", str(Path("imm.onnx").resolve()),
        "--learned-detector", "--detector-model", str(Path("detector.onnx").resolve()),
        "--detector-baseline-model", str(Path("stub.onnx").resolve()),
    ]


def test_duration_forwarding() -> None:
    assert build_command(["--duration-seconds", "1.5"])[-2:] == ["--duration-seconds", "1.5"]


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected(duration: str) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        build_command(["--duration-seconds", duration])


@pytest.mark.parametrize("args", [
    ["--learned-imm"], ["--learned-detector"], ["--imm-model", "orphan.onnx"],
    ["--detector-model", "orphan.onnx"], ["--detector-baseline-model", "orphan.onnx"],
])
def test_incomplete_modes_rejected_before_cargo(args: list[str]) -> None:
    with patch("eval.run_tracker.subprocess.run") as process:
        with pytest.raises(ValueError):
            run_eval(args)
        process.assert_not_called()


def test_subprocess_receives_command_and_repo_cwd(tmp_path: Path) -> None:
    with patch("eval.run_tracker.subprocess.run") as process:
        process.return_value.stdout = '{"scenario":"test","pipeline":"detector-candidate"}\n'
        args = ["--learned-detector", "--detector-model", "detector.onnx"]
        result = run_eval(args, repo_root=tmp_path)
        process.assert_called_once_with(
            build_command(args), capture_output=True, text=True, check=True, cwd=tmp_path,
        )
    assert result == [{"scenario": "test", "pipeline": "detector-candidate"}]
