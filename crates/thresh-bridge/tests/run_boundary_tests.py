#!/usr/bin/env python3
"""Fail-closed, pinned boundary and bounded algorithm-reference gate (stdlib only)."""

from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys
import sysconfig
import tomllib


TEST_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = TEST_DIRECTORY.parents[2]
PYTHON_VERSION = "3.12.13"
PYO3_VERSION = "0.29.0"
TEST_LINE = re.compile(r"^test (.+) \.\.\. (ok|FAILED|ignored)(?:,.*)?$", re.MULTILINE)
SUMMARY = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out;.*$", re.MULTILINE
)


class GateFailure(RuntimeError):
    """Required dependencies or executed coverage did not satisfy the contract."""


def load_manifest(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GateFailure("manifest must be a JSON object")
    targets = data.get("targets")
    if data.get("schema_version") != 1 or not isinstance(targets, dict) or not targets:
        raise GateFailure("manifest must have schema_version 1 and nonempty targets")
    for target, names in targets.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", target):
            raise GateFailure(f"invalid manifest target: {target!r}")
        validate_names(names, f"manifest target {target}")
    if "lib" not in targets:
        raise GateFailure("manifest must include the lib unit tests")
    return targets


def validate_names(names: object, context: str) -> None:
    if not isinstance(names, list) or not names:
        raise GateFailure(f"{context}: empty test selection")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise GateFailure(f"{context}: invalid test name")
    if len(names) != len(set(names)):
        raise GateFailure(f"{context}: duplicate test names")


def locked_python_packages() -> dict[str, str]:
    text = (TEST_DIRECTORY / "requirements.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", text, re.MULTILINE))
    baseline = {"stonesoup": "1.9.1", "numpy": "2.5.3", "scipy": "1.18.1"}
    if not pins or any(pins.get(name) != version for name, version in baseline.items()):
        raise GateFailure("Python lock does not contain the reviewed exact baseline")
    return pins


def check_package_versions(pins: dict[str, str]) -> None:
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise GateFailure(f"missing required Python dependency: {name}") from error
        if actual != expected:
            raise GateFailure(f"{name}: installed {actual}, required {expected}")
        print(f"{name}=={actual}", flush=True)


def runtime_environment() -> dict[str, str]:
    if platform.python_implementation() != "CPython" or platform.python_version() != PYTHON_VERSION:
        raise GateFailure(f"required CPython {PYTHON_VERSION}; running {platform.python_version()}")
    if sys.prefix == sys.base_prefix:
        raise GateFailure("run the boundary gate from an isolated virtual environment")
    print(f"CPython {platform.python_version()} ({sys.executable})", flush=True)
    check_package_versions(locked_python_packages())
    for name in ("numpy", "scipy", "stonesoup"):
        try:
            import_module(name)
        except Exception as error:
            raise GateFailure(f"cannot import required Python dependency {name}: {error}") from error
    lock = tomllib.loads((REPOSITORY / "Cargo.lock").read_text(encoding="utf-8"))
    versions = [package["version"] for package in lock["package"] if package["name"] == "pyo3"]
    if versions != [PYO3_VERSION]:
        raise GateFailure(f"expected locked PyO3 {PYO3_VERSION}, found {versions}")
    print(f"PyO3 {versions[0]} (Cargo.lock)", flush=True)
    environment = os.environ.copy()
    supplied = environment.get("PYO3_PYTHON")
    if supplied and os.path.abspath(supplied) != os.path.abspath(sys.executable):
        raise GateFailure("run this script with the same interpreter selected by PYO3_PYTHON")
    # Embedding does not automatically load a virtualenv's site-packages. These
    # are child-process settings only; the Rust library never mutates either.
    environment["PYO3_PYTHON"] = sys.executable
    environment["PYTHONPATH"] = os.pathsep.join(site.getsitepackages())
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CARGO_TERM_COLOR"] = "never"
    configure_shared_library(environment)
    return environment


def configure_shared_library(environment: dict[str, str]) -> None:
    # We execute test binaries directly, outside Cargo's dynamic-library-path
    # setup. Linux must also be able to load this interpreter's shared library.
    if sys.platform != "linux":
        return
    library_directory = sysconfig.get_config_var("LIBDIR")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if not library_directory or not library_name:
        raise GateFailure("Python shared-library location is unavailable")
    if not (Path(library_directory) / library_name).is_file():
        raise GateFailure("Python shared library is missing from its configured location")
    prior = environment.get("LD_LIBRARY_PATH")
    directories = [library_directory] + ([prior] if prior else [])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(directories)


def run_command(
    command: list[str], environment: dict[str, str], timeout: int = 1800, display_output: bool = True,
) -> str:
    print("+ " + " ".join(command), flush=True)
    try:
        result = subprocess.run(
            command, cwd=REPOSITORY, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateFailure(f"failed child command {command[0]}: {error}") from error
    if display_output or result.returncode != 0:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise GateFailure(f"child command exited {result.returncode}: {' '.join(command)}")
    return result.stdout


def json_records(output: str) -> list[dict]:
    records = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def verify_target_inventory(targets: dict[str, list[str]], environment: dict[str, str]) -> None:
    output = run_command(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], environment,
        display_output=False,
    )
    records = [record for record in json_records(output) if "packages" in record]
    if len(records) != 1:
        raise GateFailure("cargo metadata did not report exactly one workspace")
    packages = [package for package in records[0]["packages"] if package["name"] == "thresh-bridge"]
    if len(packages) != 1:
        raise GateFailure("cargo metadata did not identify thresh-bridge")
    actual = {target["name"] for target in packages[0]["targets"] if "test" in target["kind"]}
    if actual | {"lib"} != set(targets):
        raise GateFailure(f"manifest target mismatch: discovered {sorted(actual | {'lib'})}, declared {sorted(targets)}")


def build_binaries(targets: dict[str, list[str]], environment: dict[str, str]) -> dict[str, str]:
    command = [
        "cargo", "test", "--locked", "-p", "thresh-bridge", "--no-default-features",
        "--features", "stonesoup", "--no-run", "--message-format=json", "--lib",
    ]
    for target in targets:
        if target != "lib":
            command.extend(["--test", target])
    output = run_command(command, environment, display_output=False)
    binaries = {}
    for record in json_records(output):
        if record.get("reason") != "compiler-artifact" or not record.get("executable"):
            continue
        if not record.get("profile", {}).get("test"):
            continue
        target = record["target"]
        name = "lib" if "lib" in target["kind"] else target["name"]
        if name in targets:
            if name in binaries:
                raise GateFailure(f"multiple executable artifacts for {name}")
            binaries[name] = record["executable"]
    if set(binaries) != set(targets):
        raise GateFailure(f"missing built test targets: {sorted(set(targets) - set(binaries))}")
    return binaries


def verify_discovery(output: str, expected: list[str], target: str) -> None:
    validate_names(expected, target)
    discovered = re.findall(r"^(.+): test$", output, re.MULTILINE)
    validate_names(discovered, f"discovered {target}")
    if set(discovered) != set(expected):
        missing = sorted(set(expected) - set(discovered))
        extra = sorted(set(discovered) - set(expected))
        raise GateFailure(f"{target}: manifest mismatch; missing={missing}, unlisted={extra}")


def verify_execution(output: str, expected: list[str], target: str) -> int:
    validate_names(expected, target)
    entries = TEST_LINE.findall(output)
    names = [name for name, _ in entries]
    validate_names(names, f"executed {target}")
    if set(names) != set(expected) or any(status != "ok" for _, status in entries):
        raise GateFailure(f"{target}: missing, unexpected, failed, or ignored required tests")
    summaries = SUMMARY.findall(output)
    if summaries != [("ok", str(len(expected)), "0", "0", "0", "0")]:
        raise GateFailure(f"{target}: expected exactly {len(expected)} passed and zero failed/ignored/filtered tests")
    return len(expected)


def execute_manifest(targets: dict[str, list[str]], environment: dict[str, str]) -> int:
    verify_target_inventory(targets, environment)
    binaries = build_binaries(targets, environment)
    # Discover every target before running any, so stale manifests fail early.
    for target, expected in targets.items():
        output = run_command([binaries[target], "--list", "--format", "terse"], environment)
        verify_discovery(output, expected, target)
    count = 0
    for target, expected in targets.items():
        output = run_command(
            [binaries[target], "--format", "pretty", "--color", "never", "--test-threads=1"], environment
        )
        count += verify_execution(output, expected, target)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TEST_DIRECTORY / "boundary-test-manifest.json")
    arguments = parser.parse_args()
    try:
        targets = load_manifest(arguments.manifest)
        count = execute_manifest(targets, runtime_environment())
    except (GateFailure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Boundary gate FAILED: {error}", file=sys.stderr)
        return 1
    print(f"Boundary gate PASSED: {count} required tests across {len(targets)} targets; zero ignored.")
    print("Boundary and bounded linear-Gaussian JPDA/GM-PHD reference checks only; the bridge "
          "provides no MHT/IMM reference (native mht-tracking and imm-estimation contracts, "
          "with interim gaps). Whole-track accuracy and linked-library deployment remain "
          "separate gaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
