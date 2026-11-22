"""Failure-mode self-tests for the boundary gate; no third-party dependencies."""

from importlib import metadata
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_boundary_tests as runner


def output(status="ok", passed=1, failed=0, ignored=0, filtered=0):
    result = "FAILED" if failed else "ok"
    return (
        f"running 1 test\ntest required ... {status}\n"
        f"test result: {result}. {passed} passed; {failed} failed; {ignored} ignored; "
        f"0 measured; {filtered} filtered out; finished in 0.00s\n"
    )


class BoundaryRunnerTests(unittest.TestCase):
    def test_success_requires_named_assertions_and_exact_counts(self):
        runner.verify_discovery("required: test\n1 test, 0 benchmarks\n", ["required"], "fixture")
        self.assertEqual(runner.verify_execution(output(), ["required"], "fixture"), 1)

    def test_empty_selection_fails(self):
        with self.assertRaises(runner.GateFailure):
            runner.verify_discovery("", [], "fixture")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"schema_version": 1, "targets": {}}', encoding="utf-8")
            with self.assertRaises(runner.GateFailure):
                runner.load_manifest(manifest)

    def test_missing_or_renamed_required_test_fails(self):
        with self.assertRaises(runner.GateFailure):
            runner.verify_discovery("renamed: test\n", ["required"], "fixture")

    def test_unlisted_test_fails(self):
        with self.assertRaises(runner.GateFailure):
            runner.verify_discovery("required: test\nnew: test\n", ["required"], "fixture")

    def test_ignored_required_test_fails_even_with_successful_exit(self):
        ignored_output = output("ignored", passed=0, ignored=1)
        with self.assertRaises(runner.GateFailure):
            runner.verify_execution(ignored_output, ["required"], "fixture")

    def test_failed_required_assertion_fails(self):
        failed_output = output("FAILED", passed=0, failed=1)
        with self.assertRaises(runner.GateFailure):
            runner.verify_execution(failed_output, ["required"], "fixture")

    def test_failed_child_exit_fails(self):
        with self.assertRaises(runner.GateFailure):
            runner.run_command([sys.executable, "-c", "raise SystemExit(7)"], os.environ.copy())

    def test_missing_dependency_fails_before_cargo(self):
        with patch.object(metadata, "version", side_effect=metadata.PackageNotFoundError("numpy")):
            with self.assertRaisesRegex(runner.GateFailure, "missing required Python dependency: numpy"):
                runner.check_package_versions({"numpy": "2.5.3"})

    def test_incorrect_dependency_version_fails(self):
        with patch.object(metadata, "version", return_value="0.0.0"):
            with self.assertRaises(runner.GateFailure):
                runner.check_package_versions({"numpy": "2.5.3"})

    def test_linux_direct_binaries_receive_shared_library_path(self):
        environment = {"LD_LIBRARY_PATH": "/existing"}
        values = {"LIBDIR": str(Path(__file__).parent), "LDLIBRARY": Path(__file__).name}
        with patch.object(sys, "platform", "linux"):
            with patch.object(runner.sysconfig, "get_config_var", side_effect=values.get):
                runner.configure_shared_library(environment)
        self.assertEqual(environment["LD_LIBRARY_PATH"], str(Path(__file__).parent) + os.pathsep + "/existing")

    def test_missing_linux_shared_library_fails(self):
        with patch.object(sys, "platform", "linux"):
            with patch.object(runner.sysconfig, "get_config_var", return_value=None):
                with self.assertRaises(runner.GateFailure):
                    runner.configure_shared_library({})

    def test_zero_executed_tests_cannot_pass(self):
        with self.assertRaises(runner.GateFailure):
            runner.verify_execution(
                "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;\n",
                ["required"], "fixture",
            )

    def test_filtered_counts_and_duplicate_result_lines_fail(self):
        filtered_output = output(filtered=1)
        with self.assertRaises(runner.GateFailure):
            runner.verify_execution(filtered_output, ["required"], "fixture")
        duplicate_output = "test required ... ok\n" + output()
        with self.assertRaises(runner.GateFailure):
            runner.verify_execution(duplicate_output, ["required"], "fixture")

    def test_missing_target_cannot_disappear_from_manifest(self):
        metadata_output = '{"packages":[{"name":"thresh-bridge","targets":[{"name":"new_contract","kind":["test"]}]}]}'
        with patch.object(runner, "run_command", return_value=metadata_output):
            with self.assertRaises(runner.GateFailure):
                runner.verify_target_inventory({"lib": ["required"]}, {})


if __name__ == "__main__":
    unittest.main()
