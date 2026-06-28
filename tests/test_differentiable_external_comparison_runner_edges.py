# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external runner comparison edges.
"""Runner, tooling, and payload edge tests for external comparisons."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from _differentiable_external_comparison_edges import executable_runner

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison


def test_enzyme_and_catalyst_runner_timeouts_become_runtime_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner timeouts should be classified as runtime hard gaps."""
    enzyme_runner = executable_runner(tmp_path, "enzyme_runner.py")
    catalyst_runner = executable_runner(tmp_path, "catalyst_runner.py")
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(enzyme_runner))
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(catalyst_runner))
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: True)
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    def timeout_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="runner", timeout=0.01)

    monkeypatch.setattr(cast(Any, comparison).subprocess, "run", timeout_run)

    enzyme_row = comparison._enzyme_row()
    catalyst_row = comparison._catalyst_row()

    assert enzyme_row.status == "hard_gap"
    assert enzyme_row.failure_class == "runtime_error"
    assert "timed out" in str(enzyme_row.setup_instructions)
    assert catalyst_row.status == "hard_gap"
    assert catalyst_row.failure_class == "runtime_error"
    assert "timed out" in str(catalyst_row.setup_instructions)


def test_enzyme_and_catalyst_runner_failures_report_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner non-zero exits should keep stderr evidence in the hard gap."""
    enzyme_runner = executable_runner(tmp_path, "enzyme_runner.py")
    catalyst_runner = executable_runner(tmp_path, "catalyst_runner.py")
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(enzyme_runner))
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(catalyst_runner))
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: True)
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    def failed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(args=["runner"], returncode=2, stdout="", stderr="")

    monkeypatch.setattr(cast(Any, comparison).subprocess, "run", failed_run)

    assert "no stderr" in str(comparison._enzyme_row().setup_instructions)
    assert "no stderr" in str(comparison._catalyst_row().setup_instructions)


def test_catalyst_runner_correctness_mismatch_is_hard_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalyst runner mismatches should not become success rows."""
    runner = tmp_path / "wrong_catalyst_runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'value': 1.0, 'gradient': [0.0, 0.0]}))\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    row = comparison._catalyst_row()

    assert row.status == "hard_gap"
    assert row.failure_class == "correctness_mismatch"
    assert "SCPN reference" in str(row.setup_instructions)


def test_runner_json_and_numeric_validators_reject_invalid_payloads() -> None:
    """Runner payload validators should reject invalid scalar, gradient, and toolchain data."""
    with pytest.raises(ValueError, match="finite real scalar"):
        comparison._as_finite_scalar("runner value", [1.0])
    with pytest.raises(ValueError, match="must be finite"):
        comparison._as_finite_scalar("runner value", float("nan"))
    with pytest.raises(ValueError, match="finite real numeric"):
        comparison._as_gradient_vector("runner gradient", [True, False], 2)
    with pytest.raises(ValueError, match="shape"):
        comparison._as_gradient_vector("runner gradient", [1.0], 2)
    with pytest.raises(ValueError, match="finite real numeric"):
        comparison._as_gradient_vector("runner gradient", [1.0, float("inf")], 2)
    with pytest.raises(ValueError, match="metadata must be an object"):
        comparison._as_toolchain_metadata(["clang"], label="Enzyme")
    with pytest.raises(ValueError, match="empty keys"):
        comparison._as_toolchain_metadata({"clang": ""}, label="Enzyme")
    assert comparison._as_toolchain_metadata(None, label="Enzyme") == {
        "enzyme": "configured-runner"
    }


def test_runner_timeout_environment_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner timeout environment variables should be numeric and positive."""
    monkeypatch.setenv("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(ValueError, match="must be numeric"):
        comparison._enzyme_runner_timeout_seconds()
    monkeypatch.setenv("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        comparison._enzyme_runner_timeout_seconds()

    monkeypatch.setenv("SCPN_CATALYST_RUNNER_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(ValueError, match="must be numeric"):
        comparison._catalyst_runner_timeout_seconds()
    monkeypatch.setenv("SCPN_CATALYST_RUNNER_TIMEOUT_SECONDS", "-1")
    with pytest.raises(ValueError, match="must be positive"):
        comparison._catalyst_runner_timeout_seconds()


def test_runner_admission_rejects_missing_null_and_nonexistent_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner path admission should reject unset, null-byte, and missing paths."""
    monkeypatch.delenv("SCPN_ENZYME_RUNNER", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        comparison._validated_runner_from_env("SCPN_ENZYME_RUNNER", label="Enzyme")

    monkeypatch.setattr(
        cast(Any, comparison).os,
        "environ",
        {"SCPN_ENZYME_RUNNER": "/tmp/bad\x00runner"},
    )
    with pytest.raises(RuntimeError, match="null byte"):
        comparison._validated_runner_from_env("SCPN_ENZYME_RUNNER", label="Enzyme")
    monkeypatch.undo()
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", "/tmp/definitely-missing-scpn-runner")

    with pytest.raises(RuntimeError, match="existing executable"):
        comparison._validated_runner_from_env("SCPN_ENZYME_RUNNER", label="Enzyme")


def test_tooling_availability_and_runner_configuration_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tooling probes should expose executable, plugin, and import availability."""
    runner = executable_runner(tmp_path, "enzyme_runner.py")
    plugin = tmp_path / "enzyme.so"
    plugin.write_bytes(b"plugin")
    monkeypatch.setattr(cast(Any, comparison).shutil, "which", lambda name: None)
    monkeypatch.delenv("ENZYME_LLVM_PLUGIN", raising=False)
    assert comparison._enzyme_tooling_available() is False
    monkeypatch.setenv("ENZYME_LLVM_PLUGIN", str(plugin))
    assert comparison._enzyme_tooling_available() is True
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    assert comparison._enzyme_runner_configured() is True
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", "relative")
    assert comparison._enzyme_runner_configured() is False

    def unavailable_import(name: str) -> object:
        if name == "catalyst":
            raise ImportError(name)
        return ModuleType(name)

    monkeypatch.setattr(comparison, "import_module", unavailable_import)
    monkeypatch.setattr(
        cast(Any, comparison).shutil,
        "which",
        lambda name: "/usr/bin/mlir-opt" if name == "mlir-opt" else None,
    )
    assert comparison._catalyst_tooling_available() is True
    monkeypatch.setattr(cast(Any, comparison).shutil, "which", lambda name: None)
    assert comparison._catalyst_tooling_available() is False
    monkeypatch.setattr(comparison, "import_module", lambda name: ModuleType(name))
    assert comparison._catalyst_tooling_available() is True


def test_installed_version_records_executables_and_import_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version probes should report executables, missing imports, and unknown versions."""
    monkeypatch.setattr(
        cast(Any, comparison).shutil, "which", lambda package: f"/usr/bin/{package}"
    )
    assert comparison._installed_version("llvm") == "executable:/usr/bin/llvm"

    metadata_module: Any = comparison.__dict__["metadata"]
    monkeypatch.setattr(
        metadata_module,
        "version",
        lambda package: (_ for _ in ()).throw(metadata_module.PackageNotFoundError(package)),
    )

    def import_probe(name: str) -> object:
        if name == "missing_pkg":
            raise RuntimeError("not importable")
        return SimpleNamespace()

    monkeypatch.setattr(comparison, "import_module", import_probe)
    assert comparison._installed_version("missing_pkg") == "not_installed"
    assert comparison._installed_version("unknown_version_pkg") == "importable_unknown_version"


def test_runner_rejects_non_object_json_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner JSON payloads should be objects with typed numeric evidence."""
    runner = tmp_path / "array_runner.py"
    runner.write_text("#!/usr/bin/env python3\nprint('[1, 2]')\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))

    with pytest.raises(ValueError, match="JSON must be an object"):
        comparison._run_enzyme_reference(np.array([0.2, -0.4], dtype=np.float64))
    with pytest.raises(ValueError, match="JSON must be an object"):
        comparison._run_catalyst_reference(np.array([0.2, -0.4], dtype=np.float64))


def test_runner_reference_payload_contains_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured runner requests should include the bounded objective contract."""
    runner = tmp_path / "recording_runner.py"
    capture = tmp_path / "request.json"
    runner.write_text(
        "\n".join(
            (
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                f"pathlib.Path({json.dumps(str(capture))}).write_text(sys.stdin.read(), encoding='utf-8')",
                "print(json.dumps({'value': 1.0, 'gradient': [0.0, 0.0]}))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))

    value, gradient, toolchain = comparison._run_enzyme_reference(
        np.array([0.2, -0.4], dtype=np.float64)
    )

    request = json.loads(capture.read_text(encoding="utf-8"))
    assert value == 1.0
    np.testing.assert_allclose(gradient, np.array([0.0, 0.0], dtype=np.float64))
    assert toolchain == {"enzyme": "configured-runner"}
    assert request["schema"] == "scpn_qc_enzyme_runner_request_v1"
    assert request["objective"] == "cos(x0)+0.25*sin(x1)"
    assert request["gradient_contract"] == ["-sin(x0)", "0.25*cos(x1)"]
