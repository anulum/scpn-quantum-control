# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR toolchain probe hardening tests
# scpn-quantum-control -- MLIR toolchain probe hardening tests
"""Tests for Enzyme/MLIR maturity-audit toolchain probe hardening."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from scpn_quantum_control import run_enzyme_mlir_maturity_audit


def _write_tool(directory: Path, command: str, script: str) -> Path:
    """Create one executable command in a temporary PATH."""
    path = directory / command
    path.write_text(script, encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_version_tool(directory: Path, command: str, version: str) -> Path:
    """Create an executable version-reporting command in a temporary PATH."""
    return _write_tool(
        directory,
        command,
        f"#!/bin/sh\nprintf '%s\\n' '{version}'\n",
    )


@contextmanager
def _temporary_path(directory: Path) -> Iterator[None]:
    """Restrict PATH to ``directory`` and restore the process environment."""
    previous = os.environ.get("PATH")
    os.environ["PATH"] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous


def test_enzyme_mlir_maturity_audit_runs_admitted_path_tools(
    tmp_path: Path,
) -> None:
    """The public maturity audit probes absolute executable PATH tools only."""
    expected_versions = {
        "enzyme": "fake enzyme 18.1.3",
        "opt": "fake opt 18.1.3",
        "mlir-opt": "fake mlir-opt 18.1.3",
        "clang": "fake clang 18.1.3",
    }
    tools = {
        command: _write_version_tool(tmp_path, command, version)
        for command, version in expected_versions.items()
    }
    with _temporary_path(tmp_path):
        result = run_enzyme_mlir_maturity_audit()
    payload = cast("dict[str, Any]", result.to_dict())

    assert all(status.available for status in result.toolchain.values())
    for command, status in result.toolchain.items():
        assert status.executable == str(tools[command].resolve())
        assert Path(status.executable).is_absolute()
        assert status.version == expected_versions[command]
        assert status.failure_class is None
        assert payload["toolchain"][command]["available"] is True
    assert result.ready_for_provider_exceedance is False
    assert not any("toolchain unavailable" in gap for gap in result.hard_gaps)


def test_enzyme_mlir_maturity_audit_rejects_relative_probe_paths() -> None:
    """Relative toolchain paths fail closed before any version subprocess runs."""
    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"relative/{command}",
    )

    assert all(not status.available for status in result.toolchain.values())
    assert all(status.executable is None for status in result.toolchain.values())
    assert {status.failure_class for status in result.toolchain.values()} == {
        "version_probe_failed"
    }
    assert {
        "enzyme toolchain unavailable",
        "opt toolchain unavailable",
        "mlir-opt toolchain unavailable",
        "clang toolchain unavailable",
    }.issubset(set(result.hard_gaps))


def test_enzyme_mlir_maturity_audit_reports_missing_path_tools(tmp_path: Path) -> None:
    """Report absent PATH commands through the public audit hard-gap surface."""
    empty_path = tmp_path / "empty"
    empty_path.mkdir()

    with _temporary_path(empty_path):
        result = run_enzyme_mlir_maturity_audit()

    assert {status.failure_class for status in result.toolchain.values()} == {"toolchain_missing"}
    assert all(status.executable is None for status in result.toolchain.values())


def test_enzyme_mlir_maturity_audit_rejects_nonfile_and_nonexecutable_probes(
    tmp_path: Path,
) -> None:
    """Reject public probe results that cannot be executed as local files."""
    nonexecutable = tmp_path / "nonexecutable"
    nonexecutable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nonexecutable.chmod(0o600)

    for target in (tmp_path, nonexecutable):

        def toolchain_probe(_command: str, bound_target: Path = target) -> str:
            return str(bound_target)

        result = run_enzyme_mlir_maturity_audit(toolchain_probe=toolchain_probe)
        assert {status.failure_class for status in result.toolchain.values()} == {
            "version_probe_failed"
        }


def test_enzyme_mlir_maturity_audit_handles_real_probe_process_failures(
    tmp_path: Path,
) -> None:
    """Fail closed when admitted commands cannot start or emit version text."""
    scripts = (
        "#!/does/not/exist\n",
        "#!/bin/sh\nexit 0\n",
    )
    for index, script in enumerate(scripts):
        tool_path = tmp_path / f"case-{index}"
        tool_path.mkdir()
        for command in ("enzyme", "opt", "mlir-opt", "clang"):
            _write_tool(tool_path, command, script)

        with _temporary_path(tool_path):
            result = run_enzyme_mlir_maturity_audit()

        assert {status.failure_class for status in result.toolchain.values()} == {
            "version_probe_failed"
        }
        assert all(status.version is None for status in result.toolchain.values())
