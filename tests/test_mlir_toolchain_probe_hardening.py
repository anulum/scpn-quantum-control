# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR toolchain probe hardening tests
"""Tests for Enzyme/MLIR maturity-audit toolchain probe hardening."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control import run_enzyme_mlir_maturity_audit


def _write_version_tool(directory: Path, command: str, version: str) -> Path:
    """Create an executable version-reporting command in a temporary PATH."""
    path = directory / command
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def test_enzyme_mlir_maturity_audit_runs_admitted_path_tools(
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv("PATH", str(tmp_path))

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
