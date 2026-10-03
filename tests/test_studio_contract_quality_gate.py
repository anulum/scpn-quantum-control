# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared contract ownership and conformance
"""Qualify executable conformance orchestration and explicit failure propagation."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

from tools import studio_contract_quality_gates as quality


def test_builder_binds_every_real_consumer() -> None:
    """Return five executable consumers with actual native and WASM test owners."""
    gates = quality.build_contract_conformance_gates("locked-python")
    assert [name for name, _, _ in gates] == [
        "workspace/python",
        "workspace/typescript",
        "program-source/python",
        "program-source/rust",
        "program-source/wasm",
    ]
    assert (
        quality.build_contract_conformance_gates(
            "locked-python", repo_root=Path(quality.__file__).resolve().parents[1]
        )
        == gates
    )
    assert gates[0][2][0] == "locked-python"
    assert gates[3][2] == [
        "cargo",
        "test",
        "--locked",
        "--test",
        "program_source",
        "shared_source_corpus",
    ]
    assert "programCompiler.test.ts" in " ".join(gates[4][2])


def test_default_cli_checks_actual_ownership(capsys: pytest.CaptureFixture[str]) -> None:
    """Readonly CLI checks actual registry and required aggregate with no child launch."""
    assert quality.main([]) == 0
    assert "5 required real consumers" in capsys.readouterr().out


@pytest.mark.parametrize("error", [ValueError("drift"), KeyError("owner"), OSError("missing")])
def test_invalid_ownership_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    """Refuse bad registry/workflow without a success-shaped result."""

    def reject(python: str) -> list[quality.Gate]:
        raise error

    monkeypatch.setattr(quality, "build_contract_conformance_gates", reject)
    assert quality.main([]) == 1
    assert "ownership refused" in capsys.readouterr().err


@pytest.mark.parametrize("status", [3, -9])
def test_consumer_failure_stops_the_required_cohort(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """A failed actual command cannot fall through to the next consumer."""
    calls: list[list[str]] = []

    def fail(
        command: list[str], *, cwd: Path, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert timeout == 120 and not check and cwd.is_dir()
        return subprocess.CompletedProcess(command, status)

    monkeypatch.setattr("tools.studio_contract_quality_gates.subprocess.run", fail)
    assert quality.main(["--run"]) == (status if status > 0 else 1)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["missing", "timeout"])
def test_unavailable_runtime_and_timeout_fail_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """Runtime launch failures remain observable and are never retried."""
    calls = 0

    def refuse(
        command: list[str], *, cwd: Path, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        raise OSError("runtime missing")

    monkeypatch.setattr("tools.studio_contract_quality_gates.subprocess.run", refuse)
    assert quality.main(["--run"]) == (124 if kind == "timeout" else 127)
    assert calls == 1
    assert ("timed out" if kind == "timeout" else "unavailable") in capsys.readouterr().err


def test_module_entrypoint_checks_real_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute the public module entrypoint against the actual checkout."""
    monkeypatch.setattr("sys.argv", ["studio_contract_quality_gates"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(Path(quality.__file__)), run_name="__main__")
    assert caught.value.code == 0
