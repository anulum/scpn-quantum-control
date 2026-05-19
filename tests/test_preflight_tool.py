# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for local preflight tool
"""Tests for the local CI preflight helper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_preflight = _load_tool_module("preflight_for_tests", "preflight.py")


def test_static_gates_include_documentation_surface_gate():
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert "documentation-surface" in gate_map
    assert "tools/audit_documentation_surface.py" in gate_map["documentation-surface"]
    assert "--allowlist" in gate_map["documentation-surface"]
    assert "tools/documentation_surface_allowlist.json" in gate_map["documentation-surface"]
    assert "--fail-on-findings" in gate_map["documentation-surface"]


def test_run_gate_reports_pass(monkeypatch, capsys):
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(_preflight.subprocess, "run", fake_run)

    assert _preflight.run_gate("unit", ["tool"]) is True
    assert "PASS  unit" in capsys.readouterr().out


def test_run_gate_reports_failure_tail(monkeypatch, capsys):
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="\n".join(f"out-{idx}" for idx in range(12)),
            stderr="\n".join(f"err-{idx}" for idx in range(12)),
        )

    monkeypatch.setattr(_preflight.subprocess, "run", fake_run)

    assert _preflight.run_gate("unit", ["tool"]) is False
    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "out-2" in output
    assert "out-11" in output
    assert "err-2" in output
    assert "err-11" in output
    assert "        out-1\n" not in output


def test_main_skips_tests_with_no_tests_flag(monkeypatch, capsys):
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-tests"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 0
    assert calls == ["lint", "bandit"]
    assert "ALL CLEAR" in capsys.readouterr().out


def test_main_uses_plain_pytest_when_coverage_is_disabled(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-coverage"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 0
    assert calls == ["lint", "pytest", "bandit"]


def test_main_stops_on_first_failed_gate(monkeypatch, capsys):
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"]), ("type", ["type"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-tests"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return name != "lint"

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 1
    assert calls == ["lint"]
    assert "BLOCKED: lint" in capsys.readouterr().out
