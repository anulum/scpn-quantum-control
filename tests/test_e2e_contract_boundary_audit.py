# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — E2E boundary audit tests
"""Tests for the E2E contract boundary audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_AUDIT_TOOL = ROOT / "tools" / "audit_e2e_contract_boundaries.py"
_SPEC = importlib.util.spec_from_file_location("audit_e2e_contract_boundaries", _AUDIT_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

audit_boundaries = _MODULE.audit_boundaries
audits_to_json = _MODULE.audits_to_json
format_audits = _MODULE.format_audits
main = _MODULE.main


def test_boundary_audit_matches_expected_test_file_names(tmp_path: Path) -> None:
    (tmp_path / "test_hardware_runner.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "test_bridge_properties.py").write_text("def test_b():\n    assert True\n")
    (tmp_path / "test_snn_neurocore_e2e.py").write_text("def test_c():\n    assert True\n")
    (tmp_path / "test_orchestrator_adapter.py").write_text("def test_d():\n    assert True\n")

    audits = {item.key: item for item in audit_boundaries(tmp_path)}

    assert audits["hardware_qpu"].covered
    assert audits["bridge"].matching_files == ("test_bridge_properties.py",)
    assert audits["sc_neurocore"].covered
    assert audits["phase_orchestrator"].covered
    assert not audits["notebook"].covered
    assert not audits["example"].covered


def test_boundary_audit_json_is_machine_readable(tmp_path: Path) -> None:
    (tmp_path / "test_example_workflow.py").write_text("def test_example():\n    assert True\n")

    decoded = json.loads(audits_to_json(audit_boundaries(tmp_path)))

    assert {item["key"] for item in decoded} >= {"hardware_qpu", "example"}
    example = next(item for item in decoded if item["key"] == "example")
    assert example["covered"] is True


def test_boundary_audit_summary_lists_missing_boundaries(tmp_path: Path) -> None:
    (tmp_path / "test_notebook_smoke.py").write_text("def test_notebook():\n    assert True\n")

    summary = format_audits(audit_boundaries(tmp_path))

    assert "E2E contract boundary audit summary:" in summary
    assert "notebook: covered" in summary
    assert "hardware_qpu: missing" in summary


def test_boundary_audit_cli_fail_on_missing(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "test_bridge_contract.py").write_text("def test_bridge():\n    assert True\n")

    assert main(["--tests-root", str(tmp_path), "--json"]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert any(item["key"] == "bridge" and item["covered"] for item in decoded)
    assert main(["--tests-root", str(tmp_path), "--fail-on-missing"]) == 1
