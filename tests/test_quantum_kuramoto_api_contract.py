# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum kuramoto API contract tests
# scpn-quantum-control -- S6 API contract tests
"""Tests for the S6 quantum-kuramoto API contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_api_contract_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "export_quantum_kuramoto_api_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_quantum_kuramoto_api_contract", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S6 API-contract script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_api_contract = _load_api_contract_module().build_api_contract


def test_api_contract_keeps_skeleton_blocked() -> None:
    payload = build_api_contract()

    assert payload["schema"] == "s6_quantum_kuramoto_api_contract_v1"
    assert payload["package_skeleton_allowed"] is False
    assert "before any separate package skeleton" in payload["reason"]


def test_api_contract_validates_proposed_export_names() -> None:
    payload = build_api_contract()

    assert payload["contract_passed"] is True
    assert payload["errors"] == []
    assert all(row["target_valid"] for row in payload["rows"])
    assert all(row["proposed_export"].startswith("quantum_kuramoto.") for row in payload["rows"])


def test_api_contract_blocks_unreviewed_runner_surface() -> None:
    payload = build_api_contract()
    modules = {row["module"] for row in payload["rows"]}
    exports = {row["proposed_export"] for row in payload["rows"]}

    assert "scpn_quantum_control.hardware.runner" not in modules
    assert "quantum_kuramoto.hardware.runner" not in exports


def test_api_contract_flags_non_reusable_rows_as_warnings() -> None:
    payload = build_api_contract()
    warnings = payload["warnings"]

    assert any("hardware.async_runner" in warning for warning in warnings)
    assert any("hardware.analog_kuramoto" in warning for warning in warnings)
    assert payload["summary"]["warning_count"] == 2
    assert payload["summary"]["immediately_promotable_exports"] == 14
