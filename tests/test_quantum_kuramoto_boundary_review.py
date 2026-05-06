# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S6 boundary review tests
"""Tests for the S6 quantum-kuramoto boundary review."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_boundary_review_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_quantum_kuramoto_boundary_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_quantum_kuramoto_boundary_review", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S6 boundary-review script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_boundary_review = _load_boundary_review_module().build_boundary_review


def test_boundary_review_keeps_skeleton_blocked() -> None:
    payload = build_boundary_review()

    assert payload["schema"] == "s6_quantum_kuramoto_boundary_review_v1"
    assert payload["package_skeleton_allowed"] is False
    assert "requires refactors" in payload["reason"]


def test_boundary_review_proposes_core_api_surface() -> None:
    payload = build_boundary_review()
    exports = {row["proposed_export"] for row in payload["proposed_public_api"]}

    assert "quantum_kuramoto.phase.xy_kuramoto" in exports
    assert "quantum_kuramoto.phase.xy_compiler" in exports
    assert "quantum_kuramoto.hardware.async_runner" in exports
    assert "quantum_kuramoto.hardware.runner" not in exports


def test_boundary_review_decides_all_needs_review_rows() -> None:
    payload = build_boundary_review()
    decisions = {row["module"]: row["decision"] for row in payload["needs_review_decisions"]}

    assert decisions["scpn_quantum_control.hardware.runner"] == "defer"
    assert (
        decisions["scpn_quantum_control.hardware.hybrid_digital_analog"] == "promote_after_facade"
    )
    assert all(decision in {"defer", "promote_after_facade"} for decision in decisions.values())
