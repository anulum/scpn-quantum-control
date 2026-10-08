# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum kuramoto boundary review tests
# scpn-quantum-control -- S6 boundary review tests
"""Tests for the S6 quantum-kuramoto boundary review."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from scripts.audit_quantum_kuramoto_split import build_split_audit


def _load_boundary_review_module() -> ModuleType:
    """Load the actual review script without executing its CLI."""
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
    """Retain the review's explicit package creation hold."""
    payload = build_boundary_review()

    assert payload["schema"] == "s6_quantum_kuramoto_boundary_review_v1"
    assert payload["package_skeleton_allowed"] is False
    assert "requires refactors" in payload["reason"]


def test_boundary_review_proposes_core_api_surface() -> None:
    """Keep the reviewed async API and exclude the full parent runner."""
    payload = build_boundary_review()
    exports = {row["proposed_export"] for row in payload["proposed_public_api"]}

    assert "quantum_kuramoto.phase.xy_kuramoto" in exports
    assert "quantum_kuramoto.phase.xy_compiler" in exports
    assert "quantum_kuramoto.hardware.async_runner" in exports
    assert "quantum_kuramoto.hardware.runner" not in exports


def test_boundary_review_decides_all_needs_review_rows() -> None:
    """Retain the explicit decisions for every review-dependent module."""
    payload = build_boundary_review()
    decisions = {row["module"]: row["decision"] for row in payload["needs_review_decisions"]}

    assert decisions["scpn_quantum_control.hardware.runner"] == "defer"
    assert (
        decisions["scpn_quantum_control.hardware.hybrid_digital_analog"] == "promote_after_facade"
    )
    assert all(decision in {"defer", "promote_after_facade"} for decision in decisions.values())


def test_boundary_review_binds_the_declared_current_audit(tmp_path: Path) -> None:
    """Load and bind the declared current audit by its actual byte digest."""
    audit = build_split_audit()
    source = tmp_path / "actual-audit.json"
    source.write_text(json.dumps(audit), encoding="utf-8")
    expected_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    from_file = build_boundary_review(source_audit=source)
    from_object = build_boundary_review(audit, source_audit=source)
    assert from_file == from_object
    assert from_file["source_audit"] == str(source)
    assert from_file["source_audit_sha256"] == expected_digest
    assert from_file["package_skeleton_allowed"] is False
    assert len(from_file["proposed_public_api"]) == 16


def test_boundary_review_keeps_in_memory_audit_unbound() -> None:
    """Avoid attributing a supplied object to an unrelated archived file."""
    payload = build_boundary_review(build_split_audit())
    assert payload["source_audit"] is None
    assert payload["source_audit_sha256"] is None


def test_boundary_review_rejects_mislabelled_input(tmp_path: Path) -> None:
    """Reject a source file whose content differs from the supplied audit."""
    audit = build_split_audit()
    source = tmp_path / "different-audit.json"
    source.write_text(json.dumps({**audit, "date": "different"}), encoding="utf-8")
    with pytest.raises(ValueError, match="declared audit source differs"):
        build_boundary_review(audit, source_audit=source)


def test_boundary_review_rejects_non_object_input_file(tmp_path: Path) -> None:
    """Reject a declared audit file containing a JSON array."""
    source = tmp_path / "array.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        build_boundary_review(source_audit=source)


def test_boundary_review_rejects_missing_rows() -> None:
    """Reject an audit that cannot provide module review rows."""
    with pytest.raises(ValueError, match="must contain rows"):
        build_boundary_review({"rows": None})


@pytest.mark.parametrize(
    "missing,expected",
    [
        ("scpn_quantum_control.accel.rust_import", "public API modules missing"),
        ("scpn_quantum_control.hardware.provenance", "needs-review module missing"),
    ],
)
def test_boundary_review_rejects_missing_required_module(missing: str, expected: str) -> None:
    """Require both the proposed API and deferred module observations."""
    audit = build_split_audit()
    rows = audit["rows"]
    assert isinstance(rows, list)
    audit["rows"] = [row for row in rows if row["module"] != missing]
    with pytest.raises(ValueError, match=expected):
        build_boundary_review(audit)
