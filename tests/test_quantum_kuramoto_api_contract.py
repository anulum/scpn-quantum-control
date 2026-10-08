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

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts.audit_quantum_kuramoto_split import build_split_audit
from scripts.export_quantum_kuramoto_boundary_review import build_boundary_review


def _load_api_contract_module() -> ModuleType:
    """Load the real API contract script without executing its CLI."""
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
    """Retain the contract's explicit package creation hold."""
    payload = build_api_contract()

    assert payload["schema"] == "s6_quantum_kuramoto_api_contract_v1"
    assert payload["package_skeleton_allowed"] is False
    assert "before any separate package skeleton" in payload["reason"]


def test_api_contract_validates_proposed_export_names() -> None:
    """Exercise actual imports for every archived reviewed export."""
    payload = build_api_contract()

    assert payload["contract_passed"] is True
    assert payload["errors"] == []
    assert all(row["target_valid"] for row in payload["rows"])
    assert all(row["proposed_export"].startswith("quantum_kuramoto.") for row in payload["rows"])


def test_api_contract_blocks_unreviewed_runner_surface() -> None:
    """Exclude the parent runner from the archived proposed exports."""
    payload = build_api_contract()
    modules = {row["module"] for row in payload["rows"]}
    exports = {row["proposed_export"] for row in payload["rows"]}

    assert "scpn_quantum_control.hardware.runner" not in modules
    assert "quantum_kuramoto.hardware.runner" not in exports


def test_api_contract_flags_non_reusable_rows_as_warnings() -> None:
    """Preserve the original archived review's two isolation warnings."""
    payload = build_api_contract()
    warnings = payload["warnings"]

    assert any("hardware.async_runner" in warning for warning in warnings)
    assert any("hardware.analog_kuramoto" in warning for warning in warnings)
    assert payload["summary"]["warning_count"] == 2
    assert payload["summary"]["immediately_promotable_exports"] == 14


def test_current_s6_cli_chain_preserves_input_provenance(tmp_path: Path) -> None:
    """Execute all three real CLIs and bind each output to its real input."""
    root = Path(__file__).resolve().parents[1]
    outputs = [tmp_path / name for name in ("audit", "review", "contract")]
    names = ["split_audit", "boundary_review", "api_contract"]
    scripts = [
        "audit_quantum_kuramoto_split.py",
        "export_quantum_kuramoto_boundary_review.py",
        "export_quantum_kuramoto_api_contract.py",
    ]
    inputs = [[], ["--audit-path"], ["--review-path"]]
    previous: Path | None = None
    for script, name, output, input_flag in zip(scripts, names, outputs, inputs, strict=True):
        command = [sys.executable, str(root / "scripts" / script)]
        if input_flag:
            assert previous is not None
            command.extend([*input_flag, str(previous)])
        command.extend(["--out-dir", str(output), "--doc-path", str(output / "report.md")])
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        previous = output / f"quantum_kuramoto_{name}_2026-05-07.json"
        assert previous.is_file()
        assert (output / "report.md").is_file()
    audit_path = outputs[0] / "quantum_kuramoto_split_audit_2026-05-07.json"
    review_path = outputs[1] / "quantum_kuramoto_boundary_review_2026-05-07.json"
    assert previous is not None
    review = json.loads(review_path.read_text(encoding="utf-8"))
    contract = json.loads(previous.read_text(encoding="utf-8"))
    assert review["source_audit"] == str(audit_path)
    assert review["source_audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert contract["source_review"] == str(review_path)
    assert contract["source_review_sha256"] == hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert contract["contract_passed"] is True
    assert contract["summary"]["importable_exports"] == 16
    assert len(contract["rows"]) == 16
    assert review["package_skeleton_allowed"] is False
    assert contract["package_skeleton_allowed"] is False


def test_api_contract_binds_declared_review(tmp_path: Path) -> None:
    """Load and digest the declared current review, including object input."""
    review = build_boundary_review(build_split_audit())
    source = tmp_path / "actual-review.json"
    source.write_text(json.dumps(review), encoding="utf-8")
    from_file = build_api_contract(source_review=source)
    assert from_file == build_api_contract(review, source_review=source)
    assert from_file["source_review"] == str(source)
    assert from_file["source_review_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_api_contract_keeps_in_memory_review_unbound() -> None:
    """Leave an object-only review free of an invented archived reference."""
    contract = build_api_contract(build_boundary_review(build_split_audit()))
    assert contract["source_review"] is None
    assert contract["source_review_sha256"] is None


def test_api_contract_rejects_mislabelled_input(tmp_path: Path) -> None:
    """Reject source attribution to a file with different review content."""
    review = build_boundary_review(build_split_audit())
    source = tmp_path / "different-review.json"
    source.write_text(json.dumps({**review, "date": "different"}), encoding="utf-8")
    with pytest.raises(ValueError, match="declared review source differs"):
        build_api_contract(review, source_review=source)


def test_api_contract_rejects_non_object_review_file(tmp_path: Path) -> None:
    """Reject a review file containing a JSON array."""
    source = tmp_path / "array.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        build_api_contract(source_review=source)


def test_api_contract_rejects_unknown_schema() -> None:
    """Reject review inputs from an unknown schema."""
    with pytest.raises(ValueError, match="unexpected boundary-review schema"):
        build_api_contract({"schema": "unknown"})


def test_api_contract_rejects_missing_proposals() -> None:
    """Require explicit proposed exports before building a contract."""
    with pytest.raises(ValueError, match="must contain proposed_public_api"):
        build_api_contract({"schema": "s6_quantum_kuramoto_boundary_review_v1"})


def test_api_contract_reports_real_import_and_export_contract_errors() -> None:
    """Report actual missing imports, invalid targets, duplicate and blocked rows."""
    review = build_boundary_review(build_split_audit())
    review["proposed_public_api"] = [
        None,
        {"module": "module_that_does_not_exist", "proposed_export": "invalid!"},
        {
            "module": "scpn_quantum_control.hardware.runner",
            "proposed_export": "quantum_kuramoto.hardware.runner",
            "current_status": "needs_review",
        },
        {
            "module": "scpn_quantum_control.hardware.runner",
            "proposed_export": "quantum_kuramoto.hardware.runner",
            "current_status": "needs_review",
        },
    ]
    contract = build_api_contract(review)
    assert contract["contract_passed"] is False
    assert contract["summary"]["error_count"] == 6
    assert contract["rows"][0]["import_error"].startswith("ModuleNotFoundError:")
    assert contract["rows"][1]["blocked_source"] is True
    assert contract["rows"][2]["duplicate_export"] is True
    assert contract["summary"]["immediately_promotable_exports"] == 0
    assert contract["package_skeleton_allowed"] is False


@pytest.mark.parametrize("invalid", [False, True])
def test_api_contract_cli_renders_pass_and_refusal_reports(tmp_path: Path, invalid: bool) -> None:
    """Render warning-free and refused contracts through the real export CLI."""
    review = build_boundary_review(build_split_audit())
    if invalid:
        review["proposed_public_api"] = [
            {
                "module": "module_that_does_not_exist",
                "proposed_export": "invalid!",
                "current_status": "needs_review",
            }
        ]
    else:
        review["proposed_public_api"] = [
            row for row in review["proposed_public_api"] if row["current_status"] == "reusable"
        ]
        assert review["proposed_public_api"]
    source = tmp_path / "review.json"
    source.write_text(json.dumps(review), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / "contract.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "export_quantum_kuramoto_api_contract.py"),
            "--review-path",
            str(source),
            "--out-dir",
            str(tmp_path),
            "--doc-path",
            str(report),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    contract = json.loads(
        (tmp_path / "quantum_kuramoto_api_contract_2026-05-07.json").read_text(encoding="utf-8")
    )
    rendered = report.read_text(encoding="utf-8")
    assert contract["contract_passed"] is (not invalid)
    assert contract["package_skeleton_allowed"] is False
    assert ("## Errors" in rendered) is invalid
    assert ("## Warnings" in rendered) is invalid
    assert all(error in rendered for error in contract["errors"])
