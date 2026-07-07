# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — transform support-matrix committed-artefact tests
"""Tests for the transform-algebra support-matrix committed artefact (ST-03)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scpn_quantum_control.differentiable_transform_algebra import (
    TransformAlgebraAudit,
    run_transform_algebra_audit,
)
from scpn_quantum_control.differentiable_transform_support_matrix import (
    REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS,
    TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY,
)
from scpn_quantum_control.differentiable_transform_support_matrix_artifact import (
    DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH,
    DEFAULT_TRANSFORM_SUPPORT_MATRIX_MARKDOWN_PATH,
    TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID,
    TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA,
    TransformSupportMatrixArtifactValidation,
    build_transform_support_matrix_artifact,
    main,
    render_transform_support_matrix_markdown,
    validate_transform_support_matrix_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH
COMMITTED_MARKDOWN = REPO_ROOT / DEFAULT_TRANSFORM_SUPPORT_MATRIX_MARKDOWN_PATH


def _failed_audit() -> TransformAlgebraAudit:
    """Return an audit whose first executed case is forced to failed."""
    audit = run_transform_algebra_audit()
    first = audit.cases[0]
    tampered = dataclasses.replace(
        first,
        status="failed",
        residual=first.tolerance * 2.0,
    )
    return dataclasses.replace(audit, cases=(tampered, *audit.cases[1:]))


def test_payload_carries_all_support_rows_verbatim() -> None:
    """The payload mirrors the generated matrix row for row."""
    audit = run_transform_algebra_audit()
    payload = build_transform_support_matrix_artifact(audit)
    assert payload["schema"] == TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA
    assert payload["artifact_id"] == TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID
    assert payload["claim_boundary"] == TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY
    rows = payload["support_matrix"]
    assert isinstance(rows, list)
    assert [row["row_id"] for row in rows] == list(REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS)
    assert rows == [row.to_dict() for row in audit.support_matrix]
    assert payload["row_count"] == len(rows)
    assert payload["supported_row_count"] == sum(1 for row in rows if row["supported"])
    assert payload["blocked_row_count"] == sum(1 for row in rows if row["status"] == "blocked")
    assert payload["audit_case_count"] == len(audit.cases)


def test_failing_audit_is_never_serialised() -> None:
    """An audit with a failed case refuses to build a committed artefact."""
    with pytest.raises(ValueError, match="audit failed"):
        build_transform_support_matrix_artifact(_failed_audit())


def test_incomplete_audit_reports_missing_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing categories and missing required rows are named in the refusal."""
    import scpn_quantum_control.differentiable_transform_algebra as algebra

    audit = run_transform_algebra_audit()
    truncated = dataclasses.replace(audit, cases=audit.cases[1:])
    with pytest.raises(ValueError, match="missing categories: grad_vmap_composition"):
        build_transform_support_matrix_artifact(truncated)
    monkeypatch.setattr(
        algebra,
        "REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS",
        (*REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS, "phantom_row"),
    )
    with pytest.raises(ValueError, match="missing support rows: phantom_row"):
        build_transform_support_matrix_artifact(run_transform_algebra_audit())


def test_markdown_renders_every_row_and_the_boundary() -> None:
    """The markdown table carries one line per row plus the claim boundary."""
    payload = build_transform_support_matrix_artifact()
    rendered = render_transform_support_matrix_markdown(payload)
    assert "# Differentiable Transform-Algebra Support Matrix" in rendered
    for row_id in REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS:
        assert f"| `{row_id}` |" in rendered
    assert TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY in rendered
    assert "`n/a`" in rendered
    assert rendered.endswith("its generated status.\n")


def test_markdown_fails_closed_on_malformed_payload() -> None:
    """A payload without a row list refuses to render."""
    with pytest.raises(ValueError, match="list of row objects"):
        render_transform_support_matrix_markdown({"support_matrix": "not-a-list"})


def test_markdown_renders_malformed_cells_as_empty() -> None:
    """Non-list cell values render as empty cells instead of crashing."""
    payload = build_transform_support_matrix_artifact()
    rows = payload["support_matrix"]
    assert isinstance(rows, list)
    rows[0]["notes"] = "not-a-list"
    rendered = render_transform_support_matrix_markdown(payload)
    assert rendered.count("| `native_grad_vmap` |") == 1


def test_committed_artifact_is_current() -> None:
    """The committed JSON and markdown match a fresh audit regeneration."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    validation = validate_transform_support_matrix_artifact(committed)
    assert validation.passed, validation.errors
    rendered = render_transform_support_matrix_markdown(committed)
    assert COMMITTED_MARKDOWN.read_text(encoding="utf-8") == rendered


def test_validation_flags_tampered_fields_and_residual_drift() -> None:
    """Field tampering and beyond-tolerance residual drift are both caught."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    committed["row_count"] = 99
    committed["support_matrix"][0]["status"] = "blocked"
    committed["support_matrix"][1]["residual"] = 1.0
    committed["support_matrix"][2]["residual"] = None
    committed["support_matrix"][3]["residual"] = "not-a-number"
    validation = validate_transform_support_matrix_artifact(committed)
    assert not validation.passed
    assert any("row_count" in error for error in validation.errors)
    assert any("'status'" in error for error in validation.errors)
    assert any("drifts beyond the row tolerance" in error for error in validation.errors)
    assert any("residual presence does not match" in error for error in validation.errors)
    assert any("residual must be numeric" in error for error in validation.errors)


def test_validation_flags_row_shape_mismatches() -> None:
    """Missing rows and non-object rows are reported instead of crashing."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    short = dict(committed)
    short["support_matrix"] = committed["support_matrix"][:-1]
    validation = validate_transform_support_matrix_artifact(short)
    assert any("row list does not match" in error for error in validation.errors)
    scalar_rows = dict(committed)
    scalar_rows["support_matrix"] = ["not-a-row"] * len(committed["support_matrix"])
    validation = validate_transform_support_matrix_artifact(scalar_rows)
    assert any("must be JSON objects" in error for error in validation.errors)


def test_validation_verdict_invariants_hold() -> None:
    """The verdict dataclass rejects inconsistent pass/error combinations."""
    with pytest.raises(ValueError, match="must not carry errors"):
        TransformSupportMatrixArtifactValidation(passed=True, errors=("x",))
    with pytest.raises(ValueError, match="must explain its errors"):
        TransformSupportMatrixArtifactValidation(passed=False, errors=())


def test_main_check_passes_on_the_committed_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI check mode confirms the committed artefact is current."""
    assert main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "current" in captured.out
    assert captured.err == ""


def test_main_write_and_check_round_trip(tmp_path: Path) -> None:
    """Write mode emits both files and check mode accepts them."""
    json_path = tmp_path / "matrix.json"
    markdown_path = tmp_path / "matrix.md"
    arguments = [
        "--json-path",
        str(json_path),
        "--markdown-path",
        str(markdown_path),
    ]
    assert main(["--write", *arguments]) == 0
    assert json_path.exists() and markdown_path.exists()
    assert main(["--check", *arguments]) == 0
    markdown_path.write_text("stale rendering\n", encoding="utf-8")
    assert main(["--check", *arguments]) == 1
    tampered = json.loads(json_path.read_text(encoding="utf-8"))
    tampered["support_matrix"][0]["supported"] = False
    json_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["--check", *arguments]) == 1


def test_main_default_mode_prints_the_payload(capsys: pytest.CaptureFixture[str]) -> None:
    """Without flags the CLI prints the regenerated JSON payload."""
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA
    assert len(payload["support_matrix"]) == len(REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS)
