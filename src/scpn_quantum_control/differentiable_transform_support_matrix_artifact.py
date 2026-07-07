# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — transform-algebra support-matrix committed artefact.
"""Committed-artefact emission for the transform-algebra support matrix.

The support matrix (:mod:`scpn_quantum_control.differentiable_transform_support_matrix`)
is generated from the executable transform-algebra audit
(:mod:`scpn_quantum_control.differentiable_transform_algebra`), so it mirrors the
test battery instead of maintaining a hand-written capability claim. This module
serialises that generated matrix into one committed JSON artefact plus a rendered
markdown table, and validates the committed artefact against a fresh audit run.

Emission is fail-closed: an audit with failed cases, missing categories, or
missing support rows is never serialised. Validation compares every field
verbatim except executed residuals, which are floating-point measurements and
are therefore compared within each row's own tolerance instead of bit-exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .differentiable_claim_ledger import REPO_ROOT
from .differentiable_transform_algebra import (
    TransformAlgebraAudit,
    run_transform_algebra_audit,
)
from .differentiable_transform_support_matrix import (
    REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS,
    TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY,
)

TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA: Final[str] = (
    "scpn_qc_differentiable_transform_support_matrix_v1"
)
"""Schema identifier stamped into the committed support-matrix artefact."""

TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID: Final[str] = "diff-transform-support-matrix-20260708"
"""Artifact identifier of the committed support-matrix emission."""

DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/differentiable_transform_support_matrix_20260708.json"
)
"""Repository-relative path of the committed JSON artefact."""

DEFAULT_TRANSFORM_SUPPORT_MATRIX_MARKDOWN_PATH: Final[Path] = Path(
    "data/differentiable_phase_qnode/differentiable_transform_support_matrix_20260708.md"
)
"""Repository-relative path of the committed markdown rendering."""

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.differentiable_transform_support_matrix_artifact --write"
)

_MARKDOWN_HEADER: Final[str] = """<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Transform-Algebra Support Matrix
-->"""


@dataclass(frozen=True)
class TransformSupportMatrixArtifactValidation:
    """Validation verdict for a committed support-matrix artefact payload.

    Parameters
    ----------
    passed
        Whether the committed payload matches a fresh audit regeneration.
    errors
        Human-readable mismatch descriptions, empty when ``passed``.
    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that the verdict and its error list agree."""
        if self.passed and self.errors:
            raise ValueError("a passed validation must not carry errors")
        if not self.passed and not self.errors:
            raise ValueError("a failed validation must explain its errors")


def build_transform_support_matrix_artifact(
    audit: TransformAlgebraAudit | None = None,
) -> dict[str, object]:
    """Build the committed-artefact payload for the support matrix.

    Parameters
    ----------
    audit
        A transform-algebra audit to serialise, or ``None`` to run
        :func:`run_transform_algebra_audit` on the current tree.

    Returns
    -------
    dict[str, object]
        JSON-ready payload carrying every generated support row verbatim.

    Raises
    ------
    ValueError
        If the audit did not pass — failed cases, missing categories, or
        missing support rows are never serialised into a committed artefact.
    """
    resolved = run_transform_algebra_audit() if audit is None else audit
    if not resolved.passed:
        details: list[str] = []
        if resolved.missing_categories:
            details.append("missing categories: " + ", ".join(resolved.missing_categories))
        if resolved.missing_support_rows:
            details.append("missing support rows: " + ", ".join(resolved.missing_support_rows))
        for case in resolved.failed_cases:
            details.append(f"failed case {case.case_id} residual={case.residual}")
        for row in resolved.failed_support_rows:
            details.append(f"failed support row {row.row_id}")
        raise ValueError(
            "transform-algebra audit failed and is not serialised: " + "; ".join(details)
        )
    rows = resolved.support_matrix
    return {
        "schema": TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA,
        "artifact_id": TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY,
        "audit_case_count": len(resolved.cases),
        "audit_passed_count": len(resolved.passed_cases),
        "audit_blocked_count": len(resolved.blocked_cases),
        "required_rows": list(REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS),
        "row_count": len(rows),
        "supported_row_count": sum(1 for row in rows if row.supported),
        "blocked_row_count": sum(1 for row in rows if row.status == "blocked"),
        "support_matrix": [row.to_dict() for row in rows],
    }


def validate_transform_support_matrix_artifact(
    payload: dict[str, object],
    *,
    audit: TransformAlgebraAudit | None = None,
) -> TransformSupportMatrixArtifactValidation:
    """Validate a committed payload against a fresh audit regeneration.

    Every field is compared verbatim except executed row residuals, which are
    floating-point measurements: a committed residual matches when it differs
    from the regenerated one by at most the row's own tolerance.

    Parameters
    ----------
    payload
        The committed artefact payload (parsed JSON).
    audit
        Optional audit to regenerate from; defaults to a fresh run.

    Returns
    -------
    TransformSupportMatrixArtifactValidation
        The verdict with per-field mismatch descriptions.
    """
    errors: list[str] = []
    reference = build_transform_support_matrix_artifact(audit)
    for key, expected in reference.items():
        if key == "support_matrix":
            continue
        if payload.get(key) != expected:
            errors.append(f"field {key!r} does not match the regenerated artefact")
    committed_rows = payload.get("support_matrix")
    reference_rows = _row_list(reference)
    if not isinstance(committed_rows, list) or len(committed_rows) != len(reference_rows):
        errors.append("support_matrix row list does not match the regenerated artefact")
        committed_rows = []
    for committed, regenerated in zip(committed_rows, reference_rows, strict=False):
        if not isinstance(committed, dict):
            errors.append("support_matrix rows must be JSON objects")
            continue
        errors.extend(_row_mismatches(committed, regenerated))
    return TransformSupportMatrixArtifactValidation(
        passed=not errors,
        errors=tuple(errors),
    )


def _row_list(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return the payload's support rows, failing closed on malformed shapes."""
    rows = payload.get("support_matrix")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("support_matrix payload must carry a list of row objects")
    return rows


def _row_mismatches(committed: dict[str, object], regenerated: dict[str, object]) -> list[str]:
    """Return mismatch descriptions between one committed and regenerated row."""
    row_id = regenerated.get("row_id")
    mismatches: list[str] = []
    for key, expected in regenerated.items():
        if key == "residual":
            continue
        if committed.get(key) != expected:
            mismatches.append(f"row {row_id!r} field {key!r} does not match")
    committed_residual = committed.get("residual")
    regenerated_residual = regenerated.get("residual")
    tolerance = regenerated.get("tolerance")
    if regenerated_residual is None or committed_residual is None:
        if committed_residual != regenerated_residual:
            mismatches.append(f"row {row_id!r} residual presence does not match")
    elif (
        not isinstance(committed_residual, int | float)
        or not isinstance(regenerated_residual, int | float)
        or not isinstance(tolerance, int | float)
    ):
        mismatches.append(f"row {row_id!r} residual must be numeric")
    elif abs(float(committed_residual) - float(regenerated_residual)) > float(tolerance):
        mismatches.append(f"row {row_id!r} residual drifts beyond the row tolerance")
    return mismatches


def render_transform_support_matrix_markdown(payload: dict[str, object]) -> str:
    """Render the committed-artefact payload as a reviewer-facing markdown table.

    Parameters
    ----------
    payload
        The artefact payload from :func:`build_transform_support_matrix_artifact`.

    Returns
    -------
    str
        Markdown document with artefact metadata and one row per support row.
    """
    rows = _row_list(payload)
    lines = [
        _MARKDOWN_HEADER,
        "",
        "# Differentiable Transform-Algebra Support Matrix",
        "",
        f"- Schema: `{payload.get('schema')}`",
        f"- Artifact ID: `{payload.get('artifact_id')}`",
        f"- Supported rows: `{payload.get('supported_row_count')}/{payload.get('row_count')}`",
        f"- Fail-closed blocked rows: `{payload.get('blocked_row_count')}`",
        (
            f"- Source audit cases: `{payload.get('audit_case_count')}` "
            f"(`{payload.get('audit_passed_count')}` passed, "
            f"`{payload.get('audit_blocked_count')}` blocked)"
        ),
        f"- Claim boundary: {payload.get('claim_boundary')}",
        "",
        "| Row | Lane | Transform stack | Status | Residual | Tolerance | Blockers | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_markdown_row(row) for row in rows)
    lines.extend(
        [
            "",
            "Blocked rows are explicit fail-closed boundaries, not failures. The",
            "artefact mirrors the executable audit and cannot promote any row beyond",
            "its generated status.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_row(row: dict[str, object]) -> str:
    """Render one support row as a markdown table line."""
    residual = row.get("residual")
    residual_cell = f"{residual:.3e}" if isinstance(residual, int | float) else "n/a"
    tolerance = row.get("tolerance")
    tolerance_cell = f"{tolerance:.1e}" if isinstance(tolerance, int | float) else "n/a"
    cells = (
        f"`{row.get('row_id')}`",
        f"`{row.get('lane')}`",
        _joined_cell(row.get("transform_stack")),
        f"`{row.get('status')}`",
        f"`{residual_cell}`",
        f"`{tolerance_cell}`",
        _joined_cell(row.get("blocked_reasons")) or "—",
        _joined_cell(row.get("notes")),
    )
    return "| " + " | ".join(cells) + " |"


def _joined_cell(values: object) -> str:
    """Join a JSON string list into one escaped markdown cell."""
    if not isinstance(values, list):
        return ""
    return "<br>".join(
        str(value).replace("\n", " ").replace("|", "\\|") for value in values if str(value)
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print, write, or check the committed artefact.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``--check`` finds committed-artefact drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the JSON artefact and markdown rendering",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed artefact against a fresh audit",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPO_ROOT / DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH,
        help="committed JSON artefact path",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=REPO_ROOT / DEFAULT_TRANSFORM_SUPPORT_MATRIX_MARKDOWN_PATH,
        help="committed markdown rendering path",
    )
    args = parser.parse_args(argv)
    if args.check:
        committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        validation = validate_transform_support_matrix_artifact(committed)
        rendered = render_transform_support_matrix_markdown(committed)
        markdown_current = args.markdown_path.read_text(encoding="utf-8") == rendered
        if validation.passed and markdown_current:
            print("transform support-matrix artefact: current")
            return 0
        for error in validation.errors:
            print(error, file=sys.stderr)
        if not markdown_current:
            print("markdown rendering does not match the committed payload", file=sys.stderr)
        return 1
    payload = build_transform_support_matrix_artifact()
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.json_path.write_text(serialised, encoding="utf-8")
        args.markdown_path.write_text(
            render_transform_support_matrix_markdown(payload),
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
        print(f"wrote {args.markdown_path}")
        return 0
    print(serialised, end="")
    return 0


__all__ = [
    "DEFAULT_TRANSFORM_SUPPORT_MATRIX_JSON_PATH",
    "DEFAULT_TRANSFORM_SUPPORT_MATRIX_MARKDOWN_PATH",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_ID",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SCHEMA",
    "TransformSupportMatrixArtifactValidation",
    "build_transform_support_matrix_artifact",
    "render_transform_support_matrix_markdown",
    "validate_transform_support_matrix_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
