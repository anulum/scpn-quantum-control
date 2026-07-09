# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coupling-Recovery Evidence
"""Benchmark artefacts for bounded coupling time-series recovery."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phase.coupling_time_series_recovery import (
    COUPLING_RECOVERY_CLAIM_BOUNDARY,
    COUPLING_RECOVERY_EVIDENCE_CLASS,
    CouplingRecoverySuiteResult,
    run_coupling_recovery_suite,
)

COUPLING_RECOVERY_EVIDENCE_SCHEMA = "scpn_qc_coupling_recovery_evidence_v1"


@dataclass(frozen=True)
class CouplingRecoveryEvidenceArtifact:
    """Written BL-17 coupling-recovery artifact metadata."""

    artifact_id: str
    json_path: Path
    markdown_path: Path
    row_count: int
    boundary_count: int
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready artifact metadata."""
        return {
            "artifact_id": self.artifact_id,
            "json_path": str(self.json_path),
            "markdown_path": str(self.markdown_path),
            "row_count": self.row_count,
            "boundary_count": self.boundary_count,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
        }


def coupling_recovery_evidence_payload(
    suite: CouplingRecoverySuiteResult | None = None,
    *,
    artifact_id: str = "coupling-recovery-evidence-local",
) -> dict[str, Any]:
    """Return the BL-17 coupling-recovery evidence payload."""
    normalized_id = artifact_id.strip()
    if not normalized_id:
        raise ValueError("artifact_id must be non-empty")
    result = suite if suite is not None else run_coupling_recovery_suite()
    if result.evidence_class != COUPLING_RECOVERY_EVIDENCE_CLASS:
        raise ValueError("suite evidence_class must be functional_non_isolated")
    return {
        "schema": COUPLING_RECOVERY_EVIDENCE_SCHEMA,
        "artifact_id": normalized_id,
        "artifact_date": "2026-07-09",
        "classification": COUPLING_RECOVERY_EVIDENCE_CLASS,
        "production_eligible": False,
        "promotion_ready": False,
        "row_schema": {
            "required_run_fields": [
                "case_id",
                "family",
                "learned_couplings",
                "true_couplings",
                "abs_error",
                "max_abs_error",
                "rmse",
                "valid_fraction",
                "design_rank",
                "condition_number",
                "noise_std",
                "missing_fraction",
                "tolerance",
                "passed",
                "claim_boundary",
            ],
            "required_boundary_fields": [
                "boundary_id",
                "status",
                "reason",
                "claim_boundary",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "suite": result.to_dict(),
        "rows": [record.to_dict() for record in result.records],
        "boundary_rows": [row.to_dict() for row in result.boundary_rows],
        "passed": result.passed,
        "claim_boundary": result.claim_boundary,
    }


def render_coupling_recovery_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the BL-17 payload as bounded Markdown evidence."""
    rows = payload["rows"]
    boundary_rows = payload["boundary_rows"]
    lines = [
        "# Coupling-Recovery Evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Artifact id: `{payload['artifact_id']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Executable Rows",
        "",
        "| Case | Family | Max error | RMSE | Valid rows | Rank | Passed |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{case}` | `{family}` | {max_error:.6g} | {rmse:.6g} | "
            "{valid:.3f} | {rank} | `{passed}` |".format(
                case=row["case_id"],
                family=row["family"],
                max_error=float(row["max_abs_error"]),
                rmse=float(row["rmse"]),
                valid=float(row["valid_fraction"]),
                rank=int(row["design_rank"]),
                passed=row["passed"],
            )
        )
    if boundary_rows:
        lines.extend(
            [
                "",
                "## Boundary Rows",
                "",
                "| Boundary | Status | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for row in boundary_rows:
            lines.append(
                "| `{boundary}` | `{status}` | {reason} |".format(
                    boundary=row["boundary_id"],
                    status=row["status"],
                    reason=row["reason"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_coupling_recovery_evidence_artifact(
    output_path: str | Path,
    *,
    markdown_path: str | Path | None = None,
    suite: CouplingRecoverySuiteResult | None = None,
    artifact_id: str = "coupling-recovery-evidence-local",
) -> CouplingRecoveryEvidenceArtifact:
    """Write JSON and Markdown BL-17 coupling-recovery artefacts."""
    json_destination = Path(output_path)
    if json_destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    markdown_destination = (
        json_destination.with_suffix(".md") if markdown_path is None else Path(markdown_path)
    )
    if markdown_destination.suffix.lower() != ".md":
        raise ValueError("markdown_path must end with .md")
    payload = coupling_recovery_evidence_payload(suite, artifact_id=artifact_id)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_destination.write_text(
        render_coupling_recovery_evidence_markdown(payload),
        encoding="utf-8",
    )
    return CouplingRecoveryEvidenceArtifact(
        artifact_id=str(payload["artifact_id"]),
        json_path=json_destination,
        markdown_path=markdown_destination,
        row_count=len(payload["rows"]),
        boundary_count=len(payload["boundary_rows"]),
        classification=COUPLING_RECOVERY_EVIDENCE_CLASS,
        claim_boundary=COUPLING_RECOVERY_CLAIM_BOUNDARY,
    )


__all__ = [
    "COUPLING_RECOVERY_EVIDENCE_SCHEMA",
    "CouplingRecoveryEvidenceArtifact",
    "coupling_recovery_evidence_payload",
    "render_coupling_recovery_evidence_markdown",
    "write_coupling_recovery_evidence_artifact",
]
