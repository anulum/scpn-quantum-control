# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-System Objective Evidence
"""Benchmark artefacts for bounded Lindblad and MCWF objectives."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phase.open_system_objectives import (
    OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
    OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS,
    OpenSystemObjectiveSuiteResult,
    run_open_system_objective_suite,
)

OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA = "scpn_qc_open_system_objective_evidence_v1"


@dataclass(frozen=True)
class OpenSystemObjectiveEvidenceArtifact:
    """Written BL-16 open-system objective artifact metadata."""

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


def open_system_objective_evidence_payload(
    suite: OpenSystemObjectiveSuiteResult | None = None,
    *,
    artifact_id: str = "open-system-objective-evidence-local",
) -> dict[str, Any]:
    """Return the BL-16 open-system objective evidence payload."""
    normalized_id = artifact_id.strip()
    if not normalized_id:
        raise ValueError("artifact_id must be non-empty")
    result = suite if suite is not None else run_open_system_objective_suite()
    if result.evidence_class != OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS:
        raise ValueError("suite evidence_class must be functional_non_isolated")
    return {
        "schema": OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA,
        "artifact_id": normalized_id,
        "artifact_date": "2026-07-09",
        "classification": OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS,
        "production_eligible": False,
        "promotion_ready": False,
        "row_schema": {
            "required_run_fields": [
                "case_id",
                "backend",
                "params",
                "value",
                "gradient",
                "evaluations",
                "final_order_parameter",
                "invariant_certificate",
                "reproducibility_certificate",
                "evidence_class",
                "claim_boundary",
            ],
            "required_boundary_fields": [
                "case_id",
                "backend",
                "status",
                "failure_class",
                "setup_instructions",
                "evidence_class",
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


def render_open_system_objective_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the BL-16 payload as bounded Markdown evidence."""
    suite = payload["suite"]
    rows = payload["rows"]
    boundary_rows = payload["boundary_rows"]
    lines = [
        "# Open-System Objective Evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Artifact id: `{payload['artifact_id']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- Objective cases: `{suite['case_count']}`",
        f"- Executable rows: `{suite['record_count']}`",
        f"- Boundary rows: `{len(boundary_rows)}`",
        f"- Backends: `{', '.join(suite['backend_names'])}`",
        "",
        "## Executable Rows",
        "",
        "| Case | Backend | Objective | Gradient | Final R | Certificate |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        certificate = row["invariant_certificate"] or row["reproducibility_certificate"]
        assert isinstance(certificate, dict)
        certificate_status = "passed" if certificate["passed"] else "failed"
        lines.append(
            "| `{case}` | `{backend}` | {value:.12g} | `{gradient}` | "
            "{final_r:.12g} | `{status}` |".format(
                case=row["case_id"],
                backend=row["backend"],
                value=float(row["value"]),
                gradient=", ".join(f"{float(component):.6g}" for component in row["gradient"]),
                final_r=float(row["final_order_parameter"]),
                status=certificate_status,
            )
        )
    if boundary_rows:
        lines.extend(
            [
                "",
                "## Boundary Rows",
                "",
                "| Case | Backend | Failure class | Boundary |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in boundary_rows:
            lines.append(
                "| `{case}` | `{backend}` | `{failure}` | {boundary} |".format(
                    case=row["case_id"],
                    backend=row["backend"],
                    failure=row["failure_class"],
                    boundary=row["setup_instructions"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_open_system_objective_evidence_artifact(
    output_path: str | Path,
    *,
    markdown_path: str | Path | None = None,
    suite: OpenSystemObjectiveSuiteResult | None = None,
    artifact_id: str = "open-system-objective-evidence-local",
) -> OpenSystemObjectiveEvidenceArtifact:
    """Write JSON and Markdown BL-16 open-system objective artefacts."""
    json_destination = Path(output_path)
    if json_destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    markdown_destination = (
        json_destination.with_suffix(".md") if markdown_path is None else Path(markdown_path)
    )
    if markdown_destination.suffix.lower() != ".md":
        raise ValueError("markdown_path must end with .md")
    payload = open_system_objective_evidence_payload(suite, artifact_id=artifact_id)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_destination.write_text(
        render_open_system_objective_evidence_markdown(payload),
        encoding="utf-8",
    )
    return OpenSystemObjectiveEvidenceArtifact(
        artifact_id=str(payload["artifact_id"]),
        json_path=json_destination,
        markdown_path=markdown_destination,
        row_count=len(payload["rows"]),
        boundary_count=len(payload["boundary_rows"]),
        classification=OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS,
        claim_boundary=OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
    )


__all__ = [
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA",
    "OpenSystemObjectiveEvidenceArtifact",
    "open_system_objective_evidence_payload",
    "render_open_system_objective_evidence_markdown",
    "write_open_system_objective_evidence_artifact",
]
