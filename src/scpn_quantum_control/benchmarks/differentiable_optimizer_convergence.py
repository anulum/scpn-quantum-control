# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Optimizer Convergence Evidence
"""Benchmark artefacts for ground-state optimizer convergence rows."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phase.optimizer_convergence_suite import (
    GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
    GroundStateOptimizerConvergenceSuiteResult,
    run_ground_state_optimizer_convergence_suite,
)

GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA = "scpn_qc_ground_state_optimizer_convergence_v1"


@dataclass(frozen=True)
class GroundStateOptimizerConvergenceArtifact:
    """Written ground-state optimizer convergence artefact metadata."""

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


def ground_state_optimizer_convergence_payload(
    suite: GroundStateOptimizerConvergenceSuiteResult | None = None,
    *,
    artifact_id: str = "ground-state-optimizer-convergence-local",
) -> dict[str, Any]:
    """Return the BL-15 optimizer convergence artifact payload."""
    normalized_id = artifact_id.strip()
    if not normalized_id:
        raise ValueError("artifact_id must be non-empty")
    result = suite if suite is not None else run_ground_state_optimizer_convergence_suite()
    if result.evidence_class != GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS:
        raise ValueError("suite evidence_class must be functional_non_isolated")
    return {
        "schema": GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA,
        "artifact_id": normalized_id,
        "artifact_date": "2026-07-09",
        "classification": GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        "production_eligible": False,
        "promotion_ready": False,
        "row_schema": {
            "required_run_fields": [
                "case_id",
                "optimizer",
                "method",
                "status",
                "best_value",
                "exact_ground_energy",
                "iterations",
                "evaluations",
                "wall_time_seconds",
                "certificate",
                "evidence_class",
                "claim_boundary",
            ],
            "required_boundary_fields": [
                "case_id",
                "optimizer",
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


def render_ground_state_optimizer_convergence_markdown(payload: dict[str, Any]) -> str:
    """Render optimizer convergence payload as bounded Markdown evidence."""
    suite = payload["suite"]
    rows = payload["rows"]
    boundary_rows = payload["boundary_rows"]
    lines = [
        "# Ground-State Optimizer Convergence Evidence",
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
        f"- Optimizers: `{', '.join(suite['optimizer_names'])}`",
        "",
        "## Executable Rows",
        "",
        "| Case | Optimizer | Best energy | Energy error | Parameter distance | Evaluations | Passed |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        certificate = row["certificate"]
        lines.append(
            "| `{case}` | `{optimizer}` | {best:.12g} | {energy_error:.3g} | "
            "{distance:.3g} | {evaluations} | `{passed}` |".format(
                case=row["case_id"],
                optimizer=row["optimizer"],
                best=float(row["best_value"]),
                energy_error=float(certificate["energy_error"]),
                distance=float(certificate["parameter_distance"]),
                evaluations=int(row["evaluations"]),
                passed=certificate["passed"],
            )
        )
    if boundary_rows:
        lines.extend(
            [
                "",
                "## Boundary Rows",
                "",
                "| Case | Optimizer | Failure class | Boundary |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in boundary_rows:
            lines.append(
                "| `{case}` | `{optimizer}` | `{failure}` | {boundary} |".format(
                    case=row["case_id"],
                    optimizer=row["optimizer"],
                    failure=row["failure_class"],
                    boundary=row["setup_instructions"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_ground_state_optimizer_convergence_artifact(
    output_path: str | Path,
    *,
    markdown_path: str | Path | None = None,
    suite: GroundStateOptimizerConvergenceSuiteResult | None = None,
    artifact_id: str = "ground-state-optimizer-convergence-local",
) -> GroundStateOptimizerConvergenceArtifact:
    """Write JSON and Markdown BL-15 optimizer convergence artefacts."""
    json_destination = Path(output_path)
    if json_destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    markdown_destination = (
        json_destination.with_suffix(".md") if markdown_path is None else Path(markdown_path)
    )
    if markdown_destination.suffix.lower() != ".md":
        raise ValueError("markdown_path must end with .md")
    payload = ground_state_optimizer_convergence_payload(suite, artifact_id=artifact_id)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_destination.write_text(
        render_ground_state_optimizer_convergence_markdown(payload),
        encoding="utf-8",
    )
    return GroundStateOptimizerConvergenceArtifact(
        artifact_id=str(payload["artifact_id"]),
        json_path=json_destination,
        markdown_path=markdown_destination,
        row_count=len(payload["rows"]),
        boundary_count=len(payload["boundary_rows"]),
        classification=GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        claim_boundary=GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    )


__all__ = [
    "GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA",
    "GroundStateOptimizerConvergenceArtifact",
    "ground_state_optimizer_convergence_payload",
    "render_ground_state_optimizer_convergence_markdown",
    "write_ground_state_optimizer_convergence_artifact",
]
