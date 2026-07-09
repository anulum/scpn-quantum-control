# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Witness Evidence
"""Benchmark artefacts for bounded synchronisation-witness runs."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phase.synchronisation_witness import (
    SYNC_WITNESS_CLAIM_BOUNDARY,
    SYNC_WITNESS_EVIDENCE_CLASS,
    SyncWitnessSuiteResult,
    run_sync_witness_suite,
)

SYNC_WITNESS_EVIDENCE_SCHEMA = "scpn_qc_sync_witness_evidence_v1"


@dataclass(frozen=True)
class SyncWitnessEvidenceArtifact:
    """Written BL-18 synchronisation-witness artifact metadata."""

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


def sync_witness_evidence_payload(
    suite: SyncWitnessSuiteResult | None = None,
    *,
    artifact_id: str = "sync-witness-evidence-local",
) -> dict[str, Any]:
    """Return the BL-18 synchronisation-witness evidence payload."""
    normalized_id = artifact_id.strip()
    if not normalized_id:
        raise ValueError("artifact_id must be non-empty")
    result = suite if suite is not None else run_sync_witness_suite()
    if result.evidence_class != SYNC_WITNESS_EVIDENCE_CLASS:
        raise ValueError("suite evidence_class must be functional_non_isolated")
    return {
        "schema": SYNC_WITNESS_EVIDENCE_SCHEMA,
        "artifact_id": normalized_id,
        "artifact_date": "2026-07-09",
        "classification": SYNC_WITNESS_EVIDENCE_CLASS,
        "production_eligible": False,
        "promotion_ready": False,
        "row_schema": {
            "required_run_fields": [
                "case_id",
                "regime",
                "n_nodes",
                "order_parameter",
                "order_parameter_harmonic2",
                "order_parameter_std",
                "thresholds",
                "betti0_curve",
                "betti1_curve",
                "h0_persistence",
                "h1_persistence",
                "persistent_component_count",
                "dominant_h1_persistence",
                "reference_scale",
                "n_bootstrap",
                "noise_std",
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


def render_sync_witness_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the BL-18 payload as bounded Markdown evidence."""
    rows = payload["rows"]
    boundary_rows = payload["boundary_rows"]
    lines = [
        "# Synchronisation-Witness Evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Artifact id: `{payload['artifact_id']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Executable Rows",
        "",
        "| Case | Regime | Order r1 | Components | Dominant H1 | Passed |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{case}` | `{regime}` | {order:.6g} | {components} | "
            "{dominant:.6g} | `{passed}` |".format(
                case=row["case_id"],
                regime=row["regime"],
                order=float(row["order_parameter"]),
                components=int(row["persistent_component_count"]),
                dominant=float(row["dominant_h1_persistence"]),
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


def write_sync_witness_evidence_artifact(
    output_path: str | Path,
    *,
    markdown_path: str | Path | None = None,
    suite: SyncWitnessSuiteResult | None = None,
    artifact_id: str = "sync-witness-evidence-local",
) -> SyncWitnessEvidenceArtifact:
    """Write JSON and Markdown BL-18 synchronisation-witness artefacts."""
    json_destination = Path(output_path)
    if json_destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    markdown_destination = (
        json_destination.with_suffix(".md") if markdown_path is None else Path(markdown_path)
    )
    if markdown_destination.suffix.lower() != ".md":
        raise ValueError("markdown_path must end with .md")
    payload = sync_witness_evidence_payload(suite, artifact_id=artifact_id)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_destination.write_text(
        render_sync_witness_evidence_markdown(payload),
        encoding="utf-8",
    )
    return SyncWitnessEvidenceArtifact(
        artifact_id=str(payload["artifact_id"]),
        json_path=json_destination,
        markdown_path=markdown_destination,
        row_count=len(payload["rows"]),
        boundary_count=len(payload["boundary_rows"]),
        classification=SYNC_WITNESS_EVIDENCE_CLASS,
        claim_boundary=SYNC_WITNESS_CLAIM_BOUNDARY,
    )


__all__ = [
    "SYNC_WITNESS_EVIDENCE_SCHEMA",
    "SyncWitnessEvidenceArtifact",
    "render_sync_witness_evidence_markdown",
    "sync_witness_evidence_payload",
    "write_sync_witness_evidence_artifact",
]
