# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 lane registry
"""Deterministic Paper 0 lane registry for source-bounded downstream flows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .axiom_ii_infoton_geometry_validation import (
    CLAIM_BOUNDARY as AXIOM_II_INFOTON_GEOMETRY_CLAIM_BOUNDARY,
)
from .axiom_ii_infoton_geometry_validation import (
    SOURCE_LEDGER_SPAN as AXIOM_II_INFOTON_GEOMETRY_LEDGER_SPAN,
)
from .foundational_viability_postulate_validation import (
    CLAIM_BOUNDARY as FOUNDATIONAL_VIABILITY_POSTULATE_CLAIM_BOUNDARY,
)
from .foundational_viability_postulate_validation import (
    SOURCE_LEDGER_SPAN as FOUNDATIONAL_VIABILITY_POSTULATE_LEDGER_SPAN,
)
from .front_matter_context_validation import (
    CLAIM_BOUNDARY as FRONT_MATTER_CONTEXT_CLAIM_BOUNDARY,
)
from .front_matter_context_validation import (
    SOURCE_LEDGER_SPAN as FRONT_MATTER_CONTEXT_LEDGER_SPAN,
)
from .spec_loader import DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE

PAPER0_LANE_REGISTRY_SCHEMA = "paper0_lane_registry_v1"
PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY = (
    "Paper 0 lane registry is source-bounded; it does not claim external empirical "
    "validation and has no QPU submission and no hardware execution path."
)
HARDWARE_BOUNDARY_LABEL = "no_qpu_submission; no_hardware_submission; source_only_projection_lane"
PAPER0_LANE_RECORD_SOURCE_STATUS = "source_methodology_no_experiment"

_FRONT_MATTER_CONTEXT_SPEC_KEY = "front_matter_context.collection_identity"


@dataclass(frozen=True, slots=True)
class Paper0LaneRecord:
    """Immutable, JSON-compatible record for one Paper 0 downstream lane."""

    lane_id: str
    source_status: str
    evidence_class: str
    blockers: tuple[str, ...]
    claim_boundary: str
    source_ledger_span: tuple[str, str]
    no_qpu_submission: bool
    no_hardware_submission: bool
    hardware_boundary: str
    related_spec_bundle_path: str | None = None
    related_spec_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready lane record."""
        return asdict(self)


def _sorted_records() -> tuple[Paper0LaneRecord, ...]:
    records = (
        Paper0LaneRecord(
            lane_id="P0L01_front_matter_context",
            source_status=PAPER0_LANE_RECORD_SOURCE_STATUS,
            evidence_class="source_boundary_context",
            blockers=("no_external_validation_submission", "downstream_boundary_projection"),
            claim_boundary=(
                f"{FRONT_MATTER_CONTEXT_CLAIM_BOUNDARY} and no external validation submission."
            ),
            source_ledger_span=FRONT_MATTER_CONTEXT_LEDGER_SPAN,
            no_qpu_submission=True,
            no_hardware_submission=True,
            hardware_boundary=HARDWARE_BOUNDARY_LABEL,
            related_spec_bundle_path=DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE,
            related_spec_key=_FRONT_MATTER_CONTEXT_SPEC_KEY,
        ),
        Paper0LaneRecord(
            lane_id="P0L02_axiom_ii_infoton_geometry",
            source_status=PAPER0_LANE_RECORD_SOURCE_STATUS,
            evidence_class="source_derivation_slice",
            blockers=(
                "downstream_validation_requires_new_hardware",
                "empirical_evidence_not_claimed",
            ),
            claim_boundary=(
                f"{AXIOM_II_INFOTON_GEOMETRY_CLAIM_BOUNDARY} no external validation claim."
            ),
            source_ledger_span=AXIOM_II_INFOTON_GEOMETRY_LEDGER_SPAN,
            no_qpu_submission=True,
            no_hardware_submission=True,
            hardware_boundary=HARDWARE_BOUNDARY_LABEL,
        ),
        Paper0LaneRecord(
            lane_id="P0L03_foundational_viability_postulate",
            source_status=PAPER0_LANE_RECORD_SOURCE_STATUS,
            evidence_class="source_postulate_bundle",
            blockers=("no_hardware_submission_path",),
            claim_boundary=(
                f"{FOUNDATIONAL_VIABILITY_POSTULATE_CLAIM_BOUNDARY} "
                "not a readiness or hardware claim."
            ),
            source_ledger_span=FOUNDATIONAL_VIABILITY_POSTULATE_LEDGER_SPAN,
            no_qpu_submission=True,
            no_hardware_submission=True,
            hardware_boundary=HARDWARE_BOUNDARY_LABEL,
        ),
    )
    return tuple(sorted(records, key=lambda record: record.lane_id))


_LANE_REGISTRY = _sorted_records()


def paper0_lane_registry_lanes() -> tuple[Paper0LaneRecord, ...]:
    """Return deterministic Paper 0 lane records."""

    return _LANE_REGISTRY


def paper0_lane_registry_payload() -> dict[str, Any]:
    """Return JSON-compatible Paper 0 lane registry payload."""

    lanes = [lane.to_dict() for lane in paper0_lane_registry_lanes()]
    return {
        "schema": PAPER0_LANE_REGISTRY_SCHEMA,
        "spdx_license_identifier": "AGPL-3.0-or-later",
        "claim_boundary": PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY,
        "no_qpu_submission": True,
        "no_hardware_submission": True,
        "lane_count": len(lanes),
        "lanes": lanes,
    }


def paper0_lane_registry_json(payload: dict[str, Any] | None = None) -> str:
    """Return canonical JSON text for the Paper 0 lane registry."""

    return json.dumps(payload or paper0_lane_registry_payload(), indent=2, sort_keys=True) + "\n"


def paper0_lane_registry_markdown(payload: dict[str, Any] | None = None) -> str:
    """Return canonical Markdown text for the Paper 0 lane registry."""

    registry = payload or paper0_lane_registry_payload()
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- Paper 0 lane registry -->",
        "",
        "# Paper 0 Lane Registry",
        "",
        "Status: source-bounded programme registry; no QPU or hardware submission.",
        "",
        "This generated artefact turns the processed Paper 0 source register into",
        "downstream experimental lanes. It is an index of candidate work and",
        "blockers, not external validation evidence.",
        "",
        "## Registry summary",
        "",
        f"- Schema: `{registry['schema']}`",
        f"- Lane count: `{registry['lane_count']}`",
        f"- QPU submission enabled: `{not registry['no_qpu_submission']}`",
        f"- Hardware submission enabled: `{not registry['no_hardware_submission']}`",
        "",
        "## Lanes",
        "",
        "| Lane ID | Evidence class | Source span | Blockers | Related spec |",
        "|---|---|---|---|---|",
    ]
    for lane in registry["lanes"]:
        span = " to ".join(lane["source_ledger_span"])
        blockers = ", ".join(lane["blockers"]) or "`none`"
        related_spec = lane.get("related_spec_bundle_path") or "`none`"
        lines.append(
            "| `{lane_id}` | `{evidence_class}` | `{span}` | {blockers} | {related_spec} |".format(
                lane_id=lane["lane_id"],
                evidence_class=lane["evidence_class"],
                span=span,
                blockers=blockers,
                related_spec=related_spec,
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility gate",
            "",
            "Regenerate and compare this registry with:",
            "",
            "```bash",
            "scpn-bench paper0-lane-registry-gate",
            "```",
            "",
            "## Claim boundary",
            "",
            str(registry["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "Paper0LaneRecord",
    "PAPER0_LANE_REGISTRY_CLAIM_BOUNDARY",
    "PAPER0_LANE_REGISTRY_SCHEMA",
    "HARDWARE_BOUNDARY_LABEL",
    "paper0_lane_registry_json",
    "paper0_lane_registry_lanes",
    "paper0_lane_registry_markdown",
    "paper0_lane_registry_payload",
]
