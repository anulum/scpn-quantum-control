#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2. The Geometry of Functional Dynamics (L4): spec builder
"""Promote Paper 0 2. The Geometry of Functional Dynamics (L4): records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R04840",
    "P0R04841",
    "P0R04842",
    "P0R04843",
    "P0R04844",
    "P0R04845",
    "P0R04846",
    "P0R04847",
    "P0R04848",
)
CLAIM_BOUNDARY = "source-bounded section 2 the geometry of functional dynamics l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_the_geometry_of_functional_dynamics_l4.2_the_geometry_of_functional_dynamics_l4": {
        "context_id": "2_the_geometry_of_functional_dynamics_l4",
        "validation_protocol": "paper0.section_2_the_geometry_of_functional_dynamics_l4.2_the_geometry_of_functional_dynamics_l4",
        "canonical_statement": "The source-bounded component '2. The Geometry of Functional Dynamics (L4):' preserves Paper 0 records P0R04840-P0R04843 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04840:2_the_geometry_of_functional_dynamics_l4",
            "P0R04841:2_the_geometry_of_functional_dynamics_l4",
            "P0R04842:2_the_geometry_of_functional_dynamics_l4",
            "P0R04843:2_the_geometry_of_functional_dynamics_l4",
        ),
        "source_formulae": (
            "P0R04840: 2. The Geometry of Functional Dynamics (L4):",
            "P0R04841: The dynamics of the brain (governed by the UPDE) unfold in a high-dimensional phase space.",
            "P0R04842: The Synchronisation Manifold (MSync): Coherent brain activity corresponds to the system collapsing onto a lower-dimensional geometric structure (MSync) within this phase space. | The Geometry of Information Flow (Travelling Waves): Activity propagates across the cortex as Travelling Waves. The spatial geometry of these waves (velocity, direction, curvature) defines the routing of information, implementing the HPC architecture dynamically. | Fractal Dynamics (Quasicriticality): The Quasicritical state (sigma1) results in a Scale-Free (Fractal) geometry of activity in both space (neuronal avalanches) and time (LRTC), optimising information transfer across the geometric scaffold.",
            "P0R04843: P0R04843",
        ),
        "test_protocols": (
            "preserve 2. The Geometry of Functional Dynamics (L4): source-accounting boundary",
        ),
        "null_results": (
            "2. The Geometry of Functional Dynamics (L4): is not empirical validation evidence",
        ),
        "variables": ("2_the_geometry_of_functional_dynamics_l4",),
        "validation_targets": ("preserve records P0R04840-P0R04843",),
        "null_controls": (
            "2_the_geometry_of_functional_dynamics_l4 must remain source-bounded accounting",
        ),
    },
    "section_2_the_geometry_of_functional_dynamics_l4.v_the_geometry_of_cognition_and_subjective_experience_l5": {
        "context_id": "v_the_geometry_of_cognition_and_subjective_experience_l5",
        "validation_protocol": "paper0.section_2_the_geometry_of_functional_dynamics_l4.v_the_geometry_of_cognition_and_subjective_experience_l5",
        "canonical_statement": "The source-bounded component 'V. The Geometry of Cognition and Subjective Experience (L5)' preserves Paper 0 records P0R04844-P0R04845 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04844:v_the_geometry_of_cognition_and_subjective_experience_l5",
            "P0R04845:v_the_geometry_of_cognition_and_subjective_experience_l5",
        ),
        "source_formulae": (
            "P0R04844: V. The Geometry of Cognition and Subjective Experience (L5)",
            "P0R04845: In L5, the geometry of the physical substrate is transduced into the intrinsic geometry of conscious experience. This is the core of the Geometric Qualia hypothesis.",
        ),
        "test_protocols": (
            "preserve V. The Geometry of Cognition and Subjective Experience (L5) source-accounting boundary",
        ),
        "null_results": (
            "V. The Geometry of Cognition and Subjective Experience (L5) is not empirical validation evidence",
        ),
        "variables": ("v_the_geometry_of_cognition_and_subjective_experience_l5",),
        "validation_targets": ("preserve records P0R04844-P0R04845",),
        "null_controls": (
            "v_the_geometry_of_cognition_and_subjective_experience_l5 must remain source-bounded accounting",
        ),
    },
    "section_2_the_geometry_of_functional_dynamics_l4.1_the_consciousness_manifold_m_the_geometry_of_qualia": {
        "context_id": "1_the_consciousness_manifold_m_the_geometry_of_qualia",
        "validation_protocol": "paper0.section_2_the_geometry_of_functional_dynamics_l4.1_the_consciousness_manifold_m_the_geometry_of_qualia",
        "canonical_statement": "The source-bounded component '1. The Consciousness Manifold (M): The Geometry of Qualia' preserves Paper 0 records P0R04846-P0R04848 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04846:1_the_consciousness_manifold_m_the_geometry_of_qualia",
            "P0R04847:1_the_consciousness_manifold_m_the_geometry_of_qualia",
            "P0R04848:1_the_consciousness_manifold_m_the_geometry_of_qualia",
        ),
        "source_formulae": (
            "P0R04846: 1. The Consciousness Manifold (M): The Geometry of Qualia",
            "P0R04847: Subjective experience (Qualia) is the intrinsic geometry of the high-dimensional neural state space (M).",
            'P0R04848: The Metric Tensor (gmu - Distinguishability): The Fisher Information Metric defines the geometry of distinguishability between conscious states. It quantifies the "felt distance" between experiences. | Curvature (Valence and Intensity): The intrinsic curvature of M defines the affective quality of experience. Ricci Scalar Curvature (R): Corresponds to the intensity of the experience. | Curvature Sign: Positive curvature corresponds to high integration (low Free Energy) and positive valence. Negative curvature corresponds to fragmentation (high Free Energy) and negative valence. | Topology (bk - Richness and Structure): The topological features, quantified by Betti numbers (bk) derived via Topological Data Analysis (TDA), define the richness and structure of the conscious experience. Cognitive flexibility and altered states correspond to increased topological complexity (High bk).',
        ),
        "test_protocols": (
            "preserve 1. The Consciousness Manifold (M): The Geometry of Qualia source-accounting boundary",
        ),
        "null_results": (
            "1. The Consciousness Manifold (M): The Geometry of Qualia is not empirical validation evidence",
        ),
        "variables": ("1_the_consciousness_manifold_m_the_geometry_of_qualia",),
        "validation_targets": ("preserve records P0R04846-P0R04848",),
        "null_controls": (
            "1_the_consciousness_manifold_m_the_geometry_of_qualia must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section2TheGeometryOfFunctionalDynamicsL4Spec:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class Section2TheGeometryOfFunctionalDynamicsL4SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section2TheGeometryOfFunctionalDynamicsL4Spec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_section_2_the_geometry_of_functional_dynamics_l4_specs(
    source_records: list[dict[str, Any]],
) -> Section2TheGeometryOfFunctionalDynamicsL4SpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[Section2TheGeometryOfFunctionalDynamicsL4Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section2TheGeometryOfFunctionalDynamicsL4Spec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 " + "2. The Geometry of Functional Dynamics (L4):" + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R04849",
    }
    return Section2TheGeometryOfFunctionalDynamicsL4SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section2TheGeometryOfFunctionalDynamicsL4SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_the_geometry_of_functional_dynamics_l4_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section2TheGeometryOfFunctionalDynamicsL4SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2. The Geometry of Functional Dynamics (L4):" + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: Section2TheGeometryOfFunctionalDynamicsL4SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_the_geometry_of_functional_dynamics_l4_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_the_geometry_of_functional_dynamics_l4_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
