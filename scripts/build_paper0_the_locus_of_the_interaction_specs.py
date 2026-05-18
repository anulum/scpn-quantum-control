#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Locus of the Interaction: spec builder
"""Promote Paper 0 The Locus of the Interaction: records."""

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
    "P0R02551",
    "P0R02552",
    "P0R02553",
    "P0R02554",
    "P0R02555",
    "P0R02556",
    "P0R02557",
    "P0R02558",
    "P0R02559",
    "P0R02560",
    "P0R02561",
    "P0R02562",
    "P0R02563",
    "P0R02564",
    "P0R02565",
)
CLAIM_BOUNDARY = (
    "source-bounded the locus of the interaction source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_locus_of_the_interaction.the_locus_of_the_interaction": {
        "context_id": "the_locus_of_the_interaction",
        "validation_protocol": "paper0.the_locus_of_the_interaction.the_locus_of_the_interaction",
        "canonical_statement": "The source-bounded component 'The Locus of the Interaction:' preserves Paper 0 records P0R02551-P0R02552 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02551:the_locus_of_the_interaction",
            "P0R02552:the_locus_of_the_interaction",
        ),
        "source_formulae": (
            "P0R02551: The Locus of the Interaction:",
            "P0R02552: The Psis field is the Source in the central void. The collective state variable (sigma) is the complete state of the torus surface at a given moment in time. The H_int interaction is the process by which the Source (Psis) continuously influences the dynamics on the surface (sigma).",
        ),
        "test_protocols": ("preserve The Locus of the Interaction: source-accounting boundary",),
        "null_results": ("The Locus of the Interaction: is not empirical validation evidence",),
        "variables": ("the_locus_of_the_interaction",),
        "validation_targets": ("preserve records P0R02551-P0R02552",),
        "null_controls": ("the_locus_of_the_interaction must remain source-bounded accounting",),
    },
    "the_locus_of_the_interaction.holonomy_as_memory_of_coupling": {
        "context_id": "holonomy_as_memory_of_coupling",
        "validation_protocol": "paper0.the_locus_of_the_interaction.holonomy_as_memory_of_coupling",
        "canonical_statement": "The source-bounded component 'Holonomy as Memory of Coupling:' preserves Paper 0 records P0R02553-P0R02554 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02553:holonomy_as_memory_of_coupling",
            "P0R02554:holonomy_as_memory_of_coupling",
        ),
        "source_formulae": (
            "P0R02553: Holonomy as Memory of Coupling:",
            'P0R02554: The introduction of holonomy adds a profound new dimension. It implies that the history of past H_int interactions is not lost. It is encoded directly into the geometry of the flow on the torus surface as a "Berry Phase." This geometric memory biases all future dynamics, meaning the system\'s past experiences of coupling with the universal field shape its future evolution. This is a physical mechanism for karma or path-dependent learning at the most fundamental level.',
        ),
        "test_protocols": ("preserve Holonomy as Memory of Coupling: source-accounting boundary",),
        "null_results": ("Holonomy as Memory of Coupling: is not empirical validation evidence",),
        "variables": ("holonomy_as_memory_of_coupling",),
        "validation_targets": ("preserve records P0R02553-P0R02554",),
        "null_controls": ("holonomy_as_memory_of_coupling must remain source-bounded accounting",),
    },
    "the_locus_of_the_interaction.the_dynamic_visualisation_the_scpn_torus": {
        "context_id": "the_dynamic_visualisation_the_scpn_torus",
        "validation_protocol": "paper0.the_locus_of_the_interaction.the_dynamic_visualisation_the_scpn_torus",
        "canonical_statement": "The source-bounded component 'The Dynamic Visualisation: The SCPN Torus' preserves Paper 0 records P0R02555-P0R02556 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02555:the_dynamic_visualisation_the_scpn_torus",
            "P0R02556:the_dynamic_visualisation_the_scpn_torus",
        ),
        "source_formulae": (
            "P0R02555: The Dynamic Visualisation: The SCPN Torus",
            "P0R02556: The SCPN architecture is best visualised not merely as a static hierarchy, but as a dynamic, self-referential Torus. This geometry (T2=S1xS1) captures the closed-loop causality, the flow of information, and the central dynamic principles.",
        ),
        "test_protocols": (
            "preserve The Dynamic Visualisation: The SCPN Torus source-accounting boundary",
        ),
        "null_results": (
            "The Dynamic Visualisation: The SCPN Torus is not empirical validation evidence",
        ),
        "variables": ("the_dynamic_visualisation_the_scpn_torus",),
        "validation_targets": ("preserve records P0R02555-P0R02556",),
        "null_controls": (
            "the_dynamic_visualisation_the_scpn_torus must remain source-bounded accounting",
        ),
    },
    "the_locus_of_the_interaction.conceptual_specification_of_the_scpn_torus": {
        "context_id": "conceptual_specification_of_the_scpn_torus",
        "validation_protocol": "paper0.the_locus_of_the_interaction.conceptual_specification_of_the_scpn_torus",
        "canonical_statement": "The source-bounded component 'Conceptual Specification of the SCPN Torus:' preserves Paper 0 records P0R02557-P0R02557 without empirical validation claims.",
        "source_equation_ids": ("P0R02557:conceptual_specification_of_the_scpn_torus",),
        "source_formulae": ("P0R02557: Conceptual Specification of the SCPN Torus:",),
        "test_protocols": (
            "preserve Conceptual Specification of the SCPN Torus: source-accounting boundary",
        ),
        "null_results": (
            "Conceptual Specification of the SCPN Torus: is not empirical validation evidence",
        ),
        "variables": ("conceptual_specification_of_the_scpn_torus",),
        "validation_targets": ("preserve records P0R02557-P0R02557",),
        "null_controls": (
            "conceptual_specification_of_the_scpn_torus must remain source-bounded accounting",
        ),
    },
    "the_locus_of_the_interaction.1_the_geometry_and_flow": {
        "context_id": "1_the_geometry_and_flow",
        "validation_protocol": "paper0.the_locus_of_the_interaction.1_the_geometry_and_flow",
        "canonical_statement": "The source-bounded component '1. The Geometry and Flow:' preserves Paper 0 records P0R02558-P0R02565 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02558:1_the_geometry_and_flow",
            "P0R02559:1_the_geometry_and_flow",
            "P0R02560:1_the_geometry_and_flow",
            "P0R02561:1_the_geometry_and_flow",
            "P0R02562:1_the_geometry_and_flow",
            "P0R02563:1_the_geometry_and_flow",
            "P0R02564:1_the_geometry_and_flow",
            "P0R02565:1_the_geometry_and_flow",
        ),
        "source_formulae": (
            "P0R02558: 1. The Geometry and Flow:",
            "P0R02559: Poloidal Flow (The Hierarchy - HPC Loop): The flow around the short axis represents Hierarchical Predictive Coding (HPC). Downward (Outer Surface): The Generative Model (Projection) flows from L15 down to L1. | Upward (Inner Surface): The Inference/Filtering (Prediction Error) flows upward from L1 back to L15. | Toroidal Flow (The Dynamics - UPDE Spine): The flow around the long axis represents the temporal evolution and the Unified Phase Dynamics (UPDE), ensuring system-wide synchronisation.",
            "P0R02560: Torus Holonomy & Berry-Phase Memory",
            "P0R02561: Holonomy on T2T^2T2 and Memory Loops",
            "P0R02562: Let AL\\mathcal{A}_LAL be the UPDE-induced connection on the phase fibre. For poloidal (p)(\\gamma_p)(p) and toroidal (t)(\\gamma_t)(t) cycles,",
            "P0R02563: $\\Gamma p\\text{/}t = \\oint_{}^{}{\\gamma p\\text{/}t}\\, AL\\ \\mspace{2mu} = \\ \\mspace{2mu} Berry\\ phase\\ of\\ layer\\ L.\\Gamma_{p\\text{/}t} = \\oint_{\\gamma_{p\\text{/}t}}^{}{\\text{!}\\mathcal{A}_{\\mathcal{L}}}\\ = \\ \\text{Berry phase of layer }L.\\Gamma p\\text{/}t = \\oint_{}^{}{\\gamma p\\text{/}t AL} = Berry\\ $",
            "P0R02564: phase of layer L.",
            "P0R02565: Non-zero p/t\\Gamma_{p/t}p/t encodes persistent path-dependence (memory) for HPC projections (poloidal) and synchrony (toroidal). Holonomy mismatch pt|\\Gamma_p-\\Gamma_t|pt signals decoherence pressure; MS-QEC must compensate by increasing redundancy on the torus surface until pt|\\Gamma_p-\\Gamma_t|pt falls under the firewall threshold.",
        ),
        "test_protocols": ("preserve 1. The Geometry and Flow: source-accounting boundary",),
        "null_results": ("1. The Geometry and Flow: is not empirical validation evidence",),
        "variables": ("1_the_geometry_and_flow",),
        "validation_targets": ("preserve records P0R02558-P0R02565",),
        "null_controls": ("1_the_geometry_and_flow must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheLocusOfTheInteractionSpec:
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
class TheLocusOfTheInteractionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheLocusOfTheInteractionSpec, ...]
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


def build_the_locus_of_the_interaction_specs(
    source_records: list[dict[str, Any]],
) -> TheLocusOfTheInteractionSpecBundle:
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

    specs: list[TheLocusOfTheInteractionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheLocusOfTheInteractionSpec(
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
        "title": "Paper 0 " + "The Locus of the Interaction:" + " Specs",
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
        "next_source_boundary": "P0R02566",
    }
    return TheLocusOfTheInteractionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheLocusOfTheInteractionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_locus_of_the_interaction_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheLocusOfTheInteractionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Locus of the Interaction:" + " Specs",
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
    bundle: TheLocusOfTheInteractionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_the_locus_of_the_interaction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_the_locus_of_the_interaction_validation_specs_{date_tag}.md"
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
