#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2.7 The Fisher Info Metric: The Geometry of Interaction spec builder
"""Promote Paper 0 2.7 The Fisher Info Metric: The Geometry of Interaction records."""

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
    "P0R01993",
    "P0R01994",
    "P0R01995",
    "P0R01996",
    "P0R01997",
    "P0R01998",
    "P0R01999",
    "P0R02000",
    "P0R02001",
    "P0R02002",
    "P0R02003",
    "P0R02004",
    "P0R02005",
    "P0R02006",
    "P0R02007",
    "P0R02008",
    "P0R02009",
    "P0R02010",
)
CLAIM_BOUNDARY = "source-bounded section 2 7 the fisher info metric the geometry of interaction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_7_the_fisher_info_metric_the_geometry_of_interaction.2_7_the_fisher_info_metric_the_geometry_of_interaction": {
        "context_id": "2_7_the_fisher_info_metric_the_geometry_of_interaction",
        "validation_protocol": "paper0.section_2_7_the_fisher_info_metric_the_geometry_of_interaction.2_7_the_fisher_info_metric_the_geometry_of_interaction",
        "canonical_statement": "The source-bounded component '2.7 The Fisher Info Metric: The Geometry of Interaction' preserves Paper 0 records P0R01993-P0R02010 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01993:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01994:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01995:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01996:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01997:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01998:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R01999:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02000:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02001:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02002:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02003:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02004:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02005:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02006:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02007:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02008:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02009:2_7_the_fisher_info_metric_the_geometry_of_interaction",
            "P0R02010:2_7_the_fisher_info_metric_the_geometry_of_interaction",
        ),
        "source_formulae": (
            "P0R01993: 2.7 The Fisher Info Metric: The Geometry of Interaction",
            "P0R01994: Metric Regularization Law (Revision 11.45):",
            "P0R01995: To prevent informational-gauge divergence where det(g~F)->0,",
            "P0R01996: we introduce the Regularized Pullback Metric (greg):",
            "P0R01997: g_reg_munu = g_fim_munu + (epsilon * eta_munu) * exp(-S_path / k_b)",
            "P0R01998: Legend of Equation Components:",
            'P0R01999: g_reg_munu: The physically realizable information-geometric metric. | epsilon: A non-zero regularization constant (The "FIM-Floor"). | eta_munu: The background Minkowski metric. | S_path: The current causal path entropy of the system.',
            'P0R02000: Implementation: This ensures that even in states of near-zero information density, the "Infoton" maintains a finite propagation speed, preventing unphysical longitudinal modes.',
            "P0R02001: P0R02001",
            "P0R02002: 2.7.1 The Problem: Defining the Geometry of the Infoton's Dynamics",
            "P0R02003: Source Material: This section will synthesise the scattered statements highlighting the need for a non-Minkowskian geometry to govern the infoton, as its interactions are informational.",
            "P0R02004: P0R02004",
            "P0R02005: 2.7.2 Proposing the Fisher Information Metric (FIM) as the Solution",
            "P0R02006: Source Material: The core proposal will be constructed here, unifying the arguments for why the FIM, as the natural metric on the space of probability distributions, is the only logical choice for a force that communicates information about system states.",
            "P0R02007: P0R02007",
            'P0R02008: 2.7.3 The "Operational Pullback Protocol": Connecting Theory to Measurement',
            "P0R02009: Source Material: This critical section will formalise the protocol for deriving the FIM from the observable parameters (theta) of a given physical system, making the theory experimentally tractable and providing a concrete mathematical procedure.",
            "P0R02010: P0R02010",
        ),
        "test_protocols": (
            "preserve 2.7 The Fisher Info Metric: The Geometry of Interaction source-accounting boundary",
        ),
        "null_results": (
            "2.7 The Fisher Info Metric: The Geometry of Interaction is not empirical validation evidence",
        ),
        "variables": ("2_7_the_fisher_info_metric_the_geometry_of_interaction",),
        "validation_targets": ("preserve records P0R01993-P0R02010",),
        "null_controls": (
            "2_7_the_fisher_info_metric_the_geometry_of_interaction must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section27TheFisherInfoMetricTheGeometryOfInteractionSpec:
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
class Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section27TheFisherInfoMetricTheGeometryOfInteractionSpec, ...]
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


def build_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_specs(
    source_records: list[dict[str, Any]],
) -> Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle:
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

    specs: list[Section27TheFisherInfoMetricTheGeometryOfInteractionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section27TheFisherInfoMetricTheGeometryOfInteractionSpec(
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
        "title": "Paper 0 " + "2.7 The Fisher Info Metric: The Geometry of Interaction" + " Specs",
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
        "next_source_boundary": "P0R02011",
    }
    return Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2.7 The Fisher Info Metric: The Geometry of Interaction" + " Specs",
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
    bundle: Section27TheFisherInfoMetricTheGeometryOfInteractionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation_specs_{date_tag}.md"
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
