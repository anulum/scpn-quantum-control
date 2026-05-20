#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 1. The Lipid Landscape and Criticality spec builder
"""Promote Paper 0 1. The Lipid Landscape and Criticality records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R04746",
    "P0R04747",
    "P0R04748",
    "P0R04749",
    "P0R04750",
    "P0R04751",
    "P0R04752",
    "P0R04753",
)
CLAIM_BOUNDARY = "source-bounded section 1 the lipid landscape and criticality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_1_the_lipid_landscape_and_criticality.1_the_lipid_landscape_and_criticality": {
        "context_id": "1_the_lipid_landscape_and_criticality",
        "validation_protocol": "paper0.section_1_the_lipid_landscape_and_criticality.1_the_lipid_landscape_and_criticality",
        "canonical_statement": "The source-bounded component '1. The Lipid Landscape and Criticality' preserves Paper 0 records P0R04746-P0R04747 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04746:1_the_lipid_landscape_and_criticality",
            "P0R04747:1_the_lipid_landscape_and_criticality",
        ),
        "source_formulae": (
            "P0R04746: 1. The Lipid Landscape and Criticality",
            "P0R04747: Neuronal membranes are enriched in Polyunsaturated Fatty Acids (PUFAs), particularly DHA. PUFAs increase membrane fluidity, facilitating the rapid conformational changes required for signalling. The membrane maintains a state analogous to Quasicriticality (sigma1), poised between ordered and disordered phases.",
        ),
        "test_protocols": (
            "preserve 1. The Lipid Landscape and Criticality source-accounting boundary",
        ),
        "null_results": (
            "1. The Lipid Landscape and Criticality is not empirical validation evidence",
        ),
        "variables": ("1_the_lipid_landscape_and_criticality",),
        "validation_targets": ("preserve records P0R04746-P0R04747",),
        "null_controls": (
            "1_the_lipid_landscape_and_criticality must remain source-bounded accounting",
        ),
    },
    "section_1_the_lipid_landscape_and_criticality.2_the_central_role_of_cholesterol": {
        "context_id": "2_the_central_role_of_cholesterol",
        "validation_protocol": "paper0.section_1_the_lipid_landscape_and_criticality.2_the_central_role_of_cholesterol",
        "canonical_statement": "The source-bounded component '2. The Central Role of Cholesterol' preserves Paper 0 records P0R04748-P0R04750 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04748:2_the_central_role_of_cholesterol",
            "P0R04749:2_the_central_role_of_cholesterol",
            "P0R04750:2_the_central_role_of_cholesterol",
        ),
        "source_formulae": (
            "P0R04748: 2. The Central Role of Cholesterol",
            "P0R04749: Cholesterol constitutes up to 25% of the CNS lipid content and acts as the master regulator of membrane dynamics.",
            "P0R04750: Fluidity Buffer and Criticality Maintenance: Cholesterol maintains the membrane in the optimal Liquid-Ordered (Lo) phase. It buffers against temperature fluctuations, ensuring the stability of the membrane's critical state. | The Dielectric Interface (L1/L2): Cholesterol significantly lowers the dielectric constant (r) of the membrane core. This impacts capacitance (Cm) and strengthens electrostatic interactions within the membrane, potentially facilitating stronger coupling for IET mechanisms targeting transmembrane proteins. | Synaptic Function (L2): Cholesterol is essential for the high membrane curvature required for synaptic vesicles and the efficiency of the SNARE complex formation during neurotransmitter release.",
        ),
        "test_protocols": (
            "preserve 2. The Central Role of Cholesterol source-accounting boundary",
        ),
        "null_results": (
            "2. The Central Role of Cholesterol is not empirical validation evidence",
        ),
        "variables": ("2_the_central_role_of_cholesterol",),
        "validation_targets": ("preserve records P0R04748-P0R04750",),
        "null_controls": (
            "2_the_central_role_of_cholesterol must remain source-bounded accounting",
        ),
    },
    "section_1_the_lipid_landscape_and_criticality.3_lipid_rafts_the_organising_platforms_for_iet": {
        "context_id": "3_lipid_rafts_the_organising_platforms_for_iet",
        "validation_protocol": "paper0.section_1_the_lipid_landscape_and_criticality.3_lipid_rafts_the_organising_platforms_for_iet",
        "canonical_statement": "The source-bounded component '3. Lipid Rafts: The Organising Platforms for IET' preserves Paper 0 records P0R04751-P0R04753 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04751:3_lipid_rafts_the_organising_platforms_for_iet",
            "P0R04752:3_lipid_rafts_the_organising_platforms_for_iet",
            "P0R04753:3_lipid_rafts_the_organising_platforms_for_iet",
        ),
        "source_formulae": (
            "P0R04751: 3. Lipid Rafts: The Organising Platforms for IET",
            "P0R04752: Lipid Rafts are specialised microdomains enriched in cholesterol and sphingolipids. They act as platforms that concentrate signalling machinery.",
            "P0R04753: Compartmentalisation of Signalling: Rafts cluster receptors, ion channels, and signalling complexes, enhancing the speed and specificity of L2 transduction. | Lipid Rafts as IET Hubs: The highly ordered structure of the lipid raft makes it a primary site for Psi-field modulation via the Quantum Potential (Q). LIET=gIETPsi(x)Q(Raft). Furthermore, the organised geometry maximises the local Fisher Information Metric (gmu), enhancing the coupling strength as defined by the Informational Coupling Lagrangian (LInformationalPsidet(gmu). | Anaesthesia and Consciousness: The disruption of lipid raft integrity is a proposed mechanism for general anaesthesia. Anaesthetics increase membrane disorder, disrupting L2 signalling and decoupling the Psi-field from the physical substrate, leading to L5 dissolution.",
        ),
        "test_protocols": (
            "preserve 3. Lipid Rafts: The Organising Platforms for IET source-accounting boundary",
        ),
        "null_results": (
            "3. Lipid Rafts: The Organising Platforms for IET is not empirical validation evidence",
        ),
        "variables": ("3_lipid_rafts_the_organising_platforms_for_iet",),
        "validation_targets": ("preserve records P0R04751-P0R04753",),
        "null_controls": (
            "3_lipid_rafts_the_organising_platforms_for_iet must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section1TheLipidLandscapeAndCriticalitySpec:
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
class Section1TheLipidLandscapeAndCriticalitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section1TheLipidLandscapeAndCriticalitySpec, ...]
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


def build_section_1_the_lipid_landscape_and_criticality_specs(
    source_records: list[dict[str, Any]],
) -> Section1TheLipidLandscapeAndCriticalitySpecBundle:
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

    specs: list[Section1TheLipidLandscapeAndCriticalitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section1TheLipidLandscapeAndCriticalitySpec(
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
        "title": "Paper 0 " + "1. The Lipid Landscape and Criticality" + " Specs",
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
        "next_source_boundary": "P0R04754",
    }
    return Section1TheLipidLandscapeAndCriticalitySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section1TheLipidLandscapeAndCriticalitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_1_the_lipid_landscape_and_criticality_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section1TheLipidLandscapeAndCriticalitySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "1. The Lipid Landscape and Criticality" + " Specs",
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
    bundle: Section1TheLipidLandscapeAndCriticalitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_1_the_lipid_landscape_and_criticality_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_1_the_lipid_landscape_and_criticality_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
