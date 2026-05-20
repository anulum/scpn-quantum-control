#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Amplification Cascade: spec builder
"""Promote Paper 0 The Amplification Cascade: records."""

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
    "P0R05323",
    "P0R05324",
    "P0R05325",
    "P0R05326",
    "P0R05327",
    "P0R05328",
    "P0R05329",
    "P0R05330",
)
CLAIM_BOUNDARY = (
    "source-bounded the amplification cascade source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_amplification_cascade.the_amplification_cascade": {
        "context_id": "the_amplification_cascade",
        "validation_protocol": "paper0.the_amplification_cascade.the_amplification_cascade",
        "canonical_statement": "The source-bounded component 'The Amplification Cascade:' preserves Paper 0 records P0R05323-P0R05324 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05323:the_amplification_cascade",
            "P0R05324:the_amplification_cascade",
        ),
        "source_formulae": (
            "P0R05323: The Amplification Cascade:",
            "P0R05324: L1/L2 Modulation: The Psi-field subtly modulates the energy landscape (Q) of critical molecular components (e.g., ion channels). | QSR Enhancement: QSR enhances the sensitivity of these components to the weak Psi-field signal amidst the inherent noise of the substrate. | Cellular Amplification: The neuron, poised at its firing threshold (cellular criticality), is highly sensitive to these small modulations, which can trigger an action potential. | Network Cascade (L4): In the Quasicritical L4 network (sigma1), this single event can trigger a neuronal avalanche, rapidly amplifying the localised quantum effect into a large-scale change in network dynamics.",
        ),
        "test_protocols": ("preserve The Amplification Cascade: source-accounting boundary",),
        "null_results": ("The Amplification Cascade: is not empirical validation evidence",),
        "variables": ("the_amplification_cascade",),
        "validation_targets": ("preserve records P0R05323-P0R05324",),
        "null_controls": ("the_amplification_cascade must remain source-bounded accounting",),
    },
    "the_amplification_cascade.the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige": {
        "context_id": "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
        "validation_protocol": "paper0.the_amplification_cascade.the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
        "canonical_statement": "The source-bounded component 'The Morphogenetic Blueprint (Layers 3-4): Here, we explore the Genomic-Epigenomic-Morphogenetic (Layer 3) layer, framing DNA as a resonant fractal antenna that reads and writes information via CISS and bioelectric codes. This blueprint guides the formation of the organism, whose tissues then learn to oscillate in unison in the Cellular-Tissue Synchronisation (Layer 4) layer, where coherent rhythms emerge from networks operating at quasicriticality.' preserves Paper 0 records P0R05325-P0R05330 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05325:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
            "P0R05326:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
            "P0R05327:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
            "P0R05328:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
            "P0R05329:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
            "P0R05330:the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
        ),
        "source_formulae": (
            "P0R05325: The Morphogenetic Blueprint (Layers 3-4): Here, we explore the Genomic-Epigenomic-Morphogenetic (Layer 3) layer, framing DNA as a resonant fractal antenna that reads and writes information via CISS and bioelectric codes. This blueprint guides the formation of the organism, whose tissues then learn to oscillate in unison in the Cellular-Tissue Synchronisation (Layer 4) layer, where coherent rhythms emerge from networks operating at quasicriticality.",
            "P0R05326: [IMAGE:]",
            "P0R05327: Fig.: The Signature of Criticality",
            "P0R05328: Blue power-law distribution of avalanche sizes -> straight line on log-log scales. | Annotated with tau 1.5 and equation P(S) S^tau. | Red dashed curve showing exponential decay as the Subcritical/Chaotic Regime. | Both axes set to logarithmic scaling with scientific clarity.",
            "P0R05329: [IMAGE:]",
            'P0R05330: DNA double helix (blue + red strands). | Electrons travelling along the sugar-phosphate backbone with spin filtering. | Spin-Up (green) preferred, Spin-Down (orange) suppressed. | Magnetic field loop around DNA, annotated as "Resonant Antenna for Psi-Field Coupling."',
        ),
        "test_protocols": (
            "preserve The Morphogenetic Blueprint (Layers 3-4): Here, we explore the Genomic-Epigenomic-Morphogenetic (Layer 3) layer, framing DNA as a resonant fractal antenna that reads and writes information via CISS and bioelectric codes. This blueprint guides the formation of the organism, whose tissues then learn to oscillate in unison in the Cellular-Tissue Synchronisation (Layer 4) layer, where coherent rhythms emerge from networks operating at quasicriticality. source-accounting boundary",
        ),
        "null_results": (
            "The Morphogenetic Blueprint (Layers 3-4): Here, we explore the Genomic-Epigenomic-Morphogenetic (Layer 3) layer, framing DNA as a resonant fractal antenna that reads and writes information via CISS and bioelectric codes. This blueprint guides the formation of the organism, whose tissues then learn to oscillate in unison in the Cellular-Tissue Synchronisation (Layer 4) layer, where coherent rhythms emerge from networks operating at quasicriticality. is not empirical validation evidence",
        ),
        "variables": ("the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",),
        "validation_targets": ("preserve records P0R05325-P0R05330",),
        "null_controls": (
            "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheAmplificationCascadeSpec:
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
class TheAmplificationCascadeSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheAmplificationCascadeSpec, ...]
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


def build_the_amplification_cascade_specs(
    source_records: list[dict[str, Any]],
) -> TheAmplificationCascadeSpecBundle:
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

    specs: list[TheAmplificationCascadeSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheAmplificationCascadeSpec(
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
        "title": "Paper 0 " + "The Amplification Cascade:" + " Specs",
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
        "next_source_boundary": "P0R05331",
    }
    return TheAmplificationCascadeSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheAmplificationCascadeSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_amplification_cascade_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheAmplificationCascadeSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Amplification Cascade:" + " Specs",
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
    bundle: TheAmplificationCascadeSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_the_amplification_cascade_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_the_amplification_cascade_validation_specs_{date_tag}.md"
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
