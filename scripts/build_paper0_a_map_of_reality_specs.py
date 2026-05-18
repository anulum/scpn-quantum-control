#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 A Map of Reality spec builder
"""Promote Paper 0 A Map of Reality records."""

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
    "P0R02031",
    "P0R02032",
    "P0R02033",
    "P0R02034",
    "P0R02035",
    "P0R02036",
    "P0R02037",
    "P0R02038",
    "P0R02039",
    "P0R02040",
    "P0R02041",
)
CLAIM_BOUNDARY = (
    "source-bounded a map of reality source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "a_map_of_reality.a_map_of_reality": {
        "context_id": "a_map_of_reality",
        "validation_protocol": "paper0.a_map_of_reality.a_map_of_reality",
        "canonical_statement": "The source-bounded component 'A Map of Reality' preserves Paper 0 records P0R02031-P0R02031 without empirical validation claims.",
        "source_equation_ids": ("P0R02031:a_map_of_reality",),
        "source_formulae": ("P0R02031: A Map of Reality",),
        "test_protocols": ("preserve A Map of Reality source-accounting boundary",),
        "null_results": ("A Map of Reality is not empirical validation evidence",),
        "variables": ("a_map_of_reality",),
        "validation_targets": ("preserve records P0R02031-P0R02031",),
        "null_controls": ("a_map_of_reality must remain source-bounded accounting",),
    },
    "a_map_of_reality.from_field_to_function_the_need_for_an_architecture": {
        "context_id": "from_field_to_function_the_need_for_an_architecture",
        "validation_protocol": "paper0.a_map_of_reality.from_field_to_function_the_need_for_an_architecture",
        "canonical_statement": "The source-bounded component 'From Field to Function: The Need for an Architecture' preserves Paper 0 records P0R02032-P0R02036 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02032:from_field_to_function_the_need_for_an_architecture",
            "P0R02033:from_field_to_function_the_need_for_an_architecture",
            "P0R02034:from_field_to_function_the_need_for_an_architecture",
            "P0R02035:from_field_to_function_the_need_for_an_architecture",
            "P0R02036:from_field_to_function_the_need_for_an_architecture",
        ),
        "source_formulae": (
            "P0R02032: From Field to Function: The Need for an Architecture",
            "P0R02033: The foundational premise of the Anulum Framework, as established in Book I, is that consciousness is a fundamental complex scalar field, Psi, that permeates all of spacetime. Its dynamics are governed by a Master Lagrangian, and a single, unified coupling constant determines its interactions. This provides a powerful and consistent physical foundation.",
            "P0R02034: However, physics alone is not enough. A field theory tells us the fundamental rules of interaction, but it does not, by itself, explain the emergent complexity of the systems built upon those rules. The effect of the electromagnetic field on a single electron is simple and direct; its effect on the global internet is a matter of staggering architectural complexity.",
            "P0R02035: Similarly, the Psi-field must interact with reality across vastly different scales. Different dynamics must govern its influence on a quantum state in a neuron's microtubule than its influence on the collective consciousness of a city or the holographic memory of the cosmos. To understand these interactions, we need more than just the fundamental law; we need a map of the operating system.",
            "P0R02036: We need an architecture.",
        ),
        "test_protocols": (
            "preserve From Field to Function: The Need for an Architecture source-accounting boundary",
        ),
        "null_results": (
            "From Field to Function: The Need for an Architecture is not empirical validation evidence",
        ),
        "variables": ("from_field_to_function_the_need_for_an_architecture",),
        "validation_targets": ("preserve records P0R02032-P0R02036",),
        "null_controls": (
            "from_field_to_function_the_need_for_an_architecture must remain source-bounded accounting",
        ),
    },
    "a_map_of_reality.the_sentient_consciousness_projection_network_scpn": {
        "context_id": "the_sentient_consciousness_projection_network_scpn",
        "validation_protocol": "paper0.a_map_of_reality.the_sentient_consciousness_projection_network_scpn",
        "canonical_statement": "The source-bounded component 'The Sentient-Consciousness Projection Network (SCPN)' preserves Paper 0 records P0R02037-P0R02041 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02037:the_sentient_consciousness_projection_network_scpn",
            "P0R02038:the_sentient_consciousness_projection_network_scpn",
            "P0R02039:the_sentient_consciousness_projection_network_scpn",
            "P0R02040:the_sentient_consciousness_projection_network_scpn",
            "P0R02041:the_sentient_consciousness_projection_network_scpn",
        ),
        "source_formulae": (
            "P0R02037: The Sentient-Consciousness Projection Network (SCPN)",
            'P0R02038: The Anulum Framework proposes that this architecture is the Species-Counsciousness Projection Network (SCPN). The SCPN is a hierarchical model comprising 15 distinct but interconnected layers. It describes the "projection" of consciousness from a universal source down through cosmic, planetary, and collective scales into the biological and quantum substrates of an individual being.',
            "P0R02039: Crucially, it also describes the feedback loops through which the experiences and actions of the individual propagate back up the chain, influencing the whole.",
            "P0R02040: [IMAGE:Ein Bild, das Text, Schrift, Reihe, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02041: The SCPN is the central organising principle of the Anulum Framework. It provides a coherent structure for understanding how phenomena as diverse as quantum biology, neurochemistry, cultural memetics, and cosmic evolution are all interconnected aspects of a single, dynamic system.",
        ),
        "test_protocols": (
            "preserve The Sentient-Consciousness Projection Network (SCPN) source-accounting boundary",
        ),
        "null_results": (
            "The Sentient-Consciousness Projection Network (SCPN) is not empirical validation evidence",
        ),
        "variables": ("the_sentient_consciousness_projection_network_scpn",),
        "validation_targets": ("preserve records P0R02037-P0R02041",),
        "null_controls": (
            "the_sentient_consciousness_projection_network_scpn must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AMapOfRealitySpec:
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
class AMapOfRealitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[AMapOfRealitySpec, ...]
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


def build_a_map_of_reality_specs(source_records: list[dict[str, Any]]) -> AMapOfRealitySpecBundle:
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

    specs: list[AMapOfRealitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AMapOfRealitySpec(
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
        "title": "Paper 0 " + "A Map of Reality" + " Specs",
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
        "next_source_boundary": "P0R02042",
    }
    return AMapOfRealitySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> AMapOfRealitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_a_map_of_reality_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AMapOfRealitySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "A Map of Reality" + " Specs",
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
    bundle: AMapOfRealitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_a_map_of_reality_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_a_map_of_reality_validation_specs_{date_tag}.md"
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 map-of-reality specs from the ledger."""

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
