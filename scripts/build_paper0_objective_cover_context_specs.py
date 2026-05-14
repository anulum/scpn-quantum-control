#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 objective cover context spec builder
"""Promote Paper 0 objective, cover, and collection context records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(218, 249))
CLAIM_BOUNDARY = "source-bounded objective and cover context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "objective_cover_context.objective_statement": {
        "context_id": "objective_statement",
        "validation_protocol": "paper0.objective_cover_context.objective_statement",
        "canonical_statement": (
            "The cover objective maps multi-scale consciousness/biology interaction into "
            "a self-organising, self-optimising, and self-correcting universe model."
        ),
        "source_equation_ids": (
            "P0R00218:objective_marker",
            "P0R00219:multi_scale_consciousness_biology_mapping",
            "P0R00220:self_organising_self_correcting_universe_model",
        ),
        "source_formulae": (
            "Multi-Scale Interaction of Consciousness and Biology",
            "self-organising, self-optimising, and self-correcting universe",
        ),
        "test_protocols": ("preserve objective as positioning context",),
        "null_results": ("objective text is not validation evidence",),
        "variables": ("multi_scale_interaction", "consciousness", "biology"),
        "validation_targets": (
            "preserve objective text",
            "preserve self-correcting universe positioning",
            "reject objective as empirical evidence",
        ),
        "null_controls": (
            "objective-as-evidence control must be rejected",
            "missing-objective control must be rejected",
        ),
    },
    "objective_cover_context.cover_identity": {
        "context_id": "cover_identity",
        "validation_protocol": "paper0.objective_cover_context.cover_identity",
        "canonical_statement": (
            "The cover identifies the project title, architecture framing, discipline framing, "
            "frontier label, and authorship context."
        ),
        "source_equation_ids": (
            "P0R00221:project_title",
            "P0R00222:architecture_title",
            "P0R00223:discipline_positioning",
            "P0R00224:frontier_label",
            "P0R00225:authorship_context",
        ),
        "source_formulae": (
            "God of the Math",
            "The Architecture of Being",
            "Consciousness Engineering / Field Architecture",
        ),
        "test_protocols": ("preserve title and discipline-positioning context",),
        "null_results": ("cover identity is not a scientific result",),
        "variables": ("Field_Architecture", "Consciousness_Engineering"),
        "validation_targets": (
            "preserve title context",
            "preserve discipline framing",
            "preserve authorship context",
        ),
        "null_controls": (
            "cover-title-as-proof control must be rejected",
            "missing-discipline-framing control must be rejected",
        ),
    },
    "objective_cover_context.image_and_quote_context": {
        "context_id": "image_and_quote_context",
        "validation_protocol": "paper0.objective_cover_context.image_and_quote_context",
        "canonical_statement": (
            "Image markers and the Korzybski quote are preserved as non-computational context."
        ),
        "source_equation_ids": (
            "P0R00226-P0R00227:image_markers",
            "P0R00228-P0R00229:korzybski_quote",
            "P0R00231:image_marker",
        ),
        "source_formulae": ("map is not the territory", "3 image marker records"),
        "test_protocols": ("count image markers without interpreting them as evidence",),
        "null_results": ("image descriptions are not machine-checked figures",),
        "variables": ("image_marker_count", "quote_context"),
        "validation_targets": (
            "preserve image marker count",
            "preserve quote attribution context",
            "reject image markers as data",
        ),
        "null_controls": (
            "image-marker-as-data control must be rejected",
            "missing-image-marker control must be rejected",
        ),
    },
    "objective_cover_context.cyclic_operator_positioning": {
        "context_id": "cyclic_operator_positioning",
        "validation_protocol": "paper0.objective_cover_context.cyclic_operator_positioning",
        "canonical_statement": (
            "The cover positions SCPN as an active-inference architecture with a recursive "
            "participatory loop and Meta Metatron Cycle/cyclic-operator claim boundary."
        ),
        "source_equation_ids": (
            "P0R00230:active_inference_engine_positioning",
            "P0R00232-P0R00235:participatory_recursive_loop_positioning",
            "P0R00236:meta_metatron_cycle_positioning",
        ),
        "source_formulae": (
            "cosmic-scale active inference engine",
            "recursive loop",
            "Meta Metatron Cycle",
            "cyclic operator",
        ),
        "test_protocols": ("preserve cyclic-operator positioning as later validation target",),
        "null_results": ("cover positioning is not a cyclic-operator derivation",),
        "variables": ("active_inference_engine", "Meta_Metatron_Cycle", "cyclic_operator"),
        "validation_targets": (
            "preserve active-inference positioning",
            "preserve recursive-loop positioning",
            "preserve cyclic-operator boundary",
        ),
        "null_controls": (
            "cover-as-cyclic-operator-proof control must be rejected",
            "missing-recursive-loop-positioning control must be rejected",
        ),
    },
    "objective_cover_context.book_ii_collection_identity": {
        "context_id": "book_ii_collection_identity",
        "validation_protocol": "paper0.objective_cover_context.book_ii_collection_identity",
        "canonical_statement": (
            "The cover repeats Book II identity and the five-book collection order before "
            "the Positioning Preface starts at P0R00249."
        ),
        "source_equation_ids": (
            "P0R00238:psi_marker",
            "P0R00239:book_ii_interdisciplinary_purpose",
            "P0R00240-P0R00242:book_ii_title",
            "P0R00243-P0R00248:collection_order",
        ),
        "source_formulae": (
            "Book II",
            "The Sentient-Consciousness Projection Network",
            "An Architecture for Reality",
            "five-book collection",
        ),
        "test_protocols": ("preserve Book II identity and next-section boundary",),
        "null_results": ("book identity is not validation evidence",),
        "variables": ("Book_II", "SCPN", "collection_order"),
        "validation_targets": (
            "preserve Book II identity",
            "preserve five-book collection order",
            "preserve Positioning Preface boundary",
        ),
        "null_controls": (
            "book-title-as-evidence control must be rejected",
            "missing-positioning-preface-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ObjectiveCoverContextSpec:
    """Objective and cover context spec promoted from Paper 0 records."""

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
class ObjectiveCoverContextSpecBundle:
    """Objective and cover context specs plus source coverage summary."""

    specs: tuple[ObjectiveCoverContextSpec, ...]
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


def build_objective_cover_context_specs(
    source_records: list[dict[str, Any]],
) -> ObjectiveCoverContextSpecBundle:
    """Build source-covered objective and cover context specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    image_marker_count = sum(1 for record in anchors if str(record["text"]).startswith("[IMAGE:"))
    specs: list[ObjectiveCoverContextSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ObjectiveCoverContextSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="source_context_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Objective Cover Context Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "image_marker_count": image_marker_count,
        "collection_book_count": 5,
        "positioning_preface_boundary": "P0R00249",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return ObjectiveCoverContextSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> ObjectiveCoverContextSpecBundle:
    """Build objective cover context specs from the canonical ledger."""
    return build_objective_cover_context_specs(load_jsonl(path))


def write_outputs(
    bundle: ObjectiveCoverContextSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown objective cover context spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_objective_cover_context_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_objective_cover_context_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: ObjectiveCoverContextSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Objective Cover Context Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Image markers: {bundle.summary['image_marker_count']}",
        f"- Collection books: {bundle.summary['collection_book_count']}",
        f"- Positioning Preface boundary: {bundle.summary['positioning_preface_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 objective cover context validation specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0 if bundle.summary["coverage_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
