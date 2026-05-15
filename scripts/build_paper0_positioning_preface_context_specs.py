#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Positioning Preface context spec builder
"""Promote Paper 0 Positioning Preface and Author's Note context records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(249, 268))
BLANK_SEPARATOR_IDS = ("P0R00260", "P0R00267")
CLAIM_BOUNDARY = "source-bounded Positioning Preface context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "positioning_preface_context.book_ii_threshold": {
        "context_id": "book_ii_threshold",
        "validation_protocol": "paper0.positioning_preface_context.book_ii_threshold",
        "canonical_statement": (
            "The Positioning Preface opens Book II as a threshold text for the SCPN."
        ),
        "source_equation_ids": (
            "P0R00249:positioning_preface_title",
            "P0R00250:book_ii_subtitle",
            "P0R00251:humanity_threshold_positioning",
        ),
        "source_formulae": (
            "Positioning Preface",
            "Book II - The Sentient-Consciousness Projection Network",
            "Humanity stands at a threshold",
        ),
        "test_protocols": ("preserve Book II threshold positioning",),
        "null_results": ("threshold preface text is not empirical evidence",),
        "variables": ("Book_II", "SCPN", "threshold"),
        "validation_targets": (
            "preserve Positioning Preface boundary",
            "preserve Book II subtitle",
            "reject preface as validation evidence",
        ),
        "null_controls": (
            "preface-as-evidence control must be rejected",
            "missing-Book-II-subtitle control must be rejected",
        ),
    },
    "positioning_preface_context.discipline_positioning": {
        "context_id": "discipline_positioning",
        "validation_protocol": "paper0.positioning_preface_context.discipline_positioning",
        "canonical_statement": (
            "The preface positions consciousness as structural, then names Field Architecture, "
            "Consciousness Engineering, and Noetic Field Theory with a rigour boundary."
        ),
        "source_equation_ids": ("P0R00251-P0R00255:discipline_positioning_claims",),
        "source_formulae": (
            "structural principle of reality",
            "Field Architecture",
            "Consciousness Engineering",
            "Noetic Field Theory",
            "explicit equations, testable couplings",
        ),
        "test_protocols": ("preserve discipline-positioning claims as hypotheses",),
        "null_results": ("discipline framing is not a measured coupling",),
        "variables": ("Field_Architecture", "Consciousness_Engineering", "Noetic_Field_Theory"),
        "validation_targets": (
            "preserve structural-consciousness positioning",
            "preserve Field Architecture and Consciousness Engineering distinction",
            "preserve rigour and testability boundary",
        ),
        "null_controls": (
            "discipline-framing-as-proof control must be rejected",
            "missing-testability-boundary control must be rejected",
        ),
    },
    "positioning_preface_context.architecture_manual_boundary": {
        "context_id": "architecture_manual_boundary",
        "validation_protocol": "paper0.positioning_preface_context.architecture_manual_boundary",
        "canonical_statement": (
            "The preface frames the work as an architecture manual with projection layers, "
            "VIBRANA, symbolic operators, and self-reference/closure warning."
        ),
        "source_equation_ids": (
            "P0R00256:architecture_manual_scope",
            "P0R00257-P0R00258:foundation_stone_and_practical_horizon",
            "P0R00259:self_reference_closure_warning",
        ),
        "source_formulae": (
            "projection layers",
            "VIBRANA",
            "foundation stone",
            "infinite regress",
            "self-reference",
        ),
        "test_protocols": ("preserve architecture-manual and closure-warning context",),
        "null_results": ("architecture-manual framing is not an implementation protocol",),
        "variables": ("VIBRANA", "self_reference", "closure"),
        "validation_targets": (
            "preserve projection-layer scope",
            "preserve VIBRANA operator framing",
            "preserve self-reference closure warning",
        ),
        "null_controls": (
            "manual-framing-as-protocol control must be rejected",
            "missing-closure-warning control must be rejected",
        ),
    },
    "positioning_preface_context.dual_register_author_note": {
        "context_id": "dual_register_author_note",
        "validation_protocol": "paper0.positioning_preface_context.dual_register_author_note",
        "canonical_statement": (
            "The Author's Note states that the manuscript operates in academic and visionary registers."
        ),
        "source_equation_ids": (
            "P0R00261:authors_note_title",
            "P0R00262:two_frequencies_statement",
            "P0R00263:academic_register",
            "P0R00264:visionary_register",
            "P0R00265:dual_voice_boundary",
        ),
        "source_formulae": (
            "two frequencies",
            "academic register",
            "visionary register",
            "foundational architecture manual",
        ),
        "test_protocols": ("preserve dual-register note as interpretive boundary",),
        "null_results": ("dual-register framing is not validation evidence",),
        "variables": ("academic_register", "visionary_register"),
        "validation_targets": (
            "preserve two-frequency framing",
            "preserve academic register boundary",
            "preserve visionary register boundary",
        ),
        "null_controls": (
            "authors-note-as-evidence control must be rejected",
            "missing-dual-register-boundary control must be rejected",
        ),
    },
    "positioning_preface_context.separator_and_part_boundary": {
        "context_id": "separator_and_part_boundary",
        "validation_protocol": "paper0.positioning_preface_context.separator_and_part_boundary",
        "canonical_statement": (
            "Blank separators and one image marker are counted, and Part I starts at P0R00268."
        ),
        "source_equation_ids": (
            "P0R00260:blank_separator",
            "P0R00266:image_marker",
            "P0R00267:blank_separator",
        ),
        "source_formulae": (
            "2 blank separator records",
            "1 image marker",
            "Part I boundary P0R00268",
        ),
        "test_protocols": ("preserve source accounting and next-section boundary",),
        "null_results": ("image and blank records are not validation evidence",),
        "variables": ("blank_separator_count", "image_marker_count", "part_i_boundary"),
        "validation_targets": (
            "preserve blank separator count",
            "preserve image marker count",
            "preserve Part I boundary",
        ),
        "null_controls": (
            "blank-skip control must be rejected",
            "missing-Part-I-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PositioningPrefaceContextSpec:
    """Positioning Preface context spec promoted from Paper 0 records."""

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
class PositioningPrefaceContextSpecBundle:
    """Positioning Preface context specs plus source coverage summary."""

    specs: tuple[PositioningPrefaceContextSpec, ...]
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


def build_positioning_preface_context_specs(
    source_records: list[dict[str, Any]],
) -> PositioningPrefaceContextSpecBundle:
    """Build source-covered Positioning Preface context specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    blank_separator_count = sum(
        1 for ledger_id in BLANK_SEPARATOR_IDS if not str(records_by_ledger[ledger_id]["text"])
    )
    image_marker_count = sum(1 for record in anchors if str(record["text"]).startswith("[IMAGE:"))
    specs: list[PositioningPrefaceContextSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PositioningPrefaceContextSpec(
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
        "title": "Paper 0 Positioning Preface Context Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": blank_separator_count,
        "image_marker_count": image_marker_count,
        "part_i_boundary": "P0R00268",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return PositioningPrefaceContextSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> PositioningPrefaceContextSpecBundle:
    """Build Positioning Preface context specs from the canonical ledger."""
    return build_positioning_preface_context_specs(load_jsonl(path))


def write_outputs(
    bundle: PositioningPrefaceContextSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Positioning Preface context spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_positioning_preface_context_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_positioning_preface_context_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: PositioningPrefaceContextSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Positioning Preface Context Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Blank separators: {bundle.summary['blank_separator_count']}",
        f"- Image markers: {bundle.summary['image_marker_count']}",
        f"- Part I boundary: {bundle.summary['part_i_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 Positioning Preface context validation specs."""
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
