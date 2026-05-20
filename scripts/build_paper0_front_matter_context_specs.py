#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 front matter context spec builder
"""Promote Paper 0 front matter and fragmented ToC context records."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(18, 105))
BLANK_PLACEHOLDER_IDS = tuple(f"P0R{number:05d}" for number in range(59, 104))
CLAIM_BOUNDARY = "source-bounded front matter context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "front_matter_context.collection_identity": {
        "context_id": "collection_identity",
        "validation_protocol": "paper0.front_matter_context.collection_identity",
        "canonical_statement": (
            "The front matter identifies author, collection position, Book II location, "
            "and the hypothesis/falsifiability status of Paper 0."
        ),
        "source_equation_ids": (
            "P0R00018:author_identity",
            "P0R00019:collection_location",
            "P0R00020-P0R00024:book_collection",
            "P0R00025:author_note_claim_boundary",
        ),
        "source_formulae": (
            "Paper 0 operates as the foundation stone for Book II",
            "assertions are framed as hypotheses with falsifiable observables",
        ),
        "test_protocols": (
            "preserve author and collection context",
            "reject front matter as empirical evidence",
        ),
        "null_results": ("front matter identity is not a validation result",),
        "variables": ("Book_I", "Book_II", "Book_III", "Book_IV", "Book_V"),
        "validation_targets": (
            "preserve five-book collection order",
            "preserve Book II SCPN position",
            "preserve falsifiable-observable boundary",
        ),
        "null_controls": (
            "front-matter-as-evidence control must be rejected",
            "missing-Book-II-position control must be rejected",
            "missing-falsifiability-boundary control must be rejected",
        ),
    },
    "front_matter_context.master_publication_topology": {
        "context_id": "master_publication_topology",
        "validation_protocol": "paper0.front_matter_context.master_publication_topology",
        "canonical_statement": (
            "The master publication list maps Paper 0, the 16 layer monographs, "
            "and Papers 17-20 validation and synthesis suite."
        ),
        "source_equation_ids": (
            "P0R00026:master_publications_toc",
            "P0R00027-P0R00055:publication_topology",
        ),
        "source_formulae": (
            "Part II: The 16 Layer-Specific Monographs",
            "Part III: The Critical Validation & Synthesis Suite",
        ),
        "test_protocols": (
            "preserve 16 layer monograph count",
            "preserve Papers 17-20 validation suite count",
        ),
        "null_results": ("publication topology is a roadmap, not validation evidence",),
        "variables": ("Paper_0", "Papers_1_16", "Papers_17_20"),
        "validation_targets": (
            "preserve Paper 0 foundation position",
            "preserve layer monograph count",
            "preserve validation suite paper count",
        ),
        "null_controls": (
            "missing-layer-monograph control must be rejected",
            "missing-validation-suite control must be rejected",
            "toc-as-proof control must be rejected",
        ),
    },
    "front_matter_context.chapter_structure_marker": {
        "context_id": "chapter_structure_marker",
        "validation_protocol": "paper0.front_matter_context.chapter_structure_marker",
        "canonical_statement": (
            "The source marks the local Paper 0 chapter-structure table before "
            "the blank placeholder block."
        ),
        "source_equation_ids": (
            "P0R00056:foundational_framework_part_marker",
            "P0R00057:paper0_marker",
            "P0R00058:chapter_structure_toc_marker",
        ),
        "source_formulae": ("Chapter Structure and Table of Content",),
        "test_protocols": ("preserve local ToC marker and context-only status",),
        "null_results": ("chapter markers are not validation evidence",),
        "variables": ("chapter_structure_toc",),
        "validation_targets": (
            "preserve Part I marker",
            "preserve Paper 0 marker",
            "preserve chapter-structure marker",
        ),
        "null_controls": (
            "chapter-marker-as-equation control must be rejected",
            "missing-chapter-marker control must be rejected",
        ),
    },
    "front_matter_context.blank_toc_placeholders": {
        "context_id": "blank_toc_placeholders",
        "validation_protocol": "paper0.front_matter_context.blank_toc_placeholders",
        "canonical_statement": (
            "The canonical ledger contains 45 blank ToC placeholder records that must "
            "remain counted rather than silently skipped."
        ),
        "source_equation_ids": ("P0R00059-P0R00103:blank_toc_placeholders",),
        "source_formulae": ("45 blank placeholder records",),
        "test_protocols": ("count blank placeholders as consumed source records",),
        "null_results": ("blank placeholders are source-accounting artefacts",),
        "variables": ("blank_placeholder_count",),
        "validation_targets": (
            "preserve blank placeholder count",
            "prevent silent ledger skipping",
        ),
        "null_controls": (
            "blank-record-skip control must be rejected",
            "negative-placeholder-count control must be rejected",
        ),
    },
    "front_matter_context.fragmented_toc_warning": {
        "context_id": "fragmented_toc_warning",
        "validation_protocol": "paper0.front_matter_context.fragmented_toc_warning",
        "canonical_statement": (
            "The source explicitly warns that the ToC is fragmented and currently incorrect."
        ),
        "source_equation_ids": ("P0R00104:fragmented_toc_warning",),
        "source_formulae": ("fragemented and currently incorect",),
        "test_protocols": (
            "preserve the warning as a quality boundary",
            "reject fragmented ToC context as direct claim evidence",
        ),
        "null_results": ("fragmented ToC warning blocks claim promotion from this ToC",),
        "variables": ("fragmented_toc_warning_present",),
        "validation_targets": (
            "preserve source warning",
            "classify this run as context boundary",
            "prevent ToC-driven evidence promotion",
        ),
        "null_controls": (
            "toc-as-empirical-evidence control must be rejected",
            "missing-fragment-warning control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class FrontMatterContextSpec:
    """Front matter context spec promoted from Paper 0 records."""

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
class FrontMatterContextSpecBundle:
    """Front matter context specs plus source coverage summary."""

    specs: tuple[FrontMatterContextSpec, ...]
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


def build_front_matter_context_specs(
    source_records: list[dict[str, Any]],
) -> FrontMatterContextSpecBundle:
    """Build source-covered front matter context specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    blank_placeholder_count = sum(
        1 for ledger_id in BLANK_PLACEHOLDER_IDS if not str(records_by_ledger[ledger_id]["text"])
    )
    fragmented_warning_present = "fragemented and currently incorect" in str(
        records_by_ledger["P0R00104"]["text"]
    )

    specs: list[FrontMatterContextSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FrontMatterContextSpec(
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
        "title": "Paper 0 Front Matter Context Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "collection_book_count": 5,
        "layer_monograph_count": 16,
        "validation_suite_paper_count": 4,
        "blank_placeholder_count": blank_placeholder_count,
        "fragmented_toc_warning_present": fragmented_warning_present,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return FrontMatterContextSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> FrontMatterContextSpecBundle:
    """Build front matter context specs from the canonical ledger."""
    return build_front_matter_context_specs(load_jsonl(path))


def write_outputs(
    bundle: FrontMatterContextSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown front matter context spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_front_matter_context_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_front_matter_context_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: FrontMatterContextSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Front Matter Context Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Collection books: {bundle.summary['collection_book_count']}",
        f"- Layer monographs: {bundle.summary['layer_monograph_count']}",
        f"- Validation suite papers: {bundle.summary['validation_suite_paper_count']}",
        f"- Blank ToC placeholders: {bundle.summary['blank_placeholder_count']}",
        f"- Fragmented ToC warning present: {bundle.summary['fragmented_toc_warning_present']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 front matter context validation specs."""
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
