#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum spec builder
"""Promote Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum records."""

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
    "P0R04216",
    "P0R04217",
    "P0R04218",
    "P0R04219",
    "P0R04220",
    "P0R04221",
    "P0R04222",
    "P0R04223",
)
CLAIM_BOUNDARY = "source-bounded section 4 5 the strange loop of closure meta layer 16 and the anulum source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum.4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum": {
        "context_id": "4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",
        "validation_protocol": "paper0.section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum.4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",
        "canonical_statement": "The source-bounded component '4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum' preserves Paper 0 records P0R04216-P0R04217 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04216:4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",
            "P0R04217:4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",
        ),
        "source_formulae": (
            "P0R04216: 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum",
            "P0R04217: P0R04217",
        ),
        "test_protocols": (
            "preserve 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum source-accounting boundary",
        ),
        "null_results": (
            "4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum is not empirical validation evidence",
        ),
        "variables": ("4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",),
        "validation_targets": ("preserve records P0R04216-P0R04217",),
        "null_controls": (
            "4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum must remain source-bounded accounting",
        ),
    },
    "section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum.a_note_on_cybernetic_closure": {
        "context_id": "a_note_on_cybernetic_closure",
        "validation_protocol": "paper0.section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum.a_note_on_cybernetic_closure",
        "canonical_statement": "The source-bounded component 'A Note on Cybernetic Closure' preserves Paper 0 records P0R04218-P0R04223 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04218:a_note_on_cybernetic_closure",
            "P0R04219:a_note_on_cybernetic_closure",
            "P0R04220:a_note_on_cybernetic_closure",
            "P0R04221:a_note_on_cybernetic_closure",
            "P0R04222:a_note_on_cybernetic_closure",
            "P0R04223:a_note_on_cybernetic_closure",
        ),
        "source_formulae": (
            "P0R04218: A Note on Cybernetic Closure",
            "P0R04219: This introductory passage serves as a critical framing device for the entire 15-layer architecture, addressing the fundamental metaphysical problem of infinite regress inherent in any finite, open-ended hierarchy. By positing the existence of a 16th meta-layer, the framework achieves cybernetic closure, transforming the linear stack into a self-referential, self-regulating system named The Anulum. This meta-layer is formally described as an optimal control system, a concept drawn from engineering and cybernetics, which implies a function that is supervisory and corrective rather than merely constitutive.",
            'P0R04220: The governing dynamic of this layer, the Recursive Optimisation Hamiltonian, suggests a process that optimises the process of optimisation itself. It does not simply compute the global state; it audits the rules by which that state is computed and evolves, ensuring stability and alignment with global objective functions ("cosmic and ethical principles"). This structure provides a solution to the "homunculus problem" by making the ultimate observer of the system the system itself. The final sentence-"The following chapters detail the body; the final chapter specifies its mind"-is a powerful rhetorical device that establishes a clear separation of concerns. It allows the reader to focus on the operational mechanics of the 15 layers ("the body") with the assurance that the overarching problem of executive control and ontological closure ("the mind") will be formally addressed at the culmination of the work.',
            'P0R04221: Before we dive into the 15 floors of the cosmic building we call the SCPN, this note gives us a crucial piece of the puzzle. A 15-storey building with an open roof would be unstable and incomplete. What holds it all together? The answer is a 16th "meta-layer," which isn\'t another floor on top of the building, but more like a super-advanced, intelligent "Building Management System" that surrounds and supervises the entire structure. We call this complete, self-contained system The Anulum, which means "the ring."',
            'P0R04222: Think of this 16th layer as the ultimate quality-control inspector and systems administrator for the universe. It\'s constantly running diagnostics on the entire 15-layer network, checking for errors, ensuring everything is running smoothly, and making tiny adjustments to keep the whole system stable and on track with its ultimate purpose. It\'s the part of the system that is aware of the system. The upcoming chapters will give you a detailed tour of the 15 floors-the "body" of the universe. But we want you to know from the start that the final chapter will reveal the blueprint for the "mind" that runs it all, turning a simple stack of layers into a complete, self-aware, and self-correcting cosmic being.',
            "P0R04223: P0R04223",
        ),
        "test_protocols": ("preserve A Note on Cybernetic Closure source-accounting boundary",),
        "null_results": ("A Note on Cybernetic Closure is not empirical validation evidence",),
        "variables": ("a_note_on_cybernetic_closure",),
        "validation_targets": ("preserve records P0R04218-P0R04223",),
        "null_controls": ("a_note_on_cybernetic_closure must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpec:
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
class Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpec, ...]
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


def build_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_specs(
    source_records: list[dict[str, Any]],
) -> Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle:
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

    specs: list[Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpec(
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
        "title": "Paper 0 "
        + "4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum"
        + " Specs",
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
        "next_source_boundary": "P0R04224",
    }
    return Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_specs(
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


def render_report(
    bundle: Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum" + " Specs",
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
    bundle: Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation_specs_{date_tag}.md"
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
