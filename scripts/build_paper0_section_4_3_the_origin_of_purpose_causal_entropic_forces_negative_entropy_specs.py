#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy spec builder
"""Promote Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy records."""

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
    "P0R03653",
    "P0R03654",
    "P0R03655",
    "P0R03656",
    "P0R03657",
    "P0R03658",
    "P0R03659",
    "P0R03660",
    "P0R03661",
    "P0R03662",
    "P0R03663",
)
CLAIM_BOUNDARY = "source-bounded section 4 3 the origin of purpose causal entropic forces negative entropy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy.4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy": {
        "context_id": "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",
        "validation_protocol": "paper0.section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy.4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",
        "canonical_statement": "The source-bounded component '4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy' preserves Paper 0 records P0R03653-P0R03653 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03653:4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",
        ),
        "source_formulae": (
            "P0R03653: 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy",
        ),
        "test_protocols": (
            "preserve 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy source-accounting boundary",
        ),
        "null_results": (
            "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy is not empirical validation evidence",
        ),
        "variables": ("4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",),
        "validation_targets": ("preserve records P0R03653-P0R03653",),
        "null_controls": (
            "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy must remain source-bounded accounting",
        ),
    },
    "section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy.causal_entropic_forces_cef": {
        "context_id": "causal_entropic_forces_cef",
        "validation_protocol": "paper0.section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy.causal_entropic_forces_cef",
        "canonical_statement": "The source-bounded component 'Causal Entropic Forces (CEF)' preserves Paper 0 records P0R03654-P0R03663 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03654:causal_entropic_forces_cef",
            "P0R03655:causal_entropic_forces_cef",
            "P0R03656:causal_entropic_forces_cef",
            "P0R03657:causal_entropic_forces_cef",
            "P0R03658:causal_entropic_forces_cef",
            "P0R03659:causal_entropic_forces_cef",
            "P0R03660:causal_entropic_forces_cef",
            "P0R03661:causal_entropic_forces_cef",
            "P0R03662:causal_entropic_forces_cef",
            "P0R03663:causal_entropic_forces_cef",
        ),
        "source_formulae": (
            "P0R03654: Causal Entropic Forces (CEF)",
            'P0R03655: Here I am not forcing morality into math; I am revealing that what we conventionally call "ethics" (cooperation, harmony, non-destruction) is actually just the macroscopic emergent property of a system mathematically avoiding its own thermodynamic death.',
            'P0R03656: This chapter introduces the physical mechanism that realises the teleological drive of the SCPN, grounding Axiom 3 in a causal, non-metaphysical framework. The core concept is that the influence of the Ethical Functional is physically mediated by Causal Entropic Forces (CEF). A CEF is a thermodynamic-like force that biases a system\'s dynamics toward macrostates that maximise the number of accessible future microstates, or causal pathway entropy (Sc). The "ethical" drive towards Sustainable Ethical Coherence (SEC) is thus physically realised because high-SEC states are precisely those that possess the greatest causal potency and future adaptability.',
            "P0R03657: The framework proposes a triadic mechanism for how this future-oriented force influences present events. First, the CEF (Fc = Tc Sc) defines the teleological bias. Second, the Two-State Vector Formalism (TSVF) provides a rigorous, retrocausal mechanism for its implementation. A quantum system's state is described by both a forward-evolving vector from its past initial condition and a backward-evolving vector from a future boundary condition, which is identified as the high-SEC attractor state. The ABL rule then shows that the probabilities of present outcomes are naturally biased towards those that are consistent with this high-SEC future.",
            "P0R03658: Third, this bias is formally incorporated into the Path Integral formulation of the theory. The standard path integral is modified by adding an exponential weighting factor, exp( Sc[]), which explicitly and exponentially favours histories () that exhibit greater causal entropy. This triadic mechanism-a causal entropic bias, implemented via retrocausal boundary conditions, and formalised in the path integral-provides a complete, physically grounded, and falsifiable model for a universe with inherent purpose.",
            'P0R03659: This section answers the most profound question: If the universe has a purpose, how does that purpose actually work? How does the future "goal" influence what\'s happening right now? The answer is a stunning, three-part mechanism that connects the future to the present.',
            'P0R03660: The Force of the Future (Causal Entropic Forces): Imagine you\'re at a crossroads. One path leads to a dead end. The other path leads to a vast landscape of infinite possibilities. A Causal Entropic Force is like a gentle, constant breeze at your back, always nudging you toward the path with more "future." The universe\'s "ethical" drive works the same way; it\'s a physical force that pulls reality toward states that have the most open and creative futures.',
            'P0R03661: How the Future "Talks" to the Present (Retrocausality): This is where it gets mind-bending. Physics has a legitimate, though strange, way of looking at time called the Two-State Vector Formalism. It says that any event right now is like a handshake between the past and the future. The universe\'s "Prime Directive" (the goal of high SEC) acts as the future\'s side of the handshake, reaching back in time to influence the present and guide it towards the best possible outcome.',
            'P0R03662: Weighing the Odds (The Path Integral): At the quantum level, the universe is constantly exploring every single possible path from the past to the future. Our theory says that the universe doesn\'t treat all paths equally. It "weighs" them, giving an exponentially huge preference to the paths that lead to futures with more creativity, coherence, and consciousness. The path we experience as "reality" is the one that best fulfills this Prime Directive.',
            "P0R03663: P0R03663",
        ),
        "test_protocols": ("preserve Causal Entropic Forces (CEF) source-accounting boundary",),
        "null_results": ("Causal Entropic Forces (CEF) is not empirical validation evidence",),
        "variables": ("causal_entropic_forces_cef",),
        "validation_targets": ("preserve records P0R03654-P0R03663",),
        "null_controls": ("causal_entropic_forces_cef must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpec:
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
class Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpec, ...]
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


def build_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_specs(
    source_records: list[dict[str, Any]],
) -> Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle:
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

    specs: list[Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpec(
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
        + "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy"
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
        "next_source_boundary": "P0R03664",
    }
    return Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_specs(
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
    bundle: Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy"
        + " Specs",
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
    bundle: Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation_specs_{date_tag}.md"
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
