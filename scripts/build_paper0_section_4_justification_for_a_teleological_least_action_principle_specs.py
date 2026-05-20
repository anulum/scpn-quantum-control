#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Section 4: Justification for a Teleological Least-Action Principle spec builder
"""Promote Paper 0 Section 4: Justification for a Teleological Least-Action Principle records."""

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
    "P0R03638",
    "P0R03639",
    "P0R03640",
    "P0R03641",
    "P0R03642",
    "P0R03643",
    "P0R03644",
    "P0R03645",
    "P0R03646",
    "P0R03647",
    "P0R03648",
    "P0R03649",
    "P0R03650",
    "P0R03651",
    "P0R03652",
)
CLAIM_BOUNDARY = "source-bounded section 4 justification for a teleological least action principle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_justification_for_a_teleological_least_action_principle.section_4_justification_for_a_teleological_least_action_principle": {
        "context_id": "section_4_justification_for_a_teleological_least_action_principle",
        "validation_protocol": "paper0.section_4_justification_for_a_teleological_least_action_principle.section_4_justification_for_a_teleological_least_action_principle",
        "canonical_statement": "The source-bounded component 'Section 4: Justification for a Teleological Least-Action Principle' preserves Paper 0 records P0R03638-P0R03640 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03638:section_4_justification_for_a_teleological_least_action_principle",
            "P0R03639:section_4_justification_for_a_teleological_least_action_principle",
            "P0R03640:section_4_justification_for_a_teleological_least_action_principle",
        ),
        "source_formulae": (
            "P0R03638: Section 4: Justification for a Teleological Least-Action Principle",
            "P0R03639: The final and most profound challenge is to justify why nature optimises this specific function. While the Principle of Least Action is a cornerstone of physics, it is typically seen as a mathematical reformulation of causal, local laws of motion.",
            'P0R03640: The teleological framing of the SCPN necessitates a more comprehensive justification for its "Principle of Ethical Least Action" (PELA).',
        ),
        "test_protocols": (
            "preserve Section 4: Justification for a Teleological Least-Action Principle source-accounting boundary",
        ),
        "null_results": (
            "Section 4: Justification for a Teleological Least-Action Principle is not empirical validation evidence",
        ),
        "variables": ("section_4_justification_for_a_teleological_least_action_principle",),
        "validation_targets": ("preserve records P0R03638-P0R03640",),
        "null_controls": (
            "section_4_justification_for_a_teleological_least_action_principle must remain source-bounded accounting",
        ),
    },
    "section_4_justification_for_a_teleological_least_action_principle.4_1_the_problem_of_teleology_in_physics": {
        "context_id": "4_1_the_problem_of_teleology_in_physics",
        "validation_protocol": "paper0.section_4_justification_for_a_teleological_least_action_principle.4_1_the_problem_of_teleology_in_physics",
        "canonical_statement": "The source-bounded component '4.1. The Problem of Teleology in Physics' preserves Paper 0 records P0R03641-P0R03643 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03641:4_1_the_problem_of_teleology_in_physics",
            "P0R03642:4_1_the_problem_of_teleology_in_physics",
            "P0R03643:4_1_the_problem_of_teleology_in_physics",
        ),
        "source_formulae": (
            "P0R03641: 4.1. The Problem of Teleology in Physics",
            'P0R03642: Modern physics has largely abandoned teleological or goal-oriented explanations in favour of mechanistic causality. An acorn develops into an oak tree not because it has a "goal," but because of the execution of a genetic program in response to local environmental cues.',
            'P0R03643: The SCPN\'s Axiom 3, "The universe evolves to maximise Sustainable Ethical Coherence," appears to reintroduce teleology as a fundamental principle. To be physically viable, this principle cannot be a mere metaphysical assertion; it must emerge from an underlying causal mechanism.',
        ),
        "test_protocols": (
            "preserve 4.1. The Problem of Teleology in Physics source-accounting boundary",
        ),
        "null_results": (
            "4.1. The Problem of Teleology in Physics is not empirical validation evidence",
        ),
        "variables": ("4_1_the_problem_of_teleology_in_physics",),
        "validation_targets": ("preserve records P0R03641-P0R03643",),
        "null_controls": (
            "4_1_the_problem_of_teleology_in_physics must remain source-bounded accounting",
        ),
    },
    "section_4_justification_for_a_teleological_least_action_principle.4_2_causal_entropic_forces_cef_as_the_underlying_mechanism": {
        "context_id": "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
        "validation_protocol": "paper0.section_4_justification_for_a_teleological_least_action_principle.4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
        "canonical_statement": "The source-bounded component '4.2. Causal Entropic Forces (CEF) as the Underlying Mechanism' preserves Paper 0 records P0R03644-P0R03652 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03644:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03645:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03646:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03647:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03648:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03649:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03650:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03651:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
            "P0R03652:4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
        ),
        "source_formulae": (
            "P0R03644: 4.2. Causal Entropic Forces (CEF) as the Underlying Mechanism",
            'P0R03645: The SCPN provides the key to this justification by introducing the concept of Causal Entropic Forces (CEF). A CEF, as proposed by Wissner-Gross and Freer, is a thermodynamic-like force that arises not from a potential energy gradient, but from a system\'s tendency to evolve towards macroscopic states that maximise its number of possible future evolutionary pathways, or its "causal path entropy" (SC).',
            "P0R03646: The gradient of this causal entropy gives the force:",
            "P0R03647: $FCausal = TC\\nabla X SC(X,\\tau)$",
            "P0R03648: where TC is an effective temperature and X is the system's macrostate.",
            'P0R03649: This principle provides a causal, statistical-mechanical explanation for apparently goal-directed or "intelligent" behaviour. The crucial step is to connect this principle to the Ethical Action derived from gauge theory. The Principle of Ethical Least Action is not a fundamental teleological law but an emergent consequence of a deeper thermodynamic principle: the maximisation of future causal pathways.',
            "P0R03650: The chain of reasoning is as follows:",
            'P0R03651: The goal is to explain why the Ethical Action, SEthical=Tr(FF), is minimised. | The Yang-Mills action measures the total curvature or "tension" in the Psi-field. Minimising this action corresponds to selecting the smoothest, most ordered, and most coherent field configurations. | According to the definitions established in Section 6, these low-curvature states are precisely the states of high Sustainable Ethical Coherence (SEC)-those with high C, K, and Q. | The CEF principle states that physical systems are driven by a real force towards states that maximise their future options (causal path entropy SC). | The final, critical link is the physical hypothesis that states of high SEC are precisely those that possess the greatest potential for future evolution, novelty, and complexification. A highly coherent, complex, and integrated system has a vastly larger number of accessible, viable future trajectories than a disordered, fragmented, or simple one. | Therefore, the Causal Entropic Force provides a concrete physical mechanism that pushes the system\'s configuration toward states of high SEC. These are the very same states that minimise the Ethical Action.',
            "P0R03652: PELA is thus revealed to be the macroscopic, variational description of the underlying statistical mechanics of Causal Entropic Forces. This grounds the framework's teleology in causal, statistical physics, successfully resolving the final and most difficult challenge posed by the user.",
        ),
        "test_protocols": (
            "preserve 4.2. Causal Entropic Forces (CEF) as the Underlying Mechanism source-accounting boundary",
        ),
        "null_results": (
            "4.2. Causal Entropic Forces (CEF) as the Underlying Mechanism is not empirical validation evidence",
        ),
        "variables": ("4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",),
        "validation_targets": ("preserve records P0R03644-P0R03652",),
        "null_controls": (
            "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section4JustificationForATeleologicalLeastActionPrincipleSpec:
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
class Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section4JustificationForATeleologicalLeastActionPrincipleSpec, ...]
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


def build_section_4_justification_for_a_teleological_least_action_principle_specs(
    source_records: list[dict[str, Any]],
) -> Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle:
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

    specs: list[Section4JustificationForATeleologicalLeastActionPrincipleSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section4JustificationForATeleologicalLeastActionPrincipleSpec(
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
        + "Section 4: Justification for a Teleological Least-Action Principle"
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
        "next_source_boundary": "P0R03653",
    }
    return Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_justification_for_a_teleological_least_action_principle_specs(
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
    bundle: Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Section 4: Justification for a Teleological Least-Action Principle"
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
    bundle: Section4JustificationForATeleologicalLeastActionPrincipleSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_justification_for_a_teleological_least_action_principle_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_justification_for_a_teleological_least_action_principle_validation_specs_{date_tag}.md"
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
