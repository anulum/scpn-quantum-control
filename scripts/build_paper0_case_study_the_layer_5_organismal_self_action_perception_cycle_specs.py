#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle spec builder
"""Promote Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle records."""

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
    "P0R02177",
    "P0R02178",
    "P0R02179",
    "P0R02180",
    "P0R02181",
    "P0R02182",
    "P0R02183",
    "P0R02184",
    "P0R02185",
    "P0R02186",
    "P0R02187",
    "P0R02188",
)
CLAIM_BOUNDARY = "source-bounded case study the layer 5 organismal self action perception cycle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "case_study_the_layer_5_organismal_self_action_perception_cycle.case_study_the_layer_5_organismal_self_action_perception_cycle": {
        "context_id": "case_study_the_layer_5_organismal_self_action_perception_cycle",
        "validation_protocol": "paper0.case_study_the_layer_5_organismal_self_action_perception_cycle.case_study_the_layer_5_organismal_self_action_perception_cycle",
        "canonical_statement": "The source-bounded component 'Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle' preserves Paper 0 records P0R02177-P0R02188 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02177:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02178:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02179:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02180:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02181:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02182:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02183:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02184:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02185:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02186:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02187:case_study_the_layer_5_organismal_self_action_perception_cycle",
            "P0R02188:case_study_the_layer_5_organismal_self_action_perception_cycle",
        ),
        "source_formulae": (
            "P0R02177: Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle",
            'P0R02178: This section presents a neuro-anatomically grounded model of the organismal Self (Layer 5) as a continuous, four-phase action-perception cycle, providing a concrete implementation of the active inference framework. The model reframes the Self not as a static entity but as a dynamic process aimed at minimising variational free energy. Each phase of this "four-stroke engine" is mapped to the specific computational functions of major brain networks.',
            "P0R02179: Phase 1, Policy Selection, is assigned to the basal ganglia. This subcortical system acts as a high-stakes conflict resolution hub, evaluating potential actions (policies) based on learned reward predictions. Its output is not excitatory but rather a selective disinhibition, a sophisticated control mechanism that releases one chosen action from tonic suppression while holding all others in check, thus ensuring coherent, unitary action.",
            "P0R02180: Phase 2, Prediction Generation, is the domain of the cerebellum, which functions as a universal forward model. Upon receiving an efference copy of the selected motor command, the cerebellum computes a high-fidelity, real-time simulation of the expected sensory consequences. This detailed prediction is then projected to the relevant cortical areas, preparing them for the incoming sensory stream.",
            'P0R02181: Phase 3, Error Processing, occurs in the cortex, which serves as the primary comparator. Perception is defined here not as the passive reception of sensory data, but as the active process of subtracting the cerebellar prediction from the actual sensory input. The resulting signal, the prediction error, is the only information that propagates up the cortical hierarchy. This residual "surprise" is the fundamental currency for learning, driving synaptic plasticity throughout the entire network to refine future predictions and policies.',
            "P0R02182: Phase 4, Model Consolidation, is achieved during sleep. This offline phase is critical for system maintenance and long-term learning. During NREM sleep, hippocampal replay, coordinated with cortical slow oscillations, facilitates the transfer of episodic memories into the neocortical long-term store (a process linking Layer 5 to Layer 9). Concurrently, synaptic homeostasis resets the network to a state of quasicriticality. REM sleep, conversely, serves to refine the generative model itself by running offline simulations (dreams), allowing the system to explore policy-outcome space without real-world consequence. This comprehensive, four-phase cycle provides a complete, end-to-end neuro-computational architecture for an actively inferring Self.",
            "P0R02183: This section reveals that your \"Self\" isn't a thing, it's a process-a constantly running, four-stroke engine that allows you to interact with the world. It's the cycle of how you decide, predict, act, and learn.",
            "P0R02184: Stroke 1: Decide (The Ignition Spark). This happens in a deep, ancient part of your brain called the basal ganglia. It's the ultimate decision-maker. It looks at all your possible next moves, weighs the pros and cons based on past experience, and then picks one winner. It does this by slamming the brakes on every option except the chosen one, ensuring you do one thing at a time.",
            "P0R02185: Stroke 2: Predict (The Fuel Injection). Once you've decided to act, your cerebellum-your brain's super-powerful simulation deck-kicks in. If you decide to pick up a coffee cup, your cerebellum instantly runs a perfect simulation of exactly how that should feel, look, and sound. It sends this \"movie trailer\" of the future to the rest of your brain.",
            'P0R02186: Stroke 3: Act & Compare (The Power Stroke). Now your cortex-the big, wrinkly part of your brain-takes over. It sends the command to your muscles while simultaneously watching two screens: the "movie trailer" from the cerebellum and the "live feed" from your senses. Your brain doesn\'t care about the parts that match. It\'s only looking for the difference, the "surprise." If the cup is hotter than you predicted, that "surprise!" signal is the only thing that gets flagged. This surprise is the most important signal in your brain; it\'s the raw material for all learning.',
            "P0R02187: Stroke 4: Learn & Reboot (The Exhaust). This critical step happens when you sleep. During deep sleep, your brain is like a librarian, taking the important lessons of the day and filing them away for long-term storage. It also performs a vital system cleanup, resetting all the connections so you're ready for the next day. During dream (REM) sleep, your brain runs wild simulations, like a virtual reality game, testing out new ideas and strategies in a safe environment. This entire cycle-Decide, Predict, Act, Learn-is the engine that creates your experience of being a Self.",
            "P0R02188: P0R02188",
        ),
        "test_protocols": (
            "preserve Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle source-accounting boundary",
        ),
        "null_results": (
            "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle is not empirical validation evidence",
        ),
        "variables": ("case_study_the_layer_5_organismal_self_action_perception_cycle",),
        "validation_targets": ("preserve records P0R02177-P0R02188",),
        "null_controls": (
            "case_study_the_layer_5_organismal_self_action_perception_cycle must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpec:
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
class CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpec, ...]
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


def build_case_study_the_layer_5_organismal_self_action_perception_cycle_specs(
    source_records: list[dict[str, Any]],
) -> CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle:
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

    specs: list[CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpec(
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
        + "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle"
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
        "next_source_boundary": "P0R02189",
    }
    return CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_case_study_the_layer_5_organismal_self_action_perception_cycle_specs(
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


def render_report(bundle: CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle"
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
    bundle: CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_case_study_the_layer_5_organismal_self_action_perception_cycle_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_case_study_the_layer_5_organismal_self_action_perception_cycle_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Layer 5 organismal self-action case-study specs."""

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
