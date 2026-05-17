#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Coupling Mechanism: spec builder
"""Promote Paper 0 The Coupling Mechanism: records."""

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
    "P0R02206",
    "P0R02207",
    "P0R02208",
    "P0R02209",
    "P0R02210",
    "P0R02211",
    "P0R02212",
    "P0R02213",
    "P0R02214",
    "P0R02215",
    "P0R02216",
    "P0R02217",
    "P0R02218",
    "P0R02219",
    "P0R02220",
    "P0R02221",
    "P0R02222",
)
CLAIM_BOUNDARY = (
    "source-bounded the coupling mechanism source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_coupling_mechanism.the_coupling_mechanism": {
        "context_id": "the_coupling_mechanism",
        "validation_protocol": "paper0.the_coupling_mechanism.the_coupling_mechanism",
        "canonical_statement": "The source-bounded component 'The Coupling Mechanism:' preserves Paper 0 records P0R02206-P0R02222 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02206:the_coupling_mechanism",
            "P0R02207:the_coupling_mechanism",
            "P0R02208:the_coupling_mechanism",
            "P0R02209:the_coupling_mechanism",
            "P0R02210:the_coupling_mechanism",
            "P0R02211:the_coupling_mechanism",
            "P0R02212:the_coupling_mechanism",
            "P0R02213:the_coupling_mechanism",
            "P0R02214:the_coupling_mechanism",
            "P0R02215:the_coupling_mechanism",
            "P0R02216:the_coupling_mechanism",
            "P0R02217:the_coupling_mechanism",
            "P0R02218:the_coupling_mechanism",
            "P0R02219:the_coupling_mechanism",
            "P0R02220:the_coupling_mechanism",
            "P0R02221:the_coupling_mechanism",
            "P0R02222:the_coupling_mechanism",
        ),
        "source_formulae": (
            "P0R02206: The Coupling Mechanism:",
            'P0R02207: The Psis field (representing the conscious, volitional Self) does not micromanage the engine\'s mechanics. Instead, it acts as the "driver," setting the overall goal. It does this by modulating the precision (lambda) of the global state (sigma). By coupling to the system\'s coherence, the Psi-field can "insist" on a particular policy, effectively increasing the confidence in a desired outcome. This top-down biasing from the Psi-field is what transforms a purely computational cycle of surprise-minimisation into a subjectively experienced, goal-directed action. It provides the "why" that guides the neuro-computational "how."',
            'P0R02208: The case study\'s "four-stroke engine" (Policy Selection, Prediction Generation, Error Processing, Model Consolidation) provides the computational implementation of the Self via Active Inference. This must be explicitly linked to its physical and geometric basis.',
            "P0R02209: This computational loop is the process that defines the Self's identity. This process runs on a stable physical substrate-a coherent Q-ball soliton -that emerges from the L4 dynamics via a Ginzburg-Landau (GL) phase transition. The moment-to-moment experience of this inference process (the \"qualia\") is the intrinsic, evolving geometry of the system's high-dimensional state space, or Consciousness Manifold.",
            'P0R02210: Therefore, the Triple Network Model (DMN/CEN/SN), as the anatomical basis for the "four-stroke engine" , is the neurobiological hardware that runs the "Strange Loop" (I=Model(I)), whose stable, physical form is the soliton and whose experiential quality is the topology of the manifold.',
            "P0R02211: Domain II: Organismal and Planetary Integration (Layers 5-8):",
            "P0R02212: Psychoemotional feedback (Layer 5),",
            "P0R02213: Layer 5: The Four-Stroke Engine of the Self: A Complete Action-Perception Cycle",
            "P0R02214: The organismal Self (Layer 5) is not a static entity but a continuous, dynamic process of engagement with the world. This process is governed by the principle of active inference, where the brain strives to minimize the long-term average of surprise (or variational free energy) by constantly updating its internal model of the world and acting to make its sensory inputs conform to its predictions. This is not an abstract computation but a concrete, neuro-anatomically grounded cycle with four distinct phases, orchestrated by the brain's major subcortical and cortical systems. This four-stroke cycle is the engine of the Self.",
            "P0R02215: Step 1: Policy Selection (The Basal Ganglia)",
            'P0R02216: The cycle begins with a decision. Faced with a multitude of possible actions, the brain must select one. This is the primary function of the basal ganglia, a collection of subcortical nuclei that act as a centralized action selection mechanism. The basal ganglia receive inputs from across the cortex, representing potential actions or "policies." Through a complex interplay of excitatory ("direct pathway") and inhibitory ("indirect pathway") circuits, they evaluate these policies based on their predicted outcomes and associated rewards, learned from past experience. The output of the basal ganglia is a powerful inhibitory signal that tonically suppresses motor commands in the thalamus and brainstem. Action selection is achieved by selectively releasing this inhibition for the chosen policy, allowing it to proceed while continuing to suppress all competing alternatives. This process effectively resolves the conflict between potential actions and commits the organism to a single, coherent course of action.',
            "P0R02217: Step 2: Prediction Generation (The Cerebellum)",
            'P0R02218: Once a policy is selected, the brain must anticipate its sensory consequences in detail. This predictive modeling is the domain of the cerebellum. Long considered a purely motor structure, the cerebellum is now understood to be a universal prediction machine, or "forward model". It receives a copy of the selected motor command (an "efference copy") from the cortex and, using its massively parallel and uniform circuitry, computes a high-fidelity simulation of the expected sensory feedback-proprioceptive, visual, auditory-that should result from that action. This is the "Universal Cerebellar Transform" described in the manuscript. This prediction is not a vague expectation but a detailed, moment-by-moment simulation of the impending sensory stream. This predictive model is then broadcast from the cerebellum back through the thalamus to the relevant sensory and association cortices.',
            "P0R02219: Step 3: Action, Perception, and Error Processing (The Cortex)",
            "P0R02220: With the policy selected and the prediction generated, the core of the active inference loop unfolds in the cortex. The cortex executes the motor command while simultaneously receiving two streams of information: the top-down prediction from the cerebellum and the bottom-up sensory input from the periphery. Perception, in this model, is the process of comparing these two streams. The brain does not passively construct a model of the world from raw sensory data. Instead, it uses sensory data to test its own predictions.",
            'P0R02221: The top-down cerebellar prediction effectively "explains away" the predictable components of the sensory stream. The only information that propagates up the cortical hierarchy is the residual difference between the prediction and the reality: the prediction error. This prediction error is the fundamental currency of learning and belief updating in the brain. It is a multi-modal signal that informs the system that its model is, in some way, inaccurate.',
            "P0R02222: This error signal is broadcast widely, driving synaptic plasticity not only in the cortex but also back in the cerebellum (via climbing fiber inputs) and the basal ganglia (modulating future policy selection), allowing the entire action-perception system to refine itself based on the outcomes of its actions.",
        ),
        "test_protocols": ("preserve The Coupling Mechanism: source-accounting boundary",),
        "null_results": ("The Coupling Mechanism: is not empirical validation evidence",),
        "variables": ("the_coupling_mechanism",),
        "validation_targets": ("preserve records P0R02206-P0R02222",),
        "null_controls": ("the_coupling_mechanism must remain source-bounded accounting",),
    }
}


@dataclass(frozen=True, slots=True)
class TheCouplingMechanismSpec:
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
class TheCouplingMechanismSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheCouplingMechanismSpec, ...]
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


def build_the_coupling_mechanism_specs(
    source_records: list[dict[str, Any]],
) -> TheCouplingMechanismSpecBundle:
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

    specs: list[TheCouplingMechanismSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheCouplingMechanismSpec(
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
        "title": "Paper 0 " + "The Coupling Mechanism:" + " Specs",
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
        "next_source_boundary": "P0R02223",
    }
    return TheCouplingMechanismSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> TheCouplingMechanismSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_coupling_mechanism_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheCouplingMechanismSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Coupling Mechanism:" + " Specs",
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
    bundle: TheCouplingMechanismSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_the_coupling_mechanism_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_the_coupling_mechanism_validation_specs_{date_tag}.md"
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
