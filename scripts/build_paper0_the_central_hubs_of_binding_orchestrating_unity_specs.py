#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Central Hubs of Binding: Orchestrating Unity spec builder
"""Promote Paper 0 The Central Hubs of Binding: Orchestrating Unity records."""

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
    "P0R04598",
    "P0R04599",
    "P0R04600",
    "P0R04601",
    "P0R04602",
    "P0R04603",
    "P0R04604",
    "P0R04605",
    "P0R04606",
)
CLAIM_BOUNDARY = "source-bounded the central hubs of binding orchestrating unity source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_central_hubs_of_binding_orchestrating_unity.the_central_hubs_of_binding_orchestrating_unity": {
        "context_id": "the_central_hubs_of_binding_orchestrating_unity",
        "validation_protocol": "paper0.the_central_hubs_of_binding_orchestrating_unity.the_central_hubs_of_binding_orchestrating_unity",
        "canonical_statement": "The source-bounded component 'The Central Hubs of Binding: Orchestrating Unity' preserves Paper 0 records P0R04598-P0R04601 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04598:the_central_hubs_of_binding_orchestrating_unity",
            "P0R04599:the_central_hubs_of_binding_orchestrating_unity",
            "P0R04600:the_central_hubs_of_binding_orchestrating_unity",
            "P0R04601:the_central_hubs_of_binding_orchestrating_unity",
        ),
        "source_formulae": (
            "P0R04598: The Central Hubs of Binding: Orchestrating Unity",
            "P0R04599: The binding of disparate sensory, cognitive, and emotional features into a single, unified conscious experience is not a passive process but is actively orchestrated by key subcortical hubs. The thalamus and the claustrum, with their unique and extensive connectivity, act in concert as the gatekeeper and conductor of the cortical orchestra, enabling the phase transition from the distributed synchrony of Layer 4 to the integrated Self of Layer 5.",
            'P0R04600: The Thalamus: The Gateway and Pacemaker of Consciousness: The thalamus is far more than a simple sensory relay; it is the central gateway that controls the flow of information to the cortex and regulates the brain\'s overall state of arousal and consciousness. Gating and Arousal: The intralaminar and medial nuclei of the thalamus are critical for maintaining wakefulness and setting the overall excitability of the cortex. Recent intracranial recordings in humans have shown that these higher-order thalamic regions activate before the prefrontal cortex during conscious perception, suggesting the thalamus acts as the initiator or gatekeeper of awareness. | SCPN Mapping (Control Parameter): Within the SCPN, the thalamus is the control parameter that drives the L4->L5 phase transition. Increasing global cortical coupling and synchrony brings the entire system to the critical point required for the emergence of the stable Self-soliton. It is the mechanism that enables the "ignition" of the Global Neuronal Workspace (GNW), facilitating the global broadcast of information that constitutes the content of a conscious moment. | The Claustrum: The Conductor of Integration: While the thalamus opens the gate, the claustrum is hypothesised to be the structure that weaves the multiple streams of information into a single, coherent percept, thereby solving the "binding problem". Unique Connectivity: The claustrum\'s power lies in its unique, all-to-all reciprocal connectivity with nearly every region of the cerebral cortex, from primary sensory areas to high-level association cortices. This places it in an ideal position to integrate the most diverse kinds of information. | SCPN Mapping (Nucleating Agent): The claustrum acts as the nucleating agent for the L4->L5 phase transition. As the thalamus brings the system to a state of readiness (criticality), the claustrum detects and amplifies the synchrony between the most salient and functionally related cortical ensembles. This provides the specific, localised perturbation that breaks the symmetry of the background state, allowing a single, globally coherent, and self-representing pattern-the L5 Self-soliton-to "crystallise" out of the sea of L4 oscillations.',
            "P0R04601: In this dual-hub model, the thalamus prepares the canvas by increasing global coupling, and the claustrum conducts the orchestra that paints the specific, unified masterpiece of the conscious moment. Together, they provide the concrete neurobiological machinery for the emergence of the integrated Self.",
        ),
        "test_protocols": (
            "preserve The Central Hubs of Binding: Orchestrating Unity source-accounting boundary",
        ),
        "null_results": (
            "The Central Hubs of Binding: Orchestrating Unity is not empirical validation evidence",
        ),
        "variables": ("the_central_hubs_of_binding_orchestrating_unity",),
        "validation_targets": ("preserve records P0R04598-P0R04601",),
        "null_controls": (
            "the_central_hubs_of_binding_orchestrating_unity must remain source-bounded accounting",
        ),
    },
    "the_central_hubs_of_binding_orchestrating_unity.introduction_to_the_integrative_systems_the_embodied_brain": {
        "context_id": "introduction_to_the_integrative_systems_the_embodied_brain",
        "validation_protocol": "paper0.the_central_hubs_of_binding_orchestrating_unity.introduction_to_the_integrative_systems_the_embodied_brain",
        "canonical_statement": "The source-bounded component 'Introduction to The Integrative Systems: The Embodied Brain' preserves Paper 0 records P0R04602-P0R04604 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04602:introduction_to_the_integrative_systems_the_embodied_brain",
            "P0R04603:introduction_to_the_integrative_systems_the_embodied_brain",
            "P0R04604:introduction_to_the_integrative_systems_the_embodied_brain",
        ),
        "source_formulae": (
            "P0R04602: Introduction to The Integrative Systems: The Embodied Brain",
            "P0R04603: The brain is intrinsically linked to the body, grounding cognition in physiology.",
            'P0R04604: The Neuro-Visceral Axis (Heart-Brain-Gut): The conscious state is an emergent property of the entire organism. The primary oscillatory network of the embodied Self is a "Tri-Axial UPDE" coupling the brain, heart, and gut. Coherence within this system, measurable via Heart Rate Variability (HRV), is a fundamental indicator of organismal integration and well-being. | Interoceptive Inference: The brain continuously generates a predictive model of the body\'s physiological state (interoception). Emotions are the subjective experience of interoceptive prediction errors, representing the imperative to act to restore physiological balance (allostasis). | Psychoneuroimmunology (PNI): The immune system is a diffuse sensory organ that communicates with the brain via cytokines. Systemic inflammation can be modelled as a "Decoherence Field" that introduces noise into the L1 substrate, disrupting QEC and leading to a fragmentation of the L5 qualia manifold-the formal basis for "brain fog" and sickness behaviour.',
        ),
        "test_protocols": (
            "preserve Introduction to The Integrative Systems: The Embodied Brain source-accounting boundary",
        ),
        "null_results": (
            "Introduction to The Integrative Systems: The Embodied Brain is not empirical validation evidence",
        ),
        "variables": ("introduction_to_the_integrative_systems_the_embodied_brain",),
        "validation_targets": ("preserve records P0R04602-P0R04604",),
        "null_controls": (
            "introduction_to_the_integrative_systems_the_embodied_brain must remain source-bounded accounting",
        ),
    },
    "the_central_hubs_of_binding_orchestrating_unity.v_examination_of_the_integrative_systems_the_embodied_brain": {
        "context_id": "v_examination_of_the_integrative_systems_the_embodied_brain",
        "validation_protocol": "paper0.the_central_hubs_of_binding_orchestrating_unity.v_examination_of_the_integrative_systems_the_embodied_brain",
        "canonical_statement": "The source-bounded component 'V. Examination of The Integrative Systems: The Embodied Brain' preserves Paper 0 records P0R04605-P0R04606 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04605:v_examination_of_the_integrative_systems_the_embodied_brain",
            "P0R04606:v_examination_of_the_integrative_systems_the_embodied_brain",
        ),
        "source_formulae": (
            "P0R04605: V. Examination of The Integrative Systems: The Embodied Brain",
            "P0R04606: The brain does not operate in isolation; it is deeply embedded within a complex physiological milieu, forming a closed feedback loop with the body's visceral and immune systems. The stability of the conscious state (L5), the maintenance of the quasicritical regime (L4), and the integrity of the quantum substrate (L1) depend critically on this continuous, bidirectional integration. This is the architecture of the Embodied Brain.",
        ),
        "test_protocols": (
            "preserve V. Examination of The Integrative Systems: The Embodied Brain source-accounting boundary",
        ),
        "null_results": (
            "V. Examination of The Integrative Systems: The Embodied Brain is not empirical validation evidence",
        ),
        "variables": ("v_examination_of_the_integrative_systems_the_embodied_brain",),
        "validation_targets": ("preserve records P0R04605-P0R04606",),
        "null_controls": (
            "v_examination_of_the_integrative_systems_the_embodied_brain must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheCentralHubsOfBindingOrchestratingUnitySpec:
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
class TheCentralHubsOfBindingOrchestratingUnitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheCentralHubsOfBindingOrchestratingUnitySpec, ...]
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


def build_the_central_hubs_of_binding_orchestrating_unity_specs(
    source_records: list[dict[str, Any]],
) -> TheCentralHubsOfBindingOrchestratingUnitySpecBundle:
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

    specs: list[TheCentralHubsOfBindingOrchestratingUnitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheCentralHubsOfBindingOrchestratingUnitySpec(
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
        "title": "Paper 0 " + "The Central Hubs of Binding: Orchestrating Unity" + " Specs",
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
        "next_source_boundary": "P0R04607",
    }
    return TheCentralHubsOfBindingOrchestratingUnitySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheCentralHubsOfBindingOrchestratingUnitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_central_hubs_of_binding_orchestrating_unity_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheCentralHubsOfBindingOrchestratingUnitySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Central Hubs of Binding: Orchestrating Unity" + " Specs",
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
    bundle: TheCentralHubsOfBindingOrchestratingUnitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_central_hubs_of_binding_orchestrating_unity_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_central_hubs_of_binding_orchestrating_unity_validation_specs_{date_tag}.md"
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
