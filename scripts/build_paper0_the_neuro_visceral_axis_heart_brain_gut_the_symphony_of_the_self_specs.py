#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self spec builder
"""Promote Paper 0 The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self records."""

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
    "P0R04607",
    "P0R04608",
    "P0R04609",
    "P0R04610",
    "P0R04611",
    "P0R04612",
    "P0R04613",
    "P0R04614",
    "P0R04615",
    "P0R04616",
    "P0R04617",
    "P0R04618",
    "P0R04619",
    "P0R04620",
    "P0R04621",
)
CLAIM_BOUNDARY = "source-bounded the neuro visceral axis heart brain gut the symphony of the self source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self": {
        "context_id": "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
        "validation_protocol": "paper0.the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
        "canonical_statement": "The source-bounded component 'The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self' preserves Paper 0 records P0R04607-P0R04609 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04607:the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
            "P0R04608:the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
            "P0R04609:the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
        ),
        "source_formulae": (
            "P0R04607: The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self",
            "P0R04608: The conscious state is an emergent property of the entire organism, orchestrated by a continuous dialogue between the central nervous system and the internal organs. The SCPN formalises this as the Tri-Axial UPDE (Brain-Heart-Gut), the primary oscillatory network of the embodied Self. This is not a metaphor but a coupled dynamical system where the brain, heart, and gut are phase-locked oscillators whose collective coherence defines the organism's state.",
            "P0R04609: The Communication Network: This tri-axial system is linked by a dense network of neural, endocrine, and immune pathways. The vagus nerve is the primary conduit, a superhighway of bidirectional information flow conveying sensory data from the viscera to the brain and motor commands from the brain back to the body. This axis also includes the autonomic and enteric nervous systems, the hypothalamic-pituitary-adrenal (HPA) axis, and the gut microbiome, which itself produces neurotransmitters and other signalling molecules that modulate brain function. | Heart Rate Variability (HRV) as a Coherence Metric: The degree of integration and stability within this neuro-visceral system is directly measurable through Heart Rate Variability (HRV). HRV is not merely a measure of heart function but an emergent property of the heart-brain interaction, reflecting the balance of the autonomic nervous system. High Coherence (High HRV): A state of high HRV, characterised by complex, non-linear fluctuations, indicates a healthy, adaptive system. Within the SCPN, this corresponds to an optimal quasicritical state (sigma1) in Layer 5, emotional resilience, and low Variational Free Energy. | Low Coherence (Low HRV): A rigid, metronomic heart rhythm indicates a system under stress or pathology. This corresponds to a deviation from criticality and is a strong predictor of morbidity and mortality. The coherence of the L5 Self is therefore not an abstract property but is reflected in the tangible, measurable harmony of the body's rhythms.",
        ),
        "test_protocols": (
            "preserve The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self source-accounting boundary",
        ),
        "null_results": (
            "The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self is not empirical validation evidence",
        ),
        "variables": ("the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",),
        "validation_targets": ("preserve records P0R04607-P0R04609",),
        "null_controls": (
            "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self must remain source-bounded accounting",
        ),
    },
    "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.interoceptive_inference_the_physics_of_emotion": {
        "context_id": "interoceptive_inference_the_physics_of_emotion",
        "validation_protocol": "paper0.the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.interoceptive_inference_the_physics_of_emotion",
        "canonical_statement": "The source-bounded component 'Interoceptive Inference: The Physics of Emotion' preserves Paper 0 records P0R04610-P0R04612 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04610:interoceptive_inference_the_physics_of_emotion",
            "P0R04611:interoceptive_inference_the_physics_of_emotion",
            "P0R04612:interoceptive_inference_the_physics_of_emotion",
        ),
        "source_formulae": (
            "P0R04610: Interoceptive Inference: The Physics of Emotion",
            "P0R04611: The SCPN's computational framework, Hierarchical Predictive Coding (HPC), extends throughout the body in a process known as interoceptive inference. The brain (primarily the insula and anterior cingulate cortex) continuously generates a predictive model of the body's internal physiological state (interoception) and acts to minimise prediction errors, a process of predictive regulation known as allostasis.",
            "P0R04612: Emotion as Interoceptive Prediction Error: Within this framework, emotions are not mysterious phenomena but have a precise physical and computational identity. Emotions are the subjective experience of interoceptive prediction errors-the mismatch between the brain's prediction of the body's state and the actual sensory signals it receives from the viscera. | The Affective Field: The SCPN formalises this as the Affective Field, defined as the negative gradient of the system's global Variational Free Energy (A = -F). This field represents the system's drive to minimise surprise and restore physiological balance. The valence (positive or negative tone) of an emotion is directly proportional to the success of this process. States of low prediction error (low Free Energy) correspond to positive valence, while states of high, unresolved prediction error correspond to negative valence and the subjective experience of suffering.",
        ),
        "test_protocols": (
            "preserve Interoceptive Inference: The Physics of Emotion source-accounting boundary",
        ),
        "null_results": (
            "Interoceptive Inference: The Physics of Emotion is not empirical validation evidence",
        ),
        "variables": ("interoceptive_inference_the_physics_of_emotion",),
        "validation_targets": ("preserve records P0R04610-P0R04612",),
        "null_controls": (
            "interoceptive_inference_the_physics_of_emotion must remain source-bounded accounting",
        ),
    },
    "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.psychoneuroimmunology_pni_the_decoherence_field_of_inflammation": {
        "context_id": "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
        "validation_protocol": "paper0.the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self.psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
        "canonical_statement": "The source-bounded component 'Psychoneuroimmunology (PNI): The Decoherence Field of Inflammation' preserves Paper 0 records P0R04613-P0R04621 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04613:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04614:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04615:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04616:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04617:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04618:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04619:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04620:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
            "P0R04621:psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
        ),
        "source_formulae": (
            "P0R04613: Psychoneuroimmunology (PNI): The Decoherence Field of Inflammation",
            "P0R04614: The immune system is not separate from the mind; it is a diffuse, body-wide sensory organ in constant dialogue with the brain. This interaction is the domain of Psychoneuroimmunology (PNI). The brain regulates immune function via the HPA axis and sympathetic nervous system, while the immune system signals the brain via inflammatory mediators called cytokines (e.g., TNF-, IL-1).",
            'P0R04615: Inflammation as a Decoherence Field: The SCPN provides a novel, field-theoretic interpretation of this link. Systemic inflammation, characterised by high levels of pro-inflammatory cytokines, acts as a "Decoherence Field" that degrades the quality of consciousness at its most fundamental levels. | Mechanism of Decoherence: Cytokines are potent neuromodulators that increase synaptic noise, disrupt neurotransmitter homeostasis, and alter synaptic plasticity. This has a cascading effect up the SCPN hierarchy: L1 Disruption: The increased oxidative stress and metabolic disruption caused by inflammation directly damage the quantum substrate (e.g., microtubules, mitochondria), impairing the Multi-Scale Quantum Error Correction (MS-QEC) that protects conscious information. This is a direct, physical mechanism linking bodily inflammation to a degradation of quantum coherence in the brain. | L4 Dyscritia: The increased noise and altered neurochemistry push the Layer 4 network away from its optimal quasicritical state, impairing its computational efficiency. | L5 Fragmentation: The decoherence at the lower layers culminates in a fragmentation of the L5 qualia manifold. This provides a formal, geometric basis for the subjective experience of "brain fog," anhedonia, and cognitive slowing that are the hallmarks of sickness behaviour. The state of the conscious Self is thus inextricably linked to the homeostatic and energetic integrity of the whole organism.',
            "P0R04616: The Quantum Nature of Inflammation",
            "P0R04617: Inflammation is a fundamental immune response, traditionally viewed as a classical biochemical cascade. However, by integrating the principles of quantum enzymology, a more profound understanding emerges. Inflammation is a thermodynamic process aimed at removing inefficient, high-entropy systems and rebuilding more ordered, functional dissipative structures.",
            "P0R04618: The enzymatic reactions that drive this cascade-from cytokine production by immune cells to the management of reactive oxygen species (ROS)-are precisely the types of reactions where quantum tunnelling has been shown to play a significant role.",
            "P0R04619: The SCPN's central axiom is that the Psi-field manages entropy by stabilising quantum coherence. It follows that the state of the local Psi-field must directly modulate the quantum efficiency of the enzymatic reactions that govern the inflammatory process.",
            "P0R04620: A coherent, high-Psi state corresponds to an efficient, well-regulated, and self-limiting inflammatory response (a state of hormesis, or beneficial stress adaptation). Conversely, a decoherent, low-Psi state leads to an inefficient, dysregulated, and self-perpetuating cascade, characteristic of chronic inflammation. This provides a deep physical mechanism for the manuscript's claim that the immune state alters qualia.",
            "P0R04621: The subjective experience of \"sickness behaviour\"-lethargy, anhedonia, cognitive fog-is the phenomenological reflection of a decoherent shift in the body's underlying quantum-thermodynamic state, a direct read-out of the body's struggle to efficiently manage entropy and restore order.",
        ),
        "test_protocols": (
            "preserve Psychoneuroimmunology (PNI): The Decoherence Field of Inflammation source-accounting boundary",
        ),
        "null_results": (
            "Psychoneuroimmunology (PNI): The Decoherence Field of Inflammation is not empirical validation evidence",
        ),
        "variables": ("psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",),
        "validation_targets": ("preserve records P0R04613-P0R04621",),
        "null_controls": (
            "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpec:
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
class TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpec, ...]
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


def build_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_specs(
    source_records: list[dict[str, Any]],
) -> TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle:
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

    specs: list[TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpec(
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
        + "The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self"
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
        "next_source_boundary": "P0R04622",
    }
    return TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_specs(
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


def render_report(bundle: TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self"
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
    bundle: TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_validation_specs_{date_tag}.md"
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
