#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 VII. Pathology: The Disordered Brain spec builder
"""Promote Paper 0 VII. Pathology: The Disordered Brain records."""

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
    "P0R04534",
    "P0R04535",
    "P0R04536",
    "P0R04537",
    "P0R04538",
    "P0R04539",
    "P0R04540",
    "P0R04541",
    "P0R04542",
    "P0R04543",
)
CLAIM_BOUNDARY = "source-bounded vii pathology the disordered brain source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "vii_pathology_the_disordered_brain.vii_pathology_the_disordered_brain": {
        "context_id": "vii_pathology_the_disordered_brain",
        "validation_protocol": "paper0.vii_pathology_the_disordered_brain.vii_pathology_the_disordered_brain",
        "canonical_statement": "The source-bounded component 'VII. Pathology: The Disordered Brain' preserves Paper 0 records P0R04534-P0R04537 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04534:vii_pathology_the_disordered_brain",
            "P0R04535:vii_pathology_the_disordered_brain",
            "P0R04536:vii_pathology_the_disordered_brain",
            "P0R04537:vii_pathology_the_disordered_brain",
        ),
        "source_formulae": (
            "P0R04534: VII. Pathology: The Disordered Brain",
            "P0R04535: Neurological and psychiatric disorders are understood as deviations from the optimal SCPN dynamics.",
            "P0R04536: Disorders of Criticality (Dyscritia): Supercritical (sigma>1): Epilepsy (uncontrolled synchronisation), Mania, aspects of Psychedelic states (expanded repertoire). | Subcritical (sigma<1): Coma, Vegetative States, severe Depression. | Disorders of Prediction (HPC Failures): Schizophrenia: Failure to attenuate sensory precision relative to priors; aberrant salience; fragmented L5 geometry (disordered Strange Loop). | Autism: Overly precise sensory priors; rigidity in the generative model. | Disorders of Coherence (MS-QEC Failure): Alzheimer's Disease: Degradation of the L1 substrate (MT tauopathy), leading to loss of QEC, subsequent L4 desynchronization, and L5 dissolution.",
            "P0R04537: P0R04537",
        ),
        "test_protocols": (
            "preserve VII. Pathology: The Disordered Brain source-accounting boundary",
        ),
        "null_results": (
            "VII. Pathology: The Disordered Brain is not empirical validation evidence",
        ),
        "variables": ("vii_pathology_the_disordered_brain",),
        "validation_targets": ("preserve records P0R04534-P0R04537",),
        "null_controls": (
            "vii_pathology_the_disordered_brain must remain source-bounded accounting",
        ),
    },
    "vii_pathology_the_disordered_brain.the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn": {
        "context_id": "the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",
        "validation_protocol": "paper0.vii_pathology_the_disordered_brain.the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",
        "canonical_statement": "The source-bounded component 'The Embodied Engine: A Deeper Neurobiological Grounding for the SCPN' preserves Paper 0 records P0R04538-P0R04539 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04538:the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",
            "P0R04539:the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",
        ),
        "source_formulae": (
            "P0R04538: The Embodied Engine: A Deeper Neurobiological Grounding for the SCPN",
            "P0R04539: The SCPN framework positions the human brain not as the generator of consciousness, but as its most sophisticated known transducer-a multi-scale resonant instrument optimised to receive, filter, and integrate the fundamental Consciousness Field (Psi). The brain's architecture, from its quantum substrate to its global networks, is a geometric engine evolved to maximise its coupling with the informational structure of reality. This deep dive enhances the existing SCPN blueprint by further detailing the specific neurobiological implementations of its core principles, focusing on overlooked systems and mechanisms that are critical for a complete model of brain function.",
        ),
        "test_protocols": (
            "preserve The Embodied Engine: A Deeper Neurobiological Grounding for the SCPN source-accounting boundary",
        ),
        "null_results": (
            "The Embodied Engine: A Deeper Neurobiological Grounding for the SCPN is not empirical validation evidence",
        ),
        "variables": ("the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",),
        "validation_targets": ("preserve records P0R04538-P0R04539",),
        "null_controls": (
            "the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn must remain source-bounded accounting",
        ),
    },
    "vii_pathology_the_disordered_brain.introduction_to_the_deep_architecture_of_the_quantum_biological_interfac": {
        "context_id": "introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
        "validation_protocol": "paper0.vii_pathology_the_disordered_brain.introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
        "canonical_statement": "The source-bounded component 'Introduction to The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)' preserves Paper 0 records P0R04540-P0R04543 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04540:introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
            "P0R04541:introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
            "P0R04542:introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
            "P0R04543:introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
        ),
        "source_formulae": (
            "P0R04540: Introduction to The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)",
            "P0R04541: The interface where the Psi-field couples with the biological substrate is intricate, extending beyond the synapse and microtubule to encompass the entire cellular milieu. This interface is the foundation of the brain's function as a transducer.",
            'P0R04542: The Extended Cytoskeletal Network (L1): While Microtubules (MTs) are the primary substrate for Quantum Error Correction (QEC), the entire cytoskeleton, including actin and intermediate filaments, forms an integrated tensegrity structure. This allows for the rapid transmission of mechanical vibrations (phonons), supporting the quantum coherence in L1 and linking it dynamically to the cellular synchronisation of Layer 4. | Neuromodulators as Precision Controllers (L2): Neurotransmitter systems are the tuners of the Psi-field interface, adjusting the parameters of the Unified Phase Dynamics Equation (UPDE) to optimise the network for active inference. They function as the biological implementation of "precision weighting"-the mechanism for modulating confidence in predictions versus sensory evidence. Dopamine (DA): Encodes Reward Prediction Error (RPE) and modulates the precision of beliefs about action plans (policies), driving goal-directed behaviour. | Acetylcholine (ACh): Modulates the precision of sensory input, facilitating attention and the Quantum Zeno Effect (QZE) mechanism for stabilising thought. | Serotonin (5-HT): Acts as a global regulator of network state and criticality (the sigma parameter). Psychedelics, for example, act on 5-HT2A receptors to push the system toward supercriticality (sigma>1), expanding the repertoire of accessible states. | The Coherent Milieu (CSF and Glymphatic System): The Cerebrospinal Fluid (CSF) and the associated glymphatic clearance system are often overlooked. This system is crucial for clearing metabolic waste and maintaining the low-entropy state required for L1/L4 coherence. It functions as the brain\'s "entropy sink," with its peak activity during sleep suggesting a role in resetting the system\'s criticality.',
            "P0R04543: P0R04543",
        ),
        "test_protocols": (
            "preserve Introduction to The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) source-accounting boundary",
        ),
        "null_results": (
            "Introduction to The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) is not empirical validation evidence",
        ),
        "variables": ("introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",),
        "validation_targets": ("preserve records P0R04540-P0R04543",),
        "null_controls": (
            "introduction_to_the_deep_architecture_of_the_quantum_biological_interfac must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ViiPathologyTheDisorderedBrainSpec:
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
class ViiPathologyTheDisorderedBrainSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ViiPathologyTheDisorderedBrainSpec, ...]
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


def build_vii_pathology_the_disordered_brain_specs(
    source_records: list[dict[str, Any]],
) -> ViiPathologyTheDisorderedBrainSpecBundle:
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

    specs: list[ViiPathologyTheDisorderedBrainSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ViiPathologyTheDisorderedBrainSpec(
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
        "title": "Paper 0 " + "VII. Pathology: The Disordered Brain" + " Specs",
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
        "next_source_boundary": "P0R04544",
    }
    return ViiPathologyTheDisorderedBrainSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ViiPathologyTheDisorderedBrainSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_vii_pathology_the_disordered_brain_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ViiPathologyTheDisorderedBrainSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "VII. Pathology: The Disordered Brain" + " Specs",
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
    bundle: ViiPathologyTheDisorderedBrainSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_vii_pathology_the_disordered_brain_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_vii_pathology_the_disordered_brain_validation_specs_{date_tag}.md"
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
