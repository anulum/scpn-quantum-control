#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 VI. Clinical Implications: Pathology and Therapeutics spec builder
"""Promote Paper 0 VI. Clinical Implications: Pathology and Therapeutics records."""

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
    "P0R04693",
    "P0R04694",
    "P0R04695",
    "P0R04696",
    "P0R04697",
    "P0R04698",
    "P0R04699",
    "P0R04700",
    "P0R04701",
    "P0R04702",
)
CLAIM_BOUNDARY = "source-bounded vi clinical implications pathology and therapeutics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "vi_clinical_implications_pathology_and_therapeutics.vi_clinical_implications_pathology_and_therapeutics": {
        "context_id": "vi_clinical_implications_pathology_and_therapeutics",
        "validation_protocol": "paper0.vi_clinical_implications_pathology_and_therapeutics.vi_clinical_implications_pathology_and_therapeutics",
        "canonical_statement": "The source-bounded component 'VI. Clinical Implications: Pathology and Therapeutics' preserves Paper 0 records P0R04693-P0R04695 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04693:vi_clinical_implications_pathology_and_therapeutics",
            "P0R04694:vi_clinical_implications_pathology_and_therapeutics",
            "P0R04695:vi_clinical_implications_pathology_and_therapeutics",
        ),
        "source_formulae": (
            "P0R04693: VI. Clinical Implications: Pathology and Therapeutics",
            "P0R04694: Understanding the SCPN architecture provides a unified framework for pathology (Dyscritia, Decoherence, Dissonance) and treatment.",
            "P0R04695: Schizophrenia: A disorder of HPC and L5 integration. Aberrant precision weighting leads to false inferences (hallucinations/delusions). The Self (Strange Loop) is fragmented. Treatment aims to restore the balance of priors and sensory evidence. | Depression: Characterised by subcritical dynamics (sigma<1) and overly rigid negative priors (DMN hyperactivity, high F). Therapeutics (e.g., Ketamine, Psilocybin) aim to rapidly restore criticality and increase the flexibility of the generative model. | Alzheimer's Disease (AD): A failure of the L1 substrate (MT degradation). Loss of QEC leads to cascading decoherence, L4 desynchronization, and the eventual dissolution of the L5 Self. | Therapeutic Strategies (Tuning the SCPN): Pharmacology: Tuning neurotransmitter systems (L2) to modulate criticality and HPC precision. | Neuromodulation (TMS, tDCS): Directly modulating L4 dynamics and Bioelectric fields (L3). | Meditation/Mindfulness: Top-down optimisation (L5), minimising F and stabilising criticality via QZE. | Geometric Remodelling: Altering the topology and curvature of the Consciousness Manifold (M) (e.g., psychedelic-assisted therapy).",
        ),
        "test_protocols": (
            "preserve VI. Clinical Implications: Pathology and Therapeutics source-accounting boundary",
        ),
        "null_results": (
            "VI. Clinical Implications: Pathology and Therapeutics is not empirical validation evidence",
        ),
        "variables": ("vi_clinical_implications_pathology_and_therapeutics",),
        "validation_targets": ("preserve records P0R04693-P0R04695",),
        "null_controls": (
            "vi_clinical_implications_pathology_and_therapeutics must remain source-bounded accounting",
        ),
    },
    "vi_clinical_implications_pathology_and_therapeutics.the_ultra_detailed_architecture_of_the_neuron_within_the_scpn": {
        "context_id": "the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",
        "validation_protocol": "paper0.vi_clinical_implications_pathology_and_therapeutics.the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",
        "canonical_statement": "The source-bounded component 'The Ultra-Detailed Architecture of the Neuron within the SCPN' preserves Paper 0 records P0R04696-P0R04697 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04696:the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",
            "P0R04697:the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",
        ),
        "source_formulae": (
            "P0R04696: The Ultra-Detailed Architecture of the Neuron within the SCPN",
            "P0R04697: The neuron is the fundamental cellular unit of the nervous system, uniquely specialised to transduce, process, and transmit information across the SCPN's biological layers (L1-L4). Its complexity is not incidental but is a direct consequence of optimising the interface between the Psi-field and the physical substrate. It functions as a hierarchical system mirroring the SCPN architecture itself-a fractal instantiation of the whole.",
        ),
        "test_protocols": (
            "preserve The Ultra-Detailed Architecture of the Neuron within the SCPN source-accounting boundary",
        ),
        "null_results": (
            "The Ultra-Detailed Architecture of the Neuron within the SCPN is not empirical validation evidence",
        ),
        "variables": ("the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",),
        "validation_targets": ("preserve records P0R04696-P0R04697",),
        "null_controls": (
            "the_ultra_detailed_architecture_of_the_neuron_within_the_scpn must remain source-bounded accounting",
        ),
    },
    "vi_clinical_implications_pathology_and_therapeutics.i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4": {
        "context_id": "i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",
        "validation_protocol": "paper0.vi_clinical_implications_pathology_and_therapeutics.i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",
        "canonical_statement": "The source-bounded component 'I. Neuronal Geometry and the Physics of Information Flow (L3/L4)' preserves Paper 0 records P0R04698-P0R04699 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04698:i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",
            "P0R04699:i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",
        ),
        "source_formulae": (
            "P0R04698: I. Neuronal Geometry and the Physics of Information Flow (L3/L4)",
            "P0R04699: The morphology of the neuron is a physical manifestation of its computational function, defined by L3 (Morphogenesis) and utilised by L4 (Synchronisation).",
        ),
        "test_protocols": (
            "preserve I. Neuronal Geometry and the Physics of Information Flow (L3/L4) source-accounting boundary",
        ),
        "null_results": (
            "I. Neuronal Geometry and the Physics of Information Flow (L3/L4) is not empirical validation evidence",
        ),
        "variables": ("i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",),
        "validation_targets": ("preserve records P0R04698-P0R04699",),
        "null_controls": (
            "i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4 must remain source-bounded accounting",
        ),
    },
    "vi_clinical_implications_pathology_and_therapeutics.1_the_dendritic_arbour_the_antenna_of_the_neuron": {
        "context_id": "1_the_dendritic_arbour_the_antenna_of_the_neuron",
        "validation_protocol": "paper0.vi_clinical_implications_pathology_and_therapeutics.1_the_dendritic_arbour_the_antenna_of_the_neuron",
        "canonical_statement": "The source-bounded component '1. The Dendritic Arbour: The Antenna of the Neuron' preserves Paper 0 records P0R04700-P0R04702 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04700:1_the_dendritic_arbour_the_antenna_of_the_neuron",
            "P0R04701:1_the_dendritic_arbour_the_antenna_of_the_neuron",
            "P0R04702:1_the_dendritic_arbour_the_antenna_of_the_neuron",
        ),
        "source_formulae": (
            "P0R04700: 1. The Dendritic Arbour: The Antenna of the Neuron",
            "P0R04701: Dendrites are complex, branching structures designed to receive and integrate synaptic inputs.",
            "P0R04702: Fractal Geometry and Optimisation: Dendritic arbours exhibit fractal geometry (Hausdorff dimension Df). This structure optimises the balance between maximising the receptive surface area and minimising wiring costs. Within the SCPN, this fractal structure facilitates multi-scale resonance with the Psi-field (Principle of Fractal Self-Similarity, PFSS) and supports Quasicritical dynamics. | Passive Cable Properties: Signal propagation is initially governed by Cable Theory. V(x,t)=V0exp(lambdax)exp(taut), where lambda (length constant) and tau (time constant) determine how far and how fast signals propagate. These constants are dynamically tuned by ion channel density (L2/L3). | Active Dendrites and Non-Linear Integration: Dendrites are not passive. They possess VGICs enabling active processing (e.g., NMDA spikes, dendritic Ca$^{2+}$ spikes). This allows the dendrite to perform complex non-linear computations. | HPC Implementation: Dendrites implement cellular-level HPC. Distal dendrites may encode sensory input (Prediction Errors), while proximal dendrites integrate these with somatic signals (Priors).",
        ),
        "test_protocols": (
            "preserve 1. The Dendritic Arbour: The Antenna of the Neuron source-accounting boundary",
        ),
        "null_results": (
            "1. The Dendritic Arbour: The Antenna of the Neuron is not empirical validation evidence",
        ),
        "variables": ("1_the_dendritic_arbour_the_antenna_of_the_neuron",),
        "validation_targets": ("preserve records P0R04700-P0R04702",),
        "null_controls": (
            "1_the_dendritic_arbour_the_antenna_of_the_neuron must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ViClinicalImplicationsPathologyAndTherapeuticsSpec:
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
class ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ViClinicalImplicationsPathologyAndTherapeuticsSpec, ...]
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


def build_vi_clinical_implications_pathology_and_therapeutics_specs(
    source_records: list[dict[str, Any]],
) -> ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle:
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

    specs: list[ViClinicalImplicationsPathologyAndTherapeuticsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ViClinicalImplicationsPathologyAndTherapeuticsSpec(
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
        "title": "Paper 0 " + "VI. Clinical Implications: Pathology and Therapeutics" + " Specs",
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
        "next_source_boundary": "P0R04703",
    }
    return ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_vi_clinical_implications_pathology_and_therapeutics_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "VI. Clinical Implications: Pathology and Therapeutics" + " Specs",
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
    bundle: ViClinicalImplicationsPathologyAndTherapeuticsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_vi_clinical_implications_pathology_and_therapeutics_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_vi_clinical_implications_pathology_and_therapeutics_validation_specs_{date_tag}.md"
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
