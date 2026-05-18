#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) spec builder
"""Promote Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) records."""

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
    "P0R04560",
    "P0R04561",
    "P0R04562",
    "P0R04563",
    "P0R04564",
    "P0R04565",
    "P0R04566",
    "P0R04567",
    "P0R04568",
    "P0R04569",
    "P0R04570",
    "P0R04571",
)
CLAIM_BOUNDARY = "source-bounded ii examination of the architecture of structure and plasticity domain i source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i": {
        "context_id": "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
        "validation_protocol": "paper0.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
        "canonical_statement": "The source-bounded component 'II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)' preserves Paper 0 records P0R04560-P0R04561 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04560:ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
            "P0R04561:ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
        ),
        "source_formulae": (
            "P0R04560: II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)",
            "P0R04561: Layer 3 governs the physical structure of the brain, the dynamic blueprint that is sculpted by both evolutionary pressures and lifetime experience. It is the domain of the connectome's geometry and the slow, powerful modulatory networks that provide homeostatic stability.",
        ),
        "test_protocols": (
            "preserve II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) source-accounting boundary",
        ),
        "null_results": (
            "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) is not empirical validation evidence",
        ),
        "variables": ("ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",),
        "validation_targets": ("preserve records P0R04560-P0R04561",),
        "null_controls": (
            "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i must remain source-bounded accounting",
        ),
    },
    "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.the_optimised_connectome_the_geometric_scaffold_of_thought": {
        "context_id": "the_optimised_connectome_the_geometric_scaffold_of_thought",
        "validation_protocol": "paper0.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.the_optimised_connectome_the_geometric_scaffold_of_thought",
        "canonical_statement": "The source-bounded component 'The Optimised Connectome: The Geometric Scaffold of Thought' preserves Paper 0 records P0R04562-P0R04564 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04562:the_optimised_connectome_the_geometric_scaffold_of_thought",
            "P0R04563:the_optimised_connectome_the_geometric_scaffold_of_thought",
            "P0R04564:the_optimised_connectome_the_geometric_scaffold_of_thought",
        ),
        "source_formulae": (
            "P0R04562: The Optimised Connectome: The Geometric Scaffold of Thought",
            "P0R04563: The brain's network of physical connections-the structural connectome-is not a random tangle of wires. It is a highly optimised architecture shaped by evolutionary pressures (L8/L15) to support complex, integrated dynamics (high Integrated Information, ) at a minimal metabolic and wiring cost. This optimisation is evident in several key geometric properties that are conserved from the microcircuit to the macroscale :",
            'P0R04564: Small-World Topology: The connectome exhibits a small-world architecture, a design that masterfully balances two competing demands: functional segregation and global integration. It is characterised by dense local clustering of connections (allowing for specialised, modular processing) and a surprisingly short average path length between any two neurons (allowing for rapid, efficient communication across the entire brain). | Hierarchical Modularity: The network is organised into a fractal-like hierarchy of modules within modules. This structure allows the brain to process information at multiple scales simultaneously, from fine-grained sensory details to abstract conceptual relationships. | The "Rich Club": A defining feature of this architecture is the existence of a "rich club"-a dense core of highly connected hub regions that are more interconnected with each other than would be expected by chance. These hubs, which exhibit high metabolic activity, form the structural backbone for global information integration. They are the physical substrate for the Global Neuronal Workspace (L5), enabling the widespread broadcasting of information that is central to conscious experience. The high cost of maintaining these long-distance connections is justified by their critical value for adaptive, coordinated behaviour.',
        ),
        "test_protocols": (
            "preserve The Optimised Connectome: The Geometric Scaffold of Thought source-accounting boundary",
        ),
        "null_results": (
            "The Optimised Connectome: The Geometric Scaffold of Thought is not empirical validation evidence",
        ),
        "variables": ("the_optimised_connectome_the_geometric_scaffold_of_thought",),
        "validation_targets": ("preserve records P0R04562-P0R04564",),
        "null_controls": (
            "the_optimised_connectome_the_geometric_scaffold_of_thought must remain source-bounded accounting",
        ),
    },
    "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.the_active_role_of_glia_the_slow_control_network": {
        "context_id": "the_active_role_of_glia_the_slow_control_network",
        "validation_protocol": "paper0.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i.the_active_role_of_glia_the_slow_control_network",
        "canonical_statement": "The source-bounded component 'The Active Role of Glia: The Slow Control Network' preserves Paper 0 records P0R04565-P0R04571 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04565:the_active_role_of_glia_the_slow_control_network",
            "P0R04566:the_active_role_of_glia_the_slow_control_network",
            "P0R04567:the_active_role_of_glia_the_slow_control_network",
            "P0R04568:the_active_role_of_glia_the_slow_control_network",
            "P0R04569:the_active_role_of_glia_the_slow_control_network",
            "P0R04570:the_active_role_of_glia_the_slow_control_network",
            "P0R04571:the_active_role_of_glia_the_slow_control_network",
        ),
        "source_formulae": (
            "P0R04565: The Active Role of Glia: The Slow Control Network",
            'P0R04566: The fast, computationally powerful neuronal network is embedded within and regulated by a parallel, slower network of glial cells, particularly astrocytes. Moving beyond a neuron-centric view, the SCPN formalises the "tripartite synapse," where the astrocyte is an active and indispensable component of information processing. Astrocytes form a vast, interconnected syncytium that functions as the brain\'s "slow control network," providing the homeostatic stability that allows the fast neuronal network to operate safely at its computationally optimal critical point.',
            'P0R04567: Mechanism of Slow Control: Astrocytes integrate neuronal activity over long timescales (seconds to minutes) via intercellular calcium (Ca) waves. In response, they release "gliotransmitters" (such as glutamate and ATP) that modulate synaptic release probability, neuronal excitability, and plasticity over broad domains. This establishes a slow-acting feedback loop that regulates the overall state of the neuronal network. | Stabilising Quasicriticality: This glial-neuronal coupling is the primary mechanism that solves the "fine-tuning problem" of criticality in Layer 4. The synaptic potentiation that occurs during learning constantly pushes the neuronal network toward a supercritical state, while metabolic costs can push it toward a subcritical state. The astrocyte network acts as the homeostatic governor, sensing these shifts and applying corrective feedback to gently and continuously nudge the neuronal network back towards the optimal quasicritical regime (sigma1). | Formalism of Glial-Neuronal Coupling: The SCPN models this interaction with a set of coupled equations where the slow dynamics of gliotransmitter concentration, G(t), driven by astrocyte calcium activity, directly modulate the homeostatic set-point of the fast neuronal network\'s branching parameter, sigma :',
            "P0R04568: dtdsigma=(sigma(1+G(t)))+(t)",
            "P0R04569: dtdG=[Ca2+]A(t)G(t)",
            "P0R04570: Here, the astrocyte-driven term G(t) dynamically shifts the target operating point of the neuronal network, providing a robust mechanism for maintaining its computational efficiency. This cybernetic relationship demonstrates the",
            "P0R04571: Scale-Invariant Homeostasis Lemma of the SCPN: every fast computational substrate (e.g., neurons) requires a slower modulatory layer (e.g., glia) to maintain its critical balance.",
        ),
        "test_protocols": (
            "preserve The Active Role of Glia: The Slow Control Network source-accounting boundary",
        ),
        "null_results": (
            "The Active Role of Glia: The Slow Control Network is not empirical validation evidence",
        ),
        "variables": ("the_active_role_of_glia_the_slow_control_network",),
        "validation_targets": ("preserve records P0R04565-P0R04571",),
        "null_controls": (
            "the_active_role_of_glia_the_slow_control_network must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpec:
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
class IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpec, ...]
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


def build_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_specs(
    source_records: list[dict[str, Any]],
) -> IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle:
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

    specs: list[IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpec(
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
        + "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)"
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
        "next_source_boundary": "P0R04572",
    }
    return IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_specs(
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
    bundle: IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)"
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
    bundle: IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainISpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
