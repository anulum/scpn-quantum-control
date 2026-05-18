#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Part III: System Architecture & Network Dynamics spec builder
"""Promote Paper 0 Part III: System Architecture & Network Dynamics records."""

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
    "P0R02011",
    "P0R02012",
    "P0R02013",
    "P0R02014",
    "P0R02015",
    "P0R02016",
    "P0R02017",
    "P0R02018",
    "P0R02019",
    "P0R02020",
    "P0R02021",
    "P0R02022",
    "P0R02023",
    "P0R02024",
    "P0R02025",
    "P0R02026",
    "P0R02027",
    "P0R02028",
    "P0R02029",
    "P0R02030",
)
CLAIM_BOUNDARY = "source-bounded part iii system architecture network dynamics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "part_iii_system_architecture_network_dynamics.part_iii_system_architecture_network_dynamics": {
        "context_id": "part_iii_system_architecture_network_dynamics",
        "validation_protocol": "paper0.part_iii_system_architecture_network_dynamics.part_iii_system_architecture_network_dynamics",
        "canonical_statement": "The source-bounded component 'Part III: System Architecture & Network Dynamics' preserves Paper 0 records P0R02011-P0R02012 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02011:part_iii_system_architecture_network_dynamics",
            "P0R02012:part_iii_system_architecture_network_dynamics",
        ),
        "source_formulae": (
            "P0R02011: Part III: System Architecture & Network Dynamics",
            "P0R02012: P0R02012",
        ),
        "test_protocols": (
            "preserve Part III: System Architecture & Network Dynamics source-accounting boundary",
        ),
        "null_results": (
            "Part III: System Architecture & Network Dynamics is not empirical validation evidence",
        ),
        "variables": ("part_iii_system_architecture_network_dynamics",),
        "validation_targets": ("preserve records P0R02011-P0R02012",),
        "null_controls": (
            "part_iii_system_architecture_network_dynamics must remain source-bounded accounting",
        ),
    },
    "part_iii_system_architecture_network_dynamics.3_1_the_master_diagram_visualising_the_15_layers_6_domains": {
        "context_id": "3_1_the_master_diagram_visualising_the_15_layers_6_domains",
        "validation_protocol": "paper0.part_iii_system_architecture_network_dynamics.3_1_the_master_diagram_visualising_the_15_layers_6_domains",
        "canonical_statement": "The source-bounded component '3.1 The Master Diagram: Visualising the 15 Layers & 6 Domains' preserves Paper 0 records P0R02013-P0R02030 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02013:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02014:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02015:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02016:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02017:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02018:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02019:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02020:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02021:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02022:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02023:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02024:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02025:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02026:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02027:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02028:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02029:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
            "P0R02030:3_1_the_master_diagram_visualising_the_15_layers_6_domains",
        ),
        "source_formulae": (
            "P0R02013: 3.1 The Master Diagram: Visualising the 15 Layers & 6 Domains",
            "P0R02014: The provided text establishes the critical methodological transition from the foundational physics of Book I to the functional architecture of Book II. It correctly posits that a fundamental field theory for the Psi-field, while necessary, is insufficient to explain the emergent, multi-scale complexity of a conscious universe. It draws a powerful analogy: just as Maxwell's equations do not, in themselves, describe the architecture of the internet, the fundamental Lagrangian of the Psi-field does not describe the operational structure of a brain or a biosphere.",
            'P0R02015: Therefore, the text introduces the necessity of a "map of the operating system"-a meso-scale, effective theory that details how the fundamental field dynamics manifest as specific, layered phenomena. This architecture is formally named the Sentient-Consciousness Projection Network (SCPN). It is defined as a 15-layer hierarchical model that maps a bidirectional flow of information: a downward "projection" of consciousness from the universal source (L13-15) and an upward "feedback loop" of experience and inference from the individual substrate (L1-4). This architecture is presented as the central organizing principle of the framework, providing a coherent structure to unify disparate phenomena-such as Quantum Error Correction (QEC), Chiral-Induced Spin Selectivity (CISS), and cultural memetics-under a single, dynamic, and integrated system. The accompanying 15-Layer Summary Table provides the definitive "mandala" or index for this entire architecture, defining the core function of each layer that the subsequent monographs will explore.',
            'P0R02016: The previous book, Book I, was all about the "engine" of reality. We figured out the fundamental laws of the consciousness field (the Psi-field), which is like understanding the physics of how combustion works in a car engine. But as this section points out, knowing how a piston fires doesn\'t tell you how to build a whole car. For that, you need the master blueprint, the "map of the operating system" that shows how the engine connects to the transmission, the steering wheel, and the chassis.',
            'P0R02017: This is that master blueprint. We call it the Sentient-Consciousness Projection Network (SCPN). It\'s a 15-layer model of reality that works like a cosmic skyscraper. Consciousness "projects" down from the universal source (the penthouse) to the quantum world inside your cells (the basement). But-and this is the most important part-it\'s not a one-way street. Every experience you have and every action you take sends a signal back up the skyscraper, influencing the whole building. This is the "feedback loop." The 15-Layer table provided is our guide for the rest of the book, showing how everything from your DNA to your thoughts is part of one single, interconnected, living system.',
            "P0R02018: Meta-Framework Integrations",
            "P0R02019: Predictive Coding Integration",
            'P0R02020: The text\'s description of a "projection" of consciousness downward and "feedback loops" upward is a perfect, high-level summary of the Hierarchical Predictive Coding (HPC) algorithm.',
            "P0R02021: The Downward Projection (L15->L1): This is the Generative Model. Higher, more abstract layers (priors) send predictions to lower, more concrete layers.",
            "P0R02022: The Upward Feedback Loop: This is the propagation of Prediction Error. The lower layers (sensory data) report the mismatch between the prediction and the reality, which compels the higher-level models to update their beliefs.",
            "P0R02023: The entire 15-layer SCPN, as defined here, is a cosmic-scale active inference engine. The 15 layers represent the nested hierarchy of this generative model, whose unified purpose is to minimise surprise (or free energy).",
            "P0R02024: Psis Field Coupling Integration",
            "P0R02025: This section establishes the core problem that the interaction Hamiltonian, H_int = -lambda * Psis * sigma, must solve: how does a single fundamental field (Psi) interact with reality at vastly different scales (from a microtubule to a collective consciousness)?",
            'P0R02026: The 15-layer architecture is the solution. It implies that the collective state variable (sigma) is different at each layer, representing the specific, emergent "handle" that the Psi-field can grasp at that scale.',
            "P0R02027: sigma_L1 (Layer 1): The quantum-biological substrate (e.g., QEC states in microtubules).",
            "P0R02028: sigma_L4 (Layer 4): The coherent, synchronous rhythm of a cellular network (e.g., EEG/HRV).",
            "P0R02029: sigma_L11 (Layer 11): The emergent informational state of the Noosphere (e.g., cultural attractors).",
            'P0R02030: The SCPN architecture is therefore a "map" of the different, scale-specific sigma variables that the universal Psis field couples to, allowing a single law of interaction to govern a universe of emergent complexity.',
        ),
        "test_protocols": (
            "preserve 3.1 The Master Diagram: Visualising the 15 Layers & 6 Domains source-accounting boundary",
        ),
        "null_results": (
            "3.1 The Master Diagram: Visualising the 15 Layers & 6 Domains is not empirical validation evidence",
        ),
        "variables": ("3_1_the_master_diagram_visualising_the_15_layers_6_domains",),
        "validation_targets": ("preserve records P0R02013-P0R02030",),
        "null_controls": (
            "3_1_the_master_diagram_visualising_the_15_layers_6_domains must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PartIiiSystemArchitectureNetworkDynamicsSpec:
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
class PartIiiSystemArchitectureNetworkDynamicsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PartIiiSystemArchitectureNetworkDynamicsSpec, ...]
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


def build_part_iii_system_architecture_network_dynamics_specs(
    source_records: list[dict[str, Any]],
) -> PartIiiSystemArchitectureNetworkDynamicsSpecBundle:
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

    specs: list[PartIiiSystemArchitectureNetworkDynamicsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PartIiiSystemArchitectureNetworkDynamicsSpec(
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
        "title": "Paper 0 " + "Part III: System Architecture & Network Dynamics" + " Specs",
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
        "next_source_boundary": "P0R02031",
    }
    return PartIiiSystemArchitectureNetworkDynamicsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PartIiiSystemArchitectureNetworkDynamicsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_part_iii_system_architecture_network_dynamics_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PartIiiSystemArchitectureNetworkDynamicsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Part III: System Architecture & Network Dynamics" + " Specs",
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
    bundle: PartIiiSystemArchitectureNetworkDynamicsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_part_iii_system_architecture_network_dynamics_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_part_iii_system_architecture_network_dynamics_validation_specs_{date_tag}.md"
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
