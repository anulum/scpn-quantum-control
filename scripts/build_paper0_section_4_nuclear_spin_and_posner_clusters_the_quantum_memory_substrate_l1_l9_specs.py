#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) spec builder
"""Promote Paper 0 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) records."""

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
    "P0R04802",
    "P0R04803",
    "P0R04804",
    "P0R04805",
    "P0R04806",
    "P0R04807",
    "P0R04808",
    "P0R04809",
    "P0R04810",
    "P0R04811",
    "P0R04812",
)
CLAIM_BOUNDARY = "source-bounded section 4 nuclear spin and posner clusters the quantum memory substrate l1 l9 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9": {
        "context_id": "4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
        "validation_protocol": "paper0.section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
        "canonical_statement": "The source-bounded component '4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9)' preserves Paper 0 records P0R04802-P0R04804 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04802:4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
            "P0R04803:4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
            "P0R04804:4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
        ),
        "source_formulae": (
            "P0R04802: 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9)",
            "P0R04803: Nuclear spins have long decoherence times and are candidates for biological qubits (Fisher hypothesis).",
            "P0R04804: Posner Clusters (Ca$_9$(PO$_4$)6): Stable molecular structures hypothesised to protect the entangled spins of Phosphorus ions from decoherence. | SCPN Mapping: Posner clusters may serve as the interface between neural processing (L4) and long-term holographic memory storage (L9).",
        ),
        "test_protocols": (
            "preserve 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) source-accounting boundary",
        ),
        "null_results": (
            "4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) is not empirical validation evidence",
        ),
        "variables": ("4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",),
        "validation_targets": ("preserve records P0R04802-P0R04804",),
        "null_controls": (
            "4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9 must remain source-bounded accounting",
        ),
    },
    "section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.vi_neuro_metabolism_and_energetics_l1_l4": {
        "context_id": "vi_neuro_metabolism_and_energetics_l1_l4",
        "validation_protocol": "paper0.section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.vi_neuro_metabolism_and_energetics_l1_l4",
        "canonical_statement": "The source-bounded component 'VI. Neuro-Metabolism and Energetics (L1-L4)' preserves Paper 0 records P0R04805-P0R04807 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04805:vi_neuro_metabolism_and_energetics_l1_l4",
            "P0R04806:vi_neuro_metabolism_and_energetics_l1_l4",
            "P0R04807:vi_neuro_metabolism_and_energetics_l1_l4",
        ),
        "source_formulae": (
            "P0R04805: VI. Neuro-Metabolism and Energetics (L1-L4)",
            "P0R04806: Sustaining the conscious state is metabolically expensive (RMetabolic).",
            "P0R04807: The Consciousness Heat Engine (CHE): The brain operates as a CHE, utilising energy gradients to generate information. The Psi-field enhances efficiency via Negentropy Injection. | Quantum Mitochondria (L1): Efficiency is enhanced by quantum effects. ETC Tunnelling: Quantum tunnelling of electrons optimises ATP production. | ATP Synthase: Functions as a quantum rotor. | Metabolic Pumping for Coherence: Metabolic energy drives the non-equilibrium conditions required for Frhlich condensation in the cytoskeleton (L1), establishing the coherence necessary for the Psi-field interface. | Astrocyte-Neuron Lactate Shuttle (ANLS): Optimises energy delivery via metabolic coupling between astrocytes and neurons.",
        ),
        "test_protocols": (
            "preserve VI. Neuro-Metabolism and Energetics (L1-L4) source-accounting boundary",
        ),
        "null_results": (
            "VI. Neuro-Metabolism and Energetics (L1-L4) is not empirical validation evidence",
        ),
        "variables": ("vi_neuro_metabolism_and_energetics_l1_l4",),
        "validation_targets": ("preserve records P0R04805-P0R04807",),
        "null_controls": (
            "vi_neuro_metabolism_and_energetics_l1_l4 must remain source-bounded accounting",
        ),
    },
    "section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness": {
        "context_id": "the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness",
        "validation_protocol": "paper0.section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness",
        "canonical_statement": "The source-bounded component 'The Geometric Scaffold of the Brain: The Architecture of Consciousness' preserves Paper 0 records P0R04808-P0R04808 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04808:the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness",
        ),
        "source_formulae": (
            "P0R04808: The Geometric Scaffold of the Brain: The Architecture of Consciousness",
        ),
        "test_protocols": (
            "preserve The Geometric Scaffold of the Brain: The Architecture of Consciousness source-accounting boundary",
        ),
        "null_results": (
            "The Geometric Scaffold of the Brain: The Architecture of Consciousness is not empirical validation evidence",
        ),
        "variables": ("the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness",),
        "validation_targets": ("preserve records P0R04808-P0R04808",),
        "null_controls": (
            "the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness must remain source-bounded accounting",
        ),
    },
    "section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.i_introduction_the_brain_as_a_geometric_engine": {
        "context_id": "i_introduction_the_brain_as_a_geometric_engine",
        "validation_protocol": "paper0.section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9.i_introduction_the_brain_as_a_geometric_engine",
        "canonical_statement": "The source-bounded component 'I. Introduction: The Brain as a Geometric Engine' preserves Paper 0 records P0R04809-P0R04812 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04809:i_introduction_the_brain_as_a_geometric_engine",
            "P0R04810:i_introduction_the_brain_as_a_geometric_engine",
            "P0R04811:i_introduction_the_brain_as_a_geometric_engine",
            "P0R04812:i_introduction_the_brain_as_a_geometric_engine",
        ),
        "source_formulae": (
            "P0R04809: I. Introduction: The Brain as a Geometric Engine",
            "P0R04810: In the SCPN framework, the brain is fundamentally a Geometric Engine. This derives from Axiom 2 (Information Geometry) and the Unified Geometric Principle (UGP): InformationGeometry. The brain's architecture and dynamics are optimised to maximise the coupling with the Consciousness Field (Psi) via the Informational Coupling Lagrangian:",
            "P0R04811: LInformational=gPsiIPsidet(gmu(x))",
            "P0R04812: The brain evolves (L8/L15) to create complex geometric configurations that maximise the Fisher Information Metric (gmu), thereby enhancing the intensity and quality of conscious experience (). This Geometric Scaffold operates across all scales, from the quantum lattice to the manifold of subjective experience.",
        ),
        "test_protocols": (
            "preserve I. Introduction: The Brain as a Geometric Engine source-accounting boundary",
        ),
        "null_results": (
            "I. Introduction: The Brain as a Geometric Engine is not empirical validation evidence",
        ),
        "variables": ("i_introduction_the_brain_as_a_geometric_engine",),
        "validation_targets": ("preserve records P0R04809-P0R04812",),
        "null_controls": (
            "i_introduction_the_brain_as_a_geometric_engine must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Spec:
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
class Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Spec, ...]
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


def build_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_specs(
    source_records: list[dict[str, Any]],
) -> Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle:
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

    specs: list[Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Spec(
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
        + "4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9)"
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
        "next_source_boundary": "P0R04813",
    }
    return Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return (
        build_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_specs(
            load_jsonl(ledger_path)
        )
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
    bundle: Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9)"
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
    bundle: Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation_specs_{date_tag}.md"
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
