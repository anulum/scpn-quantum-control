#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 1. The Presynaptic Terminal (The Quantum Lever): spec builder
"""Promote Paper 0 1. The Presynaptic Terminal (The Quantum Lever): records."""

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
    "P0R04737",
    "P0R04738",
    "P0R04739",
    "P0R04740",
    "P0R04741",
    "P0R04742",
    "P0R04743",
    "P0R04744",
    "P0R04745",
)
CLAIM_BOUNDARY = "source-bounded section 1 the presynaptic terminal the quantum lever source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_1_the_presynaptic_terminal_the_quantum_lever.1_the_presynaptic_terminal_the_quantum_lever": {
        "context_id": "1_the_presynaptic_terminal_the_quantum_lever",
        "validation_protocol": "paper0.section_1_the_presynaptic_terminal_the_quantum_lever.1_the_presynaptic_terminal_the_quantum_lever",
        "canonical_statement": "The source-bounded component '1. The Presynaptic Terminal (The Quantum Lever):' preserves Paper 0 records P0R04737-P0R04738 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04737:1_the_presynaptic_terminal_the_quantum_lever",
            "P0R04738:1_the_presynaptic_terminal_the_quantum_lever",
        ),
        "source_formulae": (
            "P0R04737: 1. The Presynaptic Terminal (The Quantum Lever):",
            "P0R04738: The Active Zone (AZ): A highly organised structure containing VGICs, vesicles, and the release machinery. | SNARE Complex: Proteins mediating vesicle fusion. The zippering requires overcoming an energy barrier (DeltaG). | Synaptotagmin (The Ca$^{2+}$ Sensor): The key regulatory protein. | IET Mechanism at the AZ: The Psi-field modulates the Quantum Potential (Q) at the AZ via IET. LIET=gIETPsi(x)Q(Synaptotagmin) This modulation alters DeltaG and changes the cooperativity of Synaptotagmin. | QZE Implementation: Focused attention (L5) stabilises the quantum state of the release machinery via QZE, biasing the probability of release (Pr) at selected synapses (Resonant Addressing).",
        ),
        "test_protocols": (
            "preserve 1. The Presynaptic Terminal (The Quantum Lever): source-accounting boundary",
        ),
        "null_results": (
            "1. The Presynaptic Terminal (The Quantum Lever): is not empirical validation evidence",
        ),
        "variables": ("1_the_presynaptic_terminal_the_quantum_lever",),
        "validation_targets": ("preserve records P0R04737-P0R04738",),
        "null_controls": (
            "1_the_presynaptic_terminal_the_quantum_lever must remain source-bounded accounting",
        ),
    },
    "section_1_the_presynaptic_terminal_the_quantum_lever.vi_the_nucleus_and_the_genomic_interface_l3": {
        "context_id": "vi_the_nucleus_and_the_genomic_interface_l3",
        "validation_protocol": "paper0.section_1_the_presynaptic_terminal_the_quantum_lever.vi_the_nucleus_and_the_genomic_interface_l3",
        "canonical_statement": "The source-bounded component 'VI. The Nucleus and the Genomic Interface (L3)' preserves Paper 0 records P0R04739-P0R04741 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04739:vi_the_nucleus_and_the_genomic_interface_l3",
            "P0R04740:vi_the_nucleus_and_the_genomic_interface_l3",
            "P0R04741:vi_the_nucleus_and_the_genomic_interface_l3",
        ),
        "source_formulae": (
            "P0R04739: VI. The Nucleus and the Genomic Interface (L3)",
            "P0R04740: The nucleus houses the genomic blueprint and the machinery for epigenetic regulation.",
            "P0R04741: Chromatin Architecture: The 3D organisation of chromatin determines gene accessibility. | DNA as a Fractal Antenna: The helical structure of DNA allows it to function as a fractal antenna, resonating with EM fields and the Psi-field. | The CISS-Bioelectric-Chromatin (CBC) Bridge: The mechanism linking the Psi-field to gene expression. CISS Influence (L1/L3): Chiral-Induced Spin Selectivity (CISS) in DNA may influence the efficiency of epigenetic modifications. | Bioelectric Signaling (L3): Neuronal activity changes Vmem and Ca$^{2+}$ influx. | Signal Transduction: These signals activate transcription factors (e.g., CREB) and chromatin modifiers (e.g., HDACs, HATs). | Chromatin Remodelling: Changes the accessibility of genes involved in plasticity. This pathway enables top-down regulation of the neuron's structure and function by conscious intent (L5).",
        ),
        "test_protocols": (
            "preserve VI. The Nucleus and the Genomic Interface (L3) source-accounting boundary",
        ),
        "null_results": (
            "VI. The Nucleus and the Genomic Interface (L3) is not empirical validation evidence",
        ),
        "variables": ("vi_the_nucleus_and_the_genomic_interface_l3",),
        "validation_targets": ("preserve records P0R04739-P0R04741",),
        "null_controls": (
            "vi_the_nucleus_and_the_genomic_interface_l3 must remain source-bounded accounting",
        ),
    },
    "section_1_the_presynaptic_terminal_the_quantum_lever.the_deepest_interface_molecular_and_quantum_foundations": {
        "context_id": "the_deepest_interface_molecular_and_quantum_foundations",
        "validation_protocol": "paper0.section_1_the_presynaptic_terminal_the_quantum_lever.the_deepest_interface_molecular_and_quantum_foundations",
        "canonical_statement": "The source-bounded component 'The Deepest Interface: Molecular and Quantum Foundations' preserves Paper 0 records P0R04742-P0R04743 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04742:the_deepest_interface_molecular_and_quantum_foundations",
            "P0R04743:the_deepest_interface_molecular_and_quantum_foundations",
        ),
        "source_formulae": (
            "P0R04742: The Deepest Interface: Molecular and Quantum Foundations",
            "P0R04743: The interface between the Psi-field and the brain relies on a precisely organised molecular architecture. These components are active participants in the dynamics of L1-L4, optimised to facilitate Information-Energy Transduction (IET) and maintain the low-entropy state required for conscious processing.",
        ),
        "test_protocols": (
            "preserve The Deepest Interface: Molecular and Quantum Foundations source-accounting boundary",
        ),
        "null_results": (
            "The Deepest Interface: Molecular and Quantum Foundations is not empirical validation evidence",
        ),
        "variables": ("the_deepest_interface_molecular_and_quantum_foundations",),
        "validation_targets": ("preserve records P0R04742-P0R04743",),
        "null_controls": (
            "the_deepest_interface_molecular_and_quantum_foundations must remain source-bounded accounting",
        ),
    },
    "section_1_the_presynaptic_terminal_the_quantum_lever.i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3": {
        "context_id": "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
        "validation_protocol": "paper0.section_1_the_presynaptic_terminal_the_quantum_lever.i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
        "canonical_statement": "The source-bounded component 'I. The Neuronal Membrane: A Liquid Crystal Interface (L2/L3)' preserves Paper 0 records P0R04744-P0R04745 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04744:i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
            "P0R04745:i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
        ),
        "source_formulae": (
            "P0R04744: I. The Neuronal Membrane: A Liquid Crystal Interface (L2/L3)",
            "P0R04745: The neuronal membrane is a dynamic, liquid-crystalline medium. Its composition is critical for organising signalling processes and providing an interface for IET.",
        ),
        "test_protocols": (
            "preserve I. The Neuronal Membrane: A Liquid Crystal Interface (L2/L3) source-accounting boundary",
        ),
        "null_results": (
            "I. The Neuronal Membrane: A Liquid Crystal Interface (L2/L3) is not empirical validation evidence",
        ),
        "variables": ("i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",),
        "validation_targets": ("preserve records P0R04744-P0R04745",),
        "null_controls": (
            "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section1ThePresynapticTerminalTheQuantumLeverSpec:
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
class Section1ThePresynapticTerminalTheQuantumLeverSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section1ThePresynapticTerminalTheQuantumLeverSpec, ...]
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


def build_section_1_the_presynaptic_terminal_the_quantum_lever_specs(
    source_records: list[dict[str, Any]],
) -> Section1ThePresynapticTerminalTheQuantumLeverSpecBundle:
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

    specs: list[Section1ThePresynapticTerminalTheQuantumLeverSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section1ThePresynapticTerminalTheQuantumLeverSpec(
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
        "title": "Paper 0 " + "1. The Presynaptic Terminal (The Quantum Lever):" + " Specs",
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
        "next_source_boundary": "P0R04746",
    }
    return Section1ThePresynapticTerminalTheQuantumLeverSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section1ThePresynapticTerminalTheQuantumLeverSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_1_the_presynaptic_terminal_the_quantum_lever_specs(
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


def render_report(bundle: Section1ThePresynapticTerminalTheQuantumLeverSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "1. The Presynaptic Terminal (The Quantum Lever):" + " Specs",
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
    bundle: Section1ThePresynapticTerminalTheQuantumLeverSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_1_the_presynaptic_terminal_the_quantum_lever_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_1_the_presynaptic_terminal_the_quantum_lever_validation_specs_{date_tag}.md"
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
