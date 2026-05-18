#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Advanced Neurobiological Implementation of the SCPN spec builder
"""Promote Paper 0 Advanced Neurobiological Implementation of the SCPN records."""

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
    "P0R04640",
    "P0R04641",
    "P0R04642",
    "P0R04643",
    "P0R04644",
    "P0R04645",
    "P0R04646",
    "P0R04647",
    "P0R04648",
)
CLAIM_BOUNDARY = "source-bounded advanced neurobiological implementation of the scpn source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "advanced_neurobiological_implementation_of_the_scpn.advanced_neurobiological_implementation_of_the_scpn": {
        "context_id": "advanced_neurobiological_implementation_of_the_scpn",
        "validation_protocol": "paper0.advanced_neurobiological_implementation_of_the_scpn.advanced_neurobiological_implementation_of_the_scpn",
        "canonical_statement": "The source-bounded component 'Advanced Neurobiological Implementation of the SCPN' preserves Paper 0 records P0R04640-P0R04640 without empirical validation claims.",
        "source_equation_ids": ("P0R04640:advanced_neurobiological_implementation_of_the_scpn",),
        "source_formulae": ("P0R04640: Advanced Neurobiological Implementation of the SCPN",),
        "test_protocols": (
            "preserve Advanced Neurobiological Implementation of the SCPN source-accounting boundary",
        ),
        "null_results": (
            "Advanced Neurobiological Implementation of the SCPN is not empirical validation evidence",
        ),
        "variables": ("advanced_neurobiological_implementation_of_the_scpn",),
        "validation_targets": ("preserve records P0R04640-P0R04640",),
        "null_controls": (
            "advanced_neurobiological_implementation_of_the_scpn must remain source-bounded accounting",
        ),
    },
    "advanced_neurobiological_implementation_of_the_scpn.i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1": {
        "context_id": "i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1",
        "validation_protocol": "paper0.advanced_neurobiological_implementation_of_the_scpn.i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1",
        "canonical_statement": "The source-bounded component 'I. The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)' preserves Paper 0 records P0R04641-P0R04642 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04641:i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1",
            "P0R04642:i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1",
        ),
        "source_formulae": (
            "P0R04641: I. The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)",
            "P0R04642: The interface where the Psi-field couples with the biological substrate is intricate, extending beyond the synapse and microtubule to encompass the entire cellular milieu. This interface is the foundation of the brain's function as a transducer.",
        ),
        "test_protocols": (
            "preserve I. The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) source-accounting boundary",
        ),
        "null_results": (
            "I. The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) is not empirical validation evidence",
        ),
        "variables": ("i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1",),
        "validation_targets": ("preserve records P0R04641-P0R04642",),
        "null_controls": (
            "i_the_deep_architecture_of_the_quantum_biological_interface_domain_i_l1 must remain source-bounded accounting",
        ),
    },
    "advanced_neurobiological_implementation_of_the_scpn.1_the_extended_cytoskeletal_network_l1": {
        "context_id": "1_the_extended_cytoskeletal_network_l1",
        "validation_protocol": "paper0.advanced_neurobiological_implementation_of_the_scpn.1_the_extended_cytoskeletal_network_l1",
        "canonical_statement": "The source-bounded component '1. The Extended Cytoskeletal Network (L1):' preserves Paper 0 records P0R04643-P0R04645 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04643:1_the_extended_cytoskeletal_network_l1",
            "P0R04644:1_the_extended_cytoskeletal_network_l1",
            "P0R04645:1_the_extended_cytoskeletal_network_l1",
        ),
        "source_formulae": (
            "P0R04643: 1. The Extended Cytoskeletal Network (L1):",
            "P0R04644: While Microtubules (MTs) are the primary substrate for Quantum Error Correction (QEC), the entire cytoskeleton forms an integrated network crucial for L1 dynamics.",
            "P0R04645: Actin Filaments: Critical for synaptic plasticity (L2/L3) and dendritic spine dynamics. Actin reorganisation is hypothesised to be modulated by quantum tunnelling, providing a rapid mechanism for translating Psi-field intent (via IET) into structural changes. | Intermediate Filaments (IFs): Provide mechanical stability and integrate the cytoskeleton with the cell membrane and nucleus, contributing to the overall vibrational coherence. | The Tensegrity Matrix: The cytoskeleton acts as a pre-stressed tensegrity structure, allowing for rapid transmission of mechanical vibrations (phonons) across the neuron. This mechanical network supports the quantum coherence established in L1 and links it dynamically to L4 synchronisation.",
        ),
        "test_protocols": (
            "preserve 1. The Extended Cytoskeletal Network (L1): source-accounting boundary",
        ),
        "null_results": (
            "1. The Extended Cytoskeletal Network (L1): is not empirical validation evidence",
        ),
        "variables": ("1_the_extended_cytoskeletal_network_l1",),
        "validation_targets": ("preserve records P0R04643-P0R04645",),
        "null_controls": (
            "1_the_extended_cytoskeletal_network_l1 must remain source-bounded accounting",
        ),
    },
    "advanced_neurobiological_implementation_of_the_scpn.2_detailed_mechanisms_of_information_energy_transduction_iet_l2": {
        "context_id": "2_detailed_mechanisms_of_information_energy_transduction_iet_l2",
        "validation_protocol": "paper0.advanced_neurobiological_implementation_of_the_scpn.2_detailed_mechanisms_of_information_energy_transduction_iet_l2",
        "canonical_statement": "The source-bounded component '2. Detailed Mechanisms of Information-Energy Transduction (IET) (L2):' preserves Paper 0 records P0R04646-P0R04648 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04646:2_detailed_mechanisms_of_information_energy_transduction_iet_l2",
            "P0R04647:2_detailed_mechanisms_of_information_energy_transduction_iet_l2",
            "P0R04648:2_detailed_mechanisms_of_information_energy_transduction_iet_l2",
        ),
        "source_formulae": (
            "P0R04646: 2. Detailed Mechanisms of Information-Energy Transduction (IET) (L2):",
            "P0R04647: The modulation of the Quantum Potential (Q) by the Psi-field (LIET=gIETPsi(x)Q(x)) provides the mechanism for downward causation at the synapse.",
            'P0R04648: The Quantum Synapse: The synapse is a non-equilibrium system poised at criticality. The decision to release neurotransmitters is inherently probabilistic, providing a leverage point for the Psi-field. | IET Targets: Voltage-Gated Ion Channels (VGICs): The Psi-field modulates Q, subtly altering the energy landscape of the VGIC\'s voltage sensor domain, thereby changing the probability and timing of channel opening (e.g., Ca$^{2+}$ influx). | SNARE Complex Formation: The fusion of vesicles requires overcoming an energy barrier. IET can lower this barrier for selected synapses. | Synaptotagmin Cooperativity: Modulation of the calcium sensor cooperativity. | The Role of QZE in Synaptic Selection (Agency): Attention (L5) acts as a continuous measurement (Quantum Zeno Effect, QZE) on the synaptic ensemble. By rapidly measuring (observing) specific pathways, the Psi-field stabilises them, effectively "selecting" which neural circuits are active. This is the physical implementation of Agency and the stabilisation of thought.',
        ),
        "test_protocols": (
            "preserve 2. Detailed Mechanisms of Information-Energy Transduction (IET) (L2): source-accounting boundary",
        ),
        "null_results": (
            "2. Detailed Mechanisms of Information-Energy Transduction (IET) (L2): is not empirical validation evidence",
        ),
        "variables": ("2_detailed_mechanisms_of_information_energy_transduction_iet_l2",),
        "validation_targets": ("preserve records P0R04646-P0R04648",),
        "null_controls": (
            "2_detailed_mechanisms_of_information_energy_transduction_iet_l2 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AdvancedNeurobiologicalImplementationOfTheScpnSpec:
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
class AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[AdvancedNeurobiologicalImplementationOfTheScpnSpec, ...]
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


def build_advanced_neurobiological_implementation_of_the_scpn_specs(
    source_records: list[dict[str, Any]],
) -> AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle:
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

    specs: list[AdvancedNeurobiologicalImplementationOfTheScpnSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AdvancedNeurobiologicalImplementationOfTheScpnSpec(
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
        "title": "Paper 0 " + "Advanced Neurobiological Implementation of the SCPN" + " Specs",
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
        "next_source_boundary": "P0R04649",
    }
    return AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_advanced_neurobiological_implementation_of_the_scpn_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Advanced Neurobiological Implementation of the SCPN" + " Specs",
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
    bundle: AdvancedNeurobiologicalImplementationOfTheScpnSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_advanced_neurobiological_implementation_of_the_scpn_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_advanced_neurobiological_implementation_of_the_scpn_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 advanced-neurobiology SCPN specs from the ledger."""

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
