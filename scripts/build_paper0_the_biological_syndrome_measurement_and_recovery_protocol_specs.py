#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Biological Syndrome Measurement and Recovery Protocol spec builder
"""Promote Paper 0 The Biological Syndrome Measurement and Recovery Protocol records."""

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
    "P0R03076",
    "P0R03077",
    "P0R03078",
    "P0R03079",
    "P0R03080",
    "P0R03081",
    "P0R03082",
    "P0R03083",
    "P0R03084",
    "P0R03085",
    "P0R03086",
    "P0R03087",
    "P0R03088",
    "P0R03089",
    "P0R03090",
    "P0R03091",
    "P0R03092",
    "P0R03093",
    "P0R03094",
    "P0R03095",
    "P0R03096",
    "P0R03097",
    "P0R03098",
)
CLAIM_BOUNDARY = "source-bounded the biological syndrome measurement and recovery protocol source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_biological_syndrome_measurement_and_recovery_protocol.the_biological_syndrome_measurement_and_recovery_protocol": {
        "context_id": "the_biological_syndrome_measurement_and_recovery_protocol",
        "validation_protocol": "paper0.the_biological_syndrome_measurement_and_recovery_protocol.the_biological_syndrome_measurement_and_recovery_protocol",
        "canonical_statement": "The source-bounded component 'The Biological Syndrome Measurement and Recovery Protocol' preserves Paper 0 records P0R03076-P0R03098 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03076:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03077:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03078:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03079:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03080:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03081:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03082:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03083:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03084:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03085:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03086:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03087:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03088:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03089:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03090:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03091:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03092:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03093:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03094:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03095:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03096:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03097:the_biological_syndrome_measurement_and_recovery_protocol",
            "P0R03098:the_biological_syndrome_measurement_and_recovery_protocol",
        ),
        "source_formulae": (
            "P0R03076: The Biological Syndrome Measurement and Recovery Protocol",
            "P0R03077: P0R03077",
            "P0R03078: While a large topological energy gap ($\\Delta \\approx 1.64$ eV) provides passive shielding against thermal excitation, true Quantum Error Correction (QEC) requires an active, dynamic cycle: the continuous measurement of error syndromes via ancilla qubits, followed by the application of corrective unitary operations, all without collapsing the protected logical state. For the SCPN framework to be physically viable at Layer 1, these abstract quantum information processes must be mapped to explicit biochemical machinery.",
            "P0R03079: We formalize the Biological QEC cycle as a three-step thermodynamic and spin-chemical process occurring at the microtubule-cytosol interface.",
            "P0R03080: The Ancilla Qubits: Posner Clusters and Interfacial Water",
            "P0R03081: The logical qubits ($|\\psi_{log}\\rangle$) are encoded in the collective dipole and conformational states of the microtubule (MT) tubulin lattice. To measure errors without destroying this logical state, the system requires ancillary qubits ($|a\\rangle$) that can become entangled with the MT lattice, be measured, and be reset.",
            'P0R03082: The framework identifies the nuclear spins of Phosphorus-31 ($^{31}\\text{P}$) within Calcium Phosphate "Posner" clusters ($\\text{Ca}_9(\\text{PO}_4)_6$), suspended in the QED Coherence Domains (CDs) of interfacial water, as the primary ancilla qubits. Due to their $S=0$ singlet ground state, these clusters possess exceptionally long decoherence times.',
            "P0R03083: The interaction Hamiltonian between the logical MT lattice and the Posner ancilla is mediated by the local magnetic field fluctuations ($\\mathbf{B}_{loc}$) generated by the tubulin dipoles:",
            "P0R03084: $$H_{entangle} = \\sum_{i} \\gamma_P \\mathbf{I}_{ancilla} \\cdot \\mathbf{B}_{loc}(\\sigma_i^{MT})$$",
            'P0R03085: As thermal noise induces a localized phase or bit-flip error in the MT lattice, this interaction entangles the error state with the nuclear spin state of the proximate Posner ancilla, effectively transferring the error "syndrome" into the ancilla without measuring the logical superposition itself.',
            "P0R03086: The Syndrome Measurement: Spin-Dependent Enzymatic Hydrolysis",
            'P0R03087: In standard quantum computing, a macroscopic observer measures the ancilla. In the biological substrate, the "measurement" is a biochemical reaction whose rate is highly sensitive to the quantum state of the ancilla.',
            "P0R03088: This is executed via the Radical Pair Mechanism (RPM) in associated ATPase enzymes (e.g., motor proteins or kinases bound to the MT lattice). When the Posner cluster binds to the enzyme, its nuclear spin state acts as a spin-valve. If the ancilla has absorbed an error syndrome (shifting from a singlet to a triplet-like projection), the spin-orbit coupling drastically alters the singlet-to-triplet interconversion rate of the enzyme's radical pair intermediates.",
            "P0R03089: Therefore, the quantum measurement of the syndrome is transduced into a classical, binary biochemical outcome: the triggering (or suppression) of ATP hydrolysis.",
            "P0R03090: $$M_{syn} \\rightarrow \\Gamma_{ATP} (\\rho_{ancilla})$$",
            "P0R03091: The $\\Psi$-field, acting via Guided Einselection, biases this measurement, ensuring that the collapse of the ancilla state cleanly discretizes the syndrome.",
            "P0R03092: The Recovery Operation: Phonon-Mediated Topological Correction",
            "P0R03093: The final step is the application of the recovery operator ($U_R$). The hydrolysis of ATP releases a highly localized, discrete packet of free energy ($\\approx 0.49$ eV). Because the MT is a prestressed tensegrity lattice, this energy is not lost as diffuse heat; it is channeled as a targeted, high-frequency mechanical vibration-a coherent phonon.",
            "P0R03094: This phonon propagates through the lattice, applying a highly specific mechanical strain to the exact tubulin dimer that suffered the initial error. This mechanical strain effectively applies the required Pauli-$X$ or Pauli-$Z$ gate, forcing the tubulin dimer's conformational state back into alignment with the logical code space:",
            "P0R03095: $$U_R = \\exp\\left(-i \\frac{H_{phonon} t}{\\hbar}\\right)$$",
            "P0R03096: Once the correction is applied, the Posner cluster is released and its spin state is reset via thermalization in the bulk cytosol, ready for the next cycle.",
            "P0R03097: We explicitly define the ancilla (Posner nuclear spins), the measurement (spin-dependent ATP hydrolysis), and the recovery (phonon-mediated conformational shifts), the SCPN formally grounds its MS-QEC architecture in known, falsifiable biochemical reality. The organism is thus revealed to be a continuously running, fault-tolerant quantum Turing machine, powered by metabolism and stabilized by the $\\Psi$-field.",
            "P0R03098: P0R03098",
        ),
        "test_protocols": (
            "preserve The Biological Syndrome Measurement and Recovery Protocol source-accounting boundary",
        ),
        "null_results": (
            "The Biological Syndrome Measurement and Recovery Protocol is not empirical validation evidence",
        ),
        "variables": ("the_biological_syndrome_measurement_and_recovery_protocol",),
        "validation_targets": ("preserve records P0R03076-P0R03098",),
        "null_controls": (
            "the_biological_syndrome_measurement_and_recovery_protocol must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpec:
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
class TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpec, ...]
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


def build_the_biological_syndrome_measurement_and_recovery_protocol_specs(
    source_records: list[dict[str, Any]],
) -> TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle:
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

    specs: list[TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpec(
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
        + "The Biological Syndrome Measurement and Recovery Protocol"
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
        "next_source_boundary": "P0R03099",
    }
    return TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_biological_syndrome_measurement_and_recovery_protocol_specs(
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


def render_report(bundle: TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Biological Syndrome Measurement and Recovery Protocol" + " Specs",
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
    bundle: TheBiologicalSyndromeMeasurementAndRecoveryProtocolSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_biological_syndrome_measurement_and_recovery_protocol_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_biological_syndrome_measurement_and_recovery_protocol_validation_specs_{date_tag}.md"
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
