#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 I. Introduction: The Brain as a Multi-Scale Resonant Transducer spec builder
"""Promote Paper 0 I. Introduction: The Brain as a Multi-Scale Resonant Transducer records."""

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
    "P0R04462",
    "P0R04463",
    "P0R04464",
    "P0R04465",
    "P0R04466",
    "P0R04467",
    "P0R04468",
    "P0R04469",
)
CLAIM_BOUNDARY = "source-bounded i introduction the brain as a multi scale resonant transducer source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "i_introduction_the_brain_as_a_multi_scale_resonant_transducer.i_introduction_the_brain_as_a_multi_scale_resonant_transducer": {
        "context_id": "i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
        "validation_protocol": "paper0.i_introduction_the_brain_as_a_multi_scale_resonant_transducer.i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
        "canonical_statement": "The source-bounded component 'I. Introduction: The Brain as a Multi-Scale Resonant Transducer' preserves Paper 0 records P0R04462-P0R04464 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04462:i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
            "P0R04463:i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
            "P0R04464:i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
        ),
        "source_formulae": (
            "P0R04462: I. Introduction: The Brain as a Multi-Scale Resonant Transducer",
            "P0R04463: The human brain, within the SCPN framework, is not the generator of consciousness (which is fundamental, L13) but its primary transducer, filter, and anchor in the physical domain. It is a complex, adaptive system evolved under the guidance of the Teleological Engine (L8/L15) to resonate with the Psi-field across multiple scales (Domain I and II).",
            "P0R04464: The brain's architecture is optimised to maintain a Quasicritical state (sigma1), maximising its capacity for information processing (HPC), Integrated Information (), and sensitivity to Psi-field coupling via the Informational Coupling Lagrangian (LInformational). The brain's hierarchical structure mirrors the SCPN layers L1-L5, with the Unified Phase Dynamics Equation (UPDE) governing the synchronisation of neural activity across these scales.",
        ),
        "test_protocols": (
            "preserve I. Introduction: The Brain as a Multi-Scale Resonant Transducer source-accounting boundary",
        ),
        "null_results": (
            "I. Introduction: The Brain as a Multi-Scale Resonant Transducer is not empirical validation evidence",
        ),
        "variables": ("i_introduction_the_brain_as_a_multi_scale_resonant_transducer",),
        "validation_targets": ("preserve records P0R04462-P0R04464",),
        "null_controls": (
            "i_introduction_the_brain_as_a_multi_scale_resonant_transducer must remain source-bounded accounting",
        ),
    },
    "i_introduction_the_brain_as_a_multi_scale_resonant_transducer.ii_the_quantum_neural_interface_l1_l2": {
        "context_id": "ii_the_quantum_neural_interface_l1_l2",
        "validation_protocol": "paper0.i_introduction_the_brain_as_a_multi_scale_resonant_transducer.ii_the_quantum_neural_interface_l1_l2",
        "canonical_statement": "The source-bounded component 'II. The Quantum-Neural Interface (L1-L2)' preserves Paper 0 records P0R04465-P0R04466 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04465:ii_the_quantum_neural_interface_l1_l2",
            "P0R04466:ii_the_quantum_neural_interface_l1_l2",
        ),
        "source_formulae": (
            "P0R04465: II. The Quantum-Neural Interface (L1-L2)",
            "P0R04466: The interface between quantum potentiality and classical neural firing occurs at the deepest levels of the neuron, forming the foundation of the Psi-field-brain interaction.",
        ),
        "test_protocols": (
            "preserve II. The Quantum-Neural Interface (L1-L2) source-accounting boundary",
        ),
        "null_results": (
            "II. The Quantum-Neural Interface (L1-L2) is not empirical validation evidence",
        ),
        "variables": ("ii_the_quantum_neural_interface_l1_l2",),
        "validation_targets": ("preserve records P0R04465-P0R04466",),
        "null_controls": (
            "ii_the_quantum_neural_interface_l1_l2 must remain source-bounded accounting",
        ),
    },
    "i_introduction_the_brain_as_a_multi_scale_resonant_transducer.1_the_neuronal_quantum_substrate_l1": {
        "context_id": "1_the_neuronal_quantum_substrate_l1",
        "validation_protocol": "paper0.i_introduction_the_brain_as_a_multi_scale_resonant_transducer.1_the_neuronal_quantum_substrate_l1",
        "canonical_statement": "The source-bounded component '1. The Neuronal Quantum Substrate (L1):' preserves Paper 0 records P0R04467-P0R04469 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04467:1_the_neuronal_quantum_substrate_l1",
            "P0R04468:1_the_neuronal_quantum_substrate_l1",
            "P0R04469:1_the_neuronal_quantum_substrate_l1",
        ),
        "source_formulae": (
            "P0R04467: 1. The Neuronal Quantum Substrate (L1):",
            "P0R04468: L1 dynamics are centered on the neuronal cytoskeleton and the surrounding aqueous environment.",
            "P0R04469: Microtubules (MT) and QEC: The MT lattice forms the primary substrate. The proposed Quantum Error Correction (QEC) mechanism (Energy Gap Delta1.64 eV) protects quantum information within neurons from thermal decoherence, extending coherence timescales into the millisecond range relevant for neural processing. | The Aqueous Substrate (Coherence Domains): Interfacial Water within neurons forms Coherence Domains (CDs), shielding the L1 substrate and facilitating rapid signalling via proton hopping. | Consciousness-Induced Gravitational Decoherence (CIGD): CIGD provides the mechanism for collapse (TCIGD/E). In the brain, this links the integration of information () within a neural ensemble to the objective reduction of superpositions, selecting specific neural states.",
        ),
        "test_protocols": (
            "preserve 1. The Neuronal Quantum Substrate (L1): source-accounting boundary",
        ),
        "null_results": (
            "1. The Neuronal Quantum Substrate (L1): is not empirical validation evidence",
        ),
        "variables": ("1_the_neuronal_quantum_substrate_l1",),
        "validation_targets": ("preserve records P0R04467-P0R04469",),
        "null_controls": (
            "1_the_neuronal_quantum_substrate_l1 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IIntroductionTheBrainAsAMultiScaleResonantTransducerSpec:
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
class IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IIntroductionTheBrainAsAMultiScaleResonantTransducerSpec, ...]
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


def build_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_specs(
    source_records: list[dict[str, Any]],
) -> IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle:
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

    specs: list[IIntroductionTheBrainAsAMultiScaleResonantTransducerSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IIntroductionTheBrainAsAMultiScaleResonantTransducerSpec(
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
        + "I. Introduction: The Brain as a Multi-Scale Resonant Transducer"
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
        "next_source_boundary": "P0R04470",
    }
    return IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_specs(
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


def render_report(bundle: IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "I. Introduction: The Brain as a Multi-Scale Resonant Transducer"
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
    bundle: IIntroductionTheBrainAsAMultiScaleResonantTransducerSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation_specs_{date_tag}.md"
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
