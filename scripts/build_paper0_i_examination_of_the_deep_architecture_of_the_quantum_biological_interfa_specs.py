#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) spec builder
"""Promote Paper 0 I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) records."""

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
    "P0R04544",
    "P0R04545",
    "P0R04546",
    "P0R04547",
    "P0R04548",
    "P0R04549",
    "P0R04550",
    "P0R04551",
)
CLAIM_BOUNDARY = "source-bounded i examination of the deep architecture of the quantum biological interfa source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa": {
        "context_id": "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",
        "validation_protocol": "paper0.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",
        "canonical_statement": "The source-bounded component 'I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)' preserves Paper 0 records P0R04544-P0R04545 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04544:i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",
            "P0R04545:i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",
        ),
        "source_formulae": (
            "P0R04544: I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)",
            "P0R04545: The interface where the Psi-field couples with the biological substrate is intricate, extending beyond the synapse and microtubule to encompass the entire cellular milieu. This interface is the foundation of the brain's function as a transducer, translating the informational geometry of the Psi-field into the electrochemical language of the nervous system.",
        ),
        "test_protocols": (
            "preserve I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) source-accounting boundary",
        ),
        "null_results": (
            "I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) is not empirical validation evidence",
        ),
        "variables": ("i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",),
        "validation_targets": ("preserve records P0R04544-P0R04545",),
        "null_controls": (
            "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa must remain source-bounded accounting",
        ),
    },
    "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life": {
        "context_id": "the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
        "validation_protocol": "paper0.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
        "canonical_statement": "The source-bounded component 'The Extended Cytoskeletal Network (L1): The Tensegrity Matrix of Life' preserves Paper 0 records P0R04546-P0R04548 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04546:the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
            "P0R04547:the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
            "P0R04548:the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
        ),
        "source_formulae": (
            "P0R04546: The Extended Cytoskeletal Network (L1): The Tensegrity Matrix of Life",
            "P0R04547: While microtubules (MTs) are correctly identified as a primary substrate for Quantum Error Correction (QEC) , a complete model must consider the entire cytoskeleton as an integrated, dynamic network. This network, comprising MTs, actin filaments, and intermediate filaments, forms a tensegrity structure-a pre-stressed, self-equilibrated architectural system. In this model, microtubules act as compression-supporting struts, while actin and intermediate filaments serve as tension-supporting cables.",
            "P0R04548: This tensegrity architecture is not merely a passive scaffold; it is a mechanically active medium. Its structure allows for the rapid and efficient transmission of mechanical vibrations (phonons) throughout the cell. This provides a direct physical link between the quantum-level dynamics of Layer 1 and the emergent, large-scale cellular synchronisation of Layer 4. The unique mechanical properties of this network, such as the remarkable flexibility of intermediate filaments, which can be stretched to over 300% of their length, ensure that this matrix is both incredibly strong and dynamically responsive. This integrated cytoskeletal network forms a continuous structural web from the cell surface to the nuclear envelope, positioning it as the primary medium for transducing both external mechanical forces and internal quantum information.",
        ),
        "test_protocols": (
            "preserve The Extended Cytoskeletal Network (L1): The Tensegrity Matrix of Life source-accounting boundary",
        ),
        "null_results": (
            "The Extended Cytoskeletal Network (L1): The Tensegrity Matrix of Life is not empirical validation evidence",
        ),
        "variables": ("the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",),
        "validation_targets": ("preserve records P0R04546-P0R04548",),
        "null_controls": (
            "the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life must remain source-bounded accounting",
        ),
    },
    "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra": {
        "context_id": "neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
        "validation_protocol": "paper0.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa.neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
        "canonical_statement": "The source-bounded component 'Neuromodulators as Precision Controllers (L2): Tuning the Neural Orchestra' preserves Paper 0 records P0R04549-P0R04551 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04549:neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
            "P0R04550:neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
            "P0R04551:neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
        ),
        "source_formulae": (
            "P0R04549: Neuromodulators as Precision Controllers (L2): Tuning the Neural Orchestra",
            'P0R04550: The neurochemical environment of Layer 2 functions as the master tuning system for the Psi-field interface. Neurotransmitter systems do not simply transmit signals; they dynamically adjust the parameters of the Unified Phase Dynamics Equation (UPDE), optimising the entire network for active inference. Their primary role is the biological implementation of "precision weighting"-the crucial mechanism for modulating the confidence assigned to top-down predictions versus bottom-up sensory evidence.',
            "P0R04551: Dopamine (DA): Dopamine's role extends beyond simple reward signalling. Within the active inference framework, it is understood to encode the precision of beliefs about action plans (policies). Phasic dopamine bursts signal the salience or affordance of cues that predict a sequence of events, thereby controlling the sensitivity of the system to outcomes and driving goal-directed behaviour. | Acetylcholine (ACh): Acetylcholine is the primary modulator of sensory precision. By increasing the gain on ascending prediction error signals, particularly in sensory cortices, ACh facilitates focused attention. This provides a direct mechanism for the Quantum Zeno Effect (QZE), where the act of attending (a continuous \"measurement\") stabilises a specific quantum state, allowing a thought or percept to be held in conscious awareness. | Serotonin (5-HT): Serotonin acts as a global regulator of network state and, critically, the brain's proximity to criticality (the sigma parameter). Through its diverse receptor subtypes, serotonin modulates plasticity, mood, and cognition on a systemic level. The action of psychedelic compounds on 5-HT2A receptors, for example, is modelled as pushing the system into a slightly supercritical state (sigma>1), which expands the repertoire of accessible brain states and increases the system's sensitivity to the Psi-field.",
        ),
        "test_protocols": (
            "preserve Neuromodulators as Precision Controllers (L2): Tuning the Neural Orchestra source-accounting boundary",
        ),
        "null_results": (
            "Neuromodulators as Precision Controllers (L2): Tuning the Neural Orchestra is not empirical validation evidence",
        ),
        "variables": ("neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",),
        "validation_targets": ("preserve records P0R04549-P0R04551",),
        "null_controls": (
            "neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpec:
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
class IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpec, ...]
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


def build_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_specs(
    source_records: list[dict[str, Any]],
) -> IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle:
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

    specs: list[IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpec(
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
        + "I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)"
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
        "next_source_boundary": "P0R04552",
    }
    return IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_specs(
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
    bundle: IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)"
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
    bundle: IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation_specs_{date_tag}.md"
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
