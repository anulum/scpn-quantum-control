#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. spec builder
"""Promote Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. records."""

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
    "P0R05314",
    "P0R05315",
    "P0R05316",
    "P0R05317",
    "P0R05318",
    "P0R05319",
    "P0R05320",
    "P0R05321",
    "P0R05322",
)
CLAIM_BOUNDARY = "source-bounded the quantum engine layers 1 2 we begin with the quantum biological layer source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer": {
        "context_id": "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",
        "validation_protocol": "paper0.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",
        "canonical_statement": "The source-bounded component 'The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system.' preserves Paper 0 records P0R05314-P0R05314 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05314:the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",
        ),
        "source_formulae": (
            "P0R05314: The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system.",
        ),
        "test_protocols": (
            "preserve The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. source-accounting boundary",
        ),
        "null_results": (
            "The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. is not empirical validation evidence",
        ),
        "variables": ("the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",),
        "validation_targets": ("preserve records P0R05314-P0R05314",),
        "null_controls": (
            "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer must remain source-bounded accounting",
        ),
    },
    "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.the_quantum_classical_bridge_selection_and_amplification": {
        "context_id": "the_quantum_classical_bridge_selection_and_amplification",
        "validation_protocol": "paper0.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.the_quantum_classical_bridge_selection_and_amplification",
        "canonical_statement": "The source-bounded component 'The Quantum-Classical Bridge: Selection and Amplification' preserves Paper 0 records P0R05315-P0R05316 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05315:the_quantum_classical_bridge_selection_and_amplification",
            "P0R05316:the_quantum_classical_bridge_selection_and_amplification",
        ),
        "source_formulae": (
            "P0R05315: The Quantum-Classical Bridge: Selection and Amplification",
            "P0R05316: The transition from the quantum potentiality of L1 to the classical dynamics of L4 requires two distinct mechanisms: Selection (the emergence of classical reality) and Amplification.",
        ),
        "test_protocols": (
            "preserve The Quantum-Classical Bridge: Selection and Amplification source-accounting boundary",
        ),
        "null_results": (
            "The Quantum-Classical Bridge: Selection and Amplification is not empirical validation evidence",
        ),
        "variables": ("the_quantum_classical_bridge_selection_and_amplification",),
        "validation_targets": ("preserve records P0R05315-P0R05316",),
        "null_controls": (
            "the_quantum_classical_bridge_selection_and_amplification must remain source-bounded accounting",
        ),
    },
    "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.i_guided_einselection_the_emergence_of_classicality": {
        "context_id": "i_guided_einselection_the_emergence_of_classicality",
        "validation_protocol": "paper0.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.i_guided_einselection_the_emergence_of_classicality",
        "canonical_statement": "The source-bounded component 'I. Guided Einselection (The Emergence of Classicality)' preserves Paper 0 records P0R05317-P0R05319 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05317:i_guided_einselection_the_emergence_of_classicality",
            "P0R05318:i_guided_einselection_the_emergence_of_classicality",
            "P0R05319:i_guided_einselection_the_emergence_of_classicality",
        ),
        "source_formulae": (
            "P0R05317: I. Guided Einselection (The Emergence of Classicality)",
            'P0R05318: The transition from quantum superposition (L1) to stable classical states (L2) is formalised by Quantum Darwinism and Environment-Induced Superselection (Einselection). Classical reality emerges as the environment continuously "measures" the system, selecting stable "pointer states."',
            'P0R05319: Within the SCPN, the Psi-field acts as the primary "environment" (E). The interaction Hamiltonian (H_Int=L_IET) determines which states are selected. Crucially, the Psi-field, guided by the HPC generative model (L5 intent), actively selects the pointer basis. States aligned with the generative model are preferentially stabilised (Guided Einselection) via the Quantum Zeno Effect (QZE) and proliferated, while dissonant states decohere.',
        ),
        "test_protocols": (
            "preserve I. Guided Einselection (The Emergence of Classicality) source-accounting boundary",
        ),
        "null_results": (
            "I. Guided Einselection (The Emergence of Classicality) is not empirical validation evidence",
        ),
        "variables": ("i_guided_einselection_the_emergence_of_classicality",),
        "validation_targets": ("preserve records P0R05317-P0R05319",),
        "null_controls": (
            "i_guided_einselection_the_emergence_of_classicality must remain source-bounded accounting",
        ),
    },
    "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti": {
        "context_id": "ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
        "validation_protocol": "paper0.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer.ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
        "canonical_statement": "The source-bounded component 'II. The Amplification Mechanism: Quantum Stochastic Resonance (QSR) at Criticality' preserves Paper 0 records P0R05320-P0R05322 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05320:ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
            "P0R05321:ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
            "P0R05322:ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
        ),
        "source_formulae": (
            "P0R05320: II. The Amplification Mechanism: Quantum Stochastic Resonance (QSR) at Criticality",
            "P0R05321: The mechanism to amplify localised quantum fluctuations into macroscopic effects (e.g., altering an action potential) is the interplay of Quantum Stochastic Resonance (QSR) and the Quasicritical state of the L4 network.",
            'P0R05322: In QSR, an optimal level of noise enhances the detection of a weak signal in a non-linear system. Here, the "weak signal" is the coherent modulation of the Quantum Potential (Q) by the Psi-field via IET (L1/L2).',
        ),
        "test_protocols": (
            "preserve II. The Amplification Mechanism: Quantum Stochastic Resonance (QSR) at Criticality source-accounting boundary",
        ),
        "null_results": (
            "II. The Amplification Mechanism: Quantum Stochastic Resonance (QSR) at Criticality is not empirical validation evidence",
        ),
        "variables": ("ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",),
        "validation_targets": ("preserve records P0R05320-P0R05322",),
        "null_controls": (
            "ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpec:
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
class TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpec, ...]
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


def build_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_specs(
    source_records: list[dict[str, Any]],
) -> TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle:
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

    specs: list[TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpec(
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
        + "The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system."
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
        "next_source_boundary": "P0R05323",
    }
    return TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_specs(
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
    bundle: TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system."
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
    bundle: TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation_specs_{date_tag}.md"
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
