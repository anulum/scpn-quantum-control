#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: spec builder
"""Promote Paper 0 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: records."""

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
    "P0R04488",
    "P0R04489",
    "P0R04490",
    "P0R04491",
    "P0R04492",
    "P0R04493",
    "P0R04494",
    "P0R04495",
    "P0R04496",
    "P0R04497",
    "P0R04498",
)
CLAIM_BOUNDARY = "source-bounded section 2 cross frequency coupling cfc and hierarchical processing source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_cross_frequency_coupling_cfc_and_hierarchical_processing.2_cross_frequency_coupling_cfc_and_hierarchical_processing": {
        "context_id": "2_cross_frequency_coupling_cfc_and_hierarchical_processing",
        "validation_protocol": "paper0.section_2_cross_frequency_coupling_cfc_and_hierarchical_processing.2_cross_frequency_coupling_cfc_and_hierarchical_processing",
        "canonical_statement": "The source-bounded component '2. Cross-Frequency Coupling (CFC) and Hierarchical Processing:' preserves Paper 0 records P0R04488-P0R04492 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04488:2_cross_frequency_coupling_cfc_and_hierarchical_processing",
            "P0R04489:2_cross_frequency_coupling_cfc_and_hierarchical_processing",
            "P0R04490:2_cross_frequency_coupling_cfc_and_hierarchical_processing",
            "P0R04491:2_cross_frequency_coupling_cfc_and_hierarchical_processing",
            "P0R04492:2_cross_frequency_coupling_cfc_and_hierarchical_processing",
        ),
        "source_formulae": (
            "P0R04488: 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing:",
            "P0R04489: CFC, particularly Phase-Amplitude Coupling (PAC), is the mechanism for hierarchical integration. PAC (e.g., Theta phase modulating Gamma amplitude) nests information across temporal scales, implementing the UPDE's inter-layer coupling (CInterLayer).",
            "P0R04490: Dynamic Regimes of Theta-Gamma Coupling",
            "P0R04491: The Phase-Amplitude Coupling (PAC) formalism provides the basis for understanding how information is nested across scales. Further insights from next-generation neural mass models reveal that this coupling is not monolithic but can manifest in several distinct dynamic regimes depending on the state of the network. Driving a gamma-generating circuit (e.g., a Pyramidal-Interneuronal Network Gamma, or PING, model) with a slow theta-frequency input can produce a spectrum of behaviors: (1)",
            "P0R04492: perfect phase-locking, where gamma bursts are rigidly tied to the theta phase, resulting in periodic dynamics; (2) imperfect locking, leading to quasi-periodic dynamics where the phase relationship drifts slowly over time; and (3) chaotic dynamics, where the phase relationship becomes unpredictable. The SCPN proposes that these distinct dynamical regimes correspond to different functional and cognitive states. Perfect locking may be optimal for high-fidelity memory encoding and recall, where precise temporal sequencing is paramount. Quasi-periodicity may support the flexible and associative nature of thought, while a controlled shift towards chaotic dynamics could be the substrate for creative insight, insight problem-solving, or the expanded state repertoire observed in psychedelic states. The parameters of the UPDE, modulated by neuromodulators and glial activity, thus determine not just the presence of coupling, but its precise dynamic character.",
        ),
        "test_protocols": (
            "preserve 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: source-accounting boundary",
        ),
        "null_results": (
            "2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: is not empirical validation evidence",
        ),
        "variables": ("2_cross_frequency_coupling_cfc_and_hierarchical_processing",),
        "validation_targets": ("preserve records P0R04488-P0R04492",),
        "null_controls": (
            "2_cross_frequency_coupling_cfc_and_hierarchical_processing must remain source-bounded accounting",
        ),
    },
    "section_2_cross_frequency_coupling_cfc_and_hierarchical_processing.the_chemoarchitectural_basis_of_network_dynamics": {
        "context_id": "the_chemoarchitectural_basis_of_network_dynamics",
        "validation_protocol": "paper0.section_2_cross_frequency_coupling_cfc_and_hierarchical_processing.the_chemoarchitectural_basis_of_network_dynamics",
        "canonical_statement": "The source-bounded component 'The Chemoarchitectural Basis of Network Dynamics' preserves Paper 0 records P0R04493-P0R04498 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04493:the_chemoarchitectural_basis_of_network_dynamics",
            "P0R04494:the_chemoarchitectural_basis_of_network_dynamics",
            "P0R04495:the_chemoarchitectural_basis_of_network_dynamics",
            "P0R04496:the_chemoarchitectural_basis_of_network_dynamics",
            "P0R04497:the_chemoarchitectural_basis_of_network_dynamics",
            "P0R04498:the_chemoarchitectural_basis_of_network_dynamics",
        ),
        "source_formulae": (
            "P0R04493: The Chemoarchitectural Basis of Network Dynamics",
            "P0R04494: The spatial heterogeneity of the parameters in the Unified Phase Dynamics Equation-the intrinsic frequencies (i), coupling strengths (Kij), and local noise levels (i)-is not random. It is determined by the underlying chemoarchitecture of the cortex, providing a formal bridge between the micro-scale of neurotransmitter systems (Layer 2) and the macro-scale of emergent network dynamics (Layer 4). Recent studies combining PET atlases of receptor densities with MEG recordings of oscillatory networks provide a direct empirical basis for this mapping.",
            "P0R04495: The parameters of the UPDE for a given cortical region can be formally defined as functions of its local receptor and transporter densities:",
            'P0R04496: Intrinsic Frequency (i): The natural frequency of a cortical column is a function of its local excitation-inhibition (E/I) balance. This can be estimated from the ratio of excitatory (e.g., NMDA) to inhibitory (e.g., GABA-A) receptor densities. Regions with higher relative excitatory receptor density will exhibit higher intrinsic frequencies. | Coupling Strength (Kij): The strength of functional coupling between two regions is modulated by neuromodulatory systems. The density of specific receptors known to positively covary with network "hubness" in certain frequency bands (e.g., dopaminergic D1 and serotonergic 5-HT4 receptors for gamma-band synchrony) can be used to parameterise the coupling matrix Kij. | Local Noise (i): The amplitude of local stochastic fluctuations can be related to the density of receptors that modulate neuronal gain and membrane potential variance, such as those involved in regulating potassium channels or certain metabotropic pathways.',
            "P0R04497: This mapping provides a powerful unifying principle. It demonstrates how the brain's structural connectome (anatomy), its neurotransmitter landscape (chemistry), and its emergent oscillatory patterns (dynamics) are not separate domains but are deeply intertwined aspects of a single, coherent system. This provides a concrete, data-driven methodology for parameterising large-scale brain models like the SCPN, making the framework significantly more testable and empirically grounded.",
            "P0R04498: [TABLE]",
        ),
        "test_protocols": (
            "preserve The Chemoarchitectural Basis of Network Dynamics source-accounting boundary",
        ),
        "null_results": (
            "The Chemoarchitectural Basis of Network Dynamics is not empirical validation evidence",
        ),
        "variables": ("the_chemoarchitectural_basis_of_network_dynamics",),
        "validation_targets": ("preserve records P0R04493-P0R04498",),
        "null_controls": (
            "the_chemoarchitectural_basis_of_network_dynamics must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpec:
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
class Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpec, ...]
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


def build_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_specs(
    source_records: list[dict[str, Any]],
) -> Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle:
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

    specs: list[Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpec(
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
        + "2. Cross-Frequency Coupling (CFC) and Hierarchical Processing:"
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
        "next_source_boundary": "P0R04499",
    }
    return Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_specs(
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
    bundle: Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2. Cross-Frequency Coupling (CFC) and Hierarchical Processing:" + " Specs",
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
    bundle: Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation_specs_{date_tag}.md"
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
