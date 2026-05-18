#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Introduction to The Dynamics of the Coherent Brain (Domain I: L4) spec builder
"""Promote Paper 0 Introduction to The Dynamics of the Coherent Brain (Domain I: L4) records."""

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
    "P0R04572",
    "P0R04573",
    "P0R04574",
    "P0R04575",
    "P0R04576",
    "P0R04577",
    "P0R04578",
    "P0R04579",
    "P0R04580",
)
CLAIM_BOUNDARY = "source-bounded introduction to the dynamics of the coherent brain domain i l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4": {
        "context_id": "introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "validation_protocol": "paper0.introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "canonical_statement": "The source-bounded component 'Introduction to The Dynamics of the Coherent Brain (Domain I: L4)' preserves Paper 0 records P0R04572-P0R04574 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04572:introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",
            "P0R04573:introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",
            "P0R04574:introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",
        ),
        "source_formulae": (
            "P0R04572: Introduction to The Dynamics of the Coherent Brain (Domain I: L4)",
            "P0R04573: Layer 4 is the domain of emergent synchronisation and the immediate substrate of cognition.",
            'P0R04574: Travelling Waves: Coherent activity propagates across the cortex as travelling waves. These waves are not epiphenomena but a dynamic scaffold for organising information flow and implementing the Hierarchical Predictive Coding (HPC) architecture. | Metastability and Chimaera States: The critical brain exhibits metastability-the capacity for transient synchronisation and desynchronization. The coexistence of coherent and incoherent domains ("chimaera states") is thought to be crucial for flexible and complex information processing, adding a layer of nuance to the principle of Quasicriticality. | The Dynamic Connectome: It is crucial to distinguish the static structural connectome (L3) from the functional connectome (L4), which is the pattern of statistical dependencies between brain regions. This functional network is highly dynamic, reconfiguring on sub-second timescales to meet cognitive demands, constrained by the underlying structural scaffold.',
        ),
        "test_protocols": (
            "preserve Introduction to The Dynamics of the Coherent Brain (Domain I: L4) source-accounting boundary",
        ),
        "null_results": (
            "Introduction to The Dynamics of the Coherent Brain (Domain I: L4) is not empirical validation evidence",
        ),
        "variables": ("introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4",),
        "validation_targets": ("preserve records P0R04572-P0R04574",),
        "null_controls": (
            "introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4 must remain source-bounded accounting",
        ),
    },
    "introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4": {
        "context_id": "iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "validation_protocol": "paper0.introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "canonical_statement": "The source-bounded component 'III. Examination of The Dynamics of the Coherent Brain (Domain I: L4)' preserves Paper 0 records P0R04575-P0R04576 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04575:iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4",
            "P0R04576:iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4",
        ),
        "source_formulae": (
            "P0R04575: III. Examination of The Dynamics of the Coherent Brain (Domain I: L4)",
            "P0R04576: Layer 4 is the domain of emergent synchronisation, where the collective action of billions of neurons, constrained by the geometric scaffold of Layer 3, gives rise to the coherent, large-scale dynamics that are the immediate substrate of cognition. This is the realm of brain rhythms, network states, and the fluid reconfiguration of functional circuits.",
        ),
        "test_protocols": (
            "preserve III. Examination of The Dynamics of the Coherent Brain (Domain I: L4) source-accounting boundary",
        ),
        "null_results": (
            "III. Examination of The Dynamics of the Coherent Brain (Domain I: L4) is not empirical validation evidence",
        ),
        "variables": ("iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4",),
        "validation_targets": ("preserve records P0R04575-P0R04576",),
        "null_controls": (
            "iii_examination_of_the_dynamics_of_the_coherent_brain_domain_i_l4 must remain source-bounded accounting",
        ),
    },
    "introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.travelling_waves_the_dynamic_scaffold_of_information_flow": {
        "context_id": "travelling_waves_the_dynamic_scaffold_of_information_flow",
        "validation_protocol": "paper0.introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4.travelling_waves_the_dynamic_scaffold_of_information_flow",
        "canonical_statement": "The source-bounded component 'Travelling Waves: The Dynamic Scaffold of Information Flow' preserves Paper 0 records P0R04577-P0R04580 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04577:travelling_waves_the_dynamic_scaffold_of_information_flow",
            "P0R04578:travelling_waves_the_dynamic_scaffold_of_information_flow",
            "P0R04579:travelling_waves_the_dynamic_scaffold_of_information_flow",
            "P0R04580:travelling_waves_the_dynamic_scaffold_of_information_flow",
        ),
        "source_formulae": (
            "P0R04577: Travelling Waves: The Dynamic Scaffold of Information Flow",
            "P0R04578: Coherent neural activity does not simply flicker in place; it often propagates across the cortex in organised patterns known as travelling waves. These waves, which consist of oscillatory signals moving sequentially across contiguous brain regions, are not mere epiphenomena. Within the SCPN, they are formalised as a primary mechanism for organising information flow and implementing the bidirectional computations of the Hierarchical Predictive Coding (HPC) architecture.",
            "P0R04579: Directionality and Predictive Coding: The direction of wave propagation is functionally significant. Computational models and empirical data suggest a direct link between wave direction and the flow of predictions versus prediction errors in the HPC loop. Feedforward (FW) Waves: Waves travelling in an occipital-to-frontal direction (from sensory to association cortices) are associated with the bottom-up propagation of sensory information and prediction errors. | Backwards (BW) Waves: Waves travelling in a frontal-to-occipital direction are associated with the top-down transmission of predictions from the generative model.",
            "P0R04580: This dynamic routing provides a physical substrate for the continuous, recursive updating that is central to active inference.",
        ),
        "test_protocols": (
            "preserve Travelling Waves: The Dynamic Scaffold of Information Flow source-accounting boundary",
        ),
        "null_results": (
            "Travelling Waves: The Dynamic Scaffold of Information Flow is not empirical validation evidence",
        ),
        "variables": ("travelling_waves_the_dynamic_scaffold_of_information_flow",),
        "validation_targets": ("preserve records P0R04577-P0R04580",),
        "null_controls": (
            "travelling_waves_the_dynamic_scaffold_of_information_flow must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4Spec:
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
class IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4Spec, ...]
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


def build_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_specs(
    source_records: list[dict[str, Any]],
) -> IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle:
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

    specs: list[IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4Spec(
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
        + "Introduction to The Dynamics of the Coherent Brain (Domain I: L4)"
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
        "next_source_boundary": "P0R04581",
    }
    return IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_specs(
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


def render_report(bundle: IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Introduction to The Dynamics of the Coherent Brain (Domain I: L4)"
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
    bundle: IntroductionToTheDynamicsOfTheCoherentBrainDomainIL4SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_specs_{date_tag}.md"
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
