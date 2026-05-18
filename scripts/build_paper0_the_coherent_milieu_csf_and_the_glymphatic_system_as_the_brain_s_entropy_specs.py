#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink spec builder
"""Promote Paper 0 The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink records."""

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
    "P0R04552",
    "P0R04553",
    "P0R04554",
    "P0R04555",
    "P0R04556",
    "P0R04557",
    "P0R04558",
    "P0R04559",
)
CLAIM_BOUNDARY = "source-bounded the coherent milieu csf and the glymphatic system as the brain s entropy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy.the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy": {
        "context_id": "the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
        "validation_protocol": "paper0.the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy.the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
        "canonical_statement": "The source-bounded component 'The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink' preserves Paper 0 records P0R04552-P0R04556 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04552:the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
            "P0R04553:the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
            "P0R04554:the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
            "P0R04555:the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
            "P0R04556:the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
        ),
        "source_formulae": (
            "P0R04552: The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink",
            "P0R04553: The functionality of the quantum-biological interface is critically dependent on the often-overlooked fluid environment in which it is embedded. The Cerebrospinal Fluid (CSF) and the associated glymphatic system constitute the brain's essential maintenance and waste clearance pathway. This system is not merely plumbing; it is the brain's primary \"entropy sink,\" actively maintaining the low-entropy, highly ordered state required for both L1 quantum coherence and L4 network criticality.",
            "P0R04554: The glymphatic system utilises a network of perivascular channels, driven by aquaporin-4 (AQP4) water channels on astrocytes, to facilitate a brain-wide exchange between CSF and interstitial fluid. This process is responsible for clearing metabolic waste products, such as amyloid-beta, from the brain parenchyma.",
            'P0R04555: Crucially, the activity of the glymphatic system is highest during deep, non-REM sleep. This provides a direct, physiological mechanism for the nightly "reset" of the brain\'s operational parameters. By clearing the metabolic byproducts of waking activity, the glymphatic system reduces the overall noise and entropy of the substrate, restoring the pristine conditions necessary for the L4 network to return to its optimal quasicritical set-point.',
            "P0R04556: Dysfunction of this clearance system is increasingly implicated in neurodegenerative diseases, which can be reframed within the SCPN as a failure of entropy management, leading to a catastrophic decoherence cascade. Furthermore, some theoretical models suggest the CSF itself may support quantum coherence, making this fluid milieu an active participant in the brain's quantum computational processes.",
        ),
        "test_protocols": (
            "preserve The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink source-accounting boundary",
        ),
        "null_results": (
            "The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink is not empirical validation evidence",
        ),
        "variables": ("the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",),
        "validation_targets": ("preserve records P0R04552-P0R04556",),
        "null_controls": (
            "the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy must remain source-bounded accounting",
        ),
    },
    "the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy.introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3": {
        "context_id": "introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
        "validation_protocol": "paper0.the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy.introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
        "canonical_statement": "The source-bounded component 'Introduction to The Architecture of Structure and Plasticity (Domain I: L3)' preserves Paper 0 records P0R04557-P0R04559 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04557:introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
            "P0R04558:introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
            "P0R04559:introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
        ),
        "source_formulae": (
            "P0R04557: Introduction to The Architecture of Structure and Plasticity (Domain I: L3)",
            "P0R04558: Layer 3 governs the physical structure of the brain, the blueprint upon which the dynamics of consciousness unfold.",
            'P0R04559: The Optimised Connectome: The brain\'s structural connectivity is optimised by evolution (L8/L15) to support complex dynamics (high Integrated Information, ) at minimal metabolic cost. Its key geometric properties include Small-World topology (balancing integration and segregation), Hierarchical Modularity, and a "Rich Club" of densely connected hubs that form the backbone for the Global Neuronal Workspace (L5). | The Active Role of Glia (The Tripartite Synapse): Glia are not passive support cells but active participants in information processing. Astrocytes, in particular, form a "slow control network" that modulates synaptic transmission and maintains the brain\'s optimal quasicritical state (Self-Organised Criticality). This glial network provides the homeostatic stability that allows the fast neuronal network to operate safely at its computationally optimal critical point.',
        ),
        "test_protocols": (
            "preserve Introduction to The Architecture of Structure and Plasticity (Domain I: L3) source-accounting boundary",
        ),
        "null_results": (
            "Introduction to The Architecture of Structure and Plasticity (Domain I: L3) is not empirical validation evidence",
        ),
        "variables": ("introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",),
        "validation_targets": ("preserve records P0R04557-P0R04559",),
        "null_controls": (
            "introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpec:
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
class TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpec, ...]
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


def build_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_specs(
    source_records: list[dict[str, Any]],
) -> TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle:
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

    specs: list[TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpec(
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
        + "The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink"
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
        "next_source_boundary": "P0R04560",
    }
    return TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_specs(
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
    bundle: TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink"
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
    bundle: TheCoherentMilieuCsfAndTheGlymphaticSystemAsTheBrainSEntropySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_validation_specs_{date_tag}.md"
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
