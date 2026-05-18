#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) spec builder
"""Promote Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) records."""

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
    "P0R04786",
    "P0R04787",
    "P0R04788",
    "P0R04789",
    "P0R04790",
    "P0R04791",
    "P0R04792",
    "P0R04793",
)
CLAIM_BOUNDARY = "source-bounded iv sub synaptic and axonal architecture l1 l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iv_sub_synaptic_and_axonal_architecture_l1_l3.iv_sub_synaptic_and_axonal_architecture_l1_l3": {
        "context_id": "iv_sub_synaptic_and_axonal_architecture_l1_l3",
        "validation_protocol": "paper0.iv_sub_synaptic_and_axonal_architecture_l1_l3.iv_sub_synaptic_and_axonal_architecture_l1_l3",
        "canonical_statement": "The source-bounded component 'IV. Sub-Synaptic and Axonal Architecture (L1-L3)' preserves Paper 0 records P0R04786-P0R04786 without empirical validation claims.",
        "source_equation_ids": ("P0R04786:iv_sub_synaptic_and_axonal_architecture_l1_l3",),
        "source_formulae": ("P0R04786: IV. Sub-Synaptic and Axonal Architecture (L1-L3)",),
        "test_protocols": (
            "preserve IV. Sub-Synaptic and Axonal Architecture (L1-L3) source-accounting boundary",
        ),
        "null_results": (
            "IV. Sub-Synaptic and Axonal Architecture (L1-L3) is not empirical validation evidence",
        ),
        "variables": ("iv_sub_synaptic_and_axonal_architecture_l1_l3",),
        "validation_targets": ("preserve records P0R04786-P0R04786",),
        "null_controls": (
            "iv_sub_synaptic_and_axonal_architecture_l1_l3 must remain source-bounded accounting",
        ),
    },
    "iv_sub_synaptic_and_axonal_architecture_l1_l3.1_the_post_synaptic_density_psd": {
        "context_id": "1_the_post_synaptic_density_psd",
        "validation_protocol": "paper0.iv_sub_synaptic_and_axonal_architecture_l1_l3.1_the_post_synaptic_density_psd",
        "canonical_statement": "The source-bounded component '1. The Post-Synaptic Density (PSD):' preserves Paper 0 records P0R04787-P0R04789 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04787:1_the_post_synaptic_density_psd",
            "P0R04788:1_the_post_synaptic_density_psd",
            "P0R04789:1_the_post_synaptic_density_psd",
        ),
        "source_formulae": (
            "P0R04787: 1. The Post-Synaptic Density (PSD):",
            "P0R04788: A dense protein complex at the postsynaptic terminal.",
            'P0R04789: Organisation (Signalosomes): Scaffolding proteins (e.g., PSD-95) organise receptors and signalling enzymes into "signalosomes." This precise arrangement ensures efficient coupling between L2 activation and L3 plasticity. | CaMKII: The Molecular Memory Switch: CaMKII acts as a bistable molecular switch encoding synaptic memory. It interacts directly with the cytoskeleton (Actin and MTs), potentially providing a mechanism for encoding information directly into the L1 lattice.',
        ),
        "test_protocols": (
            "preserve 1. The Post-Synaptic Density (PSD): source-accounting boundary",
        ),
        "null_results": (
            "1. The Post-Synaptic Density (PSD): is not empirical validation evidence",
        ),
        "variables": ("1_the_post_synaptic_density_psd",),
        "validation_targets": ("preserve records P0R04787-P0R04789",),
        "null_controls": (
            "1_the_post_synaptic_density_psd must remain source-bounded accounting",
        ),
    },
    "iv_sub_synaptic_and_axonal_architecture_l1_l3.2_axonal_structure_and_transport": {
        "context_id": "2_axonal_structure_and_transport",
        "validation_protocol": "paper0.iv_sub_synaptic_and_axonal_architecture_l1_l3.2_axonal_structure_and_transport",
        "canonical_statement": "The source-bounded component '2. Axonal Structure and Transport:' preserves Paper 0 records P0R04790-P0R04791 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04790:2_axonal_structure_and_transport",
            "P0R04791:2_axonal_structure_and_transport",
        ),
        "source_formulae": (
            "P0R04790: 2. Axonal Structure and Transport:",
            "P0R04791: Periodic Membrane Skeleton (PMS): An ordered structure of Actin rings and Spectrin filaments along the axon, providing stability and organising ion channels. | Axonal Transport (L3/L1): Molecular motors (Kinesin, Dynein) move along Microtubule (L1) tracks. The efficiency of transport depends on the integrity of the L1 MT substrate.",
        ),
        "test_protocols": (
            "preserve 2. Axonal Structure and Transport: source-accounting boundary",
        ),
        "null_results": (
            "2. Axonal Structure and Transport: is not empirical validation evidence",
        ),
        "variables": ("2_axonal_structure_and_transport",),
        "validation_targets": ("preserve records P0R04790-P0R04791",),
        "null_controls": (
            "2_axonal_structure_and_transport must remain source-bounded accounting",
        ),
    },
    "iv_sub_synaptic_and_axonal_architecture_l1_l3.v_the_deep_quantum_milieu_l1": {
        "context_id": "v_the_deep_quantum_milieu_l1",
        "validation_protocol": "paper0.iv_sub_synaptic_and_axonal_architecture_l1_l3.v_the_deep_quantum_milieu_l1",
        "canonical_statement": "The source-bounded component 'V. The Deep Quantum Milieu (L1)' preserves Paper 0 records P0R04792-P0R04793 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04792:v_the_deep_quantum_milieu_l1",
            "P0R04793:v_the_deep_quantum_milieu_l1",
        ),
        "source_formulae": (
            "P0R04792: V. The Deep Quantum Milieu (L1)",
            "P0R04793: The deepest layer of the interface involves the cytoskeleton, quantum field effects, and spin dynamics.",
        ),
        "test_protocols": ("preserve V. The Deep Quantum Milieu (L1) source-accounting boundary",),
        "null_results": ("V. The Deep Quantum Milieu (L1) is not empirical validation evidence",),
        "variables": ("v_the_deep_quantum_milieu_l1",),
        "validation_targets": ("preserve records P0R04792-P0R04793",),
        "null_controls": ("v_the_deep_quantum_milieu_l1 must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class IvSubSynapticAndAxonalArchitectureL1L3Spec:
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
class IvSubSynapticAndAxonalArchitectureL1L3SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IvSubSynapticAndAxonalArchitectureL1L3Spec, ...]
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


def build_iv_sub_synaptic_and_axonal_architecture_l1_l3_specs(
    source_records: list[dict[str, Any]],
) -> IvSubSynapticAndAxonalArchitectureL1L3SpecBundle:
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

    specs: list[IvSubSynapticAndAxonalArchitectureL1L3Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IvSubSynapticAndAxonalArchitectureL1L3Spec(
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
        "title": "Paper 0 " + "IV. Sub-Synaptic and Axonal Architecture (L1-L3)" + " Specs",
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
        "next_source_boundary": "P0R04794",
    }
    return IvSubSynapticAndAxonalArchitectureL1L3SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IvSubSynapticAndAxonalArchitectureL1L3SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iv_sub_synaptic_and_axonal_architecture_l1_l3_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IvSubSynapticAndAxonalArchitectureL1L3SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "IV. Sub-Synaptic and Axonal Architecture (L1-L3)" + " Specs",
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
    bundle: IvSubSynapticAndAxonalArchitectureL1L3SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iv_sub_synaptic_and_axonal_architecture_l1_l3_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iv_sub_synaptic_and_axonal_architecture_l1_l3_validation_specs_{date_tag}.md"
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
