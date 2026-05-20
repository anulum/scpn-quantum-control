#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) spec builder
"""Promote Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) records."""

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
    "P0R04813",
    "P0R04814",
    "P0R04815",
    "P0R04816",
    "P0R04817",
    "P0R04818",
    "P0R04819",
    "P0R04820",
    "P0R04821",
    "P0R04822",
    "P0R04823",
)
CLAIM_BOUNDARY = "source-bounded ii micro scale geometry the quantum and molecular scaffold l1 l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3.ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3": {
        "context_id": "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
        "validation_protocol": "paper0.ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3.ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
        "canonical_statement": "The source-bounded component 'II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)' preserves Paper 0 records P0R04813-P0R04814 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04813:ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
            "P0R04814:ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
        ),
        "source_formulae": (
            "P0R04813: II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)",
            "P0R04814: The foundational layers (Domain I) rely on precise geometric organisation at the micro- and nano-scales to establish the quantum interface and cellular structure.",
        ),
        "test_protocols": (
            "preserve II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) source-accounting boundary",
        ),
        "null_results": (
            "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) is not empirical validation evidence",
        ),
        "variables": ("ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",),
        "validation_targets": ("preserve records P0R04813-P0R04814",),
        "null_controls": (
            "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3 must remain source-bounded accounting",
        ),
    },
    "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3.1_the_geometry_of_the_quantum_substrate_l1": {
        "context_id": "1_the_geometry_of_the_quantum_substrate_l1",
        "validation_protocol": "paper0.ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3.1_the_geometry_of_the_quantum_substrate_l1",
        "canonical_statement": "The source-bounded component '1. The Geometry of the Quantum Substrate (L1):' preserves Paper 0 records P0R04815-P0R04823 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04815:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04816:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04817:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04818:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04819:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04820:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04821:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04822:1_the_geometry_of_the_quantum_substrate_l1",
            "P0R04823:1_the_geometry_of_the_quantum_substrate_l1",
        ),
        "source_formulae": (
            "P0R04815: 1. The Geometry of the Quantum Substrate (L1):",
            "P0R04816: The Microtubule (MT) Lattice and Topological QEC: The specific geometry of the MT is critical. The helical arrangement of tubulin dimers ensures the precise alignment of their large dipole moments. This geometric configuration generates the strong interaction energy (J0.82 eV) required for Topological Quantum Error Correction (QEC). Information is encoded in the global topological properties of this lattice, protected by the resulting energy gap (Delta1.64 eV).",
            "P0R04817: QEC and the Microtubule Lattice: Fracton Order and 2D Surface Anyons",
            "P0R04818: The specific helical geometry of the Microtubule (MT) lattice supports Topological Quantum Error Correction (QEC). However, a naive application of standard anyonic braiding to the 3D MT bulk is physically prohibited; in three spatial dimensions, particle exchange paths can trivially untangle, destroying topological protection. To resolve this dimensionality constraint, the SCPN framework posits a dual topological architecture that perfectly matches the MT's physical geometry:",
            "P0R04819: 3D Fracton Topological Order (The Bulk Storage): The rigid, 3D helical lattice of the MT itself does not support mobile anyons. Instead, it supports Fracton Topological Order. Fractons are a newly understood class of topological excitations uniquely suited to rigid 3D geometries. Unlike anyons, fractons are strictly immobile (or restricted to linear sub-manifolds) because their movement is forbidden by sub-dimensional symmetries and generalized charge conservation laws-specifically, the conservation of dipole moment ($\\sum_i \\mathbf{r}_i q_i = \\text{const}$). The precise, rigid alignment of tubulin dipoles ($J \\approx 0.82 \\text{ eV}$) provides the exact physical symmetries required to lock these fracton defects in place, resulting in a large protective energy gap ($\\Delta \\approx 1.64 \\text{ eV}$). This makes the MT bulk an ultra-stable, glass-like quantum memory immune to local thermal perturbations. | 2D Surface States (The Anyonic Interface): For active computation and error syndrome measurement, mobile topological defects are required. These are restricted to the highly ordered interfacial water (Coherence Domains) wrapping the MT cylinder. This hydration layer acts as a 2D topological insulator. Within this strictly two-dimensional surface manifold, standard point-like topological defects (anyons) can exist, propagate, and undergo the protected braiding operations necessary to interface with the fracton bulk.",
            "P0R04820: This dual architecture allows the MT to act as a complete topological quantum memory: the 3D fracton bulk provides ultra-stable, immobile storage, while the 2D anyonic surface provides the mobile interface for active syndrome measurement and logical gate operations.",
            "P0R04821: P0R04821",
            "P0R04822: P0R04822",
            "P0R04823: The Geometry of Water (Coherence Domains): Interfacial water forms highly ordered, quasi-crystalline structures (QED Coherence Domains, CDs). The geometry of these CDs shields the L1 substrate from thermal decoherence and creates specific geometric pathways (Grotthuss mechanism) for rapid proton transport.",
        ),
        "test_protocols": (
            "preserve 1. The Geometry of the Quantum Substrate (L1): source-accounting boundary",
        ),
        "null_results": (
            "1. The Geometry of the Quantum Substrate (L1): is not empirical validation evidence",
        ),
        "variables": ("1_the_geometry_of_the_quantum_substrate_l1",),
        "validation_targets": ("preserve records P0R04815-P0R04823",),
        "null_controls": (
            "1_the_geometry_of_the_quantum_substrate_l1 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Spec:
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
class IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Spec, ...]
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


def build_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_specs(
    source_records: list[dict[str, Any]],
) -> IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle:
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

    specs: list[IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Spec(
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
        + "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)"
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
        "next_source_boundary": "P0R04824",
    }
    return IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_specs(
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


def render_report(bundle: IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)"
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
    bundle: IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation_specs_{date_tag}.md"
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
