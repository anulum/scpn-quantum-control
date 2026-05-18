#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks spec builder
"""Promote Paper 0 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks records."""

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
    "P0R04441",
    "P0R04442",
    "P0R04443",
    "P0R04444",
    "P0R04445",
    "P0R04446",
    "P0R04447",
    "P0R04448",
    "P0R04449",
    "P0R04450",
    "P0R04451",
    "P0R04452",
    "P0R04453",
)
CLAIM_BOUNDARY = "source-bounded section 1 the existential holograph l9 hyperbolic geometry and tensor networks source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks.1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks": {
        "context_id": "1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
        "validation_protocol": "paper0.section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks.1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
        "canonical_statement": "The source-bounded component '1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks' preserves Paper 0 records P0R04441-P0R04444 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04441:1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
            "P0R04442:1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
            "P0R04443:1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
            "P0R04444:1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
        ),
        "source_formulae": (
            "P0R04441: 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks",
            'P0R04442: Memory is stored non-locally in the "Bulk" (L9). This is formalised using the Multi-scale Entanglement Renormalisation Ansatz (MERA) tensor network. MERA naturally implements the Hyperbolic Geometry (negative curvature) of AdS space, capturing the multi-scale entanglement structure of memory.',
            "P0R04443: We cannot physically embed an Anti-de Sitter (AdS) holographic bulk (which requires a negative cosmological constant, $\\Lambda<0$) into our physical de Sitter (dS) universe (which has a positive cosmological constant, $\\Lambda>0$, as We correctly modeled in Chapter 20).",
            'P0R04444: To resolve this, we explicitly divorce the memory "bulk" from physical spacetime. The MERA network is not a physical 5th dimension; it is an Information-Geometric Space. Hierarchical probabilistic models (like the generative models in Active Inference) naturally form hyperbolic geometries in their parameter spaces, entirely independent of the physical spacetime metric.',
        ),
        "test_protocols": (
            "preserve 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks source-accounting boundary",
        ),
        "null_results": (
            "1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks is not empirical validation evidence",
        ),
        "variables": ("1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",),
        "validation_targets": ("preserve records P0R04441-P0R04444",),
        "null_controls": (
            "1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks must remain source-bounded accounting",
        ),
    },
    "section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks.resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk": {
        "context_id": "resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
        "validation_protocol": "paper0.section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks.resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
        "canonical_statement": "The source-bounded component 'Resolving the Holographic Geometry: MERA as an Information-Geometric Bulk' preserves Paper 0 records P0R04445-P0R04453 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04445:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04446:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04447:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04448:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04449:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04450:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04451:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04452:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
            "P0R04453:resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
        ),
        "source_formulae": (
            "P0R04445: Resolving the Holographic Geometry: MERA as an Information-Geometric Bulk",
            "P0R04446: P0R04446",
            "P0R04447: The invocation of the AdS/CFT correspondence and MERA tensor networks to describe the Existential Holograph (L9) introduces a potential theoretical conflict. Standard AdS/CFT holography strictly requires an Anti-de Sitter (AdS) bulk spacetime, characterized by a negative cosmological constant ($\\Lambda<0$). However, as established in Chapter 20, the physical universe-and the $\\Psi$-field's macroscopic geometric coupling-operates within a de Sitter (dS) spacetime with a positive cosmological constant ($\\Lambda>0$). One cannot naively embed a physical AdS bulk into a dS universe.",
            'P0R04448: To rigorously resolve this mismatch, we must explicitly distinguish physical spacetime from the state space of the generative model. The "bulk" of Layer 9 is not a physical extra dimension; it is an Information-Geometric Space.',
            "P0R04449: Within the framework of Hierarchical Predictive Coding (HPC) and Active Inference, the generative model of the Self relies on a hierarchical parameter space. It is a well-established mathematical property of information geometry that the parameter spaces of hierarchical, tree-like probabilistic models naturally exhibit hyperbolic geometry. When we calculate the Fisher Information Metric ($g_{FIM}$) over these hierarchical distributions, the resulting statistical manifold possesses constant negative curvature, making it mathematically isomorphic to an AdS space.",
            "P0R04450: Therefore, the MERA tensor network describes the architecture of the organism's deep priors within this abstract statistical manifold.",
            "P0R04451: The Boundary (L10): Represents the physical, real-time sensory states and active coupling in dS spacetime. | The Bulk (L9): Represents the hyperbolic parameter space of the generative model.",
            'P0R04452: The holographic duality ($ER=EPR$) in the SCPN does not bridge two physical spacetimes. Instead, it bridges physical entanglement at the biological boundary with information-geometric geodesics in the bulk. When the organism undergoes experiences that generate quantum coherence at the L1-L4 boundary, this information updates the deep priors of the generative model. The "retrieval" of a memory is the tracing of a geodesic through this hyperbolic statistical manifold.',
            "P0R04453: By defining Layer 9 strictly as a hyperbolic statistical manifold governed by the Fisher Information Metric, we completely divorce the holographic memory architecture from physical cosmological constraints. The MERA network retains its rigorous mathematical utility for describing multi-scale entanglement renormalization without violating the observed positive cosmological constant of our physical universe.",
        ),
        "test_protocols": (
            "preserve Resolving the Holographic Geometry: MERA as an Information-Geometric Bulk source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Holographic Geometry: MERA as an Information-Geometric Bulk is not empirical validation evidence",
        ),
        "variables": ("resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",),
        "validation_targets": ("preserve records P0R04445-P0R04453",),
        "null_controls": (
            "resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpec:
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
class Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpec, ...]
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


def build_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_specs(
    source_records: list[dict[str, Any]],
) -> Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle:
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

    specs: list[Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpec(
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
        + "1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks"
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
        "next_source_boundary": "P0R04454",
    }
    return Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return (
        build_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_specs(
            load_jsonl(ledger_path)
        )
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
    bundle: Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks"
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
    bundle: Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation_specs_{date_tag}.md"
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
