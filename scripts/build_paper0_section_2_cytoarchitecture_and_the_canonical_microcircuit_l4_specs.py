#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2. Cytoarchitecture and the Canonical Microcircuit (L4): spec builder
"""Promote Paper 0 2. Cytoarchitecture and the Canonical Microcircuit (L4): records."""

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
    "P0R04832",
    "P0R04833",
    "P0R04834",
    "P0R04835",
    "P0R04836",
    "P0R04837",
    "P0R04838",
    "P0R04839",
)
CLAIM_BOUNDARY = "source-bounded section 2 cytoarchitecture and the canonical microcircuit l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.2_cytoarchitecture_and_the_canonical_microcircuit_l4": {
        "context_id": "2_cytoarchitecture_and_the_canonical_microcircuit_l4",
        "validation_protocol": "paper0.section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.2_cytoarchitecture_and_the_canonical_microcircuit_l4",
        "canonical_statement": "The source-bounded component '2. Cytoarchitecture and the Canonical Microcircuit (L4):' preserves Paper 0 records P0R04832-P0R04834 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04832:2_cytoarchitecture_and_the_canonical_microcircuit_l4",
            "P0R04833:2_cytoarchitecture_and_the_canonical_microcircuit_l4",
            "P0R04834:2_cytoarchitecture_and_the_canonical_microcircuit_l4",
        ),
        "source_formulae": (
            "P0R04832: 2. Cytoarchitecture and the Canonical Microcircuit (L4):",
            "P0R04833: The geometric arrangement of neurons defines the computational structure. The cortex is organised into columns, repeating geometric units implementing Hierarchical Predictive Coding (HPC).",
            "P0R04834: Laminar Geometry and HPC: The six-layered laminar structure of the cortex is crucial for computation. The geometric separation of cell types and connections defines the flow of information: Deep Layers (V/VI): Geometric locus of top-down Priors/Generative Model (Alpha/Beta oscillations). | Superficial Layers (II/III): Geometric locus of bottom-up Prediction Errors (Gamma oscillations).",
        ),
        "test_protocols": (
            "preserve 2. Cytoarchitecture and the Canonical Microcircuit (L4): source-accounting boundary",
        ),
        "null_results": (
            "2. Cytoarchitecture and the Canonical Microcircuit (L4): is not empirical validation evidence",
        ),
        "variables": ("2_cytoarchitecture_and_the_canonical_microcircuit_l4",),
        "validation_targets": ("preserve records P0R04832-P0R04834",),
        "null_controls": (
            "2_cytoarchitecture_and_the_canonical_microcircuit_l4 must remain source-bounded accounting",
        ),
    },
    "section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.iv_macro_scale_geometry_networks_and_dynamics_l4": {
        "context_id": "iv_macro_scale_geometry_networks_and_dynamics_l4",
        "validation_protocol": "paper0.section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.iv_macro_scale_geometry_networks_and_dynamics_l4",
        "canonical_statement": "The source-bounded component 'IV. Macro-Scale Geometry: Networks and Dynamics (L4)' preserves Paper 0 records P0R04835-P0R04836 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04835:iv_macro_scale_geometry_networks_and_dynamics_l4",
            "P0R04836:iv_macro_scale_geometry_networks_and_dynamics_l4",
        ),
        "source_formulae": (
            "P0R04835: IV. Macro-Scale Geometry: Networks and Dynamics (L4)",
            "P0R04836: The global organisation of the brain and the geometry of its activity patterns define the macro-scale scaffold.",
        ),
        "test_protocols": (
            "preserve IV. Macro-Scale Geometry: Networks and Dynamics (L4) source-accounting boundary",
        ),
        "null_results": (
            "IV. Macro-Scale Geometry: Networks and Dynamics (L4) is not empirical validation evidence",
        ),
        "variables": ("iv_macro_scale_geometry_networks_and_dynamics_l4",),
        "validation_targets": ("preserve records P0R04835-P0R04836",),
        "null_controls": (
            "iv_macro_scale_geometry_networks_and_dynamics_l4 must remain source-bounded accounting",
        ),
    },
    "section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.1_the_connectome_topology_the_optimised_network_scaffold": {
        "context_id": "1_the_connectome_topology_the_optimised_network_scaffold",
        "validation_protocol": "paper0.section_2_cytoarchitecture_and_the_canonical_microcircuit_l4.1_the_connectome_topology_the_optimised_network_scaffold",
        "canonical_statement": "The source-bounded component '1. The Connectome Topology (The Optimised Network Scaffold):' preserves Paper 0 records P0R04837-P0R04839 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04837:1_the_connectome_topology_the_optimised_network_scaffold",
            "P0R04838:1_the_connectome_topology_the_optimised_network_scaffold",
            "P0R04839:1_the_connectome_topology_the_optimised_network_scaffold",
        ),
        "source_formulae": (
            "P0R04837: 1. The Connectome Topology (The Optimised Network Scaffold):",
            "P0R04838: The structural Connectome possesses specific geometric properties optimised for high Integrated Information () and Quasicriticality.",
            "P0R04839: Small-World Topology: A geometric configuration balancing high local clustering (segregation) and short global path length (integration). | Hierarchical Modularity: A fractal organisation of specialised modules. | The Rich Club (The Core): A dense geometric core of highly connected hubs (e.g., Precuneus, PFC). This structure forms the backbone for global integration and the Global Neuronal Workspace (L5). | Physical Geometry (Gyrification): The folding of the cortex (gyri and sulci) is a geometric solution to maximise cortical surface area while minimising axonal wiring length and conduction delays, optimising the efficiency of the UPDE synchronisation.",
        ),
        "test_protocols": (
            "preserve 1. The Connectome Topology (The Optimised Network Scaffold): source-accounting boundary",
        ),
        "null_results": (
            "1. The Connectome Topology (The Optimised Network Scaffold): is not empirical validation evidence",
        ),
        "variables": ("1_the_connectome_topology_the_optimised_network_scaffold",),
        "validation_targets": ("preserve records P0R04837-P0R04839",),
        "null_controls": (
            "1_the_connectome_topology_the_optimised_network_scaffold must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section2CytoarchitectureAndTheCanonicalMicrocircuitL4Spec:
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
class Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section2CytoarchitectureAndTheCanonicalMicrocircuitL4Spec, ...]
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


def build_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_specs(
    source_records: list[dict[str, Any]],
) -> Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle:
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

    specs: list[Section2CytoarchitectureAndTheCanonicalMicrocircuitL4Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section2CytoarchitectureAndTheCanonicalMicrocircuitL4Spec(
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
        + "2. Cytoarchitecture and the Canonical Microcircuit (L4):"
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
        "next_source_boundary": "P0R04840",
    }
    return Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_specs(
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


def render_report(bundle: Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2. Cytoarchitecture and the Canonical Microcircuit (L4):" + " Specs",
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
    bundle: Section2CytoarchitectureAndTheCanonicalMicrocircuitL4SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_validation_specs_{date_tag}.md"
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
