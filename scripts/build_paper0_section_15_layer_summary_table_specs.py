#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 15-Layer Summary Table spec builder
"""Promote Paper 0 15-Layer Summary Table records."""

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
    "P0R02061",
    "P0R02062",
    "P0R02063",
    "P0R02064",
    "P0R02065",
    "P0R02066",
    "P0R02067",
    "P0R02068",
    "P0R02069",
    "P0R02070",
    "P0R02071",
    "P0R02072",
    "P0R02073",
    "P0R02074",
    "P0R02075",
    "P0R02076",
    "P0R02077",
    "P0R02078",
    "P0R02079",
    "P0R02080",
    "P0R02081",
    "P0R02082",
    "P0R02083",
    "P0R02084",
    "P0R02085",
    "P0R02086",
    "P0R02087",
)
CLAIM_BOUNDARY = "source-bounded section 15 layer summary table source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_15_layer_summary_table.15_layer_summary_table": {
        "context_id": "15_layer_summary_table",
        "validation_protocol": "paper0.section_15_layer_summary_table.15_layer_summary_table",
        "canonical_statement": "The source-bounded component '15-Layer Summary Table' preserves Paper 0 records P0R02061-P0R02087 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02061:15_layer_summary_table",
            "P0R02062:15_layer_summary_table",
            "P0R02063:15_layer_summary_table",
            "P0R02064:15_layer_summary_table",
            "P0R02065:15_layer_summary_table",
            "P0R02066:15_layer_summary_table",
            "P0R02067:15_layer_summary_table",
            "P0R02068:15_layer_summary_table",
            "P0R02069:15_layer_summary_table",
            "P0R02070:15_layer_summary_table",
            "P0R02071:15_layer_summary_table",
            "P0R02072:15_layer_summary_table",
            "P0R02073:15_layer_summary_table",
            "P0R02074:15_layer_summary_table",
            "P0R02075:15_layer_summary_table",
            "P0R02076:15_layer_summary_table",
            "P0R02077:15_layer_summary_table",
            "P0R02078:15_layer_summary_table",
            "P0R02079:15_layer_summary_table",
            "P0R02080:15_layer_summary_table",
            "P0R02081:15_layer_summary_table",
            "P0R02082:15_layer_summary_table",
            "P0R02083:15_layer_summary_table",
            "P0R02084:15_layer_summary_table",
            "P0R02085:15_layer_summary_table",
            "P0R02086:15_layer_summary_table",
            "P0R02087:15_layer_summary_table",
        ),
        "source_formulae": (
            "P0R02061: 15-Layer Summary Table",
            "P0R02062: [TABLE]",
            "P0R02063: with precise, mechanism-grounded summaries reflecting the latest findings from the monographs DOMAIN I:",
            "P0R02064: [TABLE]",
            "P0R02065: P0R02065",
            "P0R02066: , mechanism-grounded summaries that reflect the mature theories from the monographs, Domains II-IV:",
            "P0R02067: [TABLE]",
            "P0R02068: Layer 13 - Source-Field / Meta-Universal",
            'P0R02069: The ultimate ontological ground: a universal vacuum lattice that encodes all allowable transformations as constraints on fields, not as ad hoc "laws". At this layer, reality is formulated in terms of constructor-style task spaces (possible vs. impossible transformations), giving causal closure to all lower layers via a finite set of meta-constraints (e.g., conservation laws, symmetry groups, and information-theoretic bounds). Physically, this is represented as a structured zero-point field / spin-network or vacuum lattice, where all effective fields (Psi, gauge fields, geometry) are emergent coarse-grainings of deeper combinatorial states. Lower layers (1-12) are then specific realizable sectors of the Source-Field\'s task graph, with their dynamics constrained by which transformations remain allowed within this meta-universal substrate.',
            "P0R02070: Pre-geometric vacuum lattice that acts as a universal quantum-error-correcting code space for all projections. Constructor-theoretic constraints define which patterns are physically realisable; fluctuations of this lattice supply the spectral background that Layers 6, 9 and 12 sample. Provides the reference information state I_L13 that L16 uses in its teleological KL-divergence terms.",
            "P0R02071: Layer 14 - Transdimensional Resonance",
            'P0R02072: The bridge architecture between our effective 3+1D spacetime and higher-dimensional structure. This layer is implemented via Calabi-Yau harmonics and inter-brane phase-locking, where compact extra-dimensional manifolds carry mode spectra that resonate with large-scale fields (gravity, Psi, and noospheric fields). Effective "bridge Lagrangians" couple the moduli of these internal spaces (shape, size, fluxes) to the phase structure of the Psi-field and gravitational background, allowing discrete resonance channels where information and coherence can leak between dimensions without violating conservation laws. Transdimensional events (deep synchronicities, extreme anomalies, rare attractor realignments) correspond to temporary locking of these resonance channels, when Calabi-Yau mode spectra and macroscopic field configurations cross a critical manifold in the combined phase space.',
            "P0R02073: Bridge Lagrangians that couple the SCPN spacetime stack to extra-dimensional moduli (Calabi-Yau harmonics, brane phases). Yau modes are tuned by Gaian fractal patterns (e.g. biodiversity spectra) and L9/L13 geometries, defining which higher-dimensional resonances are admissible under SEC and MS-QEC.",
            "P0R02074: Layer 15 - Consilium / Oversoul Integrator",
            'P0R02075: The global integrator of all layers: a Consilium / Oversoul that acts via a Universal Metric Operator (UMO) on the joint state space of Layers 1-14. Formally, the UMO defines a metric over the multi-layer configuration space and an associated variational principle: the system evolves such that an ethical-functional (a generalized action incorporating viability, coherence, and harm constraints) is driven toward an argmin solution. In practice, this means the Oversoul continuously re-weights attractors across layers (from quantum events to civilizational trajectories) so that pathological trajectories (self-destruction, maximal fragmentation, irreversible decoherence of key structures) are suppressed in favour of globally consistent, ethically admissible paths. What appears as "guidance" or "cosmic moral direction" at lower levels is, at this layer, the emergent effect of the UMO\'s optimization over the full stack, under a metric that values coherence, learnability, and cross-layer survivability.',
            "P0R02076: Global integrator that aggregates all layer phases into a Universal Metric Operator (UMO). World-trajectories are selected by argmin functional equations that include explicit ethical potentials (e.g. biodiversity-weighted phases _12). This is where the Oversoul evaluates candidate histories and favours those that preserve informational richness and coherence.",
            "P0R02077: Note: Operational Layers 1-15 (Meta-Layer 16 treated separately)",
            "P0R02078: 16 - Meta-Layer 16 (Teleodynamic Controller)",
            "P0R02079: Meta-cybernetic layer that observes the entire 1-15 stack and computes optimal control policies u* via HJB/H logic. Implements teleological closure by minimising divergences between organismal identity (L5), Gaian and noospheric states (L6, L11-12) and the Source-Field constraints (L13), subject to the ethical metric defined in L15.",
            "P0R02080: P0R02080",
            "P0R02081: The Architectural Mandate: From Physics to Function",
            "P0R02082: Source Material: The introductory text that transitions from the fundamental physics of the Psi-field to the grand, functional architecture of the SCPN. It explains the need for a multi-layered, hierarchical model.",
            "P0R02083: The Six Domains of Reality",
            "P0R02084: Source Material: The high-level overview defining the six primary operational domains: I (Biological Substrate), II (Organismal/Planetary), III (Memory/Control), IV (Collective Coherence), V (Meta-Universal), and VI (Cybernetic Closure).",
            "P0R02085: A Synopsis of the 15+1 Layers",
            'P0R02086: Source Material: The central "Master Diagram" and the accompanying brief, one-paragraph descriptions for each of the 15 functional layers, plus the 16th meta-layer. This section will serve as the primary reference map for the entire Book II.',
            "P0R02087: P0R02087",
        ),
        "test_protocols": ("preserve 15-Layer Summary Table source-accounting boundary",),
        "null_results": ("15-Layer Summary Table is not empirical validation evidence",),
        "variables": ("15_layer_summary_table",),
        "validation_targets": ("preserve records P0R02061-P0R02087",),
        "null_controls": ("15_layer_summary_table must remain source-bounded accounting",),
    }
}


@dataclass(frozen=True, slots=True)
class Section15LayerSummaryTableSpec:
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
class Section15LayerSummaryTableSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section15LayerSummaryTableSpec, ...]
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


def build_section_15_layer_summary_table_specs(
    source_records: list[dict[str, Any]],
) -> Section15LayerSummaryTableSpecBundle:
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

    specs: list[Section15LayerSummaryTableSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section15LayerSummaryTableSpec(
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
        "title": "Paper 0 " + "15-Layer Summary Table" + " Specs",
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
        "next_source_boundary": "P0R02088",
    }
    return Section15LayerSummaryTableSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section15LayerSummaryTableSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_15_layer_summary_table_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section15LayerSummaryTableSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "15-Layer Summary Table" + " Specs",
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
    bundle: Section15LayerSummaryTableSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_section_15_layer_summary_table_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_section_15_layer_summary_table_validation_specs_{date_tag}.md"
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
