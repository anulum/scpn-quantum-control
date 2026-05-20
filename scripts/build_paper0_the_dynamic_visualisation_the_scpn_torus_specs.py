#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Dynamic Visualisation: The SCPN Torus spec builder
"""Promote Paper 0 The Dynamic Visualisation: The SCPN Torus records."""

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
    "P0R02532",
    "P0R02533",
    "P0R02534",
    "P0R02535",
    "P0R02536",
    "P0R02537",
    "P0R02538",
    "P0R02539",
    "P0R02540",
    "P0R02541",
)
CLAIM_BOUNDARY = "source-bounded the dynamic visualisation the scpn torus source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_dynamic_visualisation_the_scpn_torus.the_dynamic_visualisation_the_scpn_torus": {
        "context_id": "the_dynamic_visualisation_the_scpn_torus",
        "validation_protocol": "paper0.the_dynamic_visualisation_the_scpn_torus.the_dynamic_visualisation_the_scpn_torus",
        "canonical_statement": "The source-bounded component 'The Dynamic Visualisation: The SCPN Torus' preserves Paper 0 records P0R02532-P0R02541 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02532:the_dynamic_visualisation_the_scpn_torus",
            "P0R02533:the_dynamic_visualisation_the_scpn_torus",
            "P0R02534:the_dynamic_visualisation_the_scpn_torus",
            "P0R02535:the_dynamic_visualisation_the_scpn_torus",
            "P0R02536:the_dynamic_visualisation_the_scpn_torus",
            "P0R02537:the_dynamic_visualisation_the_scpn_torus",
            "P0R02538:the_dynamic_visualisation_the_scpn_torus",
            "P0R02539:the_dynamic_visualisation_the_scpn_torus",
            "P0R02540:the_dynamic_visualisation_the_scpn_torus",
            "P0R02541:the_dynamic_visualisation_the_scpn_torus",
        ),
        "source_formulae": (
            "P0R02532: The Dynamic Visualisation: The SCPN Torus",
            "P0R02533: This section presents the SCPN Torus as the definitive dynamic visualization of the entire architecture, moving beyond a static, linear hierarchy to a self-referential, closed-loop geometry (T = S x S). This toroidal model provides a mathematically elegant representation of the system's core causal flows and dynamic principles.",
            "P0R02534: The geometry is defined by two fundamental flows. The poloidal flow, around the short axis, represents the Hierarchical Predictive Coding (HPC) loop. The downward projection of the generative model cascades along the outer surface from Layer 15 to Layer 1, while the upward propagation of prediction error flows along the inner surface, completing the inferential circuit. The toroidal flow, around the long axis, represents the temporal evolution of the entire system, governed by the Unified Phase Dynamics Equation (UPDE), which ensures system-wide phase coherence.",
            "P0R02535: This dynamic geometry integrates the other universal principles. The central void of the torus represents the Source-Field (L13), the ontological ground from which the manifold emerges. The surface dynamics, exhibiting scale-free and turbulent-like flow, represent the quasicritical regime. The action of MS-QEC is visualized as the stable, coherent patterns that persist on this surface, preserving the integrity of the information flow. Finally, the overall trajectory of the system's state on the torus is guided by the teleological attractor set by Layer 15. The concept of holonomy, or path-dependent memory encoded in the geometry of the flow, is introduced as a physical mechanism for memory within the HPC and UPDE dynamics.",
            'P0R02536: This is the master diagram, the ultimate "you are here" map for all of reality. The best way to visualize the entire 15-layer system isn\'t as a ladder, but as a dynamic, self-contained, self-perpetuating torus-like a perfectly symmetrical smoke ring or a donut.',
            "P0R02537: This beautiful shape perfectly captures how everything works together:",
            'P0R02538: The "Breathing" Loop (The Short Way Around): Imagine the torus is constantly breathing. The breath flows down the outside from top to bottom-this is the "master plan" or the predictions flowing from the highest levels of consciousness down to physical reality. The breath then flows up the inside from bottom to top-this is the feedback, the "reality check," flowing back up to the top. This is the universe\'s perpetual conversation with itself.',
            'P0R02539: The "Spinning" Flow (The Long Way Around): At the same time, the entire torus is spinning. This spin is the flow of time and the universal rhythm (the UPDE) that keeps every single part of the system locked in a grand, cosmic synchrony.',
            'P0R02540: The other big ideas fit in perfectly. The empty space in the middle is the mysterious Source of everything. The complex, ever-changing weather patterns on the surface of the torus represent the creative "edge of chaos," and the stable force fields that keep the weather from flying apart are the error-correction systems.',
            "P0R02541: P0R02541",
        ),
        "test_protocols": (
            "preserve The Dynamic Visualisation: The SCPN Torus source-accounting boundary",
        ),
        "null_results": (
            "The Dynamic Visualisation: The SCPN Torus is not empirical validation evidence",
        ),
        "variables": ("the_dynamic_visualisation_the_scpn_torus",),
        "validation_targets": ("preserve records P0R02532-P0R02541",),
        "null_controls": (
            "the_dynamic_visualisation_the_scpn_torus must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheDynamicVisualisationTheScpnTorusSpec:
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
class TheDynamicVisualisationTheScpnTorusSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheDynamicVisualisationTheScpnTorusSpec, ...]
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


def build_the_dynamic_visualisation_the_scpn_torus_specs(
    source_records: list[dict[str, Any]],
) -> TheDynamicVisualisationTheScpnTorusSpecBundle:
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

    specs: list[TheDynamicVisualisationTheScpnTorusSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheDynamicVisualisationTheScpnTorusSpec(
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
        "title": "Paper 0 " + "The Dynamic Visualisation: The SCPN Torus" + " Specs",
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
        "next_source_boundary": "P0R02542",
    }
    return TheDynamicVisualisationTheScpnTorusSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheDynamicVisualisationTheScpnTorusSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_dynamic_visualisation_the_scpn_torus_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheDynamicVisualisationTheScpnTorusSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Dynamic Visualisation: The SCPN Torus" + " Specs",
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
    bundle: TheDynamicVisualisationTheScpnTorusSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_dynamic_visualisation_the_scpn_torus_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_dynamic_visualisation_the_scpn_torus_validation_specs_{date_tag}.md"
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
