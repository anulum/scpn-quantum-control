#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): spec builder
"""Promote Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): records."""

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
    "P0R04526",
    "P0R04527",
    "P0R04528",
    "P0R04529",
    "P0R04530",
    "P0R04531",
    "P0R04532",
    "P0R04533",
)
CLAIM_BOUNDARY = "source-bounded section 4 the geometry of thought the consciousness manifold m source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_the_geometry_of_thought_the_consciousness_manifold_m.4_the_geometry_of_thought_the_consciousness_manifold_m": {
        "context_id": "4_the_geometry_of_thought_the_consciousness_manifold_m",
        "validation_protocol": "paper0.section_4_the_geometry_of_thought_the_consciousness_manifold_m.4_the_geometry_of_thought_the_consciousness_manifold_m",
        "canonical_statement": "The source-bounded component '4. The Geometry of Thought (The Consciousness Manifold M):' preserves Paper 0 records P0R04526-P0R04528 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04526:4_the_geometry_of_thought_the_consciousness_manifold_m",
            "P0R04527:4_the_geometry_of_thought_the_consciousness_manifold_m",
            "P0R04528:4_the_geometry_of_thought_the_consciousness_manifold_m",
        ),
        "source_formulae": (
            "P0R04526: 4. The Geometry of Thought (The Consciousness Manifold M):",
            "P0R04527: Cognitive states and Qualia correspond to the intrinsic geometry of the high-dimensional Neural Manifold (M).",
            "P0R04528: Topological Data Analysis (TDA): Betti numbers (bk) quantify the richness of the cognitive state. | The Physics of Valence: Valence (emotional tone) corresponds to the intrinsic curvature of M and the proximity to criticality (Valencesigma1).",
        ),
        "test_protocols": (
            "preserve 4. The Geometry of Thought (The Consciousness Manifold M): source-accounting boundary",
        ),
        "null_results": (
            "4. The Geometry of Thought (The Consciousness Manifold M): is not empirical validation evidence",
        ),
        "variables": ("4_the_geometry_of_thought_the_consciousness_manifold_m",),
        "validation_targets": ("preserve records P0R04526-P0R04528",),
        "null_controls": (
            "4_the_geometry_of_thought_the_consciousness_manifold_m must remain source-bounded accounting",
        ),
    },
    "section_4_the_geometry_of_thought_the_consciousness_manifold_m.5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis": {
        "context_id": "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
        "validation_protocol": "paper0.section_4_the_geometry_of_thought_the_consciousness_manifold_m.5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
        "canonical_statement": "The source-bounded component '5. The Neural Correlates of Consciousness (NCC) - An SCPN Synthesis:' preserves Paper 0 records P0R04529-P0R04530 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04529:5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
            "P0R04530:5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
        ),
        "source_formulae": (
            "P0R04529: 5. The Neural Correlates of Consciousness (NCC) - An SCPN Synthesis:",
            'P0R04530: IIT: The degree of consciousness is , maximised in the Posterior Hot Zone. | GNW: "Ignition" (Global Neuronal Workspace activation) corresponds to the L4/L5 phase transition (SSB), enabling global broadcasting when synchronisation (R) crosses a critical threshold (RC).',
        ),
        "test_protocols": (
            "preserve 5. The Neural Correlates of Consciousness (NCC) - An SCPN Synthesis: source-accounting boundary",
        ),
        "null_results": (
            "5. The Neural Correlates of Consciousness (NCC) - An SCPN Synthesis: is not empirical validation evidence",
        ),
        "variables": ("5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",),
        "validation_targets": ("preserve records P0R04529-P0R04530",),
        "null_controls": (
            "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis must remain source-bounded accounting",
        ),
    },
    "section_4_the_geometry_of_thought_the_consciousness_manifold_m.vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes": {
        "context_id": "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
        "validation_protocol": "paper0.section_4_the_geometry_of_thought_the_consciousness_manifold_m.vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
        "canonical_statement": "The source-bounded component 'VI. The Extended Brain: Neuro-Visceral and Neuro-Immune Axes' preserves Paper 0 records P0R04531-P0R04533 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04531:vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
            "P0R04532:vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
            "P0R04533:vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
        ),
        "source_formulae": (
            "P0R04531: VI. The Extended Brain: Neuro-Visceral and Neuro-Immune Axes",
            "P0R04532: The brain is intrinsically linked to the body.",
            "P0R04533: The Neuro-Visceral Axis (Heart-Brain-Gut): The Psychosomatic Harmonics form a coupled oscillatory network (Tri-Axial UPDE), essential for emotional processing and maintaining the stability of the Self (L5). Coherence in this axis (measured by HRV) is the primary indicator of the organismal state. | The Neuro-Immune Interface (L1/L5): The immune system is integrated via the PNI axis. Emotional states (L5) modulate immune function. Quantum coherence extends to immune cells (L1).",
        ),
        "test_protocols": (
            "preserve VI. The Extended Brain: Neuro-Visceral and Neuro-Immune Axes source-accounting boundary",
        ),
        "null_results": (
            "VI. The Extended Brain: Neuro-Visceral and Neuro-Immune Axes is not empirical validation evidence",
        ),
        "variables": ("vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",),
        "validation_targets": ("preserve records P0R04531-P0R04533",),
        "null_controls": (
            "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpec:
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
class Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpec, ...]
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


def build_section_4_the_geometry_of_thought_the_consciousness_manifold_m_specs(
    source_records: list[dict[str, Any]],
) -> Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle:
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

    specs: list[Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpec(
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
        + "4. The Geometry of Thought (The Consciousness Manifold M):"
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
        "next_source_boundary": "P0R04534",
    }
    return Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_the_geometry_of_thought_the_consciousness_manifold_m_specs(
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


def render_report(bundle: Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4. The Geometry of Thought (The Consciousness Manifold M):" + " Specs",
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
    bundle: Section4TheGeometryOfThoughtTheConsciousnessManifoldMSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation_specs_{date_tag}.md"
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
