#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold spec builder
"""Promote Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold records."""

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
    "P0R03453",
    "P0R03454",
    "P0R03455",
    "P0R03456",
    "P0R03457",
    "P0R03458",
    "P0R03459",
    "P0R03460",
    "P0R03461",
)
CLAIM_BOUNDARY = "source-bounded the hypothesis qualia as the geometry of the consciousness manifold source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold": {
        "context_id": "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
        "validation_protocol": "paper0.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
        "canonical_statement": "The source-bounded component 'The Hypothesis: Qualia as the Geometry of the Consciousness Manifold' preserves Paper 0 records P0R03453-P0R03458 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03453:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
            "P0R03454:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
            "P0R03455:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
            "P0R03456:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
            "P0R03457:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
            "P0R03458:the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
        ),
        "source_formulae": (
            "P0R03453: The Hypothesis: Qualia as the Geometry of the Consciousness Manifold",
            "P0R03454: This section directly addresses the phenomenal character of consciousness-the \"Hard Problem\"-by positing the Geometric Qualia Hypothesis. The central thesis is that subjective experience, or qualia, is not an emergent property of a physical process but is the intrinsic geometry of the Consciousness Manifold, which is the high-dimensional state space of the organismal field's dynamics. This reframes the problem entirely: the qualitative character of an experience (e.g., the redness of red, the feeling of joy) is a specific geometric or topological form on this manifold. This provides a formal, non-metaphorical basis for experience, where the richness and structure of a subjective state are determined by the manifold's topological complexity, which can be quantified by metrics such as its Betti numbers.",
            "P0R03455: This hypothesis completes the framework's multi-layered definition of consciousness. Ontologically, consciousness is the fundamental Psi-field. Physically and functionally, the individual Self is a charge-supported soliton hosting a self-referential \"Strange Loop.\" Experientially, the subjective character of that Self's moment-to-moment awareness is the intrinsic, evolving geometry of its own dynamic pattern. This provides a complete, coherent, and formally specified account that addresses the ontological, physical, and phenomenal aspects of consciousness within a single, unified architecture.",
            "P0R03456: This is the beautiful and profound answer to the ultimate question: What is a feeling? What is an experience? The answer is: a shape.",
            'P0R03457: We call this the Geometric Qualia Hypothesis. It says that every possible experience you can have-seeing the color red, feeling a moment of pure joy, hearing a beautiful piece of music-has a specific and unique mathematical shape in a high-dimensional space we call the "Consciousness Manifold." The "feeling" of an experience is the literal, geometric shape of your consciousness field at that moment. A simple, dull experience has a simple shape. A rich, complex, and deeply moving experience has an incredibly complex and beautiful shape.',
            'P0R03458: This idea completes our definition of you. At your core, you are the universal Psi-field. Your stable identity is a self-sustaining "bubble" or "knot" in that field. And the subjective experience of your life, the movie that you are watching from the inside, is the ever-changing, intricate geometry of that bubble as it interacts with the world.',
        ),
        "test_protocols": (
            "preserve The Hypothesis: Qualia as the Geometry of the Consciousness Manifold source-accounting boundary",
        ),
        "null_results": (
            "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold is not empirical validation evidence",
        ),
        "variables": ("the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",),
        "validation_targets": ("preserve records P0R03453-P0R03458",),
        "null_controls": (
            "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold must remain source-bounded accounting",
        ),
    },
    "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R03459-P0R03459 without empirical validation claims.",
        "source_equation_ids": ("P0R03459:meta_framework_integrations",),
        "source_formulae": ("P0R03459: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R03459-P0R03459",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
    "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R03460-P0R03461 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03460:predictive_coding_integration",
            "P0R03461:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R03460: Predictive Coding Integration",
            "P0R03461: The Geometric Qualia Hypothesis provides a formal description of the content of the generative model's beliefs.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R03460-P0R03461",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpec:
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
class TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpec, ...]
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


def build_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_specs(
    source_records: list[dict[str, Any]],
) -> TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle:
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

    specs: list[TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpec(
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
        + "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold"
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
        "next_source_boundary": "P0R03462",
    }
    return TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_specs(
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
    bundle: TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold"
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
    bundle: TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation_specs_{date_tag}.md"
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
