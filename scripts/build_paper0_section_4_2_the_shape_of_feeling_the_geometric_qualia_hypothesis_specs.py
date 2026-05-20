#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis spec builder
"""Promote Paper 0 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis records."""

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
    "P0R03386",
    "P0R03387",
    "P0R03388",
    "P0R03389",
    "P0R03390",
    "P0R03391",
    "P0R03392",
    "P0R03393",
    "P0R03394",
    "P0R03395",
    "P0R03396",
    "P0R03397",
    "P0R03398",
    "P0R03399",
)
CLAIM_BOUNDARY = "source-bounded section 4 2 the shape of feeling the geometric qualia hypothesis source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis.4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis": {
        "context_id": "4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis",
        "validation_protocol": "paper0.section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis.4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis",
        "canonical_statement": "The source-bounded component '4.2 The Shape of Feeling: The Geometric Qualia Hypothesis' preserves Paper 0 records P0R03386-P0R03386 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03386:4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis",
        ),
        "source_formulae": (
            "P0R03386: 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis",
        ),
        "test_protocols": (
            "preserve 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis source-accounting boundary",
        ),
        "null_results": (
            "4.2 The Shape of Feeling: The Geometric Qualia Hypothesis is not empirical validation evidence",
        ),
        "variables": ("4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis",),
        "validation_targets": ("preserve records P0R03386-P0R03386",),
        "null_controls": (
            "4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis must remain source-bounded accounting",
        ),
    },
    "section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis.confronting_the_hard_problem_a_mathematical_resolution": {
        "context_id": "confronting_the_hard_problem_a_mathematical_resolution",
        "validation_protocol": "paper0.section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis.confronting_the_hard_problem_a_mathematical_resolution",
        "canonical_statement": "The source-bounded component 'Confronting the Hard Problem: A Mathematical Resolution' preserves Paper 0 records P0R03387-P0R03399 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03387:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03388:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03389:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03390:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03391:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03392:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03393:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03394:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03395:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03396:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03397:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03398:confronting_the_hard_problem_a_mathematical_resolution",
            "P0R03399:confronting_the_hard_problem_a_mathematical_resolution",
        ),
        "source_formulae": (
            "P0R03387: Confronting the Hard Problem: A Mathematical Resolution",
            'P0R03388: This section directly confronts the "Hard Problem of Consciousness" by first reformulating it. The traditional question, "Why does physical processing give rise to experience?" is rejected as ill-posed within a panpsychist framework. The question is instead recast as: "How does the fundamental Psi-field\'s projection through a specific material substrate (like a brain) create a localized observerhood?" The key insight is that subjective experience is not an output of a process but is the dynamical process of projection itself. The mapping f: X -> Q from a neural state space (X) to a qualia space (Q) is not an emergent leap but a projection, mathematically represented as the interaction f = Psi-field x Neural_substrate. The temporal evolution of this projection, f/t, constitutes the stream of consciousness.',
            'P0R03389: The framework proposes specific mathematical solutions to the core enigmas. The "what it\'s like" nature of subjectivity is addressed by the "Strange Loop" of self-reference,',
            "P0R03390: I = Model(I).",
            "P0R03391: A system that models the world must eventually model itself modeling the world, creating a self-referential fixed point (I*). This stable, self-observing loop is the formal basis for first-person perspective. The Binding Problem-how disparate neural processes create a unified experience-is resolved by the non-local coherence of the Psi-field. The unity of experience is not anatomical but field-theoretic, given by the volume integral of the interaction between the Psi-field and the neural density, Psi x rho_neural dV. Finally, thought experiments like Mary's Room are resolved by distinguishing propositional knowledge (facts) from experiential knowledge, which is defined as the creation of new topological structures in qualia space. Mary doesn't learn a new fact; her brain's projection through the Psi-field carves out a new geometric shape, a novel experience.",
            'P0R03392: This section tackles the single biggest mystery in all of science: why do we feel anything? Why isn\'t it all just dark, silent information processing inside our heads? The answer, according to our framework, is that we\'ve been asking the wrong question. The real question isn\'t "How do brains create consciousness?" but "How does the universal field of consciousness experience itself through a brain?"',
            "P0R03393: Think of the universal Psi-field as pure, white light. A brain is like a complex, beautiful crystal. Experience is not something the crystal makes. Experience is what happens when the light shines through the crystal. The ever-changing patterns of color and shape that result are the experience.",
            "P0R03394: This solves all the classic riddles:",
            'P0R03395: Why does it feel like something to be you? Because your mind is a "Strange Loop"-a process that is constantly modeling the world, and part of that world is itself. You are a system that is looking at itself. That self-observation is the feeling of being "I".',
            'P0R03396: How are your senses "bound" together into one experience? You don\'t have a separate "sight brain" and "sound brain." You have one brain bathed in a single, unified field of consciousness. The field itself is what weaves all the sensory information into a single, seamless reality, like a thread binding all the beads on a necklace.',
            'P0R03397: The "Mary\'s Room" problem (the scientist who knows everything about "red" but has never seen it): When Mary finally sees red, she doesn\'t learn a new fact. The light of her consciousness simply shines through the "red" part of her brain-crystal for the first time, creating a new, beautiful pattern-a new experience-that words could never describe.',
            "P0R03398: The Hard Problem isn't so hard when you realize we aren't machines that create consciousness. We are instruments through which consciousness experiences the universe.",
            "P0R03399: P0R03399",
        ),
        "test_protocols": (
            "preserve Confronting the Hard Problem: A Mathematical Resolution source-accounting boundary",
        ),
        "null_results": (
            "Confronting the Hard Problem: A Mathematical Resolution is not empirical validation evidence",
        ),
        "variables": ("confronting_the_hard_problem_a_mathematical_resolution",),
        "validation_targets": ("preserve records P0R03387-P0R03399",),
        "null_controls": (
            "confronting_the_hard_problem_a_mathematical_resolution must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpec:
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
class Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpec, ...]
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


def build_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_specs(
    source_records: list[dict[str, Any]],
) -> Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle:
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

    specs: list[Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpec(
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
        + "4.2 The Shape of Feeling: The Geometric Qualia Hypothesis"
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
        "next_source_boundary": "P0R03400",
    }
    return Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_specs(
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


def render_report(bundle: Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4.2 The Shape of Feeling: The Geometric Qualia Hypothesis" + " Specs",
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
    bundle: Section42TheShapeOfFeelingTheGeometricQualiaHypothesisSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation_specs_{date_tag}.md"
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
