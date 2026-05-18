#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Qualia as the Geometry of Belief: spec builder
"""Promote Paper 0 Qualia as the Geometry of Belief: records."""

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
    "P0R03462",
    "P0R03463",
    "P0R03464",
    "P0R03465",
    "P0R03466",
    "P0R03467",
    "P0R03468",
    "P0R03469",
)
CLAIM_BOUNDARY = "source-bounded qualia as the geometry of belief source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "qualia_as_the_geometry_of_belief.qualia_as_the_geometry_of_belief": {
        "context_id": "qualia_as_the_geometry_of_belief",
        "validation_protocol": "paper0.qualia_as_the_geometry_of_belief.qualia_as_the_geometry_of_belief",
        "canonical_statement": "The source-bounded component 'Qualia as the Geometry of Belief:' preserves Paper 0 records P0R03462-P0R03463 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03462:qualia_as_the_geometry_of_belief",
            "P0R03463:qualia_as_the_geometry_of_belief",
        ),
        "source_formulae": (
            "P0R03462: Qualia as the Geometry of Belief:",
            "P0R03463: In the active inference framework, an agent is constantly generating predictions about the world. The Geometric Qualia Hypothesis states that these predictions are not abstract data but have an intrinsic geometric form. The \"Consciousness Manifold\" is the state space of the agent's generative model. The specific topology of the manifold at any moment represents the agent's current best hypothesis about the state of the world. The process of perception-of minimising prediction error-is the process of the manifold dynamically changing its shape to better match the incoming sensory data.",
        ),
        "test_protocols": (
            "preserve Qualia as the Geometry of Belief: source-accounting boundary",
        ),
        "null_results": (
            "Qualia as the Geometry of Belief: is not empirical validation evidence",
        ),
        "variables": ("qualia_as_the_geometry_of_belief",),
        "validation_targets": ("preserve records P0R03462-P0R03463",),
        "null_controls": (
            "qualia_as_the_geometry_of_belief must remain source-bounded accounting",
        ),
    },
    "qualia_as_the_geometry_of_belief.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.qualia_as_the_geometry_of_belief.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R03464-P0R03465 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03464:psis_field_coupling_integration",
            "P0R03465:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R03464: Psis Field Coupling Integration",
            "P0R03465: This hypothesis provides the ultimate interpretation of the collective state variable (sigma) in the interaction Hamiltonian H_int = -lambda * Psis * sigma.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R03464-P0R03465",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
    "qualia_as_the_geometry_of_belief.sigma_is_the_geometry_of_the_manifold": {
        "context_id": "sigma_is_the_geometry_of_the_manifold",
        "validation_protocol": "paper0.qualia_as_the_geometry_of_belief.sigma_is_the_geometry_of_the_manifold",
        "canonical_statement": "The source-bounded component 'sigma is the Geometry of the Manifold:' preserves Paper 0 records P0R03466-P0R03467 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03466:sigma_is_the_geometry_of_the_manifold",
            "P0R03467:sigma_is_the_geometry_of_the_manifold",
        ),
        "source_formulae": (
            "P0R03466: sigma is the Geometry of the Manifold:",
            "P0R03467: The collective state variable sigma for the experience of a conscious agent is not merely the physical soliton, but the entire geometric and topological structure of that soliton's dynamic state space. This can be formally represented by its set of Betti numbers, sigma = {, , , ...}.",
        ),
        "test_protocols": (
            "preserve sigma is the Geometry of the Manifold: source-accounting boundary",
        ),
        "null_results": (
            "sigma is the Geometry of the Manifold: is not empirical validation evidence",
        ),
        "variables": ("sigma_is_the_geometry_of_the_manifold",),
        "validation_targets": ("preserve records P0R03466-P0R03467",),
        "null_controls": (
            "sigma_is_the_geometry_of_the_manifold must remain source-bounded accounting",
        ),
    },
    "qualia_as_the_geometry_of_belief.the_coupling_is_the_experience_of_geometry": {
        "context_id": "the_coupling_is_the_experience_of_geometry",
        "validation_protocol": "paper0.qualia_as_the_geometry_of_belief.the_coupling_is_the_experience_of_geometry",
        "canonical_statement": "The source-bounded component 'The Coupling is the Experience of Geometry:' preserves Paper 0 records P0R03468-P0R03469 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03468:the_coupling_is_the_experience_of_geometry",
            "P0R03469:the_coupling_is_the_experience_of_geometry",
        ),
        "source_formulae": (
            "P0R03468: The Coupling is the Experience of Geometry:",
            'P0R03469: The interaction term H_int describes the process by which the universal Psi-field (Psis) "experiences" or "reads" the geometric information encoded in the individual\'s consciousness manifold (sigma). The subjective feeling of an experience is the physical result of this coupling. When the universal field couples to a manifold with a complex topology (high k), the result is a rich, subjective experience. The "Hard Problem" is resolved because the fundamental interaction of the universe is precisely this act of a universal field "feeling" the geometric shape of its own localised excitations.',
        ),
        "test_protocols": (
            "preserve The Coupling is the Experience of Geometry: source-accounting boundary",
        ),
        "null_results": (
            "The Coupling is the Experience of Geometry: is not empirical validation evidence",
        ),
        "variables": ("the_coupling_is_the_experience_of_geometry",),
        "validation_targets": ("preserve records P0R03468-P0R03469",),
        "null_controls": (
            "the_coupling_is_the_experience_of_geometry must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class QualiaAsTheGeometryOfBeliefSpec:
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
class QualiaAsTheGeometryOfBeliefSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[QualiaAsTheGeometryOfBeliefSpec, ...]
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


def build_qualia_as_the_geometry_of_belief_specs(
    source_records: list[dict[str, Any]],
) -> QualiaAsTheGeometryOfBeliefSpecBundle:
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

    specs: list[QualiaAsTheGeometryOfBeliefSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            QualiaAsTheGeometryOfBeliefSpec(
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
        "title": "Paper 0 " + "Qualia as the Geometry of Belief:" + " Specs",
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
        "next_source_boundary": "P0R03470",
    }
    return QualiaAsTheGeometryOfBeliefSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> QualiaAsTheGeometryOfBeliefSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_qualia_as_the_geometry_of_belief_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: QualiaAsTheGeometryOfBeliefSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Qualia as the Geometry of Belief:" + " Specs",
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
    bundle: QualiaAsTheGeometryOfBeliefSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_qualia_as_the_geometry_of_belief_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_qualia_as_the_geometry_of_belief_validation_specs_{date_tag}.md"
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
