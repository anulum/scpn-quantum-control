#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. Pathology (Chronodisruption): spec builder
"""Promote Paper 0 3. Pathology (Chronodisruption): records."""

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
    "P0R04983",
    "P0R04984",
    "P0R04985",
    "P0R04986",
    "P0R04987",
    "P0R04988",
    "P0R04989",
    "P0R04990",
    "P0R04991",
    "P0R04992",
)
CLAIM_BOUNDARY = "source-bounded section 3 pathology chronodisruption source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_pathology_chronodisruption.3_pathology_chronodisruption": {
        "context_id": "3_pathology_chronodisruption",
        "validation_protocol": "paper0.section_3_pathology_chronodisruption.3_pathology_chronodisruption",
        "canonical_statement": "The source-bounded component '3. Pathology (Chronodisruption):' preserves Paper 0 records P0R04983-P0R04985 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04983:3_pathology_chronodisruption",
            "P0R04984:3_pathology_chronodisruption",
            "P0R04985:3_pathology_chronodisruption",
        ),
        "source_formulae": (
            "P0R04983: 3. Pathology (Chronodisruption):",
            "P0R04984: Misalignment between internal (L4) and external (L6/L8) rhythms (e.g., Jet Lag, Shift Work) induces systemic stress (High F), impairs criticality, and promotes inflammation (Decoherence).",
            "P0R04985: P0R04985",
        ),
        "test_protocols": (
            "preserve 3. Pathology (Chronodisruption): source-accounting boundary",
        ),
        "null_results": ("3. Pathology (Chronodisruption): is not empirical validation evidence",),
        "variables": ("3_pathology_chronodisruption",),
        "validation_targets": ("preserve records P0R04983-P0R04985",),
        "null_controls": ("3_pathology_chronodisruption must remain source-bounded accounting",),
    },
    "section_3_pathology_chronodisruption.iii_the_sensory_motor_loop_action_perception_and_agency": {
        "context_id": "iii_the_sensory_motor_loop_action_perception_and_agency",
        "validation_protocol": "paper0.section_3_pathology_chronodisruption.iii_the_sensory_motor_loop_action_perception_and_agency",
        "canonical_statement": "The source-bounded component 'III. The Sensory-Motor Loop: Action, Perception, and Agency' preserves Paper 0 records P0R04986-P0R04989 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04986:iii_the_sensory_motor_loop_action_perception_and_agency",
            "P0R04987:iii_the_sensory_motor_loop_action_perception_and_agency",
            "P0R04988:iii_the_sensory_motor_loop_action_perception_and_agency",
            "P0R04989:iii_the_sensory_motor_loop_action_perception_and_agency",
        ),
        "source_formulae": (
            "P0R04986: III. The Sensory-Motor Loop: Action, Perception, and Agency",
            "P0R04987: The interaction with the environment is a closed loop formalised by HPC and Active Inference. The system minimises F via Perception (updating the model) or Action (acting on the world).",
            "P0R04988: [IMAGE:]",
            "P0R04989: Fig.: Perception-action cycle: generative model -> predictions -> data -> errors -> policy/action. Geometric transduction (L1->L5) maps signals into the Consciousness Manifold MMM.",
        ),
        "test_protocols": (
            "preserve III. The Sensory-Motor Loop: Action, Perception, and Agency source-accounting boundary",
        ),
        "null_results": (
            "III. The Sensory-Motor Loop: Action, Perception, and Agency is not empirical validation evidence",
        ),
        "variables": ("iii_the_sensory_motor_loop_action_perception_and_agency",),
        "validation_targets": ("preserve records P0R04986-P0R04989",),
        "null_controls": (
            "iii_the_sensory_motor_loop_action_perception_and_agency must remain source-bounded accounting",
        ),
    },
    "section_3_pathology_chronodisruption.1_perception_as_geometric_transduction_l1_l5": {
        "context_id": "1_perception_as_geometric_transduction_l1_l5",
        "validation_protocol": "paper0.section_3_pathology_chronodisruption.1_perception_as_geometric_transduction_l1_l5",
        "canonical_statement": "The source-bounded component '1. Perception as Geometric Transduction (L1-L5)' preserves Paper 0 records P0R04990-P0R04992 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04990:1_perception_as_geometric_transduction_l1_l5",
            "P0R04991:1_perception_as_geometric_transduction_l1_l5",
            "P0R04992:1_perception_as_geometric_transduction_l1_l5",
        ),
        "source_formulae": (
            "P0R04990: 1. Perception as Geometric Transduction (L1-L5)",
            "P0R04991: Perception is the active process of transducing external physical signals into the intrinsic geometry of the Consciousness Manifold (M, L5).",
            "P0R04992: Quantum Transduction (L1): Sensory receptors may utilise quantum enhancement (e.g., Retinal isomerisation in vision, VAT in olfaction, RPM in magnetoreception). | Hierarchical Inference (HPC): Sensory information (Prediction Errors) ascends the hierarchy, updating the Generative Model.",
        ),
        "test_protocols": (
            "preserve 1. Perception as Geometric Transduction (L1-L5) source-accounting boundary",
        ),
        "null_results": (
            "1. Perception as Geometric Transduction (L1-L5) is not empirical validation evidence",
        ),
        "variables": ("1_perception_as_geometric_transduction_l1_l5",),
        "validation_targets": ("preserve records P0R04990-P0R04992",),
        "null_controls": (
            "1_perception_as_geometric_transduction_l1_l5 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3PathologyChronodisruptionSpec:
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
class Section3PathologyChronodisruptionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3PathologyChronodisruptionSpec, ...]
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


def build_section_3_pathology_chronodisruption_specs(
    source_records: list[dict[str, Any]],
) -> Section3PathologyChronodisruptionSpecBundle:
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

    specs: list[Section3PathologyChronodisruptionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3PathologyChronodisruptionSpec(
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
        "title": "Paper 0 " + "3. Pathology (Chronodisruption):" + " Specs",
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
        "next_source_boundary": "P0R04993",
    }
    return Section3PathologyChronodisruptionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3PathologyChronodisruptionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_pathology_chronodisruption_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section3PathologyChronodisruptionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. Pathology (Chronodisruption):" + " Specs",
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
    bundle: Section3PathologyChronodisruptionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_pathology_chronodisruption_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_section_3_pathology_chronodisruption_validation_specs_{date_tag}.md"
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
