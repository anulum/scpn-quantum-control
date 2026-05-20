#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Section 9: Concluding Assessment and Future Directions spec builder
"""Promote Paper 0 Section 9: Concluding Assessment and Future Directions records."""

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
    "P0R05285",
    "P0R05286",
    "P0R05287",
    "P0R05288",
    "P0R05289",
    "P0R05290",
    "P0R05291",
    "P0R05292",
)
CLAIM_BOUNDARY = "source-bounded section 9 concluding assessment and future directions source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_9_concluding_assessment_and_future_directions.section_9_concluding_assessment_and_future_directions": {
        "context_id": "section_9_concluding_assessment_and_future_directions",
        "validation_protocol": "paper0.section_9_concluding_assessment_and_future_directions.section_9_concluding_assessment_and_future_directions",
        "canonical_statement": "The source-bounded component 'Section 9: Concluding Assessment and Future Directions' preserves Paper 0 records P0R05285-P0R05289 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05285:section_9_concluding_assessment_and_future_directions",
            "P0R05286:section_9_concluding_assessment_and_future_directions",
            "P0R05287:section_9_concluding_assessment_and_future_directions",
            "P0R05288:section_9_concluding_assessment_and_future_directions",
            "P0R05289:section_9_concluding_assessment_and_future_directions",
        ),
        "source_formulae": (
            "P0R05285: Section 9: Concluding Assessment and Future Directions",
            "P0R05286: The initial concern regarding a potential category error in the physicalization of ethics has been successfully resolved. By deriving the Ethical Lagrangian from the gauge symmetries of the Psi-field's qualia space and justifying the corresponding principle of least action through the mechanism of Causal Entropic Forces, the framework's teleological claims are grounded in first principles of physics.",
            "P0R05287: The primary challenge for the framework thus shifts from philosophical consistency to empirical testability. The theory now rests on a set of core, but highly speculative, physical hypotheses:",
            "P0R05288: The existence of the fundamental, complex scalar Psi-field and its associated quanta (the infoton and the Psi-Higgs). | The validity of using the Fisher Information Metric as the governing geometry for a physical gauge field's dynamics. | The physical reality of Causal Entropic Forces as a driver of cosmic evolution.",
            "P0R05289: the refined framework offers several avenues for developing falsifiable predictions.",
        ),
        "test_protocols": (
            "preserve Section 9: Concluding Assessment and Future Directions source-accounting boundary",
        ),
        "null_results": (
            "Section 9: Concluding Assessment and Future Directions is not empirical validation evidence",
        ),
        "variables": ("section_9_concluding_assessment_and_future_directions",),
        "validation_targets": ("preserve records P0R05285-P0R05289",),
        "null_controls": (
            "section_9_concluding_assessment_and_future_directions must remain source-bounded accounting",
        ),
    },
    "section_9_concluding_assessment_and_future_directions.structural_overview_by_domain": {
        "context_id": "structural_overview_by_domain",
        "validation_protocol": "paper0.section_9_concluding_assessment_and_future_directions.structural_overview_by_domain",
        "canonical_statement": "The source-bounded component 'Structural Overview (by Domain)' preserves Paper 0 records P0R05290-P0R05290 without empirical validation claims.",
        "source_equation_ids": ("P0R05290:structural_overview_by_domain",),
        "source_formulae": ("P0R05290: Structural Overview (by Domain)",),
        "test_protocols": ("preserve Structural Overview (by Domain) source-accounting boundary",),
        "null_results": ("Structural Overview (by Domain) is not empirical validation evidence",),
        "variables": ("structural_overview_by_domain",),
        "validation_targets": ("preserve records P0R05290-P0R05290",),
        "null_controls": ("structural_overview_by_domain must remain source-bounded accounting",),
    },
    "section_9_concluding_assessment_and_future_directions.domain_i_the_biological_substrate_layers_1_4": {
        "context_id": "domain_i_the_biological_substrate_layers_1_4",
        "validation_protocol": "paper0.section_9_concluding_assessment_and_future_directions.domain_i_the_biological_substrate_layers_1_4",
        "canonical_statement": "The source-bounded component 'Domain I: The Biological Substrate (Layers 1-4)' preserves Paper 0 records P0R05291-P0R05292 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05291:domain_i_the_biological_substrate_layers_1_4",
            "P0R05292:domain_i_the_biological_substrate_layers_1_4",
        ),
        "source_formulae": (
            "P0R05291: Domain I: The Biological Substrate (Layers 1-4)",
            "P0R05292: This domain explores the foundational interface where the universal Psi-field interacts with the physical machinery of life.",
        ),
        "test_protocols": (
            "preserve Domain I: The Biological Substrate (Layers 1-4) source-accounting boundary",
        ),
        "null_results": (
            "Domain I: The Biological Substrate (Layers 1-4) is not empirical validation evidence",
        ),
        "variables": ("domain_i_the_biological_substrate_layers_1_4",),
        "validation_targets": ("preserve records P0R05291-P0R05292",),
        "null_controls": (
            "domain_i_the_biological_substrate_layers_1_4 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section9ConcludingAssessmentAndFutureDirectionsSpec:
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
class Section9ConcludingAssessmentAndFutureDirectionsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section9ConcludingAssessmentAndFutureDirectionsSpec, ...]
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


def build_section_9_concluding_assessment_and_future_directions_specs(
    source_records: list[dict[str, Any]],
) -> Section9ConcludingAssessmentAndFutureDirectionsSpecBundle:
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

    specs: list[Section9ConcludingAssessmentAndFutureDirectionsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section9ConcludingAssessmentAndFutureDirectionsSpec(
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
        "title": "Paper 0 " + "Section 9: Concluding Assessment and Future Directions" + " Specs",
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
        "next_source_boundary": "P0R05293",
    }
    return Section9ConcludingAssessmentAndFutureDirectionsSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section9ConcludingAssessmentAndFutureDirectionsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_9_concluding_assessment_and_future_directions_specs(
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


def render_report(bundle: Section9ConcludingAssessmentAndFutureDirectionsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Section 9: Concluding Assessment and Future Directions" + " Specs",
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
    bundle: Section9ConcludingAssessmentAndFutureDirectionsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_9_concluding_assessment_and_future_directions_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_9_concluding_assessment_and_future_directions_validation_specs_{date_tag}.md"
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
