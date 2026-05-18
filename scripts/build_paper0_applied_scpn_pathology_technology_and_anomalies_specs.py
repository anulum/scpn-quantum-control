#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Applied SCPN: Pathology, Technology, and Anomalies spec builder
"""Promote Paper 0 Applied SCPN: Pathology, Technology, and Anomalies records."""

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
    "P0R06197",
    "P0R06198",
    "P0R06199",
    "P0R06200",
    "P0R06201",
    "P0R06202",
    "P0R06203",
    "P0R06204",
    "P0R06205",
)
CLAIM_BOUNDARY = "source-bounded applied scpn pathology technology and anomalies source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied_scpn_pathology_technology_and_anomalies.applied_scpn_pathology_technology_and_anomalies": {
        "context_id": "applied_scpn_pathology_technology_and_anomalies",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.applied_scpn_pathology_technology_and_anomalies",
        "canonical_statement": "The source-bounded component 'Applied SCPN: Pathology, Technology, and Anomalies' preserves Paper 0 records P0R06197-P0R06197 without empirical validation claims.",
        "source_equation_ids": ("P0R06197:applied_scpn_pathology_technology_and_anomalies",),
        "source_formulae": ("P0R06197: Applied SCPN: Pathology, Technology, and Anomalies",),
        "test_protocols": (
            "preserve Applied SCPN: Pathology, Technology, and Anomalies source-accounting boundary",
        ),
        "null_results": (
            "Applied SCPN: Pathology, Technology, and Anomalies is not empirical validation evidence",
        ),
        "variables": ("applied_scpn_pathology_technology_and_anomalies",),
        "validation_targets": ("preserve records P0R06197-P0R06197",),
        "null_controls": (
            "applied_scpn_pathology_technology_and_anomalies must remain source-bounded accounting",
        ),
    },
    "applied_scpn_pathology_technology_and_anomalies.i_pathology_and_therapeutics": {
        "context_id": "i_pathology_and_therapeutics",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.i_pathology_and_therapeutics",
        "canonical_statement": "The source-bounded component 'I. Pathology and Therapeutics' preserves Paper 0 records P0R06198-P0R06199 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06198:i_pathology_and_therapeutics",
            "P0R06199:i_pathology_and_therapeutics",
        ),
        "source_formulae": (
            "P0R06198: I. Pathology and Therapeutics",
            "P0R06199: Pathology is defined as a systemic breakdown of coherence, deviation from the quasicritical regime, or failure of error correction (MS-QEC).",
        ),
        "test_protocols": ("preserve I. Pathology and Therapeutics source-accounting boundary",),
        "null_results": ("I. Pathology and Therapeutics is not empirical validation evidence",),
        "variables": ("i_pathology_and_therapeutics",),
        "validation_targets": ("preserve records P0R06198-P0R06199",),
        "null_controls": ("i_pathology_and_therapeutics must remain source-bounded accounting",),
    },
    "applied_scpn_pathology_technology_and_anomalies.aetiology_of_disorder": {
        "context_id": "aetiology_of_disorder",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.aetiology_of_disorder",
        "canonical_statement": "The source-bounded component 'Aetiology of Disorder:' preserves Paper 0 records P0R06200-P0R06200 without empirical validation claims.",
        "source_equation_ids": ("P0R06200:aetiology_of_disorder",),
        "source_formulae": ("P0R06200: Aetiology of Disorder:",),
        "test_protocols": ("preserve Aetiology of Disorder: source-accounting boundary",),
        "null_results": ("Aetiology of Disorder: is not empirical validation evidence",),
        "variables": ("aetiology_of_disorder",),
        "validation_targets": ("preserve records P0R06200-P0R06200",),
        "null_controls": ("aetiology_of_disorder must remain source-bounded accounting",),
    },
    "applied_scpn_pathology_technology_and_anomalies.dissonance_free_energy_accumulation_sustained_accumulation_of_prediction": {
        "context_id": "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",
        "canonical_statement": "The source-bounded component 'Dissonance (Free Energy Accumulation): Sustained accumulation of Prediction Errors (EL) in the HPC architecture. PathologyIndexFGlobal.' preserves Paper 0 records P0R06201-P0R06201 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06201:dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",
        ),
        "source_formulae": (
            "P0R06201: Dissonance (Free Energy Accumulation): Sustained accumulation of Prediction Errors (EL) in the HPC architecture. PathologyIndexFGlobal.",
        ),
        "test_protocols": (
            "preserve Dissonance (Free Energy Accumulation): Sustained accumulation of Prediction Errors (EL) in the HPC architecture. PathologyIndexFGlobal. source-accounting boundary",
        ),
        "null_results": (
            "Dissonance (Free Energy Accumulation): Sustained accumulation of Prediction Errors (EL) in the HPC architecture. PathologyIndexFGlobal. is not empirical validation evidence",
        ),
        "variables": ("dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",),
        "validation_targets": ("preserve records P0R06201-P0R06201",),
        "null_controls": (
            "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction must remain source-bounded accounting",
        ),
    },
    "applied_scpn_pathology_technology_and_anomalies.deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig": {
        "context_id": "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",
        "canonical_statement": "The source-bounded component 'Deviation from Criticality: Shifts away from sigma=1. Supercriticality (sigma > 1, e.g., mania, seizures); Subcriticality (sigma < 1, e.g., depression, coma).' preserves Paper 0 records P0R06202-P0R06202 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06202:deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",
        ),
        "source_formulae": (
            "P0R06202: Deviation from Criticality: Shifts away from sigma=1. Supercriticality (sigma > 1, e.g., mania, seizures); Subcriticality (sigma < 1, e.g., depression, coma).",
        ),
        "test_protocols": (
            "preserve Deviation from Criticality: Shifts away from sigma=1. Supercriticality (sigma > 1, e.g., mania, seizures); Subcriticality (sigma < 1, e.g., depression, coma). source-accounting boundary",
        ),
        "null_results": (
            "Deviation from Criticality: Shifts away from sigma=1. Supercriticality (sigma > 1, e.g., mania, seizures); Subcriticality (sigma < 1, e.g., depression, coma). is not empirical validation evidence",
        ),
        "variables": ("deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",),
        "validation_targets": ("preserve records P0R06202-P0R06202",),
        "null_controls": (
            "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig must remain source-bounded accounting",
        ),
    },
    "applied_scpn_pathology_technology_and_anomalies.fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc": {
        "context_id": "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
        "validation_protocol": "paper0.applied_scpn_pathology_technology_and_anomalies.fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
        "canonical_statement": "The source-bounded component 'Fragmentation: Failure of integration (e.g., L5 Self fragmentation in dissociation; L11 societal polarisation).' preserves Paper 0 records P0R06203-P0R06205 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06203:fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
            "P0R06204:fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
            "P0R06205:fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
        ),
        "source_formulae": (
            "P0R06203: Fragmentation: Failure of integration (e.g., L5 Self fragmentation in dissociation; L11 societal polarisation).",
            "P0R06204: Therapeutic Principles:",
            "P0R06205: Therapies aim to restore the optimal state: Minimising Free Energy (resolving prediction errors), Restoring Criticality (tuning sigma->1), and Enhancing Synchronisation (Phase Resetting).",
        ),
        "test_protocols": (
            "preserve Fragmentation: Failure of integration (e.g., L5 Self fragmentation in dissociation; L11 societal polarisation). source-accounting boundary",
        ),
        "null_results": (
            "Fragmentation: Failure of integration (e.g., L5 Self fragmentation in dissociation; L11 societal polarisation). is not empirical validation evidence",
        ),
        "variables": ("fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",),
        "validation_targets": ("preserve records P0R06203-P0R06205",),
        "null_controls": (
            "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AppliedScpnPathologyTechnologyAndAnomaliesSpec:
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
class AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[AppliedScpnPathologyTechnologyAndAnomaliesSpec, ...]
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


def build_applied_scpn_pathology_technology_and_anomalies_specs(
    source_records: list[dict[str, Any]],
) -> AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle:
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

    specs: list[AppliedScpnPathologyTechnologyAndAnomaliesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AppliedScpnPathologyTechnologyAndAnomaliesSpec(
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
        "title": "Paper 0 " + "Applied SCPN: Pathology, Technology, and Anomalies" + " Specs",
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
        "next_source_boundary": "P0R06206",
    }
    return AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_applied_scpn_pathology_technology_and_anomalies_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Applied SCPN: Pathology, Technology, and Anomalies" + " Specs",
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
    bundle: AppliedScpnPathologyTechnologyAndAnomaliesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_applied_scpn_pathology_technology_and_anomalies_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_applied_scpn_pathology_technology_and_anomalies_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 applied-pathology technology/anomaly specs from the ledger."""

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
