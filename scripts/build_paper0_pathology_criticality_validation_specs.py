#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 pathology/criticality spec builder
"""Promote Paper 0 pathology/criticality anchors into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SPEC_SOURCE_LEDGER_IDS: dict[str, tuple[str, ...]] = {
    "applied.pathology.coherence_breakdown_index": (
        "P0R06197",
        "P0R06198",
        "P0R06199",
        "P0R06200",
        "P0R06201",
    ),
    "applied.pathology.criticality_deviation_classifier": (
        "P0R06202",
        "P0R06203",
    ),
    "applied.pathology.therapeutic_restoration_targets": (
        "P0R06204",
        "P0R06205",
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied.pathology.coherence_breakdown_index": {
        "validation_protocol": "paper0.applied.pathology.coherence_breakdown_index",
        "canonical_statement": (
            "Pathology is bounded to a computable systems-state index combining "
            "free-energy accumulation, synchronisation loss, criticality deviation, "
            "and error-correction failure."
        ),
        "variables": ("F_global", "order_parameter", "sigma", "qec_success_probability"),
        "assumptions": (
            "all terms are dimensionless finite simulator observables",
            "larger index means greater modelled systems dysregulation",
            "clinical labels are not inferred from this simulator quantity",
        ),
        "validation_targets": (
            "verify pathology index increases with free-energy accumulation",
            "verify coherence loss increases pathology index",
            "verify QEC failure probability contributes monotonically",
        ),
        "null_controls": (
            "healthy-baseline control must return lower index",
            "non-finite-observable control must be rejected",
            "negative-probability control must be rejected",
        ),
    },
    "applied.pathology.criticality_deviation_classifier": {
        "validation_protocol": "paper0.applied.pathology.criticality_deviation_classifier",
        "canonical_statement": (
            "Deviation from quasicriticality is promoted only as a sigma classifier: "
            "sigma > 1 is supercritical, sigma < 1 is subcritical, and sigma near 1 "
            "is quasicritical."
        ),
        "variables": ("sigma", "criticality_tolerance", "criticality_label"),
        "assumptions": (
            "sigma is a finite positive branching or response-ratio proxy",
            "clinical examples in the manuscript remain interpretive context only",
            "classification thresholds are explicit and tested",
        ),
        "validation_targets": (
            "verify sigma above tolerance classifies as supercritical",
            "verify sigma below tolerance classifies as subcritical",
            "verify sigma-neutral tolerance classifies as quasicritical",
        ),
        "null_controls": (
            "sigma-neutral control must not be classified as pathological",
            "non-positive-sigma control must be rejected",
            "negative-tolerance control must be rejected",
        ),
    },
    "applied.pathology.therapeutic_restoration_targets": {
        "validation_protocol": "paper0.applied.pathology.therapeutic_restoration_targets",
        "canonical_statement": (
            "Therapeutic restoration is bounded to simulator control objectives: "
            "reduce prediction-error/free-energy accumulation, tune sigma toward 1, "
            "and increase phase synchronisation."
        ),
        "variables": (
            "free_energy_step",
            "sigma_step",
            "synchronisation_step",
            "restoration_score",
        ),
        "assumptions": (
            "updates are small finite simulator controls",
            "restoration score compares before and after states only",
            "therapeutic wording is not clinical evidence or medical advice",
        ),
        "validation_targets": (
            "verify restoration update reduces pathology index",
            "verify sigma update moves toward one without overshoot hidden by clipping",
            "verify synchronisation update increases bounded order parameter",
        ),
        "null_controls": (
            "zero-update control must leave pathology index unchanged",
            "wrong-direction update control must increase pathology index",
            "out-of-range-order-parameter control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PathologyCriticalityValidationSpec:
    """Validation spec promoted from Paper 0 pathology/criticality records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: tuple[str, ...]
    assumptions: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class PathologyCriticalityValidationSpecBundle:
    """Pathology/criticality validation specs plus coverage summary."""

    specs: tuple[PathologyCriticalityValidationSpec, ...]
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


def build_pathology_criticality_validation_specs(
    source_records: list[dict[str, Any]],
) -> PathologyCriticalityValidationSpecBundle:
    """Build source-covered validation specs for pathology/criticality records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[PathologyCriticalityValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in (
        "applied.pathology.coherence_breakdown_index",
        "applied.pathology.criticality_deviation_classifier",
        "applied.pathology.therapeutic_restoration_targets",
    ):
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            PathologyCriticalityValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=(),
                source_ledger_ids=ledger_ids,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=anchor_math_ids,
                variables=tuple(str(item) for item in metadata["variables"]),
                assumptions=tuple(str(item) for item in metadata["assumptions"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="simulator-only systems metric; not clinical evidence or medical advice",
                implementation_status="validation_spec_pending_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    sorted_required = sorted(required_ids)
    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": consumed_ids == required_ids,
        "unconsumed_source_ledger_ids": sorted(required_ids - consumed_ids),
        "source_ledger_span": [sorted_required[0], sorted_required[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_avoid_invented_equation_ids": all(
            not spec.source_equation_ids and not spec.anchor_math_ids for spec in specs
        ),
        "all_specs_have_clinical_boundary": all(
            "clinical" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06197-P0R06205 are promoted as source-covered validation "
            "specifications only. Pathology and therapeutic language is bounded "
            "to finite simulator systems metrics and is not clinical evidence."
        ),
    }
    return PathologyCriticalityValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: PathologyCriticalityValidationSpecBundle) -> str:
    """Render a concise Markdown report for pathology/criticality specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Pathology/Criticality Validation Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        f"- Coverage status: `{status}`",
        f"- Source span: `{', '.join(bundle.summary['source_ledger_span'])}`",
        f"- Spec count: `{bundle.summary['spec_count']}`",
        f"- Hardware status: `{bundle.summary['hardware_status']}`",
        "",
        "## Specs",
        "",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored validation specifications only and "
            "not clinical evidence, diagnosis, treatment guidance, or medical advice.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: PathologyCriticalityValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the pathology/criticality bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_pathology_criticality_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_pathology_criticality_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    return [record for record in records if str(record.get("ledger_id")) in required_ids]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_pathology_criticality_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
