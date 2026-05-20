#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 system-robustness spec builder
"""Promote Paper 0 system-robustness records into validation specs."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = ("P0R06215", "P0R06216", "P0R06217")

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied.system_robustness.cascading_failure_percolation": {
        "validation_protocol": "paper0.applied.system_robustness.cascading_failure_percolation",
        "canonical_statement": (
            "Cascading-failure wording is bounded to weighted-graph percolation "
            "tests over finite coupling networks."
        ),
        "variables": ("coupling_matrix", "percolation_threshold", "removed_nodes"),
        "assumptions": (
            "couplings are finite symmetric non-negative graph weights",
            "cascade severity is measured by largest component loss",
            "passing the fixture is not operational security evidence",
        ),
        "validation_targets": (
            "verify thresholded graph largest component fraction is bounded",
            "verify targeted removal reduces largest component fraction",
            "verify invalid coupling matrices are rejected",
        ),
        "null_controls": (
            "empty-graph control must be fragmented",
            "complete-graph control must remain connected",
            "asymmetric-coupling control must be rejected",
        ),
    },
    "applied.system_robustness.critical_slowing_recovery": {
        "validation_protocol": "paper0.applied.system_robustness.critical_slowing_recovery",
        "canonical_statement": (
            "Critical-slowing wording is bounded to finite recovery-time scaling "
            "as sigma approaches the phase-transition point."
        ),
        "variables": ("sigma", "sigma_critical", "critical_exponent", "recovery_time"),
        "assumptions": (
            "sigma is finite positive and not exactly at the critical point",
            "recovery time is a simulator observable, not a clinical prognosis",
            "passing the fixture is not operational security evidence",
        ),
        "validation_targets": (
            "verify recovery time increases near sigma criticality",
            "verify far-from-critical control has lower recovery time",
            "verify exact-critical denominator is rejected",
        ),
        "null_controls": (
            "far-from-transition control must reduce recovery time",
            "critical-point control must be rejected",
            "non-positive-sigma control must be rejected",
        ),
    },
    "applied.system_robustness.decoherence_attack_ms_qec_boundary": {
        "validation_protocol": "paper0.applied.system_robustness.decoherence_attack_ms_qec_boundary",
        "canonical_statement": (
            "Decoherence-attack wording is bounded to an MS-QEC success-probability "
            "fixture with explicit redundancy and correction-strength controls."
        ),
        "variables": (
            "decoherence_exposure",
            "ms_qec_redundancy",
            "qec_correction_strength",
            "failure_probability",
        ),
        "assumptions": (
            "attack is a simulator stressor parameter, not an operational exploit",
            "MS-QEC redundancy monotonically improves success under finite exposure",
            "passing the fixture is not operational security evidence",
        ),
        "validation_targets": (
            "verify MS-QEC success probability is bounded in the unit interval",
            "verify redundancy reduces failure probability relative to unprotected control",
            "verify invalid redundancy or correction strength is rejected",
        ),
        "null_controls": (
            "zero-redundancy control must be rejected",
            "out-of-range-correction control must be rejected",
            "unprotected-control failure must exceed protected failure",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SystemRobustnessValidationSpec:
    """Validation spec promoted from Paper 0 system-robustness records."""

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
class SystemRobustnessValidationSpecBundle:
    """System-robustness validation specs plus coverage summary."""

    specs: tuple[SystemRobustnessValidationSpec, ...]
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


def build_system_robustness_validation_specs(
    source_records: list[dict[str, Any]],
) -> SystemRobustnessValidationSpecBundle:
    """Build source-covered validation specs for system-robustness records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = set(SOURCE_LEDGER_IDS)
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    anchor_math_ids = tuple(
        sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
    )
    specs: list[SystemRobustnessValidationSpec] = []
    for key in (
        "applied.system_robustness.cascading_failure_percolation",
        "applied.system_robustness.critical_slowing_recovery",
        "applied.system_robustness.decoherence_attack_ms_qec_boundary",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            SystemRobustnessValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=(),
                source_ledger_ids=SOURCE_LEDGER_IDS,
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
                claim_boundary=(
                    "simulator-only robustness boundary; not operational security evidence"
                ),
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(required_ids),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
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
        "all_specs_have_operational_boundary": all(
            "not operational security evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06215-P0R06217 are promoted as source-covered simulator robustness "
            "specifications only. Passing fixtures is not operational security evidence."
        ),
    }
    return SystemRobustnessValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: SystemRobustnessValidationSpecBundle) -> str:
    """Render a concise Markdown report for system-robustness specs."""
    lines = [
        "# Paper 0 System-Robustness Validation Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        "- Coverage status: `match`",
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
            "These records are source-anchored simulator robustness specifications only. "
            "Passing any fixture is not operational security evidence and does not "
            "establish real-world attack resistance, safety, or resilience.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: SystemRobustnessValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the system-robustness bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_system_robustness_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_system_robustness_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("ledger_id")) in set(SOURCE_LEDGER_IDS)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_system_robustness_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
