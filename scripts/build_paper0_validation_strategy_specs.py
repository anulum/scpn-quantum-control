#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 validation-strategy spec builder
"""Promote Paper 0 Applied SCPN and Validation records into roadmap specs."""

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

SOURCE_LEDGER_IDS = ("P0R06221", "P0R06222")

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied.validation_strategy.pathology_and_societal_phase_targets": {
        "stage": "Cross-cutting",
        "validation_protocol": "paper0.applied.validation_strategy.pathology_societal_targets",
        "canonical_statement": (
            "Applied pathology and L11 societal phase-transition claims are bounded "
            "to validation targets for criticality, free energy, MS-QEC, and spin-glass dynamics."
        ),
        "variables": ("pathology", "criticality", "free_energy", "ms_qec", "l11_spin_glass"),
        "validation_targets": (
            "map pathology target to deviation from criticality, free-energy accumulation, or MS-QEC failure",
            "map societal phase transitions to L11 spin-glass dynamics",
            "preserve quasicritical sigma equals one as a validation target",
        ),
        "null_controls": (
            "missing-pathology-target control must be rejected",
            "missing-L11-spin-glass target control must be rejected",
            "roadmap output must not be treated as empirical evidence",
        ),
    },
    "applied.validation_strategy.ethical_governance_alignment_targets": {
        "stage": "Cross-cutting",
        "validation_protocol": "paper0.applied.validation_strategy.governance_alignment_targets",
        "canonical_statement": (
            "Ethical governance and alignment claims are bounded to validation targets "
            "for SEC, ethical Lagrangian/CEF, and ethical-functional embedding."
        ),
        "variables": ("SEC", "ethical_lagrangian", "CEF", "ethical_functional"),
        "validation_targets": (
            "map governance target to SEC maximisation through ethical Lagrangian and CEF",
            "map alignment target to ethical-functional embedding as objective function",
            "record that roadmap classification is not empirical evidence",
        ),
        "null_controls": (
            "missing-governance-target control must be rejected",
            "missing-alignment-target control must be rejected",
            "roadmap output must not prescribe deployment",
        ),
    },
    "applied.validation_strategy.stage_i_foundations": {
        "stage": "Stage I",
        "validation_protocol": "paper0.applied.validation_strategy.stage_i_foundations",
        "canonical_statement": (
            "Stage I prioritises foundational validation: universal quasicriticality, "
            "biological quantum-interface tests, and geometry-of-qualia PTA/TDA."
        ),
        "variables": ("quasicriticality", "biological_quantum_interface", "qualia_geometry"),
        "validation_targets": (
            "validate universal quasicriticality before higher-level dynamics",
            "validate L1/L2 biological quantum-interface gaps and modulation claims",
            "validate L5 geometry-of-qualia claims with PTA/TDA targets",
        ),
        "null_controls": (
            "stage-order control must place Stage I before Stage II",
            "missing-foundation-domain control must be rejected",
            "roadmap output must not be treated as completed validation",
        ),
    },
    "applied.validation_strategy.stage_ii_iii_mechanisms_and_high_level": {
        "stage": "Stage II/III",
        "validation_protocol": "paper0.applied.validation_strategy.stage_ii_iii",
        "canonical_statement": (
            "Stage II prioritises UPDE and HPC mechanisms; Stage III prioritises "
            "Gaian coupling and teleology high-level dynamics."
        ),
        "variables": ("UPDE", "PAC", "HPC", "transfer_entropy", "Gaian_coupling", "teleology"),
        "validation_targets": (
            "validate UPDE through multi-scale PAC targets",
            "validate HPC through transfer-entropy targets",
            "validate L12 Gaian coupling and L15/L8 teleology after mechanism targets",
        ),
        "null_controls": (
            "stage-order control must place Stage II before Stage III",
            "missing-mechanism-domain control must be rejected",
            "missing-high-level-domain control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ValidationStrategySpec:
    """Validation roadmap spec promoted from Paper 0 Applied SCPN records."""

    key: str
    stage: str
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
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class ValidationStrategySpecBundle:
    """Validation roadmap specs plus coverage summary."""

    specs: tuple[ValidationStrategySpec, ...]
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


def build_validation_strategy_specs(
    source_records: list[dict[str, Any]],
) -> ValidationStrategySpecBundle:
    """Build source-covered validation roadmap specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = set(SOURCE_LEDGER_IDS)
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    anchor_math_ids = tuple(
        sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
    )
    specs: list[ValidationStrategySpec] = []
    for key in (
        "applied.validation_strategy.pathology_and_societal_phase_targets",
        "applied.validation_strategy.ethical_governance_alignment_targets",
        "applied.validation_strategy.stage_i_foundations",
        "applied.validation_strategy.stage_ii_iii_mechanisms_and_high_level",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            ValidationStrategySpec(
                key=key,
                stage=str(metadata["stage"]),
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
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="validation-roadmap contract; not empirical evidence",
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
        "all_specs_are_roadmap_only": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06221-P0R06222 are promoted as source-covered validation-roadmap "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return ValidationStrategySpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: ValidationStrategySpecBundle) -> str:
    """Render a concise Markdown report for validation-roadmap specs."""
    lines = [
        "# Paper 0 Validation Strategy Specs",
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
                f"- Stage: `{spec.stage}`",
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
            "These records are source-anchored validation-roadmap specifications only. "
            "Passing any fixture is not empirical evidence and does not establish "
            "that any listed target has been experimentally validated.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: ValidationStrategySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the validation-roadmap bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_validation_strategy_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_validation_strategy_specs_report_{date_tag}.md"
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
    bundle = build_validation_strategy_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
