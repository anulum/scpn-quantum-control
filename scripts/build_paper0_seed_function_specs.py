#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 seed-function spec builder
"""Promote Paper 0 Python-format teleological seed records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6363, 6378))
STRUCTURAL_SOURCE_LEDGER_IDS = ("P0R06377",)
FORMULAE_BY_SPEC = {
    "seed_function.python_format_source_boundary": (
        ("P0R06364:function_signature",),
        ("def compute_teleological_seed(prev_cycle_sec, coupling_constant_g):",),
    ),
    "seed_function.mu_squared_seed_formula": (
        ("P0R06371:mu_squared_seed",),
        ("mu_squared_seed = sqrt(prev_cycle_sec / coupling_constant_g)",),
    ),
    "seed_function.return_payload_contract": (
        ("P0R06373:ssb_bias_magnitude", "P0R06374:is_random_reset"),
        ("ssb_bias_magnitude = mu_squared_seed", "is_random_reset = False"),
    ),
    "seed_function.conformal_continuity_boundary": (
        ("P0R06375:conformal_continuity",),
        ("conformal_continuity = prev_cycle_sec > 0",),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "seed_function.python_format_source_boundary": {
        "validation_protocol": "paper0.seed_function.python_format_source_boundary",
        "canonical_statement": (
            "Python-format seed wording is bounded to a source-code-like manuscript "
            "contract for compute_teleological_seed."
        ),
        "variables": ("prev_cycle_sec", "coupling_constant_g", "compute_teleological_seed"),
        "validation_targets": (
            "preserve the source function signature",
            "require structured tachyonic mass-term purpose text",
            "reject treating manuscript code format as empirical evidence",
        ),
        "null_controls": (
            "missing-function-signature control must be rejected",
            "missing-purpose-text control must be rejected",
            "code-format-as-evidence control must be rejected",
        ),
    },
    "seed_function.mu_squared_seed_formula": {
        "validation_protocol": "paper0.seed_function.mu_squared_seed_formula",
        "canonical_statement": (
            "The mu-squared seed formula is bounded to sqrt(prev_cycle_sec / coupling_constant_g)."
        ),
        "variables": ("mu_squared_seed", "prev_cycle_sec", "coupling_constant_g"),
        "validation_targets": (
            "preserve the source seed formula",
            "reject negative SEC and non-positive coupling",
            "verify deterministic finite numeric output",
        ),
        "null_controls": (
            "negative-SEC control must be rejected",
            "zero-coupling control must be rejected",
            "non-finite-output control must be rejected",
        ),
    },
    "seed_function.return_payload_contract": {
        "validation_protocol": "paper0.seed_function.return_payload_contract",
        "canonical_statement": (
            "Return-payload wording is bounded to ssb_bias_magnitude and "
            "is_random_reset=False source fields."
        ),
        "variables": ("ssb_bias_magnitude", "mu_squared_seed", "is_random_reset"),
        "validation_targets": (
            "preserve ssb_bias_magnitude payload field",
            "preserve is_random_reset False field",
            "reject payloads that omit source fields",
        ),
        "null_controls": (
            "missing-ssb-bias-field control must be rejected",
            "random-reset-true control must be rejected",
            "missing-payload control must be rejected",
        ),
    },
    "seed_function.conformal_continuity_boundary": {
        "validation_protocol": "paper0.seed_function.conformal_continuity_boundary",
        "canonical_statement": (
            "Conformal-continuity wording is bounded to the source predicate prev_cycle_sec > 0."
        ),
        "variables": ("conformal_continuity", "prev_cycle_sec"),
        "validation_targets": (
            "preserve conformal_continuity source predicate",
            "verify positive SEC marks continuity true",
            "reject continuity as empirical MMC proof",
        ),
        "null_controls": (
            "non-positive-SEC-continuity control must be rejected",
            "missing-continuity-field control must be rejected",
            "MMC-proof control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SeedFunctionValidationSpec:
    """Validation spec promoted from Paper 0 seed-function records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
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
class SeedFunctionValidationSpecBundle:
    """Seed-function validation specs plus coverage summary."""

    specs: tuple[SeedFunctionValidationSpec, ...]
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


def build_seed_function_specs(
    source_records: list[dict[str, Any]],
) -> SeedFunctionValidationSpecBundle:
    """Build source-covered validation specs for seed-function records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[SeedFunctionValidationSpec] = []
    for key in (
        "seed_function.python_format_source_boundary",
        "seed_function.mu_squared_seed_formula",
        "seed_function.return_payload_contract",
        "seed_function.conformal_continuity_boundary",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae = FORMULAE_BY_SPEC[key]
        specs.append(
            SeedFunctionValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=equation_ids,
                source_formulae=formulae,
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=(),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="source-bounded seed-function simulator contract; not empirical evidence",
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "source_formula_ids": [
            formula_id for ids, _formulae in FORMULAE_BY_SPEC.values() for formula_id in ids
        ],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06363-P0R06377 are promoted as source-covered seed-function "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return SeedFunctionValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: SeedFunctionValidationSpecBundle) -> str:
    """Render a concise Markdown report for seed-function specs."""
    lines = [
        "# Paper 0 Seed Function Specs",
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
                f"- Source formulae: `{', '.join(spec.source_formulae)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored seed-function specifications only. "
            "Passing any fixture is not empirical evidence and does not validate "
            "teleological seeding or MMC continuity.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: SeedFunctionValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the seed-function bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_seed_function_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_seed_function_validation_specs_report_{date_tag}.md"
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
    bundle = build_seed_function_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
