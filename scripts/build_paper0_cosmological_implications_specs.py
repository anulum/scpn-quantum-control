#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 cosmological implications spec builder
"""Promote Paper 0 comparative and cosmological implications records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6290, 6311))

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "cosmological_implications.comparative_positioning_mapping": {
        "validation_protocol": "paper0.cosmological_implications.comparative_positioning_mapping",
        "canonical_statement": (
            "Comparative-positioning wording is bounded to source-listed mappings "
            "against IIT, Orch OR, GNW, FEP/predictive coding, UPDE, and L15."
        ),
        "variables": ("IIT", "Orch_OR", "GNW", "FEP", "UPDE", "L15"),
        "validation_targets": (
            "require all four source-listed comparison theories",
            "require UPDE and L15 as SCPN-specific differentiators",
            "reject treating comparative positioning as external benchmark evidence",
        ),
        "null_controls": (
            "missing-comparison-theory control must be rejected",
            "missing-UPDE-or-L15 control must be rejected",
            "benchmark-evidence control must be rejected",
        ),
    },
    "cosmological_implications.ethical_selection_claim_boundary": {
        "validation_protocol": "paper0.cosmological_implications.ethical_selection_claim_boundary",
        "canonical_statement": (
            "Ethical-selection wording is bounded to a claim that physical constants, "
            "including Lambda, are selected to maximise SEC, without empirical support."
        ),
        "variables": ("physical_constants", "Λ", "SEC", "L15"),
        "validation_targets": (
            "preserve the SEC/L15 claim scope",
            "separate symbolic constant-selection wording from observed cosmology evidence",
            "reject any claim that the fixture validates fine tuning",
        ),
        "null_controls": (
            "missing-SEC-anchor control must be rejected",
            "missing-L15-anchor control must be rejected",
            "fine-tuning-proof control must be rejected",
        ),
    },
    "cosmological_implications.lambda_optimisation_context": {
        "validation_protocol": "paper0.cosmological_implications.lambda_optimisation_context",
        "canonical_statement": (
            "Lambda optimisation wording is bounded to a simulator score over expansion "
            "balance, RG-flow window, and cosmic-attractor access."
        ),
        "variables": ("Λ", "Λ_obs", "RG_flow", "g_star", "dark_energy"),
        "validation_targets": (
            "score balanced and unbalanced Lambda-context scenarios",
            "require RG-flow and cosmic-attractor channels",
            "reject dark-energy interpretation as empirical evidence",
        ),
        "null_controls": (
            "missing-RG-flow control must be rejected",
            "missing-cosmic-attractor control must be rejected",
            "dark-energy-evidence control must be rejected",
        ),
    },
    "cosmological_implications.ethical_renormalisation_mechanism": {
        "validation_protocol": "paper0.cosmological_implications.ethical_renormalisation_mechanism",
        "canonical_statement": (
            "Ethical-renormalisation wording is bounded to preserved Yang-Mills form, "
            "renormalised C/K/Q balance, and L16 meta-optimisation across cycles."
        ),
        "variables": ("L_Ethical", "C", "K", "Q", "L16", "coupling_constants"),
        "validation_targets": (
            "require previous-cycle pathology inputs",
            "require coupling adjustment and L16 meta-optimisation channels",
            "reject cosmological-timescale wording as empirical evidence",
        ),
        "null_controls": (
            "missing-L16-meta-optimisation control must be rejected",
            "negative-coupling-adjustment control must be rejected",
            "empirical-cosmology control must be rejected",
        ),
    },
    "cosmological_implications.mmc_ccc_formalisation_boundary": {
        "validation_protocol": "paper0.cosmological_implications.mmc_ccc_formalisation_boundary",
        "canonical_statement": (
            "MMC formalisation wording is bounded to a context relation inspired by "
            "Conformal Cyclic Cosmology and does not validate CCC or MMC."
        ),
        "variables": ("MMC", "P3", "Conformal Cyclic Cosmology", "conformal_boundary"),
        "validation_targets": (
            "preserve MMC/P3 and CCC-inspired context labels",
            "require explicit non-validation boundary for CCC and MMC",
            "reject using source context as cosmological observation",
        ),
        "null_controls": (
            "missing-MMC-P3 label control must be rejected",
            "missing-CCC-context control must be rejected",
            "cosmological-observation control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CosmologicalImplicationsValidationSpec:
    """Validation spec promoted from Paper 0 cosmological implications records."""

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
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class CosmologicalImplicationsValidationSpecBundle:
    """Cosmological implications validation specs plus coverage summary."""

    specs: tuple[CosmologicalImplicationsValidationSpec, ...]
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


def build_cosmological_implications_specs(
    source_records: list[dict[str, Any]],
) -> CosmologicalImplicationsValidationSpecBundle:
    """Build source-covered validation specs for cosmological implications records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CosmologicalImplicationsValidationSpec] = []
    for key in (
        "cosmological_implications.comparative_positioning_mapping",
        "cosmological_implications.ethical_selection_claim_boundary",
        "cosmological_implications.lambda_optimisation_context",
        "cosmological_implications.ethical_renormalisation_mechanism",
        "cosmological_implications.mmc_ccc_formalisation_boundary",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            CosmologicalImplicationsValidationSpec(
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
                anchor_math_ids=(),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=(
                    "source-bounded cosmological implications simulator contract; "
                    "not empirical evidence"
                ),
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
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06290-P0R06310 are promoted as source-covered comparative and "
            "cosmological implications specifications only. Passing fixtures is "
            "not empirical evidence."
        ),
    }
    return CosmologicalImplicationsValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CosmologicalImplicationsValidationSpecBundle) -> str:
    """Render a concise Markdown report for cosmological implications specs."""
    lines = [
        "# Paper 0 Cosmological Implications Specs",
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
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored comparative and cosmological implications "
            "specifications only. Passing any fixture is not empirical evidence and "
            "does not validate fine tuning, dark energy, MMC, or CCC claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: CosmologicalImplicationsValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the cosmological implications bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_cosmological_implications_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_cosmological_implications_validation_specs_report_{date_tag}.md"
    )
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
    bundle = build_cosmological_implications_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
