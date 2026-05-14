#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Gaian safety spec builder
"""Promote Paper 0 Gaian ethic and societal-safety records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6251, 6273))

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "gaian_safety.biodiversity_phi_sec_boundary": {
        "validation_protocol": "paper0.gaian_safety.biodiversity_phi_sec_boundary",
        "canonical_statement": (
            "Biodiversity, global Phi, and SEC wording is bounded to a Gaian "
            "stability index over finite simulator variables."
        ),
        "variables": ("biodiversity", "global_phi", "SEC", "L12"),
        "validation_targets": (
            "compare protected and degraded biodiversity/Phi/SEC states",
            "label Gaian stability from bounded finite scores",
            "reject treating the claim as ecological evidence",
        ),
        "null_controls": (
            "missing-biodiversity control must be rejected",
            "missing-SEC control must be rejected",
            "ecological-evidence control must be rejected",
        ),
    },
    "gaian_safety.ethical_functional_pela_boundary": {
        "validation_protocol": "paper0.gaian_safety.ethical_functional_pela_boundary",
        "canonical_statement": (
            "The Ethical Functional and PELA wording is bounded to a simulator "
            "claim boundary linking SEC, action minimisation, and teleological gradient."
        ),
        "variables": ("ethical_functional", "SEC", "PELA", "action", "teleological_gradient"),
        "validation_targets": (
            "verify SEC-aligned state has lower action proxy",
            "verify PELA anchor is present in the source span",
            "reject normative prescription without modelled objective terms",
        ),
        "null_controls": (
            "missing-PELA-anchor control must be rejected",
            "missing-action-proxy control must be rejected",
            "normative-prescription-as-proof control must be rejected",
        ),
    },
    "gaian_safety.nths_phase_category_validation": {
        "validation_protocol": "paper0.gaian_safety.nths_phase_category_validation",
        "canonical_statement": (
            "NTHS phase wording is bounded to three finite labels: ferromagnetic "
            "coherence, spin-glass fragmentation, and paramagnetic incoherence."
        ),
        "variables": ("NTHS", "coherence", "frustration", "entropy_flux"),
        "validation_targets": (
            "classify high-coherence low-frustration states as ferromagnetic coherence",
            "classify high-frustration states as spin-glass fragmentation",
            "classify high-entropy low-coherence states as paramagnetic incoherence",
        ),
        "null_controls": (
            "missing-phase-category control must be rejected",
            "non-finite-phase-input control must be rejected",
            "societal-phase-evidence control must be rejected",
        ),
    },
    "gaian_safety.consciousness_engineering_safety_protocol": {
        "validation_protocol": "paper0.gaian_safety.consciousness_engineering_safety_protocol",
        "canonical_statement": (
            "Consciousness Engineering and Field Architecture wording is bounded "
            "to multi-layer intervention safety channels anchored in L15-L16."
        ),
        "variables": ("L1_L2", "L7", "L11", "L15", "L16", "safety_protocol"),
        "validation_targets": (
            "require neural-quantum, noosphere, and symbolic intervention channels",
            "require upward and downward cascade accounting",
            "require Layer 15-16 safety anchors",
        ),
        "null_controls": (
            "missing-intervention-channel control must be rejected",
            "missing-cascade-accounting control must be rejected",
            "missing-L15-L16-anchor control must be rejected",
        ),
    },
    "gaian_safety.governance_risk_safeguard_protocol": {
        "validation_protocol": "paper0.gaian_safety.governance_risk_safeguard_protocol",
        "canonical_statement": (
            "Governance, risk, safeguard, and participatory-ethics wording is bounded "
            "to entropy budgets, coherence metrics, recursive review, QECC-style redundancy, "
            "and long-horizon ecological/technological budgets."
        ),
        "variables": ("entropy_budget", "coherence_metric", "recursive_review", "QECC", "L15"),
        "validation_targets": (
            "score risk channels against explicit safeguard channels",
            "require entropy-constrained policy and recursive ethical review",
            "require long-horizon ecological and technological budgets",
        ),
        "null_controls": (
            "missing-entropy-budget control must be rejected",
            "missing-recursive-review control must be rejected",
            "missing-QECC-safeguard control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class GaianSafetyValidationSpec:
    """Validation spec promoted from Paper 0 Gaian safety records."""

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
class GaianSafetyValidationSpecBundle:
    """Gaian safety validation specs plus coverage summary."""

    specs: tuple[GaianSafetyValidationSpec, ...]
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


def build_gaian_safety_specs(
    source_records: list[dict[str, Any]],
) -> GaianSafetyValidationSpecBundle:
    """Build source-covered validation specs for Gaian safety records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[GaianSafetyValidationSpec] = []
    for key in (
        "gaian_safety.biodiversity_phi_sec_boundary",
        "gaian_safety.ethical_functional_pela_boundary",
        "gaian_safety.nths_phase_category_validation",
        "gaian_safety.consciousness_engineering_safety_protocol",
        "gaian_safety.governance_risk_safeguard_protocol",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            GaianSafetyValidationSpec(
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
                claim_boundary="source-bounded Gaian safety simulator contract; not empirical evidence",
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
            "P0R06251-P0R06272 are promoted as source-covered Gaian safety "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return GaianSafetyValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: GaianSafetyValidationSpecBundle) -> str:
    """Render a concise Markdown report for Gaian safety specs."""
    lines = [
        "# Paper 0 Gaian Safety Specs",
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
            "These records are source-anchored Gaian safety specifications only. "
            "Passing any fixture is not empirical evidence and does not establish "
            "that any ecological, societal, or governance claim is validated.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: GaianSafetyValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Gaian safety bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_gaian_safety_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_gaian_safety_validation_specs_report_{date_tag}.md"
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
    bundle = build_gaian_safety_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
