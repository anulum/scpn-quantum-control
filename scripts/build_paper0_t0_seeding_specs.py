#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 t0-seeding spec builder
"""Promote Paper 0 t=0 SSB seeding and spin-torsion records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6339, 6363))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06340",
    "P0R06346",
    "P0R06357",
    "P0R06358",
    "P0R06359",
    "P0R06360",
    "P0R06361",
)
FORMULAE_BY_SPEC = {
    "t0_seeding.teleological_tachyonic_potential": (
        ("P0R06344:t0_effective_potential",),
        ("V_eff(|Psi|, t -> 0+) = -mu^2(J_SEC) |Psi|^2 + lambda |Psi|^4",),
    ),
    "t0_seeding.spin_torsion_bridge_equations": (
        ("P0R06349:torsion_spin_bridge", "P0R06354:psi_torsion_spin_bridge"),
        ("torsion_ijk = 8 pi G s_ijk", "torsion_ijk = 8 pi G s_ijk_psi"),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "t0_seeding.initial_value_problem_boundary": {
        "validation_protocol": "paper0.t0_seeding.initial_value_problem_boundary",
        "canonical_statement": (
            "Initial-value wording is bounded to the source mechanism requirement that "
            "a massless symmetric t=0 boundary needs an SSB trigger."
        ),
        "variables": ("t0", "v", "SSB", "Psi_field", "conformal_bounce"),
        "validation_targets": (
            "score restored massless boundary and symmetric vacuum channels",
            "require explicit SSB trigger requirement",
            "reject treating t=0 seeding as empirical cosmology",
        ),
        "null_controls": (
            "missing-SSB-trigger control must be rejected",
            "missing-symmetric-vacuum control must be rejected",
            "empirical-cosmology control must be rejected",
        ),
    },
    "t0_seeding.j_sec_memory_bias_boundary": {
        "validation_protocol": "paper0.t0_seeding.j_sec_memory_bias_boundary",
        "canonical_statement": (
            "J_SEC memory wording is bounded to preserved Ethical Functional, conformal "
            "invariance, and prior-aeon informational geometry labels."
        ),
        "variables": ("J_SEC", "MMC", "conformal_invariant", "prior_aeon_geometry"),
        "validation_targets": (
            "require preserved J_SEC channel",
            "require conformal-invariance and prior-aeon geometry channels",
            "reject teleological memory as measured evidence",
        ),
        "null_controls": (
            "missing-J_SEC control must be rejected",
            "missing-prior-aeon-geometry control must be rejected",
            "memory-as-measurement control must be rejected",
        ),
    },
    "t0_seeding.teleological_tachyonic_potential": {
        "validation_protocol": "paper0.t0_seeding.teleological_tachyonic_potential",
        "canonical_statement": (
            "Teleological tachyonic potential wording is bounded to the source "
            "V_eff(|Psi|, t->0+) formula and a directional seed coefficient."
        ),
        "variables": ("V_eff", "Psi", "J_SEC", "mu_squared", "lambda"),
        "validation_targets": (
            "preserve the source t=0 effective-potential formula",
            "verify nonzero J_SEC seed can make the quadratic term tachyonic",
            "reject interpreting the seed as empirical evidence",
        ),
        "null_controls": (
            "missing-J_SEC-seed control must be rejected",
            "negative-lambda control must be rejected",
            "seed-as-observation control must be rejected",
        ),
    },
    "t0_seeding.spin_torsion_bridge_equations": {
        "validation_protocol": "paper0.t0_seeding.spin_torsion_bridge_equations",
        "canonical_statement": (
            "Spin-torsion bridge wording is bounded to ECSK-style torsion sourced by "
            "Psi-field spin density and to the two source torsion formulae."
        ),
        "variables": ("torsion_ijk", "G", "s_ijk", "s_ijk_psi", "ECSK"),
        "validation_targets": (
            "preserve both source torsion bridge formulae",
            "verify non-negative spin density produces non-negative torsion proxy",
            "reject treating torsion bridge as measured spacetime torsion",
        ),
        "null_controls": (
            "negative-spin-density control must be rejected",
            "missing-Psi-spin-density control must be rejected",
            "torsion-evidence control must be rejected",
        ),
    },
    "t0_seeding.conformal_invariant_torsion_boundary": {
        "validation_protocol": "paper0.t0_seeding.conformal_invariant_torsion_boundary",
        "canonical_statement": (
            "Conformal-invariant torsion wording is bounded to J_SEC preserved as "
            "T_SEC across the MMC and acting as structured t->0+ Psi-vacuum bias."
        ),
        "variables": ("J_SEC", "T_SEC", "MMC", "Psi_vacuum", "t0_plus"),
        "validation_targets": (
            "require J_SEC and T_SEC preservation labels",
            "require structured bias and non-random reset boundary labels",
            "reject conformal torsion as empirical evidence",
        ),
        "null_controls": (
            "missing-T_SEC control must be rejected",
            "missing-structured-bias control must be rejected",
            "torsion-as-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class T0SeedingValidationSpec:
    """Validation spec promoted from Paper 0 t0-seeding records."""

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
class T0SeedingValidationSpecBundle:
    """T0-seeding validation specs plus coverage summary."""

    specs: tuple[T0SeedingValidationSpec, ...]
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


def build_t0_seeding_specs(source_records: list[dict[str, Any]]) -> T0SeedingValidationSpecBundle:
    """Build source-covered validation specs for t0-seeding records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[T0SeedingValidationSpec] = []
    for key in (
        "t0_seeding.initial_value_problem_boundary",
        "t0_seeding.j_sec_memory_bias_boundary",
        "t0_seeding.teleological_tachyonic_potential",
        "t0_seeding.spin_torsion_bridge_equations",
        "t0_seeding.conformal_invariant_torsion_boundary",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae = FORMULAE_BY_SPEC.get(key, ((), ()))
        specs.append(
            T0SeedingValidationSpec(
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
                claim_boundary="source-bounded t0-seeding simulator contract; not empirical evidence",
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
            "P0R06339-P0R06362 are promoted as source-covered t0-seeding "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return T0SeedingValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: T0SeedingValidationSpecBundle) -> str:
    """Render a concise Markdown report for t0-seeding specs."""
    lines = [
        "# Paper 0 t0 Seeding Specs",
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
                f"- Source formulae: `{', '.join(spec.source_formulae) if spec.source_formulae else 'none'}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored t0-seeding specifications only. "
            "Passing any fixture is not empirical evidence and does not validate "
            "teleological seeding, ECSK torsion, MMC continuity, or conformal torsion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: T0SeedingValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the t0-seeding bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_t0_seeding_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_t0_seeding_validation_specs_report_{date_tag}.md"
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
    bundle = build_t0_seeding_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
