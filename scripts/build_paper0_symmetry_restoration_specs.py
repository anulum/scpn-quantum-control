#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 symmetry-restoration spec builder
"""Promote Paper 0 MMC symmetry-restoration records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6324, 6339))
FORMULAE_BY_SPEC = {
    "symmetry_restoration.mmc_conformal_geometry_boundary": (
        ("P0R06326:conformal_rescaling",),
        ("g_hat_mu_nu = Omega^2 g_mu_nu",),
    ),
    "symmetry_restoration.effective_potential_flip_boundary": (
        ("P0R06333:effective_potential",),
        ("V_eff(|Psi|) = (-mu^2 + c1 T_dS^2 + c2 f(R)) |Psi|^2 + lambda |Psi|^4",),
    ),
    "symmetry_restoration.vev_melting_massless_limit": (
        ("P0R06336:vev_limit", "P0R06337:mass_limits"),
        ("lim_{t -> infinity} v(t) = 0", "m_A = g v; m_h = sqrt(2 lambda) v"),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "symmetry_restoration.mmc_conformal_geometry_boundary": {
        "validation_protocol": "paper0.symmetry_restoration.mmc_conformal_geometry_boundary",
        "canonical_statement": (
            "MMC mathematics wording is bounded to conformal geometry, optional Floquet "
            "internal dynamics context, conformal rescaling, and L15 information preservation."
        ),
        "variables": ("MMC", "P3", "Omega", "g_mu_nu", "E", "L15"),
        "validation_targets": (
            "preserve conformal-rescaling source formula text",
            "require L15 information-preservation channel",
            "reject treating MMC or CCC wording as empirical cosmology",
        ),
        "null_controls": (
            "missing-conformal-rescaling control must be rejected",
            "missing-L15-preservation control must be rejected",
            "empirical-cosmology control must be rejected",
        ),
    },
    "symmetry_restoration.conformal_boundary_masslessness_constraint": {
        "validation_protocol": "paper0.symmetry_restoration.conformal_boundary_masslessness_constraint",
        "canonical_statement": (
            "Conformal-boundary wording is bounded to the claim that retained mass or "
            "physical scale violates conformal rescaling at the aeon boundary."
        ),
        "variables": ("CCC", "MMC", "mass", "physical_scale", "v"),
        "validation_targets": (
            "score mass-retention as conformal-boundary violation",
            "score masslessness as restored boundary condition",
            "reject boundary legality without explicit masslessness condition",
        ),
        "null_controls": (
            "retained-mass control must be rejected",
            "retained-scale control must be rejected",
            "missing-masslessness-boundary control must be rejected",
        ),
    },
    "symmetry_restoration.effective_potential_flip_boundary": {
        "validation_protocol": "paper0.symmetry_restoration.effective_potential_flip_boundary",
        "canonical_statement": (
            "Effective-potential wording is bounded to a quadratic-coefficient sign flip "
            "from thermal and geometric corrections in the de Sitter asymptotic context."
        ),
        "variables": ("V_eff", "mu_squared", "c1", "T_dS", "c2", "f_R", "lambda"),
        "validation_targets": (
            "preserve the source effective-potential formula text",
            "verify correction terms can flip the quadratic coefficient sign",
            "reject treating the asymptotic scenario as observed cosmology",
        ),
        "null_controls": (
            "missing-thermal-correction control must be rejected",
            "missing-geometric-correction control must be rejected",
            "observed-cosmology control must be rejected",
        ),
    },
    "symmetry_restoration.vev_melting_massless_limit": {
        "validation_protocol": "paper0.symmetry_restoration.vev_melting_massless_limit",
        "canonical_statement": (
            "VEV-melting wording is bounded to the limit v(t)->0 and consequent infoton "
            "and Psi-Higgs mass limits m_A=g v and m_h=sqrt(2 lambda) v."
        ),
        "variables": ("v", "m_A", "g", "m_h", "lambda", "U1"),
        "validation_targets": (
            "preserve VEV and mass-limit source formulae",
            "verify finite non-negative couplings produce massless limits as v approaches zero",
            "reject negative lambda or nonzero VEV boundary controls",
        ),
        "null_controls": (
            "nonzero-VEV control must be rejected",
            "negative-lambda control must be rejected",
            "massless-limit-as-observation control must be rejected",
        ),
    },
    "symmetry_restoration.legal_conformal_rescaling_boundary": {
        "validation_protocol": "paper0.symmetry_restoration.legal_conformal_rescaling_boundary",
        "canonical_statement": (
            "Legal-conformal-rescaling wording is bounded to scale shedding, dimensionless "
            "Ethical Functional preservation, and metric reset for a subsequent aeon."
        ),
        "variables": ("O_MMC", "scale_shedding", "dimensionless_E", "metric_reset"),
        "validation_targets": (
            "require scale-shedding and dimensionless-functional channels",
            "require metric-reset channel",
            "reject conformal-rescaling legality without all source channels",
        ),
        "null_controls": (
            "missing-scale-shedding control must be rejected",
            "missing-dimensionless-functional control must be rejected",
            "missing-metric-reset control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SymmetryRestorationValidationSpec:
    """Validation spec promoted from Paper 0 symmetry-restoration records."""

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
class SymmetryRestorationValidationSpecBundle:
    """Symmetry-restoration validation specs plus coverage summary."""

    specs: tuple[SymmetryRestorationValidationSpec, ...]
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


def build_symmetry_restoration_specs(
    source_records: list[dict[str, Any]],
) -> SymmetryRestorationValidationSpecBundle:
    """Build source-covered validation specs for symmetry-restoration records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[SymmetryRestorationValidationSpec] = []
    for key in (
        "symmetry_restoration.mmc_conformal_geometry_boundary",
        "symmetry_restoration.conformal_boundary_masslessness_constraint",
        "symmetry_restoration.effective_potential_flip_boundary",
        "symmetry_restoration.vev_melting_massless_limit",
        "symmetry_restoration.legal_conformal_rescaling_boundary",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae = FORMULAE_BY_SPEC.get(key, ((), ()))
        specs.append(
            SymmetryRestorationValidationSpec(
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
                claim_boundary=(
                    "source-bounded symmetry-restoration simulator contract; "
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
            "P0R06324-P0R06338 are promoted as source-covered symmetry-restoration "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return SymmetryRestorationValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: SymmetryRestorationValidationSpecBundle) -> str:
    """Render a concise Markdown report for symmetry-restoration specs."""
    lines = [
        "# Paper 0 Symmetry Restoration Specs",
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
            "These records are source-anchored symmetry-restoration specifications only. "
            "Passing any fixture is not empirical evidence and does not validate MMC, "
            "CCC, de Sitter asymptotics, or far-future conformal-boundary claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: SymmetryRestorationValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the symmetry-restoration bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_symmetry_restoration_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_symmetry_restoration_validation_specs_report_{date_tag}.md"
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
    bundle = build_symmetry_restoration_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
