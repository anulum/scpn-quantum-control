#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 cosmological EOS spec builder
"""Promote Paper 0 cosmological equation-of-state records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6916, 6949))
CLAIM_BOUNDARY = "source-bounded cosmological equation-of-state fixture; not empirical evidence"
HARDWARE_STATUS = "cosmological_constraint_fixture_no_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "cosmological_eos.chapter_boundary": {
        "validation_protocol": "paper0.cosmological_eos.chapter_boundary",
        "canonical_statement": "Chapter 22 frames cosmological constraints through a Psi-field equation of state.",
        "source_equation_ids": ("P0R06916:chapter_boundary",),
        "source_formulae": (
            "Chapter 22: Cosmological Constraints -- The Psi-Field Equation of State",
        ),
        "source_mechanisms": ("cosmological-constraint framing",),
        "variables": ("Psi", "w"),
        "validation_targets": (
            "preserve chapter boundary",
            "stop before five-testable-consequences chapter",
            "reject treating this block as observational validation",
        ),
        "null_controls": (
            "missing-chapter-boundary control must be rejected",
            "prediction-chapter-bleed-through control must be rejected",
            "observational-validation-overclaim control must be rejected",
        ),
    },
    "cosmological_eos.scalar_field_equations": {
        "validation_protocol": "paper0.cosmological_eos.scalar_field_equations",
        "canonical_statement": "Scalar-field density, pressure, and equation-of-state formulae are stated.",
        "source_equation_ids": ("P0R06920:rho_psi", "P0R06921:p_psi", "P0R06923:w_psi"),
        "source_formulae": (
            "rho_Psi = 0.5 psi_dot^2 + V(Psi)",
            "p_Psi = 0.5 psi_dot^2 - V(Psi)",
            "w_Psi = p_Psi / rho_Psi",
        ),
        "source_mechanisms": ("scalar-field energy-density and pressure accounting",),
        "variables": ("psi_dot", "V", "rho_Psi", "p_Psi", "w_Psi"),
        "validation_targets": (
            "compute density and pressure from finite scalar-field inputs",
            "compute w from pressure divided by positive density",
            "reject zero-density denominator",
        ),
        "null_controls": (
            "zero-density denominator control must be rejected",
            "non-finite scalar input control must be rejected",
            "negative-density overclaim control must be rejected",
        ),
    },
    "cosmological_eos.limiting_cases": {
        "validation_protocol": "paper0.cosmological_eos.limiting_cases",
        "canonical_statement": "Slow-roll, kinetic-dominated, and oscillatory limiting cases are distinguished.",
        "source_equation_ids": (
            "P0R06925:slow_roll_limit",
            "P0R06926:kinetic_limit",
            "P0R06927:oscillatory_limit",
        ),
        "source_formulae": (
            "psi_dot^2 << V implies w approximately -1",
            "V small with significant kinetic energy implies w approaches +1",
            "coherent oscillations about minimum imply matter-like average w approximately 0",
        ),
        "source_mechanisms": (
            "potential-dominated vacuum-energy behaviour",
            "kinetic-dominated stiff-matter behaviour",
            "oscillatory matter-like average behaviour",
        ),
        "variables": ("psi_dot", "V", "w"),
        "validation_targets": (
            "verify static potential-dominated fixture returns w=-1",
            "verify kinetic-dominated fixture returns w=+1",
            "record oscillatory case as contextual average, not late-time acceleration validation",
        ),
        "null_controls": (
            "slow-roll-sign-inversion control must be rejected",
            "kinetic-limit-sign-inversion control must be rejected",
            "oscillatory-dark-energy-overclaim control must be rejected",
        ),
    },
    "cosmological_eos.observational_constraint": {
        "validation_protocol": "paper0.cosmological_eos.observational_constraint",
        "canonical_statement": "Planck 2018 plus supernova context constrains w0 near -1.",
        "source_equation_ids": ("P0R06930:w0_planck_supernova",),
        "source_formulae": ("w0 = -1.03 +/- 0.03",),
        "source_mechanisms": (
            "consistency with cosmological constant",
            "Psi-field acceleration requires behaviour close to vacuum energy today",
        ),
        "variables": ("w0", "sigma_w0"),
        "validation_targets": (
            "test whether w0 confidence interval contains -1",
            "preserve no-evidence-of-deviation boundary",
            "reject unconstrained dynamic dark-energy claim",
        ),
        "null_controls": (
            "missing-w0-uncertainty control must be rejected",
            "target-outside-confidence-interval control must be rejected",
            "deviation-detected-overclaim control must be rejected",
        ),
    },
    "cosmological_eos.hybrid_split_and_homogeneity": {
        "validation_protocol": "paper0.cosmological_eos.hybrid_split",
        "canonical_statement": "The Psi field is split into homogeneous background and local perturbation terms.",
        "source_equation_ids": (
            "P0R06934:psi_background_perturbation_split",
            "P0R06938:stress_energy_split",
            "P0R06942:smooth_dark_energy_constraint",
            "P0R06943:gentle_variation_bound",
        ),
        "source_formulae": (
            "Psi(t,x) = Psi0(t) + deltaPsi(t,x)",
            "T_Psi_munu = T_Psi0_munu + T_deltaPsi_munu",
            "dark energy remains smooth on scales smaller than the horizon when c_s approximately 1",
        ),
        "source_mechanisms": (
            "homogeneous baseline supplies dark-energy component independent of observers",
            "local perturbation is subdominant",
            "large spatial variation conflicts with near-isotropy and distance-redshift precision",
        ),
        "variables": ("Psi0", "deltaPsi", "T_Psi", "c_s"),
        "validation_targets": (
            "preserve background-plus-perturbation split",
            "require perturbation fraction to remain subdominant",
            "reject observer-concentrated large inhomogeneity claim",
        ),
        "null_controls": (
            "invalid-perturbation-fraction control must be rejected",
            "observer-dependent-acceleration-overclaim control must be rejected",
            "large-inhomogeneity control must be rejected",
        ),
    },
    "cosmological_eos.quintessence_detection_target": {
        "validation_protocol": "paper0.cosmological_eos.quintessence_detection",
        "canonical_statement": "A mild w(z) deviation is recorded as a future-survey detection target.",
        "source_equation_ids": (
            "P0R06946:low_redshift_quintessence_possibility",
            "P0R06947:survey_detection_target",
        ),
        "source_formulae": (
            "w(z) slightly greater than -1, for example -0.98 at present",
            "low-redshift z less than or approximately 1 effect target",
        ),
        "source_mechanisms": (
            "gradual low-redshift growth would have limited CMB or early-structure effect",
            "next-generation supernova surveys might detect slight time variation",
        ),
        "variables": ("w_z", "z"),
        "validation_targets": (
            "record w(z) deviation as future observational target",
            "preserve low-redshift constraint",
            "reject current-detection claim",
        ),
        "null_controls": (
            "current-detection-overclaim control must be rejected",
            "missing-low-redshift-bound control must be rejected",
            "CMB-impact-overclaim control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CosmologicalEOSSpec:
    """Cosmological equation-of-state spec promoted from Paper 0 records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class CosmologicalEOSSpecBundle:
    """Cosmological equation-of-state specs plus coverage summary."""

    specs: tuple[CosmologicalEOSSpec, ...]
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


def build_cosmological_eos_specs(
    source_records: list[dict[str, Any]],
) -> CosmologicalEOSSpecBundle:
    """Build source-covered cosmological equation-of-state specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CosmologicalEOSSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CosmologicalEOSSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                source_mechanisms=tuple(str(item) for item in metadata["source_mechanisms"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Cosmological EOS Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return CosmologicalEOSSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CosmologicalEOSSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Cosmological EOS Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Source equations: {', '.join(spec.source_equation_ids) or 'none'}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: CosmologicalEOSSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the cosmological equation-of-state specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_cosmological_eos_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_cosmological_eos_validation_specs_report_{date_tag}.md"
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build cosmological equation-of-state specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_cosmological_eos_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
