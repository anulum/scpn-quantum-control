#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Stuart-Landau precision spec builder
"""Promote Paper 0 Stuart-Landau precision anchors into validation specs."""

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

SPEC_SOURCE_LEDGER_IDS: dict[str, tuple[str, ...]] = {
    "computational.stuart_landau_precision_upgrade": (
        "P0R06179",
        "P0R06180",
        "P0R06181",
        "P0R06182",
        "P0R06183",
        "P0R06184",
        "P0R06185",
        "P0R06186",
    ),
    "computational.precision_weighted_phase_amplitude_dynamics": (
        "P0R06187",
        "P0R06188",
    ),
    "computational.salience_radial_precision_control": (
        "P0R06189",
        "P0R06190",
        "P0R06191",
        "P0R06192",
        "P0R06193",
        "P0R06194",
        "P0R06195",
        "P0R06196",
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "computational.stuart_landau_precision_upgrade": {
        "validation_protocol": "paper0.computational.stuart_landau.precision_upgrade",
        "canonical_statement": (
            "The phase-only Kuramoto precision gap is promoted as a bounded "
            "mathematical claim: confidence requires a positive amplitude state "
            "R_j in the complex Stuart-Landau variable Z_j = R_j exp(i theta_j)."
        ),
        "variables": ("Z_j", "R_j", "theta_j", "rho_j", "omega_j", "K_jk"),
        "assumptions": (
            "amplitudes are strictly positive when polar phase equations are evaluated",
            "couplings are finite real weights in the simulator fixture",
            "biological gain interpretation remains outside the mathematical fixture",
        ),
        "validation_targets": (
            "verify complex Stuart-Landau derivative is finite for positive radius",
            "verify complex derivative exactly decomposes into radial and phase rates",
            "verify zero or negative radius is rejected before amplitude-ratio evaluation",
        ),
        "null_controls": (
            "zero-radius control must reject polar decomposition",
            "non-finite-complex-state control must reject derivative evaluation",
            "phase-only control must expose missing amplitude confidence channel",
        ),
    },
    "computational.precision_weighted_phase_amplitude_dynamics": {
        "validation_protocol": "paper0.computational.stuart_landau.polar_dynamics",
        "canonical_statement": (
            "The promoted polar equations require the phase coupling term "
            "K_jk (R_k/R_j) sin(theta_k - theta_j) and radial term "
            "R_j(rho_j - R_j^2) + sum K_jk R_k cos(theta_k - theta_j)."
        ),
        "variables": ("theta_dot_j", "R_dot_j", "R_k_over_R_j", "eta_theta", "eta_R"),
        "assumptions": (
            "phase noise and radial noise are explicit additive terms",
            "amplitude ratio is evaluated only for positive R_j",
            "uniform-amplitude limit must recover phase-only Kuramoto coupling",
        ),
        "validation_targets": (
            "verify amplitude ratio scales signed phase residuals",
            "verify radial equation matches real polar projection",
            "verify uniform-amplitude limit recovers unweighted Kuramoto phase coupling",
        ),
        "null_controls": (
            "uniform-amplitude control must remove amplitude-ratio modulation",
            "negative-amplitude control must be rejected",
            "mismatched-shape control must reject uncoupled polar arrays",
        ),
    },
    "computational.salience_radial_precision_control": {
        "validation_protocol": "paper0.computational.stuart_landau.salience_radial_control",
        "canonical_statement": (
            "Salience-network precision control is bounded to radial modulation: "
            "changing rho_j changes R_dot_j and thereby the amplitude-weighted "
            "phase residual contribution, without directly changing phase labels."
        ),
        "variables": ("rho_j", "R_j", "precision_weight", "phase_residual", "radial_gain"),
        "assumptions": (
            "rho_j is the explicit bifurcation/gain control parameter",
            "sensory/prior labels are fixture labels rather than biological confirmation",
            "claim promotion requires finite radial and phase controls",
        ),
        "validation_targets": (
            "verify increasing rho_j increases radial growth at fixed state",
            "verify high incoming amplitude dominates phase residual contribution",
            "verify high prior amplitude mutes incoming phase residual contribution",
        ),
        "null_controls": (
            "direct-phase-salience control must not bypass radial gain",
            "zero-incoming-amplitude control must remove incoming residual dominance",
            "non-finite-rho control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionValidationSpec:
    """Validation spec promoted from Paper 0 Stuart-Landau precision records."""

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
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionValidationSpecBundle:
    """Stuart-Landau precision validation specs plus coverage summary."""

    specs: tuple[StuartLandauPrecisionValidationSpec, ...]
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


def build_stuart_landau_precision_validation_specs(
    source_records: list[dict[str, Any]],
) -> StuartLandauPrecisionValidationSpecBundle:
    """Build source-covered validation specs for Stuart-Landau precision records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[StuartLandauPrecisionValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in (
        "computational.stuart_landau_precision_upgrade",
        "computational.precision_weighted_phase_amplitude_dynamics",
        "computational.salience_radial_precision_control",
    ):
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            StuartLandauPrecisionValidationSpec(
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
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06179-P0R06196 are promoted as source-covered validation "
            "specifications. No standalone equation IDs are invented for formula "
            "text lacking canonical EQ anchors; Stuart-Landau and salience "
            "interpretations remain bounded to finite simulator fixtures."
        ),
    }
    return StuartLandauPrecisionValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: StuartLandauPrecisionValidationSpecBundle) -> str:
    """Render a concise Markdown report for Stuart-Landau precision specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Stuart-Landau Precision Validation Specs",
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
                f"- Source equations: `{', '.join(spec.source_equation_ids)}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                f"- Status: `{spec.implementation_status}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "No standalone equation IDs are invented for formula text that was not "
            "assigned canonical equation anchors during Paper 0 extraction. These "
            "records are source-anchored validation specifications only; "
            "Stuart-Landau precision and salience interpretations require "
            "executable fixtures and falsification controls before claim promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: StuartLandauPrecisionValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Stuart-Landau precision bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_stuart_landau_precision_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_stuart_landau_precision_validation_specs_report_{date_tag}.md"
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
    bundle = build_stuart_landau_precision_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
