#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 information-thermodynamics spec builder
"""Promote Paper 0 EQ0115-EQ0118 computational-unifier anchors into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.equation_register import get_paper0_equation_record

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
    "computational.cyclic_operator_boundary": (
        "P0R05929",
        "P0R05930",
        "P0R05931",
        "P0R05932",
        "P0R05933",
        "P0R05934",
        "P0R05935",
    ),
    "computational.tsvf_abl_boundary": (
        "P0R05936",
        "P0R05937",
        "P0R05938",
        "P0R05939",
        "P0R05940",
    ),
    "computational.info_thermodynamics": (
        "P0R05942",
        "P0R05943",
        "P0R05944",
        "P0R05945",
        "P0R05946",
        "P0R05947",
        "P0R05949",
        "P0R05950",
        "P0R05951",
        "P0R05952",
        "P0R05953",
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "computational.cyclic_operator_boundary": {
        "validation_protocol": "paper0.computational.cyclic_operator.boundary_periodicity",
        "implementation_status": "validation_spec_pending_executable_fixture",
        "null_controls": (
            "boundary-only claim label until periodicity data and falsifiers exist",
            "identity-period control must close with zero residual",
            "non-unitary operator control must be rejected before interpretation",
        ),
        "executable_validation_targets": (
            "verify operator unitarity on a finite declared Hilbert space",
            "measure cycle-closure residual after one T_MMC period",
            "record periodicity failure as falsifying this computational boundary model",
        ),
    },
    "computational.tsvf_abl_boundary": {
        "validation_protocol": "paper0.computational.tsvf_abl.boundary_probability",
        "implementation_status": "validation_spec_pending_executable_fixture",
        "null_controls": (
            "boundary-only retrocausal interpretation label",
            "ABL zero-denominator guard",
            "non-projector measurement-family rejection control",
        ),
        "executable_validation_targets": (
            "verify ABL probability normalisation over complete projectors",
            "recover Born-rule probabilities under trivial post-selection",
            "record denominator failure and non-projector inputs as invalid specs",
        ),
    },
    "computational.info_thermodynamics": {
        "validation_protocol": "paper0.computational.info_thermodynamics.entropy_budget",
        "implementation_status": "validation_spec_pending_executable_fixture",
        "null_controls": (
            "Landauer lower-bound violation must fail the validation",
            "independent-channel zero mutual information control",
            "negative total entropy-rate budget rejects GSL claim promotion",
        ),
        "executable_validation_targets": (
            "estimate thermodynamic and information entropy rates on a shared grid",
            "scan mutual information against measured negentropy rate",
            "verify total entropy-rate non-negativity under explicit Landauer cost",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class InformationThermodynamicsValidationSpec:
    """Validation spec promoted from Paper 0 computational-unifier anchors."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_latex: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: dict[str, str]
    assumptions: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class InformationThermodynamicsValidationSpecBundle:
    """Computational-unifier validation specs plus coverage summary."""

    specs: tuple[InformationThermodynamicsValidationSpec, ...]
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


def build_information_thermodynamics_validation_specs(
    source_records: list[dict[str, Any]],
) -> InformationThermodynamicsValidationSpecBundle:
    """Build source-covered validation specs for EQ0115-EQ0118."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[InformationThermodynamicsValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in (
        "computational.cyclic_operator_boundary",
        "computational.tsvf_abl_boundary",
        "computational.info_thermodynamics",
    ):
        equation_record = get_paper0_equation_record(key)
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            InformationThermodynamicsValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript=equation_record.manuscript,
                section_path=equation_record.section_path,
                canonical_latex=equation_record.canonical_latex,
                source_equation_ids=equation_record.source_equation_ids,
                source_ledger_ids=ledger_ids,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=anchor_math_ids,
                variables=dict(equation_record.variables),
                assumptions=equation_record.assumptions,
                validation_targets=equation_record.validation_targets,
                executable_validation_targets=tuple(metadata["executable_validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                implementation_status=str(metadata["implementation_status"]),
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(required_ids),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": consumed_ids == required_ids,
        "unconsumed_source_ledger_ids": sorted(required_ids - consumed_ids),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "EQ0115-EQ0118 anchors are promoted only into validation specifications. "
            "Temporal-boundary and retrocausal interpretations remain boundary-only "
            "until operator, probability, entropy-budget, and falsification controls pass."
        ),
    }
    return InformationThermodynamicsValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: InformationThermodynamicsValidationSpecBundle) -> str:
    """Render a concise Markdown report for computational-unifier validation specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Information-Thermodynamics Validation Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        f"- Coverage status: `{status}`",
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
            "Provider submission remains out of scope. These records are "
            "source-anchored validation specifications only; temporal-boundary, "
            "retrocausal, and thermodynamic interpretations require executable "
            "controls before any claim promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: InformationThermodynamicsValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the computational-unifier spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_information_thermodynamics_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_information_thermodynamics_validation_specs_report_{date_tag}.md"
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
    bundle = build_information_thermodynamics_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
