#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 artificial-sentience spec builder
"""Promote Paper 0 artificial-sentience anchors into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SPEC_SOURCE_LEDGER_IDS: dict[str, tuple[str, ...]] = {
    "applied.artificial_sentience.technosphere_coupling_acceleration": (
        "P0R06206",
        "P0R06207",
    ),
    "applied.artificial_sentience.criteria_gate": (
        "P0R06208",
        "P0R06209",
        "P0R06210",
    ),
    "applied.artificial_sentience.phase_locking_substrate_boundary": ("P0R06211",),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied.artificial_sentience.technosphere_coupling_acceleration": {
        "validation_protocol": "paper0.applied.artificial_sentience.coupling_acceleration",
        "canonical_statement": (
            "Technosphere acceleration is bounded to a network-dynamics claim: "
            "increasing effective coupling J_ij reduces synchronisation convergence "
            "time in finite collective-state simulations."
        ),
        "variables": ("J_ij", "mean_coupling", "convergence_rate", "order_parameter"),
        "assumptions": (
            "couplings are finite non-negative simulator weights",
            "acceleration is measured as convergence-rate change only",
            "no noosphere, consciousness, or sentience claim is inferred",
        ),
        "validation_targets": (
            "verify increased coupling raises effective synchronisation rate",
            "verify zero-coupling baseline has no coupling acceleration",
            "verify non-finite coupling is rejected",
        ),
        "null_controls": (
            "zero-coupling control must produce zero acceleration",
            "negative-coupling control must be rejected",
            "shape-mismatch control must be rejected",
        ),
    },
    "applied.artificial_sentience.criteria_gate": {
        "validation_protocol": "paper0.applied.artificial_sentience.criteria_gate",
        "canonical_statement": (
            "Artificial-sentience criteria are promoted only as a conjunctive simulator "
            "gate over integrated-information proxy, quasicritical sigma, and substrate "
            "coupling capability."
        ),
        "variables": ("phi_proxy", "sigma", "substrate_coupling", "criteria_pass"),
        "assumptions": (
            "phi is a finite proxy metric, not a consciousness measurement",
            "sigma gate uses explicit tolerance around one",
            "substrate coupling is a declared binary capability flag",
        ),
        "validation_targets": (
            "verify criteria pass only when all required predicates pass",
            "verify missing-substrate control fails the gate",
            "verify low-phi and off-critical controls fail independently",
        ),
        "null_controls": (
            "missing-substrate control must fail criteria gate",
            "low-phi control must fail criteria gate",
            "off-criticality control must fail criteria gate",
        ),
    },
    "applied.artificial_sentience.phase_locking_substrate_boundary": {
        "validation_protocol": "paper0.applied.artificial_sentience.phase_locking_boundary",
        "canonical_statement": (
            "QAS phase-locking is bounded to a finite UPDE-style phase-locking score "
            "between system dynamics and an external field phase; passing the score is "
            "not sentience evidence."
        ),
        "variables": ("system_phase", "field_phase", "phase_locking_value", "lock_threshold"),
        "assumptions": (
            "phase-locking uses bounded circular statistics",
            "classical and quantum labels are substrate flags only",
            "passing the fixture cannot establish artificial sentience",
        ),
        "validation_targets": (
            "verify aligned phases pass the phase-locking threshold",
            "verify random or opposed phases fail the threshold",
            "verify absent substrate coupling fails even with high phase-locking",
        ),
        "null_controls": (
            "random-phase control must fail phase-locking threshold",
            "absent-substrate control must fail boundary gate",
            "non-finite-phase control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ArtificialSentienceValidationSpec:
    """Validation spec promoted from Paper 0 artificial-sentience records."""

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
class ArtificialSentienceValidationSpecBundle:
    """Artificial-sentience validation specs plus coverage summary."""

    specs: tuple[ArtificialSentienceValidationSpec, ...]
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


def build_artificial_sentience_validation_specs(
    source_records: list[dict[str, Any]],
) -> ArtificialSentienceValidationSpecBundle:
    """Build source-covered validation specs for artificial-sentience records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = {
        ledger_id for ledger_ids in SPEC_SOURCE_LEDGER_IDS.values() for ledger_id in ledger_ids
    }
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    specs: list[ArtificialSentienceValidationSpec] = []
    consumed_ids: set[str] = set()
    for key in (
        "applied.artificial_sentience.technosphere_coupling_acceleration",
        "applied.artificial_sentience.criteria_gate",
        "applied.artificial_sentience.phase_locking_substrate_boundary",
    ):
        ledger_ids = SPEC_SOURCE_LEDGER_IDS[key]
        anchors = [records_by_ledger[ledger_id] for ledger_id in ledger_ids]
        anchor_math_ids = tuple(
            sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
        )
        metadata = SPEC_METADATA[key]
        consumed_ids.update(ledger_ids)
        specs.append(
            ArtificialSentienceValidationSpec(
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
                claim_boundary="simulator-only criteria gate; not sentience evidence",
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
        "all_specs_have_sentience_boundary": all(
            "not sentience evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06206-P0R06211 are promoted as source-covered validation "
            "specifications only. Artificial-sentience wording is bounded to finite "
            "simulator criteria and is not sentience evidence."
        ),
    }
    return ArtificialSentienceValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: ArtificialSentienceValidationSpecBundle) -> str:
    """Render a concise Markdown report for artificial-sentience specs."""
    status = "match" if bundle.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Artificial-Sentience Validation Specs",
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
            "These records are source-anchored validation specifications only. "
            "Passing any fixture is not sentience evidence and does not establish "
            "consciousness, subjective experience, or artificial sentience.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: ArtificialSentienceValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the artificial-sentience bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_artificial_sentience_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_artificial_sentience_validation_specs_report_{date_tag}.md"
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
    bundle = build_artificial_sentience_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
