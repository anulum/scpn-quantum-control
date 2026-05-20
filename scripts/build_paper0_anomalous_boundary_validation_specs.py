#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 anomalous-boundary spec builder
"""Promote Paper 0 anomalous-phenomena records into boundary specs."""

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

SOURCE_LEDGER_IDS = ("P0R06212", "P0R06213", "P0R06214")

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "applied.anomalous_boundary.tsvf_precognition_boundary": {
        "validation_protocol": "paper0.applied.anomalous_boundary.tsvf_precognition",
        "canonical_statement": (
            "The precognition wording is bounded to TSVF/ABL post-selection "
            "probability conditioning in finite Hilbert spaces."
        ),
        "variables": ("pre_state", "post_state", "projectors", "abl_probability"),
        "assumptions": (
            "post-selection is conditioning, not backwards causal signalling",
            "probabilities are normalised over explicit projectors",
            "passing the fixture is not anomalous evidence",
        ),
        "validation_targets": (
            "verify ABL probabilities normalise under post-selection",
            "verify changing post-selection changes conditional probabilities",
            "verify no-retrocausal-signalling control preserves marginal bounds",
        ),
        "null_controls": (
            "orthogonal-post-selection control must reject zero denominator",
            "born-rule control must match when post-selection is uninformative",
            "retrocausal-signalling control must be labelled unsupported",
        ),
    },
    "applied.anomalous_boundary.entanglement_correlation_boundary": {
        "validation_protocol": "paper0.applied.anomalous_boundary.entanglement_correlation",
        "canonical_statement": (
            "Remote-perception/telepathy wording is bounded to entanglement "
            "correlation tests that can violate classical correlation limits "
            "without enabling signalling."
        ),
        "variables": ("bell_state", "measurement_angles", "chsh_value", "no_signalling_residual"),
        "assumptions": (
            "ER=EPR language remains interpretive context only",
            "CHSH correlation is not information transfer",
            "passing the fixture is not anomalous evidence",
        ),
        "validation_targets": (
            "verify Bell-state CHSH value exceeds classical bound",
            "verify product-state control stays within classical bound",
            "verify no-signalling marginal residual is zero",
        ),
        "null_controls": (
            "product-state control must not violate CHSH bound",
            "signalling control must reject marginal dependence",
            "non-normalised-state control must be rejected",
        ),
    },
    "applied.anomalous_boundary.weak_measurement_bias_boundary": {
        "validation_protocol": "paper0.applied.anomalous_boundary.weak_measurement_bias",
        "canonical_statement": (
            "Psychokinesis wording is bounded to a weak-measurement probability-bias "
            "fixture with explicit bounded intent parameter and null controls."
        ),
        "variables": (
            "prior_probability",
            "intent_bias",
            "measurement_strength",
            "biased_probability",
        ),
        "assumptions": (
            "intent is a simulator control parameter, not a mind-matter claim",
            "probabilities remain in the unit interval",
            "passing the fixture is not anomalous evidence",
        ),
        "validation_targets": (
            "verify bounded weak bias shifts probability monotonically",
            "verify zero-intent control returns prior probability",
            "verify out-of-range intent or strength is rejected",
        ),
        "null_controls": (
            "zero-intent control must produce zero probability shift",
            "saturated-probability control must remain bounded",
            "out-of-range-bias control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AnomalousBoundaryValidationSpec:
    """Validation spec promoted from Paper 0 anomalous-phenomena boundary records."""

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
class AnomalousBoundaryValidationSpecBundle:
    """Anomalous-phenomena boundary specs plus coverage summary."""

    specs: tuple[AnomalousBoundaryValidationSpec, ...]
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


def build_anomalous_boundary_validation_specs(
    source_records: list[dict[str, Any]],
) -> AnomalousBoundaryValidationSpecBundle:
    """Build source-covered anomalous-phenomena boundary validation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    required_ids = set(SOURCE_LEDGER_IDS)
    missing = sorted(required_ids - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    anchor_math_ids = tuple(
        sorted({str(math_id) for anchor in anchors for math_id in anchor.get("math_ids", [])})
    )
    specs: list[AnomalousBoundaryValidationSpec] = []
    for key in (
        "applied.anomalous_boundary.tsvf_precognition_boundary",
        "applied.anomalous_boundary.entanglement_correlation_boundary",
        "applied.anomalous_boundary.weak_measurement_bias_boundary",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            AnomalousBoundaryValidationSpec(
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
                anchor_math_ids=anchor_math_ids,
                variables=tuple(str(item) for item in metadata["variables"]),
                assumptions=tuple(str(item) for item in metadata["assumptions"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="simulator-only falsification boundary; not anomalous evidence",
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
        "all_specs_have_anomalous_evidence_boundary": all(
            "not anomalous evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06212-P0R06214 are promoted as source-covered falsification-boundary "
            "specifications only. Passing fixtures is not anomalous evidence."
        ),
    }
    return AnomalousBoundaryValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: AnomalousBoundaryValidationSpecBundle) -> str:
    """Render a concise Markdown report for anomalous-boundary specs."""
    lines = [
        "# Paper 0 Anomalous-Phenomena Boundary Validation Specs",
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
                f"- Executable targets: `{len(spec.executable_validation_targets)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored falsification-boundary specifications "
            "only. Passing any fixture is not anomalous evidence and does not "
            "establish precognition, telepathy, remote perception, or psychokinesis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: AnomalousBoundaryValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the anomalous-boundary bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_anomalous_boundary_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_anomalous_boundary_validation_specs_report_{date_tag}.md"
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
    bundle = build_anomalous_boundary_validation_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
