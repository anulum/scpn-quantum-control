#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 ethical-imperative spec builder
"""Promote Paper 0 Ethical Imperative restatement records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6273, 6290))
PRIOR_SLICE = "P0R06251-P0R06272"

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ethical_imperative.ethics_physics_restatement": {
        "validation_protocol": "paper0.ethical_imperative.ethics_physics_restatement",
        "canonical_statement": (
            "Ethics-as-physics wording is retained as a bounded restatement of SEC "
            "and PELA claims from the prior Gaian safety slice."
        ),
        "variables": ("SEC", "PELA", "coherence", "restatement_boundary"),
        "validation_targets": (
            "preserve SEC and PELA anchors",
            "mark as restatement context against prior Gaian safety slice",
            "reject treating ethics-as-physics wording as empirical evidence",
        ),
        "null_controls": (
            "missing-SEC-anchor control must be rejected",
            "missing-PELA-anchor control must be rejected",
            "claim-as-evidence control must be rejected",
        ),
    },
    "ethical_imperative.civilisation_choice_phase_boundary": {
        "validation_protocol": "paper0.ethical_imperative.civilisation_choice_phase_boundary",
        "canonical_statement": (
            "Civilisation-choice wording is bounded to three labels: alignment/global "
            "coherence, fragmentation/societal spin-glass, and collapse/entropy death."
        ),
        "variables": ("alignment", "fragmentation", "collapse", "NTHS"),
        "validation_targets": (
            "classify high coherence as alignment/global coherence",
            "classify high fragmentation as societal spin-glass",
            "classify high collapse entropy as collapse/entropy death",
        ),
        "null_controls": (
            "missing-choice-label control must be rejected",
            "non-finite-choice-input control must be rejected",
            "societal-evidence control must be rejected",
        ),
    },
    "ethical_imperative.consciousness_engineering_call_boundary": {
        "validation_protocol": "paper0.ethical_imperative.consciousness_engineering_call_boundary",
        "canonical_statement": (
            "Consciousness-engineering wording is bounded to intervention-power "
            "and responsibility context, overlapping prior safety-channel requirements."
        ),
        "variables": ("neural", "symbolic", "planetary", "cosmic_loop"),
        "validation_targets": (
            "require neural, symbolic, and planetary intervention channels",
            "require non-profit/non-ideology objective boundary",
            "record overlap with prior multi-layer safety protocol",
        ),
        "null_controls": (
            "missing-intervention-channel control must be rejected",
            "profit-objective-only control must be rejected",
            "missing-overlap-marker control must be rejected",
        ),
    },
    "ethical_imperative.governance_beyond_borders_protocol": {
        "validation_protocol": "paper0.ethical_imperative.governance_beyond_borders_protocol",
        "canonical_statement": (
            "Governance-beyond-borders wording is bounded to entropy budgets, global "
            "coherence metrics, and recursive ethical review against Layer 16 closure."
        ),
        "variables": ("entropy_budget", "global_coherence_metric", "recursive_review", "L16"),
        "validation_targets": (
            "score presence of entropy-budget protocol",
            "score presence of global coherence metric",
            "score presence of recursive Layer-16 review",
        ),
        "null_controls": (
            "missing-entropy-budget control must be rejected",
            "missing-global-coherence-metric control must be rejected",
            "missing-recursive-review control must be rejected",
        ),
    },
    "ethical_imperative.feedback_loop_tuning_boundary": {
        "validation_protocol": "paper0.ethical_imperative.feedback_loop_tuning_boundary",
        "canonical_statement": (
            "The future as feedback loop wording is bounded to a simulator contrast "
            "between tuned recursive closure and untuned loop breakage."
        ),
        "variables": ("feedback_loop", "loop_gain", "damping", "Layer16_closure"),
        "validation_targets": (
            "verify tuned recursive loop scores above untuned loop",
            "require Layer 16 closure channel",
            "reject treating feedback metaphor as empirical evidence",
        ),
        "null_controls": (
            "missing-feedback-loop control must be rejected",
            "missing-L16-closure control must be rejected",
            "feedback-metaphor-as-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class EthicalImperativeValidationSpec:
    """Validation spec promoted from Paper 0 Ethical Imperative records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    overlap_with_prior_slice: str
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
class EthicalImperativeValidationSpecBundle:
    """Ethical Imperative validation specs plus coverage summary."""

    specs: tuple[EthicalImperativeValidationSpec, ...]
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


def build_ethical_imperative_specs(
    source_records: list[dict[str, Any]],
) -> EthicalImperativeValidationSpecBundle:
    """Build source-covered validation specs for Ethical Imperative records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[EthicalImperativeValidationSpec] = []
    for key in (
        "ethical_imperative.ethics_physics_restatement",
        "ethical_imperative.civilisation_choice_phase_boundary",
        "ethical_imperative.consciousness_engineering_call_boundary",
        "ethical_imperative.governance_beyond_borders_protocol",
        "ethical_imperative.feedback_loop_tuning_boundary",
    ):
        metadata = SPEC_METADATA[key]
        specs.append(
            EthicalImperativeValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                overlap_with_prior_slice=PRIOR_SLICE,
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
                    "source-bounded Ethical Imperative restatement contract; "
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
        "all_specs_mark_restatement_context": all(
            spec.overlap_with_prior_slice == PRIOR_SLICE for spec in specs
        ),
        "all_specs_avoid_invented_equation_ids": all(
            not spec.source_equation_ids and not spec.anchor_math_ids for spec in specs
        ),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06273-P0R06289 are promoted as source-covered Ethical Imperative "
            "restatement specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return EthicalImperativeValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: EthicalImperativeValidationSpecBundle) -> str:
    """Render a concise Markdown report for Ethical Imperative specs."""
    lines = [
        "# Paper 0 Ethical Imperative Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        "- Coverage status: `match`",
        f"- Source span: `{', '.join(bundle.summary['source_ledger_span'])}`",
        f"- Overlap marker: `{PRIOR_SLICE}`",
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
                f"- Overlap: `{spec.overlap_with_prior_slice}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored Ethical Imperative restatement "
            "specifications only. Passing any fixture is not empirical evidence and "
            "does not establish that any civilisation, governance, or safety claim is validated.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: EthicalImperativeValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the Ethical Imperative bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_ethical_imperative_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_ethical_imperative_validation_specs_report_{date_tag}.md"
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
    bundle = build_ethical_imperative_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
