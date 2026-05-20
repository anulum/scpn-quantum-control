#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Status and Method continuation spec builder
"""Promote Paper 0 Status and Method continuation records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(391, 401))
BLANK_SEPARATOR_IDS = ("P0R00400",)
CLAIM_BOUNDARY = "source-bounded Status and Method continuation; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "status_method_continuation.not_absolute_truths_boundary": {
        "context_id": "not_absolute_truths_boundary",
        "validation_protocol": "paper0.status_method_continuation.not_absolute_truths_boundary",
        "canonical_statement": (
            "The continuation states what the framework is not: not absolute truth, "
            "not metaphor literalisation, and not a bypass of empirical standards."
        ),
        "source_equation_ids": (
            "P0R00391:what_this_is_not_boundary",
            "P0R00392:operational_commitments_header",
        ),
        "source_formulae": (
            "not a catalogue of absolute truths",
            "not a literalisation of every metaphor or analogy",
            "analogy-class carries no ontological load",
            "not a licence to bypass empirical standards",
        ),
        "test_protocols": ("classify not-boundaries and reject doctrine/metaphor promotion",),
        "null_results": ("not-boundary text is not empirical validation evidence",),
        "variables": ("absolute_truths", "metaphor_literalisation", "empirical_bypass"),
        "validation_targets": (
            "reject doctrine status",
            "reject ontological load for analogy-class mathematics",
            "reject bypass of empirical standards",
        ),
        "null_controls": (
            "literalised-metaphor control must be rejected",
            "empirical-bypass control must be rejected",
        ),
    },
    "status_method_continuation.commitments_continuation": {
        "context_id": "commitments_continuation",
        "validation_protocol": "paper0.status_method_continuation.commitments_continuation",
        "canonical_statement": (
            "The continuation restates the four operating commitments and adds "
            "pre-registration, out-of-sample support, claim labels, and explicit changes."
        ),
        "source_equation_ids": ("P0R00393:operational_commitments_continuation",),
        "source_formulae": (
            "Falsifiability first",
            "Hypothesis registry",
            "Tiered status",
            "Versioning and correction",
            "out-of-sample support",
        ),
        "test_protocols": ("preserve four commitment labels and support boundary",),
        "null_results": ("commitment restatement is not evidence for a claim",),
        "variables": (
            "falsifiability_first",
            "hypothesis_registry",
            "tiered_status",
            "versioning_and_correction",
        ),
        "validation_targets": (
            "preserve pre-registration and negative-result pruning",
            "preserve registry hypothesis as non-premise",
            "preserve tiered claim labelling",
            "preserve explicit change/correction requirement",
        ),
        "null_controls": (
            "registry-as-premise control must be rejected",
            "unlabelled-claim-status control must be rejected",
        ),
    },
    "status_method_continuation.axioms_as_generative_hypotheses": {
        "context_id": "axioms_as_generative_hypotheses",
        "validation_protocol": "paper0.status_method_continuation.axioms_as_generative_hypotheses",
        "canonical_statement": (
            "The axioms are starting points, not conclusions; their worth is tested "
            "by derived coherence and survived or failed tests."
        ),
        "source_equation_ids": (
            "P0R00394:how_to_read_axioms",
            "P0R00395:axioms_as_generative_hypotheses",
        ),
        "source_formulae": (
            "starting points, not conclusions",
            "generative hypotheses",
            "what follows and what fails",
            "tests it survives or fails",
        ),
        "test_protocols": ("preserve axiom status as generative hypotheses",),
        "null_results": ("axiom declaration is not proof of derived claims",),
        "variables": ("axioms", "generative_hypotheses", "survival_tests"),
        "validation_targets": (
            "preserve starting-point boundary",
            "preserve failure-question framing",
            "preserve survived/failed test criterion",
        ),
        "null_controls": (
            "axiom-as-conclusion control must be rejected",
            "untested-axiom-proof control must be rejected",
        ),
    },
    "status_method_continuation.productive_disagreement_protocol": {
        "context_id": "productive_disagreement_protocol",
        "validation_protocol": "paper0.status_method_continuation.productive_disagreement_protocol",
        "canonical_statement": (
            "Productive disagreement requires prediction/baseline comparison, empirical "
            "handles for analogies, and replacement models in the same observable slot."
        ),
        "source_equation_ids": (
            "P0R00396:productive_disagreement_header",
            "P0R00397:productive_disagreement_protocol",
        ),
        "source_formulae": (
            "stated prediction and alternative baseline",
            "empirical handle",
            "same observables, stricter assumptions, stronger fit",
        ),
        "test_protocols": ("classify disagreement moves and reject untested dismissal",),
        "null_results": ("disagreement protocol is not validation by itself",),
        "variables": ("prediction_baseline", "analogy_handle", "replacement_model"),
        "validation_targets": (
            "preserve prediction-baseline comparison",
            "preserve analogy-handle burden",
            "preserve same-slot replacement model path",
        ),
        "null_controls": (
            "analogy-without-handle control must be rejected",
            "dismiss-without-test control must be rejected",
        ),
    },
    "status_method_continuation.standing_invitation_closure": {
        "context_id": "standing_invitation_closure",
        "validation_protocol": "paper0.status_method_continuation.standing_invitation_closure",
        "canonical_statement": (
            "The continuation closes as a standing invitation: foundation stone, not "
            "capstone, with expected refutation, reinforcement, and supersession."
        ),
        "source_equation_ids": (
            "P0R00398:standing_invitation_header",
            "P0R00399:foundation_stone_not_capstone",
            "P0R00400:blank_separator",
        ),
        "source_formulae": (
            "foundation stone, not a capstone",
            "refutes some parts",
            "reinforces others",
            "supersedes many",
            "SCPN mandate boundary P0R00401",
        ),
        "test_protocols": ("preserve standing invitation and next-boundary accounting",),
        "null_results": ("standing invitation is not validation evidence",),
        "variables": ("foundation_stone", "capstone_rejection", "scp_mandate_boundary"),
        "validation_targets": (
            "preserve foundation-stone closure",
            "preserve refute/reinforce/supersede expectation",
            "preserve SCPN mandate boundary",
        ),
        "null_controls": (
            "capstone-finality control must be rejected",
            "missing-SCPN-mandate-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class StatusMethodContinuationSpec:
    """Status and Method continuation spec promoted from Paper 0 records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class StatusMethodContinuationSpecBundle:
    """Status and Method continuation specs plus source coverage summary."""

    specs: tuple[StatusMethodContinuationSpec, ...]
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


def build_status_method_continuation_specs(
    source_records: list[dict[str, Any]],
) -> StatusMethodContinuationSpecBundle:
    """Build source-covered Status and Method continuation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[StatusMethodContinuationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            StatusMethodContinuationSpec(
                key=key,
                context_id=str(metadata["context_id"]),
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
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="source_methodology_continuation_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Status and Method Continuation Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "operational_commitment_count": 4,
        "scp_mandate_boundary": "P0R00401",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return StatusMethodContinuationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> StatusMethodContinuationSpecBundle:
    """Build Status and Method continuation specs from the canonical ledger."""
    return build_status_method_continuation_specs(load_jsonl(path))


def write_outputs(
    bundle: StatusMethodContinuationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Status and Method continuation spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_status_method_continuation_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_status_method_continuation_validation_specs_report_{date_tag}.md"
    )
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: StatusMethodContinuationSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Status and Method Continuation Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Blank separators: {bundle.summary['blank_separator_count']}",
        f"- Operational commitments: {bundle.summary['operational_commitment_count']}",
        f"- SCPN mandate boundary: {bundle.summary['scp_mandate_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
        if "empirical handle" in spec.source_formulae:
            lines.append("  - Analogy burden: empirical handle required")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 Status and Method continuation validation specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0 if bundle.summary["coverage_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
