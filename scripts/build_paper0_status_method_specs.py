#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Status and Method spec builder
"""Promote Paper 0 Status and Method records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(358, 391))
BLANK_SEPARATOR_IDS = ("P0R00364", "P0R00385")
CLAIM_BOUNDARY = "source-bounded Status and Method protocol; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "status_method.living_research_programme": {
        "context_id": "living_research_programme",
        "validation_protocol": "paper0.status_method.living_research_programme",
        "canonical_statement": (
            "Status and Method frames the manuscript as a dynamic, version-controlled "
            "research programme rather than doctrine."
        ),
        "source_equation_ids": (
            "P0R00358:status_method_title",
            "P0R00359:version_controlled_hypotheses",
            "P0R00360:operational_commitments",
            "P0R00387-P0R00390:research_programme_not_doctrine",
        ),
        "source_formulae": (
            "research programme, not a finished doctrine",
            "version-controlled system of hypotheses",
            "working model",
            "not a new religion or absolute truths",
        ),
        "test_protocols": ("preserve doctrine rejection and living-programme boundary",),
        "null_results": ("methodology declaration is not empirical validation evidence",),
        "variables": ("living_programme", "version_control", "doctrine_rejection"),
        "validation_targets": (
            "preserve research-programme status",
            "preserve provisional claim boundary",
            "preserve doctrine/religion rejection",
        ),
        "null_controls": (
            "doctrine-promotion control must be rejected",
            "absolute-truth control must be rejected",
        ),
    },
    "status_method.operational_commitments": {
        "context_id": "operational_commitments",
        "validation_protocol": "paper0.status_method.operational_commitments",
        "canonical_statement": (
            "The slice codifies falsifiability, a hypothesis registry, tiered claim "
            "status, and versioning/correction as operating commitments."
        ),
        "source_equation_ids": (
            "P0R00360:operational_commitment_list",
            "P0R00362:working_model_and_metaphor_boundary",
            "P0R00363:testability_and_prediction_list",
        ),
        "source_formulae": (
            "Falsifiability first",
            "Hypothesis registry",
            "Tiered status",
            "Versioning and correction",
        ),
        "test_protocols": ("classify operational commitments into executable gates",),
        "null_results": ("commitment list is not evidence for any claim",),
        "variables": (
            "falsifiability_first",
            "hypothesis_registry",
            "tiered_status",
            "versioning_and_correction",
        ),
        "validation_targets": (
            "preserve falsifiability admission gate",
            "preserve hypothesis-registry requirement",
            "preserve claim-status tracking",
            "preserve correction/update mechanism",
        ),
        "null_controls": (
            "untestable-hypothesis control must be rejected",
            "unversioned-correction control must be rejected",
        ),
    },
    "status_method.fep_scientific_methodology": {
        "context_id": "fep_scientific_methodology",
        "validation_protocol": "paper0.status_method.fep_scientific_methodology",
        "canonical_statement": (
            "The FEP integration maps theory to a generative model, experiments to "
            "sensory evidence, falsification to prediction error, and revision to model update."
        ),
        "source_equation_ids": (
            "P0R00365:meta_framework_integrations",
            "P0R00366:predictive_coding_integration",
            "P0R00367:fep_operating_methodology",
            "P0R00368-P0R00376:scientific_active_inference_cycle",
        ),
        "source_formulae": (
            "theory as generative model",
            "experiments as sensory evidence",
            "negative result as prediction error",
            "model updating",
            "minimise free energy",
        ),
        "test_protocols": ("classify scientific inference steps by FEP role",),
        "null_results": ("FEP methodology mapping is not a measurement result",),
        "variables": ("generative_model", "sensory_evidence", "prediction_error", "model_update"),
        "validation_targets": (
            "map theory to generative model",
            "map experiments to sensory evidence",
            "map falsification to prediction error",
            "map revision to model update",
        ),
        "null_controls": (
            "unknown-inference-step control must be rejected",
            "negative-result-as-failure-only control must be rejected",
        ),
    },
    "status_method.h_int_quality_control": {
        "context_id": "h_int_quality_control",
        "validation_protocol": "paper0.status_method.h_int_quality_control",
        "canonical_statement": (
            "The methodology supplies quality control for investigating "
            "H_int by requiring empirically accessible sigma variables and "
            "empirical handles for analogies."
        ),
        "source_equation_ids": (
            "P0R00377:psi_field_coupling_integration",
            "P0R00378:H_int_quality_control",
            "P0R00379-P0R00380:empirically_accessible_sigma",
            "P0R00381-P0R00382:analogy_empirical_handle",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "empirically accessible sigma",
            "analogy-class usage",
            "empirical handle",
        ),
        "test_protocols": ("normalise H_int and reject untestable sigma/analogy claims",),
        "null_results": ("quality-control rule is not coupling confirmation",),
        "variables": ("lambda_coupling", "Psi_s", "sigma", "empirical_handle"),
        "validation_targets": (
            "preserve all Hamiltonian parameters",
            "require empirically accessible sigma",
            "require empirical handle for analogies",
        ),
        "null_controls": (
            "untestable-sigma control must be rejected",
            "analogy-without-handle control must be rejected",
        ),
    },
    "status_method.productive_disagreement": {
        "context_id": "productive_disagreement",
        "validation_protocol": "paper0.status_method.productive_disagreement",
        "canonical_statement": (
            "Productive disagreement is framed as model replacement/refinement, "
            "including tensor sigma alternatives when scalar sigma is insufficient."
        ),
        "source_equation_ids": (
            "P0R00383:productive_disagreement_header",
            "P0R00384:replacement_model_tensor_sigma",
            "P0R00385:blank_separator",
            "P0R00386:status_method_reprise",
        ),
        "source_formulae": (
            "replacement model",
            "tensor sigma_ij instead of scalar sigma",
            "refining H_int",
            "next boundary P0R00391",
        ),
        "test_protocols": ("preserve replacement-model path and next-boundary accounting",),
        "null_results": ("replacement-model proposal is not validation by itself",),
        "variables": ("sigma", "sigma_ij", "replacement_model", "next_boundary"),
        "validation_targets": (
            "preserve tensor-sigma replacement path",
            "preserve productive-disagreement mechanism",
            "preserve next-boundary accounting",
        ),
        "null_controls": (
            "disagreement-as-refutation-only control must be rejected",
            "missing-next-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class StatusMethodSpec:
    """Status and Method spec promoted from Paper 0 records."""

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
class StatusMethodSpecBundle:
    """Status and Method specs plus source coverage summary."""

    specs: tuple[StatusMethodSpec, ...]
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


def build_status_method_specs(source_records: list[dict[str, Any]]) -> StatusMethodSpecBundle:
    """Build source-covered Status and Method specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[StatusMethodSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            StatusMethodSpec(
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
                domain_review_status="source_methodology_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Status and Method Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "method_commitment_count": 4,
        "next_boundary": "P0R00391",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return StatusMethodSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> StatusMethodSpecBundle:
    """Build Status and Method specs from the canonical ledger."""
    return build_status_method_specs(load_jsonl(path))


def write_outputs(
    bundle: StatusMethodSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Status and Method spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_status_method_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_status_method_validation_specs_report_{date_tag}.md"
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: StatusMethodSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Status and Method Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Blank separators: {bundle.summary['blank_separator_count']}",
        f"- Method commitments: {bundle.summary['method_commitment_count']}",
        f"- Next boundary: {bundle.summary['next_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
        if "H_int = -lambda * Psi_s * sigma" in spec.source_formulae:
            lines.append("  - Source formula: `H_int = -lambda * Psi_s * sigma`")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 Status and Method validation specs."""
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
