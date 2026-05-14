#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 terminal boundary spec builder
"""Promote Paper 0 terminal taxonomy and EBS records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(7073, 7081))
CLAIM_BOUNDARY = "source-bounded EBS terminal protocol; no unbound empirical claim"
HARDWARE_STATUS = "boundary_protocol_no_device_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "terminal_boundary.section_boundary": {
        "boundary_id": "section",
        "validation_protocol": "paper0.terminal_boundary.section_boundary",
        "canonical_statement": "The section defines Terminal Taxonomy and Enhanced Boundary Set integration.",
        "source_equation_ids": (
            "P0R07073:section_boundary",
            "P0R07078:integration_paragraph_boundary",
        ),
        "source_formulae": ("Terminal Taxonomy and Enhanced Boundary Sets",),
        "test_protocols": ("preserve the section as a boundary-and-terminal protocol",),
        "null_results": ("section existence is not a device-run result",),
        "variables": ("terminal", "EBS", "boundary_object"),
        "validation_targets": (
            "preserve Terminal Taxonomy boundary",
            "preserve EBS integration boundary",
            "reject empirical claims without EBS binding",
        ),
        "null_controls": (
            "missing-EBS-boundary control must be rejected",
            "missing-terminal-boundary control must be rejected",
            "unbound-claim control must be rejected",
        ),
    },
    "terminal_boundary.terminal_taxonomy": {
        "boundary_id": "T1-T7",
        "validation_protocol": "paper0.terminal_boundary.terminal_taxonomy",
        "canonical_statement": "A finite T1-T7 terminal taxonomy is required for all world-facing exchanges.",
        "source_equation_ids": (
            "P0R07075:terminal_taxonomy_t1_t7",
            "P0R07076:TBL020_terminal_taxonomy",
            "P0R07080:terminal_categories",
        ),
        "source_formulae": (
            "T1 bio-measurement",
            "T2 body-side actuation",
            "T3 cognitive/linguistic input",
            "T4 environmental and planetary context",
            "T5 cosmic geometry",
            "T6 noospheric information",
            "T7 simulation control",
        ),
        "test_protocols": (
            "validate every active terminal against T1-T7",
            "reject unknown terminal IDs",
            "require active terminal subset for every run",
        ),
        "null_results": ("unknown or missing terminal configuration invalidates claim binding",),
        "variables": ("T1", "T2", "T3", "T4", "T5", "T6", "T7"),
        "validation_targets": (
            "preserve seven-terminal taxonomy",
            "preserve source-stated terminal categories",
            "reject terminal IDs outside T1-T7",
        ),
        "null_controls": (
            "unknown-terminal control must be rejected",
            "empty-terminal-subset control must be rejected",
            "terminal-category-dropout control must be rejected",
        ),
    },
    "terminal_boundary.ebs_binding": {
        "boundary_id": "EBS",
        "validation_protocol": "paper0.terminal_boundary.ebs_binding",
        "canonical_statement": "Each run binds local geometry, environmental fields, CGP, and operator state into a versioned EBS object.",
        "source_equation_ids": (
            "P0R07074:ebs_boundary_object",
            "P0R07077:ebs_id_hash_binding",
            "P0R07080:reproducible_boundary_conditions",
        ),
        "source_formulae": (
            "EBS = local bio-geometry + environmental fields + CGP + operator state",
            "run -> EBS ID and hash",
        ),
        "test_protocols": (
            "require EBS ID for every run",
            "require deterministic EBS hash",
            "require active terminals configured through EBS",
        ),
        "null_results": ("missing EBS ID or hash prevents reproducible boundary-condition claim",),
        "variables": ("ebs_id", "ebs_hash", "local_bio_geometry", "environmental_fields", "CGP"),
        "validation_targets": (
            "preserve EBS input fields",
            "preserve versioned boundary object requirement",
            "preserve EBS ID/hash traceability",
        ),
        "null_controls": (
            "missing-EBS-ID control must be rejected",
            "missing-EBS-hash control must be rejected",
            "unversioned-boundary control must be rejected",
        ),
    },
    "terminal_boundary.claim_traceability": {
        "boundary_id": "traceability",
        "validation_protocol": "paper0.terminal_boundary.claim_traceability",
        "canonical_statement": "Claims about cosmic dependence, environmental sensitivity, or consciousness modulation must trace to EBS ID and hash.",
        "source_equation_ids": (
            "P0R07077:claim_traceability",
            "P0R07080:empirical_claim_boundary",
        ),
        "source_formulae": (
            "empirical claim -> concrete reproducible boundary conditions",
            "claim -> EBS ID and hash",
        ),
        "test_protocols": (
            "require EBS-bound claim metadata",
            "reject claims lacking active terminal configuration",
            "separate mathematical closure from empirical openness",
        ),
        "null_results": (
            "unbound empirical claim is rejected until tied to EBS and terminal configuration",
        ),
        "variables": ("claim", "boundary_conditions", "active_terminals", "EBS_hash"),
        "validation_targets": (
            "preserve empirical-claim traceability boundary",
            "preserve mathematical-closure and empirical-openness distinction",
            "reject claims not tied to concrete terminal configuration",
        ),
        "null_controls": (
            "unbound-claim control must be rejected",
            "missing-active-terminal control must be rejected",
            "missing-boundary-condition control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TerminalBoundarySpec:
    """Terminal boundary spec promoted from Paper 0 records."""

    key: str
    boundary_id: str
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
class TerminalBoundarySpecBundle:
    """Terminal boundary specs plus source coverage summary."""

    specs: tuple[TerminalBoundarySpec, ...]
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


def build_terminal_boundary_specs(
    source_records: list[dict[str, Any]],
) -> TerminalBoundarySpecBundle:
    """Build source-covered terminal boundary specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[TerminalBoundarySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TerminalBoundarySpec(
                key=key,
                boundary_id=str(metadata["boundary_id"]),
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
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    table_ids = sorted(
        {str(anchor["table_id"]) for anchor in anchors if anchor.get("table_id") is not None}
    )
    summary = {
        "title": "Paper 0 Terminal Boundary Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "terminal_count": 7,
        "table_ids": table_ids,
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return TerminalBoundarySpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: TerminalBoundarySpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Terminal Boundary Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Terminal count: {bundle.summary['terminal_count']}",
        f"- Table IDs: {', '.join(bundle.summary['table_ids'])}",
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
                f"- Boundary: {spec.boundary_id}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: TerminalBoundarySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for terminal boundary specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_terminal_boundary_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_terminal_boundary_validation_specs_report_{date_tag}.md"
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build terminal boundary specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_terminal_boundary_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
