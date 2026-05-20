#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 opening foundation spec builder
"""Promote Paper 0 opening foundation and global-boundary axiom records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1, 18))
CLAIM_BOUNDARY = "source-bounded opening foundation; not empirical validation evidence"
HARDWARE_STATUS = "source_foundation_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "opening_foundation.book_identity": {
        "foundation_id": "book_identity",
        "validation_protocol": "paper0.opening_foundation.book_identity",
        "canonical_statement": "The opening identifies Paper 0 as the Book II foundation for SCPN architecture.",
        "source_equation_ids": (
            "P0R00001:book_identity",
            "P0R00004:scp_network_definition",
            "P0R00008:foundation_stone_note",
        ),
        "source_formulae": ("SCPN, Book II -- An Architecture for Reality",),
        "test_protocols": ("preserve title, abstract role, and foundation-boundary status",),
        "null_results": ("identity text is not a validation result",),
        "variables": ("SCPN", "Book II", "Paper 0"),
        "validation_targets": (
            "preserve Book II identity",
            "preserve Paper 0 foundation role",
            "reject title text as empirical evidence",
        ),
        "null_controls": (
            "identity-as-evidence control must be rejected",
            "missing-foundation-role control must be rejected",
            "wrong-book-layer control must be rejected",
        ),
    },
    "opening_foundation.quasicritical_ms_qec": {
        "foundation_id": "quasicritical_ms_qec",
        "validation_protocol": "paper0.opening_foundation.quasicritical_ms_qec",
        "canonical_statement": "The abstract frames SCPN as a self-organising quasicritical system stabilised by SOC and protected by MS-QEC.",
        "source_equation_ids": (
            "P0R00005:quasicritical_soc_ms_qec",
            "P0R00006:multi_scale_bridge",
        ),
        "source_formulae": (
            "self-organising quasicritical system",
            "Self-Organised Criticality and Multi-Scale Quantum Error Correction",
        ),
        "test_protocols": (
            "preserve quasicritical, SOC, and MS-QEC as hypotheses requiring later validation",
        ),
        "null_results": ("abstract framing alone does not establish SOC or MS-QEC in data",),
        "variables": ("SOC", "MS_QEC", "quasicriticality"),
        "validation_targets": (
            "preserve quasicritical system framing",
            "preserve SOC stabilisation claim boundary",
            "preserve MS-QEC protection claim boundary",
        ),
        "null_controls": (
            "abstract-as-SOC-proof control must be rejected",
            "missing-MS-QEC-boundary control must be rejected",
            "unwired-layer-bridge control must be rejected",
        ),
    },
    "opening_foundation.recursive_optimisation": {
        "foundation_id": "recursive_optimisation",
        "validation_protocol": "paper0.opening_foundation.recursive_optimisation",
        "canonical_statement": "Cybernetic closure is framed through H_rec at Meta-Layer 16 stabilising the L15 teleological objective.",
        "source_equation_ids": (
            "P0R00007:H_rec_cybernetic_closure",
            "P0R00008:falsifiable_observables_boundary",
        ),
        "source_formulae": ("H_rec stabilises L15 teleological objective",),
        "test_protocols": (
            "preserve recursive-optimisation framing as a later validation target",
        ),
        "null_results": ("H_rec mention is not a measured closure result",),
        "variables": ("H_rec", "L15", "Meta_Layer_16"),
        "validation_targets": (
            "preserve cybernetic closure claim boundary",
            "preserve L15/Meta-Layer 16 relation",
            "preserve falsifiable-observable requirement",
        ),
        "null_controls": (
            "H-rec-as-executed-result control must be rejected",
            "missing-L15-link control must be rejected",
            "missing-falsifier control must be rejected",
        ),
    },
    "opening_foundation.ebs_terminal_anchor": {
        "foundation_id": "ebs_terminal_anchor",
        "validation_protocol": "paper0.opening_foundation.ebs_terminal_anchor",
        "canonical_statement": "Every application, simulation, or device run is anchored in EBS and exchanges through T1-T7 terminals.",
        "source_equation_ids": (
            "P0R00011:ebs_anchor",
            "P0R00013:C0_boundary_set",
            "P0R00014:F0_terminal_set",
        ),
        "source_formulae": (
            "C_0 = {G_local, F_env, G_cosmic, O_state}",
            "F_0 = {T1, T2, T3, T4, T5, T6, T7}",
        ),
        "test_protocols": ("validate boundary set and terminal set membership",),
        "null_results": ("missing EBS or terminal binding invalidates empirical claim tracking",),
        "variables": ("C_0", "F_0", "EBS", "T1_T7"),
        "validation_targets": (
            "preserve EBS anchoring requirement",
            "preserve C0 boundary members",
            "preserve F0 terminal members",
        ),
        "null_controls": (
            "missing-C0-member control must be rejected",
            "unknown-terminal control must be rejected",
            "free-boundary control must be rejected",
        ),
    },
    "opening_foundation.global_boundary_axiom": {
        "foundation_id": "global_axiom",
        "validation_protocol": "paper0.opening_foundation.global_boundary_axiom",
        "canonical_statement": "The global axiom allows no free, untracked boundary conditions.",
        "source_equation_ids": (
            "P0R00011:ebs_anchor",
            "P0R00013:C0_boundary_set",
            "P0R00014:F0_terminal_set",
            "P0R00016:beta0_boundary_assertion",
            "P0R00017:no_free_boundary_conditions",
        ),
        "source_formulae": (
            "beta_0(E,A): all layers get boundaries from E in B and interactions occur via A subset T",
            "no free, untracked boundary conditions",
        ),
        "test_protocols": (
            "reject unknown boundary members",
            "reject unknown terminal IDs",
            "reject empty active terminal subsets",
        ),
        "null_results": ("untracked boundary conditions fail the global axiom",),
        "variables": ("beta_0", "E", "A", "B", "T"),
        "validation_targets": (
            "preserve beta0 membership assertion",
            "preserve no-free-boundary axiom",
            "preserve active-terminal subset requirement",
        ),
        "null_controls": (
            "free-boundary control must be rejected",
            "unknown-terminal control must be rejected",
            "empty-terminal-subset control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class OpeningFoundationSpec:
    """Opening foundation spec promoted from Paper 0 records."""

    key: str
    foundation_id: str
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
class OpeningFoundationSpecBundle:
    """Opening foundation specs plus source coverage summary."""

    specs: tuple[OpeningFoundationSpec, ...]
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


def build_opening_foundation_specs(
    source_records: list[dict[str, Any]],
) -> OpeningFoundationSpecBundle:
    """Build source-covered opening foundation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[OpeningFoundationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            OpeningFoundationSpec(
                key=key,
                foundation_id=str(metadata["foundation_id"]),
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

    summary = {
        "title": "Paper 0 Opening Foundation Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "boundary_set_size": 4,
        "terminal_count": 7,
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return OpeningFoundationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: OpeningFoundationSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Opening Foundation Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Boundary set size: {bundle.summary['boundary_set_size']}",
        f"- Terminal count: {bundle.summary['terminal_count']}",
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
                f"- Foundation: {spec.foundation_id}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: OpeningFoundationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for opening foundation specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_opening_foundation_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_opening_foundation_validation_specs_report_{date_tag}.md"
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build opening foundation specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_opening_foundation_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
