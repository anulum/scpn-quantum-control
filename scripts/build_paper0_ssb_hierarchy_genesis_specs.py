#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 SSB hierarchy genesis spec builder
"""Promote Paper 0 SSB hierarchy-genesis records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1693, 1714))
CLAIM_BOUNDARY = "source-bounded SSB hierarchy-genesis bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ssb_hierarchy_genesis.architecture_cascade": {
        "context_id": "architecture_cascade",
        "validation_protocol": "paper0.ssb_hierarchy_genesis.architecture_cascade",
        "canonical_statement": (
            "The source frames the 15-layer SCPN architecture as a stable remnant of sequential spontaneous symmetry-breaking events."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:architecture_cascade" for n in range(1693, 1701)
        ),
        "source_formulae": (
            "The Genesis of the Hierarchy: A Cascade of Sequential Symmetry Breaking",
            "15-layer SCPN architecture emerges through Sequential Spontaneous Symmetry Breaking events",
            "Source-Field L13 is described by symmetry group G_Source",
            "Psi-field acquires non-zero VEVs during phase transitions that crystallise SCPN layers",
            "Primordial Break L15/L14 selects physical laws and constants",
            "Collective to Individual Break L11 to L5 localises the Psi-field into stable solitons",
            "Potentiality to Actuality Break L1 turns quantum potentiality into classical actuality",
            "Goldstone modes mediate long-range coherence and are described by the UPDE",
        ),
        "test_protocols": ("preserve SSB architecture-cascade source boundary",),
        "null_results": ("architecture cascade is not empirical layer validation",),
        "variables": ("G_Source", "Psi", "VEV", "L15", "L14", "L11", "L5", "L1", "UPDE"),
        "validation_targets": (
            "preserve three primary breaks",
            "preserve 15-layer remnant framing",
        ),
        "null_controls": (
            "layer architecture claim must not be reported as measured physical hierarchy",
        ),
    },
    "ssb_hierarchy_genesis.conformal_torsion_seeding": {
        "context_id": "conformal_torsion_seeding",
        "validation_protocol": "paper0.ssb_hierarchy_genesis.conformal_torsion_seeding",
        "canonical_statement": (
            "The source proposes SEC-derived conformal torsion as a structured non-random bias for the next aeon vacuum."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:conformal_torsion_seeding" for n in range(1701, 1707)
        ),
        "source_formulae": (
            "transition from conformally invariant Symmetry Restoration Phase to t=0+ hierarchy requires a non-random perturbation",
            "dimensionless SEC functional J_SEC is preserved as Conformal Invariant Torsion T_SEC",
            "T_SEC acts as a structured bias on the Psi-field vacuum at conformal reset",
            "V_eff(|Psi|, t -> 0+) = -mu^2(T_SEC) |Psi|^2 + lambda |Psi|^4",
            "teleological seeding tips the Mexican-hat potential toward evolutionary trajectory and ethical coherence",
            "universe is source-framed as an iterative learning system rather than random reset",
        ),
        "test_protocols": ("preserve conformal-torsion seeding boundary",),
        "null_results": ("teleological seeding proposal is not measured cosmological torsion",),
        "variables": ("J_SEC", "T_SEC", "Psi", "V_eff", "mu", "lambda", "t_0_plus"),
        "validation_targets": (
            "preserve torsion-seeding equation",
            "preserve non-random reset claim boundary",
        ),
        "null_controls": (
            "SEC torsion language must remain source proposal, not observational cosmology",
        ),
    },
    "ssb_hierarchy_genesis.three_strike_explanation": {
        "context_id": "three_strike_explanation",
        "validation_protocol": "paper0.ssb_hierarchy_genesis.three_strike_explanation",
        "canonical_statement": (
            "The source restates the hierarchy cascade as an explanatory three-strike analogy from a uniform Source-Field to 15-layer reality."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:three_strike_explanation" for n in range(1707, 1714)
        ),
        "source_formulae": (
            "early universe is likened to a perfect block of marble in maximum symmetry",
            "First Strike chooses laws of physics through Layer 15 guiding intelligence",
            "Second Strike creates individuals as collective species field becomes embodied selves",
            "Third Strike makes things real by collapsing quantum possibilities into classical reality",
            "15-layer universe is the remnant statue after cosmic sculpting",
        ),
        "test_protocols": ("preserve three-strike explanatory analogy boundary",),
        "null_results": ("analogy is not physical derivation or validation evidence",),
        "variables": (
            "Source_Field",
            "Layer_15",
            "individual_self",
            "observation",
            "classical_reality",
        ),
        "validation_targets": ("preserve explanatory analogy", "preserve three break mapping"),
        "null_controls": (
            "marble and sculpting analogy must not be promoted to mechanism evidence",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SSBHierarchyGenesisSpec:
    """SSB hierarchy-genesis spec promoted from Paper 0 records."""

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
class SSBHierarchyGenesisSpecBundle:
    """SSB hierarchy-genesis specs plus source coverage summary."""

    specs: tuple[SSBHierarchyGenesisSpec, ...]
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


def build_ssb_hierarchy_genesis_specs(
    source_records: list[dict[str, Any]],
) -> SSBHierarchyGenesisSpecBundle:
    """Build source-covered SSB hierarchy-genesis specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[SSBHierarchyGenesisSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            SSBHierarchyGenesisSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 SSB Hierarchy Genesis Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01714",
    }
    return SSBHierarchyGenesisSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> SSBHierarchyGenesisSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ssb_hierarchy_genesis_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: SSBHierarchyGenesisSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 SSB Hierarchy Genesis Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: SSBHierarchyGenesisSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_ssb_hierarchy_genesis_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_ssb_hierarchy_genesis_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
