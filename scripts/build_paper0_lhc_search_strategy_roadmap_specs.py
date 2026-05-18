#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 LHC search-strategy roadmap spec builder
"""Promote Paper 0 LHC search-strategy roadmap records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1684, 1693))
CLAIM_BOUNDARY = "source-bounded LHC search-strategy roadmap bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "lhc_search_strategy_roadmap.search_signature_overview": {
        "context_id": "search_signature_overview",
        "validation_protocol": "paper0.lhc_search_strategy_roadmap.search_signature_overview",
        "canonical_statement": (
            "The source summarises LHC search signatures for a Higgs-mixed Psi-Higgs scalar as a roadmap for constraints or discovery."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:search_signature_overview" for n in range(1684, 1688)
        ),
        "source_formulae": (
            "Phenomenology and Search Strategies at the LHC",
            "ATLAS and CMS searches provide a template for constraining or discovering h_Psi",
            "exotic Higgs decays include h_SM -> h_Psi h_Psi when m_hPsi < 125 GeV",
            "resonant production includes pp -> h_Psi -> ZZ and pp -> h_Psi -> WW",
            "cascade decays include X -> h_SM h_Psi",
            "invisible decays can arise if h_Psi couples to dark-sector or infoton channels",
        ),
        "test_protocols": ("preserve LHC search-signature overview boundary",),
        "null_results": ("search-signature list is not an observed LHC excess",),
        "variables": ("h_SM", "h_Psi", "m_hPsi", "ZZ", "WW", "X", "infoton"),
        "validation_targets": (
            "preserve four LHC signature classes",
            "preserve constraint-or-discovery framing",
        ),
        "null_controls": ("roadmap channels must not imply detected Psi-Higgs events",),
    },
    "lhc_search_strategy_roadmap.table_roadmap": {
        "context_id": "table_roadmap",
        "validation_protocol": "paper0.lhc_search_strategy_roadmap.table_roadmap",
        "canonical_statement": (
            "The source anchors proposed Psi-Higgs search parameters in Table 2 as a concrete experimental roadmap."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:table_roadmap" for n in range(1688, 1690)),
        "source_formulae": (
            "Table 2: Proposed Experimental Search Parameters for the Psi-Higgs Boson",
            "TBL003 is the source table for the proposed experimental search parameters",
            "the table translates theoretical possibilities into a concrete experimental roadmap",
            "roadmap grounding is source accounting, not validation evidence",
        ),
        "test_protocols": ("preserve Table 2 roadmap source boundary",),
        "null_results": ("source table is not an executed experimental result",),
        "variables": ("TBL003", "search_parameters", "experimental_roadmap"),
        "validation_targets": ("preserve table identifier", "preserve roadmap role"),
        "null_controls": ("table roadmap must not be promoted to measured search outcome",),
    },
    "lhc_search_strategy_roadmap.ssb_cascade_transition": {
        "context_id": "ssb_cascade_transition",
        "validation_protocol": "paper0.lhc_search_strategy_roadmap.ssb_cascade_transition",
        "canonical_statement": (
            "The source closes the LHC search roadmap and transitions to the SSB cascade section on mass and solitonic self."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:ssb_cascade_transition" for n in range(1690, 1693)
        ),
        "source_formulae": (
            "structural separator after the LHC search-parameter table",
            "2.4 The SSB Cascade: Origin of Mass & The Solitonic Self",
            "next source section begins at the genesis of the hierarchy cascade",
        ),
        "test_protocols": ("preserve SSB-cascade transition boundary",),
        "null_results": ("section transition carries no empirical validation claim",),
        "variables": ("SSB_cascade", "origin_of_mass", "solitonic_self"),
        "validation_targets": ("preserve section transition", "preserve next-boundary context"),
        "null_controls": ("blank structural records must remain structural source accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class LHCSearchStrategyRoadmapSpec:
    """LHC search-strategy roadmap spec promoted from Paper 0 records."""

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
class LHCSearchStrategyRoadmapSpecBundle:
    """LHC search-strategy roadmap specs plus source coverage summary."""

    specs: tuple[LHCSearchStrategyRoadmapSpec, ...]
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


def build_lhc_search_strategy_roadmap_specs(
    source_records: list[dict[str, Any]],
) -> LHCSearchStrategyRoadmapSpecBundle:
    """Build source-covered LHC search-strategy roadmap specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[LHCSearchStrategyRoadmapSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            LHCSearchStrategyRoadmapSpec(
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
        "title": "Paper 0 LHC Search Strategy Roadmap Specs",
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
        "next_source_boundary": "P0R01693",
    }
    return LHCSearchStrategyRoadmapSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> LHCSearchStrategyRoadmapSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_lhc_search_strategy_roadmap_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: LHCSearchStrategyRoadmapSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 LHC Search Strategy Roadmap Specs",
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
    bundle: LHCSearchStrategyRoadmapSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_lhc_search_strategy_roadmap_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_lhc_search_strategy_roadmap_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
