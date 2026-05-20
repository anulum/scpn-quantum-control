#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 0 canonical review ledger
"""Create a complete canonical-review ledger from the Paper 0 block register."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.equation_register import (
    iter_paper0_equation_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_REGISTER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_exhaustive_register_2026-05-13.jsonl"
VALID_CATEGORIES = {
    "claim",
    "context",
    "equation",
    "figure",
    "mechanism",
    "structural",
    "table",
    "validation_target",
}
DOMAIN_REVIEW_CATEGORIES = {
    "claim",
    "equation",
    "mechanism",
    "validation_target",
}


@dataclass(frozen=True, slots=True)
class CanonicalLedger:
    """Complete review ledger and summary."""

    entries: list[dict[str, Any]]
    summary: dict[str, Any]


def _paper0_equation_key_map() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for record in iter_paper0_equation_records():
        for equation_id in record.source_equation_ids:
            mapping.setdefault(equation_id, []).append(record.key)
    return {key: sorted(values) for key, values in mapping.items()}


def _category_for_record(record: dict[str, Any]) -> str:
    tags = set(record.get("semantic_tags", []))
    block_type = str(record.get("block_type", ""))
    if record.get("math_ids"):
        return "equation"
    if "validation" in tags:
        return "validation_target"
    if record.get("table_id") or block_type == "Table":
        return "table"
    if record.get("image_ids"):
        return "figure"
    if record.get("is_mechanism_candidate") or "mechanism" in tags:
        return "mechanism"
    if record.get("is_claim_candidate") or "claim_language" in tags:
        return "claim"
    if not record.get("has_text"):
        return "structural"
    return "context"


def _next_action(category: str) -> str:
    return {
        "claim": "canonicalise_claim_scope_and_falsification_boundary",
        "context": "retain_as_source_context",
        "equation": "canonicalise_latex_variables_and_units",
        "figure": "link_caption_media_and_evidence_role",
        "mechanism": "map_mechanism_inputs_outputs_and_controls",
        "structural": "retain_or_reject_as_document_structure",
        "table": "extract_table_schema_and_parameter_evidence",
        "validation_target": "map_to_executable_validation_protocol",
    }[category]


def _promotion_state(category: str) -> str:
    if category in DOMAIN_REVIEW_CATEGORIES:
        return "requires_domain_review"
    return "accepted_as_context_or_structure"


def build_review_ledger(records: list[dict[str, Any]]) -> CanonicalLedger:
    """Assign controlled canonical-review categories to every extracted record."""
    equation_key_map = _paper0_equation_key_map()
    entries: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        category = _category_for_record(record)
        if category not in VALID_CATEGORIES:
            raise ValueError(f"unsupported category {category!r}")
        equation_keys: list[str] = []
        for equation_id in record.get("math_ids", []):
            equation_keys.extend(equation_key_map.get(equation_id, []))
        entries.append(
            {
                "ledger_id": f"P0R{ordinal:05d}",
                "source_record_id": record["record_id"],
                "source_block_index": int(record["block_index"]),
                "block_type": record["block_type"],
                "section_path": record.get("section_path", ""),
                "canonical_category": category,
                "promotion_state": _promotion_state(category),
                "next_action": _next_action(category),
                "paper0_equation_record_keys": sorted(set(equation_keys)),
                "math_ids": list(record.get("math_ids", [])),
                "image_ids": list(record.get("image_ids", [])),
                "table_id": record.get("table_id"),
                "semantic_tags": list(record.get("semantic_tags", [])),
                "review_status": "category_assigned",
                "domain_review_status": (
                    "pending" if category in DOMAIN_REVIEW_CATEGORIES else "not_required"
                ),
                "text": record.get("text", ""),
            }
        )

    category_counts = dict(
        sorted(Counter(entry["canonical_category"] for entry in entries).items())
    )
    promotion_counts = dict(sorted(Counter(entry["promotion_state"] for entry in entries).items()))
    domain_review_count = sum(1 for entry in entries if entry["domain_review_status"] == "pending")
    summary = {
        "source_record_count": len(records),
        "ledger_record_count": len(entries),
        "coverage_match": len(records) == len(entries),
        "category_counts": category_counts,
        "promotion_state_counts": promotion_counts,
        "requires_domain_review_count": domain_review_count,
        "context_or_structure_count": len(entries) - domain_review_count,
        "upde_equation_anchor_count": sum(
            1 for entry in entries if entry["paper0_equation_record_keys"]
        ),
        "all_entries_have_next_action": all(bool(entry["next_action"]) for entry in entries),
        "canonical_review_policy": (
            "Every source record is categorised. Scientific promotion remains pending "
            "for claim, mechanism, equation, and validation-target categories until "
            "domain review records variables, assumptions, controls, and falsifiers."
        ),
    }
    return CanonicalLedger(entries=entries, summary=summary)


def build_review_report(result: CanonicalLedger) -> str:
    """Render a concise canonical-review coverage report."""
    status = "match" if result.summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 Canonical Review Ledger",
        "",
        f"- Source records: `{result.summary['source_record_count']}`",
        f"- Ledger records: `{result.summary['ledger_record_count']}`",
        f"- Coverage status: `{status}`",
        f"- Requires domain review: `{result.summary['requires_domain_review_count']}`",
        f"- Context or structure accepted: `{result.summary['context_or_structure_count']}`",
        f"- UPDE equation anchors linked: `{result.summary['upde_equation_anchor_count']}`",
        f"- All entries have next action: `{result.summary['all_entries_have_next_action']}`",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in result.summary["category_counts"].items():
        lines.append(f"- {category}: `{count}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            result.summary["canonical_review_policy"],
            "",
        ]
    )
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""
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


def write_outputs(
    result: CanonicalLedger,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write review ledger, summary, and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / f"paper0_canonical_review_ledger_{date_tag}.jsonl"
    summary_path = output_dir / f"paper0_canonical_review_summary_{date_tag}.json"
    report_path = output_dir / f"paper0_canonical_review_report_{date_tag}.md"
    domain_queue_path = output_dir / f"paper0_domain_review_queue_{date_tag}.jsonl"
    upde_queue_path = output_dir / f"paper0_upde_anchor_review_queue_{date_tag}.jsonl"
    domain_queue = [
        entry for entry in result.entries if entry["domain_review_status"] == "pending"
    ]
    upde_queue = [entry for entry in result.entries if entry["paper0_equation_record_keys"]]
    ledger_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in result.entries) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_review_report(result), encoding="utf-8")
    domain_queue_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in domain_queue) + "\n",
        encoding="utf-8",
    )
    upde_queue_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in upde_queue) + "\n",
        encoding="utf-8",
    )
    return {
        "ledger": ledger_path,
        "summary": summary_path,
        "report": report_path,
        "domain_review_queue": domain_queue_path,
        "upde_anchor_review_queue": upde_queue_path,
    }


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    records = load_jsonl(args.register)
    result = build_review_ledger(records)
    paths = write_outputs(result, output_dir=args.output_dir, date_tag=args.date_tag)
    for key, path in paths.items():
        print(f"wrote_{key}={path}")
    print(f"ledger_record_count={result.summary['ledger_record_count']}")
    print(f"coverage_match={result.summary['coverage_match']}")
    print(f"requires_domain_review_count={result.summary['requires_domain_review_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
