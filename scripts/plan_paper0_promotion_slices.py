#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 promotion planner
"""Plan deterministic source-bounded Paper 0 promotion work orders."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.reconcile_paper0_validation_coverage import REPO_ROOT, reconcile_promoted_coverage

DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_EXTRACTION_DIR / "paper0_promotion_work_orders_2026-05-17.json"
DEFAULT_REPORT_PATH = DEFAULT_EXTRACTION_DIR / "paper0_promotion_work_orders_2026-05-17.md"
MAX_DEFAULT_RECORDS = 64
MIN_HEADER_CLOSURE_RECORDS = 8


@dataclass(frozen=True, slots=True)
class PromotionWorkOrder:
    """One deterministic source-bounded Paper 0 promotion work order."""

    order: int
    source_start: str
    source_end: str
    source_record_count: int
    first_header: str
    next_source_boundary: str | None
    section_path: str
    category_counts: dict[str, int]
    block_type_counts: dict[str, int]
    math_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    table_ids: tuple[str, ...]
    claim_boundary: str
    required_surfaces: tuple[str, ...]
    acceptance_gates: tuple[str, ...]


def load_ledger(path: Path = DEFAULT_LEDGER_PATH) -> tuple[dict[str, Any], ...]:
    """Load the canonical Paper 0 ledger."""
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
    return tuple(records)


def plan_work_orders(
    *,
    ledger_records: tuple[dict[str, Any], ...] | None = None,
    max_records: int = MAX_DEFAULT_RECORDS,
    max_orders: int = 3,
) -> tuple[PromotionWorkOrder, ...]:
    """Plan work orders for the first unpromoted reconciliation gap."""
    if max_records < MIN_HEADER_CLOSURE_RECORDS:
        raise ValueError(f"max_records must be at least {MIN_HEADER_CLOSURE_RECORDS}")
    if max_orders < 1:
        raise ValueError("max_orders must be at least 1")

    records = ledger_records or load_ledger()
    by_id = {str(record["ledger_id"]): record for record in records}
    reconciliation = reconcile_promoted_coverage(REPO_ROOT)
    gaps = reconciliation.summary["gaps"]
    if not gaps:
        return ()

    gap_start, gap_end = gaps[0]
    start_number = _ledger_number(gap_start)
    end_number = _ledger_number(gap_end)
    work_orders: list[PromotionWorkOrder] = []
    cursor = start_number
    order = 1
    while cursor <= end_number and len(work_orders) < max_orders:
        slice_start = cursor
        slice_end = _select_slice_end(by_id, slice_start, end_number, max_records)
        slice_records = tuple(
            by_id[f"P0R{number:05d}"] for number in range(slice_start, slice_end + 1)
        )
        next_boundary_number = slice_end + 1 if slice_end < end_number else None
        work_orders.append(
            _build_work_order(
                order=order,
                records=slice_records,
                next_source_boundary=f"P0R{next_boundary_number:05d}"
                if next_boundary_number
                else None,
            )
        )
        cursor = slice_end + 1
        order += 1
    return tuple(work_orders)


def render_report(work_orders: tuple[PromotionWorkOrder, ...]) -> str:
    """Render a Markdown report for planned work orders."""
    lines = [
        "# Paper 0 Promotion Work Orders",
        "",
        "- Claim boundary: planning only; not scientific validation evidence",
        f"- Work orders: {len(work_orders)}",
        "",
    ]
    for item in work_orders:
        lines.extend(
            [
                f"## Work Order {item.order}",
                "",
                f"- Source span: {item.source_start} - {item.source_end}",
                f"- Source records: {item.source_record_count}",
                f"- First header: {item.first_header}",
                f"- Next source boundary: {item.next_source_boundary}",
                f"- Section path: {item.section_path}",
                f"- Math IDs: {', '.join(item.math_ids) if item.math_ids else 'none'}",
                f"- Image IDs: {', '.join(item.image_ids) if item.image_ids else 'none'}",
                f"- Table IDs: {', '.join(item.table_ids) if item.table_ids else 'none'}",
                "",
                "### Required surfaces",
            ]
        )
        lines.extend(f"- `{surface}`" for surface in item.required_surfaces)
        lines.extend(["", "### Acceptance gates"])
        lines.extend(f"- {gate}" for gate in item.acceptance_gates)
        lines.append("")
    return "\n".join(lines)


def write_outputs(
    work_orders: tuple[PromotionWorkOrder, ...],
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Path]:
    """Write work-order JSON and Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Paper 0 Promotion Work Orders",
        "claim_boundary": "planning only; not scientific validation evidence",
        "work_order_count": len(work_orders),
        "work_orders": [_work_order_payload(item) for item in work_orders],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(work_orders), encoding="utf-8")
    return {"json": output_path, "report": report_path}


def _work_order_payload(item: PromotionWorkOrder) -> dict[str, Any]:
    """Return a stable JSON payload for one work order.

    ``source_record_count`` is the canonical field consumed by the promotion
    automation. ``record_count`` is a read-only compatibility alias for status
    tooling so queue summaries cannot silently report zero remaining records
    when they use the shorter field name.
    """
    payload = asdict(item)
    payload["record_count"] = item.source_record_count
    return payload


def main() -> int:
    """Plan and write Paper 0 promotion work orders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=MAX_DEFAULT_RECORDS)
    parser.add_argument("--max-orders", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    work_orders = plan_work_orders(max_records=args.max_records, max_orders=args.max_orders)
    outputs = write_outputs(work_orders, output_path=args.output, report_path=args.report)
    print(outputs["json"])
    print(outputs["report"])
    return 0


def _build_work_order(
    *,
    order: int,
    records: tuple[dict[str, Any], ...],
    next_source_boundary: str | None,
) -> PromotionWorkOrder:
    first = records[0]
    last = records[-1]
    first_header = next(
        (
            str(record.get("text", record.get("canonical_text", "")))
            for record in records
            if record.get("block_type") == "Header"
        ),
        str(first.get("section_path", "unknown section")),
    )
    categories = _counts(record.get("canonical_category", "unknown") for record in records)
    block_types = _counts(record.get("block_type", "unknown") for record in records)
    slug = _surface_slug(_slug(first_header), str(first["ledger_id"]), str(last["ledger_id"]))
    required_surfaces = (
        f"scripts/build_paper0_{slug}_specs.py",
        f"src/scpn_quantum_control/paper0/{slug}_validation.py",
        f"scripts/run_paper0_{slug}_fixture.py",
        f"tests/test_build_paper0_{slug}_specs.py",
        f"tests/test_paper0_{slug}_validation.py",
        f"tests/test_run_paper0_{slug}_fixture.py",
    )
    return PromotionWorkOrder(
        order=order,
        source_start=str(first["ledger_id"]),
        source_end=str(last["ledger_id"]),
        source_record_count=len(records),
        first_header=first_header,
        next_source_boundary=next_source_boundary,
        section_path=str(first.get("section_path", "")),
        category_counts=categories,
        block_type_counts=block_types,
        math_ids=tuple(
            sorted({item for record in records for item in record.get("math_ids", [])})
        ),
        image_ids=tuple(
            sorted({item for record in records for item in record.get("image_ids", [])})
        ),
        table_ids=tuple(
            sorted(
                {
                    str(record["table_id"])
                    for record in records
                    if record.get("table_id") is not None
                }
            )
        ),
        claim_boundary="work order only; source-bounded promotion required; not validation evidence",
        required_surfaces=required_surfaces,
        acceptance_gates=(
            "builder summary source span equals work-order span",
            "builder consumed_source_record_count equals source_record_count",
            "builder coverage_match is true and unconsumed_source_ledger_ids is empty",
            "all specs preserve the same claim_boundary and hardware_status",
            "fixture source span, source_record_count, component_count, and next_source_boundary match builder summary",
            "reconciliation reports missing_surface_count 0 and overlap_count 0 after integration",
            "public files contain no internal agent/vendor names",
        ),
    )


def _identifier_safe_slug(slug: str) -> str:
    if not slug or not slug[0].isalpha():
        return f"section_{slug}"
    return slug


def _surface_slug(base_slug: str, source_start: str, source_end: str) -> str:
    base_slug = _identifier_safe_slug(base_slug)
    builder = REPO_ROOT / f"scripts/build_paper0_{base_slug}_specs.py"
    runtime = REPO_ROOT / f"src/scpn_quantum_control/paper0/{base_slug}_validation.py"
    existing = [path for path in (builder, runtime) if path.exists()]
    if not existing:
        return base_slug
    for path in existing:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{base_slug}_{source_start.lower()}"
        if source_start in text and source_end in text:
            return base_slug
    return f"{base_slug}_{source_start.lower()}"


def _select_slice_end(
    by_id: dict[str, dict[str, Any]],
    start_number: int,
    gap_end_number: int,
    max_records: int,
) -> int:
    hard_end = min(gap_end_number, start_number + max_records - 1)
    for number in range(start_number + MIN_HEADER_CLOSURE_RECORDS, hard_end + 1):
        record = by_id[f"P0R{number:05d}"]
        if record.get("block_type") == "Header":
            return number - 1
    return hard_end


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _ledger_number(ledger_id: str) -> int:
    return int(ledger_id.removeprefix("P0R"))


def _ascii_safe_slug_text(text: str) -> str:
    replacements = {
        "Ψ": "Psi",
        "ψ": "psi",
        "θ": "theta",
        "λ": "lambda",
        "σ": "sigma",
        "μ": "mu",
        "ρ": "rho",
        "τ": "tau",
        "Ω": "Omega",
        "Δ": "Delta",
        "Σ": "Sigma",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "→": "->",
        "×": "x",
    }
    translated = text
    for original, replacement in replacements.items():
        translated = translated.replace(original, replacement)
    return translated.encode("ascii", "ignore").decode("ascii")


def _slug(text: str) -> str:
    lowered = text.lower().replace("psi", "psi")
    chars = [char if char.isalnum() else "_" for char in lowered]
    collapsed = "_".join(part for part in "".join(chars).split("_") if part)
    return collapsed[:72].strip("_") or "paper0_slice"


if __name__ == "__main__":
    raise SystemExit(main())
