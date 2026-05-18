# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 promotion planner tests
"""Tests for deterministic Paper 0 promotion work-order planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_paper0_promotion_slices import plan_work_orders, render_report, write_outputs


def test_plan_work_orders_starts_at_current_reconciliation_gap() -> None:
    work_orders = plan_work_orders(max_orders=2)

    assert len(work_orders) == 2
    assert work_orders[0].source_start == "P0R04572"
    assert work_orders[0].source_record_count >= 8
    assert work_orders[0].source_record_count <= 64
    assert work_orders[0].source_end < work_orders[1].source_start
    assert (
        work_orders[0].claim_boundary
        == "work order only; source-bounded promotion required; not validation evidence"
    )
    assert "scripts/build_paper0_" in work_orders[0].required_surfaces[0]
    assert "builder coverage_match is true" in work_orders[0].acceptance_gates[2]


def test_plan_work_orders_rejects_unsafe_batch_parameters() -> None:
    with pytest.raises(ValueError, match="max_records must be at least 8"):
        plan_work_orders(max_records=7)
    with pytest.raises(ValueError, match="max_orders must be at least 1"):
        plan_work_orders(max_orders=0)


def test_write_work_order_outputs(tmp_path: Path) -> None:
    work_orders = plan_work_orders(max_records=64, max_orders=1)
    outputs = write_outputs(
        work_orders,
        output_path=tmp_path / "orders.json",
        report_path=tmp_path / "orders.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["claim_boundary"] == "planning only; not scientific validation evidence"
    assert payload["work_order_count"] == 1
    assert payload["work_orders"][0]["source_start"] == "P0R04572"
    assert (
        payload["work_orders"][0]["record_count"]
        == payload["work_orders"][0]["source_record_count"]
    )
    assert payload["work_orders"][0]["record_count"] > 0
    assert "Paper 0 Promotion Work Orders" in report
    assert "Required surfaces" in render_report(work_orders)
