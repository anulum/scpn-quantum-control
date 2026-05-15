# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 validation coverage reconciliation tests
"""Tests for Paper 0 promoted validation coverage reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.reconcile_paper0_validation_coverage import (
    REPO_ROOT,
    discover_promoted_slices,
    reconcile_promoted_coverage,
    write_outputs,
)


def test_discover_promoted_slices_cover_opening_and_promoted_tail() -> None:
    slices = discover_promoted_slices(REPO_ROOT)

    assert len(slices) == 52
    assert slices[0].source_start == "P0R00001"
    assert slices[0].source_end == "P0R00017"
    assert slices[1].source_start == "P0R00018"
    assert slices[1].source_end == "P0R00104"
    assert slices[2].source_start == "P0R00105"
    assert slices[2].source_end == "P0R00217"
    assert slices[3].source_start == "P0R00218"
    assert slices[3].source_end == "P0R00248"
    assert slices[4].source_start == "P0R00249"
    assert slices[4].source_end == "P0R00267"
    assert slices[5].source_start == "P0R00268"
    assert slices[5].source_end == "P0R00306"
    assert slices[6].source_start == "P0R00307"
    assert slices[6].source_end == "P0R00332"
    assert slices[7].source_start == "P0R00333"
    assert slices[7].source_end == "P0R00357"
    assert slices[8].source_start == "P0R00358"
    assert slices[8].source_end == "P0R00390"
    assert slices[9].source_start == "P0R00391"
    assert slices[9].source_end == "P0R00400"
    assert slices[10].source_start == "P0R00401"
    assert slices[10].source_end == "P0R00435"
    assert slices[11].source_start == "P0R00436"
    assert slices[11].source_end == "P0R00463"
    assert slices[12].source_start == "P0R00464"
    assert slices[12].source_end == "P0R00505"
    assert slices[13].source_start == "P0R00506"
    assert slices[13].source_end == "P0R00544"
    assert slices[14].source_start == "P0R00545"
    assert slices[14].source_end == "P0R00577"
    assert slices[15].source_start == "P0R00578"
    assert slices[15].source_end == "P0R00609"
    assert slices[-1].source_end == "P0R07129"
    assert sum(item.source_record_count for item in slices) == 1527
    assert all(item.has_runtime_module for item in slices)
    assert all(item.has_runner for item in slices)
    assert all(item.has_builder_tests for item in slices)
    assert all(item.has_runtime_tests for item in slices)
    assert all(item.has_runner_tests for item in slices)


def test_reconcile_promoted_coverage_reports_remaining_middle_gap() -> None:
    result = reconcile_promoted_coverage(REPO_ROOT)

    assert result.summary["ledger_record_count"] == 7129
    assert result.summary["promoted_start"] == "P0R00001"
    assert result.summary["promoted_end"] == "P0R07129"
    assert result.summary["promoted_record_count"] == 1527
    assert result.summary["promoted_coverage_match"] is False
    assert result.summary["promoted_surface_integrity"] is True
    assert result.summary["gap_count"] == 1
    assert result.summary["gaps"] == [["P0R00610", "P0R06211"]]
    assert result.summary["overlap_count"] == 0
    assert result.summary["missing_surface_count"] == 0
    assert result.summary["unpromoted_prefix_count"] == 0
    assert result.summary["unpromoted_prefix_span"] == []


def test_write_reconciliation_outputs(tmp_path: Path) -> None:
    result = reconcile_promoted_coverage(REPO_ROOT)

    outputs = write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["summary"]["promoted_coverage_match"] is False
    assert payload["summary"]["promoted_surface_integrity"] is True
    assert payload["summary"]["missing_surface_count"] == 0
    assert "Paper 0 Validation Coverage Reconciliation" in report
    assert "P0R00001 - P0R07129" in report
