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

    assert len(slices) == 368
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
    assert slices[16].source_start == "P0R00610"
    assert slices[16].source_end == "P0R00634"
    assert slices[17].source_start == "P0R00635"
    assert slices[17].source_end == "P0R00669"
    assert slices[18].source_start == "P0R00670"
    assert slices[18].source_end == "P0R00702"
    assert slices[19].source_start == "P0R00703"
    assert slices[19].source_end == "P0R00716"
    assert slices[20].source_start == "P0R00717"
    assert slices[20].source_end == "P0R00732"
    assert slices[21].source_start == "P0R00733"
    assert slices[21].source_end == "P0R00746"
    assert slices[22].source_start == "P0R00747"
    assert slices[22].source_end == "P0R00756"
    assert slices[23].source_start == "P0R00757"
    assert slices[23].source_end == "P0R00760"
    assert slices[24].source_start == "P0R00761"
    assert slices[24].source_end == "P0R00769"
    assert slices[25].source_start == "P0R00770"
    assert slices[25].source_end == "P0R00774"
    assert slices[26].source_start == "P0R00775"
    assert slices[26].source_end == "P0R00781"
    assert slices[27].source_start == "P0R00782"
    assert slices[27].source_end == "P0R00790"
    assert slices[28].source_start == "P0R00791"
    assert slices[28].source_end == "P0R00799"
    assert slices[29].source_start == "P0R00800"
    assert slices[29].source_end == "P0R00810"
    assert slices[30].source_start == "P0R00811"
    assert slices[30].source_end == "P0R00817"
    assert slices[31].source_start == "P0R00818"
    assert slices[31].source_end == "P0R00837"
    assert slices[32].source_start == "P0R00838"
    assert slices[32].source_end == "P0R00904"
    assert slices[33].source_start == "P0R00905"
    assert slices[33].source_end == "P0R00986"
    assert slices[34].source_start == "P0R00987"
    assert slices[34].source_end == "P0R01017"
    assert slices[35].source_start == "P0R01018"
    assert slices[35].source_end == "P0R01077"
    assert slices[36].source_start == "P0R01078"
    assert slices[36].source_end == "P0R01102"
    assert slices[37].source_start == "P0R01103"
    assert slices[37].source_end == "P0R01134"
    assert slices[38].source_start == "P0R01135"
    assert slices[38].source_end == "P0R01188"
    assert slices[39].source_start == "P0R01189"
    assert slices[39].source_end == "P0R01241"
    assert slices[40].source_start == "P0R01242"
    assert slices[40].source_end == "P0R01271"
    assert slices[41].source_start == "P0R01272"
    assert slices[41].source_end == "P0R01332"
    assert slices[42].source_start == "P0R01333"
    assert slices[42].source_end == "P0R01383"
    assert slices[43].source_start == "P0R01384"
    assert slices[43].source_end == "P0R01421"
    assert slices[44].source_start == "P0R01422"
    assert slices[44].source_end == "P0R01509"
    assert slices[45].source_start == "P0R01510"
    assert slices[45].source_end == "P0R01581"
    assert slices[46].source_start == "P0R01582"
    assert slices[46].source_end == "P0R01596"
    assert slices[47].source_start == "P0R01597"
    assert slices[47].source_end == "P0R01622"
    assert slices[48].source_start == "P0R01623"
    assert slices[48].source_end == "P0R01637"
    assert slices[49].source_start == "P0R01638"
    assert slices[49].source_end == "P0R01646"
    assert slices[50].source_start == "P0R01647"
    assert slices[50].source_end == "P0R01654"
    assert slices[51].source_start == "P0R01655"
    assert slices[51].source_end == "P0R01668"
    assert slices[52].source_start == "P0R01669"
    assert slices[52].source_end == "P0R01683"
    assert slices[53].source_start == "P0R01684"
    assert slices[53].source_end == "P0R01692"
    assert slices[54].source_start == "P0R01693"
    assert slices[54].source_end == "P0R01713"
    assert slices[55].source_start == "P0R01714"
    assert slices[55].source_end == "P0R01726"
    assert slices[-1].source_end == "P0R07129"
    assert sum(item.source_record_count for item in slices) == 6311
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
    assert result.summary["promoted_record_count"] == 6311
    assert result.summary["promoted_coverage_match"] is False
    assert result.summary["promoted_surface_integrity"] is True
    assert result.summary["gap_count"] == 1
    assert result.summary["gaps"] == [["P0R05285", "P0R06211"]]
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
