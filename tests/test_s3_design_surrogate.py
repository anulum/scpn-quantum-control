# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 design surrogate rehearsal
"""Tests for the S3 no-QPU surrogate rehearsal harness."""

from __future__ import annotations

from scpn_quantum_control.benchmarks.s3_design_protocol import generate_s3_candidate_grid
from scripts.train_s3_design_surrogate import _rows_for_sizes, _train_holdout


def test_generate_s3_candidate_grid_contains_both_families() -> None:
    candidates = generate_s3_candidate_grid()

    assert {candidate.family for candidate in candidates} == {"ansatz", "pulse"}
    assert len(candidates) >= 20


def test_train_holdout_reports_finite_metrics() -> None:
    rows = _rows_for_sizes([3, 4])
    summary = _train_holdout(rows)

    assert summary["train_rows"] > summary["holdout_rows"] > 0
    assert summary["holdout_metrics"]["rmse"] >= 0.0
    assert set(summary["family_metrics"]) == {"ansatz", "pulse"}
