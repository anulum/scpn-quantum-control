# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 ansatz observable validation
"""Tests for S3 no-QPU ansatz observable validation."""

from __future__ import annotations

from scripts.validate_s3_ansatz_observables import _aggregate, _rows


def test_rows_validate_promoted_ansatz_candidates_without_hardware() -> None:
    rows = _rows([3], top_k=2)

    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["hardware_submission"] is False for row in rows)
    assert all(float(row["energy_absolute_error"]) >= 0.0 for row in rows)


def test_aggregate_reports_best_candidate() -> None:
    aggregate = _aggregate(_rows([3], top_k=2))

    assert aggregate["row_count"] == 2
    assert aggregate["best_energy_absolute_error"] >= 0.0
    assert aggregate["best_candidate_by_energy"]
