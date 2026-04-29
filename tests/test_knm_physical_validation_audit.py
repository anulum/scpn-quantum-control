# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for K_nm Physical Validation Audit Runner
"""Tests for K_nm physical validation audit helpers."""

from __future__ import annotations

import json

import numpy as np

from scripts.run_knm_physical_validation_audit import (
    compare_measured_couplings,
    load_measured_couplings,
)


def test_compare_measured_couplings_marks_missing_dataset_open():
    result = compare_measured_couplings(np.ones((2, 2)), None)

    assert result["available"] is False
    assert result["status"] == "missing_measured_system_dataset"


def test_compare_measured_couplings_validates_with_uncertainty():
    K = np.array([[0.0, 0.302], [0.302, 0.0]])
    measured = {
        "system": "unit-test",
        "unit": "dimensionless",
        "normalisation": "already normalised",
        "couplings": [{"i": 1, "j": 2, "value": 0.301, "uncertainty": 0.002}],
    }

    result = compare_measured_couplings(K, measured)

    assert result["available"] is True
    assert result["status"] == "validated_with_measured_dataset"
    assert result["matched_edges"] == 1


def test_load_measured_couplings_requires_couplings_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"system": "bad"}), encoding="utf-8")

    try:
        load_measured_couplings(path)
    except ValueError as exc:
        assert "couplings list" in str(exc)
    else:
        raise AssertionError("Expected invalid measured coupling schema to fail")
