# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-pinned DLA tests
"""Tests for layout-pinned IQM DLA minimal planning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_iqm_layout_pinned_dla_minimal.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_iqm_layout_pinned_dla_minimal", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM layout-pinned runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_layout_rejects_invalid_values() -> None:
    module = _load_module()

    assert module.parse_layout("9,4,3,8") == (9, 4, 3, 8)
    with pytest.raises(ValueError, match="exactly four"):
        module.parse_layout("9,4,3")
    with pytest.raises(ValueError, match="unique"):
        module.parse_layout("9,4,4,8")
    with pytest.raises(ValueError, match="non-negative"):
        module.parse_layout("9,4,3,-1")


def test_build_plan_pairs_even_odd_on_same_layout() -> None:
    module = _load_module()

    rows = module.build_plan(layout=(9, 4, 3, 8), shots=256)

    assert len(rows) == 8
    assert sum(row["kind"] == "dla_parity" for row in rows) == 6
    assert sum(row["kind"] == "readout_baseline" for row in rows) == 2
    assert {tuple(row["requested_initial_layout"]) for row in rows} == {(9, 4, 3, 8)}
    by_depth = {}
    for row in rows:
        if row["kind"] == "dla_parity":
            by_depth.setdefault(row["meta"]["depth"], set()).add(row["meta"]["sector"])
    assert by_depth == {4: {"even", "odd"}, 6: {"even", "odd"}, 10: {"even", "odd"}}


def test_analyse_counts_reports_parity_leakage_and_retention() -> None:
    module = _load_module()

    stats = module.analyse_counts({"1100": 7, "0001": 3}, initial="0011", n_qubits=4)

    assert stats["total_shots"] == 10
    assert stats["expected_parity"] == 0
    assert stats["in_sector_counts"] == 7
    assert stats["leakage_counts"] == 3
    assert stats["parity_leakage"] == pytest.approx(0.3)
    assert stats["initial_state_retention"] == pytest.approx(0.7)


def test_public_copy_removes_raw_job_id_from_timeout_record() -> None:
    module = _load_module()

    clean = module._public_copy(
        {
            "schema": "private",
            "execute": True,
            "platform": "IQM Resonance",
            "requested_initial_layout": [11, 6, 5, 10],
            "records": [
                {
                    "status": "timeout_cancelled",
                    "job_id": "raw-private-id",
                    "job_id_sha256": "hash-only",
                }
            ],
        }
    )

    assert clean["schema"] == "scpn_iqm_dla_layout_pinned_repeat_v1_sanitized"
    assert "job_id" not in clean["records"][0]
    assert clean["records"][0]["job_id_sha256"] == "hash-only"
