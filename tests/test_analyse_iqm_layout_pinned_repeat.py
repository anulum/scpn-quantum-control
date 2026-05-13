# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-pinned repeat analysis tests
"""Tests for reusable IQM layout-pinned repeat analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyse_iqm_layout_pinned_repeat.py"
Q13_INPUT = (
    REPO_ROOT
    / "data"
    / "iqm_paper_replication"
    / "iqm_dla_layout_pinned_repeat_q13-8-9-14_2026-05-13_executed.json"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_iqm_layout_pinned_repeat", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM layout-pinned analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q13_layout_repeat_analysis_matches_recorded_counts() -> None:
    module = _load_module()

    summary = module.analyse_layout_repeat(Q13_INPUT)

    assert summary["requested_initial_layout"] == [13, 8, 9, 14]
    assert summary["total_circuits"] == 8
    assert summary["total_shots"] == 2048
    assert summary["n_signs_matching_ibm_phase2"] == 1
    by_depth = {row["depth"]: row for row in summary["depth_summaries"]}
    assert by_depth[4]["iqm_leakage_even"] == 106 / 256
    assert by_depth[4]["iqm_leakage_odd"] == 117 / 256
    assert by_depth[10]["sign_matches_ibm_phase2"] is True
    assert all("job_id" not in row for row in summary["depth_summaries"])


def test_layout_slug_is_stable() -> None:
    module = _load_module()

    assert module._layout_slug([13, 8, 9, 14]) == "q13-8-9-14"
