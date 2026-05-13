# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM DLA parity analysis tests
"""Regression tests for the IQM DLA/parity minimal analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyse_iqm_dla_parity.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_iqm_dla_parity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM DLA analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyse_iqm_minimal_uses_sanitized_public_input() -> None:
    module = _load_module()

    summary = module.analyse()

    assert summary["provider"] == "iqm"
    assert summary["tier"] == "dla_parity_minimal"
    assert summary["n_depths_tested"] == 3
    assert summary["n_depths_compared_to_ibm_phase2"] == 3
    assert summary["n_signs_matching_ibm_phase2"] == 1
    assert all("job_id" not in row for row in summary["depth_summaries"])


def test_depth_four_counts_match_recorded_iqm_run() -> None:
    module = _load_module()

    summary = module.analyse()
    by_depth = {row["depth"]: row for row in summary["depth_summaries"]}
    depth_four = by_depth[4]

    assert depth_four["iqm_even_leakage_counts"] == 117
    assert depth_four["iqm_odd_leakage_counts"] == 128
    assert depth_four["iqm_even_shots"] == 256
    assert depth_four["iqm_odd_shots"] == 256
    assert depth_four["iqm_asymmetry_relative"] < 0.0
    assert depth_four["ibm_phase2_asymmetry_relative"] > 0.0
    assert depth_four["sign_matches_ibm_phase2"] is False
    assert depth_four["standard_error_difference"] > 0.0
    assert -1.1 < depth_four["z_difference"] < -0.8
