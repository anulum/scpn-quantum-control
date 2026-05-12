# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 state/layout analysis tests
"""Tests for the Phase 3 state/layout DLA analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyse_phase3_state_layout_dla.py"
    spec = importlib.util.spec_from_file_location("analyse_phase3_state_layout_dla", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load state/layout analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_analysis_matches_preregistered_shape() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_analysis(payload, input_sha256="fixture")

    assert summary["schema"] == "scpn_phase3_state_layout_dla_analysis_v1"
    assert summary["backend"] == "ibm_marrakesh"
    assert summary["job_ids"] == ["ibm-run-aabcf620230b1438", "ibm-run-eea172711aa52b78"]
    assert len(summary["state_depth_layout_rows"]) == 60
    assert len(summary["comparison_rows"]) == 48
    assert len(summary["layout_summaries"]) == 3
    assert len(summary["readout_rows"]) == 15


def test_build_analysis_records_claim_boundaries_and_flags() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_analysis(payload, input_sha256="fixture")

    blocked = summary["claim_boundary"]["blocked"]
    assert "DLA-parity-only causality" in blocked
    assert "quantum advantage" in blocked
    assert set(summary["decision_flags"]) == {
        "layout_spread_exceeds_mean_original_contrast",
        "original_contrast_mixed_sign",
        "within_sector_controls_significant",
    }


def test_original_comparison_has_all_layout_depth_tests() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_analysis(payload, input_sha256="fixture")
    original = summary["fisher_by_comparison"]["original_E0_minus_O0"]

    assert original["n_tests"] == 12
    assert sum(original["signs"].values()) == 12
