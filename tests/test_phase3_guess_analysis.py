# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Phase 3 GUESS analysis
"""Tests for the Phase 3 GUESS DLA analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyse_phase3_guess_dla.py"
    spec = importlib.util.spec_from_file_location("analyse_phase3_guess_dla", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 3 GUESS analysis script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase3_guess_analysis_shape_and_jobs() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_summary(payload, input_path=module.DEFAULT_INPUT)

    assert summary["schema"] == "scpn_phase3_guess_dla_analysis_v1"
    assert summary["backend"] == "ibm_marrakesh"
    assert summary["job_ids"] == ["d7tt5lkt738s73cib64g", "d7tt7oaudops7398fdt0"]
    assert summary["n_witness_rows"] == 24
    assert summary["n_fit_rows"] == 16


def test_phase3_guess_analysis_records_claim_boundary() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_summary(payload, input_path=module.DEFAULT_INPUT)

    blocked = summary["claim_boundary"]["blocked"]
    assert "universal GUESS mitigation performance" in blocked
    assert "full confusion-matrix readout mitigation" in blocked
    assert set(summary["decision_flags"]) == {
        "raw_has_any_usable_guess_witness",
        "corrected_has_any_usable_guess_witness",
        "all_corrected_fits_usable",
    }


def test_exact_state_readout_correction_is_applied() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    witness_rows = module.build_witness_rows(payload)

    assert len(witness_rows) == 24
    assert all(row.mean_leakage_corrected is not None for row in witness_rows)
    assert all(0.0 <= row.mean_survival_corrected <= 1.0 for row in witness_rows)
