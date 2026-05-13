# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout campaign tests
"""Tests for IQM layout campaign summary generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyse_iqm_layout_campaign.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_iqm_layout_campaign", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM layout campaign script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_summary_separates_completed_and_cancelled_blocks() -> None:
    module = _load_module()

    summary = module.summarise_campaign()

    assert summary["completed_blocks"] == 6
    assert summary["completed_circuits"] == 48
    assert summary["completed_shots"] == 12288
    assert summary["cancelled_submitted_circuits"] == 1
    assert summary["cancelled_layout_blocks"][0]["status_after_cancel"] == "CANCELLED"
    assert "job_id" not in summary["cancelled_layout_blocks"][0]
    assert summary["completed_layout_blocks"][-1]["layout"] == [2, 7, 12, 13]
    assert summary["completed_layout_blocks"][-1]["sign_matches_vs_ibm_phase2"] == "3/3"
