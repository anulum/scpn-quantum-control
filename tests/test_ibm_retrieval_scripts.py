# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IBM retrieval script tests
"""Tests for IBM retrieval helper credential hygiene."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retrieve_completed_jobs_has_no_embedded_ibm_instance() -> None:
    script = REPO_ROOT / "scripts" / "retrieve_completed_jobs.py"
    text = script.read_text(encoding="utf-8")

    assert "crn:v1:bluemix" not in text
    assert "SCPN_IBM_CRN" in text
    assert "SCPN_IBM_INSTANCE" in text


def test_retrieve_completed_jobs_uses_repo_root_results_dir() -> None:
    script = REPO_ROOT / "scripts" / "retrieve_completed_jobs.py"
    text = script.read_text(encoding="utf-8")

    assert "REPO_ROOT = Path(__file__).resolve().parents[1]" in text
    assert 'Path("results/march_2026")' not in text
