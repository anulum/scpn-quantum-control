# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for hardware job dossiers
"""Tests for the standard hardware-job dossier schema."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.job_dossier import HardwareJobDossier


def _dossier() -> HardwareJobDossier:
    return HardwareJobDossier(
        job_id="job",
        title="Title",
        purpose="Purpose",
        hypothesis="Hypothesis",
        falsification_condition="Falsification",
        expected_observables=("observable",),
        circuit_summary={"depth": 3},
        qpu_budget={"seconds": 5},
        platform_fit={"ibm": "ready"},
        risks_and_confounds=("risk",),
        decision_tree={"positive": "continue"},
        paper_impact="Paper",
        follow_up_avenue="Follow-up",
        possibilities_opened=("possibility",),
        claim_boundary="Boundary",
        reproducibility_package={"manifest": "path"},
    )


def test_hardware_job_dossier_serialises_to_dict_and_markdown() -> None:
    dossier = _dossier()

    data = dossier.to_dict()
    markdown = dossier.to_markdown()

    assert data["job_id"] == "job"
    assert data["expected_observables"] == ["observable"]
    assert "## Falsification Condition" in markdown
    assert "- `depth`: 3" in markdown


def test_hardware_job_dossier_rejects_missing_required_sections() -> None:
    with pytest.raises(ValueError, match="expected_observables"):
        HardwareJobDossier(
            job_id="job",
            title="Title",
            purpose="Purpose",
            hypothesis="Hypothesis",
            falsification_condition="Falsification",
            expected_observables=(),
            circuit_summary={"depth": 3},
            qpu_budget={"seconds": 5},
            platform_fit={"ibm": "ready"},
            risks_and_confounds=("risk",),
            decision_tree={"positive": "continue"},
            paper_impact="Paper",
            follow_up_avenue="Follow-up",
            possibilities_opened=("possibility",),
            claim_boundary="Boundary",
            reproducibility_package={"manifest": "path"},
        )
