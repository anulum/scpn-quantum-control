# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — s4 neutral atom preregistration tests
# scpn-quantum-control -- S4 neutral-atom preregistration tests
"""Tests for S4 neutral-atom provider-object preregistration."""

from __future__ import annotations

from scripts.export_s4_multi_hardware_readiness import build_readiness_payload
from scripts.export_s4_neutral_atom_preregistration import build_neutral_atom_dossier


def test_neutral_atom_dossier_is_non_submitting() -> None:
    dossier = build_neutral_atom_dossier(build_readiness_payload())

    assert dossier.job_id == "s4_neutral_atom_provider_object_review"
    assert dossier.qpu_budget["hardware_submission"] is False
    assert dossier.qpu_budget["cloud_contact"] is False
    assert "does not import provider SDK constructors" in dossier.claim_boundary


def test_neutral_atom_dossier_covers_pulser_and_bloqade() -> None:
    dossier = build_neutral_atom_dossier(build_readiness_payload())

    assert dossier.circuit_summary["providers"] == "pulser,bloqade"
    assert dossier.circuit_summary["platform"] == "neutral_atoms"
    assert dossier.circuit_summary["pulser_schema"] == "native_ahs_v1"
    assert dossier.circuit_summary["bloqade_schema"] == "native_ahs_v1"
    assert set(dossier.platform_fit) >= {"pulser", "bloqade"}


def test_neutral_atom_dossier_records_comparator_and_credit_gates() -> None:
    dossier = build_neutral_atom_dossier(build_readiness_payload())

    assert set(dossier.decision_tree) == {"accepted", "manual_review", "fail"}
    assert any("matched digital comparator" in item for item in dossier.expected_observables)
    assert any("provider credit" in item for item in dossier.prerequisites)
