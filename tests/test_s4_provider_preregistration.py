# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S4 provider preregistration tests
"""Tests for S4 IBM pulse-level calibration preregistration."""

from __future__ import annotations

from scripts.export_s4_multi_hardware_readiness import build_readiness_payload
from scripts.export_s4_provider_preregistration import build_ibm_pulse_dossier


def test_ibm_pulse_dossier_is_non_submitting() -> None:
    dossier = build_ibm_pulse_dossier(build_readiness_payload())

    assert dossier.job_id == "s4_ibm_pulse_calibration_review"
    assert dossier.qpu_budget["hardware_submission"] is False
    assert dossier.qpu_budget["cloud_contact"] is False
    assert "does not create a pulse Schedule" in dossier.claim_boundary
    assert dossier.circuit_summary["openpulse_readiness_status"] in {"ready", "blocked"}


def test_ibm_pulse_dossier_preserves_provider_summary() -> None:
    dossier = build_ibm_pulse_dossier(build_readiness_payload())

    assert dossier.circuit_summary["provider"] == "ibm_pulse"
    assert dossier.circuit_summary["platform"] == "circuit_qed"
    assert dossier.circuit_summary["native_schema"] == "exchange_resonator_v1"
    assert dossier.circuit_summary["n_oscillators"] == 4
    assert dossier.platform_fit["gate_based_comparator"] == "required_before_execution"
    assert dossier.platform_fit["ibm_openpulse_readiness"] in {"ready", "blocked"}


def test_ibm_pulse_dossier_has_review_decision_tree() -> None:
    dossier = build_ibm_pulse_dossier(build_readiness_payload())

    assert set(dossier.decision_tree) == {"accepted", "manual_review", "fail"}
    assert any("channel map" in item for item in dossier.expected_observables)
    assert any("QPU time" in item for item in dossier.prerequisites)
    assert "openpulse_readiness" in dossier.reproducibility_package
