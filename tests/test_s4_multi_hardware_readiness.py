# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S4 multi-hardware readiness tests
"""Tests for no-submit S4 multi-hardware readiness artefacts."""

from __future__ import annotations

from scripts.export_s4_multi_hardware_readiness import build_readiness_payload


def test_s4_readiness_payload_is_non_submitting() -> None:
    payload = build_readiness_payload()

    assert payload["hardware_submission"] is False
    assert payload["cloud_contact"] is False
    assert payload["qpu_budget_requested_seconds"] == 0.0
    assert "no non-IBM hardware result" in payload["blocked_claims"]


def test_s4_provider_plans_are_approval_gated() -> None:
    payload = build_readiness_payload()
    plans = {plan["provider"]: plan for plan in payload["provider_plans"]}

    assert set(plans) == {"pulser", "bloqade", "ibm_pulse"}
    for plan in plans.values():
        assert plan["export"]["can_submit"] is False
        assert plan["execution_plan"]["approved"] is False
        assert plan["execution_plan"]["can_execute"] is False
        assert "execution_plan_only_no_provider_contact" in plan["execution_plan"]["limitations"]


def test_s4_programmes_cover_neutral_atom_and_circuit_qed_paths() -> None:
    payload = build_readiness_payload()

    neutral = payload["programmes"]["neutral_atoms"]
    circuit_qed = payload["programmes"]["circuit_qed"]

    assert neutral["platform"] == "neutral_atoms"
    assert neutral["payload"]["schema"] == "native_ahs_v1"
    assert circuit_qed["platform"] == "circuit_qed"
    assert circuit_qed["payload"]["schema"] == "exchange_resonator_v1"
