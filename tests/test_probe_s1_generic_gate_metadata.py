# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S1 generic gate metadata probe
"""Tests for no-submit S1 generic gate metadata probe helpers."""

from __future__ import annotations

from scripts.probe_s1_generic_gate_metadata import build_decision_document


def test_build_decision_document_from_generic_gate_metadata() -> None:
    document = build_decision_document(
        {
            "provider": "generic_gate",
            "backend_name": "openqasm3_target",
            "n_qubits": 32,
            "supported_features": [
                "cross_shot_batches",
                "mid_circuit_measurement",
                "conditional_control",
                "conditional_reset",
            ],
            "max_shots": 2048,
            "max_circuits": 8,
        }
    )

    assert document["hardware_submission"] is False
    assert document["network_access"] is False
    assert document["capability_decision"]["status"] == "ready"
    assert document["capability_decision"]["backend_name"] == "openqasm3_target"
