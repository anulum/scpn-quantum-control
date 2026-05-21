# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S1 IBM metadata probe
"""Tests for no-submit S1 IBM metadata probe helpers."""

from __future__ import annotations

import json

from scripts.probe_s1_ibm_metadata import (
    build_decision_document,
    load_snapshot_from_metadata_json,
)


def test_load_snapshot_from_metadata_json_and_build_decision(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "ibm",
                "backend_name": "ibm_template",
                "n_qubits": 100,
                "supported_features": [
                    "cross_shot_batches",
                    "mid_circuit_measurement",
                    "conditional_control",
                    "conditional_reset",
                    "pulse_control",
                    "drive_channel_access",
                ],
                "max_shots": 4096,
                "max_circuits": 16,
                "metadata": {
                    "openpulse_profile": {
                        "supports_pulse_control": True,
                        "supports_drive_channel_access": True,
                        "supports_measure_channel_access": False,
                        "supports_control_channel_access": False,
                        "n_control_channels": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_snapshot_from_metadata_json(metadata_path)
    document = build_decision_document(snapshot)

    assert document["hardware_submission"] is False
    assert document["credential_string_argument_supported"] is False
    assert document["capability_decision"]["status"] == "ready"
    assert document["package_budget"]["circuits"] == 2
    assert document["openpulse_readiness_status"] == "ready"
    assert document["openpulse_blockers"] == []
    assert document["openpulse_readiness"]["ready"] is True
    assert document["openpulse_readiness"]["hardware_submission"] is False
