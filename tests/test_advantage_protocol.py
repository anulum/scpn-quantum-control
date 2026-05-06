# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S2 scaling protocol
"""Tests for the S2 scaling benchmark protocol."""

from __future__ import annotations

import pytest

from scpn_quantum_control.benchmarks.advantage_protocol import (
    ScalingBaseline,
    default_s2_scaling_protocol,
    validate_scaling_rows,
)


def test_default_s2_scaling_protocol_contains_required_baselines_and_boundaries() -> None:
    protocol = default_s2_scaling_protocol()
    data = protocol.to_dict()

    assert protocol.sizes == (4, 6, 8, 10, 12, 14, 16, 18, 20)
    assert "mps_tensor_network" in protocol.required_baselines
    assert "aer_statevector" in protocol.required_baselines
    assert "qpu_hardware" not in protocol.required_baselines
    assert data["output_schema"]["row_keys"][0] == "protocol_id"
    assert "must not claim broad quantum advantage" in protocol.claim_boundary
    assert any("MPS" in item for item in protocol.falsification)


def test_scaling_baseline_rejects_empty_metrics() -> None:
    with pytest.raises(ValueError, match="metrics"):
        ScalingBaseline(
            kind="classical_ode",
            label="bad",
            required=True,
            max_qubits=4,
            metrics=(),
            claim_boundary="boundary",
        )


def test_validate_scaling_rows_accepts_required_ok_or_skipped_rows() -> None:
    protocol = default_s2_scaling_protocol()
    rows = [
        {key: None for key in protocol.output_schema["row_keys"]}
        for _ in protocol.required_baselines
    ]
    for row, baseline in zip(rows, protocol.required_baselines, strict=True):
        row["protocol_id"] = protocol.protocol_id
        row["n_qubits"] = 4
        row["baseline"] = baseline
        row["status"] = "skipped"
        row["metric_payload"] = {}
        row["command"] = "test"
        row["machine"] = "test"
        row["dependencies"] = {}
        row["git_commit"] = "test"
        row["notes"] = []

    validation = validate_scaling_rows(protocol, rows)

    assert validation.valid is True
    assert validation.missing_required == ()


def test_validate_scaling_rows_rejects_missing_baseline_and_unknown_status() -> None:
    protocol = default_s2_scaling_protocol()

    validation = validate_scaling_rows(
        protocol,
        [
            {
                "protocol_id": protocol.protocol_id,
                "n_qubits": 4,
                "baseline": "classical_ode",
                "status": "not-a-status",
            }
        ],
    )

    assert validation.valid is False
    assert "dense_eigh" in validation.missing_required
    assert any("invalid status" in item for item in validation.invalid_rows)
