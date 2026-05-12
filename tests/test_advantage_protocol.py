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
        row["notes"] = ["size-gated"]

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
    assert "n=4:dense_eigh" in validation.missing_required
    assert any("invalid status" in item for item in validation.invalid_rows)


def _complete_row(protocol_id: str, n_qubits: int, baseline: str) -> dict:
    return {
        "protocol_id": protocol_id,
        "n_qubits": n_qubits,
        "baseline": baseline,
        "status": "skipped",
        "wall_time_ms": None,
        "memory_bytes": None,
        "metric_payload": {},
        "command": "test",
        "machine": "test",
        "dependencies": {},
        "git_commit": "test",
        "notes": ["size-gated test row"],
    }


def test_validate_scaling_rows_requires_each_required_baseline_per_size() -> None:
    protocol = default_s2_scaling_protocol()
    rows = [
        _complete_row(protocol.protocol_id, 4, baseline)
        for baseline in protocol.required_baselines
    ]
    rows.extend(
        [
            _complete_row(protocol.protocol_id, 6, "classical_ode"),
            _complete_row(protocol.protocol_id, 6, "dense_eigh"),
        ]
    )

    validation = validate_scaling_rows(protocol, rows)

    assert validation.valid is False
    assert "n=6:mps_tensor_network" in validation.missing_required
    assert "n=6:sparse_eigsh" in validation.missing_required


def test_validate_scaling_rows_rejects_unbounded_ok_and_unexplained_skip() -> None:
    protocol = default_s2_scaling_protocol()
    rows = [
        _complete_row(protocol.protocol_id, 4, baseline)
        for baseline in protocol.required_baselines
    ]
    rows[0]["status"] = "ok"
    rows[0]["wall_time_ms"] = -1.0
    rows[0]["memory_bytes"] = -5
    rows[1]["notes"] = []

    validation = validate_scaling_rows(protocol, rows)

    assert validation.valid is False
    assert any(
        "wall_time_ms must be finite and non-negative" in item for item in validation.invalid_rows
    )
    assert any(
        "memory_bytes must be a non-negative integer" in item for item in validation.invalid_rows
    )
    assert any("skipped row requires notes" in item for item in validation.invalid_rows)
