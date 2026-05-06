# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S2 lite scaling harness
"""Tests for S2 lite scaling harness."""

from __future__ import annotations

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)
from scripts.bench_s2_scaling_lite import build_rows


def test_build_rows_emits_required_protocol_baselines() -> None:
    rows = build_rows((4,))
    validation = validate_scaling_rows(default_s2_scaling_protocol(), rows)

    assert validation.valid is True
    assert {row["baseline"] for row in rows} >= set(
        default_s2_scaling_protocol().required_baselines
    )
    assert any(row["baseline"] == "dense_eigh" and row["status"] == "ok" for row in rows)
    assert any(row["baseline"] == "sparse_eigsh" and row["status"] == "ok" for row in rows)
    assert any(
        row["baseline"] == "mps_tensor_network"
        and row["status"] == "ok"
        and "worst_cut_discarded_weight" in row["metric_payload"]
        for row in rows
    )
    assert any(
        row["baseline"] == "aer_statevector"
        and row["status"] == "ok"
        and "circuit_depth" in row["metric_payload"]
        for row in rows
    )
    assert all(row["memory_bytes"] is not None for row in rows if row["status"] == "ok")
    assert all(
        "peak_tracemalloc_bytes" in row["metric_payload"] for row in rows if row["status"] == "ok"
    )


def test_build_rows_respects_size_gates() -> None:
    rows = build_rows(
        (6,),
        max_dense_qubits=4,
        max_sparse_qubits=4,
        max_tn_qubits=4,
        max_statevector_qubits=4,
    )

    gated = {row["baseline"]: row["status"] for row in rows}

    assert gated["classical_ode"] == "ok"
    assert gated["dense_eigh"] == "skipped"
    assert gated["sparse_eigsh"] == "skipped"
    assert gated["mps_tensor_network"] == "skipped"
    assert gated["aer_statevector"] == "skipped"
