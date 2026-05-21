# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological QEC Pipeline Tests
"""Tests for end-to-end biological surface-code execution payloads."""

import numpy as np

from scpn_quantum_control.qec import (
    run_biological_qec_batch_execution,
    run_biological_qec_execution,
)


def test_run_biological_qec_execution_success_payload():
    """E2E biological pipeline returns successful decode and serialisable payload."""
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=float,
    )
    z_errors = np.zeros(3, dtype=np.int8)
    z_errors[1] = 1
    result = run_biological_qec_execution(
        K,
        z_errors,
        threshold=1e-8,
        node_domains={0: "L1", 1: "L1", 2: "L2", 3: "L2"},
        metadata={"campaign": "e2e"},
    )
    payload = result.to_payload()

    assert result.success is True
    assert result.residual_syndrome_weight == 0
    assert payload["metadata"]["campaign"] == "e2e"
    assert payload["code_summary"]["n_data_qubits"] == 3
    assert payload["diagnostics"]["n_nodes"] == 4
    assert payload["diagnostics"]["inter_domain_coupling"]["L1"]["L2"] > 0.0
    assert payload["decode_backend"] in {
        "rust_exact_mwpm",
        "python_mwpm",
        "python_fallback_high_defect",
    }


def test_run_biological_qec_batch_execution_aggregate_payload():
    """Batch biological pipeline returns aggregate campaign-level metrics."""
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=float,
    )
    z_error_matrix = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )

    batch = run_biological_qec_batch_execution(
        K,
        z_error_matrix,
        threshold=1e-8,
        metadata={"campaign": "batch"},
    )
    payload = batch.to_payload()

    assert batch.n_runs == 2
    assert 0 <= batch.n_success <= batch.n_runs
    assert 0.0 <= batch.success_rate <= 1.0
    assert payload["n_runs"] == 2
    assert payload["metadata"]["campaign"] == "batch"
    assert len(payload["runs"]) == 2
    assert payload["mean_residual_syndrome_weight"] >= 0.0
