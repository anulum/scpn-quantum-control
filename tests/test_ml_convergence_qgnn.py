# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 QGNN convergence tests
"""Real graph-conditioned Phase-QNode tests for the QGNN example."""

from scpn_quantum_control.ml_examples import (
    FrameworkStatus,
    qgnn_example_spec,
    qgnn_framework_rows,
    run_qgnn_convergence_example,
)


def test_qgnn_example_converges_monotonically_and_replays() -> None:
    """Fit seeded K_nm graphs through the existing QGNN trainer."""
    certificate = run_qgnn_convergence_example()

    assert certificate.spec == qgnn_example_spec()
    assert certificate.passed
    assert certificate.best_loss <= 5e-3
    assert certificate.loss_drop >= 0.45
    assert certificate.deterministic_replay
    assert all(
        current <= previous
        for previous, current in zip(certificate.loss_history, certificate.loss_history[1:])
    )
    assert dict(certificate.details)["nodes_per_graph"] == 3


def test_qgnn_framework_matrix_has_no_blank_or_invented_cells() -> None:
    """Record the native chained gradient and explicit absent adapters."""
    rows = qgnn_framework_rows()

    assert rows[0].status is FrameworkStatus.RAN
    assert rows[0].required and rows[0].gate_passed
    assert {row.framework for row in rows[1:]} == {"jax", "pytorch", "tensorflow"}
    assert all(row.status is FrameworkStatus.NOT_APPLICABLE for row in rows[1:])
    assert all(row.reason for row in rows)
