# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 QSNN convergence tests
"""Real parameter-shift and spike-readout tests for the QSNN example."""

from scpn_quantum_control.ml_examples import (
    FrameworkStatus,
    qsnn_example_spec,
    qsnn_framework_rows,
    run_qsnn_convergence_example,
)


def test_qsnn_example_meets_loss_and_final_spike_gates() -> None:
    """Train one quantum synapse and verify only the final spike boundary."""
    certificate = run_qsnn_convergence_example()
    details = dict(certificate.details)

    assert certificate.spec == qsnn_example_spec()
    assert certificate.passed
    assert certificate.best_loss <= 1e-5
    assert certificate.loss_drop >= 0.7
    assert certificate.metric_name == "final_spike_silenced"
    assert details["final_spikes"] == [0]
    assert details["temporal_coding"] == "not_modelled_by_this_dense-layer convergence example"
    assert details["neuromorphic_hardware"] is False


def test_qsnn_framework_matrix_preserves_spiking_and_hardware_boundaries() -> None:
    """Expose native execution and refuse invented framework/hardware rows."""
    rows = {row.framework: row for row in qsnn_framework_rows()}

    assert rows["scpn_qsnn_statevector"].status is FrameworkStatus.RAN
    assert rows["scpn_qsnn_statevector"].gate_passed
    assert rows["jax"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["pytorch"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["tensorflow"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["neuromorphic_hardware"].status is FrameworkStatus.UNSUPPORTED
