# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example QSNN convergence tests
"""Real parameter-shift and spike-readout tests for the QSNN example."""

import sys
from cProfile import Profile
from dataclasses import replace
from inspect import getfile
from types import CodeType
from typing import cast

import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.ml_examples import (
    FrameworkStatus,
    ModelFamily,
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
    assert certificate.spec.example_id == "qsnn_single_synapse_silencing_v2"
    assert certificate.spec.max_steps == 1024
    predecessor = cast(dict[str, object], details["predecessor"])
    predecessor_spec = cast(dict[str, object], predecessor["spec"])
    assert predecessor_spec["max_steps"] == 16
    assert predecessor_spec["target_loss"] == certificate.spec.target_loss
    assert predecessor_spec["min_loss_drop"] == certificate.spec.min_loss_drop
    assert predecessor["passed"] is False
    assert predecessor["target_reached"] is False
    assert predecessor["deterministic_replay"] is True


def test_qsnn_framework_matrix_preserves_spiking_and_hardware_boundaries() -> None:
    """Expose native execution and refuse invented framework/hardware rows."""
    rows = {row.framework: row for row in qsnn_framework_rows()}

    assert rows["scpn_qsnn_statevector"].status is FrameworkStatus.RAN
    assert rows["scpn_qsnn_statevector"].gate_passed
    assert rows["jax"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["pytorch"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["tensorflow"].status is FrameworkStatus.NOT_APPLICABLE
    assert rows["neuromorphic_hardware"].status is FrameworkStatus.UNSUPPORTED


def test_native_qsnn_row_matches_the_observed_training_result() -> None:
    """Do not report native convergence independently of the actual frozen run."""
    certificate = run_qsnn_convergence_example()
    native = qsnn_framework_rows()[0]
    assert native.executed
    assert native.passed is certificate.passed
    assert native.gate_passed is certificate.passed
    assert native.max_abs_error is None


def test_failed_predecessor_remains_failed_in_native_framework_evidence() -> None:
    """Retain the real failed 16-step run rather than labelling its native row passed."""
    certificate = run_qsnn_convergence_example(protocol="legacy_16_step")
    assert not certificate.passed
    assert certificate.spec == qsnn_example_spec(protocol="legacy_16_step")
    row = qsnn_framework_rows(certificate=certificate)[0]
    assert row.executed and row.required
    assert row.status is FrameworkStatus.FAILED
    assert row.passed is False and row.gate_passed is False
    assert row.max_abs_error is None
    assert certificate.spec.example_id in row.reason


@pytest.mark.parametrize("corruption", ["family", "protocol"])
def test_qsnn_framework_row_refuses_a_foreign_certificate(corruption: str) -> None:
    """Reject an altered task identity on a real completed predecessor run.

    Parameters
    ----------
    corruption
        Foreign model family or an unregistered protocol identifier.

    """
    certificate = run_qsnn_convergence_example(protocol="legacy_16_step")
    if corruption == "family":
        spec = replace(certificate.spec, family=ModelFamily.QNN)
    else:
        spec = replace(certificate.spec, example_id="unknown-protocol")
    with pytest.raises(ValueError, match="registered frozen protocol"):
        qsnn_framework_rows(certificate=replace(certificate, spec=spec))


def test_unknown_qsnn_protocol_refuses_before_running_a_trainer() -> None:
    """Unknown task versions produce no substituted training result."""
    with pytest.raises(ValueError, match="unknown QSNN convergence protocol"):
        qsnn_example_spec(protocol="unknown")
    with pytest.raises(ValueError, match="unknown QSNN convergence protocol"):
        run_qsnn_convergence_example(protocol="unknown")


def test_qsnn_protocol_cost_matches_observed_statevector_constructions() -> None:
    """Count every real primary, replay and predecessor circuit evaluation."""
    previous = sys.getprofile()
    profiler = Profile()
    try:
        profiler.enable()
        certificate = run_qsnn_convergence_example()
    finally:
        profiler.disable()
        sys.setprofile(previous)
    target = (getfile(Statevector.from_instruction), "from_instruction")
    constructions = sum(
        entry.callcount
        for entry in profiler.getstats()
        if isinstance(entry.code, CodeType)
        and (entry.code.co_filename, entry.code.co_name) == target
    )
    assert constructions > 0
    assert dict(certificate.details)["total_statevector_evaluations"] == constructions
