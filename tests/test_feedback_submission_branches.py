# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the S1 feedback submission package
"""Guard and circuit-introspection tests for the S1 feedback submission package.

Covers the package argument guards, the conditional-reset readiness reason and
the conditional/operation circuit-introspection helpers including control-flow
block recursion and the legacy-condition path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_submission import (
    FeedbackBudgetEstimate,
    FeedbackCircuitSummary,
    _circuit_has_conditional_operation,
    _circuit_has_conditionals,
    _circuit_has_operation,
    assess_platform_readiness,
    build_s1_feedback_submission_package,
    default_s1_platforms,
)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.array([0.2, 0.5], dtype=np.float64),
    )


def _conditional_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(0)
    return circuit


def test_package_rejects_empty_experiment_id() -> None:
    """An empty experiment id is rejected."""
    with pytest.raises(ValueError, match="experiment_id must be non-empty"):
        build_s1_feedback_submission_package(_controller(), experiment_id="")


def test_package_rejects_non_positive_rounds() -> None:
    """A non-positive round count is rejected."""
    with pytest.raises(ValueError, match="n_rounds must be positive"):
        build_s1_feedback_submission_package(_controller(), n_rounds=0)


def test_readiness_flags_unsupported_conditional_reset() -> None:
    """A conditional-reset payload on a platform without support is flagged."""
    summary = FeedbackCircuitSummary(
        n_qubits=2,
        n_clbits=2,
        depth=4,
        operation_counts={"reset": 1},
        has_mid_circuit_measurement=False,
        has_conditional_control=False,
        has_conditional_reset=True,
        n_rounds=1,
    )
    budget = FeedbackBudgetEstimate(
        circuits=1, shots_per_circuit=1024, repetitions=1, estimated_execution_seconds=1.0
    )
    platform = next(p for p in default_s1_platforms() if not p.supports_conditional_reset)
    readiness = assess_platform_readiness(platform, summary, budget)
    assert "payload requires conditional reset" in readiness.reasons


def test_circuit_has_conditionals_detects_control_flow() -> None:
    """A control-flow circuit is detected as conditional."""
    assert _circuit_has_conditionals(_conditional_circuit()) is True


def test_circuit_has_conditionals_detects_unconditional_control_flow() -> None:
    """A control-flow op without a classical condition is still detected."""
    circuit = QuantumCircuit(1)
    with circuit.for_loop(range(2)):
        circuit.x(0)
    assert _circuit_has_conditionals(circuit) is True


def test_circuit_has_conditionals_false_for_plain_circuit() -> None:
    """A plain circuit has no conditional control flow."""
    plain = QuantumCircuit(1)
    plain.h(0)
    assert _circuit_has_conditionals(plain) is False


def test_circuit_has_operation_top_level() -> None:
    """A top-level operation is detected by name."""
    plain = QuantumCircuit(1)
    plain.h(0)
    assert _circuit_has_operation(plain, "h") is True


def test_circuit_has_operation_inside_control_flow_block() -> None:
    """An operation nested inside a control-flow block is detected."""
    assert _circuit_has_operation(_conditional_circuit(), "x") is True


def test_circuit_has_conditional_operation_in_control_flow_block() -> None:
    """A control-flow block carrying the target operation is detected."""
    assert _circuit_has_conditional_operation(_conditional_circuit(), "x") is True


def test_circuit_has_conditional_operation_legacy_condition() -> None:
    """A legacy condition-bearing operation matching the name is detected."""
    legacy_instruction = SimpleNamespace(
        operation=SimpleNamespace(name="reset", condition=object(), blocks=())
    )
    fake_circuit = SimpleNamespace(data=[legacy_instruction])
    assert _circuit_has_conditional_operation(cast(QuantumCircuit, fake_circuit), "reset") is True
