# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the real-time latency scenarios
"""Branch tests for the dynamic and open-loop latency scenario circuit builders.

Covers the sparse-feedback closed-loop branch and the dense-feedback open-loop
branch that the default scenarios do not exercise.
"""

from __future__ import annotations

from scpn_quantum_control.hardware.realtime_latency_scenarios import (
    RealtimeLatencyScenario,
    build_dynamic_feedback_circuit,
    build_open_loop_reference_circuit,
)


def _scenario(*, dense_feedback: bool) -> RealtimeLatencyScenario:
    return RealtimeLatencyScenario(
        lane="test",
        rounds=1,
        shots=128,
        dense_feedback=dense_feedback,
        gain_profile=(0.1, 0.2, 0.3),
    )


def test_dynamic_feedback_sparse_branch() -> None:
    """The closed-loop builder takes the single-gate sparse-feedback branch."""
    circuit = build_dynamic_feedback_circuit(_scenario(dense_feedback=False))
    assert circuit.num_qubits == 4


def test_open_loop_dense_branch() -> None:
    """The open-loop reference builder takes the dense-feedback gate branch."""
    circuit = build_open_loop_reference_circuit(_scenario(dense_feedback=True))
    assert circuit.num_qubits == 4
