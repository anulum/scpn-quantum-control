# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — realtime latency scenarios module
"""Dedicated realtime-control latency scenarios independent from S1 scientific batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class RealtimeLatencyScenario:
    """Descriptor for a latency scenario lane."""

    lane: str
    rounds: int
    shots: int
    dense_feedback: bool
    gain_profile: tuple[float, float, float]


def _bounded_gain(value: float) -> float:
    return float(np.clip(value, 0.01, 0.95))


def build_dynamic_feedback_circuit(scenario: RealtimeLatencyScenario) -> QuantumCircuit:
    """Build dynamic-circuit lane with mid-circuit monitor and classical conditioning."""
    qc = QuantumCircuit(4, scenario.rounds)
    plant = (0, 1, 2)
    monitor = 3
    for r in range(scenario.rounds):
        g0, g1, g2 = (_bounded_gain(v) for v in scenario.gain_profile)
        qc.ry(g0, plant[0])
        qc.ry(g1, plant[1])
        qc.ry(g2, plant[2])
        qc.cx(plant[0], monitor)
        qc.crz(g1, plant[1], monitor)
        qc.cx(plant[2], monitor)
        qc.measure(monitor, r)
        if scenario.dense_feedback:
            with qc.if_test((qc.clbits[r], 1)):
                qc.x(plant[0])
                qc.z(plant[1])
                qc.y(plant[2])
        else:
            with qc.if_test((qc.clbits[r], 1)):
                qc.x(plant[0])
        qc.h(monitor)
    qc.measure_all(add_bits=True)
    return qc


def build_open_loop_reference_circuit(scenario: RealtimeLatencyScenario) -> QuantumCircuit:
    """Build open-loop reference without dynamic classical feedback branches."""
    qc = QuantumCircuit(4)
    plant = (0, 1, 2)
    monitor = 3
    for _ in range(scenario.rounds):
        g0, g1, g2 = (_bounded_gain(v) for v in scenario.gain_profile)
        qc.ry(g0, plant[0])
        qc.ry(g1, plant[1])
        qc.ry(g2, plant[2])
        qc.cx(plant[0], monitor)
        qc.crz(g1, plant[1], monitor)
        qc.cx(plant[2], monitor)
        qc.x(plant[0])
        if scenario.dense_feedback:
            qc.z(plant[1])
            qc.y(plant[2])
        qc.h(monitor)
    qc.measure_all(add_bits=True)
    return qc


def default_realtime_latency_scenarios(shots: int) -> tuple[RealtimeLatencyScenario, ...]:
    """Return default dedicated control-latency scenarios."""
    return (
        RealtimeLatencyScenario(
            lane="rt_adaptive_dense",
            rounds=3,
            shots=shots,
            dense_feedback=True,
            gain_profile=(0.23, 0.41, 0.67),
        ),
        RealtimeLatencyScenario(
            lane="rt_adaptive_sparse",
            rounds=3,
            shots=shots,
            dense_feedback=False,
            gain_profile=(0.19, 0.37, 0.59),
        ),
    )


__all__ = [
    "RealtimeLatencyScenario",
    "build_dynamic_feedback_circuit",
    "build_open_loop_reference_circuit",
    "default_realtime_latency_scenarios",
]
