# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — realtime latency scenarios tests
"""Tests for dedicated realtime-control latency scenarios."""

from __future__ import annotations

from scpn_quantum_control.hardware.realtime_latency_scenarios import (
    build_dynamic_feedback_circuit,
    build_open_loop_reference_circuit,
    default_realtime_latency_scenarios,
)


def test_default_realtime_latency_scenarios_are_dedicated_non_s1() -> None:
    scenarios = default_realtime_latency_scenarios(2048)
    lanes = {scenario.lane for scenario in scenarios}
    assert lanes == {"rt_adaptive_dense", "rt_adaptive_sparse"}
    assert all(s.rounds == 3 for s in scenarios)
    assert all(s.shots == 2048 for s in scenarios)


def test_dynamic_feedback_circuit_contains_midcircuit_measurement_and_conditionals() -> None:
    scenario = default_realtime_latency_scenarios(1024)[0]
    circuit = build_dynamic_feedback_circuit(scenario)
    assert circuit.num_qubits == 4
    assert circuit.num_clbits >= scenario.rounds
    measure_count = sum(
        1 for instruction in circuit.data if instruction.operation.name == "measure"
    )
    assert measure_count >= scenario.rounds
    has_control_flow = any(instruction.operation.name == "if_else" for instruction in circuit.data)
    assert has_control_flow


def test_open_loop_reference_contains_no_conditionals() -> None:
    scenario = default_realtime_latency_scenarios(1024)[1]
    circuit = build_open_loop_reference_circuit(scenario)
    assert circuit.num_qubits == 4
    has_control_flow = any(instruction.operation.name == "if_else" for instruction in circuit.data)
    assert not has_control_flow
