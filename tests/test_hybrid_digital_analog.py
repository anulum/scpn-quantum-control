# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Hybrid Digital-Analog Execution
"""Tests for split analog-native and digital-residual execution plans."""

import numpy as np
import pytest

from scpn_quantum_control.hardware import backends as be
from scpn_quantum_control.hardware.hybrid_digital_analog import (
    HybridDigitalAnalogBackend,
    HybridRoute,
    _partition_numpy,
    compile_hybrid_digital_analog,
    partition_kuramoto_couplings,
)
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem, compile_hybrid_program


def _inputs():
    K = np.array(
        [
            [0.0, 0.8, -0.4, 0.1],
            [0.8, 0.0, 0.3, 0.0],
            [-0.4, 0.3, 0.0, -0.2],
            [0.1, 0.0, -0.2, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float64)
    return K, omega


def test_partition_selects_largest_couplings_and_preserves_residual():
    K, _omega = _inputs()
    partition = partition_kuramoto_couplings(K, max_analog_couplers=2)

    analog_edges = {
        (assignment.source, assignment.target)
        for assignment in partition.assignments
        if assignment.route == HybridRoute.ANALOG
    }
    digital_edges = {
        (assignment.source, assignment.target)
        for assignment in partition.assignments
        if assignment.route == HybridRoute.DIGITAL
    }
    assert analog_edges == {(0, 1), (0, 2)}
    assert digital_edges == {(0, 3), (1, 2), (2, 3)}
    np.testing.assert_allclose(partition.analog_K_nm + partition.digital_K_nm, K)
    assert partition.n_analog_couplings == 2
    assert partition.n_digital_couplings == 3


def test_hybrid_program_contains_analog_block_and_digital_residual():
    K, omega = _inputs()
    program = compile_hybrid_digital_analog(
        K,
        omega,
        platform="circuit_qed",
        duration=1.25,
        max_analog_couplers=2,
        trotter_steps=3,
    )

    assert program.payload["schema"] == "hybrid_digital_analog_v1"
    assert program.n_oscillators == 4
    assert program.n_analog_couplers == 2
    assert program.n_digital_couplers == 3
    assert program.analog_program.n_couplers == 2
    assert program.payload["schedule"][0]["route"] == "analog_native"
    digital_payload = program.payload["schedule"][1]["payload"]
    assert digital_payload["schema"] == "digital_residual_qasm2_v1"
    assert digital_payload["has_residual"] is True
    assert "qreg q[4];" in digital_payload["qasm2"]
    assert program.metadata["digital_local_detunings"] == "zero_residual_to_avoid_double_counting"


def test_threshold_can_route_only_hardware_native_large_terms():
    K, omega = _inputs()
    program = compile_hybrid_digital_analog(
        K,
        omega,
        platform="neutral_atoms",
        duration=0.75,
        analog_threshold=0.35,
    )

    analog_edges = {
        (assignment.source, assignment.target)
        for assignment in program.partition.assignments
        if assignment.route == HybridRoute.ANALOG
    }
    assert analog_edges == {(0, 1), (0, 2)}
    assert program.analog_program.payload["schema"] == "native_ahs_v1"
    assert program.partition.n_digital_couplings == 3


def test_zero_analog_budget_compiles_identity_analog_couplers_and_digital_work():
    K, omega = _inputs()
    program = compile_hybrid_digital_analog(
        K,
        omega,
        platform="continuous_variable",
        duration=0.5,
        max_analog_couplers=0,
    )

    assert program.n_analog_couplers == 0
    assert program.analog_program.n_couplers == 0
    assert program.n_digital_couplers == 5
    assert program.payload["schedule"][1]["payload"]["has_residual"] is True


def test_all_analog_route_uses_empty_residual_circuit():
    K, omega = _inputs()
    program = compile_hybrid_digital_analog(
        K,
        omega,
        platform="circuit_qed",
        duration=1.0,
        max_analog_couplers=99,
    )

    assert program.n_analog_couplers == 5
    assert program.n_digital_couplers == 0
    assert program.digital_circuit.num_qubits == 4
    assert program.digital_circuit.size() == 0
    assert program.payload["schedule"][1]["payload"]["has_residual"] is False


def test_backend_registry_exposes_hybrid_compiler():
    backend = be.get_backend("hybrid_digital_analog")
    assert backend.name == "hybrid_digital_analog"
    assert backend.is_available() is True
    assert "hybrid_digital_analog" in be.list_backends(auto_discover=False)


def test_kuramoto_core_facade_compiles_hybrid_program():
    K, omega = _inputs()
    problem = build_kuramoto_problem(K, omega, metadata={"case": "hybrid_facade"})
    program = compile_hybrid_program(
        problem,
        platform="circuit_qed",
        duration=1.0,
        max_analog_couplers=1,
    )
    assert program.metadata["case"] == "hybrid_facade"
    assert program.n_analog_couplers == 1
    assert program.n_digital_couplers == 4


def test_backend_validation_rejects_invalid_split_parameters():
    K, omega = _inputs()
    problem = build_kuramoto_problem(K, omega)
    backend = HybridDigitalAnalogBackend()
    with pytest.raises(ValueError, match="duration"):
        backend.compile(problem, duration=0.0)
    with pytest.raises(ValueError, match="max_analog_couplers"):
        backend.compile(problem, duration=1.0, max_analog_couplers=-1)
    with pytest.raises(ValueError, match="analog_threshold"):
        backend.compile(problem, duration=1.0, analog_threshold=-0.1)
    with pytest.raises(ValueError, match="trotter_steps"):
        backend.compile(problem, duration=1.0, trotter_steps=0)


def test_numpy_partition_matches_expected_route_codes():
    K, _omega = _inputs()
    analog, digital, rows, cols, codes = _partition_numpy(
        K,
        analog_budget=2,
        analog_threshold=0.0,
        zero_threshold=1e-12,
    )
    np.testing.assert_array_equal(rows, np.array([0, 0, 0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(cols, np.array([1, 2, 3, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(codes, np.array([1, 1, 0, 0, 0], dtype=np.int64))
    np.testing.assert_allclose(analog + digital, K)
