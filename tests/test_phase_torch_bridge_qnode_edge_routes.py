# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch Phase-QNode edge-route tests
"""Registered Phase-QNode edge-route tests for the PyTorch bridge."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
)

FloatArray = NDArray[np.float64]


def _all_gate_circuit() -> tuple[PhaseQNodeCircuit, FloatArray]:
    """Return a compact circuit that covers the registered PyTorch gate families."""

    operations = (
        PhaseQNodeOperation("h", (0,)),
        PhaseQNodeOperation("x", (1,)),
        PhaseQNodeOperation("y", (2,)),
        PhaseQNodeOperation("z", (0,)),
        PhaseQNodeOperation("s", (1,)),
        PhaseQNodeOperation("t", (2,)),
        PhaseQNodeOperation("sx", (0,)),
        PhaseQNodeOperation("rx", (0,), parameter_index=0),
        PhaseQNodeOperation("ry", (1,), parameter_index=1),
        PhaseQNodeOperation("rz", (2,), parameter_index=2),
        PhaseQNodeOperation("phase", (0,), parameter_index=3),
        PhaseQNodeOperation("cnot", (0, 1)),
        PhaseQNodeOperation("cz", (1, 2)),
        PhaseQNodeOperation("cy", (0, 2)),
        PhaseQNodeOperation("swap", (1, 2)),
        PhaseQNodeOperation("ch", (0, 1)),
        PhaseQNodeOperation("cs", (1, 2)),
        PhaseQNodeOperation("ct", (0, 2)),
        PhaseQNodeOperation("ccnot", (0, 1, 2)),
        PhaseQNodeOperation("ccz", (0, 1, 2)),
        PhaseQNodeOperation("cswap", (0, 1, 2)),
        PhaseQNodeOperation("crx", (0, 1), parameter_index=4),
        PhaseQNodeOperation("cry", (1, 2), parameter_index=5),
        PhaseQNodeOperation("crz", (0, 2), parameter_index=6),
        PhaseQNodeOperation("rxx", (0, 1), parameter_index=7),
        PhaseQNodeOperation("ryy", (1, 2), parameter_index=8),
        PhaseQNodeOperation("rzz", (0, 2), parameter_index=9),
    )
    observable = SparsePauliHamiltonian(
        (
            PauliTerm(0.25, ((0, "x"),)),
            PauliTerm(-0.5, ((1, "y"),)),
            PauliTerm(0.75, ((2, "z"),)),
        ),
    )
    params = np.linspace(-0.4, 0.5, 10, dtype=np.float64)
    return PhaseQNodeCircuit(n_qubits=3, operations=operations, observable=observable), params


def test_torch_phase_qnode_value_and_grad_covers_registered_gate_families() -> None:
    """The PyTorch statevector route should execute all registered local gate families."""

    circuit, params = _all_gate_circuit()

    result = torch_phase_qnode_value_and_grad(circuit, params, tolerance=1e-8)

    assert result.passed
    assert result.gradient.shape == params.shape
    assert result.max_abs_error <= 1e-8
    assert (
        result.to_dict()["claim_boundary"] == "registered_phase_qnode_torch_statevector_lowering"
    )


def test_torch_phase_qnode_observable_routes_cover_dense_and_covariance() -> None:
    """The PyTorch statevector route should execute dense and covariance observables."""

    dense = DenseHermitianObservable(
        np.array([[1.0, 0.25], [0.25, -1.0]], dtype=np.complex128),
    )
    dense_circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("ry", (0,), parameter_index=0),),
        observable=dense,
    )
    dense_result = torch_phase_qnode_value_and_grad(
        dense_circuit,
        np.array([0.3], dtype=np.float64),
        tolerance=1e-8,
    )

    covariance = PauliCovarianceObservable(
        left=PauliTerm(1.0, ((0, "x"),)),
        right=PauliTerm(1.0, ((0, "z"),)),
    )
    covariance_circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("ry", (0,), parameter_index=0),),
        observable=covariance,
    )
    covariance_result = torch_phase_qnode_value_and_grad(
        covariance_circuit,
        np.array([0.2], dtype=np.float64),
        tolerance=1e-8,
    )

    assert dense_result.passed
    assert covariance_result.passed
    assert dense_result.state.shape == (2,)
    assert covariance_result.gradient.shape == (1,)


def test_torch_phase_qnode_routes_fail_closed_for_unsupported_circuits() -> None:
    """PyTorch Phase-QNode routes should reject unsupported registered-circuit reports."""

    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("rx", (0,), parameter_index=1),),
        observable="z",
    )
    params = np.array([0.2], dtype=np.float64)

    with pytest.raises(PhaseQNodeSupportError):
        torch_phase_qnode_value_and_grad(circuit, params)
    with pytest.raises(PhaseQNodeSupportError):
        torch_phase_qnode_transform_audit(
            circuit,
            params,
            params_batch=np.vstack([params, params + 0.1]),
        )
    with pytest.raises(PhaseQNodeSupportError):
        torch_phase_qnode_compile_audit(circuit, params)


def test_torch_phase_qnode_transform_audit_rejects_bad_batch_row() -> None:
    """Transform audits should validate every row in the parameter batch."""

    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("rx", (0,), parameter_index=0),),
        observable="z",
    )
    params = np.array([0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="params_batch"):
        torch_phase_qnode_transform_audit(
            circuit,
            params,
            params_batch=np.array([[0.2], [np.nan]], dtype=np.float64),
        )
