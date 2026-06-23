# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto compiler input-hardening tests
"""Adversarial input validation and resource-guard tests for the Kuramoto compiler.

Covers two failure surfaces for arbitrary ``K_nm``/``omega`` input:

* the :class:`KuramotoProblem` validation contract (shape, finiteness, symmetry,
  real dtype), and
* the resource guards that fail closed before a pathological ``n`` builds an
  ``O(n**2)`` Pauli operator or materialises a ``2**n`` matrix across the sparse
  Hamiltonian and Trotter-circuit compile paths.

The resource guards are exercised with a tiny budget rather than a huge array, so
the tests assert the fail-closed boundary without allocating pathological memory.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import (
    knm_to_hamiltonian,
    knm_to_sparse_matrix,
)
from scpn_quantum_control.compile_budget import (
    DEFAULT_PAULI_BUDGET_ENV,
    PauliOperatorBudgetError,
)
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.kuramoto_core import (
    build_kuramoto_problem,
    compile_hamiltonian,
    compile_trotter_circuit,
    validate_kuramoto_inputs,
)


def _ring(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a valid symmetric nearest-neighbour ring problem of size ``n``."""
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = 0.5
    omega = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return K, omega


# --- KuramotoProblem validation contract (sub-item 1) ---------------------------


def test_rejects_non_square_coupling() -> None:
    """A non-square coupling matrix is rejected with a shape message."""
    with pytest.raises(ValueError, match="K_nm must be a square matrix"):
        build_kuramoto_problem(np.zeros((3, 4)), np.zeros(3))


def test_rejects_empty_coupling() -> None:
    """An empty coupling matrix has no oscillators and is rejected."""
    with pytest.raises(ValueError, match="at least one oscillator"):
        build_kuramoto_problem(np.zeros((0, 0)), np.zeros(0))


def test_rejects_omega_length_mismatch() -> None:
    """A frequency vector whose length differs from ``n`` is rejected."""
    with pytest.raises(ValueError, match=r"omega must have shape"):
        build_kuramoto_problem(np.zeros((3, 3)), np.zeros(2))


def test_rejects_non_finite_coupling() -> None:
    """A non-finite coupling entry is rejected before any compile."""
    K = np.zeros((2, 2))
    K[0, 1] = K[1, 0] = np.inf
    with pytest.raises(ValueError, match="K_nm must contain only finite values"):
        build_kuramoto_problem(K, np.zeros(2))


def test_rejects_nan_frequency() -> None:
    """A NaN natural frequency is rejected."""
    with pytest.raises(ValueError, match="omega must contain only finite values"):
        build_kuramoto_problem(np.zeros((2, 2)), np.array([0.0, np.nan]))


def test_rejects_asymmetric_coupling() -> None:
    """An asymmetric coupling matrix is rejected for the gate-model mapping."""
    K = np.array([[0.0, 1.0], [0.3, 0.0]])
    with pytest.raises(ValueError, match="K_nm must be symmetric"):
        build_kuramoto_problem(K, np.zeros(2))


def test_rejects_complex_coupling_dtype() -> None:
    """A complex coupling matrix is not a real numeric array."""
    K = np.zeros((2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="real numeric"):
        build_kuramoto_problem(K, np.zeros(2))


def test_rejects_string_coupling_dtype() -> None:
    """A string-typed coupling matrix is not a real numeric array."""
    K = np.array([["0", "1"], ["1", "0"]])
    with pytest.raises(ValueError, match="real numeric"):
        build_kuramoto_problem(K, np.zeros(2))


def test_accepts_valid_problem_and_zeroes_diagonal() -> None:
    """A valid problem is frozen, exposes ``n``, and clears the self-coupling."""
    K, omega = _ring(4)
    K[2, 2] = 9.0  # self-coupling must be discarded by validation
    problem = build_kuramoto_problem(K, omega, metadata={"label": "ring"})

    assert problem.n_oscillators == 4
    assert problem.K_nm[2, 2] == 0.0
    assert problem.metadata["label"] == "ring"
    with pytest.raises(ValueError):
        problem.K_nm[0, 0] = 1.0  # frozen, read-only array


def test_large_finite_magnitudes_are_not_rejected() -> None:
    """Unphysically large but finite couplings validate; magnitude is not bounded."""
    K = np.array([[0.0, 1e18], [1e18, 0.0]])
    problem = build_kuramoto_problem(K, np.array([1e15, -1e15]))
    assert np.isfinite(problem.K_nm).all()


def test_validate_returns_symmetrised_copy() -> None:
    """The public validator returns copies and leaves the diagonal cleared."""
    K, omega = _ring(3)
    K_out, omega_out = validate_kuramoto_inputs(K, omega)
    assert np.all(np.diag(K_out) == 0.0)
    assert K_out is not K
    assert omega_out is not omega


# --- Sparse Hamiltonian + Trotter resource guards (sub-items 2, 3) --------------


def test_sparse_matrix_fails_closed_before_exponential_allocation() -> None:
    """``knm_to_sparse_matrix`` fails closed before a 2^n materialisation."""
    K, omega = _ring(8)
    with pytest.raises(DenseAllocationError, match="sparse XY Hamiltonian matrix"):
        knm_to_sparse_matrix(K, omega, max_gib=1e-9)


def test_sparse_operator_fails_closed_under_tiny_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``knm_to_hamiltonian`` fails closed when the operator budget is tiny."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "0.0000001")
    K, omega = _ring(64)
    with pytest.raises(PauliOperatorBudgetError, match="XY/XXZ Pauli Hamiltonian"):
        knm_to_hamiltonian(K, omega)


def test_compile_hamiltonian_inherits_operator_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade compile path inherits the sparse-operator guard."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "0.0000001")
    K, omega = _ring(64)
    problem = build_kuramoto_problem(K, omega)
    with pytest.raises(PauliOperatorBudgetError):
        compile_hamiltonian(problem)


def test_trotter_circuit_inherits_operator_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Trotter compile path fails closed via its Hamiltonian construction."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "0.0000001")
    K, omega = _ring(32)
    problem = build_kuramoto_problem(K, omega)
    with pytest.raises(PauliOperatorBudgetError):
        compile_trotter_circuit(problem, time=1.0, trotter_steps=2)


def test_small_problem_compiles_under_default_budget() -> None:
    """A small problem compiles to an operator and a circuit with no guard trip."""
    K, omega = _ring(4)
    problem = build_kuramoto_problem(K, omega)

    operator = compile_hamiltonian(problem)
    circuit = compile_trotter_circuit(problem, time=0.5, trotter_steps=2)

    assert isinstance(operator, SparsePauliOp)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4


def test_compiler_symmetrises_mildly_asymmetric_direct_input() -> None:
    """The compiler defensively symmetrises a direct asymmetric ``K`` (defence in depth).

    ``KuramotoProblem`` rejects asymmetric ``K`` at construction; the low-level
    compiler is still called directly elsewhere, so it symmetrises rather than
    trusting its caller.
    """
    K = np.array([[0.0, 0.4], [0.2, 0.0]])
    omega = np.zeros(2)
    operator = knm_to_hamiltonian(K, omega)
    assert isinstance(operator, SparsePauliOp)
