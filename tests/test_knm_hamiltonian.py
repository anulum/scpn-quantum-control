# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Knm Hamiltonian
"""Tests for bridge/knm_hamiltonian.py."""

from collections.abc import Callable
from threading import Event
from time import monotonic
from typing import Literal, cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_ansatz,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
    knm_to_sparse_matrix,
    knm_to_xxz_hamiltonian,
)
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
)

SIZES = [2, 3, 4, 6, 8, 16]


@pytest.mark.parametrize("surface", [knm_to_hamiltonian, knm_to_xxz_hamiltonian])
def test_pauli_export_large_finite_symmetry_preserves_coefficients(
    surface: Callable[[NDArray[np.float64], NDArray[np.float64]], SparsePauliOp],
) -> None:
    """Symmetrisation cannot overflow a representable Pauli coefficient."""
    coupling = np.array([[0.0, 1e308], [1e308, 0.0]])
    original = coupling.copy()
    with np.errstate(over="raise", invalid="raise"):
        actual = surface(coupling, np.zeros(2))
    assert dict(actual.to_list()) == {"XX": -1e308 + 0j, "YY": -1e308 + 0j}
    np.testing.assert_array_equal(coupling, original)


def test_pauli_xxz_export_refuses_overflowed_coefficient() -> None:
    """Finite input multiplication cannot publish a nonfinite Pauli coefficient."""
    coupling = np.array([[0.0, 1e308], [1e308, 0.0]])
    original = coupling.copy()
    with pytest.raises(ValueError, match="XXZ coefficient is not finite"):
        knm_to_xxz_hamiltonian(coupling, np.zeros(2), delta=2.0)
    np.testing.assert_array_equal(coupling, original)


@pytest.mark.parametrize("invalid", ["shape", "complex", "frequency", "coupling", "delta"])
def test_pauli_xxz_export_rejects_invalid_numeric_inputs(invalid: str) -> None:
    """Public Pauli export rejects malformed inputs before constructing an operator."""
    coupling = np.zeros((2, 2))
    frequency = np.zeros(2)
    delta = 0.0
    if invalid == "shape":
        coupling = np.zeros((2, 3))
    elif invalid == "complex":
        coupling = coupling.astype(complex)
    elif invalid == "frequency":
        frequency[0] = np.inf
    elif invalid == "coupling":
        coupling[0, 1] = np.nan
    else:
        delta = np.inf
    with pytest.raises(ValueError, match="Pauli Hamiltonian input"):
        knm_to_xxz_hamiltonian(coupling, frequency, delta)


def test_native_only_dense_export_rejects_nonzero_small_anisotropy() -> None:
    """A requested XY native kernel cannot silently discard a nonzero XXZ term."""
    with pytest.raises(ValueError, match="delta=0 only"):
        knm_to_dense_matrix(np.zeros((2, 2)), np.zeros(2), delta=5e-14, backend="rust")


def test_auto_dense_export_preserves_selected_small_xxz_term() -> None:
    """Canonical sparsity filtering is not replaced by Qiskit's default simplification tolerance."""
    delta = 5e-14
    coupling = np.array([[0.0, 0.5], [0.5, 0.0]])
    expected = np.diag([-0.5 * delta, 0.5 * delta, 0.5 * delta, -0.5 * delta]).astype(complex)
    expected[1, 2] = expected[2, 1] = -1.0
    actual = knm_to_dense_matrix(coupling, np.zeros(2), delta=delta, backend="auto")
    np.testing.assert_array_equal(actual, expected)


def test_python_dense_export_refuses_overflowed_frequency_sum() -> None:
    """Finite input frequencies cannot authorize an infinite dense output."""
    baseline = active_reserved_bytes()
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="invalid numeric data"),
    ):
        knm_to_dense_matrix(np.zeros((2, 2)), np.array([1e308, 1e308]), backend="python")
    assert active_reserved_bytes() == baseline


def test_dense_export_reservation_releases_after_real_success() -> None:
    """A real export's temporary charge does not remain after its scope returns."""
    before = active_reserved_bytes()
    result = knm_to_dense_matrix(np.zeros((2, 2)), np.array([1.0, 2.0]), backend="python")
    np.testing.assert_array_equal(result, np.diag([-3, -1, 1, 3]))
    assert active_reserved_bytes() == before


def test_dense_export_cancellation_and_deadline_preserve_inputs() -> None:
    """No expired/cancelled export allocates or leaves a charged scope behind."""
    before = active_reserved_bytes()
    coupling = np.zeros((2, 2))
    frequencies = np.array([1.0, 2.0])
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        knm_to_dense_matrix(coupling, frequencies, backend="python", cancelled=cancelled)
    with pytest.raises(TimeoutError):
        knm_to_dense_matrix(
            coupling, frequencies, backend="python", deadline_monotonic=monotonic() - 1
        )
    assert active_reserved_bytes() == before
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))
    np.testing.assert_array_equal(frequencies, [1.0, 2.0])


def test_dense_export_refuses_total_buffers_before_dispatch() -> None:
    """An output-sized cap cannot also admit live conversion/intermediate buffers."""
    coupling = np.zeros((2, 2))
    frequencies = np.array([1.0, 2.0])
    output_bytes = 4 * 4 * 16
    with pytest.raises(DenseAllocationError, match="execution memory"):
        knm_to_dense_matrix(
            coupling, frequencies, max_dense_gib=output_bytes / 1024**3, backend="python"
        )
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))
    np.testing.assert_array_equal(frequencies, [1.0, 2.0])


def test_dense_export_refusal_precedes_input_finite_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity and lifecycle refusal cannot materialise input validation masks."""
    coupling = np.zeros((2, 2))
    frequencies = np.array([1.0, 2.0])
    original = (coupling.tobytes(), frequencies.tobytes())
    before = active_reserved_bytes()
    cancelled = Event()
    cancelled.set()
    finite = Mock(wraps=np.isfinite)
    with monkeypatch.context() as patch:
        patch.setattr(np, "isfinite", finite)
        with pytest.raises(DenseAllocationError, match="execution memory"):
            knm_to_dense_matrix(
                coupling, frequencies, max_dense_gib=256 / 1024**3, backend="python"
            )
        with pytest.raises(ExecutionCancelledError):
            knm_to_dense_matrix(coupling, frequencies, backend="python", cancelled=cancelled)
        with pytest.raises(TimeoutError):
            knm_to_dense_matrix(
                coupling, frequencies, backend="python", deadline_monotonic=monotonic() - 1
            )
    assert not any(isinstance(call.args[0], np.ndarray) for call in finite.call_args_list)
    assert (coupling.tobytes(), frequencies.tobytes()) == original
    assert active_reserved_bytes() == before
    actual = knm_to_dense_matrix(coupling, frequencies, backend="python")
    np.testing.assert_array_equal(actual, np.diag([-3, -1, 1, 3]))
    assert active_reserved_bytes() == before


@pytest.mark.parametrize("invalid", ["coupling", "frequency"])
def test_dense_export_invalid_input_releases_admitted_mask_scope(invalid: str) -> None:
    """Invalid admitted data releases the scope and permits a real healthy retry."""
    coupling = np.zeros((2, 2))
    frequencies = np.array([1.0, 2.0])
    if invalid == "coupling":
        coupling[0, 1] = np.nan
    else:
        frequencies[0] = np.inf
    original = (coupling.tobytes(), frequencies.tobytes())
    before = active_reserved_bytes()
    with pytest.raises(ValueError, match="dense Hamiltonian inputs must be finite"):
        knm_to_dense_matrix(coupling, frequencies, backend="python")
    assert (coupling.tobytes(), frequencies.tobytes()) == original
    assert active_reserved_bytes() == before
    actual = knm_to_dense_matrix(np.zeros((2, 2)), np.array([1.0, 2.0]), backend="python")
    np.testing.assert_array_equal(actual, np.diag([-3, -1, 1, 3]))
    assert active_reserved_bytes() == before


def test_explicit_python_dense_export_retains_pauli_convention() -> None:
    """Route a real small dense export and compare an independent Pauli oracle."""
    coupling = np.array([[0.0, 0.25], [0.25, 0.0]])
    frequencies = np.array([1.0, 2.0])
    matrix = knm_to_dense_matrix(coupling, frequencies, backend="python")
    expected = np.array(
        [[-3, 0, 0, 0], [0, -1, -0.5, 0], [0, -0.5, 1, 0], [0, 0, 0, 3]],
        dtype=complex,
    )
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-14)


def test_native_only_export_does_not_substitute_python_for_xxz() -> None:
    """Refuse the unsupported anisotropic native request before any kernel call."""
    with pytest.raises(ValueError, match="native.*delta"):
        knm_to_dense_matrix(np.zeros((2, 2)), np.zeros(2), delta=0.25, backend="rust")


@pytest.mark.parametrize("shape", [(2,), (2, 1), (3, 3)])
def test_dense_ffi_shape_refuses_at_python_boundary(shape: tuple[int, ...]) -> None:
    """A malformed coupling array cannot reach native allocation or fallback."""
    with pytest.raises(ValueError, match="shape"):
        knm_to_dense_matrix(np.zeros(shape), np.zeros(2), backend="rust")


@pytest.mark.parametrize("n", SIZES)
def test_knm_paper27_symmetric(n: int) -> None:
    """Build a symmetric Paper-27 coupling matrix at each supported size."""
    K = build_knm_paper27(L=n)
    assert K.shape == (n, n)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_knm_paper27_cross_hierarchy() -> None:
    """Retain the two canonical cross-hierarchy coupling boosts."""
    K = build_knm_paper27()
    assert K[0, 15] >= 0.05
    assert K[4, 6] >= 0.15


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_hamiltonian_hermitian(n: int) -> None:
    """Compile a Hermitian XY Hamiltonian at representative sizes."""
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_hamiltonian_qubit_count(n: int) -> None:
    """Preserve one Hamiltonian qubit per oscillator."""
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == n


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_ansatz_qubit_count(n: int) -> None:
    """Preserve one ansatz qubit per coupling-matrix row."""
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=1)
    assert qc.num_qubits == n


@pytest.mark.parametrize("n,reps", [(2, 1), (3, 2), (4, 2), (6, 1)])
def test_ansatz_has_parameters(n: int, reps: int) -> None:
    """Allocate two rotation parameters per qubit and repetition."""
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=reps)
    assert qc.num_parameters == n * 2 * reps


def test_pauli_ordering_energy_on_zero_state() -> None:
    """<0...0|H|0...0> must equal -sum(omega) to verify qubit labeling."""
    from qiskit.quantum_info import Statevector

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)

    sv = Statevector.from_int(0, dims=2**n)
    E = float(sv.expectation_value(H).real)
    # H = -sum(omega_i * Z_i) - sum(K_ij * (XX+YY))
    # |0...0>: <Z_i>=+1, <XX>=<YY>=0
    np.testing.assert_allclose(E, -np.sum(omega), atol=1e-12)


def test_knm_omega_shape_mismatch() -> None:
    """Reject frequency vectors that do not match the coupling order."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:3]  # 3 != 4
    with pytest.raises(ValueError, match="rows but omega has"):
        knm_to_hamiltonian(K, omega)


def test_pauli_ordering_single_flip() -> None:
    """Flipping qubit 0 should change energy by +2*omega[0]."""
    from qiskit import QuantumCircuit as QC
    from qiskit.quantum_info import Statevector

    n = 3
    omega = np.array([1.0, 2.0, 3.0])
    K = np.zeros((n, n))  # no coupling → only Z terms
    H = knm_to_hamiltonian(K, omega)

    # |0,0,0>: E = -(1+2+3) = -6
    sv0 = Statevector.from_int(0, dims=2**n)
    E0 = float(sv0.expectation_value(H).real)
    np.testing.assert_allclose(E0, -6.0, atol=1e-12)

    # Flip qubit 0: <Z_0> = -1, others still +1 → E = +1 -2 -3 = -4
    qc = QC(n)
    qc.x(0)
    sv1 = Statevector.from_instruction(qc)
    E1 = float(sv1.expectation_value(H).real)
    np.testing.assert_allclose(E1, -4.0, atol=1e-12)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_kuramoto_ring_symmetric(n: int) -> None:
    """Build a symmetric nearest-neighbour ring without duplicate edges."""
    K, omega = build_kuramoto_ring(n, coupling=0.5, rng_seed=0)
    assert K.shape == (n, n)
    np.testing.assert_allclose(K, K.T, atol=1e-15)
    assert len(omega) == n
    assert np.count_nonzero(K) == 2 * n


@pytest.mark.parametrize("n", [3, 4, 6])
def test_kuramoto_ring_hamiltonian(n: int) -> None:
    """Compile each generated Kuramoto ring into a Hermitian Hamiltonian."""
    K, omega = build_kuramoto_ring(n, coupling=1.0, rng_seed=42)
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == n
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_kuramoto_ring_custom_omega() -> None:
    """Preserve an explicitly supplied natural-frequency vector."""
    omega_in = np.array([1.0, 2.0, 3.0])
    K, omega_out = build_kuramoto_ring(3, omega=omega_in)
    np.testing.assert_array_equal(omega_out, omega_in)


def test_kuramoto_ring_has_no_self_coupling_and_expected_edges() -> None:
    """Mutation guard: ring construction must not add diagonal or extra edges."""
    K, omega = build_kuramoto_ring(5, coupling=0.75, omega=np.arange(5, dtype=float))

    np.testing.assert_allclose(np.diag(K), 0.0)
    assert np.count_nonzero(K) == 10
    assert K[0, 1] == pytest.approx(0.75)
    assert K[0, 4] == pytest.approx(0.75)
    assert K[0, 2] == pytest.approx(0.0)
    np.testing.assert_array_equal(omega, np.arange(5, dtype=float))


def test_ansatz_threshold_includes_equal_weight_edges_only() -> None:
    """Mutation guard: threshold comparison is inclusive at equality."""
    K = np.array(
        [
            [0.0, 0.2, 0.199],
            [0.2, 0.0, 0.5],
            [0.199, 0.5, 0.0],
        ]
    )

    qc = knm_to_ansatz(K, reps=1, threshold=0.2)
    cz_edges = [
        tuple(inst.qubits) for inst in qc.data if getattr(inst.operation, "name", "") == "cz"
    ]

    assert len(cz_edges) == 2


def test_xxz_delta_zero_matches_xy_and_sparse_dense_paths() -> None:
    """Mutation guard: delta=0 path must match the public XY helper exactly."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]

    H_xy = knm_to_hamiltonian(K, omega).to_matrix()
    H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=0.0).to_matrix()
    H_sparse = knm_to_sparse_matrix(K, omega, delta=0.0).toarray()
    H_dense = knm_to_dense_matrix(K, omega, delta=0.0)

    np.testing.assert_allclose(H_xxz, H_xy, atol=1e-12)
    np.testing.assert_allclose(H_sparse, H_xy, atol=1e-12)
    np.testing.assert_allclose(H_dense, H_xy, atol=1e-12)


@pytest.mark.parametrize("invalid", ["omega_rank", "dtype", "delta", "backend"])
def test_dense_export_rejects_malformed_public_dispatch_inputs(invalid: str) -> None:
    """Malformed dense input metadata refuses before either numerical backend runs."""
    coupling = np.zeros((1, 1))
    frequency = np.zeros((1, 1)) if invalid == "omega_rank" else np.zeros(1)
    if invalid == "dtype":
        coupling = np.array([[True]])
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="shape|dtype|finite|backend"):
        knm_to_dense_matrix(
            coupling,
            frequency,
            delta=float("nan") if invalid == "delta" else 0.0,
            backend=cast(
                Literal["auto", "python", "rust"], "missing" if invalid == "backend" else "python"
            ),
        )
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(
        knm_to_dense_matrix(np.zeros((1, 1)), np.array([1.0]), backend="python"),
        np.diag([-1.0, 1.0]),
    )
    assert active_reserved_bytes() == baseline
