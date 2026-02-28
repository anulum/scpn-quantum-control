"""Property-based tests for bridge module converters."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from scpn_quantum_control.bridge import (
    OMEGA_N_16,
    angle_to_probability,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
    probability_to_angle,
)
from scpn_quantum_control.bridge.sc_to_quantum import bitstream_to_statevector

# --- probability <-> angle roundtrip ---


@given(p=st.floats(min_value=0.0, max_value=1.0))
def test_prob_angle_roundtrip(p: float):
    theta = probability_to_angle(p)
    p_back = angle_to_probability(theta)
    assert abs(p_back - p) < 1e-10


@given(theta=st.floats(min_value=0.0, max_value=np.pi))
def test_angle_prob_roundtrip(theta: float):
    p = angle_to_probability(theta)
    theta_back = probability_to_angle(p)
    assert abs(theta_back - theta) < 1e-10


@given(p=st.floats(min_value=0.0, max_value=1.0))
def test_angle_in_valid_range(p: float):
    theta = probability_to_angle(p)
    assert 0.0 <= theta <= np.pi + 1e-12


@given(p=st.floats(min_value=-10.0, max_value=10.0))
def test_probability_to_angle_clamps(p: float):
    assume(np.isfinite(p))
    theta = probability_to_angle(p)
    assert 0.0 <= theta <= np.pi + 1e-12


# --- bitstream_to_statevector ---


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_bitstream_statevector_normalized(p: float):
    bits = np.array([1] * int(p * 100) + [0] * int((1 - p) * 100), dtype=np.uint8)
    if len(bits) == 0:
        bits = np.array([0], dtype=np.uint8)
    sv = bitstream_to_statevector(bits)
    norm = np.sum(sv**2)
    assert abs(norm - 1.0) < 1e-10


# --- Knm matrix properties ---


@given(L=st.integers(min_value=2, max_value=16))
def test_knm_symmetric(L: int):
    K = build_knm_paper27(L=L)
    assert np.allclose(K, K.T)


@given(L=st.integers(min_value=2, max_value=16))
def test_knm_positive(L: int):
    K = build_knm_paper27(L=L)
    assert np.all(K >= 0)


@given(L=st.integers(min_value=2, max_value=16))
def test_knm_diagonal_is_base(L: int):
    K = build_knm_paper27(L=L)
    # diag(K) = K_base * exp(0) = 0.45
    assert np.allclose(np.diag(K), 0.45)


@given(L=st.integers(min_value=2, max_value=16))
def test_knm_shape(L: int):
    K = build_knm_paper27(L=L)
    assert K.shape == (L, L)


# --- Hamiltonian properties ---


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=20)
def test_hamiltonian_hermitian(n: int):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T), "Hamiltonian not Hermitian"


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=20)
def test_hamiltonian_qubit_count(n: int):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == n


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=20)
def test_hamiltonian_real_eigenvalues(n: int):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    eigvals = np.linalg.eigvalsh(mat)
    assert np.all(np.isreal(eigvals))


# --- Ansatz properties ---


@given(n=st.integers(min_value=2, max_value=6), reps=st.integers(min_value=1, max_value=3))
@settings(max_examples=20)
def test_ansatz_qubit_count(n: int, reps: int):
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=reps)
    assert qc.num_qubits == n


@given(n=st.integers(min_value=2, max_value=6), reps=st.integers(min_value=1, max_value=3))
@settings(max_examples=20)
def test_ansatz_param_count(n: int, reps: int):
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=reps)
    assert qc.num_parameters == n * 2 * reps
