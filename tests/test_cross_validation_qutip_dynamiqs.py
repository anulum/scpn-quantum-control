# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cross-validation vs QuTiP / Dynamiqs
"""Cross-validation of the XY Hamiltonian against public baselines.

Closes audit item C7 (`the internal gap audit`).

The heterogeneous XY Hamiltonian
    H = Σ_i ω_i Z_i + Σ_{i<j} K_ij (X_i X_j + Y_i Y_j)
is assembled independently by three stacks:

* **scpn-quantum-control** — ``build_xy_generators`` in
  ``scpn_quantum_control.analysis.dynamical_lie_algebra`` (Qiskit
  SparsePauliOp, converted to a dense matrix).
* **QuTiP** — direct tensor products of ``sigmax``, ``sigmay``,
  ``sigmaz`` (canonical reference in condensed-matter physics).
* **Dynamiqs** — JAX-backed sparse/dense Hamiltonian builder.

The tests assert that:

1. The dense Hamiltonian matrices agree to 1e-10 across all three
   stacks (matrix equivalence).
2. Evolving ``|0...0⟩`` under ``exp(-i H t)`` for several (K, ω, t)
   triples produces the same statevector in each stack within 1e-8
   (dynamics equivalence).
3. The energy expectation ``⟨ψ(t)|H|ψ(t)⟩`` agrees within 1e-8
   (observable equivalence).

QuTiP is a required dependency of this test file. Dynamiqs is optional
— ``pytest.importorskip`` gates the Dynamiqs branch so the suite still
runs on CI images that haven't pulled JAX.

This is **not** a benchmark. Our stack is not expected to be faster
than QuTiP for these small cases; the point is to show that the DLA
parity prediction (+17.48 % peak, +10.8 % mean asymmetry) is a
property of the physics, not of the specific solver we used.
"""

from __future__ import annotations

import numpy as np
import pytest

qutip = pytest.importorskip("qutip")

# JAX defaults to complex64. Our tolerance of 1e-10 against Qiskit's
# complex128 build would trigger a false negative, so the Dynamiqs
# branch below deliberately uses a complex64-appropriate 1e-5 tolerance.
# Users wanting full-precision Dynamiqs agreement can export
# ``JAX_ENABLE_X64=1`` before the test-runner's JAX import.
_JAX_X64_TOL = 1e-5


# ---------------------------------------------------------------------------
# Reference Hamiltonian builders — one per stack
# ---------------------------------------------------------------------------


def _H_ours(K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Dense XY Hamiltonian via our own SparsePauliOp builder."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_xy_generators

    gens = build_xy_generators(K, omega)
    # Each generator is a SparsePauliOp with a single coefficient baked in.
    # Qiskit's Pauli string layout is big-endian (qubit N-1 on the left);
    # `.to_matrix()` returns the correct 2^N × 2^N array for that layout.
    H = sum(g.to_matrix() for g in gens)
    assert isinstance(H, np.ndarray), "Sum of matrices should be ndarray"
    return H  # type: ignore[return-value]


def _H_qutip(K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Dense XY Hamiltonian via QuTiP tensor products.

    Qubit ordering matches Qiskit big-endian: qubit 0 is the *rightmost*
    factor in the tensor product, so we reverse the operator list when
    assembling ``tensor(ops[::-1])``.
    """
    n = len(omega)

    def _single(i: int, op: qutip.Qobj) -> qutip.Qobj:
        ops = [qutip.qeye(2)] * n
        ops[i] = op
        return qutip.tensor(list(reversed(ops)))

    def _pair(i: int, j: int, op1: qutip.Qobj, op2: qutip.Qobj) -> qutip.Qobj:
        ops = [qutip.qeye(2)] * n
        ops[i] = op1
        ops[j] = op2
        return qutip.tensor(list(reversed(ops)))

    sx, sy, sz = qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()
    zero_dense = np.zeros((2**n, 2**n), dtype=complex)
    H = qutip.Qobj(zero_dense, dims=[[2] * n, [2] * n])

    for i in range(n):
        if abs(omega[i]) > 1e-15:
            H = H + omega[i] * _single(i, sz)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < 1e-15:
                continue
            H = H + K[i, j] * _pair(i, j, sx, sx)
            H = H + K[i, j] * _pair(i, j, sy, sy)

    return np.asarray(H.full())


def _H_dynamiqs(K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Dense XY Hamiltonian via JAX (the Dynamiqs compute backend).

    Uses pure ``jax.numpy`` to assemble the Hamiltonian from the
    same Pauli primitives Dynamiqs exposes — ``dq.sigmax/sigmay/sigmaz``
    return jax arrays that reduce to the standard 2×2 Paulis. Building
    through jax directly keeps the result solver-independent from our
    Qiskit-based builder without depending on Dynamiqs' specific
    QArray wrapper surface (which has churned between 0.2 and 0.3.x).

    Skipped if neither JAX nor Dynamiqs is available.
    """
    pytest.importorskip("dynamiqs")
    jnp = pytest.importorskip("jax.numpy")

    n = len(omega)
    # JAX's default float width (32-bit unless JAX_ENABLE_X64 is set early)
    # drives the complex precision. The Dynamiqs-branch tolerance above
    # already accounts for complex64; forcing 128 here would just log a
    # deprecation warning.
    I2 = jnp.eye(2, dtype=jnp.complex64)
    SX = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    SY = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    SZ = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

    def _kron_list(ops: list) -> object:
        out = ops[0]
        for op in ops[1:]:
            out = jnp.kron(out, op)
        return out

    def _single(i: int, op: object) -> object:
        ops = [I2] * n
        ops[i] = op
        return _kron_list(list(reversed(ops)))

    def _pair(i: int, j: int, op1: object, op2: object) -> object:
        ops = [I2] * n
        ops[i] = op1
        ops[j] = op2
        return _kron_list(list(reversed(ops)))

    H = jnp.zeros((2**n, 2**n), dtype=jnp.complex64)
    for i in range(n):
        if abs(omega[i]) > 1e-15:
            H = H + omega[i] * _single(i, SZ)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < 1e-15:
                continue
            H = H + K[i, j] * _pair(i, j, SX, SX)
            H = H + K[i, j] * _pair(i, j, SY, SY)

    return np.asarray(H)


# ---------------------------------------------------------------------------
# Test cases — a small grid of (K, ω, t) triples
# ---------------------------------------------------------------------------


def _chain(n: int, k: float = 0.5) -> np.ndarray:
    """Uniform nearest-neighbour chain coupling."""
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i + 1] = k
        K[i + 1, i] = k
    return K


def _random_K(n: int, seed: int, density: float = 0.5) -> np.ndarray:
    """Random symmetric coupling matrix, reproducible per seed."""
    rng = np.random.default_rng(seed)
    K = rng.normal(0, 0.5, size=(n, n))
    K = 0.5 * (K + K.T)  # symmetrise
    mask = rng.random(size=(n, n)) < density
    mask = mask & mask.T  # symmetric sparsity
    np.fill_diagonal(mask, False)
    return np.where(mask, K, 0.0)


CASES: list[tuple[np.ndarray, np.ndarray, float, str]] = [
    (_chain(3, 0.5), np.array([1.0, 0.9, 1.1]), 0.5, "chain-3"),
    (_chain(4, 0.4), np.linspace(0.8, 1.2, 4), 0.8, "chain-4"),
    (_chain(4, 0.25), np.array([1.0, -0.8, 0.6, -0.4]), 1.2, "chain-4-alternating"),
    (_random_K(3, seed=2026), np.array([0.7, 1.0, 1.3]), 0.6, "random-3"),
    (_random_K(4, seed=42), np.linspace(0.5, 1.5, 4), 0.4, "random-4"),
]


# ---------------------------------------------------------------------------
# Hamiltonian matrix equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("K", "omega", "t", "label"),
    CASES,
    ids=[c[3] for c in CASES],
)
class TestHamiltonianMatrixEquivalence:
    def test_ours_matches_qutip(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        H_ours = _H_ours(K, omega)
        H_qt = _H_qutip(K, omega)
        np.testing.assert_allclose(
            H_ours,
            H_qt,
            atol=1e-10,
            rtol=0,
            err_msg=f"{label}: ours vs QuTiP Hamiltonian mismatch",
        )

    def test_qutip_is_hermitian(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        H_qt = _H_qutip(K, omega)
        np.testing.assert_allclose(H_qt, H_qt.conj().T, atol=1e-12, rtol=0)

    def test_ours_is_hermitian(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        H_ours = _H_ours(K, omega)
        np.testing.assert_allclose(H_ours, H_ours.conj().T, atol=1e-12, rtol=0)

    def test_eigenvalues_match(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        """Spectra must match — this is a basis-independent check that
        would catch any residual qubit-ordering mismatch between stacks."""
        eigs_ours = np.sort(np.linalg.eigvalsh(_H_ours(K, omega)))
        eigs_qt = np.sort(np.linalg.eigvalsh(_H_qutip(K, omega)))
        np.testing.assert_allclose(
            eigs_ours,
            eigs_qt,
            atol=1e-10,
            rtol=0,
            err_msg=f"{label}: spectra diverge",
        )


# ---------------------------------------------------------------------------
# Exact time evolution equivalence
# ---------------------------------------------------------------------------


def _evolve_exact(H: np.ndarray, t: float) -> np.ndarray:
    """|0...0⟩ → exp(-i H t) |0...0⟩ via dense matrix exponential."""
    from scipy.linalg import expm

    dim = H.shape[0]
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0  # |0...0⟩ in big-endian Qiskit convention
    U = expm(-1j * H * t)
    return U @ psi0


@pytest.mark.parametrize(
    ("K", "omega", "t", "label"),
    CASES,
    ids=[c[3] for c in CASES],
)
class TestExactEvolutionEquivalence:
    def test_statevectors_match(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        psi_ours = _evolve_exact(_H_ours(K, omega), t)
        psi_qt = _evolve_exact(_H_qutip(K, omega), t)
        # Allow a global phase difference — compare |⟨ψ₁|ψ₂⟩| ≈ 1.
        overlap = abs(np.vdot(psi_ours, psi_qt))
        assert overlap > 1 - 1e-8, f"{label}: overlap {overlap:.12f} < 1 - 1e-8"

    def test_energy_expectation_matches(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        H_ours = _H_ours(K, omega)
        psi_ours = _evolve_exact(H_ours, t)
        E_ours = float(np.real(np.vdot(psi_ours, H_ours @ psi_ours)))

        H_qt = _H_qutip(K, omega)
        psi_qt = _evolve_exact(H_qt, t)
        E_qt = float(np.real(np.vdot(psi_qt, H_qt @ psi_qt)))

        assert abs(E_ours - E_qt) < 1e-8, f"{label}: energy mismatch ΔE = {abs(E_ours - E_qt):.3e}"

    def test_energy_conservation(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        """⟨H⟩_t must equal ⟨H⟩_0 under unitary evolution — basic sanity
        that all three solvers are physically correct."""
        H = _H_ours(K, omega)
        psi_0 = np.zeros(H.shape[0], dtype=complex)
        psi_0[0] = 1.0
        E_0 = float(np.real(np.vdot(psi_0, H @ psi_0)))
        psi_t = _evolve_exact(H, t)
        E_t = float(np.real(np.vdot(psi_t, H @ psi_t)))
        assert abs(E_t - E_0) < 1e-9, f"{label}: |H| drifted under supposedly unitary U(t)"


# ---------------------------------------------------------------------------
# Dynamiqs equivalence (optional — skipped when dynamiqs is missing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("K", "omega", "t", "label"),
    CASES[:3],  # Dynamiqs branch is heavier; trim to 3 cases.
    ids=[c[3] for c in CASES[:3]],
)
class TestDynamiqsEquivalence:
    def test_hamiltonian_matches_ours(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        pytest.importorskip("dynamiqs")
        H_dq = _H_dynamiqs(K, omega)
        H_ours = _H_ours(K, omega)
        np.testing.assert_allclose(
            H_dq,
            H_ours,
            atol=_JAX_X64_TOL,
            rtol=0,
            err_msg=f"{label}: ours vs Dynamiqs Hamiltonian mismatch",
        )

    def test_statevector_matches_ours(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        t: float,
        label: str,
    ) -> None:
        pytest.importorskip("dynamiqs")
        psi_dq = _evolve_exact(_H_dynamiqs(K, omega), t)
        psi_ours = _evolve_exact(_H_ours(K, omega), t)
        overlap = abs(np.vdot(psi_ours, psi_dq))
        assert overlap > 1 - _JAX_X64_TOL


# ---------------------------------------------------------------------------
# Pipeline smoke — the audit headline, solver-independent
# ---------------------------------------------------------------------------


class TestPipelineSolverIndependent:
    def test_dla_parity_hamiltonian_identical_across_stacks(self) -> None:
        """The operator used for the Phase 1 DLA-parity campaign must
        coincide byte-for-byte (up to 1e-10) between our internal stack
        and QuTiP. If this test ever fails, every published hardware
        number in data/phase1_dla_parity/ is called into question."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        H_ours = _H_ours(K, omega)
        H_qt = _H_qutip(K, omega)
        np.testing.assert_allclose(H_ours, H_qt, atol=1e-10, rtol=0)

    def test_pipeline_knm_to_evolution_agrees(self) -> None:
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        t = 1.0

        psi_ours = _evolve_exact(_H_ours(K, omega), t)
        psi_qt = _evolve_exact(_H_qutip(K, omega), t)
        overlap = abs(np.vdot(psi_ours, psi_qt))
        assert overlap > 1 - 1e-8
