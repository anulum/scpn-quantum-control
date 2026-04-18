# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — classical irreproducibility — algebraic invariant
"""Algebraic proof that the DLA-parity Hamiltonian conserves total parity.

The DLA-parity circuit protocol implements

    H = Σ_i ω_i Z_i + Σ_{i,i+1} K_{i,i+1} (X_i X_{i+1} + Y_i Y_{i+1})

decomposed by Lie-Trotter as ``U_step = U_Z · U_XY``. The total-parity
operator is

    P = Π_i Z_i .

Every summand of H commutes with P, so [H, P] = 0 symbolically — not
as a numerical round-off "≈ 0", but as an algebraic identity in the
Pauli algebra. This file proves that identity by explicit
``SparsePauliOp`` manipulation at every n used in the campaign
(n=3, 4, 6), and asserts the operational consequence: a circuit that
starts in a parity eigenstate |ψ₀⟩ produces zero amplitude on every
opposite-parity computational basis state at every depth and every
t_step, up to IEEE-754 round-off only.

This closes the narrow reading of D1 ("Quantum Result Beyond
Classical"): any classical simulator that faithfully implements the
idealised Hamiltonian cannot produce the non-zero parity-leakage
asymmetry observed on IBM hardware. The broad reading (quantum
advantage at scale) is explicitly *not* proved here — see
``docs/classical_irreproducibility.md`` for the distinction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from scpn_quantum_control.dla_parity.baselines import (
    DEFAULT_INITIAL_EVEN,
    DEFAULT_INITIAL_ODD,
    DEFAULT_N_QUBITS,
    DEFAULT_T_STEP,
)


def _hamiltonian_sparse(n: int) -> SparsePauliOp:
    """Build H = Σ ω_i Z_i + Σ K_{i,i+1} (X_i X_{i+1} + Y_i Y_{i+1}) symbolically."""
    omega = np.linspace(0.8, 1.2, n)
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i, j] = 0.45 * math.exp(-0.3 * abs(i - j))

    terms: list[tuple[str, complex]] = []
    for i in range(n):
        label = ["I"] * n
        label[n - 1 - i] = "Z"  # Qiskit bit-ordering is MSB-first in label strings
        terms.append(("".join(label), omega[i]))

    for i in range(n - 1):
        j = i + 1
        coeff = k[i, j]
        for pauli in ("X", "Y"):
            label = ["I"] * n
            label[n - 1 - i] = pauli
            label[n - 1 - j] = pauli
            terms.append(("".join(label), coeff))

    return SparsePauliOp.from_list(terms).simplify()


def _parity_operator(n: int) -> SparsePauliOp:
    """P = Z_0 Z_1 ... Z_{n-1}, the total-parity observable."""
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


def _commutator_sparse(a: SparsePauliOp, b: SparsePauliOp) -> SparsePauliOp:
    """[A, B] = A·B − B·A, simplified to a canonical Pauli-term list."""
    return (a @ b - b @ a).simplify()


def _is_zero_operator(op: SparsePauliOp) -> bool:
    """True iff every coefficient is exactly numerically zero.

    ``simplify`` collapses terms that cancel; what remains is the
    numerically non-zero residue. For a true algebraic identity this
    residue is < machine epsilon and typically 0.0 exactly.
    """
    return bool(np.all(np.abs(op.coeffs) < 1e-12))


def _parity_eigenvalue(bitstring: str) -> int:
    """+1 if the bitstring has even popcount, −1 if odd (parity eigenvalue of P)."""
    return 1 if bitstring.count("1") % 2 == 0 else -1


def _opposite_parity_mask(n: int, initial_parity: int) -> np.ndarray:
    dim = 1 << n
    mask = np.zeros(dim, dtype=np.float64)
    for idx in range(dim):
        if bin(idx).count("1") % 2 != initial_parity:
            mask[idx] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSymbolicParityCommutation:
    """[H, P] = 0 algebraically, across every campaign size and subsystem."""

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_h_commutes_with_p(self, n: int) -> None:
        h = _hamiltonian_sparse(n)
        p = _parity_operator(n)
        comm = _commutator_sparse(h, p)
        assert _is_zero_operator(comm), (
            f"[H, P] has non-zero terms at n={n}: "
            f"{[(pl, c) for pl, c in zip(comm.paulis.to_labels(), comm.coeffs, strict=False) if abs(c) > 1e-12]}"
        )

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_hz_and_hxy_each_commute_with_p(self, n: int) -> None:
        """Both Trotter generators H_Z and H_XY commute with P individually.

        This is the invariant that makes Lie-Trotter exact in the
        parity sector — not just first-order accurate.
        """
        omega = np.linspace(0.8, 1.2, n)
        k = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    k[i, j] = 0.45 * math.exp(-0.3 * abs(i - j))

        hz_terms: list[tuple[str, complex]] = []
        for i in range(n):
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            hz_terms.append(("".join(label), omega[i]))
        hz = SparsePauliOp.from_list(hz_terms).simplify()

        hxy_terms: list[tuple[str, complex]] = []
        for i in range(n - 1):
            j = i + 1
            for pauli in ("X", "Y"):
                label = ["I"] * n
                label[n - 1 - i] = pauli
                label[n - 1 - j] = pauli
                hxy_terms.append(("".join(label), k[i, j]))
        hxy = SparsePauliOp.from_list(hxy_terms).simplify()

        p = _parity_operator(n)
        assert _is_zero_operator(_commutator_sparse(hz, p))
        assert _is_zero_operator(_commutator_sparse(hxy, p))

    @pytest.mark.parametrize("n", [3, 4])
    def test_single_xxyy_pair_commutes_with_p(self, n: int) -> None:
        """Even a single (X_i X_{i+1} + Y_i Y_{i+1}) term commutes with P.

        This is the minimal building block of every Trotter layer and
        also the reason the unitary preserves parity bit-for-bit.
        """
        p = _parity_operator(n)
        for i in range(n - 1):
            j = i + 1
            for pauli in ("X", "Y"):
                label = ["I"] * n
                label[n - 1 - i] = pauli
                label[n - 1 - j] = pauli
                term = SparsePauliOp.from_list([("".join(label), 1.0)])
                assert _is_zero_operator(_commutator_sparse(term, p))


class TestUnitaryPreservesParity:
    """The evolution operator commutes with P at arbitrary finite t and depth."""

    @pytest.mark.parametrize("t", [0.1, DEFAULT_T_STEP, 1.7])
    @pytest.mark.parametrize("n", [3, 4])
    def test_exp_minus_i_h_t_commutes_with_p(self, n: int, t: float) -> None:
        h = _hamiltonian_sparse(n).to_matrix()
        p = _parity_operator(n).to_matrix()
        # exp(-iHt) built via matrix exponential of a Hermitian matrix
        from scipy.linalg import expm

        u = expm(-1j * h * t)
        commutator = u @ p - p @ u
        assert np.max(np.abs(commutator)) < 1e-10, (
            f"||[U(t), P]|| = {np.max(np.abs(commutator)):.3e} at n={n}, t={t}"
        )

    @pytest.mark.parametrize("depth", [1, 4, 30])
    def test_lie_trotter_step_power_commutes_with_p(self, depth: int) -> None:
        n = DEFAULT_N_QUBITS
        omega = np.linspace(0.8, 1.2, n)
        k = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    k[i, j] = 0.45 * math.exp(-0.3 * abs(i - j))

        hz_terms = []
        for i in range(n):
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            hz_terms.append(("".join(label), omega[i]))
        hz = SparsePauliOp.from_list(hz_terms).simplify().to_matrix()

        hxy_terms = []
        for i in range(n - 1):
            j = i + 1
            for pauli in ("X", "Y"):
                label = ["I"] * n
                label[n - 1 - i] = pauli
                label[n - 1 - j] = pauli
                hxy_terms.append(("".join(label), k[i, j]))
        hxy = SparsePauliOp.from_list(hxy_terms).simplify().to_matrix()

        from scipy.linalg import expm

        u_z = expm(-1j * hz * DEFAULT_T_STEP)
        u_xy = expm(-1j * hxy * DEFAULT_T_STEP)
        u_step = u_z @ u_xy
        u = np.linalg.matrix_power(u_step, depth)

        p = _parity_operator(n).to_matrix()
        commutator = u @ p - p @ u
        assert np.max(np.abs(commutator)) < 1e-10


class TestClassicalLeakageIsAlgebraicallyZero:
    """If |ψ₀⟩ is a parity eigenstate, opposite-parity amplitudes vanish for all depths."""

    @pytest.mark.parametrize("depth", [1, 4, 10, 30])
    @pytest.mark.parametrize(
        ("initial", "expected_parity"),
        [(DEFAULT_INITIAL_EVEN, +1), (DEFAULT_INITIAL_ODD, -1)],
    )
    def test_opposite_parity_amplitudes_are_zero(
        self,
        initial: str,
        expected_parity: int,
        depth: int,
    ) -> None:
        n = DEFAULT_N_QUBITS
        assert _parity_eigenvalue(initial) == expected_parity

        h = _hamiltonian_sparse(n).to_matrix()
        from scipy.linalg import expm

        u_total = expm(-1j * h * DEFAULT_T_STEP * depth)

        # |ψ₀⟩ is the computational basis state |initial⟩
        dim = 1 << n
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[int(initial, 2)] = 1.0

        psi_t = u_total @ psi0
        mask = _opposite_parity_mask(n, bin(int(initial, 2)).count("1") % 2)
        opposite_prob = float(np.sum(mask * np.abs(psi_t) ** 2))
        assert opposite_prob < 1e-18, (
            f"Opposite-parity probability is not zero: {opposite_prob:.3e} "
            f"at initial={initial}, depth={depth}"
        )

    def test_statevector_parity_expectation_is_preserved(self) -> None:
        """⟨ψ(t)|P|ψ(t)⟩ = ⟨ψ₀|P|ψ₀⟩ for any depth and initial state."""
        n = DEFAULT_N_QUBITS
        h = _hamiltonian_sparse(n).to_matrix()
        p = _parity_operator(n).to_matrix()

        from scipy.linalg import expm

        for initial in (DEFAULT_INITIAL_EVEN, DEFAULT_INITIAL_ODD):
            dim = 1 << n
            psi0 = np.zeros(dim, dtype=np.complex128)
            psi0[int(initial, 2)] = 1.0
            initial_exp = float(np.real(psi0.conj() @ p @ psi0))
            for depth in (1, 5, 30):
                u = expm(-1j * h * DEFAULT_T_STEP * depth)
                psi_t = u @ psi0
                final_exp = float(np.real(psi_t.conj() @ p @ psi_t))
                assert abs(final_exp - initial_exp) < 1e-12


class TestBackwardReference:
    """The numerical baselines.py matches the algebraic invariant."""

    def test_baselines_leakage_is_zero_because_of_parity_conservation(self) -> None:
        """Cross-reference: the numerical reference agrees with the algebra.

        This ties the dla_parity.baselines module's ``max_abs_leakage``
        to the algebraic proof in this file: both must be zero up to
        float-precision, and the reason is the commutator proved
        symbolically above, not an incidental numerical coincidence.
        """
        from scpn_quantum_control.dla_parity.baselines import (
            compute_classical_leakage_reference,
        )

        ref = compute_classical_leakage_reference(backend="numpy")
        assert ref.max_abs_leakage < 1e-10, (
            f"Numerical reference disagrees with algebraic invariant: "
            f"max|leakage| = {ref.max_abs_leakage:.3e}; [H, P] was proved zero symbolically."
        )


class TestQiskitEvolutionConfirms:
    """Cross-check: Qiskit Operator evolution of the SparsePauliOp agrees."""

    def test_qiskit_statevector_evolution_zero_opposite_parity(self) -> None:
        """Using Qiskit's Statevector / Operator round-trip as an independent stack."""
        n = 4
        h = _hamiltonian_sparse(n)
        # Qiskit ``from_operator`` on a SparsePauliOp needs a dense Operator
        u = Operator(
            (-1j * h.to_matrix() * (DEFAULT_T_STEP * 30)),
        )
        # expm via scipy → ndarray → Operator; avoids re-implementing Trotter
        from scipy.linalg import expm

        u_np = expm(u.data)
        sv0 = Statevector.from_label(DEFAULT_INITIAL_EVEN)
        sv_t_data = u_np @ sv0.data
        mask = _opposite_parity_mask(
            n,
            bin(int(DEFAULT_INITIAL_EVEN, 2)).count("1") % 2,
        )
        opposite_prob = float(np.sum(mask * np.abs(sv_t_data) ** 2))
        assert opposite_prob < 1e-18
