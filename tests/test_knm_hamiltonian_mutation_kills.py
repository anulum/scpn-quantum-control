# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — knm_hamiltonian mutation-killing tests
"""Tests that kill specific real-miss mutants surfaced by mutmut.

The first mutmut baseline against
``src/scpn_quantum_control/bridge/knm_hamiltonian.py`` surfaced 81
survivors, most of which are meaningful behaviour the existing
test suite does not constrain. This file adds targeted assertions
aimed at the most important ones:

* **Sign convention on XX / YY coefficients** — mutants 106 and 115
  flip the ``-K[i, j]`` sign. These would invert the
  Kuramoto-XY mapping. The existing tests only check presence of
  the XX / YY strings, not their coefficients.
* **Off-by-one in pair loop** — mutant 172 rewrites
  ``range(i + 1, n)`` to ``range(i + 2, n)``, dropping every
  nearest-neighbour pair. The resulting Hamiltonian has missing
  coupling terms.
* **None propagation** — mutants 71 (``K[i, j] = None``) and 150
  (``H_op = None``) substitute ``None`` for numeric values; the
  existing tests do not check the return type.
* **Default L value** — mutant 19 changes ``L: int = 16`` to
  ``L: int = 17``, which would return a 17x17 K matrix with
  garbage in the extra row/column. The existing default-argument
  test does not exercise shape.
* **``if L > 15`` threshold** — mutant 45 rewrites to
  ``if L >= 15``, applying the L1-L16 cross-hierarchy boost to
  L = 15 (which has no index 15). Would raise IndexError or
  silently expand.
* **Cross-hierarchy boost index** — mutant 61 rewrites
  ``max(K[4, 6], 0.15)`` to ``max(K[5, 6], 0.15)``; the final
  ``K[4, 6]`` then depends on the wrong base value.
* **OMEGA_N_16 paper-data values** — the 16 frequencies from
  Paper 27 Table 1 are currently unverified against their source.
  Any mutation to an element of the array survives because no
  test asserts the exact values. This file adds that assertion
  with the paper-cited tolerance.
* **Paper 27 Table 2 anchor values** — the ``anchors`` dict is
  similarly unverified. Added: exact-value assertion.

These tests are intentionally tight — they should fail loudly if
the scientific content changes, because those values are paper
references. If the paper itself changes, update both at once.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)


class TestSignConventionOnCouplingTerms:
    """Kill mutants 106, 115: sign flips on XX / YY coefficients."""

    def test_xx_coefficient_is_minus_k(self) -> None:
        # 2-qubit K matrix with one non-zero coupling.
        K = np.array([[0.0, 0.7], [0.7, 0.0]])
        omega = np.array([0.0, 0.0])
        h = knm_to_hamiltonian(K, omega)
        # Find the XX term and assert its coefficient is exactly -K.
        labels = [p.to_label() for p in h.paulis]
        coeffs = h.coeffs
        # SparsePauliOp label strings are MSB-first over qubits.
        xx_index = labels.index("XX")
        assert np.real(coeffs[xx_index]) == pytest.approx(-0.7, abs=1e-12)
        assert abs(np.imag(coeffs[xx_index])) < 1e-12

    def test_yy_coefficient_is_minus_k(self) -> None:
        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        omega = np.array([0.0, 0.0])
        h = knm_to_hamiltonian(K, omega)
        labels = [p.to_label() for p in h.paulis]
        coeffs = h.coeffs
        yy_index = labels.index("YY")
        assert np.real(coeffs[yy_index]) == pytest.approx(-0.5, abs=1e-12)
        assert abs(np.imag(coeffs[yy_index])) < 1e-12

    def test_zz_coefficient_is_minus_k_delta(self) -> None:
        # Non-zero delta so the ZZ term exists.
        K = np.array([[0.0, 0.3], [0.3, 0.0]])
        omega = np.array([0.0, 0.0])
        h = knm_to_xxz_hamiltonian(K, omega, delta=0.9)
        labels = [p.to_label() for p in h.paulis]
        coeffs = h.coeffs
        # XX / YY coefficients are -K; ZZ coefficient is -K * delta.
        zz_index = labels.index("ZZ")
        assert np.real(coeffs[zz_index]) == pytest.approx(-0.3 * 0.9, abs=1e-12)

    def test_z_coefficient_is_minus_omega(self) -> None:
        K = np.zeros((2, 2))  # no coupling, only onsite
        omega = np.array([1.25, -0.7])
        h = knm_to_hamiltonian(K, omega)
        labels = [p.to_label() for p in h.paulis]
        coeffs = h.coeffs
        # ``reversed`` in the compiler means qubit 0 occupies the
        # rightmost label position and qubit 1 the leftmost.
        iz_index = labels.index("IZ")  # Z on qubit 0
        zi_index = labels.index("ZI")  # Z on qubit 1
        assert np.real(coeffs[iz_index]) == pytest.approx(-1.25, abs=1e-12)
        assert np.real(coeffs[zi_index]) == pytest.approx(0.7, abs=1e-12)


class TestLoopIteratesAllPairs:
    """Kill mutant 172: ``range(i + 1, n) -> range(i + 2, n)`` drops nearest-neighbour pairs."""

    def test_n_coupling_pairs_equals_n_choose_2(self) -> None:
        # 4-qubit fully-connected K. Number of (XX + YY) terms = 2 * C(4, 2) = 12.
        n = 4
        K = np.full((n, n), 0.3) - 0.3 * np.eye(n)
        omega = np.zeros(n)
        h = knm_to_hamiltonian(K, omega)
        labels = [p.to_label() for p in h.paulis]
        n_xx = sum(1 for lbl in labels if lbl.count("X") == 2 and lbl.count("I") == n - 2)
        n_yy = sum(1 for lbl in labels if lbl.count("Y") == 2 and lbl.count("I") == n - 2)
        expected = n * (n - 1) // 2
        assert n_xx == expected
        assert n_yy == expected

    def test_includes_nearest_neighbour_0_1(self) -> None:
        # If ``range(i + 1, n)`` were ``range(i + 2, n)``, i=0 would
        # start at j=2, skipping j=1. The (0, 1) pair must be present.
        K = np.array([[0.0, 0.9, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
        omega = np.zeros(3)
        h = knm_to_hamiltonian(K, omega)
        labels = {p.to_label() for p in h.paulis}
        # Qubits 0 and 1 → XX on right two positions → "IXX" / "IYY".
        assert "IXX" in labels
        assert "IYY" in labels


class TestReturnTypesAreNotNone:
    """Kill mutants 71, 150: ``None`` substitution on assignments."""

    def test_build_kuramoto_ring_returns_numpy_arrays_not_none(self) -> None:
        K, omega = build_kuramoto_ring(n=4, coupling=0.5)
        assert isinstance(K, np.ndarray)
        assert isinstance(omega, np.ndarray)
        assert K.shape == (4, 4)
        assert omega.shape == (4,)
        # Every assigned coupling cell is a finite float, not None / NaN.
        assert np.all(np.isfinite(K))

    def test_knm_to_sparse_matrix_returns_sparse_op_not_none(self) -> None:
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_sparse_matrix

        K = np.array([[0.0, 0.4], [0.4, 0.0]])
        omega = np.array([1.0, 1.0])
        mat = knm_to_sparse_matrix(K, omega)
        assert mat is not None
        assert mat.shape == (4, 4)

    def test_knm_to_hamiltonian_returns_sparse_pauli_op(self) -> None:
        K = np.array([[0.0, 0.4], [0.4, 0.0]])
        omega = np.array([1.0, 1.0])
        h = knm_to_hamiltonian(K, omega)
        assert isinstance(h, SparsePauliOp)


class TestBuildKnmPaper27Defaults:
    """Kill mutants 19, 45 and 61 around L-default and L-dependent anchors."""

    def test_default_L_is_16(self) -> None:
        # Kills mutant 19 (``L: int = 16 → L: int = 17``).
        K = build_knm_paper27()
        assert K.shape == (16, 16)

    def test_L_greater_15_applies_L1_L16_boost(self) -> None:
        # Kills mutant 45 (``if L > 15`` → ``if L >= 15``); at L=15 the
        # boost touches K[0, 15] which doesn't exist.
        K_15 = build_knm_paper27(L=15)
        assert K_15.shape == (15, 15)
        # L=16 applies the boost: K[0, 15] >= 0.05.
        K_16 = build_knm_paper27(L=16)
        assert K_16[0, 15] >= 0.05 - 1e-12
        assert K_16[15, 0] >= 0.05 - 1e-12

    def test_l5_l7_boost_uses_correct_indices(self) -> None:
        # Kills mutant 61 (``max(K[4, 6], 0.15) → max(K[5, 6], 0.15)``).
        # After the boost, K[4, 6] must be at least 0.15 AND equal to
        # K[6, 4] (symmetric). If the mutation fires, K[4, 6] is
        # seeded from K[5, 6] (the i=5 row) which is a different
        # base-exponential value.
        K = build_knm_paper27(L=8)
        assert K[4, 6] >= 0.15 - 1e-12
        assert K[4, 6] == pytest.approx(K[6, 4], abs=1e-12)

    def test_anchor_values_exact(self) -> None:
        # Kill the 8 mutants that target the anchor-dict keys / values.
        K = build_knm_paper27(L=16)
        assert K[0, 1] == pytest.approx(0.302, abs=1e-12)
        assert K[1, 0] == pytest.approx(0.302, abs=1e-12)
        assert K[1, 2] == pytest.approx(0.201, abs=1e-12)
        assert K[2, 3] == pytest.approx(0.252, abs=1e-12)
        assert K[3, 4] == pytest.approx(0.154, abs=1e-12)


class TestOmega16ExactPaperValues:
    """Kill mutants 2-17: OMEGA_N_16 element-value mutations.

    The array is Paper 27 Table 1 verbatim. If the scientific source
    changes, this assertion blocks the commit until the paper
    citation is updated — exactly the gate a paper-data array
    deserves.
    """

    def test_exact_values_match_paper27_table_1(self) -> None:
        expected = np.array(
            [
                1.329,
                2.610,
                0.844,
                1.520,
                0.710,
                3.780,
                1.055,
                0.625,
                2.210,
                1.740,
                0.480,
                3.210,
                0.915,
                1.410,
                2.830,
                0.991,
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(OMEGA_N_16, expected)

    def test_shape_is_16(self) -> None:
        assert OMEGA_N_16.shape == (16,)

    def test_dtype_is_float64(self) -> None:
        assert OMEGA_N_16.dtype == np.float64
