# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Dla Parity Theorem
"""Tests for the DLA parity theorem."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.dla_parity_theorem import (
    DLAParityTheoremResult,
    decompose_state_by_parity,
    parity_operator,
    parity_sector_dimensions,
    predicted_dla_dimension,
    project_to_parity_sector,
    su_dimension,
    verify_all_known,
    verify_theorem,
)


class TestPredictedDimension:
    """Verify the closed-form DLA dimension across representative sizes."""

    def test_n2(self) -> None:
        """Return six generators for the two-qubit parity sectors."""
        assert predicted_dla_dimension(2) == 6

    def test_n3(self) -> None:
        """Return thirty generators for the three-qubit parity sectors."""
        assert predicted_dla_dimension(3) == 30

    def test_n4(self) -> None:
        """Return 126 generators for the four-qubit parity sectors."""
        assert predicted_dla_dimension(4) == 126

    def test_n5(self) -> None:
        """Return 510 generators for the five-qubit parity sectors."""
        assert predicted_dla_dimension(5) == 510

    def test_n6(self) -> None:
        """Return 2,046 generators for the six-qubit parity sectors."""
        assert predicted_dla_dimension(6) == 2046

    def test_formula(self) -> None:
        """Match the closed-form expression over qubit counts two through seven."""
        for n in range(2, 8):
            assert predicted_dla_dimension(n) == 2 ** (2 * n - 1) - 2


class TestParitySectorDimensions:
    """Verify equal even and odd Hilbert-space partitions."""

    def test_n2(self) -> None:
        """Split the two-qubit Hilbert space into two-dimensional sectors."""
        d_even, d_odd = parity_sector_dimensions(2)
        assert d_even == 2
        assert d_odd == 2

    def test_n4(self) -> None:
        """Split the four-qubit Hilbert space into eight-dimensional sectors."""
        d_even, d_odd = parity_sector_dimensions(4)
        assert d_even == 8
        assert d_odd == 8

    def test_sum_equals_hilbert(self) -> None:
        """Reconstruct the complete Hilbert dimension from both sectors."""
        for n in range(2, 8):
            d_e, d_o = parity_sector_dimensions(n)
            assert d_e + d_o == 2**n


class TestSUDimension:
    """Verify the special-unitary algebra dimension formula."""

    def test_su2(self) -> None:
        """Return three generators for su(2)."""
        assert su_dimension(2) == 3

    def test_su4(self) -> None:
        """Return fifteen generators for su(4)."""
        assert su_dimension(4) == 15

    def test_su8(self) -> None:
        """Return sixty-three generators for su(8)."""
        assert su_dimension(8) == 63


class TestDLADecomposition:
    """Verify DLA = su(even) ⊕ su(odd) = 2 × su(2^(N-1))."""

    def test_decomposition_matches_formula(self) -> None:
        """Match the sector decomposition to the closed-form DLA dimension."""
        for n in range(2, 8):
            d_e, d_o = parity_sector_dimensions(n)
            dim_from_decomp = su_dimension(d_e) + su_dimension(d_o)
            dim_from_formula = predicted_dla_dimension(n)
            assert dim_from_decomp == dim_from_formula, (
                f"N={n}: decomposition gives {dim_from_decomp}, formula gives {dim_from_formula}"
            )


class TestVerifyTheorem:
    """Verify computed dimensions against the theorem result record."""

    def test_n4_matches(self) -> None:
        """Admit the verified four-qubit dimension and sector algebras."""
        result = verify_theorem(4, 126)
        assert isinstance(result, DLAParityTheoremResult)
        assert result.matches
        assert result.predicted_dim == 126
        assert result.su_even_expected == 63
        assert result.su_odd_expected == 63

    def test_n5_matches(self) -> None:
        """Admit the verified five-qubit dimension."""
        result = verify_theorem(5, 510)
        assert result.matches

    def test_wrong_dim_fails(self) -> None:
        """Reject a computed dimension that differs from the theorem."""
        result = verify_theorem(4, 100)
        assert not result.matches


class TestVerifyAllKnown:
    """Verify the complete set of computationally confirmed dimensions."""

    def test_all_match(self) -> None:
        """Match every stored two-through-five-qubit dimension."""
        results = verify_all_known()
        assert len(results) == 4  # N=2,3,4,5
        for r in results:
            assert r.matches, f"N={r.n_qubits}: expected {r.predicted_dim}, got {r.computed_dim}"


class TestParityOperator:
    """Verify parity-operator spectral and algebraic invariants."""

    def test_n2_eigenvalues(self) -> None:
        """Produce balanced positive and negative two-qubit eigenvalues."""
        P = parity_operator(2)
        eigenvalues = np.sort(np.linalg.eigvalsh(P))
        # |00⟩, |11⟩ → even (+1); |01⟩, |10⟩ → odd (-1)
        np.testing.assert_array_almost_equal(eigenvalues, [-1, -1, 1, 1])

    def test_n3_trace(self) -> None:
        """Cancel equal even and odd multiplicities in the three-qubit trace."""
        P = parity_operator(3)
        # 4 even states, 4 odd → trace = 4 - 4 = 0
        assert np.trace(P) == pytest.approx(0.0)

    def test_involution(self) -> None:
        """Square the parity operator to the identity."""
        P = parity_operator(3)
        np.testing.assert_array_almost_equal(P @ P, np.eye(8))

    def test_diagonal(self) -> None:
        """Keep the computational-basis parity operator diagonal."""
        P = parity_operator(4)
        assert np.allclose(P, np.diag(np.diag(P)))

    def test_diagonal_matches_popcount_ordering(self) -> None:
        """Mutation guard: basis ordering must follow (-1)^popcount(i)."""
        P = parity_operator(4)
        expected = np.array([1.0 - 2.0 * (bin(i).count("1") % 2) for i in range(16)])

        np.testing.assert_array_equal(np.diag(P), expected)


class TestProjectToParity:
    """Verify complementary even and odd state projections."""

    def test_even_projection(self) -> None:
        """Preserve an even computational-basis state."""
        # |00⟩ is even parity
        state = np.array([1, 0, 0, 0], dtype=complex)
        proj = project_to_parity_sector(state, 0, 2)
        np.testing.assert_array_almost_equal(proj, [1, 0, 0, 0])

    def test_odd_projection(self) -> None:
        """Preserve an odd computational-basis state."""
        # |01⟩ is odd parity
        state = np.array([0, 1, 0, 0], dtype=complex)
        proj = project_to_parity_sector(state, 1, 2)
        np.testing.assert_array_almost_equal(proj, [0, 1, 0, 0])

    def test_mixed_state(self) -> None:
        """Split a mixed-parity superposition into equal-norm components."""
        # (|00⟩ + |01⟩)/√2 → even part = |00⟩/√2, odd part = |01⟩/√2
        state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        even = project_to_parity_sector(state, 0, 2)
        odd = project_to_parity_sector(state, 1, 2)
        assert np.linalg.norm(even) == pytest.approx(1 / np.sqrt(2))
        assert np.linalg.norm(odd) == pytest.approx(1 / np.sqrt(2))

    def test_even_and_odd_projections_are_orthogonal_and_reconstruct(self) -> None:
        """Mutation guard: parity masks must be complementary projectors."""
        rng = np.random.default_rng(7)
        state = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        even = project_to_parity_sector(state, 0, 4)
        odd = project_to_parity_sector(state, 1, 4)

        np.testing.assert_allclose(np.vdot(even, odd), 0.0, atol=1e-14)
        np.testing.assert_allclose(even + odd, state, atol=1e-14)


class TestDecomposeByParity:
    """Verify parity weights returned by state decomposition."""

    def test_pure_even(self) -> None:
        """Assign unit weight to a pure even state."""
        state = np.array([1, 0, 0, 0], dtype=complex)
        result = decompose_state_by_parity(state, 2)
        assert result["even_weight"] == pytest.approx(1.0)
        assert result["odd_weight"] == pytest.approx(0.0)

    def test_equal_superposition(self) -> None:
        """Assign equal weights to an even-odd superposition."""
        state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        result = decompose_state_by_parity(state, 2)
        assert result["even_weight"] == pytest.approx(0.5)
        assert result["odd_weight"] == pytest.approx(0.5)

    def test_weights_sum_to_one(self) -> None:
        """Conserve total weight for a normalised random state."""
        rng = np.random.default_rng(42)
        state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        state /= np.linalg.norm(state)
        result = decompose_state_by_parity(state, 3)
        total = result["even_weight"] + result["odd_weight"]
        assert total == pytest.approx(1.0)

    def test_unnormalised_weights_sum_to_state_norm_squared(self) -> None:
        """Mutation guard: decomposition must not silently normalise weights."""
        state = np.array([2 + 0j, 1j, 3 + 0j, -4j], dtype=complex)
        result = decompose_state_by_parity(state, 2)

        total = result["even_weight"] + result["odd_weight"]
        assert total == pytest.approx(float(np.vdot(state, state).real))
        assert result["parity_imbalance"] == pytest.approx(
            abs(result["even_weight"] - result["odd_weight"])
        )
