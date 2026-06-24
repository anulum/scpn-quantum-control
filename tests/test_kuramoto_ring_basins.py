# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto ring basin-of-attraction tests
"""Multi-angle tests for the Kuramoto-ring basins of attraction.

Covers the ring coupling matrix (symmetry, zero diagonal, row sum, neighbour band), the twisted
states and their winding numbers, the closed-form twisted-state eigenvalues cross-checked against
the dense networked Jacobian, the nearest-neighbour stability law ``|q| < N/4``, the gradient-flow
relaxation (monotone interaction-energy descent to a stable twisted state), and the Monte-Carlo
basin estimate (synchronised basin below one for short range and filling to one at long range, the
fractions normalised, the winding distribution symmetric) together with the input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.accel import (
    BasinEstimate,
    estimate_ring_basins,
    is_twisted_state_stable,
    kuramoto_interaction_energy,
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
    ring_coupling_matrix,
    twisted_state,
    twisted_state_eigenvalues,
    winding_number,
)


class TestRingCouplingMatrix:
    def test_nearest_neighbour_structure(self) -> None:
        matrix = ring_coupling_matrix(6, 1, coupling=2.0)
        assert matrix.shape == (6, 6)
        np.testing.assert_allclose(matrix, matrix.T)  # symmetric
        assert np.all(np.diagonal(matrix) == 0.0)  # no self-coupling
        np.testing.assert_allclose(matrix.sum(axis=1), 2.0)  # row sum equals K
        # each node couples only to its two neighbours, weight K/(2R) = 1.0
        assert matrix[0, 1] == pytest.approx(1.0)
        assert matrix[0, 5] == pytest.approx(1.0)
        assert matrix[0, 2] == 0.0

    def test_range_two_band(self) -> None:
        matrix = ring_coupling_matrix(8, 2, coupling=4.0)
        np.testing.assert_allclose(matrix.sum(axis=1), 4.0)
        assert matrix[0, 1] == pytest.approx(1.0)  # K/(2R) = 4/4
        assert matrix[0, 2] == pytest.approx(1.0)
        assert matrix[0, 3] == 0.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n must be at least 3"):
            ring_coupling_matrix(2, 1)
        with pytest.raises(ValueError, match=r"coupling_range must be in \[1, 3\]"):
            ring_coupling_matrix(7, 4)
        with pytest.raises(ValueError, match=r"coupling_range must be in \[1, 3\]"):
            ring_coupling_matrix(7, 0)
        with pytest.raises(ValueError, match="coupling must be non-zero"):
            ring_coupling_matrix(6, 1, coupling=0.0)


class TestTwistedStateAndWinding:
    def test_twisted_state_values(self) -> None:
        state = twisted_state(4, 1)
        np.testing.assert_allclose(state, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2])

    def test_winding_round_trip(self) -> None:
        # Within |q| <= N/2 the winding number recovers the twist that built the state.
        for q in range(-3, 4):
            assert winding_number(twisted_state(16, q)) == q

    def test_synchronised_state_has_zero_winding(self) -> None:
        assert winding_number(np.zeros(10)) == 0

    def test_twisted_state_validation(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            twisted_state(0, 1)

    def test_winding_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            winding_number(np.zeros((3, 2)))
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            winding_number(np.zeros(0))


class TestTwistedStateEigenvalues:
    def test_goldstone_mode_is_zero(self) -> None:
        eigenvalues = twisted_state_eigenvalues(12, 2, 1)
        assert eigenvalues[0] == pytest.approx(0.0, abs=1e-12)

    def test_match_dense_jacobian_spectrum(self) -> None:
        # The closed-form eigenvalues must equal the spectrum of the actual networked Jacobian at the
        # twisted state, for nearest-neighbour and longer-range coupling alike.
        for coupling_range, winding in ((1, 1), (2, 1), (3, 2)):
            n = 14
            matrix = ring_coupling_matrix(n, coupling_range, coupling=1.3)
            jacobian = networked_kuramoto_jacobian(twisted_state(n, winding), matrix)
            dense_spectrum = np.sort(np.linalg.eigvals(jacobian).real)
            closed_form = np.sort(
                twisted_state_eigenvalues(n, winding, coupling_range, coupling=1.3)
            )
            np.testing.assert_allclose(closed_form, dense_spectrum, atol=1e-10)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n must be at least 3"):
            twisted_state_eigenvalues(2, 0, 1)
        with pytest.raises(ValueError, match=r"coupling_range must be in \[1, 3\]"):
            twisted_state_eigenvalues(7, 0, 5)


class TestTwistedStateStability:
    def test_nearest_neighbour_quarter_law(self) -> None:
        # q-twisted states are stable iff |q| < N/4 for nearest-neighbour coupling.
        n = 16  # N/4 = 4
        stable = [q for q in range(n) if is_twisted_state_stable(n, q, 1)]
        assert stable == [0, 1, 2, 3, 13, 14, 15]  # |q| < 4 with q and N-q both counted

    def test_marginal_state_is_not_stable(self) -> None:
        # q = N/4 gives cos(2πq/N) = 0, a marginal (non-asymptotic) state.
        assert not is_twisted_state_stable(16, 4, 1)

    def test_long_range_only_sync_is_stable(self) -> None:
        n = 16
        stable = [q for q in range(n) if is_twisted_state_stable(n, q, 7)]
        assert stable == [0]


class TestGradientFlowRelaxation:
    def test_interaction_energy_decreases_to_a_stable_twisted_state(self) -> None:
        # The symmetric ring is a gradient flow of the interaction energy, so relaxation lowers the
        # energy monotonically and settles on a stable twisted state.
        n = 16
        matrix = ring_coupling_matrix(n, 1)
        rng = np.random.default_rng(4)
        phases = rng.uniform(-math.pi, math.pi, size=n)
        energies = [kuramoto_interaction_energy(phases, matrix)]
        for _ in range(4000):
            force = networked_kuramoto_force(phases, matrix)
            if float(np.max(np.abs(force))) < 1e-7:
                break
            k2 = networked_kuramoto_force(phases + 0.05 * force, matrix)
            k3 = networked_kuramoto_force(phases + 0.05 * k2, matrix)
            k4 = networked_kuramoto_force(phases + 0.1 * k3, matrix)
            phases = phases + (0.1 / 6.0) * (force + 2.0 * k2 + 2.0 * k3 + k4)
            energies.append(kuramoto_interaction_energy(phases, matrix))
        assert np.all(np.diff(energies) <= 1e-9)  # monotone descent
        assert is_twisted_state_stable(n, winding_number(phases), 1)


class TestBasinEstimate:
    def test_short_range_basin_is_mixed_and_normalised(self) -> None:
        estimate = estimate_ring_basins(16, 1, 60, seed=0)
        assert isinstance(estimate, BasinEstimate)
        assert estimate.n_converged > 0
        assert 0.0 < estimate.sync_basin_fraction < 1.0  # twisted states capture part of the basin
        assert estimate.converged_fraction <= 1.0
        total = sum(estimate.basin_fraction(int(q)) for q in estimate.winding_values)
        assert total == pytest.approx(1.0)
        # every winding reached is a stable twisted state
        assert all(is_twisted_state_stable(16, int(q), 1) for q in estimate.winding_values)
        # the ring is symmetric, so the mean winding is close to zero
        assert abs(estimate.mean_winding) < 0.5

    def test_long_range_basin_is_all_synchronised(self) -> None:
        estimate = estimate_ring_basins(16, 7, 40, seed=1)
        assert estimate.sync_basin_fraction == pytest.approx(1.0)
        assert estimate.winding_values.tolist() == [0]

    def test_reproducible_for_a_fixed_seed(self) -> None:
        first = estimate_ring_basins(14, 1, 30, seed=2)
        second = estimate_ring_basins(14, 1, 30, seed=2)
        np.testing.assert_array_equal(first.winding_values, second.winding_values)
        np.testing.assert_array_equal(first.winding_counts, second.winding_counts)

    def test_absent_winding_has_zero_fraction(self) -> None:
        estimate = estimate_ring_basins(16, 7, 20, seed=3)
        assert estimate.basin_fraction(5) == 0.0  # never reached

    def test_no_convergence_yields_empty_tally(self) -> None:
        # A single step with an unreachable tolerance leaves every sample unconverged.
        estimate = estimate_ring_basins(16, 1, 4, seed=5, max_steps=1, force_tolerance=1e-12)
        assert estimate.n_converged == 0
        assert estimate.converged_fraction == 0.0
        assert estimate.sync_basin_fraction == 0.0
        assert estimate.mean_winding == 0.0
        assert estimate.basin_fraction(0) == 0.0
        assert estimate.winding_values.size == 0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            estimate_ring_basins(6, 1, 0, seed=0)
        with pytest.raises(ValueError, match="dt must be positive"):
            estimate_ring_basins(6, 1, 4, dt=0.0, seed=0)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            estimate_ring_basins(6, 1, 4, max_steps=0, seed=0)
        with pytest.raises(ValueError, match="force_tolerance must be positive"):
            estimate_ring_basins(6, 1, 4, force_tolerance=0.0, seed=0)
