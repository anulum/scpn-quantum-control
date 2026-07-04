# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Explosive-synchronisation continuation and hysteresis tests
"""Multi-angle tests for the Kuramoto explosive-synchronisation continuation.

Covers the explosive hysteresis of the triadic (2-simplex) model (a finite-width loop, a coherent
branch collapsing at a backward saddle-node, a forward branch that stays incoherent across the
range), the continuous transition of the pairwise model through the same machinery (coincident
branches, zero loop width), the continuation-branch bookkeeping (direction, terminal phases,
reproducibility) and the input validation of both the sweep and the loop builders.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel import (
    ContinuationBranch,
    HysteresisLoop,
    MeanFieldForce,
    continuation_sweep,
    hysteresis_loop,
    mean_field_force,
    triadic_hysteresis_loop,
)


def _cauchy_frequencies(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega = np.tan(np.pi * (rng.uniform(size=n) - 0.5))  # unit half-width Lorentzian, K_c = 2
    return omega - float(np.mean(omega))


def _incoherent_phases(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=n)


_GRID = np.linspace(1.0, 8.0, 12)
_N = 64


@pytest.fixture(scope="module")
def triadic_loop() -> HysteresisLoop:
    omega = _cauchy_frequencies(_N, 11)
    return triadic_hysteresis_loop(
        omega, _GRID, _incoherent_phases(_N, 3), np.zeros(_N), dt=0.05, n_steps=500
    )


@pytest.fixture(scope="module")
def pairwise_loop() -> HysteresisLoop:
    omega = _cauchy_frequencies(_N, 11)
    return hysteresis_loop(
        omega,
        mean_field_force,
        _GRID,
        _incoherent_phases(_N, 3),
        np.zeros(_N),
        dt=0.05,
        n_steps=500,
        separation_tolerance=0.15,
    )


class TestTriadicExplosiveHysteresis:
    def test_loop_is_hysteretic_with_finite_width(self, triadic_loop: HysteresisLoop) -> None:
        # The r² gain leaves a bistable window: the branches separate over a finite coupling range.
        assert triadic_loop.is_hysteretic
        assert triadic_loop.hysteresis_width > 1.0
        assert triadic_loop.max_branch_separation > 0.5

    def test_coherent_branch_collapses_at_a_backward_saddle_node(
        self, triadic_loop: HysteresisLoop
    ) -> None:
        # Seeded coherent at the top coupling, the backward branch stays synchronised then collapses
        # at a finite saddle-node inside the swept range.
        assert triadic_loop.backward_transition_coupling is not None
        assert _GRID[0] < triadic_loop.backward_transition_coupling < _GRID[-1]
        assert triadic_loop.backward.order_parameters[0] > 0.6  # coherent at the highest coupling
        realigned_backward = triadic_loop.backward.order_parameters[::-1]
        assert realigned_backward[0] < 0.3  # collapsed at the lowest coupling

    def test_forward_branch_stays_incoherent_in_range(self, triadic_loop: HysteresisLoop) -> None:
        # The incoherent state is linearly stable for the triadic gain, so the forward branch never
        # synchronises within the range and the forward transition lies beyond the grid.
        assert triadic_loop.forward_transition_coupling is None
        assert float(triadic_loop.forward.order_parameters.max()) < 0.4

    def test_separation_window_matches_reported_width(self, triadic_loop: HysteresisLoop) -> None:
        separated = np.flatnonzero(triadic_loop.branch_separation > 0.1)
        expected = float(_GRID[separated[-1]] - _GRID[separated[0]])
        assert triadic_loop.hysteresis_width == pytest.approx(expected)


class TestPairwiseContinuousTransition:
    def test_branches_coincide_with_no_hysteresis(self, pairwise_loop: HysteresisLoop) -> None:
        # The pairwise mean field has a continuous transition: forward and backward branches overlap.
        assert not pairwise_loop.is_hysteretic
        assert pairwise_loop.hysteresis_width == 0.0
        assert pairwise_loop.max_branch_separation < 0.15

    def test_forward_and_backward_transitions_match(self, pairwise_loop: HysteresisLoop) -> None:
        assert pairwise_loop.forward_transition_coupling is not None
        assert pairwise_loop.backward_transition_coupling is not None
        assert pairwise_loop.forward_transition_coupling == pytest.approx(
            pairwise_loop.backward_transition_coupling
        )

    def test_forward_branch_synchronises_above_critical(
        self, pairwise_loop: HysteresisLoop
    ) -> None:
        # Above the critical coupling K_c = 2 the order parameter rises into coherence.
        assert float(pairwise_loop.forward.order_parameters[-1]) > 0.6


class TestContinuationBranch:
    def test_ascending_sweep_metadata(self) -> None:
        omega = _cauchy_frequencies(_N, 5)
        branch = continuation_sweep(
            omega, mean_field_force, np.linspace(1.0, 6.0, 6), np.zeros(_N), dt=0.05, n_steps=300
        )
        assert isinstance(branch, ContinuationBranch)
        assert branch.direction == "ascending"
        assert branch.is_ascending
        assert branch.terminal_phases.shape == (6, _N)
        assert branch.order_parameters.shape == (6,)
        assert np.all(branch.order_parameters >= 0.0)
        assert np.all(branch.order_parameters <= 1.0)

    def test_descending_sweep_metadata(self) -> None:
        omega = _cauchy_frequencies(_N, 5)
        branch = continuation_sweep(
            omega, mean_field_force, np.linspace(6.0, 1.0, 6), np.zeros(_N), dt=0.05, n_steps=300
        )
        assert branch.direction == "descending"
        assert not branch.is_ascending

    def test_default_settle_window_matches_explicit_quarter(self) -> None:
        omega = _cauchy_frequencies(_N, 5)
        grid = np.linspace(1.0, 6.0, 4)
        seed = np.zeros(_N)
        default = continuation_sweep(omega, mean_field_force, grid, seed, dt=0.05, n_steps=400)
        explicit = continuation_sweep(
            omega, mean_field_force, grid, seed, dt=0.05, n_steps=400, settle_steps=100
        )
        np.testing.assert_allclose(default.order_parameters, explicit.order_parameters)

    def test_loop_is_reproducible(self) -> None:
        omega = _cauchy_frequencies(48, 1)
        grid = np.linspace(1.0, 6.0, 5)
        incoherent = _incoherent_phases(48, 2)
        coherent = np.zeros(48)
        first = triadic_hysteresis_loop(omega, grid, incoherent, coherent, dt=0.05, n_steps=300)
        second = triadic_hysteresis_loop(omega, grid, incoherent, coherent, dt=0.05, n_steps=300)
        np.testing.assert_array_equal(
            first.forward.order_parameters, second.forward.order_parameters
        )
        np.testing.assert_array_equal(
            first.backward.order_parameters, second.backward.order_parameters
        )


class TestSweepValidation:
    _OMEGA = np.zeros(3)
    _GRID = np.array([1.0, 2.0])
    _PHASES = np.zeros(3)

    def test_omega_must_be_one_dimensional(self) -> None:
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            continuation_sweep(
                np.zeros((3, 1)), mean_field_force, self._GRID, self._PHASES, dt=0.1, n_steps=4
            )

    def test_omega_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            continuation_sweep(
                np.zeros(0), mean_field_force, self._GRID, np.zeros(0), dt=0.1, n_steps=4
            )

    def test_initial_phases_shape(self) -> None:
        with pytest.raises(ValueError, match="initial_phases must have shape"):
            continuation_sweep(
                self._OMEGA, mean_field_force, self._GRID, np.zeros(2), dt=0.1, n_steps=4
            )

    def test_coupling_grid_needs_two_points(self) -> None:
        with pytest.raises(ValueError, match="at least two couplings"):
            continuation_sweep(
                self._OMEGA, mean_field_force, np.array([1.0]), self._PHASES, dt=0.1, n_steps=4
            )

    def test_coupling_grid_must_be_monotonic(self) -> None:
        with pytest.raises(ValueError, match="strictly monotonic"):
            continuation_sweep(
                self._OMEGA,
                mean_field_force,
                np.array([1.0, 2.0, 1.5]),
                self._PHASES,
                dt=0.1,
                n_steps=4,
            )

    def test_dt_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            continuation_sweep(
                self._OMEGA, mean_field_force, self._GRID, self._PHASES, dt=0.0, n_steps=4
            )

    def test_n_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be positive"):
            continuation_sweep(
                self._OMEGA, mean_field_force, self._GRID, self._PHASES, dt=0.1, n_steps=0
            )

    def test_settle_steps_range(self) -> None:
        with pytest.raises(ValueError, match=r"settle_steps must be in \[1, 4\]"):
            continuation_sweep(
                self._OMEGA,
                mean_field_force,
                self._GRID,
                self._PHASES,
                dt=0.1,
                n_steps=4,
                settle_steps=0,
            )
        with pytest.raises(ValueError, match=r"settle_steps must be in \[1, 4\]"):
            continuation_sweep(
                self._OMEGA,
                mean_field_force,
                self._GRID,
                self._PHASES,
                dt=0.1,
                n_steps=4,
                settle_steps=5,
            )


class TestLoopValidation:
    _OMEGA = np.zeros(3)
    _PHASES = np.zeros(3)

    def test_coupling_grid_must_be_strictly_ascending(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            hysteresis_loop(
                self._OMEGA,
                mean_field_force,
                np.array([2.0, 1.0]),
                self._PHASES,
                self._PHASES,
                dt=0.1,
                n_steps=4,
            )

    def test_coupling_grid_must_have_two_points(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            hysteresis_loop(
                self._OMEGA,
                mean_field_force,
                np.array([1.0]),
                self._PHASES,
                self._PHASES,
                dt=0.1,
                n_steps=4,
            )

    def test_separation_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="separation_tolerance must be positive"):
            hysteresis_loop(
                self._OMEGA,
                mean_field_force,
                np.array([1.0, 2.0]),
                self._PHASES,
                self._PHASES,
                dt=0.1,
                n_steps=4,
                separation_tolerance=0.0,
            )

    def test_delegates_sweep_validation(self) -> None:
        with pytest.raises(ValueError, match="initial_phases must have shape"):
            hysteresis_loop(
                self._OMEGA,
                mean_field_force,
                np.array([1.0, 2.0]),
                np.zeros(2),
                self._PHASES,
                dt=0.1,
                n_steps=4,
            )


def test_mean_field_force_type_alias_is_exported() -> None:
    assert MeanFieldForce is not None
