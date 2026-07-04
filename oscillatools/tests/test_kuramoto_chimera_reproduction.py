# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Dynamical reproduction of a two-population Kuramoto–Sakaguchi chimera
r"""Reproduce a chimera state from the two-population Kuramoto–Sakaguchi dynamics.

``tests/test_kuramoto_chimera.py`` checks the chimera diagnostics against **hand-built** phase
arrays — a synthetic state assembled to be chimeric — so it never shows that the dynamics themselves
produce a chimera. This module supplies that reproduction: it evolves the Abrams–Mirollo–Strogatz–
Wiley (PRL 101, 084103, 2008) two-population model and reads the resulting state with the production
chimera diagnostics.

Two populations of ``N`` identical phase oscillators are coupled with strength ``μ`` within a
population and ``ν < μ`` between them, under a phase frustration ``α = π/2 − β``:

``θ̇_i = Σ_j K_ij sin(θ_j − θ_i − α)``,

the production Kuramoto–Sakaguchi force (:func:`~oscillatools.accel.sakaguchi_kuramoto.sakaguchi_force`,
the physically meaningful, multi-language-dispatched quantity) driven through a textbook RK4 step.
Seeded with one population coherent and the other incoherent, the system settles into a **chimera**:
one population phase-locks (order parameter ≈ 1) while the other never locks, its coherence breathing
chaotically well below unity. The production ``community_order_parameters`` / ``chimera_index`` read
exactly this coexistence.

At finite ``N`` the two-population chimera is a long-lived chaotic transient (Wolfrum & Omel'chenko,
PRE 84, 015201, 2011); it is an attractor only in the thermodynamic limit. The assertions below are
therefore made over a defined post-formation window in which the chimera is unambiguously present, and
they are bands, not point values — but the dynamics are fully seeded and the production force is
bit-identical across its Rust and Python tiers, so the trajectory (and every statistic below) is
reproducible exactly. The contrast case — stronger inter-population coupling — synchronises both
populations, so the diagnostics separate a genuine chimera from ordinary synchrony by orders of
magnitude, not by a threshold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_chimera import chimera_index, community_order_parameters
from oscillatools.accel.sakaguchi_kuramoto import sakaguchi_force

_N = 128  # oscillators per population
_DT = 0.05
_BETA = 0.1
_ALPHA = math.pi / 2.0 - _BETA  # phase frustration near π/2, where chimeras live
_SEED = 20260702

# Chimera regime: strong intra-population, weak inter-population coupling (A = μ − ν = 0.5, μ + ν = 1).
_CHIMERA_INTRA, _CHIMERA_INTER = 0.75, 0.25
_CHIMERA_STEPS = 1200
# Synchronising contrast: strong inter-population coupling (A = 0.2) pulls both populations together.
_SYNC_INTRA, _SYNC_INTER = 0.6, 0.4
_SYNC_STEPS = 700
# Skip the first T = 10 (formation) and read the post-formation window in which the state has settled.
_SETTLE = 200


def _block_coupling(n: int, intra: float, inter: float) -> NDArray[np.float64]:
    r"""Two-population block coupling matrix: ``intra`` within each population, ``inter`` between.

    Each entry is normalised by the population size ``n`` so the mean-field force is size-independent;
    the self-coupling diagonal is zeroed (the Sakaguchi force excludes ``k = j`` regardless).
    """
    total = 2 * n
    coupling = np.empty((total, total), dtype=np.float64)
    coupling[:n, :n] = coupling[n:, n:] = intra / n
    coupling[:n, n:] = coupling[n:, :n] = inter / n
    np.fill_diagonal(coupling, 0.0)
    return coupling


def _chimera_initial_condition(n: int, seed: int) -> NDArray[np.float64]:
    r"""One population near-coherent (phases ≈ 0), the other incoherent (uniform on the circle)."""
    rng = np.random.default_rng(seed)
    theta0 = np.empty(2 * n, dtype=np.float64)
    theta0[:n] = 0.01 * rng.standard_normal(n)
    theta0[n:] = rng.uniform(-math.pi, math.pi, n)
    return theta0


def _integrate(
    theta0: NDArray[np.float64],
    coupling: NDArray[np.float64],
    frustration: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""RK4 trajectory of the identical-oscillator Kuramoto–Sakaguchi flow ``θ̇ = F(θ)``.

    The right-hand side is the production :func:`sakaguchi_force`; classical fourth-order Runge–Kutta
    is the textbook wrapper. Returns the ``(n_steps + 1, 2N)`` phase trajectory including the initial
    state. The natural frequencies are identical (taken as zero), the defining condition of the
    Abrams–Mirollo–Strogatz–Wiley chimera.
    """
    trajectory = np.empty((n_steps + 1, theta0.size), dtype=np.float64)
    trajectory[0] = current = theta0.copy()
    for step in range(n_steps):
        k1 = sakaguchi_force(current, coupling, frustration)
        k2 = sakaguchi_force(current + 0.5 * _DT * k1, coupling, frustration)
        k3 = sakaguchi_force(current + 0.5 * _DT * k2, coupling, frustration)
        k4 = sakaguchi_force(current + _DT * k3, coupling, frustration)
        current = current + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    return trajectory


def _communities(n: int) -> list[NDArray[np.int_]]:
    return [np.arange(n), np.arange(n, 2 * n)]


def _windowed(intra: float, inter: float, n_steps: int) -> tuple[NDArray[np.float64], float]:
    r"""Evolve the two-population model and return ``(community order parameters, chimera index)``
    over the post-formation window ``[_SETTLE, n_steps]``."""
    coupling = _block_coupling(_N, intra, inter)
    theta0 = _chimera_initial_condition(_N, _SEED)
    trajectory = _integrate(theta0, coupling, _ALPHA, n_steps)
    window = trajectory[_SETTLE:n_steps]
    communities = _communities(_N)
    return community_order_parameters(window, communities), chimera_index(window, communities)


@pytest.fixture(scope="module")
def chimera_state() -> tuple[NDArray[np.float64], float]:
    return _windowed(_CHIMERA_INTRA, _CHIMERA_INTER, _CHIMERA_STEPS)


@pytest.fixture(scope="module")
def synchronised_state() -> tuple[NDArray[np.float64], float]:
    return _windowed(_SYNC_INTRA, _SYNC_INTER, _SYNC_STEPS)


class TestTwoPopulationChimera:
    def test_one_population_locks_while_the_other_breathes(
        self, chimera_state: tuple[NDArray[np.float64], float]
    ) -> None:
        community_r, _ = chimera_state
        coherent, incoherent = community_r[:, 0], community_r[:, 1]
        # Population 1 phase-locks and stays locked for the whole window.
        assert coherent.min() > 0.99
        # Population 2 never locks: its coherence stays below unity, dips deep, and breathes — the
        # defining coexistence of a chimera, not a stationary partially synchronised branch.
        assert incoherent.max() < 0.95
        assert incoherent.mean() < 0.85
        assert incoherent.min() < 0.5
        assert incoherent.std() > 0.08

    def test_chimera_index_is_strongly_positive(
        self, chimera_state: tuple[NDArray[np.float64], float]
    ) -> None:
        _, chi = chimera_state
        # Time-averaged variance of the community order parameters — zero for uniform synchrony or
        # uniform incoherence, clearly positive for the locked/unlocked chimera coexistence.
        assert chi > 0.01

    def test_strong_intercoupling_synchronises_both_populations(
        self, synchronised_state: tuple[NDArray[np.float64], float]
    ) -> None:
        community_r, chi = synchronised_state
        # With A = 0.2 the inter-population coupling is strong enough to lock both populations, so
        # there is no chimera: both order parameters sit near unity and the index collapses.
        assert community_r[:, 0].min() > 0.95
        assert community_r[:, 1].min() > 0.9
        assert community_r[:, 1].std() < 0.05
        assert chi < 0.01

    def test_chimera_and_synchronised_regimes_are_sharply_separated(
        self,
        chimera_state: tuple[NDArray[np.float64], float],
        synchronised_state: tuple[NDArray[np.float64], float],
    ) -> None:
        chimera_r, chimera_chi = chimera_state
        sync_r, sync_chi = synchronised_state
        # The reproduction is unambiguous: the chimera index and the breathing amplitude of the
        # unlocked population exceed the synchronised regime by more than an order of magnitude, so
        # the diagnostics separate the states by scale, not by a hand-tuned threshold.
        assert chimera_chi > 20.0 * sync_chi
        assert chimera_r[:, 1].std() > 10.0 * sync_r[:, 1].std()
