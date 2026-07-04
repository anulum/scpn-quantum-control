# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the higher-order Watanabe–Strogatz reduction
"""Module-specific tests for :mod:`kuramoto_higher_order_watanabe_strogatz`.

The contracts: at ``p = 1`` the higher-order reduction is identical to the classical Watanabe–Strogatz
flow; for ``p ≥ 2`` the reconstructed ``e^{ipθ}`` reproduce a direct ``N``-oscillator integration of the
``p``-th harmonic coupling to integrator precision (the exact-reduction property); the ``SU(1,1)``
invariant is conserved while the Möbius parameters remain bounded (a non-synchronising regime); the
``p``-th harmonic order parameter behaves; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_higher_order_watanabe_strogatz import (
    HigherOrderWatanabeStrogatzTrajectory,
    integrate_higher_order_watanabe_strogatz,
)
from oscillatools.accel.kuramoto_watanabe_strogatz import integrate_watanabe_strogatz

_N = 12
_DT = 0.02
_STEPS = 2000


def _initial(seed: int = 0) -> NDArray[np.float64]:
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=_N)


def _direct_harmonic_field(
    initial: NDArray[np.float64],
    omega: float,
    coupling: float,
    harmonic: int,
) -> NDArray[np.complex128]:
    """Direct ``N``-oscillator integration of θ̇ = ω + K Im(Z_p e^{-ipθ}); returns e^{ipθ}(t)."""
    theta = initial.copy()
    images = [np.exp(1j * harmonic * theta)]

    def field(angle: NDArray[np.float64]) -> NDArray[np.float64]:
        order = np.mean(np.exp(1j * harmonic * angle))
        return np.asarray(omega + coupling * np.imag(order * np.exp(-1j * harmonic * angle)))

    for _ in range(_STEPS):
        k1 = field(theta)
        k2 = field(theta + 0.5 * _DT * k1)
        k3 = field(theta + 0.5 * _DT * k2)
        k4 = field(theta + _DT * k3)
        theta = theta + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        images.append(np.exp(1j * harmonic * theta))
    return np.asarray(images, dtype=np.complex128)


def test_first_harmonic_is_the_classical_reduction() -> None:
    initial = _initial()
    higher = integrate_higher_order_watanabe_strogatz(
        initial, omega=0.5, coupling=1.5, harmonic=1, dt=_DT, n_steps=_STEPS
    )
    classical = integrate_watanabe_strogatz(
        initial, omega=0.5, coupling=1.5, dt=_DT, n_steps=_STEPS
    )
    assert isinstance(higher, HigherOrderWatanabeStrogatzTrajectory)
    assert higher.alpha == pytest.approx(classical.alpha, abs=1e-12)
    assert higher.beta == pytest.approx(classical.beta, abs=1e-12)
    assert higher.order_parameter == pytest.approx(classical.order_parameter, abs=1e-12)


@pytest.mark.parametrize("harmonic", [2, 3])
def test_reduction_reproduces_direct_integration(harmonic: int) -> None:
    initial = _initial()
    reduced = integrate_higher_order_watanabe_strogatz(
        initial, omega=0.5, coupling=1.5, harmonic=harmonic, dt=_DT, n_steps=_STEPS
    )
    direct = _direct_harmonic_field(initial, 0.5, 1.5, harmonic)
    reduced_images = np.exp(1j * reduced.harmonic_phases)
    assert reduced_images == pytest.approx(direct, abs=1e-4)
    # the p-th harmonic order parameter synchronises under attractive coupling
    assert abs(reduced.order_parameter[-1]) > 0.99


@pytest.mark.parametrize("harmonic", [1, 2, 3])
def test_invariant_conserved_in_a_bounded_regime(harmonic: int) -> None:
    # repulsive coupling desynchronises, so the Möbius parameters stay bounded and the SU(1,1)
    # invariant is conserved to machine precision (no synchronisation coordinate divergence)
    reduced = integrate_higher_order_watanabe_strogatz(
        _initial(), omega=0.5, coupling=-1.0, harmonic=harmonic, dt=_DT, n_steps=_STEPS
    )
    assert np.max(np.abs(reduced.invariant - 1.0)) < 1e-8
    assert np.max(np.abs(reduced.alpha)) < 5.0
    assert abs(reduced.order_parameter[-1]) < 0.1


def test_trajectory_container() -> None:
    initial = _initial(2)
    reduced = integrate_higher_order_watanabe_strogatz(
        initial, omega=0.0, coupling=1.0, harmonic=2, dt=0.05, n_steps=10
    )
    assert reduced.harmonic == 2
    assert reduced.alpha.shape == (11,)
    assert reduced.harmonic_phases.shape == (11, _N)
    assert reduced.constants == pytest.approx(np.exp(2j * initial))
    assert reduced.times[-1] == pytest.approx(10 * 0.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_phases": np.zeros((2, 2))}, "must be a non-empty one-dimensional"),
        ({"initial_phases": np.full(_N, np.nan)}, "initial_phases must be finite"),
        ({"omega": np.inf}, "omega must be finite"),
        ({"coupling": np.inf}, "coupling must be finite"),
        ({"harmonic": 0}, "harmonic must be positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"dt": np.inf}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "initial_phases": _initial(),
        "omega": 0.5,
        "coupling": 1.0,
        "harmonic": 2,
        "dt": _DT,
        "n_steps": 10,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        integrate_higher_order_watanabe_strogatz(
            call["initial_phases"],
            omega=call["omega"],
            coupling=call["coupling"],
            harmonic=call["harmonic"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
