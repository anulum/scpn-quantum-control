# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for nonlocally coupled Stuart–Landau oscillators
"""Module-specific tests for :mod:`stuart_landau`.

The contracts: an uncoupled Stuart–Landau oscillator relaxes to the limit cycle of radius ``√λ``
(amplitude dynamics the phase-only models lack); the real-coordinate Jacobian matches finite
differences to machine precision; strong nonlocal symmetry-breaking coupling drives the ensemble to
an inhomogeneous oscillation-death steady state (oscillations cease, amplitudes stay spatially
patterned); the order parameter and amplitude diagnostics behave; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.stuart_landau import (
    StuartLandauTrajectory,
    amplitudes,
    integrate_stuart_landau,
    is_oscillation_death,
    stuart_landau_field,
    stuart_landau_jacobian,
    stuart_landau_order_parameter,
)

_N = 30
_OMEGA = 2.0


def _random_state(seed: int, scale: float = 0.3) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.uniform(-scale, scale, _N) + 1j * rng.uniform(-scale, scale, _N), dtype=np.complex128
    )


@pytest.mark.parametrize("lam", [1.0, 4.0])
def test_uncoupled_relaxes_to_limit_cycle_radius_sqrt_lambda(lam: float) -> None:
    trajectory = integrate_stuart_landau(_random_state(0), lam, _OMEGA, 0.0, 1, 0.02, 4000)
    assert isinstance(trajectory, StuartLandauTrajectory)
    final = amplitudes(trajectory.terminal_state)
    assert final == pytest.approx(np.full(_N, np.sqrt(lam)), abs=1e-3)


def test_jacobian_matches_finite_differences() -> None:
    lam, coupling, radius = 1.0, 3.0, 3
    state = _random_state(1, scale=1.0)
    jacobian = stuart_landau_jacobian(state, lam, _OMEGA, coupling, radius)

    def real_field(packed: NDArray[np.float64]) -> NDArray[np.float64]:
        derivative = stuart_landau_field(
            packed[:_N] + 1j * packed[_N:], lam, _OMEGA, coupling, radius
        )
        return np.concatenate([derivative.real, derivative.imag])

    packed = np.concatenate([state.real, state.imag])
    eps = 1e-6
    finite = np.zeros((2 * _N, 2 * _N), dtype=np.float64)
    for column in range(2 * _N):
        plus = packed.copy()
        minus = packed.copy()
        plus[column] += eps
        minus[column] -= eps
        finite[:, column] = (real_field(plus) - real_field(minus)) / (2.0 * eps)
    assert jacobian == pytest.approx(finite, abs=1e-7)


def test_strong_coupling_reaches_inhomogeneous_oscillation_death() -> None:
    lam, coupling, radius = 1.0, 24.0, 6
    trajectory = integrate_stuart_landau(
        _random_state(3, scale=1.5), lam, _OMEGA, coupling, radius, 0.01, 12000
    )
    final = trajectory.terminal_state
    # oscillations have ceased (a steady state)
    assert is_oscillation_death(final, lam, _OMEGA, coupling, radius)
    final_amplitudes = amplitudes(final)
    # the steady state is spatially inhomogeneous (impossible for the phase-only models)
    assert final_amplitudes.std() > 0.05
    # and it is a genuine death state, not the uncoupled limit cycle of radius √λ = 1
    assert final_amplitudes.mean() < 0.95


def test_order_parameter_and_amplitudes() -> None:
    synchronous = np.full(_N, 0.7 + 0.7j, dtype=np.complex128)
    assert abs(stuart_landau_order_parameter(synchronous)) == pytest.approx(1.0)
    assert amplitudes(synchronous) == pytest.approx(np.full(_N, np.hypot(0.7, 0.7)))
    # a dead oscillator (z = 0) is taken to have phase zero and amplitude zero
    mixed = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    assert amplitudes(mixed) == pytest.approx([0.0, 1.0])
    assert abs(stuart_landau_order_parameter(mixed)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("field", {"state": np.zeros(1, dtype=np.complex128)}, "at least two oscillators"),
        ("field", {"radius": 0}, "radius must be in"),
        ("field", {"radius": _N}, "radius must be in"),
        ("field", {"omega": np.inf}, "omega must be finite"),
        ("field", {"coupling": np.inf}, "coupling must be finite"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
        ("death", {"velocity_tolerance": 0.0}, "velocity_tolerance must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    state = _random_state(0)
    with pytest.raises(ValueError, match=message):
        if call == "field":
            args: dict[str, Any] = {
                "state": state,
                "lam": 1.0,
                "omega": _OMEGA,
                "coupling": 1.0,
                "radius": 2,
            }
            args.update(kwargs)
            stuart_landau_field(
                args["state"], args["lam"], args["omega"], args["coupling"], args["radius"]
            )
        elif call == "integrate":
            args = {"dt": 0.02, "n_steps": 10}
            args.update(kwargs)
            integrate_stuart_landau(state, 1.0, _OMEGA, 1.0, 2, args["dt"], args["n_steps"])
        else:
            integrate = integrate_stuart_landau(state, 1.0, _OMEGA, 1.0, 2, 0.02, 10)
            is_oscillation_death(
                integrate.terminal_state,
                1.0,
                _OMEGA,
                1.0,
                2,
                velocity_tolerance=kwargs["velocity_tolerance"],
            )
