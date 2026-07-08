# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto simulator reference tests
"""Tests for the WASM Kuramoto simulator's Python reference (ST-11)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio.kuramoto_reference import (  # noqa: E402
    MAX_OSCILLATORS,
    MAX_STEPS,
    decode_output,
    encode_kuramoto_input,
    order_parameter,
    simulate,
)

_N = 6
_OMEGA: NDArray[np.float64] = np.linspace(-0.5, 0.5, _N, dtype=np.float64)
_THETA0: NDArray[np.float64] = np.linspace(0.0, 2.0, _N, dtype=np.float64)


def _uniform_matrix(coupling: float, n: int) -> NDArray[np.float64]:
    """Return the all-to-all matrix ``K_ij = coupling/n`` with a zero diagonal."""
    k = np.full((n, n), coupling / n, dtype=np.float64)
    np.fill_diagonal(k, 0.0)
    return k


def test_encode_lengths_for_both_modes() -> None:
    """The packed input length matches the kernel's canonical layout."""
    mean = encode_kuramoto_input("mean-field", _OMEGA, _THETA0, steps=10, dt=0.05, coupling=1.0)
    assert len(mean) == 32 + 2 * _N * 8
    net = encode_kuramoto_input(
        "networked",
        _OMEGA,
        _THETA0,
        steps=10,
        dt=0.05,
        coupling=1.0,
        k_nm=_uniform_matrix(1.0, _N),
    )
    assert len(net) == 32 + (2 * _N + _N * _N) * 8


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "spiral"}, "unknown mode"),
        ({"steps": 0}, "steps must be"),
        ({"steps": MAX_STEPS + 1}, "steps must be"),
        ({"dt": 0.0}, "dt must be"),
        ({"dt": float("nan")}, "dt must be"),
        ({"coupling": float("inf")}, "coupling must be"),
    ],
)
def test_encode_fails_closed_on_bad_scalars(kwargs: dict[str, object], match: str) -> None:
    """Out-of-range scalars are rejected before packing."""
    base = {
        "mode": "mean-field",
        "omega": _OMEGA,
        "theta0": _THETA0,
        "steps": 10,
        "dt": 0.05,
        "coupling": 1.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        encode_kuramoto_input(**base)  # type: ignore[arg-type]


def test_encode_fails_closed_on_shape_and_matrix_errors() -> None:
    """Shape mismatches and misused coupling matrices fail closed."""
    with pytest.raises(ValueError, match="equal length"):
        encode_kuramoto_input("mean-field", _OMEGA, _THETA0[:-1], steps=10, dt=0.05, coupling=1.0)
    with pytest.raises(ValueError, match="n must be"):
        encode_kuramoto_input(
            "mean-field",
            np.zeros(MAX_OSCILLATORS + 1),
            np.zeros(MAX_OSCILLATORS + 1),
            steps=10,
            dt=0.05,
            coupling=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        encode_kuramoto_input(
            "mean-field", np.array([np.nan, 0.0]), np.zeros(2), steps=10, dt=0.05, coupling=1.0
        )
    with pytest.raises(ValueError, match="networked mode requires"):
        encode_kuramoto_input("networked", _OMEGA, _THETA0, steps=10, dt=0.05, coupling=1.0)
    with pytest.raises(ValueError, match="k_nm must have shape"):
        encode_kuramoto_input(
            "networked", _OMEGA, _THETA0, steps=10, dt=0.05, coupling=1.0, k_nm=np.zeros((2, 2))
        )
    with pytest.raises(ValueError, match="does not take a coupling matrix"):
        encode_kuramoto_input(
            "mean-field",
            _OMEGA,
            _THETA0,
            steps=10,
            dt=0.05,
            coupling=1.0,
            k_nm=_uniform_matrix(1.0, _N),
        )


def test_order_parameter_bounds() -> None:
    """R spans a fully incoherent spread (0) to a phase-locked state (1)."""
    assert order_parameter(np.array([])) == 0.0
    assert abs(order_parameter(np.full(8, 1.3)) - 1.0) < 1e-12
    assert (
        order_parameter(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float64)) < 1e-12
    )


def test_simulate_is_deterministic_and_bounded() -> None:
    """Two runs agree exactly and every R sample is a valid magnitude."""
    first = simulate("mean-field", _OMEGA, _THETA0, steps=200, dt=0.01, coupling=1.4)
    second = simulate("mean-field", _OMEGA, _THETA0, steps=200, dt=0.01, coupling=1.4)
    np.testing.assert_array_equal(first.order_parameter, second.order_parameter)
    assert first.order_parameter.shape == (201,)
    assert first.theta_final.shape == (_N,)
    assert np.all((first.order_parameter >= 0.0) & (first.order_parameter <= 1.0 + 1e-9))


def test_strong_coupling_synchronises() -> None:
    """Strong coupling raises the order parameter toward a locked state."""
    run = simulate("mean-field", _OMEGA, _THETA0, steps=1500, dt=0.01, coupling=4.0)
    assert run.order_parameter[-1] > run.order_parameter[0]
    assert run.order_parameter[-1] > 0.9


def test_mean_field_matches_uniform_network() -> None:
    """The mean-field kernel equals the all-to-all network with K_ij = K/N."""
    mean = simulate("mean-field", _OMEGA, _THETA0, steps=300, dt=0.01, coupling=1.5)
    net = simulate(
        "networked",
        _OMEGA,
        _THETA0,
        steps=300,
        dt=0.01,
        coupling=1.5,
        k_nm=_uniform_matrix(1.5, _N),
    )
    np.testing.assert_allclose(mean.order_parameter, net.order_parameter, rtol=1e-9, atol=1e-12)


def test_decode_output_round_trips() -> None:
    """decode_output is the inverse of the kernel's serialised output block."""
    run = simulate("mean-field", _OMEGA, _THETA0, steps=20, dt=0.05, coupling=1.0)
    values = [*run.order_parameter.tolist(), *run.theta_final.tolist()]
    raw = struct.pack(f"<{len(values)}d", *values)
    decoded = decode_output(raw, n=_N, steps=20)
    np.testing.assert_array_equal(decoded.order_parameter, run.order_parameter)
    np.testing.assert_array_equal(decoded.theta_final, run.theta_final)
    with pytest.raises(ValueError, match="output must be"):
        decode_output(raw[:-8], n=_N, steps=20)


def test_reference_tracks_the_scipy_baseline() -> None:
    """The fixed-step reference agrees with SciPy's adaptive Kuramoto ODE."""
    baseline = pytest.importorskip(
        "scpn_quantum_control.benchmarks.classical_baselines",
        reason="classical baselines require SciPy.",
    )
    n = 8
    omega = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    theta0 = np.linspace(0.0, 1.5, n, dtype=np.float64)
    k = _uniform_matrix(2.0, n)
    t_max = 4.0
    dt = 0.002
    steps = int(round(t_max / dt))

    reference = simulate("networked", omega, theta0, steps=steps, dt=dt, coupling=2.0, k_nm=k)
    scipy_run = baseline.scipy_ode_baseline(k, omega, t_max=t_max, dt=dt, theta0=theta0)

    # compare the final order parameter and a mid-trajectory sample
    assert abs(reference.order_parameter[-1] - scipy_run.order_parameter[-1]) < 5e-3
    mid = steps // 2
    assert abs(reference.order_parameter[mid] - scipy_run.order_parameter[mid]) < 5e-3
