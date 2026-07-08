# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio Kuramoto simulator reference
"""Python reference for the WASM Kuramoto live simulator (ST-11).

Mirrors the Rust kernel in ``scpn_quantum_engine/studio_wasm_kernel/src/kuramoto.rs``:
the same canonical little-endian wire format, the same two coupling kernels
(mean-field + networked), the same fixed-step RK4 integrator, and the same
``R(t)`` order-parameter output. It exists to (a) pin the browser kernel to a
committed, physics-validated fixture and (b) let :mod:`kuramoto_scenario_artifact`
emit that fixture. The simulator is a live visualisation, not a bit-exact
evidence claim, so the browser panel is checked against this reference within a
tolerance rather than bit-for-bit.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

KURAMOTO_INPUT_VERSION: Final[int] = 1
"""Version stamped into the canonical simulator input (matches the Rust kernel)."""

MAX_OSCILLATORS: Final[int] = 128
"""Fail-closed oscillator-count boundary (matches ``kuramoto::MAX_OSCILLATORS``)."""

MAX_STEPS: Final[int] = 4096
"""Fail-closed step-count boundary (matches ``kuramoto::MAX_STEPS``)."""

_HEADER_LEN: Final[int] = 32

KuramotoMode = Literal["mean-field", "networked"]
_MODE_CODES: Final[dict[KuramotoMode, int]] = {"mean-field": 0, "networked": 1}


@dataclass(frozen=True)
class KuramotoRun:
    """The order-parameter trajectory and final phases of one simulation."""

    order_parameter: NDArray[np.float64]
    theta_final: NDArray[np.float64]


def _as_finite(name: str, values: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    """Return ``values`` as a finite float64 array or fail closed."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def encode_kuramoto_input(
    mode: KuramotoMode,
    omega: NDArray[np.float64] | list[float],
    theta0: NDArray[np.float64] | list[float],
    *,
    steps: int,
    dt: float,
    coupling: float,
    k_nm: NDArray[np.float64] | None = None,
) -> bytes:
    """Pack a simulation request into the kernel's canonical little-endian input.

    The networked kernel carries a row-major ``n × n`` matrix; the mean-field
    kernel carries none. Bounds are enforced here so an over-large request fails
    closed before it is ever handed to the browser kernel.
    """
    if mode not in _MODE_CODES:
        raise ValueError(f"unknown mode: {mode!r}")
    omega_arr = _as_finite("omega", omega)
    theta_arr = _as_finite("theta0", theta0)
    n = int(omega_arr.shape[0])
    if omega_arr.ndim != 1 or theta_arr.shape != omega_arr.shape:
        raise ValueError("omega and theta0 must be 1-D arrays of equal length")
    if n < 1 or n > MAX_OSCILLATORS:
        raise ValueError(f"n must be in 1..={MAX_OSCILLATORS}, got {n}")
    if steps < 1 or steps > MAX_STEPS:
        raise ValueError(f"steps must be in 1..={MAX_STEPS}, got {steps}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive float")
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")

    payload = bytearray()
    payload.extend(struct.pack("<I", KURAMOTO_INPUT_VERSION))
    payload.extend(struct.pack("<I", _MODE_CODES[mode]))
    payload.extend(struct.pack("<I", n))
    payload.extend(struct.pack("<I", int(steps)))
    payload.extend(struct.pack("<d", float(dt)))
    payload.extend(struct.pack("<d", float(coupling)))
    for value in omega_arr:
        payload.extend(struct.pack("<d", float(value)))
    for value in theta_arr:
        payload.extend(struct.pack("<d", float(value)))
    if mode == "networked":
        if k_nm is None:
            raise ValueError("networked mode requires a coupling matrix")
        k_arr = _as_finite("k_nm", k_nm)
        if k_arr.shape != (n, n):
            raise ValueError(f"k_nm must have shape {(n, n)}, got {k_arr.shape}")
        for value in k_arr.reshape(-1):
            payload.extend(struct.pack("<d", float(value)))
    elif k_nm is not None:
        raise ValueError("mean-field mode does not take a coupling matrix")
    return bytes(payload)


def order_parameter(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter ``R = |(1/N) Σ_j exp(i θ_j)|``."""
    if theta.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * theta))))


def _derivative(
    mode: KuramotoMode,
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    k_nm: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return ``dθ/dt`` for the selected kernel."""
    if mode == "mean-field":
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        mean_field: NDArray[np.float64] = omega + coupling * (
            sin_mean * np.cos(theta) - cos_mean * np.sin(theta)
        )
        return mean_field
    if mode != "networked":
        raise ValueError(f"unknown mode: {mode!r}")
    if k_nm is None:
        raise ValueError("networked mode requires a coupling matrix")
    expected_shape = (int(theta.shape[0]), int(theta.shape[0]))
    if k_nm.shape != expected_shape:
        raise ValueError(f"k_nm must have shape {expected_shape}, got {k_nm.shape}")
    phase_delta = theta[None, :] - theta[:, None]
    networked: NDArray[np.float64] = omega + np.sum(k_nm * np.sin(phase_delta), axis=1)
    return networked


def simulate(
    mode: KuramotoMode,
    omega: NDArray[np.float64] | list[float],
    theta0: NDArray[np.float64] | list[float],
    *,
    steps: int,
    dt: float,
    coupling: float,
    k_nm: NDArray[np.float64] | None = None,
) -> KuramotoRun:
    """Integrate the request with fixed-step RK4, mirroring the Rust kernel."""
    omega_arr = _as_finite("omega", omega)
    theta = _as_finite("theta0", theta0).copy()
    k_arr = None if k_nm is None else _as_finite("k_nm", k_nm)
    if omega_arr.ndim != 1 or theta.shape != omega_arr.shape:
        raise ValueError("omega and theta0 must be 1-D arrays of equal length")
    if mode == "mean-field" and k_arr is not None:
        raise ValueError("mean-field mode does not take a coupling matrix")

    r_series = np.empty(steps + 1, dtype=np.float64)
    r_series[0] = order_parameter(theta)
    for step in range(steps):
        k1 = _derivative(mode, theta, omega_arr, coupling, k_arr)
        k2 = _derivative(mode, theta + 0.5 * dt * k1, omega_arr, coupling, k_arr)
        k3 = _derivative(mode, theta + 0.5 * dt * k2, omega_arr, coupling, k_arr)
        k4 = _derivative(mode, theta + dt * k3, omega_arr, coupling, k_arr)
        theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        r_series[step + 1] = order_parameter(theta)
    return KuramotoRun(order_parameter=r_series, theta_final=theta)


def decode_output(raw: bytes, *, n: int, steps: int) -> KuramotoRun:
    """Decode the kernel's ``[R(t) ; θ_final]`` little-endian f64 output block."""
    expected = (steps + 1 + n) * 8
    if len(raw) != expected:
        raise ValueError(f"output must be {expected} bytes, got {len(raw)}")
    values = np.array(struct.unpack(f"<{steps + 1 + n}d", raw), dtype=np.float64)
    return KuramotoRun(order_parameter=values[: steps + 1], theta_final=values[steps + 1 :])


__all__ = [
    "KURAMOTO_INPUT_VERSION",
    "MAX_OSCILLATORS",
    "MAX_STEPS",
    "KuramotoMode",
    "KuramotoRun",
    "decode_output",
    "encode_kuramoto_input",
    "order_parameter",
    "simulate",
]
