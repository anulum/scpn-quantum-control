# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Floquet Kuramoto
"""Floquet-Kuramoto: periodically driven XY synchronization.

Modulate the coupling: K(t) = K_0 * (1 + delta * cos(Omega * t))
with heterogeneous natural frequencies omega_i.

If R(t) responds at frequency Omega/2 (subharmonic), this is a
discrete time crystal signature: the system spontaneously breaks
the discrete time-translation symmetry of the drive.

The heterogeneous frequencies provide effective disorder that may
stabilize the DTC phase via many-body localization (MBL).

Prior art: All published DTCs use homogeneous frequencies.
Floquet-Kuramoto with heterogeneous omega_i is completely open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .._rust_accel import optional_rust_engine
from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

DTC_SUBHARMONIC_THRESHOLD = 0.1
"""Minimum subharmonic-power ratio used to flag a DTC candidate."""


@dataclass
class FloquetResult:
    """Floquet-Kuramoto simulation result."""

    times: FloatArray
    R_values: FloatArray
    drive_signal: FloatArray
    subharmonic_ratio: float  # power at Omega/2 / power at Omega
    is_dtc_candidate: bool  # subharmonic_ratio > DTC_SUBHARMONIC_THRESHOLD


def _build_H_matrix(
    K: FloatArray,
    omega: FloatArray,
    *,
    max_dense_gib: float | None = None,
) -> ComplexArray:
    """Build dense Hamiltonian matrix."""
    return np.asarray(
        knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib), dtype=np.complex128
    )


def _order_parameter(psi: ComplexArray, n: int) -> float:
    """R from statevector via single-qubit expectations.

    Rust fast path uses bitwise Pauli expectations (no Qiskit overhead).
    """
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        psi_re = np.ascontiguousarray(psi.real)
        psi_im = np.ascontiguousarray(psi.imag)
        exp_x, exp_y = _engine.all_xy_expectations(psi_re, psi_im, n)
        phases = np.arctan2(
            np.asarray(exp_y, dtype=np.float64),
            np.asarray(exp_x, dtype=np.float64),
        )
        return float(abs(np.mean(np.exp(1j * phases))))
    except AttributeError:
        pass

    from ..hardware.classical import _xy_expectations_vectorized

    exp_x, exp_y = _xy_expectations_vectorized(np.ascontiguousarray(psi), n)
    phases = np.arctan2(exp_y, exp_x)
    return float(abs(np.mean(np.exp(1j * phases))))


def floquet_evolve(
    K_topology: FloatArray,
    omega: FloatArray,
    K_base: float,
    drive_amplitude: float,
    drive_frequency: float,
    n_periods: int = 10,
    steps_per_period: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> FloquetResult:
    """Evolve under periodically driven Kuramoto-XY Hamiltonian.

    K(t) = K_base * (1 + drive_amplitude * cos(drive_frequency * t)) * K_topology

    Args:
        K_topology: normalized coupling matrix (max=1)
        omega: natural frequencies
        K_base: base coupling strength
        drive_amplitude: delta in K(t) = K_base*(1 + delta*cos(Omega*t))
        drive_frequency: Omega (angular frequency of periodic drive)
        n_periods: number of drive periods to simulate
        steps_per_period: time steps per period
        max_dense_gib: optional dense allocation budget in GiB
    """
    n = len(omega)
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=5,
        max_gib=max_dense_gib,
        label="Floquet dense evolution workspace",
    )
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=1,
        object_count=2,
        max_gib=max_dense_gib,
        label="Floquet dense state workspace",
    )
    dim = 2**n
    T_drive = 2 * np.pi / drive_frequency
    dt = T_drive / steps_per_period
    n_steps = n_periods * steps_per_period

    # Initial state: |+⟩^n (equal superposition — nontrivial phases)
    psi: ComplexArray = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    times = np.zeros(n_steps + 1, dtype=np.float64)
    R_values = np.zeros(n_steps + 1, dtype=np.float64)
    drive_signal = np.zeros(n_steps + 1, dtype=np.float64)

    R_values[0] = _order_parameter(psi, n)
    drive_signal[0] = K_base * (1 + drive_amplitude)

    # Pre-build the frequency-only Hamiltonian (time-independent part)
    H_freq = _build_H_matrix(np.zeros_like(K_topology), omega, max_dense_gib=max_dense_gib)

    for step in range(n_steps):
        t = (step + 0.5) * dt  # midpoint
        K_t = K_base * (1 + drive_amplitude * np.cos(drive_frequency * t))

        # Full Hamiltonian at midpoint
        H_coupling = _build_H_matrix(
            K_t * K_topology,
            np.zeros(n, dtype=np.float64),
            max_dense_gib=max_dense_gib,
        )
        H_total = H_coupling + H_freq

        # Propagate: |psi(t+dt)⟩ = exp(-iHdt)|psi(t)⟩
        U = expm(-1j * H_total * dt)
        psi = U @ psi

        times[step + 1] = (step + 1) * dt
        R_values[step + 1] = _order_parameter(psi, n)
        drive_signal[step + 1] = K_t

    # Spectral analysis: subharmonic detection
    sub_ratio = _subharmonic_ratio(R_values[1:], drive_frequency, dt)

    return FloquetResult(
        times=times,
        R_values=R_values,
        drive_signal=drive_signal,
        subharmonic_ratio=sub_ratio,
        is_dtc_candidate=sub_ratio > DTC_SUBHARMONIC_THRESHOLD,
    )


def _subharmonic_ratio(signal: FloatArray, drive_freq: float, dt: float) -> float:
    """Ratio of spectral power at Omega/2 to power at Omega.

    > 0 indicates subharmonic response (DTC signature).
    """
    n = len(signal)
    if n < 4:
        return 0.0

    # Remove DC component
    signal_ac = signal - np.mean(signal)

    fft = np.fft.rfft(signal_ac)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft) ** 2

    # Find bins closest to Omega and Omega/2
    omega_target = drive_freq / (2 * np.pi)
    omega_half = omega_target / 2

    idx_omega = int(np.argmin(np.abs(freqs - omega_target)))
    idx_half = int(np.argmin(np.abs(freqs - omega_half)))

    # Window: sum power in ±1 bin around target
    def _windowed_power(idx: int) -> float:
        lo = max(0, idx - 1)
        hi = min(len(power), idx + 2)
        return float(np.sum(power[lo:hi]))

    p_omega = _windowed_power(idx_omega)
    p_half = _windowed_power(idx_half)

    if p_omega < 1e-30:
        return 0.0

    return float(p_half / p_omega)


def scan_drive_amplitude(
    K_topology: FloatArray,
    omega: FloatArray,
    K_base: float,
    drive_frequency: float,
    amplitudes: FloatArray | None = None,
    n_periods: int = 8,
    steps_per_period: int = 16,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, list[float]]:
    """Scan subharmonic ratio across drive amplitudes.

    DTC phase boundary: where subharmonic_ratio jumps from ~0 to > 0.
    """
    if amplitudes is None:
        scan_amplitudes = np.asarray(np.linspace(0.1, 2.0, 10), dtype=np.float64)
    else:
        scan_amplitudes = amplitudes

    results: dict[str, list[float]] = {
        "amplitude": [],
        "subharmonic_ratio": [],
        "mean_R": [],
        "is_dtc": [],
    }

    for amp in scan_amplitudes:
        fr = floquet_evolve(
            K_topology,
            omega,
            K_base,
            float(amp),
            drive_frequency,
            n_periods,
            steps_per_period,
            max_dense_gib=max_dense_gib,
        )
        results["amplitude"].append(float(amp))
        results["subharmonic_ratio"].append(fr.subharmonic_ratio)
        results["mean_R"].append(float(np.mean(fr.R_values)))
        results["is_dtc"].append(float(fr.is_dtc_candidate))

    return results
