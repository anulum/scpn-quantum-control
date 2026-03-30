# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""OTOC as a synchronization transition probe.

Information scrambling (measured by the OTOC decay rate λ_Q) peaks
at the synchronization critical point K_c. This connects quantum
chaos theory to the Kuramoto phase transition:

    K < K_c: incoherent phase, weak scrambling (integrable-like)
    K = K_c: critical point, maximum scrambling (quantum chaos peak)
    K > K_c: synchronized phase, reduced scrambling (ordered, less chaotic)

This provides a new experimental observable for detecting the sync
transition on NISQ hardware: instead of measuring R (which requires
3-basis tomography), measure the OTOC (which requires only forward
and backward evolution). On hardware, the backward evolution can be
approximated via Loschmidt echo or randomized benchmarking.

OTOC is a standard many-body diagnostic (Maldacena-Shenker-Stanford 2016).
Applied here to the synchronization transition in Kuramoto-XY.
OTOC on NISQ: Swingle (2018), Mi et al. (Google, Science 2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .otoc import compute_otoc


@dataclass
class OTOCSyncScanResult:
    """OTOC scan across coupling strengths."""

    K_base_values: list[float]
    lyapunov_values: list[float | None]
    scrambling_times: list[float | None]
    otoc_final_values: list[float]
    R_classical: list[float]
    peak_scrambling_K: float | None
    n_qubits: int


def otoc_sync_scan(
    K: np.ndarray,
    omega: np.ndarray,
    K_base_range: np.ndarray | None = None,
    n_K_values: int = 15,
    t_max: float = 2.0,
    n_time_points: int = 20,
    w_qubit: int = 0,
    v_qubit: int | None = None,
) -> OTOCSyncScanResult:
    """Scan OTOC vs coupling strength to detect sync transition.

    At each K_base, computes the OTOC, extracts λ_Q and t*.
    Also computes classical R for comparison.
    """
    from ..hardware.classical import classical_kuramoto_reference

    n = K.shape[0]
    if K_base_range is None:
        K_base_range = np.linspace(0.01, 2.0, n_K_values)

    times = np.linspace(0, t_max, n_time_points)

    lyapunov_vals: list[float | None] = []
    scrambling_vals: list[float | None] = []
    otoc_finals: list[float] = []
    R_classical: list[float] = []

    for k_base in K_base_range:
        K_scaled = K * k_base

        otoc_result = compute_otoc(
            K_scaled,
            omega,
            times=times,
            w_qubit=w_qubit,
            v_qubit=v_qubit,
        )
        lyapunov_vals.append(otoc_result.lyapunov_estimate)
        scrambling_vals.append(otoc_result.scrambling_time)
        otoc_finals.append(float(otoc_result.otoc_values[-1]))

        cl = classical_kuramoto_reference(n, t_max=t_max, dt=0.01, K=K_scaled, omega=omega)
        R_classical.append(float(cl["R"][-1]))

    # Find peak scrambling (maximum λ_Q)
    valid_lyap = [(i, v) for i, v in enumerate(lyapunov_vals) if v is not None]
    if valid_lyap:
        peak_idx = max(valid_lyap, key=lambda x: x[1])[0]
        peak_K = float(K_base_range[peak_idx])
    else:
        peak_K = None

    return OTOCSyncScanResult(
        K_base_values=list(K_base_range),
        lyapunov_values=lyapunov_vals,
        scrambling_times=scrambling_vals,
        otoc_final_values=otoc_finals,
        R_classical=R_classical,
        peak_scrambling_K=peak_K,
        n_qubits=n,
    )


def compare_otoc_vs_R(scan: OTOCSyncScanResult) -> dict:
    """Analyze correlation between OTOC scrambling and classical R.

    If scrambling peaks where R transitions (0.3 < R < 0.7), the
    OTOC successfully detects the synchronization transition.
    """
    K_vals = np.array(scan.K_base_values)
    R_vals = np.array(scan.R_classical)

    # Find classical transition (R crosses 0.5)
    above = R_vals >= 0.5
    if np.any(above) and not np.all(above):
        trans_idx = int(np.argmax(above))
        K_c_classical = float(K_vals[trans_idx])
    else:
        K_c_classical = None

    return {
        "K_c_classical": K_c_classical,
        "K_c_otoc": scan.peak_scrambling_K,
        "delta_K_c": (
            abs(scan.peak_scrambling_K - K_c_classical)
            if scan.peak_scrambling_K is not None and K_c_classical is not None
            else None
        ),
        "otoc_detects_transition": (
            scan.peak_scrambling_K is not None
            and K_c_classical is not None
            and abs(scan.peak_scrambling_K - K_c_classical) < 0.5
        ),
    }
