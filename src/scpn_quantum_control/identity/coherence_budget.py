# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coherence Budget
"""Coherence budget calculator for identity quantum circuits.

Estimates the maximum circuit depth at which fidelity remains above
a given threshold, using the Heron r2 noise model calibration data.
The coherence budget is the quantitative limit on how complex a
quantum identity representation can be on NISQ hardware.
"""

from __future__ import annotations

import numpy as np

from ..hardware.noise_model import (
    CZ_ERROR_RATE,
    READOUT_ERROR_RATE,
    SINGLE_GATE_TIME_US,
    T1_US,
    T2_US,
    TWO_GATE_TIME_US,
)


def fidelity_at_depth(
    depth: int,
    n_qubits: int,
    *,
    t1_us: float = T1_US,
    t2_us: float = T2_US,
    cz_error: float = CZ_ERROR_RATE,
    readout_error: float = READOUT_ERROR_RATE,
    two_qubit_fraction: float = 0.4,
) -> float:
    """Estimate circuit fidelity at a given gate depth.

    Model: F = F_gate^(n_gates) * F_readout^(n_qubits) * F_decoherence
    where F_decoherence = exp(-t_total / T2) approximately.

    Args:
        depth: Total gate layers.
        n_qubits: Number of qubits in the circuit.
        two_qubit_fraction: Fraction of layers that are two-qubit gates.

    Returns:
        Estimated fidelity in [0, 1].
    """
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if depth == 0:
        return 1.0

    n_two_qubit_layers = int(depth * two_qubit_fraction)
    n_single_qubit_layers = depth - n_two_qubit_layers

    # Gate fidelity: each two-qubit gate has error rate cz_error
    # Applied to ~n_qubits/2 pairs per layer
    n_two_qubit_gates = n_two_qubit_layers * max(1, n_qubits // 2)
    f_gate = (1.0 - cz_error) ** n_two_qubit_gates

    # Readout fidelity
    f_readout = (1.0 - readout_error) ** n_qubits

    # Decoherence: total circuit time
    t_total_us = (
        n_single_qubit_layers * SINGLE_GATE_TIME_US + n_two_qubit_layers * TWO_GATE_TIME_US
    )
    f_decoherence = np.exp(-t_total_us / t2_us) ** n_qubits

    return float(f_gate * f_readout * f_decoherence)


def coherence_budget(
    n_qubits: int,
    *,
    fidelity_threshold: float = 0.5,
    max_depth: int = 2000,
    t1_us: float = T1_US,
    t2_us: float = T2_US,
    cz_error: float = CZ_ERROR_RATE,
    readout_error: float = READOUT_ERROR_RATE,
    two_qubit_fraction: float = 0.4,
) -> dict:
    """Compute the maximum circuit depth before fidelity drops below threshold.

    Returns dict with max_depth (the budget), fidelity_at_max,
    fidelity_curve (sampled at key depths), and hardware_params.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if not 0.0 < fidelity_threshold < 1.0:
        raise ValueError(f"fidelity_threshold must be in (0, 1), got {fidelity_threshold}")

    kwargs = dict(
        t1_us=t1_us,
        t2_us=t2_us,
        cz_error=cz_error,
        readout_error=readout_error,
        two_qubit_fraction=two_qubit_fraction,
    )

    # Binary search for the depth where fidelity crosses threshold
    lo, hi = 0, max_depth
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fidelity_at_depth(mid, n_qubits, **kwargs) >= fidelity_threshold:
            lo = mid
        else:
            hi = mid - 1

    budget_depth = lo

    # Sample fidelity at key depths
    sample_depths = sorted(set([1, 10, 50, 100, 250, 500, 1000, budget_depth]))
    sample_depths = [d for d in sample_depths if d <= max_depth]
    fidelity_curve = {d: fidelity_at_depth(d, n_qubits, **kwargs) for d in sample_depths}

    return {
        "n_qubits": n_qubits,
        "fidelity_threshold": fidelity_threshold,
        "max_depth": budget_depth,
        "fidelity_at_max": fidelity_at_depth(budget_depth, n_qubits, **kwargs),
        "fidelity_curve": fidelity_curve,
        "hardware_params": {
            "t1_us": t1_us,
            "t2_us": t2_us,
            "cz_error": cz_error,
            "readout_error": readout_error,
        },
    }
