# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Surface code error budget for Kuramoto-XY simulation.

Combines three error sources to determine the minimum code distance
d and total physical qubit count for a target logical error rate:

    1. Trotter error: epsilon_T from commutator bounds
    2. Gate error: epsilon_G from physical error rates (CZ, 1Q)
    3. Logical error: epsilon_L(d, p_phys) from surface code threshold

Total error budget:
    epsilon_total = epsilon_T + epsilon_G + epsilon_L

Surface code logical error rate (sub-threshold):
    p_L(d) ≈ A × (p_phys / p_th)^((d+1)/2)

where p_th ≈ 0.01 (threshold), A ≈ 0.1 (empirical prefactor).

Resource estimation:
    Physical qubits per oscillator: 2d² - 1
    Total physical qubits: n_osc × (2d² - 1)
    QEC rounds per Trotter step: d (for full syndrome extraction)
    Total QEC overhead: n_trotter_steps × d × t_syndrome
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..phase.trotter_error import commutator_norm_bound, trotter_error_bound

# Surface code threshold. Google Willow (2024): achieved below-threshold
# operation at d=7 with p_phys ≈ 0.3%.
SURFACE_CODE_THRESHOLD = 0.01
SURFACE_CODE_PREFACTOR = 0.1  # empirical A in p_L = A × (p/p_th)^((d+1)/2)


@dataclass
class ErrorBudget:
    """Error budget for fault-tolerant Kuramoto-XY simulation."""

    n_oscillators: int
    trotter_error: float  # epsilon_T
    gate_error_per_step: float  # epsilon_G per Trotter step
    logical_error_rate: float  # epsilon_L per round
    total_error: float  # epsilon_T + n_steps × epsilon_G + n_steps × epsilon_L
    code_distance: int  # minimum d for target error
    physical_qubits_per_osc: int  # 2d² - 1
    total_physical_qubits: int
    n_trotter_steps: int
    qec_rounds_total: int  # n_steps × d
    commutator_bound: float
    frequency_heterogeneity: float


def logical_error_rate(
    code_distance: int,
    p_physical: float,
    p_threshold: float = SURFACE_CODE_THRESHOLD,
    prefactor: float = SURFACE_CODE_PREFACTOR,
) -> float:
    """Surface code logical error rate at given distance and physical rate.

    p_L = A × (p_phys / p_th)^((d+1)/2)
    """
    ratio = p_physical / p_threshold
    if ratio >= 1.0:
        return 1.0  # above threshold
    return float(prefactor * ratio ** ((code_distance + 1) / 2.0))


def minimum_code_distance(
    target_logical_rate: float,
    p_physical: float,
    p_threshold: float = SURFACE_CODE_THRESHOLD,
    prefactor: float = SURFACE_CODE_PREFACTOR,
    max_distance: int = 51,
) -> int:
    """Find minimum odd d such that p_L(d) <= target."""
    for d in range(3, max_distance + 1, 2):
        if logical_error_rate(d, p_physical, p_threshold, prefactor) <= target_logical_rate:
            return d
    return max_distance


def compute_error_budget(
    K: np.ndarray,
    omega: np.ndarray,
    t_total: float = 1.0,
    trotter_order: int = 1,
    target_total_error: float = 0.01,
    p_physical: float = 0.003,
    cz_error: float = 0.005,
) -> ErrorBudget:
    """Compute full error budget for Kuramoto-XY simulation.

    Allocates the target total error across three channels:
        - 1/3 to Trotter
        - 1/3 to gate errors
        - 1/3 to logical errors

    Then determines the Trotter steps needed and code distance.

    Args:
        K: coupling matrix (n × n)
        omega: natural frequencies
        t_total: total simulation time
        trotter_order: 1 or 2
        target_total_error: allowed total error
        p_physical: physical gate error rate
        cz_error: CZ gate error rate
    """
    n = K.shape[0]

    # Allocate error budget: equal thirds
    eps_trotter_budget = target_total_error / 3.0
    eps_logical_budget = target_total_error / 3.0

    # Trotter error: find n_steps such that eps_T <= budget
    comm_bound = commutator_norm_bound(K, omega)
    if comm_bound < 1e-15:
        n_steps = 1
        eps_trotter = 0.0
    else:
        # Binary search for minimum steps
        n_steps = 1
        while n_steps < 100000:
            eps_t = trotter_error_bound(K, omega, t_total, n_steps, order=trotter_order)
            if eps_t <= eps_trotter_budget:
                break
            n_steps *= 2
        eps_trotter = trotter_error_bound(K, omega, t_total, n_steps, order=trotter_order)

    # Gate error per step: n_osc CZ gates + 2*n_osc single-qubit gates
    n_cz_per_step = n * (n - 1) // 2
    n_1q_per_step = 2 * n
    eps_gate_per_step = n_cz_per_step * cz_error + n_1q_per_step * p_physical
    eps_gate_total = n_steps * eps_gate_per_step

    # Logical error: find minimum distance
    target_logical_per_round = eps_logical_budget / max(n_steps, 1)
    d = minimum_code_distance(target_logical_per_round, p_physical)
    eps_logical_per_round = logical_error_rate(d, p_physical)
    eps_logical_total = n_steps * eps_logical_per_round

    total_error = eps_trotter + eps_gate_total + eps_logical_total
    phys_per_osc = 2 * d * d - 1
    total_phys = n * phys_per_osc
    qec_rounds = n_steps * d

    freq_het = float(np.std(omega) / max(np.mean(np.abs(omega)), 1e-15))

    return ErrorBudget(
        n_oscillators=n,
        trotter_error=eps_trotter,
        gate_error_per_step=eps_gate_per_step,
        logical_error_rate=eps_logical_per_round,
        total_error=total_error,
        code_distance=d,
        physical_qubits_per_osc=phys_per_osc,
        total_physical_qubits=total_phys,
        n_trotter_steps=n_steps,
        qec_rounds_total=qec_rounds,
        commutator_bound=comm_bound,
        frequency_heterogeneity=freq_het,
    )


def compare_error_budgets(
    K: np.ndarray,
    omega: np.ndarray,
    p_physical_values: np.ndarray | None = None,
) -> list[ErrorBudget]:
    """Compare error budgets across hardware generations.

    Default: current Heron (0.3%), Willow-like (0.1%), future (0.01%).
    """
    if p_physical_values is None:
        p_physical_values = np.array([0.003, 0.001, 0.0001])

    return [compute_error_budget(K, omega, p_physical=float(p)) for p in p_physical_values]
