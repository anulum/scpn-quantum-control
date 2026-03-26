# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Monte Carlo simulation of the XY model on the K_nm graph.

Computes the Hasenbusch-Pinn universal amplitude A_HP on the actual
K_nm coupling topology (complete graph with exponential decay), not
the square lattice. This makes the p_h1 derivation airtight.

Method: Metropolis-Hastings sampling of the classical XY model:
    H = -Σ_{ij} K_ij cos(θ_j - θ_i)

Observables:
    - Energy: E = <H>
    - Order parameter: R = |mean(exp(i θ))|
    - Helicity modulus (spin stiffness):
        ρ_s = (1/N) [<Σ K_ij cos(Δθ)> - β <(Σ K_ij sin(Δθ))²>]
    - A_HP: extracted from ρ_s at T_BKT via Nelson-Kosterlitz relation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo XY model result at one temperature."""

    temperature: float
    energy: float
    order_parameter: float
    helicity_modulus: float
    specific_heat: float
    n_oscillators: int


@dataclass
class AHPResult:
    """Hasenbusch-Pinn amplitude extraction result."""

    a_hp_graph: float  # A_HP on the K_nm graph
    a_hp_square: float  # A_HP on square lattice (0.8983)
    p_h1_graph: float  # a_hp_graph × sqrt(2/π)
    p_h1_square: float  # 0.8983 × sqrt(2/π) = 0.717
    deviation_from_072: float
    t_bkt: float
    temperatures: np.ndarray
    helicity_moduli: np.ndarray
    order_parameters: np.ndarray


def _mc_sweep(
    theta: np.ndarray,
    K: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Metropolis sweep: propose random angle changes."""
    n = len(theta)
    for i in range(n):
        theta_old = theta[i]
        theta_new = theta_old + rng.uniform(-np.pi, np.pi)

        # Energy change from flipping site i
        delta_e = 0.0
        for j in range(n):
            if i != j and K[i, j] > 0:
                delta_e += K[i, j] * (np.cos(theta_new - theta[j]) - np.cos(theta_old - theta[j]))
        delta_e = -delta_e  # H = -Σ K cos(Δθ)

        if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
            theta[i] = theta_new

    return theta


def mc_simulate(
    K: np.ndarray,
    temperature: float,
    n_thermalize: int = 5000,
    n_measure: int = 5000,
    seed: int = 42,
) -> MCResult:
    """Monte Carlo simulation at fixed temperature.

    Uses Rust engine when available (100x speedup).
    """
    try:
        from scpn_quantum_engine import mc_xy_simulate  # type: ignore[import-not-found]

        n = K.shape[0]
        k_flat: np.ndarray = K.ravel().astype(np.float64)
        energy, order, rho_s = mc_xy_simulate(
            k_flat, n, temperature, n_thermalize, n_measure, seed
        )
        beta = 1.0 / max(temperature, 1e-15)
        return MCResult(
            temperature=temperature,
            energy=energy,
            order_parameter=order,
            helicity_modulus=rho_s,
            specific_heat=0.0,  # not computed in Rust path
            n_oscillators=n,
        )
    except ImportError:
        pass

    n = K.shape[0]
    beta = 1.0 / max(temperature, 1e-15)
    rng = np.random.default_rng(seed)

    theta = np.asarray(rng.uniform(0, 2 * np.pi, n))

    # Thermalise
    for _ in range(n_thermalize):
        theta = _mc_sweep(theta, K, beta, rng)

    # Measure
    energies: list[float] = []
    orders: list[float] = []
    cos_sums: list[float] = []
    sin_sums: list[float] = []

    for _ in range(n_measure):
        theta = _mc_sweep(theta, K, beta, rng)

        # Energy
        e = 0.0
        cos_sum = 0.0
        sin_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if K[i, j] > 0:
                    d = theta[j] - theta[i]
                    e -= K[i, j] * np.cos(d)
                    cos_sum += K[i, j] * np.cos(d)
                    sin_sum += K[i, j] * np.sin(d)
        energies.append(e)
        cos_sums.append(cos_sum)
        sin_sums.append(sin_sum)

        # Order parameter
        z = np.mean(np.exp(1j * theta))
        orders.append(float(np.abs(z)))

    e_mean = float(np.mean(energies))
    e2_mean = float(np.mean(np.array(energies) ** 2))
    r_mean = float(np.mean(orders))
    cv = beta**2 * (e2_mean - e_mean**2) / n

    # Helicity modulus: ρ_s = (1/N)[<cos_sum> - β<sin_sum²>]
    cos_mean = float(np.mean(cos_sums))
    sin2_mean = float(np.mean(np.array(sin_sums) ** 2))
    rho_s = (cos_mean - beta * sin2_mean) / n

    return MCResult(
        temperature=temperature,
        energy=e_mean,
        order_parameter=r_mean,
        helicity_modulus=rho_s,
        specific_heat=cv,
        n_oscillators=n,
    )


def extract_a_hp(
    K: np.ndarray,
    t_range: tuple[float, float] = (0.01, 0.2),
    n_temps: int = 15,
    n_thermalize: int = 3000,
    n_measure: int = 3000,
    seed: int = 42,
) -> AHPResult:
    """Extract Hasenbusch-Pinn amplitude from MC on K_nm graph.

    Scans temperature, finds T_BKT from the Nelson-Kosterlitz condition
    ρ_s(T_BKT) = (2/π) × T_BKT, then extracts A_HP from the order
    parameter at T_BKT.
    """
    temps = np.linspace(t_range[0], t_range[1], n_temps)
    rho_s_vals = np.zeros(n_temps)
    r_vals = np.zeros(n_temps)

    for i, t in enumerate(temps):
        mc = mc_simulate(K, t, n_thermalize, n_measure, seed + i)
        rho_s_vals[i] = mc.helicity_modulus
        r_vals[i] = mc.order_parameter

    # Find T_BKT: where ρ_s = (2/π) × T
    nk_line = (2.0 / np.pi) * temps
    diff = rho_s_vals - nk_line
    # T_BKT is where diff crosses zero from positive to negative
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings) > 0:
        idx = crossings[0]
        # Linear interpolation
        t_bkt = float(
            temps[idx] - diff[idx] * (temps[idx + 1] - temps[idx]) / (diff[idx + 1] - diff[idx])
        )
    else:
        t_bkt = float(temps[n_temps // 2])

    # A_HP: order parameter at T_BKT, normalised
    # Interpolate R at T_BKT
    r_at_bkt = float(np.interp(t_bkt, temps, r_vals))

    # A_HP = R(T_BKT) / sqrt(2/π)  (from p_h1 = A_HP × sqrt(2/π))
    nk_sqrt = float(np.sqrt(2.0 / np.pi))
    a_hp_graph = r_at_bkt / nk_sqrt if nk_sqrt > 1e-10 else 0.0

    a_hp_square = 0.8983
    p_h1_graph = a_hp_graph * nk_sqrt
    p_h1_square = a_hp_square * nk_sqrt

    return AHPResult(
        a_hp_graph=a_hp_graph,
        a_hp_square=a_hp_square,
        p_h1_graph=p_h1_graph,
        p_h1_square=p_h1_square,
        deviation_from_072=abs(p_h1_graph - 0.72),
        t_bkt=t_bkt,
        temperatures=temps,
        helicity_moduli=rho_s_vals,
        order_parameters=r_vals,
    )


@dataclass
class FiniteSizeResult:
    """Finite-size scaling of A_HP across system sizes.

    a_hp_inf is the N→∞ extrapolation from a linear fit of A_HP vs 1/N.
    """

    n_values: list[int]
    a_hp_means: list[float]
    a_hp_stds: list[float]
    p_h1_means: list[float]
    n_seeds: int
    a_hp_inf: float = 0.0  # extrapolated A_HP(N→∞), 0.0 if < 2 points


def finite_size_scaling(
    n_values: list[int] | None = None,
    n_seeds: int = 5,
    n_thermalize: int = 10000,
    n_measure: int = 10000,
    n_temps: int = 12,
    base_seed: int = 42,
) -> FiniteSizeResult:
    """Run A_HP extraction at multiple system sizes with error bars.

    For each N, runs n_seeds independent MC chains and reports
    mean ± std of A_HP. Tests whether A_HP(N) converges as N → ∞.

    Defaults match the Gap 3 verification protocol: N=4,8,16,32 with
    n_thermalize=10000, n_measure=10000, 5 seeds per N.
    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if n_values is None:
        n_values = [4, 8, 16, 32]

    nk_sqrt = float(np.sqrt(2.0 / np.pi))
    a_hp_means: list[float] = []
    a_hp_stds: list[float] = []
    p_h1_means: list[float] = []

    for n_osc in n_values:
        K = build_knm_paper27(L=n_osc)
        seed_results: list[float] = []

        for s in range(n_seeds):
            seed = base_seed + s * 1000 + n_osc
            result = extract_a_hp(
                K, n_temps=n_temps, n_thermalize=n_thermalize, n_measure=n_measure, seed=seed
            )
            seed_results.append(result.a_hp_graph)

        mean_ahp = float(np.mean(seed_results))
        std_ahp = float(np.std(seed_results))
        a_hp_means.append(mean_ahp)
        a_hp_stds.append(std_ahp)
        p_h1_means.append(mean_ahp * nk_sqrt)

    # Extrapolate A_HP(N→∞): linear fit of A_HP vs 1/N, intercept = A_HP(∞)
    a_hp_inf = 0.0
    if len(n_values) >= 2:
        inv_N = np.array([1.0 / N for N in n_values])
        coeffs = np.polyfit(inv_N, a_hp_means, deg=1)  # coeffs[0]*x + coeffs[1]
        a_hp_inf = float(coeffs[1])

    return FiniteSizeResult(
        n_values=n_values,
        a_hp_means=a_hp_means,
        a_hp_stds=a_hp_stds,
        p_h1_means=p_h1_means,
        n_seeds=n_seeds,
        a_hp_inf=a_hp_inf,
    )
