# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-Scale Quantum Error Correction
"""Hierarchical QEC across SCPN layers via concatenated surface codes.

The SCPN architecture defines 15+1 layers grouped into domains:
    Domain I  (L1–L4):  Biological substrate
    Domain II (L5–L8):  Organismal & planetary
    Domain III (L9–L12): Memory, control, collective
    Domain IV (L13–L15): Meta-universal
    Domain V  (L16):     Cybernetic closure (Anulum)

Multi-Scale QEC (MS-QEC) maps this hierarchy to concatenated quantum
error correction: each domain operates a surface code whose logical
qubits are the physical qubits of the next domain. Errors at lower
layers are corrected by syndromes extracted at higher layers.

The inter-level coupling follows SCPN K_nm: stronger coupling between
adjacent levels, exponential decay across levels. This determines how
much syndrome information flows between QEC levels.

Concatenated threshold (Knill 2005, Aharonov & Ben-Or 1997):
    p_L(level k) = A × (p_L(level k-1) / p_th)^((d_k + 1) / 2)

With each concatenation level, the logical error rate decreases
doubly-exponentially IF the physical rate is below threshold.

Ref:
    - Knill, Nature 434, 39 (2005)
    - Aharonov & Ben-Or, STOC 1997
    - Gottesman, arXiv:0904.2557 (2009)
    - SCPN Paper 27: MS-QEC concept (Šotek, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .error_budget import (
    SURFACE_CODE_PREFACTOR,
    SURFACE_CODE_THRESHOLD,
    logical_error_rate,
    minimum_code_distance,
)

try:
    from scpn_quantum_engine import (
        concatenated_logical_rate_rust as _concat_rust,
    )
    from scpn_quantum_engine import (
        knm_domain_coupling as _knm_coupling_rust,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# SCPN domain boundaries (0-indexed layer numbers)
SCPN_DOMAINS = {
    "biological": (0, 3),  # L1–L4
    "organismal": (4, 7),  # L5–L8
    "collective": (8, 11),  # L9–L12
    "meta": (12, 14),  # L13–L15
    "closure": (15, 15),  # L16
}


@dataclass
class QECLevel:
    """Single level in the MS-QEC hierarchy.

    Each level wraps a surface code operating on the logical qubits
    from the level below (or physical qubits at the bottom level).
    """

    level: int
    domain_name: str
    code_distance: int
    layer_range: tuple[int, int]
    physical_error_rate: float
    logical_error_rate: float
    physical_qubits_per_logical: int  # 2d² − 1
    n_logical_qubits: int
    total_physical_qubits: int
    knm_coupling_to_next: float  # K_nm to level+1


@dataclass
class MultiscaleQECResult:
    """Result of multi-scale QEC analysis."""

    levels: list[QECLevel]
    effective_logical_rate: float
    total_physical_qubits: int
    concatenation_depth: int
    below_threshold: bool
    double_exponential_suppression: bool
    summary: str


@dataclass
class SyndromeFlow:
    """Syndrome propagation between QEC levels."""

    source_level: int
    target_level: int
    syndrome_weight: float  # K_nm coupling strength
    correction_capacity: float  # max correctable weight at target
    information_flow: float  # bits of syndrome per QEC round


def knm_between_domains(
    K: np.ndarray,
    domain_a: tuple[int, int],
    domain_b: tuple[int, int],
) -> float:
    """Average K_nm coupling between two SCPN domains.

    Computes mean coupling strength across all layer pairs
    between the two domains. Uses Rust engine when available.
    """
    if _HAS_RUST:
        return float(_knm_coupling_rust(K, domain_a[0], domain_a[1], domain_b[0], domain_b[1]))

    n = K.shape[0]
    total = 0.0
    count = 0
    for i in range(domain_a[0], min(domain_a[1] + 1, n)):
        for j in range(domain_b[0], min(domain_b[1] + 1, n)):
            if i != j and i < n and j < n:
                total += K[i, j]
                count += 1
    return total / max(count, 1)


def concatenated_logical_rate(
    p_physical: float,
    distances: list[int],
    p_threshold: float = SURFACE_CODE_THRESHOLD,
    prefactor: float = SURFACE_CODE_PREFACTOR,
) -> list[float]:
    """Compute logical error rate at each concatenation level.

    Level 0: p_L(0) = A × (p_phys / p_th)^((d_0+1)/2)
    Level k: p_L(k) = A × (p_L(k-1) / p_th)^((d_k+1)/2)

    Returns list of p_L values, one per level.
    Uses Rust engine when available.
    """
    if not distances:
        return []

    if _HAS_RUST:
        d_arr = np.array(distances, dtype=np.int64)
        return list(_concat_rust(p_physical, d_arr, p_threshold, prefactor))

    rates: list[float] = []
    p_current = p_physical
    for d in distances:
        p_logical = logical_error_rate(d, p_current, p_threshold, prefactor)
        rates.append(p_logical)
        p_current = p_logical
    return rates


def syndrome_flow_between_levels(
    K: np.ndarray,
    level_a: QECLevel,
    level_b: QECLevel,
) -> SyndromeFlow:
    """Compute syndrome information flow between two QEC levels.

    The syndrome weight is proportional to K_nm coupling between
    the corresponding SCPN domains. Higher coupling means more
    syndrome information can flow, enabling better correction.
    """
    coupling = knm_between_domains(K, level_a.layer_range, level_b.layer_range)
    # Correction capacity: (d-1)/2 errors correctable
    correction_cap = (level_b.code_distance - 1) / 2.0
    # Information flow: coupling × log2(d) syndrome bits per round
    info_flow = coupling * np.log2(max(level_b.code_distance, 2))

    return SyndromeFlow(
        source_level=level_a.level,
        target_level=level_b.level,
        syndrome_weight=coupling,
        correction_capacity=correction_cap,
        information_flow=info_flow,
    )


def _active_domains(n: int) -> list[tuple[str, tuple[int, int]]]:
    """Determine active SCPN domains based on K matrix size."""
    domain_list = []
    for name, (start, end) in SCPN_DOMAINS.items():
        if start < n:
            domain_list.append((name, (start, min(end, n - 1))))
    return domain_list


def _auto_distances(
    n_levels: int,
    p_physical: float,
    target_logical_rate: float,
) -> list[int]:
    """Auto-compute code distances for equal suppression allocation."""
    per_level_target = target_logical_rate ** (1.0 / max(n_levels, 1))
    distances: list[int] = []
    p_current = p_physical
    for _ in range(n_levels):
        d = minimum_code_distance(per_level_target, p_current)
        distances.append(d)
        p_current = logical_error_rate(d, p_current)
    return distances


def _build_levels(
    domain_list: list[tuple[str, tuple[int, int]]],
    distances: list[int],
    rates: list[float],
    K: np.ndarray,
    p_physical: float,
    n_oscillators_per_level: int,
) -> tuple[list[QECLevel], int]:
    """Build QEC level objects and compute total physical qubits."""
    levels: list[QECLevel] = []
    total_phys = 0
    n_levels = len(domain_list)

    for i, (name, layer_range) in enumerate(domain_list):
        d = distances[i]
        phys_per_logical = 2 * d * d - 1
        level_phys = n_oscillators_per_level * phys_per_logical

        if i + 1 < n_levels:
            coupling = knm_between_domains(K, layer_range, domain_list[i + 1][1])
        else:
            coupling = 0.0

        levels.append(
            QECLevel(
                level=i,
                domain_name=name,
                code_distance=d,
                layer_range=layer_range,
                physical_error_rate=p_physical if i == 0 else rates[i - 1],
                logical_error_rate=rates[i],
                physical_qubits_per_logical=phys_per_logical,
                n_logical_qubits=n_oscillators_per_level,
                total_physical_qubits=level_phys,
                knm_coupling_to_next=coupling,
            )
        )
        total_phys += level_phys

    return levels, total_phys


def _check_double_exponential(rates: list[float], below_threshold: bool) -> bool:
    """Check if error rates decrease faster than exponentially."""
    if len(rates) < 2 or not below_threshold:
        return False
    log_rates = [np.log10(max(r, 1e-300)) for r in rates if r > 0]
    if len(log_rates) < 2:
        return False
    ratios = [
        log_rates[i + 1] / log_rates[i] for i in range(len(log_rates) - 1) if log_rates[i] != 0
    ]
    return all(r > 1.5 for r in ratios) if ratios else False


def build_multiscale_qec(
    K: np.ndarray,
    n_oscillators_per_level: int | None = None,
    p_physical: float = 0.003,
    target_logical_rate: float = 1e-10,
    distances: list[int] | None = None,
) -> MultiscaleQECResult:
    """Build multi-scale QEC hierarchy mapped to SCPN domains."""
    n = K.shape[0]
    if n_oscillators_per_level is None:
        n_oscillators_per_level = 4

    domain_list = _active_domains(n)
    n_levels = len(domain_list)

    if distances is None:
        distances = _auto_distances(n_levels, p_physical, target_logical_rate)
    elif len(distances) != n_levels:
        raise ValueError(f"distances length ({len(distances)}) != n_levels ({n_levels})")

    rates = concatenated_logical_rate(p_physical, distances)
    levels, total_phys = _build_levels(
        domain_list,
        distances,
        rates,
        K,
        p_physical,
        n_oscillators_per_level,
    )

    effective_rate = rates[-1] if rates else p_physical
    below_threshold = p_physical < SURFACE_CODE_THRESHOLD
    double_exp = _check_double_exponential(rates, below_threshold)

    summary = "; ".join(
        [
            f"MS-QEC: {n_levels} levels, {total_phys} physical qubits",
            f"p_phys={p_physical:.1e} → p_L={effective_rate:.1e}",
            f"distances={distances}",
            f"below threshold: {below_threshold}",
            f"double-exp suppression: {double_exp}",
        ]
    )

    return MultiscaleQECResult(
        levels=levels,
        effective_logical_rate=effective_rate,
        total_physical_qubits=total_phys,
        concatenation_depth=n_levels,
        below_threshold=below_threshold,
        double_exponential_suppression=double_exp,
        summary=summary,
    )


def syndrome_flow_analysis(
    K: np.ndarray,
    result: MultiscaleQECResult,
) -> list[SyndromeFlow]:
    """Analyse syndrome information flow between all adjacent levels.

    Returns a list of SyndromeFlow objects describing how error
    correction information propagates through the SCPN hierarchy.
    """
    flows: list[SyndromeFlow] = []
    for i in range(len(result.levels) - 1):
        flow = syndrome_flow_between_levels(K, result.levels[i], result.levels[i + 1])
        flows.append(flow)
    return flows
