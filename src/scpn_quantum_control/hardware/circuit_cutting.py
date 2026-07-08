# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit Cutting
"""Circuit cutting for scaling to 32-64 oscillators.

Circuit cutting (Peng et al., PRL 125, 150504 (2020)) decomposes a
large circuit into smaller sub-circuits that run independently, then
reconstructs the full result via classical post-processing.

For the Kuramoto-XY system with n oscillators:
    - Partition into k sub-systems of n/k oscillators each
    - Cut the inter-partition coupling gates
    - Each cut gate requires 4^c sub-circuit evaluations (c = cut count)
    - Classical overhead: O(4^c) per cut

Trade-off: fewer qubits per QPU run vs exponential classical overhead.

Optimal partition: minimise cuts while fitting sub-circuits into
available hardware (e.g., 127 qubits on IBM Eagle/Heron).

For 32 oscillators:
    - 2 partitions of 16: cut count = inter-partition couplings
    - 4 partitions of 8: more cuts, but fits smaller QPUs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

DEFAULT_MAX_PARTITION_SIZE: Final[int] = 16
DEFAULT_SCALING_N_VALUES: Final[tuple[int, ...]] = (16, 24, 32, 48, 64, 96, 128)


@dataclass
class CircuitCuttingPlan:
    """Circuit cutting resource plan."""

    n_oscillators: int
    n_partitions: int
    partition_sizes: list[int]
    n_cuts: int  # inter-partition gates cut
    classical_overhead: float  # 4^n_cuts
    max_partition_qubits: int
    fits_on_heron: bool  # max_partition <= 127 qubits


def count_inter_partition_couplings(
    K: NDArray[np.float64],
    partition: list[list[int]],
) -> int:
    """Count non-zero coupling terms between partitions."""
    cuts = 0
    for pi in range(len(partition)):
        for pj in range(pi + 1, len(partition)):
            for i in partition[pi]:
                for j in partition[pj]:
                    if abs(K[i, j]) > 1e-10:
                        cuts += 1
    return cuts


def optimal_partition(
    K: NDArray[np.float64],
    max_partition_size: int = DEFAULT_MAX_PARTITION_SIZE,
) -> list[list[int]]:
    """Partition oscillators to minimise inter-partition cuts.

    Simple greedy: contiguous blocks of max_partition_size.
    For all-to-all K_nm, any partition has the same cut count,
    so contiguous is as good as any.
    """
    n = _validate_partition_inputs(K, max_partition_size)
    partitions: list[list[int]] = []
    for start in range(0, n, max_partition_size):
        end = min(start + max_partition_size, n)
        partitions.append(list(range(start, end)))
    return partitions


def circuit_cutting_plan(
    K: NDArray[np.float64],
    max_partition_size: int = DEFAULT_MAX_PARTITION_SIZE,
    heron_qubits: int = 127,
) -> CircuitCuttingPlan:
    """Compute circuit cutting resource plan.

    Args:
        K: coupling matrix (n × n)
        max_partition_size: maximum oscillators per partition
        heron_qubits: available qubits per QPU
    """
    n = _validate_partition_inputs(K, max_partition_size)
    if not isinstance(heron_qubits, int):
        raise TypeError("heron_qubits must be an integer")
    if heron_qubits < 1:
        raise ValueError("heron_qubits must be >= 1")
    partition = optimal_partition(K, max_partition_size)
    n_cuts = count_inter_partition_couplings(K, partition)
    overhead = 4.0**n_cuts if n_cuts < 30 else float("inf")
    sizes = [len(p) for p in partition]
    max_size = max(sizes)

    return CircuitCuttingPlan(
        n_oscillators=n,
        n_partitions=len(partition),
        partition_sizes=sizes,
        n_cuts=n_cuts,
        classical_overhead=overhead,
        max_partition_qubits=max_size,
        fits_on_heron=max_size <= heron_qubits,
    )


def scaling_analysis(
    n_values: list[int] | None = None,
    max_partition_size: int = DEFAULT_MAX_PARTITION_SIZE,
) -> dict[str, list[float]]:
    """Analyse circuit cutting overhead across system sizes."""
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if n_values is None:
        n_values = list(DEFAULT_SCALING_N_VALUES)

    results: dict[str, list[float]] = {
        "n_oscillators": [],
        "n_partitions": [],
        "n_cuts": [],
        "log4_overhead": [],
        "fits_heron": [],
    }

    for n in n_values:
        if not isinstance(n, int):
            raise TypeError("n_values must contain integers")
        if n < 2:
            raise ValueError("n_values must be >= 2")
        K = build_knm_paper27(L=n)
        plan = circuit_cutting_plan(K, max_partition_size)
        results["n_oscillators"].append(n)
        results["n_partitions"].append(plan.n_partitions)
        results["n_cuts"].append(plan.n_cuts)
        results["log4_overhead"].append(float(plan.n_cuts * np.log10(4)))
        results["fits_heron"].append(plan.fits_on_heron)

    return results


def _validate_partition_inputs(K: NDArray[np.float64], max_partition_size: int) -> int:
    """Return the oscillator count after validating partition planner inputs."""
    if not isinstance(max_partition_size, int):
        raise TypeError("max_partition_size must be an integer")
    if max_partition_size < 1:
        raise ValueError("max_partition_size must be >= 1")
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError("K must be a square 2-D coupling matrix")
    if K_arr.shape[0] < 2:
        raise ValueError("K must describe at least two oscillators")
    if not np.all(np.isfinite(K_arr)):
        raise ValueError("K must be finite")
    return int(K_arr.shape[0])
