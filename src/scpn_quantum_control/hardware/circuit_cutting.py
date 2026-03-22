# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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

import numpy as np


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
    K: np.ndarray,
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
    K: np.ndarray,
    max_partition_size: int = 16,
) -> list[list[int]]:
    """Partition oscillators to minimise inter-partition cuts.

    Simple greedy: contiguous blocks of max_partition_size.
    For all-to-all K_nm, any partition has the same cut count,
    so contiguous is as good as any.
    """
    n = K.shape[0]
    partitions: list[list[int]] = []
    for start in range(0, n, max_partition_size):
        end = min(start + max_partition_size, n)
        partitions.append(list(range(start, end)))
    return partitions


def circuit_cutting_plan(
    K: np.ndarray,
    max_partition_size: int = 16,
    heron_qubits: int = 127,
) -> CircuitCuttingPlan:
    """Compute circuit cutting resource plan.

    Args:
        K: coupling matrix (n × n)
        max_partition_size: maximum oscillators per partition
        heron_qubits: available qubits per QPU
    """
    n = K.shape[0]
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
    max_partition_size: int = 16,
) -> dict[str, list]:
    """Analyse circuit cutting overhead across system sizes."""
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if n_values is None:
        n_values = [16, 24, 32, 48, 64]

    results: dict[str, list] = {
        "n_oscillators": [],
        "n_partitions": [],
        "n_cuts": [],
        "log4_overhead": [],
        "fits_heron": [],
    }

    for n in n_values:
        K = build_knm_paper27(L=n)
        plan = circuit_cutting_plan(K, max_partition_size)
        results["n_oscillators"].append(n)
        results["n_partitions"].append(plan.n_partitions)
        results["n_cuts"].append(plan.n_cuts)
        results["log4_overhead"].append(float(plan.n_cuts * np.log10(4)))
        results["fits_heron"].append(plan.fits_on_heron)

    return results
