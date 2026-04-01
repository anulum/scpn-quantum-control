# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gpu Baseline
"""GPU baseline comparison for quantum simulation.

cuQuantum (NVIDIA) and cuTensorNet can simulate quantum circuits
on GPU with statevector or tensor network methods. This module
estimates the GPU resources needed to classically simulate the
Kuramoto-XY system and compares with QPU resources.

GPU statevector simulation:
    Memory: 2^n × 16 bytes (complex128)
    Time: O(2^n × n_gates) FLOPs
    n=16: 1 MB, instant
    n=30: 16 GB, minutes
    n=40: 16 TB, infeasible

GPU tensor network (cuTensorNet):
    Memory: O(chi^2 × n) where chi = bond dimension
    Time: O(chi^3 × n × n_gates)
    Advantage over statevector when entanglement is bounded

The quantum advantage boundary is where GPU time exceeds QPU time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUBaselineResult:
    """GPU classical baseline resource estimate."""

    n_qubits: int
    n_gates: int
    statevector_memory_gb: float
    statevector_flops: float
    estimated_gpu_time_s: float  # on A100 (312 TFLOPS FP64)
    qpu_time_s: float  # estimated QPU execution
    gpu_faster: bool
    crossover_n: int  # estimated n where QPU wins


# A100 GPU: 312 TFLOPS FP64, 80 GB HBM
A100_TFLOPS = 312e12
A100_MEMORY_GB = 80.0


def statevector_memory_gb(n: int) -> float:
    """GPU memory for statevector simulation."""
    return float((2**n) * 16 / 1e9)


def statevector_flops(n: int, n_gates: int) -> float:
    """FLOPs for statevector simulation: each gate is O(2^n) operations."""
    return float(n_gates * (2**n) * 10)


def estimate_gpu_time(n: int, n_gates: int, tflops: float = A100_TFLOPS) -> float:
    """Estimated GPU wall time in seconds."""
    flops = statevector_flops(n, n_gates)
    return flops / tflops


def estimate_qpu_time(n: int, n_gates: int, gate_time_us: float = 0.5) -> float:
    """Estimated QPU wall time in seconds.

    Assumes sequential gate execution (conservative).
    """
    return n_gates * gate_time_us * 1e-6


def gate_count_xy_trotter(n: int, reps: int = 10) -> int:
    """Gate count for XY Trotter circuit: reps × (n(n-1) CZ + 2n RZ)."""
    n_cz = n * (n - 1) // 2
    n_rz = 2 * n
    return reps * (n_cz + n_rz)


def gpu_baseline_comparison(
    n: int,
    trotter_reps: int = 10,
) -> GPUBaselineResult:
    """Compare GPU vs QPU for given system size."""
    n_gates = gate_count_xy_trotter(n, trotter_reps)
    mem = statevector_memory_gb(n)
    flops = statevector_flops(n, n_gates)
    gpu_time = estimate_gpu_time(n, n_gates)
    qpu_time = estimate_qpu_time(n, n_gates)

    gpu_faster = gpu_time < qpu_time

    # Find crossover: where GPU time > QPU time
    crossover = n
    for test_n in range(n, 100):
        test_gates = gate_count_xy_trotter(test_n, trotter_reps)
        if estimate_gpu_time(test_n, test_gates) > estimate_qpu_time(test_n, test_gates):
            crossover = test_n
            break
        if statevector_memory_gb(test_n) > A100_MEMORY_GB:
            crossover = test_n
            break

    return GPUBaselineResult(
        n_qubits=n,
        n_gates=n_gates,
        statevector_memory_gb=mem,
        statevector_flops=flops,
        estimated_gpu_time_s=gpu_time,
        qpu_time_s=qpu_time,
        gpu_faster=gpu_faster,
        crossover_n=crossover,
    )


def scaling_comparison(
    n_values: list[int] | None = None,
) -> dict[str, list]:
    """Compare GPU vs QPU across system sizes."""
    if n_values is None:
        n_values = [4, 8, 16, 24, 32, 40]

    results: dict[str, list] = {
        "n": [],
        "gpu_time_s": [],
        "qpu_time_s": [],
        "memory_gb": [],
        "gpu_faster": [],
    }
    for n in n_values:
        r = gpu_baseline_comparison(n)
        results["n"].append(n)
        results["gpu_time_s"].append(r.estimated_gpu_time_s)
        results["qpu_time_s"].append(r.qpu_time_s)
        results["memory_gb"].append(r.statevector_memory_gb)
        results["gpu_faster"].append(r.gpu_faster)

    return results
