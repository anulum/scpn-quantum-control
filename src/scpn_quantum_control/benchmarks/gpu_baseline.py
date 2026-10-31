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

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
ShardExecutor = Callable[[ComplexArray, ComplexArray, NDArray[np.float64]], ComplexArray]


@dataclass
class GPUBaselineResult:
    """GPU classical baseline resource estimate.

    Every time in this record is an analytic estimate from peak throughput,
    not a measurement; ``gpu_faster`` compares two estimates.
    """

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
) -> dict[str, list[float]]:
    """Compare GPU vs QPU across system sizes."""
    if n_values is None:
        n_values = [4, 8, 16, 24, 32, 40]

    results: dict[str, list[float]] = {
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


@dataclass(frozen=True)
class BatchEvolutionItem:
    """One independent evolution ``exp(-i H t) psi`` in a batch.

    Attributes
    ----------
    label:
        Unique non-empty name; results are keyed by it.
    hamiltonian:
        Hermitian ``(d, d)`` matrix.
    state:
        Initial ``(d,)`` state.
    time:
        Finite evolution time.

    """

    label: str
    hamiltonian: ComplexArray
    state: ComplexArray
    time: float


@dataclass(frozen=True)
class ShardRecord:
    """Measured execution of one shard.

    Attributes
    ----------
    index:
        Shard position in the plan.
    labels:
        Item labels in the shard, in plan order.
    seconds:
        Measured wall-clock seconds of the shard kernel.

    """

    index: int
    labels: tuple[str, ...]
    seconds: float


@dataclass(frozen=True)
class BatchExecutionRecord:
    """Placement, precision and cost evidence of a batched run.

    Attributes
    ----------
    requested_device:
        ``cpu`` or ``gpu`` as requested.
    executed_device:
        Device the kernels actually ran on; always ``cpu`` here.
    fallback:
        Whether a GPU request ran on the CPU with explicit permission.
    accelerator_evidence:
        Always ``False``: there is no GPU kernel, so no run is accelerator
        evidence.
    library:
        Library and version performing the kernels.
    precision:
        Element type of states and Hamiltonians.
    reduction_order:
        How results combine; items are independent, so nothing is reduced.
    rng:
        Random streams used; none.
    shard_size:
        Items per shard.
    shards:
        Shards executed by this call, excluding resumed ones.
    transfer_seconds:
        Host-device transfer time; zero, because execution is host-only.
    compile_seconds:
        Kernel compilation time; zero, because nothing is compiled.

    """

    requested_device: str
    executed_device: str
    fallback: bool
    accelerator_evidence: bool
    library: str
    precision: str
    reduction_order: str
    rng: str
    shard_size: int
    shards: tuple[ShardRecord, ...]
    transfer_seconds: float = 0.0
    compile_seconds: float = 0.0


@dataclass(frozen=True)
class BatchCheckpoint:
    """Results of completed shards, bound to the plan that produced them.

    Attributes
    ----------
    plan_digest:
        SHA-256 of the sorted items and shard size.
    completed_shards:
        Indices of shards whose results are kept.
    states:
        Final states of the completed items, by label.

    """

    plan_digest: str
    completed_shards: tuple[int, ...]
    states: Mapping[str, ComplexArray] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchEvolutionResult:
    """Final states by label and the execution record.

    Attributes
    ----------
    states:
        Final state of every item, by label.
    record:
        Placement and cost evidence.
    resumed_shards:
        Shards taken from the checkpoint instead of being recomputed.

    """

    states: Mapping[str, ComplexArray]
    record: BatchExecutionRecord
    resumed_shards: tuple[int, ...] = ()


class AcceleratorUnavailableError(ValueError):
    """A GPU run was requested, no GPU kernel exists and fallback was not allowed."""


class BatchExecutionInterruptedError(RuntimeError):
    """A shard failed; ``checkpoint`` keeps every shard completed before it.

    Attributes
    ----------
    checkpoint:
        Resumable state with the completed shards.
    shard_index:
        Index of the failed shard.

    """

    def __init__(self, checkpoint: BatchCheckpoint, shard_index: int) -> None:
        """Record the resumable checkpoint and the failed shard.

        Parameters
        ----------
        checkpoint:
            Completed shards and their states.
        shard_index:
            Index of the failed shard.

        """
        super().__init__(
            f"batched evolution interrupted at shard {shard_index}; "
            f"{len(checkpoint.completed_shards)} shard(s) completed and resumable"
        )
        self.checkpoint = checkpoint
        self.shard_index = shard_index


def evolve_shard_on_cpu(
    hamiltonians: ComplexArray, states: ComplexArray, times: NDArray[np.float64]
) -> ComplexArray:
    """Evolve a stack of states with stacked Hermitian eigendecompositions.

    Parameters
    ----------
    hamiltonians:
        ``(k, d, d)`` Hermitian matrices.
    states:
        ``(k, d)`` initial states.
    times:
        ``(k,)`` evolution times.

    Returns
    -------
    numpy.ndarray
        ``(k, d)`` states ``V exp(-i Lambda t) V^dagger psi``.

    """
    values, vectors = np.linalg.eigh(hamiltonians)
    coefficients = np.einsum("kji,kj->ki", vectors.conj(), states)
    phases = np.exp(-1j * values * times[:, None])
    return np.asarray(np.einsum("kij,kj->ki", vectors, phases * coefficients), dtype=np.complex128)


def run_batched_evolutions(
    items: Sequence[BatchEvolutionItem],
    *,
    shard_size: int = 8,
    device: str = "cpu",
    allow_cpu_fallback: bool = False,
    checkpoint: BatchCheckpoint | None = None,
    shard_executor: ShardExecutor = evolve_shard_on_cpu,
) -> BatchEvolutionResult:
    """Run independent evolutions in label-ordered shards with placement evidence.

    Items are sorted by label before sharding, so results do not depend on
    input order. A failing shard raises with a checkpoint of every earlier
    shard; passing that checkpoint back resumes without recomputing them.

    Parameters
    ----------
    items:
        Independent evolutions of one common dimension, with unique labels.
    shard_size:
        Positive number of items per shard.
    device:
        ``cpu``, or ``gpu``, which has no kernel here.
    allow_cpu_fallback:
        Run a ``gpu`` request on the CPU and record it as a fallback.
    checkpoint:
        Checkpoint of the same plan from an interrupted run.
    shard_executor:
        Kernel applied to each shard; the default stacked CPU kernel.

    Returns
    -------
    BatchEvolutionResult
        Final states by label and the execution record.

    Raises
    ------
    AcceleratorUnavailableError
        If ``gpu`` is requested without permission to fall back.
    BatchExecutionInterruptedError
        If a shard kernel fails; its checkpoint is resumable.
    ValueError
        If items, shard size, device or checkpoint are malformed.

    """
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if device == "gpu" and not allow_cpu_fallback:
        raise AcceleratorUnavailableError(
            "no GPU kernel exists for batched evolution; pass allow_cpu_fallback=True "
            "to run on the CPU with the fallback recorded"
        )
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    batch = _stacked_batch(items)
    done: dict[str, ComplexArray] = {}
    resumed: tuple[int, ...] = ()
    if checkpoint is not None:
        if checkpoint.plan_digest != _plan_digest(batch, shard_size):
            raise ValueError("checkpoint belongs to a different batch plan")
        done = {label: np.array(state) for label, state in checkpoint.states.items()}
        resumed = tuple(checkpoint.completed_shards)
    skipped = frozenset(resumed)
    records: list[ShardRecord] = []
    completed = list(resumed)
    for index, start in enumerate(range(0, len(batch.labels), shard_size)):
        if index in skipped:
            continue
        stop = start + shard_size
        labels = batch.labels[start:stop]
        started = time.perf_counter()
        try:
            evolved = shard_executor(
                batch.hamiltonians[start:stop], batch.states[start:stop], batch.times[start:stop]
            )
        except Exception as error:
            raise BatchExecutionInterruptedError(
                BatchCheckpoint(_plan_digest(batch, shard_size), tuple(completed), dict(done)),
                index,
            ) from error
        seconds = time.perf_counter() - started
        for label, state in zip(labels, evolved, strict=True):
            done[label] = np.array(state, dtype=np.complex128)
        completed.append(index)
        records.append(ShardRecord(index, labels, seconds))
    record = BatchExecutionRecord(
        requested_device=device,
        executed_device="cpu",
        fallback=device == "gpu",
        accelerator_evidence=False,
        library=f"numpy {np.__version__}",
        precision="complex128",
        reduction_order="none: items are independent and never combined",
        rng="none",
        shard_size=shard_size,
        shards=tuple(records),
    )
    return BatchEvolutionResult(states=done, record=record, resumed_shards=resumed)


@dataclass(frozen=True)
class _StackedBatch:
    """Label-ordered items stacked once into contiguous arrays."""

    labels: tuple[str, ...]
    hamiltonians: ComplexArray
    states: ComplexArray
    times: NDArray[np.float64]


def _stacked_batch(items: Sequence[BatchEvolutionItem]) -> _StackedBatch:
    """Check labels and shapes, stack the items, then check values in one pass.

    Label and shape problems are found before value problems. A value problem
    names the first offending item in label order.
    """
    if not items:
        raise ValueError("a batch needs at least one item")
    ordered = sorted(items, key=lambda item: item.label)
    labels = tuple(item.label for item in ordered)
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("item labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("item labels must be unique")
    dimension = np.shape(ordered[0].state)
    for item in ordered:
        if (
            len(dimension) != 1
            or np.shape(item.state) != dimension
            or np.shape(item.hamiltonian) != 2 * dimension
        ):
            raise ValueError(f"item {item.label!r} does not share the batch dimension")
    hamiltonians = np.array([item.hamiltonian for item in ordered], dtype=np.complex128)
    states = np.array([item.state for item in ordered], dtype=np.complex128)
    times = np.array([item.time for item in ordered], dtype=np.float64)
    finite = np.asarray(
        np.isfinite(hamiltonians).all(axis=(1, 2)) & np.isfinite(states).all(axis=1),
        dtype=np.bool_,
    )
    _refuse_first(~finite, labels, "must be finite")
    _refuse_first(~np.isfinite(times), labels, "time must be finite")
    scale = np.maximum(1.0, np.abs(hamiltonians).max(axis=(1, 2), initial=0.0))
    asymmetry = np.abs(hamiltonians - np.conj(np.swapaxes(hamiltonians, 1, 2)))
    _refuse_first(
        asymmetry.max(axis=(1, 2), initial=0.0) > 1e-12 * scale,
        labels,
        "Hamiltonian must be Hermitian",
    )
    for array in (hamiltonians, states, times):
        array.setflags(write=False)
    return _StackedBatch(labels, hamiltonians, states, times)


def _refuse_first(failing: NDArray[np.bool_], labels: tuple[str, ...], problem: str) -> None:
    """Raise for the first item flagged in ``failing``, if any."""
    if failing.any():
        raise ValueError(f"item {labels[int(np.argmax(failing))]!r} {problem}")


def _plan_digest(batch: _StackedBatch, shard_size: int) -> str:
    """Hash the shard size, the length-prefixed labels and the stacked arrays."""
    digest = hashlib.sha256(str(shard_size).encode())
    for label in batch.labels:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(np.asarray(batch.hamiltonians.shape, dtype=np.int64).tobytes())
    for array in (batch.hamiltonians, batch.states, batch.times):
        digest.update(array.tobytes())
    return digest.hexdigest()
