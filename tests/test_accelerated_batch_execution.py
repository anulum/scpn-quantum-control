# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — batched evolution placement and resumability qualification
"""Batched independent evolutions: parity, label order, resumption and placement.

Each item is a driven two-level system ``H = (W/2) X + (D/2) Z`` started in
``|0>``. Its closed-form propagator is
``cos(E t/2) I - i sin(E t/2) (W X + D Z) / E`` with ``E = sqrt(W^2 + D^2)``.
States are compared under the frozen float64 budget ``1e-10 + 1e-8*|ref|``.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks.gpu_baseline import (
    AcceleratorUnavailableError,
    BatchCheckpoint,
    BatchEvolutionItem,
    BatchExecutionInterruptedError,
    evolve_shard_on_cpu,
    gpu_baseline_comparison,
    run_batched_evolutions,
)

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.diag([1.0, -1.0]).astype(np.complex128)
_ZERO = np.array([1.0, 0.0], dtype=np.complex128)


def _item(index: int) -> BatchEvolutionItem:
    """Return the ``index``-th driven two-level evolution."""
    drive, detuning, time = 0.3 + 0.17 * index, -0.4 + 0.11 * index, 0.5 + 0.23 * index
    return BatchEvolutionItem(
        label=f"item-{index:02d}",
        hamiltonian=0.5 * drive * _X + 0.5 * detuning * _Z,
        state=_ZERO.copy(),
        time=time,
    )


def _analytic(index: int) -> NDArray[np.complex128]:
    """Return the closed-form final state of item ``index``."""
    drive, detuning, time = 0.3 + 0.17 * index, -0.4 + 0.11 * index, 0.5 + 0.23 * index
    energy = math.hypot(drive, detuning)
    propagator = (
        math.cos(energy * time / 2.0) * np.eye(2)
        - 1j * math.sin(energy * time / 2.0) * (drive * _X + detuning * _Z) / energy
    )
    return np.asarray(propagator @ _ZERO, dtype=np.complex128)


def _within_budget(actual: object, reference: object) -> bool:
    """Apply the frozen float64 test budget elementwise."""
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    return bool(
        np.all(
            np.abs(actual_array - reference_array) <= 1.0e-10 + 1.0e-8 * np.abs(reference_array)
        )
    )


def test_accelerated_batch_execution_01() -> None:
    """A batch of independent evolutions matches each closed-form state."""
    items = [_item(index) for index in range(10)]
    result = run_batched_evolutions(items, shard_size=3)
    for index in range(10):
        assert _within_budget(result.states[f"item-{index:02d}"], _analytic(index))
    record = result.record
    assert [shard.labels for shard in record.shards] == [
        ("item-00", "item-01", "item-02"),
        ("item-03", "item-04", "item-05"),
        ("item-06", "item-07", "item-08"),
        ("item-09",),
    ]
    assert all(shard.seconds >= 0.0 for shard in record.shards)
    assert (record.requested_device, record.executed_device) == ("cpu", "cpu")
    assert record.fallback is False and record.accelerator_evidence is False
    assert record.precision == "complex128" and record.rng == "none"
    assert record.library == f"numpy {np.__version__}"
    assert (record.transfer_seconds, record.compile_seconds) == (0.0, 0.0)
    assert result.resumed_shards == ()


def test_accelerated_batch_execution_02() -> None:
    """Permuting the input order leaves every labelled result bit-identical."""
    items = [_item(index) for index in range(7)]
    reference = run_batched_evolutions(items, shard_size=2)
    permuted = run_batched_evolutions([items[i] for i in (4, 0, 6, 2, 5, 1, 3)], shard_size=2)
    assert set(permuted.states) == set(reference.states)
    for label, state in reference.states.items():
        np.testing.assert_array_equal(permuted.states[label], state)
    assert [shard.labels for shard in permuted.record.shards] == [
        shard.labels for shard in reference.record.shards
    ]


def test_accelerated_batch_execution_03() -> None:
    """A failed shard leaves a resumable checkpoint; resuming skips finished shards."""
    items = [_item(index) for index in range(8)]
    calls: list[int] = []

    def failing_at_third_shard(
        hamiltonians: NDArray[np.complex128],
        states: NDArray[np.complex128],
        times: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        calls.append(len(calls))
        if len(calls) == 3:
            raise RuntimeError("simulated device loss")
        return evolve_shard_on_cpu(hamiltonians, states, times)

    with pytest.raises(BatchExecutionInterruptedError) as interruption:
        run_batched_evolutions(items, shard_size=2, shard_executor=failing_at_third_shard)
    checkpoint = interruption.value.checkpoint
    assert interruption.value.shard_index == 2
    assert isinstance(interruption.value.__cause__, RuntimeError)
    assert checkpoint.completed_shards == (0, 1)
    assert sorted(checkpoint.states) == ["item-00", "item-01", "item-02", "item-03"]
    saved = {label: state.copy() for label, state in checkpoint.states.items()}

    resumed_calls: list[int] = []

    def counting(
        hamiltonians: NDArray[np.complex128],
        states: NDArray[np.complex128],
        times: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        resumed_calls.append(len(resumed_calls))
        return evolve_shard_on_cpu(hamiltonians, states, times)

    resumed = run_batched_evolutions(
        items, shard_size=2, checkpoint=checkpoint, shard_executor=counting
    )
    assert len(resumed_calls) == 2
    assert resumed.resumed_shards == (0, 1)
    assert [shard.index for shard in resumed.record.shards] == [2, 3]
    uninterrupted = run_batched_evolutions(items, shard_size=2)
    for label, state in uninterrupted.states.items():
        np.testing.assert_array_equal(resumed.states[label], state)
    for label, state in saved.items():
        np.testing.assert_array_equal(checkpoint.states[label], state)


def test_checkpoint_from_another_plan_is_refused() -> None:
    """A checkpoint is bound to its items and shard size."""
    items = [_item(index) for index in range(4)]
    checkpoint = BatchCheckpoint(plan_digest="0" * 64, completed_shards=(0,))
    with pytest.raises(ValueError, match="different batch plan"):
        run_batched_evolutions(items, shard_size=2, checkpoint=checkpoint)


def test_checkpoint_digest_separates_label_boundaries() -> None:
    """Labels ``ab, c`` and ``a, bc`` are different plans for a checkpoint."""

    def failing_second_shard(
        hamiltonians: NDArray[np.complex128],
        states: NDArray[np.complex128],
        times: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        if times[0] == _item(1).time:
            raise RuntimeError("simulated device loss")
        return evolve_shard_on_cpu(hamiltonians, states, times)

    first = [_with(0, label="ab"), _with(1, label="c")]
    with pytest.raises(BatchExecutionInterruptedError) as interruption:
        run_batched_evolutions(first, shard_size=1, shard_executor=failing_second_shard)
    second = [_with(0, label="a"), _with(1, label="bc")]
    with pytest.raises(ValueError, match="different batch plan"):
        run_batched_evolutions(second, shard_size=1, checkpoint=interruption.value.checkpoint)
    resumed = run_batched_evolutions(first, shard_size=1, checkpoint=interruption.value.checkpoint)
    assert resumed.resumed_shards == (0,)
    assert _within_budget(resumed.states["c"], _analytic(1))


def test_accelerated_batch_execution_04() -> None:
    """A GPU request never returns CPU results as accelerator evidence."""
    items = [_item(index) for index in range(3)]
    with pytest.raises(AcceleratorUnavailableError, match="no GPU kernel"):
        run_batched_evolutions(items, device="gpu")
    fallback = run_batched_evolutions(items, device="gpu", allow_cpu_fallback=True)
    assert fallback.record.requested_device == "gpu"
    assert fallback.record.executed_device == "cpu"
    assert fallback.record.fallback is True
    assert fallback.record.accelerator_evidence is False
    with pytest.raises(ValueError, match="'cpu' or 'gpu'"):
        run_batched_evolutions(items, device="tpu")


def _with(index: int, **changes: Any) -> BatchEvolutionItem:
    """Return item ``index`` with fields replaced."""
    base = _item(index)
    fields: dict[str, object] = {
        "label": base.label,
        "hamiltonian": base.hamiltonian,
        "state": base.state,
        "time": base.time,
    }
    fields.update(changes)
    return BatchEvolutionItem(**cast(dict[str, Any], fields))


@pytest.mark.parametrize(
    ("items", "options", "message"),
    [
        ([], {}, "at least one item"),
        ([_item(0), _with(1, label="item-00")], {}, "unique"),
        ([_with(0, label="")], {}, "non-empty"),
        ([_item(0), _with(1, state=np.ones(3, dtype=np.complex128))], {}, "batch dimension"),
        ([_with(0, state=np.ones((2, 1), dtype=np.complex128))], {}, "batch dimension"),
        (
            [_with(0, hamiltonian=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))],
            {},
            "Hermitian",
        ),
        ([_with(0, state=np.array([np.nan, 0.0], dtype=np.complex128))], {}, "finite"),
        ([_with(0, time=math.inf)], {}, "time must be finite"),
        ([_item(0)], {"shard_size": 0}, "shard_size"),
        ([_item(0)], {"shard_size": True}, "shard_size"),
    ],
    ids=[
        "empty",
        "duplicate-label",
        "empty-label",
        "dimension",
        "matrix-state",
        "non-hermitian",
        "nan-state",
        "infinite-time",
        "zero-shard",
        "bool-shard",
    ],
)
def test_malformed_batches_refuse(
    items: list[BatchEvolutionItem], options: dict[str, Any], message: str
) -> None:
    """Malformed items or shard sizes refuse before any kernel runs.

    Parameters
    ----------
    items
        Batch items.
    options
        Keyword options for the run.
    message
        Expected refusal text.

    """
    with pytest.raises(ValueError, match=message):
        run_batched_evolutions(items, **options)


def test_resource_estimates_beyond_the_scan_keep_the_requested_size() -> None:
    """Analytic estimates past the scanned sizes report the request as crossover."""
    estimate = gpu_baseline_comparison(100, trotter_reps=1)
    assert estimate.crossover_n == 100
    assert estimate.statevector_memory_gb > 1.0e20
