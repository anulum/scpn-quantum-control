# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — execution-memory admission tests
"""Check declared buffer lifetimes against independent integer byte oracles."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_memory import (
    ExecutionBuffer,
    ExecutionMemoryDecision,
    ExecutionMemoryPlan,
    MemoryCapacity,
    check_execution_memory,
    dataclass_storage_bytes,
    json_encoded_bytes,
    require_execution_memory,
)


def test_all_live_roles_and_concurrent_jobs_are_charged() -> None:
    """Sum float32 forward, complex intermediates, tape and exported matrices."""
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer("state", "forward", (8,), "float32"),
            ExecutionBuffer("temporary", "intermediate", (8,), "complex128", 2),
            ExecutionBuffer("tape", "adjoint", (3, 8), "float64"),
            ExecutionBuffer("output", "dense_output", (8, 8), "complex64"),
        ),
        concurrency=3,
    )
    expected = 3 * (8 * 4 + 2 * 8 * 16 + 3 * 8 * 8 + 8 * 8 * 8)
    assert plan.bytes_required == expected
    decision = check_execution_memory(plan, MemoryCapacity(20_000, 10_000), max_bytes=expected)
    assert decision.allowed
    assert decision.bytes_required == expected
    assert decision.budget_bytes == expected
    tighter = check_execution_memory(plan, MemoryCapacity(10_000, 9_000), max_bytes=expected)
    assert not tighter.allowed
    assert tighter.budget_bytes == 2700
    assert tighter.bytes_required == expected


@pytest.mark.parametrize("delta, allowed", [(-1, True), (0, True), (1, False)])
def test_inclusive_exact_byte_boundary(delta: int, allowed: bool) -> None:
    """Admit B-1 and B while refusing B+1 without floating-point rounding."""
    plan = ExecutionMemoryPlan(
        (ExecutionBuffer("output", "dense_output", (128 + delta,), "uint8"),)
    )
    decision = check_execution_memory(plan, MemoryCapacity(4096), max_bytes=128)
    assert decision.allowed is allowed
    assert decision.bytes_required == 128 + delta
    assert decision.budget_bytes == 128
    assert bool(decision.blockers) is not allowed


@pytest.mark.parametrize("shape", [(sys.maxsize, 2), (10**100, 1)])
def test_unaddressable_shape_refuses_before_materialisation(shape: tuple[int, ...]) -> None:
    """Reject impossible native dimensions using only bounded integer arithmetic."""
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionBuffer("state", "forward", shape, "uint8")


def test_impossible_hilbert_dimension_never_builds_an_exponential_integer() -> None:
    """Reject absurd exponent and object multiplicity before exponentiation."""
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionBuffer.hilbert("output", "dense_output", 10**100, rank=2)
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionBuffer.hilbert("output", "dense_output", 1, count=sys.maxsize)


def test_sum_and_concurrency_cannot_overflow_native_address_size() -> None:
    """Bound the total even when each declared buffer is individually addressable."""
    buffer = ExecutionBuffer("a", "forward", (sys.maxsize // 2 + 1,), "uint8")
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionMemoryPlan((buffer,), concurrency=2)
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionMemoryPlan((buffer, ExecutionBuffer("b", "adjoint", buffer.shape, "uint8")))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_dimension_and_concurrency_are_not_coerced(value: object) -> None:
    """Reject booleans, fractions and nonpositive sizes at the public constructor."""
    with pytest.raises((TypeError, ValueError)):
        ExecutionBuffer("state", "forward", cast(Any, (value,)), "float64")
    with pytest.raises((TypeError, ValueError)):
        ExecutionMemoryPlan(
            (ExecutionBuffer("state", "forward", (1,), "float64"),), cast(Any, value)
        )


def test_unsupported_buffer_metadata_refuses() -> None:
    """Reject ambiguous names, roles, shapes, object dtype and duplicate identity."""
    for name, role, shape, dtype in [
        ("", "forward", (1,), "float64"),
        ("state", "unknown", (1,), "float64"),
        ("state", "forward", (), "float64"),
        ("state", "forward", (1,), "object"),
        ("state", "forward", (1,), "V0"),
    ]:
        with pytest.raises(ValueError):
            ExecutionBuffer(name, cast(Any, role), shape, dtype)
    buffer = ExecutionBuffer("state", "forward", (1,), "float64")
    with pytest.raises(ValueError, match="unique"):
        ExecutionMemoryPlan((buffer, buffer))
    with pytest.raises(ValueError, match="non-empty"):
        ExecutionMemoryPlan(())


def test_unknown_host_and_device_capacity_refuse() -> None:
    """An explicit requested cap cannot turn unknown capacity into admission."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (2,), "float64"),))
    unknown = check_execution_memory(plan, MemoryCapacity(None, 1000), max_bytes=100)
    assert not unknown.allowed and "host_capacity_unknown" in unknown.blockers
    device = check_execution_memory(plan, MemoryCapacity(1000), require_device=True)
    assert not device.allowed and "device_capacity_unknown" in device.blockers
    device_ok = check_execution_memory(plan, MemoryCapacity(1000, 900, 64), require_device=True)
    assert device_ok.allowed and device_ok.budget_bytes == 19


def test_snapshot_policy_does_not_override_visible_process_ceiling() -> None:
    """Bound admission by thirty percent of the tightest observed allowance."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (31,), "uint8"),))
    decision = check_execution_memory(plan, MemoryCapacity(1000, 100), max_bytes=10_000)
    assert not decision.allowed and decision.budget_bytes == 30
    assert not check_execution_memory(plan, MemoryCapacity(1000, 0)).allowed


def test_live_small_plan_uses_real_host_and_cgroup_snapshot() -> None:
    """Exercise the public runtime admission on actual host metadata without allocating."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (2,), "float64"),))
    decision = require_execution_memory(plan, max_gib=1 / 1024)
    assert decision.allowed and decision.bytes_required == 16
    assert decision.capacity.host_available_bytes is not None


def test_unknown_capacity_does_not_fall_back_to_default_gib() -> None:
    """A complete absence of observations returns refusal with zero allowance."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (2,), "float64"),))
    decision = check_execution_memory(plan, MemoryCapacity(None))
    assert not decision.allowed and decision.budget_bytes == 0
    assert decision.blockers == ("host_capacity_unknown", "declared_buffers_exceed_budget")


@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_invalid_capacity_observations_refuse(value: object) -> None:
    """All capacity fields reject negative, boolean or fractional observations."""
    for capacity in [(value, None, None), (1000, value, None), (1000, None, value)]:
        with pytest.raises((TypeError, ValueError)):
            MemoryCapacity(*cast(Any, capacity))


def test_invalid_count_policy_and_native_request_refuse() -> None:
    """Invalid control metadata cannot widen admission or select a fallback route."""
    with pytest.raises(ValueError, match="count"):
        ExecutionBuffer("state", "forward", (2,), "float64", count=0)
    buffer = ExecutionBuffer("state", "forward", (2,), "float64")
    with pytest.raises(TypeError, match="ExecutionBuffer"):
        ExecutionMemoryPlan(cast(Any, ("state",)))
    plan = ExecutionMemoryPlan((buffer,))
    with pytest.raises(TypeError, match="boolean"):
        check_execution_memory(plan, MemoryCapacity(1000), require_device=cast(Any, "yes"))
    with pytest.raises(TypeError, match="max_bytes"):
        check_execution_memory(plan, MemoryCapacity(1000), max_bytes=True)
    with pytest.raises(ValueError, match="native_symbol"):
        require_execution_memory(plan, native_symbol="")


@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_snapshot_decision_rejects_malformed_numeric_charges(value: object) -> None:
    """Malformed records cannot reduce or corrupt the shared byte ledger."""
    capacity = MemoryCapacity(1000)
    with pytest.raises((TypeError, ValueError)):
        ExecutionMemoryDecision(cast(Any, value), 100, capacity, ())
    with pytest.raises((TypeError, ValueError)):
        ExecutionMemoryDecision(16, cast(Any, value), capacity, ())


def test_snapshot_decision_rejects_inconsistent_admission_metadata() -> None:
    """Refusal reasons cannot be erased to admit unknown or insufficient capacity."""
    with pytest.raises(ValueError, match="inconsistent"):
        ExecutionMemoryDecision(16, 15, MemoryCapacity(1000), ())
    with pytest.raises(ValueError, match="inconsistent"):
        ExecutionMemoryDecision(16, 100, MemoryCapacity(None), ())
    with pytest.raises(DenseAllocationError, match="addressable"):
        ExecutionMemoryDecision(sys.maxsize + 1, sys.maxsize + 1, MemoryCapacity(1000), ())
    with pytest.raises(TypeError, match="MemoryCapacity"):
        ExecutionMemoryDecision(16, 100, cast(Any, None), ())
    for blockers in [[], ("",), ("refused", "refused")]:
        with pytest.raises(ValueError, match="blockers"):
            ExecutionMemoryDecision(16, 100, MemoryCapacity(1000), cast(Any, blockers))
    refusal = ExecutionMemoryDecision(16, 0, MemoryCapacity(None), ("host_capacity_unknown",))
    assert not refusal.allowed


def test_record_storage_declaration_covers_real_ssa_schema() -> None:
    """Declare actual production SSA records using independent object-size bounds."""
    from scpn_quantum_control.program_ad_effect_ir import ProgramADSSAValue

    record = ProgramADSSAValue("%0", 0, 0, (), "float64")
    declared = dataclass_storage_bytes(ProgramADSSAValue, payload_bytes=128)
    assert declared >= sys.getsizeof(record) + sys.getsizeof(vars(record)) + 128
    assert dataclass_storage_bytes(ProgramADSSAValue, payload_bytes=128, count=3) >= 3 * (
        sys.getsizeof(record) + sys.getsizeof(vars(record)) + 128
    )
    with pytest.raises(DenseAllocationError, match="addressable"):
        dataclass_storage_bytes(ProgramADSSAValue, payload_bytes=sys.maxsize)
    with pytest.raises(TypeError, match="dataclass"):
        dataclass_storage_bytes(str)
    with pytest.raises(TypeError, match="integer"):
        dataclass_storage_bytes(ProgramADSSAValue, payload_bytes=True)
    with pytest.raises(ValueError, match="negative"):
        dataclass_storage_bytes(ProgramADSSAValue, payload_bytes=-1)
    with pytest.raises(ValueError, match="count"):
        dataclass_storage_bytes(ProgramADSSAValue, count=0)


def test_json_storage_matches_actual_compact_ascii_wire() -> None:
    """The independent standard encoder agrees on escaping and nested wire sizes."""
    payload: dict[str, object] = {
        "channel": 'ω\n𝄞\t\x00"',
        "values": [None, True, False, -1, 0.25],
        "shape": (2, 3),
        "empty": {},
        "controls": "".join(chr(codepoint) for codepoint in range(128)),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    assert json_encoded_bytes(payload) == len(encoded.encode("ascii"))
    with pytest.raises(DenseAllocationError, match="addressability"):
        json_encoded_bytes(sys.maxsize + 1)
    with pytest.raises(TypeError, match="unsupported"):
        json_encoded_bytes(object())
    with pytest.raises(TypeError, match="keys"):
        json_encoded_bytes({1: "value"})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        json_encoded_bytes(cyclic)


def test_json_storage_refuses_unaddressable_container_nesting() -> None:
    """A real deeply nested container returns a typed refusal instead of recursing unchecked."""
    root: list[object] = []
    current = root
    for _ in range(sys.getrecursionlimit() + 1):
        child: list[object] = []
        current.append(child)
        current = child
    with pytest.raises(ValueError, match="nesting"):
        json_encoded_bytes(root)


@pytest.mark.parametrize(
    "payload", ["ω\n𝄞", {"k": [1, True, None]}, [1, 2], (3, 4), 50, False, None, 0.5]
)
def test_json_encoded_budget_uses_exact_wire_boundary(payload: object) -> None:
    """Real compact JSON bytes admit B and B+1 while refusing a B-1 cap."""
    size = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
    for cap in (size, size + 1):
        assert json_encoded_bytes(payload, max_bytes=cap) == size
    with pytest.raises(DenseAllocationError, match="budget"):
        json_encoded_bytes(payload, max_bytes=size - 1)


@pytest.mark.parametrize("cap", [True, 0, -1])
def test_json_encoded_budget_rejects_invalid_cap(cap: int) -> None:
    """Malformed byte caps cannot silently widen JSON output admission."""
    with pytest.raises((TypeError, ValueError), match="max_bytes"):
        json_encoded_bytes({}, max_bytes=cap)


def test_json_encoded_bytes_refuses_protocol_bearing_container() -> None:
    """A custom container cannot underdeclare the standard encoder's actual output."""

    class MisreportedList(list[int]):
        """Expose a length protocol inconsistent with the actual JSON payload."""

        def __len__(self) -> int:
            """Report an empty declaration while retaining actual entries."""
            return 0

    payload = MisreportedList([1, 2, 3])
    assert json.dumps(payload, separators=(",", ":")) == "[1,2,3]"
    with pytest.raises(TypeError, match="unsupported"):
        json_encoded_bytes(payload)


def test_live_admission_refuses_subbyte_budget_and_exhausted_controller(tmp_path: Path) -> None:
    """Real budget rounding and controller files cannot authorize an empty allowance."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (2,), "float64"),))
    with pytest.raises(DenseAllocationError, match="below one byte"):
        require_execution_memory(plan, max_gib=1e-30)
    (tmp_path / "memory.max").write_text("4096")
    (tmp_path / "memory.current").write_text("4096")
    with pytest.raises(DenseAllocationError, match="refused"):
        require_execution_memory(plan, cgroup_root=tmp_path)
