# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — declared execution-buffer memory admission
"""Bound simultaneously live buffers before allocation or native dispatch.

Plans describe explicit buffers, not discovered kernel allocations. Snapshot
admission does not reserve memory, measure device capacity, or protect against
another process allocating after the snapshot. A device caller must supply its
actual observed available bytes; missing capacity refuses that device route.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ._rust_accel import optional_rust_engine
from .dense_budget import (
    DEFAULT_DENSE_BUDGET_CAP_GIB,
    GIB,
    DenseAllocationError,
    cgroup_headroom_bytes,
    dense_budget_bytes,
    host_available_memory_bytes,
)

BufferRole = Literal["forward", "intermediate", "adjoint", "dense_output"]


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionBuffer:
    """One named fixed-width numeric buffer with a simultaneously live count.

    Parameters
    ----------
    name
        Unique identity within an execution plan.
    role
        Forward state, intermediate, adjoint tape or requested dense output.
    shape
        Non-empty tuple of positive native-addressable dimensions.
    dtype
        NumPy fixed-width numeric dtype spelling; object/variable storage refuses.
    count
        Number of buffers with this shape live in one job.

    """

    name: str
    role: BufferRole
    shape: tuple[int, ...]
    dtype: str = "complex128"
    count: int = 1

    def __post_init__(self) -> None:
        """Validate metadata and addressability without allocating a buffer."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("buffer name must be non-empty")
        if self.role not in ("forward", "intermediate", "adjoint", "dense_output"):
            raise ValueError("unknown buffer role")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("buffer shape must be a non-empty tuple")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biufc" or dtype.itemsize < 1:
            raise ValueError("dtype must have fixed-width numeric storage")
        _positive(self.count, "count")
        _ = self.bytes_required

    @property
    def bytes_required(self) -> int:
        """Return exact bytes, refusing native-size overflow before each product."""
        size = np.dtype(self.dtype).itemsize
        for dimension in (*self.shape, self.count):
            _positive(dimension, "dimension/count")
            if dimension > sys.maxsize // size:
                raise DenseAllocationError("buffer exceeds native addressable memory")
            size *= dimension
        return size

    @classmethod
    def hilbert(
        cls,
        name: str,
        role: BufferRole,
        n_qubits: int,
        *,
        rank: int = 1,
        dtype: str = "complex128",
        count: int = 1,
    ) -> ExecutionBuffer:
        """Declare a Hilbert buffer with an exponent bound checked before shifting.

        Parameters
        ----------
        name
            Buffer identity.
        role
            Execution stage owning the buffer.
        n_qubits
            Positive qubit count.
        rank
            One for state, two for matrix, four for a superoperator.
        dtype
            Fixed-width numeric dtype spelling.
        count
            Simultaneously live object count.

        Returns
        -------
        ExecutionBuffer
            Addressable metadata; no dense storage is created.

        Raises
        ------
        DenseAllocationError
            If the exponent already exceeds native addressability.

        """
        exponent = _positive(n_qubits, "n_qubits") * _positive(rank, "rank")
        if exponent >= sys.maxsize.bit_length():
            raise DenseAllocationError("Hilbert buffer exceeds native addressable memory")
        return cls(name, role, (1 << n_qubits,) * rank, dtype, count)


@dataclass(frozen=True, slots=True)
class ExecutionMemoryPlan:
    """All declared simultaneously live buffers across concurrent jobs.

    Parameters
    ----------
    buffers
        Immutable buffer declarations with unique names.
    concurrency
        Number of simultaneous instances of this complete job.

    """

    buffers: tuple[ExecutionBuffer, ...]
    concurrency: int = 1

    def __post_init__(self) -> None:
        """Validate identity, immutability, concurrency and total addressability."""
        if not isinstance(self.buffers, tuple) or not self.buffers:
            raise ValueError("buffers must be a non-empty tuple")
        if any(not isinstance(buffer, ExecutionBuffer) for buffer in self.buffers):
            raise TypeError("buffers must contain ExecutionBuffer declarations")
        if len({buffer.name for buffer in self.buffers}) != len(self.buffers):
            raise ValueError("buffer names must be unique")
        _positive(self.concurrency, "concurrency")
        _ = self.bytes_required

    @property
    def bytes_required(self) -> int:
        """Sum every live declaration and concurrency without native-size overflow."""
        total = 0
        for buffer in self.buffers:
            size = buffer.bytes_required
            if size > sys.maxsize - total:
                raise DenseAllocationError("execution plan exceeds native addressable memory")
            total += size
        if self.concurrency > sys.maxsize // total:
            raise DenseAllocationError("concurrent plan exceeds native addressable memory")
        return total * self.concurrency


@dataclass(frozen=True, slots=True)
class MemoryCapacity:
    """Observed byte allowances; ``None`` explicitly preserves unavailable data.

    Parameters
    ----------
    host_available_bytes
        Host free-memory observation, or unknown.
    cgroup_available_bytes
        Visible process-controller headroom, or no finite observed limit.
    device_available_bytes
        Caller-observed free memory on the requested device, or unknown.

    """

    host_available_bytes: int | None
    cgroup_available_bytes: int | None = None
    device_available_bytes: int | None = None

    def __post_init__(self) -> None:
        """Reject malformed observations without coercion or invented capacity."""
        for value in (
            self.host_available_bytes,
            self.cgroup_available_bytes,
            self.device_available_bytes,
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("memory capacity must be integer bytes or None")
                if value < 0:
                    raise ValueError("memory capacity cannot be negative")


@dataclass(frozen=True, slots=True)
class ExecutionMemoryDecision:
    """Snapshot decision for declared buffers, with observed capacity and blockers."""

    bytes_required: int
    budget_bytes: int
    capacity: MemoryCapacity
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject malformed snapshot records before they can charge a reservation."""
        _positive(self.bytes_required, "bytes_required")
        if self.bytes_required > sys.maxsize:
            raise DenseAllocationError("decision exceeds native addressable memory")
        if isinstance(self.budget_bytes, bool) or not isinstance(self.budget_bytes, int):
            raise TypeError("budget_bytes must be integer bytes")
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes cannot be negative")
        if not isinstance(self.capacity, MemoryCapacity):
            raise TypeError("decision capacity must be a MemoryCapacity")
        if not isinstance(self.blockers, tuple) or any(
            not isinstance(blocker, str) or not blocker for blocker in self.blockers
        ):
            raise ValueError("decision blockers must be a tuple of non-empty strings")
        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("decision blockers must be unique")
        if not self.blockers and (
            self.capacity.host_available_bytes is None or self.bytes_required > self.budget_bytes
        ):
            raise ValueError("inconsistent admitted execution memory decision")

    @property
    def allowed(self) -> bool:
        """Whether all declared-memory admission checks succeeded."""
        return not self.blockers


def check_execution_memory(
    plan: ExecutionMemoryPlan,
    capacity: MemoryCapacity,
    *,
    max_bytes: int | None = None,
    require_device: bool = False,
) -> ExecutionMemoryDecision:
    """Compare declared execution memory with an explicit capacity observation.

    Parameters
    ----------
    plan
        Complete declared live-buffer set and concurrency.
    capacity
        Supplied capacity snapshot; this function does not discover hardware.
    max_bytes
        Optional tighter requested cap, which cannot override observed ceilings.
    require_device
        Whether unknown device headroom must refuse this request.

    Returns
    -------
    ExecutionMemoryDecision
        Exact byte decision, never an allocation reservation or device qualification.

    """
    if not isinstance(require_device, bool):
        raise TypeError("require_device must be a boolean")
    if max_bytes is not None:
        _positive(max_bytes, "max_bytes")
    blockers: list[str] = []
    if capacity.host_available_bytes is None:
        blockers.append("host_capacity_unknown")
    if require_device and capacity.device_available_bytes is None:
        blockers.append("device_capacity_unknown")
    observations = [capacity.host_available_bytes, capacity.cgroup_available_bytes]
    if require_device:
        observations.append(capacity.device_available_bytes)
    available = min((value for value in observations if value is not None), default=0)
    cap = int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB) if max_bytes is None else max_bytes
    budget = min(cap, available * 3 // 10)
    required = plan.bytes_required
    if required > budget:
        blockers.append("declared_buffers_exceed_budget")
    return ExecutionMemoryDecision(required, budget, capacity, tuple(blockers))


def require_execution_memory(
    plan: ExecutionMemoryPlan,
    *,
    max_gib: float | None = None,
    cgroup_root: Path | None = None,
    device_available_bytes: int | None = None,
    require_device: bool = False,
    native_symbol: str | None = None,
) -> ExecutionMemoryDecision:
    """Read real process capacity and refuse before allocation/native dispatch.

    Parameters
    ----------
    plan
        Complete declared simultaneously live buffers and job count.
    max_gib
        Optional requested cap; existing environment/default policy still applies.
    cgroup_root
        Optional explicit controller snapshot root; default discovers membership.
    device_available_bytes
        Actual device observation supplied by the device-owning caller.
    require_device
        Refuse when the device observation is absent.
    native_symbol
        Required callable Rust entry, or None for no native requirement.

    Returns
    -------
    ExecutionMemoryDecision
        Accepted snapshot for declared buffers, not a reservation.

    Raises
    ------
    DenseAllocationError
        Unknown/insufficient capacity or missing requested native callable.

    """
    if native_symbol is not None:
        if not native_symbol or not isinstance(native_symbol, str):
            raise ValueError("native_symbol must be a non-empty string")
        engine = optional_rust_engine()
        if not callable(getattr(engine, native_symbol, None)):
            raise DenseAllocationError("requested native entry is unavailable; fallback refused")
    capacity = MemoryCapacity(
        host_available_memory_bytes(),
        cgroup_headroom_bytes(cgroup_root),
        device_available_bytes,
    )
    requested_bytes = dense_budget_bytes(max_gib)
    if requested_bytes == 0:
        raise DenseAllocationError("execution memory budget is below one byte")
    decision = check_execution_memory(
        plan, capacity, max_bytes=requested_bytes, require_device=require_device
    )
    if not decision.allowed:
        raise DenseAllocationError("execution memory refused: " + ", ".join(decision.blockers))
    return decision


def dataclass_storage_bytes(
    record_type: type[object], *, payload_bytes: int = 0, count: int = 1
) -> int:
    """Declare interpreter storage for fixed-schema records and separate payloads.

    Parameters
    ----------
    record_type
        Actual dataclass type whose instance and field storage is declared.
    payload_bytes
        Additional bytes per record for strings, references or nested payloads.
    count
        Simultaneously retained records with this declaration.

    Returns
    -------
    int
        Checked bytes based on interpreter object sizes and per-field mappings.
        Shared fields are conservatively charged separately. General allocator
        overhead and undeclared nested payloads are outside this declaration.

    Raises
    ------
    TypeError
        Non-dataclass schema or noninteger payload/count metadata.
    ValueError
        Negative payload bytes or nonpositive count.
    DenseAllocationError
        Declared storage exceeds native addressability.

    """
    if not isinstance(record_type, type) or not is_dataclass(record_type):
        raise TypeError("record_type must be a dataclass type")
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
        raise TypeError("payload_bytes must be integer bytes")
    if payload_bytes < 0:
        raise ValueError("payload_bytes cannot be negative")
    _positive(count, "count")
    instance = object.__new__(record_type)
    storage = (
        sys.getsizeof(instance)
        + sys.getsizeof(getattr(instance, "__dict__", {}))
        + len(fields(record_type)) * sys.getsizeof({"": None})
        + payload_bytes
    )
    return ExecutionBuffer(
        "record_storage", "intermediate", (storage,), "uint8", count
    ).bytes_required


def json_encoded_bytes(value: object, *, max_bytes: int = sys.maxsize) -> int:
    """Count compact ASCII JSON bytes without materialising the encoded document.

    Parameters
    ----------
    value
        Primitive scalar, string-keyed dictionary, list or tuple. Integer values
        must be native-addressable; cyclic or protocol-bearing objects refuse.
    max_bytes
        Positive requested encoded-byte cap, bounded by native addressability.

    Returns
    -------
    int
        Exact bytes for ensure_ascii=True and compact JSON separators.

    Raises
    ------
    TypeError
        Unsupported object or nonstring dictionary key.
    ValueError
        Cyclic container graph or nesting beyond the interpreter limit.
    DenseAllocationError
        Integer or encoded size exceeds native addressability or the requested cap.

    """
    limit = min(_positive(max_bytes, "max_bytes"), sys.maxsize)
    active: set[int] = set()

    def add(size: int, increment: int) -> int:
        if increment > limit - size:
            raise DenseAllocationError(
                "JSON storage exceeds requested/native addressability budget"
            )
        return size + increment

    def count(item: object) -> int:
        if type(item) is str:
            size = add(0, 2)
            for character in item:
                codepoint = ord(character)
                if codepoint in (34, 92, 8, 9, 10, 12, 13):
                    size = add(size, 2)
                elif codepoint < 32 or 126 < codepoint <= 65535:
                    size = add(size, 6)
                elif codepoint > 65535:
                    size = add(size, 12)
                else:
                    size = add(size, 1)
            return size
        if item is None or type(item) in (bool, float):
            return add(0, len(json.dumps(item)))
        if type(item) is int:
            if not -sys.maxsize - 1 <= item <= sys.maxsize:
                raise DenseAllocationError("JSON integer exceeds native addressability")
            return add(0, len(str(item)))
        if type(item) not in (dict, list, tuple):
            raise TypeError("unsupported JSON storage object")
        assert isinstance(item, (dict, list, tuple))
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic JSON storage object")
        active.add(identity)
        try:
            size = add(2, max(0, len(item) - 1))
            if isinstance(item, dict):
                for key, entry in item.items():
                    if type(key) is not str:
                        raise TypeError("JSON storage keys must be strings")
                    size = add(size, count(key))
                    size = add(size, 1)
                    size = add(size, count(entry))
            else:
                for entry in item:
                    size = add(size, count(entry))
            return size
        finally:
            active.remove(identity)

    try:
        size = count(value)
    except RecursionError as exc:
        raise ValueError("JSON storage nesting exceeds the interpreter limit") from exc
    return size
