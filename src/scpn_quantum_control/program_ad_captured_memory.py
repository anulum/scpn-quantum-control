# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — captured state workspace admission
"""Declare traversal storage before identity tables and reference lists grow."""

from __future__ import annotations

import struct
import sys
from bisect import bisect_right
from types import CodeType
from typing import cast

from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import ExecutionMemoryReservation

_CAPTURED_STATE_MAX_NODES = 4096
_POINTER_BYTES = struct.calcsize("P")
_IDENTITY_BYTES = sys.getsizeof((1 << (_POINTER_BYTES * 8)) - 1)
_SCOPE_BYTES = sys.getsizeof((0, ())) + _IDENTITY_BYTES
_EMPTY_TUPLE_BYTES = sys.getsizeof(())
_CODE_DEPTHS = 65
_EMPTY_CODE_SIZES = (0,) * _CODE_DEPTHS
_CODE_ENTRY_BYTES = (
    sys.getsizeof({0: 0}) + sys.getsizeof({0}) + 3 * _IDENTITY_BYTES + 2 * _POINTER_BYTES
)


class _CapturedCodeRefusal(ValueError):
    """Captured code cannot be inspected without an unsupported protocol."""


def _constant_representation_bytes(
    value: object,
    memory: _CapturedStateMemory,
    function_depth: int,
    depth: int = 0,
) -> int:
    """Bound immutable constant formatting without executing object protocols."""
    if depth >= _CODE_DEPTHS:
        raise _CapturedCodeRefusal("captured code constant nesting is unsupported")
    memory.visit(function_depth + depth + 2)
    kind = type(value)
    if value is None or value is Ellipsis or kind is bool:
        return 8
    if kind is int:
        number = cast(int, value)
        return (number.bit_length() * 30104 + 99999) // 100000 + 3
    if kind is float:
        return 32
    if kind is complex:
        return 72
    if kind is str or kind is bytes:
        return 40 * len(cast(str | bytes, value)) + 8
    if kind is tuple or kind is frozenset:
        items = cast(tuple[object, ...] | frozenset[object], value)
        return 16 + sum(
            _constant_representation_bytes(item, memory, function_depth, depth + 1) + 2
            for item in items
        )
    if kind is CodeType:
        code = cast(CodeType, value)
        for item in code.co_consts:
            _constant_representation_bytes(item, memory, function_depth, depth + 1)
        return 4 * (len(code.co_name) + len(code.co_filename)) + 256
    raise _CapturedCodeRefusal("captured code constant storage is unsupported")


def _container_storage_sizes() -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    """Measure this interpreter's insertion-only set and append-only list tables.

    Shared layout metadata is collected once at module import. Temporary
    calibration containers are released; caller captures are never inspected.
    """
    identities: set[int] = set()
    references: list[None] = []
    set_counts: list[int] = []
    set_bytes: list[int] = []
    list_counts: list[int] = []
    list_bytes: list[int] = []
    for count in range(_CAPTURED_STATE_MAX_NODES + 1):
        set_size = sys.getsizeof(identities)
        list_size = sys.getsizeof(references)
        if not set_bytes or set_bytes[-1] != set_size:
            set_counts.append(count)
            set_bytes.append(set_size)
        if not list_bytes or list_bytes[-1] != list_size:
            list_counts.append(count)
            list_bytes.append(list_size)
        identities.add(count)
        references.append(None)
    return tuple(set_counts), tuple(set_bytes), tuple(list_counts), tuple(list_bytes)


_SET_COUNTS, _SET_BYTES, _LIST_COUNTS, _LIST_BYTES = _container_storage_sizes()


class _CapturedStateMemory:
    """Own prospective table, reference and traversal-frame storage."""

    __slots__ = (
        "reservation",
        "digest_bytes",
        "fingerprint_bytes",
        "frame_bytes",
        "seen_capacity",
        "scope_capacity",
        "reference_capacity",
        "depth_capacity",
        "code_sizes",
        "code_bytes",
        "charged_bytes",
    )

    def __init__(
        self,
        reservation: ExecutionMemoryReservation,
        digest_bytes: int,
        fingerprint_bytes: int,
        frame_bytes: int,
    ) -> None:
        """Admit empty traversal structures before creating their containers."""
        self.reservation = reservation
        self.digest_bytes = digest_bytes
        self.fingerprint_bytes = fingerprint_bytes
        self.frame_bytes = frame_bytes
        self.seen_capacity = 0
        self.scope_capacity = 0
        self.reference_capacity = 0
        self.depth_capacity = 0
        self.code_sizes = _EMPTY_CODE_SIZES
        self.code_bytes = 0
        self.charged_bytes = 0
        self._admit()

    def visit(self, depth: int) -> None:
        """Retain enough stack capacity before descending into a captured value."""
        self.depth_capacity = max(self.depth_capacity, depth + 1)
        self._admit()

    def seen(self, count: int) -> None:
        """Admit prospective identity-set table and integer-key storage."""
        self.seen_capacity = max(self.seen_capacity, self._capacity(count))
        self._admit()

    def scope(self, count: int) -> None:
        """Admit module-scope tuples and their insertion-only set table."""
        self.scope_capacity = max(self.scope_capacity, self._capacity(count))
        self._admit()

    def reference(self, count: int) -> None:
        """Admit code-reference list growth and the simultaneously returned tuple."""
        self.reference_capacity = max(self.reference_capacity, self._capacity(count))
        self._admit()

    def code(self, code: CodeType, depth: int) -> None:
        """Admit disassembly tables and formatting before standard inspection.

        Line/label tables, collected global names and iterator state are bounded
        by their immutable bytecode/line/exception metadata. Constants must be
        compiler-native immutable values; an opaque representation is refused
        before ``dis.get_instructions`` can call it. Per-depth high-water charges
        retain a caller's name tuple while visiting a nested captured function.
        """
        size = 4096 + _CODE_ENTRY_BYTES * (
            len(code.co_code) // 2 + len(code.co_linetable) + len(code.co_exceptiontable)
        )
        self._code_size(size, depth)
        largest = max(
            (_constant_representation_bytes(value, self, depth) for value in code.co_consts),
            default=0,
        )
        for names in (code.co_names, code.co_varnames, code.co_cellvars, code.co_freevars):
            largest = max(largest, max((4 * len(name) + 16 for name in names), default=0))
        self._code_size(size + 4 * largest, depth)

    def _code_size(self, size: int, depth: int) -> None:
        """Charge prospective code storage before updating bounded bookkeeping."""
        previous = self.code_sizes[depth]
        if size > previous:
            self.code_bytes += size - previous
            self._admit()
            self.code_sizes = (*self.code_sizes[:depth], size, *self.code_sizes[depth + 1 :])
        self.reservation.checkpoint()

    @staticmethod
    def _capacity(count: int) -> int:
        """Grow small tables in bounded batches without a host query per entry."""
        return min(_CAPTURED_STATE_MAX_NODES, ((count + 31) // 32) * 32)

    def _admit(self) -> None:
        """Grow the one owned reservation; a refusal preserves its prior charge."""
        required = (
            self.digest_bytes
            + self.fingerprint_bytes
            + sys.getsizeof(self)
            + _SET_BYTES[bisect_right(_SET_COUNTS, self.seen_capacity) - 1]
            + self.seen_capacity * _IDENTITY_BYTES
            + _SET_BYTES[bisect_right(_SET_COUNTS, self.scope_capacity) - 1]
            + self.scope_capacity * _SCOPE_BYTES
            + _LIST_BYTES[bisect_right(_LIST_COUNTS, self.reference_capacity) - 1]
            + _EMPTY_TUPLE_BYTES
            + self.reference_capacity * _POINTER_BYTES
            + self.depth_capacity * self.frame_bytes
            + self.code_bytes
            + 3 * sys.getsizeof(_EMPTY_CODE_SIZES)
            + _CODE_DEPTHS * _IDENTITY_BYTES
        )
        if required > self.charged_bytes:
            plan = ExecutionMemoryPlan(
                (
                    ExecutionBuffer(
                        "captured_state_workspace", "intermediate", (required,), "uint8"
                    ),
                )
            )
            self.reservation.resize(plan)
            self.charged_bytes = required
        self.reservation.checkpoint()
