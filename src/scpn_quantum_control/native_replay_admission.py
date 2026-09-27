# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Native replay input-conversion ownership
"""Admit native replay input copies before PyO3 extraction.

This boundary declares source and numeric input conversion storage and charges
numeric, parser metadata and output declarations received from Rust. These are
conservative owned declarations; vendor allocator overhead and caller-retained
objects still require separate qualification.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import cast

import numpy as np

from .dense_budget import DenseAllocationError
from .execution_memory import BufferRole, ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory

_NATIVE_REAL_SCALAR_TYPES = frozenset(
    (
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    )
)


@contextmanager
def native_replay_input_scope(
    serialization: object, inputs: object = None
) -> Iterator[ExecutionMemoryReservation]:
    """Own UTF-8 and numeric copies before native parsing or sequence extraction.

    Parameters
    ----------
    serialization
        Plain non-empty string containing Program AD IR metadata.
    inputs
        Plain list, tuple, range or one-dimensional real numeric ndarray;
        absent for metadata-only parsing. Opaque iterators refuse.

    Yields
    ------
    ExecutionMemoryReservation
        Shared process/thread reservation inheriting parent lifecycle policy.

    Raises
    ------
    ValueError
        Source or numeric input metadata does not meet the static contract.
    DenseAllocationError
        Input copy storage exceeds native addressability or current capacity.

    """
    if type(serialization) is not str or not serialization:
        raise ValueError("native Program AD serialization must be a plain non-empty string")
    buffers = [
        ExecutionBuffer(
            "native_replay_source_copies", "forward", (len(serialization), 4), "uint8", 2
        )
    ]
    if inputs is not None:
        if type(inputs) is np.ndarray:
            if inputs.ndim != 1 or inputs.dtype.kind not in "iuf":
                raise ValueError("native Program AD inputs require a one-dimensional real array")
            size = int(inputs.size)
            width = max(8, inputs.dtype.itemsize)
        elif type(inputs) in (list, tuple, range):
            try:
                size = len(cast(Sequence[object], inputs))
            except OverflowError as error:
                raise DenseAllocationError(
                    "native Program AD input length exceeds addressability"
                ) from error
            width = 8
        else:
            raise ValueError("native Program AD inputs require a plain static numeric sequence")
        buffers.extend(
            (
                ExecutionBuffer(
                    "native_replay_numeric_copies", "forward", (max(1, size), width), "uint8", 2
                ),
                ExecutionBuffer(
                    "native_replay_boxed_numbers",
                    "intermediate",
                    (max(1, size), sys.getsizeof(0.0)),
                    "uint8",
                ),
            )
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        if type(inputs) in (list, tuple):
            for value in cast(Sequence[object], inputs):
                reservation.checkpoint()
                if type(value) not in _NATIVE_REAL_SCALAR_TYPES:
                    raise ValueError("native Program AD inputs must contain real numeric scalars")
        yield reservation
        reservation.checkpoint()


def _admit_native_replay_workspace(
    reservation: ExecutionMemoryReservation,
    input_bytes: int,
    forward_bytes: int,
    adjoint_bytes: int,
    intermediate_bytes: int,
) -> None:
    """Charge cumulative native declarations in addition to retained input copies."""
    if isinstance(input_bytes, bool) or not isinstance(input_bytes, int) or input_bytes < 1:
        raise ValueError("native replay input storage must be positive integer bytes")
    buffers = [
        ExecutionBuffer(
            "native_replay_input_storage",
            "forward",
            (input_bytes,),
            "uint8",
        )
    ]
    declarations: tuple[tuple[str, BufferRole, int], ...] = (
        ("native_replay_retained_values", "forward", forward_bytes),
        ("native_replay_retained_adjoints", "adjoint", adjoint_bytes),
        ("native_replay_declared_intermediates", "intermediate", intermediate_bytes),
    )
    for name, role, size in declarations:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                "native replay storage declarations must be non-negative integer bytes"
            )
        if size:
            buffers.append(ExecutionBuffer(name, role, (size,), "uint8"))
    reservation.resize(ExecutionMemoryPlan(tuple(buffers)))


def _native_replay_python_string_bytes(encoded_bytes: int) -> int:
    """Declare runtime Unicode header and worst-width payload before conversion."""
    if isinstance(encoded_bytes, bool) or not isinstance(encoded_bytes, int) or encoded_bytes < 1:
        raise ValueError("native JSON bytes must be positive integer bytes")
    header = max(sys.getsizeof(value) for value in ("", "a", "\u0080", "\u0100", "\U00010000"))
    return ExecutionBuffer(
        "native_replay_python_string", "intermediate", (header + 4 * encoded_bytes,), "uint8"
    ).bytes_required
