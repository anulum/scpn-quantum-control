# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD tape binding
"""Check the content of a runtime tape without changing its historical codec.

The digest accompanies the live callable binding. It detects changes between
observations; it neither authenticates manually reconstructed records nor locks
caller-owned storage. Standalone historical records have no live binding.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from collections.abc import Callable
from dataclasses import fields

import numpy as np
from numpy.typing import NDArray

from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import reserve_execution_memory
from .program_ad_captured_state import _CAPTURED_STATE_DIGEST_BYTES
from .program_ad_effect_ir import ProgramADEffectIR, parse_program_ad_effect_ir
from .whole_program_ad_result import WholeProgramADResult

_TAPE_DIGEST_BYTES = _CAPTURED_STATE_DIGEST_BYTES


class _MissingCapturedAdjoint(ValueError):
    """Refuse a bound tape whose required adjoint metadata was removed."""


class _UnsupportedCapturedAdjoint(ValueError):
    """Refuse a bound tape whose supported adjoint contract was replaced."""


def _bind_program_tape(result: WholeProgramADResult, checkpoint: Callable[[], None]) -> None:
    """Seal the private binding once, before the runtime result is exposed."""
    binding = result.captured_state
    if binding is None:
        raise ValueError("captured derivative tape requires an unbound runtime capture")
    digest = _program_tape_digest(result, checkpoint)
    object.__setattr__(binding, "tape_digest", digest)


def _require_bound_program_tape(result: WholeProgramADResult) -> None:
    """Check tape content and observed live state before exposing a derivative."""
    binding = result.captured_state
    if binding is None:
        return
    if binding.tape_digest is None or _program_tape_digest(result) != binding.tape_digest:
        if result.adjoint_result is None:
            raise _MissingCapturedAdjoint(
                "captured derivative tape changed after program capture: missing adjoint metadata"
            )
        if result.adjoint_result.supported is False:
            raise _UnsupportedCapturedAdjoint(
                "captured derivative tape changed after program capture: unsupported adjoint metadata"
            )
        raise ValueError("captured derivative tape changed after program capture")
    binding.require_current()


def _require_program_ad_ir_serialization(program_ir: ProgramADEffectIR) -> None:
    """Require a numerical typed-IR input to describe its actual wire payload."""
    parsed = parse_program_ad_effect_ir(program_ir.serialization)
    if parsed != program_ir:
        raise ValueError("program AD IR typed rows do not match serialization")


def _program_tape_digest(
    result: WholeProgramADResult, checkpoint: Callable[[], None] | None = None
) -> bytes:
    from .program_ad_adjoint import ProgramADAdjointResult, ProgramADAdjointStep
    from .program_ad_effect_ir import (
        ProgramADAliasEdge,
        ProgramADControlRegion,
        ProgramADEffect,
        ProgramADPhiNode,
        ProgramADSSAValue,
    )
    from .whole_program_ad_result import WholeProgramIRNode

    record_types = (
        ProgramADAdjointResult,
        ProgramADAdjointStep,
        ProgramADEffectIR,
        ProgramADAliasEdge,
        ProgramADControlRegion,
        ProgramADEffect,
        ProgramADPhiNode,
        ProgramADSSAValue,
        WholeProgramIRNode,
    )
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer("tape_digest", "intermediate", (_TAPE_DIGEST_BYTES,), "uint8"),
            ExecutionBuffer(
                "tape_inspection_stack", "intermediate", (64 * sys.getsizeof((None,)),), "uint8"
            ),
        )
    )
    with reserve_execution_memory(plan) as reservation:
        digest = hashlib.sha256()
        copy_capacity = -1

        def visit(value: object, depth: int) -> None:
            reservation.checkpoint()
            if checkpoint is not None:
                checkpoint()
            if depth > 64:
                raise ValueError("captured derivative tape contains unsupported nesting")
            if value is None:
                digest.update(b"none;")
            elif type(value) is bool:
                digest.update(b"true;" if value else b"false;")
            elif type(value) is int:
                digest.update(b"integer;")
                length = max(1, (value.bit_length() + 8) // 8)
                payload(value, length)
            elif type(value) is float:
                digest.update(b"float;" + struct.pack("!d", value))
            elif type(value) is str:
                digest.update(b"string;")
                capacity = len(value)
                if not value.isascii():
                    capacity = 0
                    for index, character in enumerate(value):
                        if index % 4096 == 0:
                            reservation.checkpoint()
                        ordinal = ord(character)
                        capacity += (
                            1
                            if ordinal < 0x80
                            else 2
                            if ordinal < 0x800
                            else 3
                            if ordinal < 0x10000
                            else 4
                        )
                payload(value, capacity)
            elif type(value) is tuple:
                digest.update(b"tuple;" + struct.pack("!Q", len(value)))
                for item in value:
                    visit(item, depth + 1)
            elif type(value) is np.ndarray:
                digest.update(b"array;")
                visit(value.dtype.str, depth + 1)
                visit(value.shape, depth + 1)
                if value.dtype != np.dtype(np.float64):
                    raise ValueError("captured derivative tape requires float64 storage")
                payload(value, value.nbytes)
            elif type(value) in record_types:
                digest.update(type(value).__name__.encode("ascii"))
                record_type = record_types[record_types.index(type(value))]
                for field in fields(record_type):
                    if field.name != "captured_state":
                        visit(field.name, depth + 1)
                        visit(getattr(value, field.name), depth + 1)
            else:
                raise ValueError("captured derivative tape contains unsupported storage")

        def payload(value: int | str | NDArray[np.float64], capacity: int) -> None:
            nonlocal copy_capacity
            if capacity > copy_capacity:
                reservation.resize(
                    ExecutionMemoryPlan(
                        (
                            *plan.buffers,
                            ExecutionBuffer(
                                "tape_payload_copy",
                                "intermediate",
                                (capacity + sys.getsizeof(b""),),
                                "uint8",
                            ),
                        )
                    )
                )
                copy_capacity = capacity
            if isinstance(value, int):
                data = value.to_bytes(capacity, "big", signed=True)
            elif isinstance(value, str):
                data = value.encode("utf-8", errors="surrogatepass")
            else:
                data = value.tobytes(order="C")
            reservation.checkpoint()
            digest.update(struct.pack("!Q", len(data)))
            digest.update(data)
            reservation.checkpoint()

        visit(result.value, 0)
        visit(result.gradient, 0)
        visit(result.parameter_names, 0)
        visit(result.trainable, 0)
        visit(result.ir_nodes, 0)
        visit(result.program_ir, 0)
        visit(result.adjoint_result, 0)
        reservation.checkpoint()
        return digest.digest()
