# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded source admission
"""Read and locate objective source within an existing memory reservation."""

from __future__ import annotations

import ast
import inspect
import io
import os
import re
import stat
import sys
import tokenize
from collections.abc import Callable
from types import CodeType

from .dense_budget import DenseAllocationError
from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import ExecutionMemoryReservation

_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
_FUNCTION_LINE = re.compile(r"^(\s*def\s)|(\s*async\s+def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)")


def read_source_lines(
    filename: str,
    observation: os.stat_result,
    reservation: ExecutionMemoryReservation,
) -> tuple[list[str], ExecutionMemoryPlan]:
    """Read at most the observed regular-file bytes with owned admission.

    Parameters
    ----------
    filename
        Source path already observed as a regular file.
    observation
        Stat snapshot whose identity and size must remain unchanged.
    reservation
        Dedicated source owner resized before buffers, decoding and lines are built.

    Returns
    -------
    tuple
        Universal-newline source lines and the complete retained declaration.

    Raises
    ------
    DenseAllocationError
        Capacity refusal, file replacement, growth, truncation or read failure.
    ExecutionCancelledError
        This source owner or an ancestor has been cancelled.
    TimeoutError
        This source owner or an ancestor deadline has elapsed.
    RuntimeError
        The reservation is closed or belongs to another thread or process.

    """
    if not stat.S_ISREG(observation.st_mode):
        raise DenseAllocationError("objective source must be an observed regular file")
    size = observation.st_size
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer("source_read_bytes", "intermediate", (max(1, size),), "uint8", 5),
            ExecutionBuffer(
                "source_decode_text", "intermediate", (max(1, size + 1),), "uint32", 4
            ),
            ExecutionBuffer(
                "source_read_headers",
                "intermediate",
                (sys.getsizeof(bytearray()) + 5 * sys.getsizeof(b"") + 4 * sys.getsizeof("𐀀"),),
                "uint8",
            ),
        )
    )
    reservation.resize(plan)
    identity = tuple(getattr(observation, field) for field in _IDENTITY_FIELDS)
    try:
        with open(
            filename,
            "rb",
            buffering=0,
            opener=lambda path, flags: os.open(path, flags | os.O_NONBLOCK),
        ) as source_file:
            current = os.fstat(source_file.fileno())
            if tuple(getattr(current, field) for field in _IDENTITY_FIELDS) != identity:
                raise DenseAllocationError("objective source changed before bounded read")
            reservation.checkpoint()
            raw = bytearray(size)
            offset = 0
            while offset < size:
                reservation.checkpoint()
                amount = source_file.readinto(memoryview(raw)[offset : min(size, offset + 65536)])
                if amount is None or amount == 0:
                    raise DenseAllocationError("objective source truncated during bounded read")
                offset += amount
            reservation.checkpoint()
            if source_file.read(1):
                raise DenseAllocationError("objective source grew during bounded read")
            current = os.fstat(source_file.fileno())
            if tuple(getattr(current, field) for field in _IDENTITY_FIELDS) != identity:
                raise DenseAllocationError("objective source changed during bounded read")
    except OSError as exc:
        raise DenseAllocationError(
            "observed objective source became unavailable during read"
        ) from exc
    with io.BytesIO(raw) as encoding_source:
        encoding, _ = tokenize.detect_encoding(encoding_source.readline)
    reservation.checkpoint()
    text = raw.decode(encoding)
    reservation.checkpoint()
    if len(text) > size:
        raise DenseAllocationError("source codec exceeded the declared character bound")
    line_bound = text.count("\n") + text.count("\r") + 1
    line_bytes = 2 * sys.getsizeof([]) + line_bound * (2 * sys.getsizeof(0) + sys.getsizeof("𐀀"))
    plan = ExecutionMemoryPlan(
        (
            *plan.buffers,
            ExecutionBuffer("source_read_lines", "intermediate", (line_bytes,), "uint8"),
        )
    )
    reservation.resize(plan)
    with io.StringIO(text, newline=None) as decoded_source:
        lines = decoded_source.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    reservation.checkpoint()
    return lines, plan


def objective_source_block(
    objective: Callable[..., object], lines: list[str]
) -> tuple[list[str], int]:
    """Locate function, method or class source in already admitted file lines.

    Parameters
    ----------
    objective
        Python callable, unwrapped using its standard decorator metadata.
    lines
        Admitted file contents with universal newlines.

    Returns
    -------
    tuple
        Source block and its one-based first line, including decorators.

    Raises
    ------
    OSError
        The callable has no matching definition in these file contents.

    """
    value = inspect.unwrap(objective)
    if inspect.isclass(value):
        start = _class_line(ast.parse("".join(lines)), value.__qualname__, "")
        if start is None:
            raise OSError("could not find class definition")
    else:
        if inspect.ismethod(value):
            value = value.__func__
        code = getattr(value, "__code__", None)
        if not isinstance(code, CodeType):
            raise OSError("could not find objective code object")
        start = code.co_firstlineno - 1
        if not 0 <= start < len(lines):
            raise OSError("objective line is outside source file")
        while start > 0 and not _FUNCTION_LINE.match(lines[start]):
            start -= 1
    return inspect.getblock(lines[start:]), start + 1


def _class_line(node: ast.AST, qualname: str, prefix: str) -> int | None:
    """Find a class by its actual Python lexical qualified name."""
    if isinstance(node, ast.ClassDef):
        prefix += node.name
        if prefix == qualname:
            return (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
        prefix += "."
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix += node.name + ".<locals>."
    for child in ast.iter_child_nodes(node):
        result = _class_line(child, qualname, prefix)
        if result is not None:
            return result
    return None
