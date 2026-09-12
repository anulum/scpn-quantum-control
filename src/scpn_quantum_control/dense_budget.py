# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dense Hilbert-space allocation budget guards
"""Memory guards for dense Hilbert-space allocations."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._cgroup_paths import memory_cgroup_paths

DEFAULT_DENSE_BUDGET_ENV: Final = "SCPN_MAX_DENSE_GIB"
DEFAULT_DENSE_RAM_FRACTION: Final = 0.30
DEFAULT_DENSE_BUDGET_CAP_GIB: Final = 8.0
DEFAULT_DENSE_EIGENSOLVER_OBJECTS: Final = 4
GIB: Final = 1024**3

DEFAULT_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
"""Fallback root when process membership metadata is unavailable.

Explicit per-call roots retain the flat controller snapshot interface. Default
Linux calls discover mounted memory controllers through process metadata.
"""

DEFAULT_PROC_ROOT: Final = Path("/proc/self")
"""Process membership and mount metadata used by default memory admission."""

CGROUP_V2_LIMIT: Final = "memory.max"
CGROUP_V2_USAGE: Final = "memory.current"
CGROUP_V1_LIMIT: Final = "memory/memory.limit_in_bytes"
CGROUP_V1_USAGE: Final = "memory/memory.usage_in_bytes"

CGROUP_UNLIMITED_THRESHOLD: Final = 1 << 62
"""Above this, a numeric cgroup limit means "no limit".

cgroup v2 writes the literal ``max``. cgroup v1 has no such spelling and writes
a sentinel near the pointer maximum instead, commonly
``9223372036854771712``, so any value this large is read as unlimited rather
than as an allowance larger than any real machine.
"""


class DenseAllocationError(MemoryError):
    """Raised before an unsafe dense Hilbert-space allocation is attempted."""


@dataclass(frozen=True)
class DenseAllocationEstimate:
    """Estimated memory for one dense Hilbert-space object."""

    n_qubits: int
    dimension: int
    shape: tuple[int, ...]
    dtype: str
    bytes_required: int
    budget_bytes: int
    label: str
    object_count: int = 1

    @property
    def gib_required(self) -> float:
        """Memory required in GiB."""
        return self.bytes_required / GIB

    @property
    def budget_gib(self) -> float:
        """Budget in GiB."""
        return self.budget_bytes / GIB


def _positive_integer(value: object, name: str) -> int:
    """Validate integral allocation metadata without boolean or fractional coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(value)


def hilbert_dimension(n_qubits: int) -> int:
    """Return ``2**n_qubits`` for a positive non-boolean integer qubit count."""
    return 1 << _positive_integer(n_qubits, "n_qubits")


def dense_object_bytes(
    n_qubits: int,
    *,
    dtype: np.dtype[Any] | type | str = np.complex128,
    rank: int = 2,
) -> int:
    """Return bytes needed for a dense Hilbert vector/matrix/superoperator."""
    rank = _positive_integer(rank, "rank")
    dim = hilbert_dimension(n_qubits)
    return int((dim**rank) * np.dtype(dtype).itemsize)


def host_available_memory_bytes() -> int | None:
    """Return available host memory when discoverable without extra dependencies.

    This is what the machine has free, which is not what a containerised process
    is allowed to use. Callers that need the process allowance should use
    :func:`available_memory_bytes`.

    Returns
    -------
    int | None
        Free host bytes, or ``None`` when the platform does not report them.

    """
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if (
        not isinstance(pages, int)
        or not isinstance(page_size, int)
        or pages <= 0
        or page_size <= 0
    ):
        return None
    return pages * page_size


def _read_cgroup_int(path: Path) -> int | None:
    """Return one non-negative integer from a cgroup file, or ``None``.

    Every failure mode a real mount presents is treated the same way: a missing
    file, an unreadable one, the literal ``max``, and any malformed content all
    yield ``None``. Only ASCII decimal digits are accepted after whitespace
    stripping, not Python integer syntax such as signs or underscores. The
    caller distinguishes an unknown limit from unknown usage under a known
    limit; the latter cannot establish any available headroom.

    Parameters
    ----------
    path
        Absolute path to the cgroup control file.

    Returns
    -------
    int | None
        The parsed value, or ``None`` when it is absent or not a plain integer.

    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not raw.isascii() or not raw.isdecimal():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value


def cgroup_headroom_bytes(cgroup_root: Path | None = None) -> int | None:
    """Return the memory a cgroup still allows this process, if it is limited.

    Default calls resolve process membership and visible mount ancestors from
    proc metadata, taking the minimum headroom across their limits. An explicit
    root retains the flat v2-then-v1 snapshot interface. A container that
    has already consumed most of its allowance reports what is left rather than
    what it was granted. A known finite limit with unreadable or malformed
    usage reports zero verified headroom, never an assumed unused allowance.
    This is an admission snapshot, not a memory reservation or OOM guarantee.

    Parameters
    ----------
    cgroup_root
        Explicit flat snapshot root. When omitted, discover process paths via
        :data:`DEFAULT_PROC_ROOT`; unavailable proc metadata falls back to
        :data:`DEFAULT_CGROUP_ROOT`. Malformed or unmappable membership refuses
        default admission with zero verified headroom. Hidden ancestors above
        the namespace mount are not observable; this is not an OOM guarantee.

    Returns
    -------
    int | None
        Remaining verified bytes, zero when usage under a finite limit cannot
        be established, or ``None`` when no readable finite limit applies.

    """
    if cgroup_root is None:
        try:
            paths = memory_cgroup_paths(DEFAULT_PROC_ROOT)
        except ValueError:
            return 0
        if paths is not None:
            headrooms: list[int] = []
            for entry in paths:
                if not entry.directory.is_dir():
                    return 0
                if (
                    entry.version == 1
                    and not entry.leaf
                    and _read_cgroup_int(entry.directory / "memory.use_hierarchy") == 0
                ):
                    continue
                limit_name = "memory.max" if entry.version == 2 else "memory.limit_in_bytes"
                usage_name = "memory.current" if entry.version == 2 else "memory.usage_in_bytes"
                limit = _read_cgroup_int(entry.directory / limit_name)
                if limit is None or limit >= CGROUP_UNLIMITED_THRESHOLD:
                    continue
                usage = _read_cgroup_int(entry.directory / usage_name)
                headrooms.append(0 if usage is None else max(0, limit - usage))
            return min(headrooms) if headrooms else None
    root = DEFAULT_CGROUP_ROOT if cgroup_root is None else cgroup_root
    for limit_name, usage_name in (
        (CGROUP_V2_LIMIT, CGROUP_V2_USAGE),
        (CGROUP_V1_LIMIT, CGROUP_V1_USAGE),
    ):
        limit = _read_cgroup_int(root / limit_name)
        if limit is None or limit >= CGROUP_UNLIMITED_THRESHOLD:
            continue
        usage = _read_cgroup_int(root / usage_name)
        return 0 if usage is None else max(0, limit - usage)
    return None


def available_memory_bytes(cgroup_root: Path | None = None) -> int | None:
    """Return a snapshot of host availability constrained by visible cgroups.

    The smaller of free host memory and any cgroup headroom. On an unrestricted
    host this is the host figure unchanged; inside a memory-limited container it
    is the container's remaining allowance, which is the number the previous
    implementation could not see.

    Parameters
    ----------
    cgroup_root
        Explicit flat controller snapshot root, or ``None`` for process
        membership discovery. See :func:`cgroup_headroom_bytes` for fallback
        and namespace visibility limits. This does not reserve memory.

    Returns
    -------
    int | None
        Usable bytes, or ``None`` when neither source reports anything.

    """
    host = host_available_memory_bytes()
    headroom = cgroup_headroom_bytes(cgroup_root)
    if host is None:
        return headroom
    if headroom is None:
        return host
    return min(host, headroom)


def _budget_bytes(value: object, name: str) -> int:
    """Convert a finite positive real GiB budget to an integer byte allowance."""
    message = f"{name} must be positive, finite and representable in bytes"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    try:
        gib = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    byte_count = gib * GIB
    if not math.isfinite(byte_count) or gib <= 0:
        raise ValueError(message)
    return int(byte_count)


def dense_budget_bytes(max_gib: float | None = None) -> int:
    """Return the dense-allocation budget in bytes."""
    if max_gib is not None:
        return _budget_bytes(max_gib, "max_gib")

    env_value = os.environ.get(DEFAULT_DENSE_BUDGET_ENV)
    if env_value is not None:
        try:
            parsed_gib = float(env_value)
        except ValueError as exc:
            raise ValueError(f"{DEFAULT_DENSE_BUDGET_ENV} must be a positive number") from exc
        return _budget_bytes(parsed_gib, DEFAULT_DENSE_BUDGET_ENV)

    available = available_memory_bytes()
    if available is None:
        return int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)
    return int(min(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB, available * DEFAULT_DENSE_RAM_FRACTION))


def estimate_dense_allocation(
    n_qubits: int,
    *,
    dtype: np.dtype[Any] | type | str = np.complex128,
    rank: int = 2,
    object_count: int = 1,
    max_gib: float | None = None,
    label: str = "dense Hilbert-space object",
) -> DenseAllocationEstimate:
    """Estimate one dense Hilbert-space allocation against the active budget."""
    object_count = _positive_integer(object_count, "object_count")
    rank = _positive_integer(rank, "rank")
    dim = hilbert_dimension(n_qubits)
    object_bytes = dense_object_bytes(n_qubits, dtype=dtype, rank=rank)
    return DenseAllocationEstimate(
        n_qubits=n_qubits,
        dimension=dim,
        shape=(dim,) * rank,
        dtype=np.dtype(dtype).name,
        bytes_required=object_bytes * object_count,
        budget_bytes=dense_budget_bytes(max_gib),
        label=label,
        object_count=object_count,
    )


def require_dense_allocation(
    n_qubits: int,
    *,
    dtype: np.dtype[Any] | type | str = np.complex128,
    rank: int = 2,
    object_count: int = 1,
    max_gib: float | None = None,
    label: str = "dense Hilbert-space object",
) -> DenseAllocationEstimate:
    """Reject allocations exceeding the budget or native addressable size.

    Validate positive non-boolean integer metadata before constructing dimensions
    or shape tuples. Impossible native sizes raise ``DenseAllocationError`` using
    a bounded diagnostic, without building exponential-sized Python integers.
    Type errors identify invalid counts; invalid budgets raise ``ValueError``.
    This checks the declared buffers, not total process memory or reservations.
    """
    n_qubits = _positive_integer(n_qubits, "n_qubits")
    rank = _positive_integer(rank, "rank")
    object_count = _positive_integer(object_count, "object_count")
    itemsize = np.dtype(dtype).itemsize
    if itemsize < 1:
        raise ValueError("dtype must have a positive fixed item size")
    exponent = n_qubits * rank
    if (
        exponent >= sys.maxsize.bit_length()
        or object_count > (sys.maxsize >> exponent) // itemsize
    ):
        raise DenseAllocationError(f"{label} exceeds native addressable memory")
    estimate = estimate_dense_allocation(
        n_qubits,
        dtype=dtype,
        rank=rank,
        object_count=object_count,
        max_gib=max_gib,
        label=label,
    )
    if estimate.bytes_required > estimate.budget_bytes:
        object_text = (
            f"{estimate.object_count} objects of shape {estimate.shape}"
            if estimate.object_count != 1
            else f"shape {estimate.shape}"
        )
        raise DenseAllocationError(
            f"{label} for n={estimate.n_qubits} requires "
            f"{estimate.gib_required:.2f} GiB for {object_text} "
            f"({estimate.dtype}), above the active dense budget "
            f"{estimate.budget_gib:.2f} GiB. Use sparse, sector, tensor-network, "
            f"or explicit hardware execution instead of dense allocation."
        )
    return estimate


def require_dense_eigensolver_workspace(
    n_qubits: int,
    *,
    dtype: np.dtype[Any] | type | str = np.complex128,
    max_gib: float | None = None,
    label: str = "dense eigensolver workspace",
) -> DenseAllocationEstimate:
    """Guard a dense Hermitian eigensolver call and retained eigenvectors.

    LAPACK-style dense Hermitian solvers can retain both the input Hamiltonian
    and eigenvector matrix while allocating additional reduction/work arrays.
    Counting one or two dense matrices underestimates the real peak workspace,
    so exact spectral probes use a shared conservative multiplier.
    """
    return require_dense_allocation(
        n_qubits,
        dtype=dtype,
        rank=2,
        object_count=DEFAULT_DENSE_EIGENSOLVER_OBJECTS,
        max_gib=max_gib,
        label=label,
    )
