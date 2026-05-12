# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dense Hilbert-space allocation budget guards
"""Memory guards for dense Hilbert-space allocations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import numpy as np

DEFAULT_DENSE_BUDGET_ENV: Final = "SCPN_MAX_DENSE_GIB"
DEFAULT_DENSE_RAM_FRACTION: Final = 0.30
DEFAULT_DENSE_BUDGET_CAP_GIB: Final = 8.0
GIB: Final = 1024**3


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

    @property
    def gib_required(self) -> float:
        """Memory required in GiB."""
        return self.bytes_required / GIB

    @property
    def budget_gib(self) -> float:
        """Budget in GiB."""
        return self.budget_bytes / GIB


def hilbert_dimension(n_qubits: int) -> int:
    """Return ``2**n_qubits`` after validating the qubit count."""
    if not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    return 1 << n_qubits


def dense_object_bytes(
    n_qubits: int,
    *,
    dtype: np.dtype | type | str = np.complex128,
    rank: int = 2,
) -> int:
    """Return bytes needed for a dense Hilbert vector/matrix/superoperator."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    dim = hilbert_dimension(n_qubits)
    return int((dim**rank) * np.dtype(dtype).itemsize)


def available_memory_bytes() -> int | None:
    """Return available host memory when discoverable without extra dependencies."""
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


def dense_budget_bytes(max_gib: float | None = None) -> int:
    """Return the dense-allocation budget in bytes."""
    if max_gib is not None:
        if max_gib <= 0:
            raise ValueError("max_gib must be positive")
        return int(max_gib * GIB)

    env_value = os.environ.get(DEFAULT_DENSE_BUDGET_ENV)
    if env_value:
        try:
            parsed_gib = float(env_value)
        except ValueError as exc:
            raise ValueError(f"{DEFAULT_DENSE_BUDGET_ENV} must be a positive number") from exc
        if parsed_gib <= 0:
            raise ValueError(f"{DEFAULT_DENSE_BUDGET_ENV} must be positive")
        return int(parsed_gib * GIB)

    available = available_memory_bytes()
    if available is None:
        return int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)
    return int(min(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB, available * DEFAULT_DENSE_RAM_FRACTION))


def estimate_dense_allocation(
    n_qubits: int,
    *,
    dtype: np.dtype | type | str = np.complex128,
    rank: int = 2,
    max_gib: float | None = None,
    label: str = "dense Hilbert-space object",
) -> DenseAllocationEstimate:
    """Estimate one dense Hilbert-space allocation against the active budget."""
    dim = hilbert_dimension(n_qubits)
    return DenseAllocationEstimate(
        n_qubits=n_qubits,
        dimension=dim,
        shape=(dim,) * rank,
        dtype=np.dtype(dtype).name,
        bytes_required=dense_object_bytes(n_qubits, dtype=dtype, rank=rank),
        budget_bytes=dense_budget_bytes(max_gib),
        label=label,
    )


def require_dense_allocation(
    n_qubits: int,
    *,
    dtype: np.dtype | type | str = np.complex128,
    rank: int = 2,
    max_gib: float | None = None,
    label: str = "dense Hilbert-space object",
) -> DenseAllocationEstimate:
    """Raise before a dense Hilbert-space allocation exceeds the budget."""
    estimate = estimate_dense_allocation(
        n_qubits,
        dtype=dtype,
        rank=rank,
        max_gib=max_gib,
        label=label,
    )
    if estimate.bytes_required > estimate.budget_bytes:
        raise DenseAllocationError(
            f"{label} for n={estimate.n_qubits} requires "
            f"{estimate.gib_required:.2f} GiB for shape {estimate.shape} "
            f"({estimate.dtype}), above the active dense budget "
            f"{estimate.budget_gib:.2f} GiB. Use sparse, sector, tensor-network, "
            f"or explicit hardware execution instead of dense allocation."
        )
    return estimate
