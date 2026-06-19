# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor Contraction Path Optimiser
"""Optimal tensor contraction paths for MPS operations.

Uses cotengra (if available) to find optimal contraction paths
for tensor network operations, reducing both memory and time
for DMRG, TEBD, and expectation value computations.

Inspired by quimb's integration with cotengra (Gray, JOSS 2018).

Requires: pip install cotengra (optional — falls back to numpy.einsum)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

TensorArray: TypeAlias = NDArray[np.float64]


class ContractionPathInfo(TypedDict, total=False):
    """Metadata describing the selected tensor contraction path."""

    flops: int
    max_size: int
    method: str
    fallback_reason: str | None
    info: str
    info_string: str


class ContractionBenchmarkResult(TypedDict):
    """Timing summary for naive and optimised tensor contractions."""

    naive_ms: float
    optimised_ms: float
    speedup: float


class _CotengraLike(Protocol):
    """Subset of the optional cotengra API used by this module."""

    def einsum_path(
        self,
        subscripts: str,
        *operands: TensorArray,
    ) -> tuple[Sequence[object], object]:
        """Return a contraction path and backend-specific metadata."""

    def einsum(self, subscripts: str, *operands: TensorArray) -> object:
        """Evaluate an Einstein summation expression."""


try:
    import cotengra as _cotengra

    _COTENGRA_AVAILABLE = True
except ImportError:
    _COTENGRA_AVAILABLE = False
    cotengra: _CotengraLike | None = None
else:
    cotengra = cast(_CotengraLike, _cotengra)


def is_cotengra_available() -> bool:
    """Check if cotengra is installed."""
    return _COTENGRA_AVAILABLE


def optimal_contraction_path(
    subscripts: str,
    *operands: TensorArray,
    optimiser: str = "auto",
) -> tuple[list[tuple[int, ...]], ContractionPathInfo]:
    """Find optimal contraction path for an einsum expression.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts (e.g., "ij,jk->ik").
    operands : arrays
        Input tensors.
    optimiser : str
        "auto" (cotengra if available), "greedy", "optimal" (numpy).

    Returns
    -------
    (path, info) where path is a list of contraction pairs and
    info contains flops, memory estimates.
    """
    if _COTENGRA_AVAILABLE and optimiser == "auto" and cotengra is not None:
        try:
            path, info = cotengra.einsum_path(subscripts, *operands)
            return _normalise_contraction_path(path), {"method": "cotengra", "info": str(info)}
        except Exception as exc:
            fallback_reason = exc.__class__.__name__

    # Fallback: numpy optimal path
    path, path_info = np.einsum_path(subscripts, *operands, optimize="optimal")
    return _normalise_contraction_path(path), {
        "flops": 0,  # numpy doesn't report flops
        "max_size": 0,
        "method": "numpy_optimal",
        "fallback_reason": locals().get("fallback_reason"),
        "info_string": path_info,
    }


def contract(
    subscripts: str,
    *operands: TensorArray,
    optimiser: str = "auto",
) -> TensorArray:
    """Contract tensors using optimal path.

    Drop-in replacement for np.einsum with path optimisation.
    """
    if _COTENGRA_AVAILABLE and optimiser == "auto" and cotengra is not None:
        return np.asarray(cotengra.einsum(subscripts, *operands))

    return np.asarray(np.einsum(subscripts, *operands, optimize="optimal"))


def benchmark_contraction(
    subscripts: str,
    *operands: TensorArray,
    n_repeats: int = 10,
) -> ContractionBenchmarkResult:
    """Benchmark contraction with and without optimisation.

    Returns dict with: naive_ms, optimised_ms, speedup
    """
    import time

    # Naive
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        np.einsum(subscripts, *operands)
    naive = (time.perf_counter() - t0) / n_repeats * 1000

    # Optimised
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        contract(subscripts, *operands)
    opt = (time.perf_counter() - t0) / n_repeats * 1000

    return {
        "naive_ms": naive,
        "optimised_ms": opt,
        "speedup": naive / opt if opt > 0 else float("inf"),
    }


def _normalise_contraction_path(path: Sequence[object]) -> list[tuple[int, ...]]:
    normalised: list[tuple[int, ...]] = []
    for item in path:
        if isinstance(item, (list, tuple)):
            normalised.append(tuple(int(index) for index in item))
    return normalised
