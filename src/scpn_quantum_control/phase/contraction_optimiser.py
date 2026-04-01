# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tensor Contraction Path Optimiser
"""Optimal tensor contraction paths for MPS operations.

Uses cotengra (if available) to find optimal contraction paths
for tensor network operations, reducing both memory and time
for DMRG, TEBD, and expectation value computations.

Inspired by quimb's integration with cotengra (Gray, JOSS 2018).

Requires: pip install cotengra (optional — falls back to numpy.einsum)
"""

from __future__ import annotations

import numpy as np

try:
    import cotengra

    _COTENGRA_AVAILABLE = True
except ImportError:
    _COTENGRA_AVAILABLE = False
    cotengra = None  # type: ignore[assignment]


def is_cotengra_available() -> bool:
    """Check if cotengra is installed."""
    return _COTENGRA_AVAILABLE


def optimal_contraction_path(
    subscripts: str,
    *operands: np.ndarray,
    optimiser: str = "auto",
) -> tuple[list[tuple[int, ...]], dict]:
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
    if _COTENGRA_AVAILABLE and optimiser == "auto":
        try:
            path, info = cotengra.einsum_path(subscripts, *operands)
            return path, {"method": "cotengra", "info": str(info)}
        except Exception:
            pass  # fall through to numpy

    # Fallback: numpy optimal path
    path, path_info = np.einsum_path(subscripts, *operands, optimize="optimal")
    return path, {
        "flops": 0,  # numpy doesn't report flops
        "max_size": 0,
        "method": "numpy_optimal",
        "info_string": path_info,
    }


def contract(
    subscripts: str,
    *operands: np.ndarray,
    optimiser: str = "auto",
) -> np.ndarray:
    """Contract tensors using optimal path.

    Drop-in replacement for np.einsum with path optimisation.
    """
    if _COTENGRA_AVAILABLE and optimiser == "auto":
        return np.asarray(cotengra.einsum(subscripts, *operands))

    return np.asarray(np.einsum(subscripts, *operands, optimize="optimal"))


def benchmark_contraction(
    subscripts: str,
    *operands: np.ndarray,
    n_repeats: int = 10,
) -> dict:
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
        "naive_ms": round(naive, 2),
        "optimised_ms": round(opt, 2),
        "speedup": round(naive / opt, 1) if opt > 0 else float("inf"),
    }
