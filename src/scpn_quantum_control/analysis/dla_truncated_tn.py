# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA tensor-network interface
"""Fail-fast interface for future DLA-truncated tensor-network simulations."""

from operator import index
from typing import Any

import numpy as np

_SUPPORTED_OBSERVABLES = frozenset({"sync_order", "dla_parity", "correlation"})


def dla_truncated_tn(
    K_nm: np.ndarray,
    max_bond_dim: int = 32,
    dla_cutoff: float = 1e-6,
    observable: str = "sync_order",
) -> dict[str, Any]:
    """
    DLA-truncated tensor-network simulation entry point.

    This module is intentionally not implemented yet. It must not return
    synthetic sync values, because those values can contaminate hardware
    campaign outputs.
    """
    _validate_inputs(K_nm, max_bond_dim, dla_cutoff, observable)
    raise NotImplementedError(
        "DLA-truncated tensor-network simulation is not implemented. "
        "Do not use this path for QPU campaigns until a real quimb/PennyLane "
        "implementation and validation tests are added."
    )


def _validate_inputs(
    K_nm: np.ndarray,
    max_bond_dim: Any,
    dla_cutoff: float,
    observable: str,
) -> None:
    K = np.asarray(K_nm, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K_nm must be a square two-dimensional coupling matrix.")
    if not np.all(np.isfinite(K)):
        raise ValueError("K_nm must contain finite values.")
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
        raise ValueError("K_nm must be symmetric.")
    _validate_positive_int(max_bond_dim, "max_bond_dim")
    cutoff = float(dla_cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("dla_cutoff must be a finite positive value.")
    if observable not in _SUPPORTED_OBSERVABLES:
        raise ValueError("observable must be one of: correlation, dla_parity, sync_order.")


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer_value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer_value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(integer_value)
