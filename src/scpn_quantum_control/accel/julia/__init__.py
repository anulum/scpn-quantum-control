# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia accel tier
"""Julia acceleration tier via ``juliacall``.

Activation is lazy: the Julia runtime boots on first
:func:`is_available` call that returns True, and the ``.jl`` source
files are ``include``'d at that moment. Subsequent calls reuse the
cached Julia ``Main`` module so the JIT warm-up cost is paid exactly
once per Python process.

If ``juliacall`` is not installed or Julia itself is not on PATH,
:func:`is_available` returns False and the module-level accessors
raise :class:`ImportError`. The dispatcher above handles that
gracefully and falls through to the next tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_JL: Any = None
_INCLUDED: bool = False
_JULIA_DIR = Path(__file__).parent


def _load() -> Any:
    """Return the ``juliacall.Main`` handle, booting Julia on first call."""
    global _JL, _INCLUDED
    if _JL is not None and _INCLUDED:
        return _JL
    try:
        from juliacall import Main as jl  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError("juliacall is not installed") from exc
    _JL = jl
    if not _INCLUDED:
        for jl_file in sorted(_JULIA_DIR.glob("*.jl")):
            jl.include(str(jl_file))
        _INCLUDED = True
    return _JL


def is_available() -> bool:
    """Return True if Julia + juliacall can run in this process.

    First call incurs the Julia boot cost (~20 s). Callers that want a
    cheap probe without warming Julia should check whether
    ``juliacall`` is importable instead.
    """
    try:
        _load()
        return True
    except Exception:
        return False


def order_parameter(theta: np.ndarray) -> float:
    """Julia-tier implementation of the Kuramoto order parameter."""
    jl = _load()
    # juliacall converts a numpy array to a Julia Vector{Float64}
    # without copying when the dtype is float64 and C-contiguous.
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return float(jl.order_parameter(arr))


def order_parameters_batch(theta_batch: np.ndarray) -> np.ndarray:
    """Julia-tier batched variant — T time-slices × N oscillators."""
    jl = _load()
    arr = np.ascontiguousarray(theta_batch, dtype=np.float64)
    return np.asarray(jl.order_parameters_batch(arr), dtype=np.float64)


__all__ = [
    "is_available",
    "order_parameter",
    "order_parameters_batch",
]
