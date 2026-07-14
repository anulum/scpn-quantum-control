# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Native Compilation Test Helpers
"""Typed fixtures shared by native compiler integration tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _eager_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Small primitive batching rule used by compiler-AD registry tests."""
    del axes, out_axes
    if len(args) != 1:
        raise ValueError("test batching rule expects one batched argument")
    batch = np.asarray(args[0], dtype=np.float64)
    return np.asarray([function(item) for item in batch], dtype=np.float64)
