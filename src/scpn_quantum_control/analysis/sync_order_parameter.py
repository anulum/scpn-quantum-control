# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Z-basis synchronisation proxy observable for measurement counts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class SyncOrderParameter:
    """Compute the count-derived Z-magnetisation synchronisation proxy.

    The public ``sync_order`` key is retained for compatibility with existing
    result artefacts. It is not the continuous Kuramoto ``X/Y`` order parameter
    ``R = |N**-1 sum_j (<X_j> + i<Y_j>)|``; counts in the computational basis
    only support the absolute mean Z-basis spin proxy emitted as
    ``sync_order_z_magnetisation``.
    """

    def __init__(self, local: bool = False, dtc_mode: bool = False) -> None:
        self.local = local
        self.dtc_mode = dtc_mode

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """Return the count-derived Z-magnetisation proxy in ``[0, 1]``.

        Parameters
        ----------
        counts:
            Measurement counts keyed by computational-basis bitstring.
        **kwargs:
            Reserved compatibility keyword arguments; ignored.

        Returns
        -------
        dict[str, float]
            ``sync_order`` and ``sync_order_z_magnetisation`` contain the same
            compatibility value. ``is_xy_kuramoto_order_parameter`` is always
            ``0.0`` because this counts-only path does not measure the X/Y
            Kuramoto order parameter.
        """
        if counts is None or len(counts) == 0:
            return {
                "sync_order": 0.0,
                "sync_order_z_magnetisation": 0.0,
                "is_xy_kuramoto_order_parameter": 0.0,
            }

        # Compatibility proxy from average computational-basis magnetisation.
        total_shots = sum(counts.values())
        magnetization = 0.0

        for bitstring, shots in counts.items():
            # Counts support Z-basis mean spin only: |0> -> +1, |1> -> -1.
            spins = np.array([1 if b == "0" else -1 for b in bitstring])
            magnetization += np.mean(spins) * shots

        sync_order = abs(magnetization) / total_shots
        return {
            "sync_order": float(sync_order),
            "sync_order_z_magnetisation": float(sync_order),
            "is_xy_kuramoto_order_parameter": 0.0,
        }
