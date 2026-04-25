# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class SyncOrderParameter:
    """
    Computes the global or local synchronization order parameter from
    expectation values or measurement counts.
    """

    def __init__(self, local: bool = False, dtc_mode: bool = False) -> None:
        self.local = local
        self.dtc_mode = dtc_mode

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """
        Returns synchronization order parameter in [0, 1].
        """
        if counts is None or len(counts) == 0:
            return {"sync_order": 0.0}

        # Simple global order parameter from average magnetization (for Kuramoto-XY)
        total_shots = sum(counts.values())
        magnetization = 0.0

        for bitstring, shots in counts.items():
            # For Kuramoto-XY mapping, we treat |0> as phase 0, |1> as phase π
            spins = np.array([1 if b == "0" else -1 for b in bitstring])
            magnetization += np.mean(spins) * shots

        sync_order = abs(magnetization) / total_shots
        return {"sync_order": float(sync_order)}
