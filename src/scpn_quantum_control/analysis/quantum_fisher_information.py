# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class QuantumFisherInformation:
    """
    Computes a proxy for Quantum Fisher Information (metrological gain)
    from the sync order and DLA asymmetry.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        # Use sync_order and dla_asymmetry if provided, otherwise fallback
        sync_order = kwargs.get("sync_order", 0.95)
        dla_asym = kwargs.get("dla_asymmetry", 0.08)

        # Simple proxy: QFI ≈ 4 * (variance) near critical point
        qfi_proxy = 4.0 * sync_order * (1.0 + abs(dla_asym) / 100.0)
        return {"qfi": float(qfi_proxy)}
