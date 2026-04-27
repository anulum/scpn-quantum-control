# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class QuantumFisherInformation:
    """
    Computes a proxy for Quantum Fisher Information (metrological gain)
    from the sync order and DLA asymmetry.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        sync_order = kwargs.get("sync_order")
        dla_asym = kwargs.get("dla_asymmetry")

        if counts and (sync_order is None or dla_asym is None):
            from .dla_parity_witness import DLAParityWitness
            from .sync_order_parameter import SyncOrderParameter

            if sync_order is None:
                sync_order = SyncOrderParameter()(counts=counts)["sync_order"]
            if dla_asym is None:
                dla_asym = DLAParityWitness()(counts=counts)["dla_asymmetry"]

        # Missing inputs mean missing signal, not an idealised target value.
        sync_order = 0.0 if sync_order is None else float(sync_order)
        dla_asym = 0.0 if dla_asym is None else float(dla_asym)

        # Simple proxy: QFI ≈ 4 * (variance) near critical point
        qfi_proxy = 4.0 * sync_order * (1.0 + abs(dla_asym) / 100.0)
        return {"qfi": float(qfi_proxy)}
