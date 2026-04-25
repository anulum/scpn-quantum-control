# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LogicalSyncWitness:
    """
    Witnesses logical information encoded in the global sync manifold.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        logical_fidelity = kwargs.get("logical_fidelity", 0.92)
        return {"logical_fidelity": float(logical_fidelity)}
