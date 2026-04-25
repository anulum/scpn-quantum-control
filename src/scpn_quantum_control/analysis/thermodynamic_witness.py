# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ThermodynamicWitness:
    """
    Computes thermodynamic work extracted across the synchronization transition.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        # Placeholder based on FIM transition strength
        work_extracted = kwargs.get("work", 1.2)
        return {"work": float(work_extracted)}
