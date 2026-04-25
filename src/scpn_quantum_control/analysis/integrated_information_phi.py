# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class IntegratedInformationPhi:
    """
    Computes a proxy for integrated information Φ from measurement data.
    Uses a simple entropy-based approximation suitable for NISQ.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        if counts is None or len(counts) == 0:
            return {"phi": 0.0}

        # Very simple proxy: normalized entropy of the distribution
        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(len(counts))
        phi_proxy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {"phi": float(phi_proxy)}
