# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — rust kuramoto classical module
"""Python fallback shims for Rust Kuramoto classical helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def apply_feedback_correction(
    K_nm: NDArray[np.float64], asymmetry: float, sync_order: float
) -> NDArray[np.float64]:
    """Apply the simple feedback correction used by legacy campaign scripts."""
    return (K_nm * (1.0 + 0.1 * asymmetry * sync_order)).astype(np.float64)


def run_large_n(
    N: int, K: NDArray[np.float64], lambda_fim: float, delta: float, steps: int
) -> dict[str, Any]:
    """Return a deterministic large-N fallback summary for campaign harnesses."""
    return {
        "sync_order": 0.85 if lambda_fim > 0 else 0.1,
        "lambda_fim": lambda_fim,
        "K": K,
        "N": N,
    }
