# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DLAParityWitness:
    """
    Computes parity asymmetry between odd (feedback) and even (projection)
    sub-blocks of the Dynamical Lie Algebra (DLA) from measurement counts.
    """

    def __init__(self, split_odd_even: bool = True) -> None:
        self.split_odd_even = split_odd_even

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """
        Args:
            counts: Qiskit measurement counts (bitstrings -> shots)
        Returns:
            Dictionary with dla_asymmetry and supporting metrics
        """
        if counts is None or len(counts) == 0:
            return {"dla_asymmetry": 0.0, "odd_robustness": 0.5, "even_robustness": 0.5}

        # Simple parity computation based on Hamming weight parity of bitstrings
        odd_shots = even_shots = 0
        for bitstring, shots in counts.items():
            parity = sum(int(b) for b in bitstring) % 2
            if parity == 1:
                odd_shots += shots
            else:
                even_shots += shots

        total = odd_shots + even_shots
        if total == 0:
            return {"dla_asymmetry": 0.0, "odd_robustness": 0.5, "even_robustness": 0.5}

        odd_robustness = odd_shots / total
        even_robustness = even_shots / total
        asymmetry = (odd_robustness - even_robustness) * 100.0  # percent

        return {
            "dla_asymmetry": asymmetry,
            "odd_robustness": odd_robustness,
            "even_robustness": even_robustness,
            "total_shots": total,
        }
