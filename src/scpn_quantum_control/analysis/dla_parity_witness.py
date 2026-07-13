# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity witness module
"""DLA parity witness observable for bitstring-count analyses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DLAParityWitness:
    """Compute DLA parity asymmetry from measurement counts.

    Odd-Hamming-weight bitstrings represent the feedback sub-block and
    even-Hamming-weight bitstrings represent the projection sub-block.

    Parameters
    ----------
    split_odd_even : bool, default=True
        Compatibility flag retained on the witness instance. The current
        observable always reports the odd/even parity split.
    """

    def __init__(self, split_odd_even: bool = True) -> None:
        self.split_odd_even = split_odd_even

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """Evaluate the parity witness.

        Parameters
        ----------
        counts : Mapping[str, int] or None, optional
            Measurement counts mapping bitstrings to shot counts. Missing or
            empty counts return the balanced negative control.
        **kwargs : Any
            Ignored compatibility keywords accepted by the observable API.

        Returns
        -------
        dict[str, float]
            DLA asymmetry percentage, odd/even robustness fractions, and the
            total shot count when at least one shot is present.
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
