# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class IntegratedInformationPhi:
    """
    Guarded integrated-information observable.

    Integrated information is not identified with output entropy. This class
    refuses to report Φ unless a real IIT/causal-state implementation is wired.
    For legacy dashboards, a labelled entropy proxy can be requested explicitly;
    that proxy is never returned under the key ``phi``.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        if not bool(kwargs.get("allow_entropy_proxy", False)):
            raise NotImplementedError(
                "IntegratedInformationPhi has no production integrated information "
                "implementation wired. Pass allow_entropy_proxy=True only for a "
                "labelled entropy diagnostic, not for Φ claims."
            )

        if counts is None or len(counts) == 0:
            return {
                "phi_available": 0.0,
                "entropy_proxy": 0.0,
                "is_integrated_information": 0.0,
            }

        total = sum(counts.values())
        if total <= 0:
            raise ValueError("counts must have a positive total.")
        if any(value < 0 for value in counts.values()):
            raise ValueError("counts must not contain negative values.")

        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(len(counts))
        entropy_proxy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "phi_available": 0.0,
            "entropy_proxy": float(entropy_proxy),
            "is_integrated_information": 0.0,
        }
