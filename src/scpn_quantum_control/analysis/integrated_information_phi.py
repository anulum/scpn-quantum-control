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
        coupling_matrix = kwargs.get("coupling_matrix")
        natural_frequencies = kwargs.get("natural_frequencies")
        if coupling_matrix is not None or natural_frequencies is not None:
            if coupling_matrix is None or natural_frequencies is None:
                raise ValueError(
                    "IntegratedInformationPhi requires both coupling_matrix and "
                    "natural_frequencies for production evaluation."
                )
            return self._compute_production_phi(coupling_matrix, natural_frequencies)

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

    @staticmethod
    def _compute_production_phi(
        coupling_matrix: Any, natural_frequencies: Any
    ) -> dict[str, float]:
        from .quantum_phi import compute_quantum_phi

        K = np.asarray(coupling_matrix, dtype=float)
        omega = np.asarray(natural_frequencies, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("coupling_matrix must be a square two-dimensional array.")
        if omega.ndim != 1 or omega.shape[0] != K.shape[0]:
            raise ValueError(
                "natural_frequencies must be a one-dimensional vector matching coupling_matrix."
            )
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)):
            raise ValueError("coupling_matrix and natural_frequencies must contain finite values.")
        if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
            raise ValueError("coupling_matrix must be symmetric.")

        result = compute_quantum_phi(K, omega)
        return {
            "phi_available": 1.0,
            "phi": float(result.phi_quantum),
            "phi_max": float(result.phi_max),
            "total_entropy": float(result.total_entropy),
            "n_qubits": float(result.n_qubits),
            "n_bipartitions": float(result.n_bipartitions),
            "mip_partition_size_a": float(len(result.mip_partition[0])),
            "mip_partition_size_b": float(len(result.mip_partition[1])),
            "is_integrated_information": 1.0,
        }
