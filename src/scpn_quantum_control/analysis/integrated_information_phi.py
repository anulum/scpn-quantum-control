# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — integrated information phi module
"""Fail-closed interface for integrated-information requests.

No Integrated Information Theory (IIT) causal model is implemented. Exact
Kuramoto-XY inputs can be routed only to an explicitly requested minimum
bipartite quantum-mutual-information diagnostic. Entropy and mutual information
are not returned under a ``phi`` key and must not be interpreted as
consciousness, sentience, cognition, or a clinical-state measure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class IntegratedInformationPhi:
    """
    Guarded integrated-information request surface.

    Integrated information is not identified with output entropy. This class
    refuses to report Φ because no IIT causal-state implementation is wired.
    Labelled entropy or bipartite-mutual-information diagnostics can be
    requested explicitly; neither is returned under the key ``phi``.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        """Reject Φ requests or return an explicitly labelled diagnostic.

        Parameters
        ----------
        counts
            Optional outcome-count mapping used only by the entropy diagnostic.
        **kwargs
            ``coupling_matrix`` and ``natural_frequencies`` select the exact
            ground-state mutual-information route, which also requires
            ``allow_mutual_information_proxy=True``. ``allow_entropy_proxy``
            explicitly selects the normalized count-entropy route.

        Returns
        -------
        dict[str, float]
            A labelled entropy or mutual-information diagnostic with
            ``phi_available=0.0`` and ``is_integrated_information=0.0``.

        Raises
        ------
        NotImplementedError
            If a diagnostic is not explicitly opted into.
        ValueError
            If inputs are incomplete, malformed, non-finite, or inconsistent.

        """
        coupling_matrix = kwargs.get("coupling_matrix")
        natural_frequencies = kwargs.get("natural_frequencies")
        if coupling_matrix is not None or natural_frequencies is not None:
            if coupling_matrix is None or natural_frequencies is None:
                raise ValueError(
                    "IntegratedInformationPhi requires both coupling_matrix and "
                    "natural_frequencies for production evaluation."
                )
            if not bool(kwargs.get("allow_mutual_information_proxy", False)):
                raise NotImplementedError(
                    "No IIT Phi implementation is wired. Pass "
                    "allow_mutual_information_proxy=True only for the explicitly "
                    "labelled bipartite mutual-information diagnostic."
                )
            return self._compute_mutual_information_proxy(coupling_matrix, natural_frequencies)

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
    def _compute_mutual_information_proxy(
        coupling_matrix: Any, natural_frequencies: Any
    ) -> dict[str, float]:
        """Compute the legacy minimum-bipartite-QMI diagnostic.

        This private adapter validates the exact-model inputs and relabels every
        output so downstream consumers cannot confuse quantum mutual
        information with IIT Φ.
        """
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
            "phi_available": 0.0,
            "mutual_information_proxy_available": 1.0,
            "minimum_bipartite_mutual_information": float(result.phi_quantum),
            "maximum_bipartite_mutual_information": float(result.phi_max),
            "total_entropy": float(result.total_entropy),
            "n_qubits": float(result.n_qubits),
            "n_bipartitions": float(result.n_bipartitions),
            "mip_partition_size_a": float(len(result.mip_partition[0])),
            "mip_partition_size_b": float(len(result.mip_partition[1])),
            "is_integrated_information": 0.0,
        }
