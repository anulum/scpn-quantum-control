# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Logical Synchronisation Witness
"""Logical synchronisation witness backed by the DLA-protected QEC sector."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..qec.dla_protected_subspace import (
    DLAProtectedSubspaceSpec,
    DLAProtectedWitnessResult,
    evaluate_dla_protected_memory,
)

FloatArray = NDArray[np.float64]


class LogicalSyncWitness:
    """Compatibility wrapper for DLA-protected logical synchronisation."""

    def __init__(self, spec: DLAProtectedSubspaceSpec | None = None) -> None:
        self._spec = spec

    def __call__(
        self,
        counts: Mapping[str, int] | None = None,
        *,
        probabilities: FloatArray | None = None,
        **kwargs: Any,
    ) -> dict[str, float | bool | list[str]]:
        """Return the legacy dictionary shape with concrete DLA metrics."""
        if counts is None and probabilities is None and "logical_fidelity" in kwargs:
            if not bool(kwargs.get("allow_fidelity_proxy", False)):
                raise NotImplementedError(
                    "LogicalSyncWitness requires counts or probabilities for production "
                    "evaluation. Set allow_fidelity_proxy=True only for the labelled "
                    "diagnostic fidelity proxy."
                )
            fidelity = _validate_logical_fidelity_proxy(kwargs["logical_fidelity"])
            return {
                "logical_sync_available": 0.0,
                "logical_fidelity_proxy": fidelity,
                "sync_weight_proxy": fidelity,
                "is_logical_sync_witness": 0.0,
            }
        result = self.evaluate(probabilities=probabilities, counts=counts)
        return {
            "logical_fidelity": result.protected_weight,
            "sync_weight": result.sync_weight,
            "logical_sync_order": result.logical_sync_order,
            "parity_leakage": result.parity_leakage,
            "code_leakage": result.code_leakage,
            "passes": result.passes,
            "failure_reasons": list(result.failure_reasons),
        }

    def evaluate(
        self,
        *,
        probabilities: FloatArray | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> DLAProtectedWitnessResult:
        """Return the typed DLA-protected witness result."""
        spec = self._spec or _infer_spec(probabilities=probabilities, counts=counts)
        return evaluate_dla_protected_memory(probabilities=probabilities, counts=counts, spec=spec)


def _infer_spec(
    *,
    probabilities: FloatArray | None,
    counts: Mapping[str, int] | None,
) -> DLAProtectedSubspaceSpec:
    if counts:
        first = next(iter(counts))
        clean = first.replace(" ", "")
        return DLAProtectedSubspaceSpec(n_logical=len(clean), code_distance=1, target_parity=0)
    if probabilities is not None:
        n_qubits = int(np.log2(np.asarray(probabilities).size))
        if 1 << n_qubits != np.asarray(probabilities).size:
            raise ValueError("probabilities length must be a power of two")
        return DLAProtectedSubspaceSpec(n_logical=n_qubits, code_distance=1, target_parity=0)
    raise ValueError("counts, probabilities, or logical_fidelity must be provided")


def _validate_logical_fidelity_proxy(value: Any) -> float:
    fidelity = float(value)
    if not np.isfinite(fidelity) or fidelity < 0.0 or fidelity > 1.0:
        raise ValueError("logical_fidelity must be a finite value in [0, 1].")
    return fidelity
