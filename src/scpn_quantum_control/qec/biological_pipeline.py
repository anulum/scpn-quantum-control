# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological QEC Pipeline
"""End-to-end biological surface-code pipeline helpers.

Provides campaign-ready orchestration that bundles:
1. Biological surface-code construction
2. Biological topology diagnostics
3. Decoder execution and residual checks
4. JSON-serialisable artefact payloads
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .biological_diagnostics import (
    BiologicalSurfaceDiagnostics,
    analyse_biological_surface_code,
)
from .biological_surface_code import BiologicalMWPMDecoder, BiologicalSurfaceCode


@dataclass(frozen=True)
class BiologicalQecExecution:
    """Structured result of one biological QEC decode execution."""

    code_summary: dict[str, int | bool]
    diagnostics: BiologicalSurfaceDiagnostics
    syndrome_weight: int
    correction_weight: int
    residual_syndrome_weight: int
    decode_backend: str
    success: bool
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-serialisable payload."""
        return {
            "code_summary": dict(self.code_summary),
            "diagnostics": asdict(self.diagnostics),
            "syndrome_weight": int(self.syndrome_weight),
            "correction_weight": int(self.correction_weight),
            "residual_syndrome_weight": int(self.residual_syndrome_weight),
            "decode_backend": str(self.decode_backend),
            "success": bool(self.success),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BiologicalQecBatchExecution:
    """Structured result of a biological QEC campaign with many error patterns."""

    runs: list[BiologicalQecExecution]
    n_runs: int
    n_success: int
    success_rate: float
    decode_backend_counts: dict[str, int]
    mean_syndrome_weight: float
    mean_correction_weight: float
    mean_residual_syndrome_weight: float
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-serialisable payload."""
        return {
            "runs": [run.to_payload() for run in self.runs],
            "n_runs": int(self.n_runs),
            "n_success": int(self.n_success),
            "success_rate": float(self.success_rate),
            "decode_backend_counts": dict(self.decode_backend_counts),
            "mean_syndrome_weight": float(self.mean_syndrome_weight),
            "mean_correction_weight": float(self.mean_correction_weight),
            "mean_residual_syndrome_weight": float(self.mean_residual_syndrome_weight),
            "metadata": dict(self.metadata),
        }


def run_biological_qec_execution(
    K: NDArray[np.float64],
    z_errors: NDArray[np.int8],
    *,
    threshold: float = 1e-5,
    node_domains: dict[int, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BiologicalQecExecution:
    """Run a full biological QEC execution for one error pattern."""
    code = BiologicalSurfaceCode(K, threshold=threshold)
    diagnostics = analyse_biological_surface_code(
        code,
        node_domains=node_domains,
        metadata=metadata,
    )
    decoder = BiologicalMWPMDecoder(code)
    syndrome = code.x_syndrome_from_z_errors(z_errors)
    correction = decoder.decode_z_errors(syndrome)
    residual = code.apply_z_correction(z_errors, correction)
    residual_syndrome = code.x_syndrome_from_z_errors(residual)

    return BiologicalQecExecution(
        code_summary=code.code_summary(),
        diagnostics=diagnostics,
        syndrome_weight=int(np.sum(syndrome)),
        correction_weight=int(np.sum(correction)),
        residual_syndrome_weight=int(np.sum(residual_syndrome)),
        decode_backend=decoder.last_decoder_backend,
        success=bool(np.all(residual_syndrome == 0)),
        metadata=dict(metadata or {}),
    )


def run_biological_qec_batch_execution(
    K: NDArray[np.float64],
    z_error_matrix: NDArray[np.int8],
    *,
    threshold: float = 1e-5,
    node_domains: dict[int, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BiologicalQecBatchExecution:
    """Run biological QEC execution over many error patterns."""
    matrix = np.asarray(z_error_matrix, dtype=np.int8)
    if matrix.ndim != 2:
        raise ValueError("z_error_matrix must be a two-dimensional array.")

    runs: list[BiologicalQecExecution] = []
    for row_idx in range(matrix.shape[0]):
        row_metadata = dict(metadata or {})
        row_metadata["row_index"] = int(row_idx)
        runs.append(
            run_biological_qec_execution(
                K,
                matrix[row_idx],
                threshold=threshold,
                node_domains=node_domains,
                metadata=row_metadata,
            )
        )

    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("z_error_matrix must contain at least one row.")

    n_success = sum(1 for run in runs if run.success)
    backend_counts: dict[str, int] = {}
    syndrome_weights = []
    correction_weights = []
    residual_weights = []
    for run in runs:
        backend_counts[run.decode_backend] = backend_counts.get(run.decode_backend, 0) + 1
        syndrome_weights.append(run.syndrome_weight)
        correction_weights.append(run.correction_weight)
        residual_weights.append(run.residual_syndrome_weight)

    return BiologicalQecBatchExecution(
        runs=runs,
        n_runs=n_runs,
        n_success=n_success,
        success_rate=float(n_success / n_runs),
        decode_backend_counts=backend_counts,
        mean_syndrome_weight=float(np.mean(syndrome_weights)),
        mean_correction_weight=float(np.mean(correction_weights)),
        mean_residual_syndrome_weight=float(np.mean(residual_weights)),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "BiologicalQecExecution",
    "BiologicalQecBatchExecution",
    "run_biological_qec_execution",
    "run_biological_qec_batch_execution",
]
