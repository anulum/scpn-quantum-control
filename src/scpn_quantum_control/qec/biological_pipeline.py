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


def run_biological_qec_execution(
    K: np.ndarray,
    z_errors: np.ndarray,
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


__all__ = [
    "BiologicalQecExecution",
    "run_biological_qec_execution",
]
