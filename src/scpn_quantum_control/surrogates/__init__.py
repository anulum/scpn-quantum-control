# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable classical surrogates
"""Classical surrogate fitting, fidelity, and exact-validation surfaces."""

from .fidelity import (
    SurrogateFidelityCertificate,
    SurrogateFidelityThresholds,
    SurrogateGradientCertificate,
    certify_surrogate_fidelity,
    certify_surrogate_gradient,
)
from .hybrid import (
    ExactValidatedSurrogateProposal,
    propose_and_validate_surrogate_step,
)
from .models import CLASSICAL_SURROGATE_CLAIM_BOUNDARY, GaussianRBFSurrogate
from .report import (
    BL45_EVIDENCE_BOUNDARY,
    BL45_EVIDENCE_SCHEMA,
    QuantumReservoirSurrogateEvidence,
    SurrogateSupportRow,
    render_quantum_reservoir_surrogate_markdown,
    write_quantum_reservoir_surrogate_evidence,
)
from .train import SurrogateFitConfig, fit_gaussian_rbf_surrogate, input_row_digests

__all__ = [
    "BL45_EVIDENCE_BOUNDARY",
    "BL45_EVIDENCE_SCHEMA",
    "CLASSICAL_SURROGATE_CLAIM_BOUNDARY",
    "ExactValidatedSurrogateProposal",
    "GaussianRBFSurrogate",
    "QuantumReservoirSurrogateEvidence",
    "SurrogateFidelityCertificate",
    "SurrogateFidelityThresholds",
    "SurrogateFitConfig",
    "SurrogateGradientCertificate",
    "SurrogateSupportRow",
    "certify_surrogate_fidelity",
    "certify_surrogate_gradient",
    "fit_gaussian_rbf_surrogate",
    "input_row_digests",
    "propose_and_validate_surrogate_step",
    "render_quantum_reservoir_surrogate_markdown",
    "write_quantum_reservoir_surrogate_evidence",
]
