# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable classical surrogates
"""Classical surrogate fitting, fidelity, and exact-validation surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY,
        QUANTUM_RESERVOIR_EVIDENCE_SCHEMA,
        QuantumReservoirSurrogateEvidence,
        SurrogateSupportRow,
        render_quantum_reservoir_surrogate_markdown,
        write_quantum_reservoir_surrogate_evidence,
    )
    from .train import SurrogateFitConfig, fit_gaussian_rbf_surrogate, input_row_digests

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "SurrogateFidelityCertificate": (
        "scpn_quantum_control.surrogates.fidelity",
        "SurrogateFidelityCertificate",
    ),
    "SurrogateFidelityThresholds": (
        "scpn_quantum_control.surrogates.fidelity",
        "SurrogateFidelityThresholds",
    ),
    "SurrogateGradientCertificate": (
        "scpn_quantum_control.surrogates.fidelity",
        "SurrogateGradientCertificate",
    ),
    "certify_surrogate_fidelity": (
        "scpn_quantum_control.surrogates.fidelity",
        "certify_surrogate_fidelity",
    ),
    "certify_surrogate_gradient": (
        "scpn_quantum_control.surrogates.fidelity",
        "certify_surrogate_gradient",
    ),
    "ExactValidatedSurrogateProposal": (
        "scpn_quantum_control.surrogates.hybrid",
        "ExactValidatedSurrogateProposal",
    ),
    "propose_and_validate_surrogate_step": (
        "scpn_quantum_control.surrogates.hybrid",
        "propose_and_validate_surrogate_step",
    ),
    "CLASSICAL_SURROGATE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.surrogates.models",
        "CLASSICAL_SURROGATE_CLAIM_BOUNDARY",
    ),
    "GaussianRBFSurrogate": ("scpn_quantum_control.surrogates.models", "GaussianRBFSurrogate"),
    "QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY": (
        "scpn_quantum_control.surrogates.report",
        "QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY",
    ),
    "QUANTUM_RESERVOIR_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.surrogates.report",
        "QUANTUM_RESERVOIR_EVIDENCE_SCHEMA",
    ),
    "QuantumReservoirSurrogateEvidence": (
        "scpn_quantum_control.surrogates.report",
        "QuantumReservoirSurrogateEvidence",
    ),
    "SurrogateSupportRow": ("scpn_quantum_control.surrogates.report", "SurrogateSupportRow"),
    "render_quantum_reservoir_surrogate_markdown": (
        "scpn_quantum_control.surrogates.report",
        "render_quantum_reservoir_surrogate_markdown",
    ),
    "write_quantum_reservoir_surrogate_evidence": (
        "scpn_quantum_control.surrogates.report",
        "write_quantum_reservoir_surrogate_evidence",
    ),
    "SurrogateFitConfig": ("scpn_quantum_control.surrogates.train", "SurrogateFitConfig"),
    "fit_gaussian_rbf_surrogate": (
        "scpn_quantum_control.surrogates.train",
        "fit_gaussian_rbf_surrogate",
    ),
    "input_row_digests": ("scpn_quantum_control.surrogates.train", "input_row_digests"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "QUANTUM_RESERVOIR_EVIDENCE_BOUNDARY",
    "QUANTUM_RESERVOIR_EVIDENCE_SCHEMA",
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
