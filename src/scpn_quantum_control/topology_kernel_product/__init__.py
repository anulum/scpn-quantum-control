# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-aware quantum-kernel facade
"""Public finite-simulator facade for the topology-kernel topology-kernel product."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .classifier import (
        KernelRidgeClassifier,
        evaluate_kernel_ridge,
        fit_kernel_ridge,
        predict_kernel_ridge,
    )
    from .evidence import (
        TOPOLOGY_KERNEL_EVIDENCE_DATE,
        TOPOLOGY_KERNEL_EVIDENCE_SCHEMA,
        KernelSupportRow,
        TopologyKernelEvidence,
        build_topology_kernel_evidence,
        render_topology_kernel_markdown,
        write_topology_kernel_evidence,
    )
    from .kernels import (
        fidelity_kernel_matrix,
        permute_edge_features,
        permute_topology,
        rbf_kernel_matrix,
        topology_digest,
        validate_feature_matrix,
        validate_topology,
    )
    from .schema import (
        TOPOLOGY_KERNEL_CLAIM_BOUNDARY,
        KernelEvaluation,
        TopologyKernelConfig,
        TopologyKernelDataset,
        TopologyKernelMatrix,
    )
    from .synthetic import (
        build_teacher_aligned_dataset,
        complete_topology,
        path_topology,
        ring_topology,
        zero_topology,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "KernelRidgeClassifier": (
        "scpn_quantum_control.topology_kernel_product.classifier",
        "KernelRidgeClassifier",
    ),
    "evaluate_kernel_ridge": (
        "scpn_quantum_control.topology_kernel_product.classifier",
        "evaluate_kernel_ridge",
    ),
    "fit_kernel_ridge": (
        "scpn_quantum_control.topology_kernel_product.classifier",
        "fit_kernel_ridge",
    ),
    "predict_kernel_ridge": (
        "scpn_quantum_control.topology_kernel_product.classifier",
        "predict_kernel_ridge",
    ),
    "TOPOLOGY_KERNEL_EVIDENCE_DATE": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "TOPOLOGY_KERNEL_EVIDENCE_DATE",
    ),
    "TOPOLOGY_KERNEL_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "TOPOLOGY_KERNEL_EVIDENCE_SCHEMA",
    ),
    "KernelSupportRow": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "KernelSupportRow",
    ),
    "TopologyKernelEvidence": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "TopologyKernelEvidence",
    ),
    "build_topology_kernel_evidence": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "build_topology_kernel_evidence",
    ),
    "render_topology_kernel_markdown": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "render_topology_kernel_markdown",
    ),
    "write_topology_kernel_evidence": (
        "scpn_quantum_control.topology_kernel_product.evidence",
        "write_topology_kernel_evidence",
    ),
    "fidelity_kernel_matrix": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "fidelity_kernel_matrix",
    ),
    "permute_edge_features": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "permute_edge_features",
    ),
    "permute_topology": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "permute_topology",
    ),
    "rbf_kernel_matrix": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "rbf_kernel_matrix",
    ),
    "topology_digest": ("scpn_quantum_control.topology_kernel_product.kernels", "topology_digest"),
    "validate_feature_matrix": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "validate_feature_matrix",
    ),
    "validate_topology": (
        "scpn_quantum_control.topology_kernel_product.kernels",
        "validate_topology",
    ),
    "TOPOLOGY_KERNEL_CLAIM_BOUNDARY": (
        "scpn_quantum_control.topology_kernel_product.schema",
        "TOPOLOGY_KERNEL_CLAIM_BOUNDARY",
    ),
    "KernelEvaluation": (
        "scpn_quantum_control.topology_kernel_product.schema",
        "KernelEvaluation",
    ),
    "TopologyKernelConfig": (
        "scpn_quantum_control.topology_kernel_product.schema",
        "TopologyKernelConfig",
    ),
    "TopologyKernelDataset": (
        "scpn_quantum_control.topology_kernel_product.schema",
        "TopologyKernelDataset",
    ),
    "TopologyKernelMatrix": (
        "scpn_quantum_control.topology_kernel_product.schema",
        "TopologyKernelMatrix",
    ),
    "build_teacher_aligned_dataset": (
        "scpn_quantum_control.topology_kernel_product.synthetic",
        "build_teacher_aligned_dataset",
    ),
    "complete_topology": (
        "scpn_quantum_control.topology_kernel_product.synthetic",
        "complete_topology",
    ),
    "path_topology": ("scpn_quantum_control.topology_kernel_product.synthetic", "path_topology"),
    "ring_topology": ("scpn_quantum_control.topology_kernel_product.synthetic", "ring_topology"),
    "zero_topology": ("scpn_quantum_control.topology_kernel_product.synthetic", "zero_topology"),
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
    "TOPOLOGY_KERNEL_EVIDENCE_DATE",
    "TOPOLOGY_KERNEL_EVIDENCE_SCHEMA",
    "TOPOLOGY_KERNEL_CLAIM_BOUNDARY",
    "KernelEvaluation",
    "KernelRidgeClassifier",
    "KernelSupportRow",
    "TopologyKernelConfig",
    "TopologyKernelDataset",
    "TopologyKernelEvidence",
    "TopologyKernelMatrix",
    "build_teacher_aligned_dataset",
    "build_topology_kernel_evidence",
    "complete_topology",
    "evaluate_kernel_ridge",
    "fidelity_kernel_matrix",
    "fit_kernel_ridge",
    "path_topology",
    "permute_edge_features",
    "permute_topology",
    "predict_kernel_ridge",
    "rbf_kernel_matrix",
    "render_topology_kernel_markdown",
    "ring_topology",
    "topology_digest",
    "validate_feature_matrix",
    "validate_topology",
    "write_topology_kernel_evidence",
    "zero_topology",
]
