# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-aware quantum-kernel facade
"""Public finite-simulator facade for the topology-kernel topology-kernel product."""

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
