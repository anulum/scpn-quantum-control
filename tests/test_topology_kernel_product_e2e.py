# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-kernel public-facade test
"""Small end-to-end use through the documented public API."""

from __future__ import annotations

import numpy as np

import scpn_quantum_control.topology_kernel_product as product


def test_public_facade_trains_and_predicts_without_provider_calls() -> None:
    config = product.TopologyKernelConfig(n_qubits=3, max_samples=4)
    topology = product.ring_topology(config.n_qubits)
    train = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    ids = ("positive", "negative")
    kernel = product.fidelity_kernel_matrix(
        train,
        train,
        topology,
        config,
        row_ids=ids,
        column_ids=ids,
    )
    model = product.fit_kernel_ridge(kernel, np.array([1, -1]), alpha=config.ridge)
    predictions = product.predict_kernel_ridge(model, kernel)
    assert predictions.tolist() == [1, -1]
    assert product.TOPOLOGY_KERNEL_CLAIM_BOUNDARY in kernel.claim_boundary
    assert set(product.__all__) >= {
        "TopologyKernelConfig",
        "fidelity_kernel_matrix",
        "fit_kernel_ridge",
        "build_topology_kernel_evidence",
    }
