# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sync order proxy contract tests
# SCPN Quantum Control - Sync-order proxy contract tests
"""Contract tests for the count-derived synchronisation proxy boundary."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter
from scpn_quantum_control.bridge import QPUDataArtifact, artifact_from_arrays
from scpn_quantum_control.qpu_compute import execute_simulator_request, make_compute_request


def _artifact() -> QPUDataArtifact:
    """Return a minimal real QPU data artifact for simulator execution."""
    return artifact_from_arrays(
        domain="unit",
        source_name="sync-proxy-contract",
        source_mode="curated",
        K_nm=np.array(
            [
                [0.0, 0.25],
                [0.25, 0.0],
            ],
            dtype=np.float64,
        ),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        normalization="unit",
        extraction_method="contract-test",
        replay_id="sync-proxy-contract:unit",
    )


def test_sync_order_proxy_exposes_legacy_alias_and_claim_boundary() -> None:
    """The counts path exposes Z-magnetisation while refusing a true-R claim."""
    result = SyncOrderParameter()(counts={"000": 75, "111": 25})

    assert result["sync_order"] == result["sync_order_z_magnetisation"]
    assert result["sync_order_z_magnetisation"] == 0.5
    assert result["is_xy_kuramoto_order_parameter"] == 0.0


def test_simulator_artifact_labels_explicit_proxy_fields() -> None:
    """Simulator results classify the new alias without breaking the legacy key."""
    artifact = _artifact()
    request = make_compute_request(
        artifact,
        kernel="sync_dla",
        shots=256,
        trotter_depth=2,
        coupling_scale=1.5,
    )

    result = execute_simulator_request(artifact, request)

    assert result.observables["sync_order"] == result.observables["sync_order_z_magnetisation"]
    assert result.observables["is_xy_kuramoto_order_parameter"] == 0.0
    assert result.observable_classification["sync_order"] == "simulated_exact_statevector"
    assert (
        result.observable_classification["sync_order_z_magnetisation"]
        == "z_magnetisation_proxy_from_counts"
    )
    assert (
        result.observable_classification["is_xy_kuramoto_order_parameter"] == "claim_boundary_flag"
    )
