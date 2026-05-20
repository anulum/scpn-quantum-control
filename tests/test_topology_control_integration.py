# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Integration Tests
"""Integration tests for topology control artefacts, QSNN policy, and hardware gate."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import scpn_quantum_control.control.topological_optimizer as legacy_topology
import scpn_quantum_control.topology_control as topology_control
from scpn_quantum_control.qsnn.quantum_neuromorphic_bridge import QuantumNeuromorphicBridge
from scpn_quantum_control.topology_control import (
    CouplingGraphBounds,
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedSPSAOptimizer,
    TopologicalDynamicCouplingPolicy,
    TopologyConstraintLedger,
    TopologyHardwareManifest,
    export_topology_optimisation_artifact,
    validate_topology_hardware_manifest,
)


def _square_coupling() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.25, 0.0, 0.25],
            [0.25, 0.0, 0.25, 0.0],
            [0.0, 0.25, 0.0, 0.25],
            [0.25, 0.0, 0.25, 0.0],
        ]
    )


def test_artifact_export_round_trips_with_stable_digest() -> None:
    K0 = _square_coupling()
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=TopologyConstraintLedger(bounds=CouplingGraphBounds(0.0, 0.5)),
        source_matrix=K0,
    )
    trace = ProjectedSPSAOptimizer(seed=7, max_steps=2).optimise(K0, objective)

    artifact = export_topology_optimisation_artifact(trace, claim_boundary="test boundary")
    payload = json.loads(artifact.to_json())
    round_trip = json.loads(artifact.to_json())

    assert payload == round_trip
    assert payload["sha256"] == artifact.sha256
    assert payload["claim_boundary"] == "test boundary"
    assert payload["steps"][-1]["objective_total"] == pytest.approx(
        trace.steps[-1].objective.total
    )


def test_topological_policy_projects_qsnn_recurrent_weights() -> None:
    bridge = QuantumNeuromorphicBridge(
        n_inputs=2,
        n_neurons=4,
        seed=11,
        recurrent_weights=_square_coupling(),
        deterministic=True,
    )
    policy = TopologicalDynamicCouplingPolicy(
        objective=CouplingTopologyObjective(
            ph_backend=NetworkCycleBackend(threshold=0.2),
            ledger=TopologyConstraintLedger(
                bounds=CouplingGraphBounds(0.0, 0.4),
                hardware_edges={(0, 1), (1, 2), (2, 3), (0, 3)},
            ),
            source_matrix=_square_coupling(),
        ),
        optimizer=ProjectedSPSAOptimizer(seed=5, max_steps=2),
    )

    result = bridge.step(np.array([1.0, 1.0]))
    projected = policy.apply(result.recurrent_weights)

    np.testing.assert_allclose(projected, projected.T)
    np.testing.assert_allclose(np.diag(projected), 0.0)
    assert np.all(projected <= 0.4)
    assert projected[0, 2] == pytest.approx(0.0)


def test_hardware_manifest_rejects_missing_preregistration_and_budget() -> None:
    manifest = TopologyHardwareManifest(
        backend_name="ibm_fez",
        qubits=(21, 22, 23, 24),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        shots=4096,
        qpu_minute_ceiling=None,
        preregistration_id="",
        objective_sha256="a" * 64,
        require_readout_calibration=True,
    )

    with pytest.raises(ValueError, match="preregistration_id"):
        validate_topology_hardware_manifest(manifest)


def test_hardware_manifest_accepts_provider_neutral_descriptor() -> None:
    descriptor = SimpleNamespace(
        name="local_aer",
        capabilities=SimpleNamespace(n_qubits=8, coupling_map=[(0, 1), (1, 2), (2, 3)]),
    )
    manifest = TopologyHardwareManifest(
        backend_name="local_aer",
        qubits=(0, 1, 2, 3),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        shots=1024,
        qpu_minute_ceiling=0.0,
        preregistration_id="no-qpu-smoke",
        objective_sha256="b" * 64,
        require_readout_calibration=False,
    )

    validated = validate_topology_hardware_manifest(manifest, backend_descriptor=descriptor)

    assert validated.backend_name == "local_aer"
    assert validated.live_submission_allowed is False


def test_legacy_wrapper_has_no_cross_repo_dependency() -> None:
    legacy_source = inspect.getsource(legacy_topology)

    forbidden = (
        "sc_neurocore",
        "scpn_phase_orchestrator",
        "scpn_fusion_core",
        "remote_data",
        "agentic_shared",
    )

    for token in forbidden:
        assert token not in legacy_source.lower()
        assert token not in inspect.getsource(topology_control).lower()
