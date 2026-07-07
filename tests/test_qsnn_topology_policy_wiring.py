# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the QSNN topology-policy wiring
"""Tests for the recurrent-coupling topology policy hook on the QSNN bridge."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.qsnn.quantum_neuromorphic_bridge import (
    QuantumNeuromorphicBridge,
    RecurrentCouplingPolicy,
)
from scpn_quantum_control.topology_control import (
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedSPSAOptimizer,
    TopologicalDynamicCouplingPolicy,
    TopologyConstraintLedger,
)


class _RecordingPolicy:
    """Deterministic stand-in policy that scales weights and counts calls."""

    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor
        self.calls = 0

    def apply(self, recurrent_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        self.calls += 1
        return recurrent_weights * self.factor


class _MalformedShapePolicy:
    def apply(self, recurrent_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((2, 3))


class _NonFinitePolicy:
    def apply(self, recurrent_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        out = recurrent_weights.copy()
        out[0, -1] = np.nan
        return out


def _bridge(**kwargs: object) -> QuantumNeuromorphicBridge:
    return QuantumNeuromorphicBridge(n_inputs=2, n_neurons=4, seed=7, **kwargs)  # type: ignore[arg-type]


def _drive(bridge: QuantumNeuromorphicBridge, steps: int) -> None:
    current = np.ones(bridge.n_inputs, dtype=np.float64)
    for _ in range(steps):
        bridge.step(current)


def test_policy_is_applied_every_step_by_default() -> None:
    policy = _RecordingPolicy()
    bridge = _bridge(topology_policy=policy)
    _drive(bridge, 3)
    assert policy.calls == 3


def test_policy_interval_throttles_application() -> None:
    policy = _RecordingPolicy()
    bridge = _bridge(topology_policy=policy, topology_policy_interval=3)
    _drive(bridge, 7)
    assert policy.calls == 2


def test_no_policy_means_no_projection_state_change() -> None:
    bridge = _bridge()
    _drive(bridge, 2)
    assert bridge.topology_policy is None
    assert bridge._steps_since_topology_projection == 0


def test_interval_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="topology_policy_interval"):
        _bridge(topology_policy=_RecordingPolicy(), topology_policy_interval=0)


def test_policy_output_keeps_bridge_invariants() -> None:
    bridge = _bridge(topology_policy=_RecordingPolicy(factor=100.0))
    _drive(bridge, 1)
    weights = bridge.recurrent_weights
    assert np.all(weights >= bridge.coupling.min_weight)
    assert np.all(weights <= bridge.coupling.max_weight)
    assert np.all(np.diag(weights) == 0.0)
    assert np.all(np.isfinite(weights))


def test_malformed_policy_shape_fails_closed() -> None:
    bridge = _bridge(topology_policy=_MalformedShapePolicy())
    with pytest.raises(ValueError, match="topology_policy output shape"):
        _drive(bridge, 1)


def test_non_finite_policy_output_fails_closed() -> None:
    bridge = _bridge(topology_policy=_NonFinitePolicy())
    with pytest.raises(ValueError, match="topology_policy output must contain only finite"):
        _drive(bridge, 1)


def test_topological_policy_satisfies_the_protocol_and_records_a_trace() -> None:
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.02),
        ledger=TopologyConstraintLedger(),
        h1_target=1.0,
        allow_approximate_ph_backend=True,
    )
    policy = TopologicalDynamicCouplingPolicy(
        objective=objective,
        optimizer=ProjectedSPSAOptimizer(seed=3, max_steps=2),
    )
    assert isinstance(policy, RecurrentCouplingPolicy)

    bridge = _bridge(topology_policy=policy, topology_policy_interval=2)
    _drive(bridge, 2)

    assert policy.last_trace is not None
    assert np.all(np.diag(bridge.recurrent_weights) == 0.0)
    assert np.all(np.isfinite(bridge.recurrent_weights))


def test_facade_exports_the_wiring_surface() -> None:
    """The package facade exposes both sides of the topology wiring."""
    for name in (
        "RecurrentCouplingPolicy",
        "TopologicalDynamicCouplingPolicy",
        "CouplingTopologyObjective",
        "TopologyConstraintLedger",
        "ProjectedSPSAOptimizer",
        "ProjectedScipyOptimizer",
        "NetworkCycleBackend",
        "RipserPHBackend",
        "TopologyOptimisationTrace",
        "export_topology_optimisation_artifact",
        "validate_topology_hardware_manifest",
    ):
        assert hasattr(scpn, name), f"facade is missing {name}"
        assert name in scpn.__all__, f"__all__ is missing {name}"
