# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design adapter tests
"""Real control-stack and observer-product integration tests for co-design."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.active_sensing_product import ActiveSensingObserverRecord
from scpn_quantum_control.analysis.adaptive_fim_feedback import (
    AdaptiveFIMConfig,
    FIMWitness,
    observer_record_from_step,
    propose_count_aware_lambda,
)
from scpn_quantum_control.codesign.adapters import (
    adaptive_fim_proposal_port,
    consume_cosimulation_port,
    consume_qaoa_mpc_port,
    consume_realtime_feedback_port,
    observer_inputs_from_products,
)
from scpn_quantum_control.codesign.contracts import ObserverInputs
from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.identity_observer_product import (
    IdentityObserverRecord,
    IdentityObserverThresholds,
    IdentitySafetyDecision,
)
from scpn_quantum_control.ssgf_geometry_gradient_product import (
    SsgfGeometryObserverRecord,
)


def test_realtime_feedback_adapter_consumes_existing_control_port() -> None:
    """Consume the real realtime-feedback controller through the control port."""
    controller = RealtimeSyncFeedbackController(
        np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        np.array([0.1, -0.1], dtype=np.float64),
    )
    evidence = consume_realtime_feedback_port(
        controller,
        policy=ClosedLoopExecutionPolicy(),
        n_rounds=2,
        seed=17,
    )

    assert evidence.adapter_id == "control_stack.realtime_feedback"
    assert evidence.authorised is True
    assert evidence.samples == 2
    assert evidence.values == tuple(step.r_live for step in controller.history)
    assert evidence.to_dict()["hardware_execution"] is False


def test_qaoa_adapter_consumes_existing_control_port() -> None:
    """Consume the real abstract QAOA-MPC optimiser through the control port."""
    controller = QAOA_MPC(
        np.array([[1.0]], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        horizon=1,
        p_layers=1,
    )
    evidence = consume_qaoa_mpc_port(
        controller,
        policy=ClosedLoopExecutionPolicy(),
        seed=5,
    )

    assert evidence.adapter_id == "control_stack.qaoa_mpc"
    assert evidence.samples == 1
    assert evidence.values in {(0.0,), (1.0,)}


def test_cosimulation_adapter_consumes_existing_control_partition() -> None:
    """Consume the real policy-gated quantum/classical partition."""
    evidence = consume_cosimulation_port(
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        np.array([0.1, -0.1], dtype=np.float64),
        policy=ClosedLoopExecutionPolicy(),
        dt=0.02,
        n_steps=2,
        max_quantum_nodes=1,
        seed=3,
    )

    assert evidence.adapter_id == "control_stack.cosimulation_partition"
    assert evidence.samples == 3
    assert len(evidence.values) == 3
    assert all(0.0 <= value <= 1.0 for value in evidence.values)


def test_observer_adapter_maps_sensing_identity_and_geometry_records() -> None:
    """Map public observer records without promoting their claim boundaries."""
    active = ActiveSensingObserverRecord(
        observer_id="active-1",
        channel="gradient_observer",
        selected_observable_id="r_global",
        expected_information_gain_nats=0.2,
        posterior_variance=0.1,
        shots=16,
        shot_policy_id="local_simulator_default",
        analytic_design_protocol_id="analytic-design",
    )
    identity_record = IdentityObserverRecord(
        energy_gap=0.5,
        transition_probability=0.01,
        adiabatic_bound=0.02,
        planned_depth=4,
        coherence_max_depth=10,
        coherence_fidelity=0.9,
        witness_status="not_requested",
        chsh_value=None,
        witness_pair=None,
    )
    thresholds = IdentityObserverThresholds(0.1, 0.1, 0.8)
    identity = IdentitySafetyDecision(
        allowed=False,
        action="hold",
        reason="energy-gap margin review",
        blockers=("energy_gap",),
        observer=identity_record,
        thresholds=thresholds,
    )
    geometry = SsgfGeometryObserverRecord(
        cost=0.2,
        r_global=0.8,
        gradient_norm=0.3,
        geometry_symmetry_residual=0.0,
        method="finite_difference",
        route_id="transform:ssgf.latent_finite_difference",
    )

    mapped = observer_inputs_from_products(
        active_sensing=active,
        identity=identity,
        geometry=geometry,
    )

    assert mapped.active_sensing_id == "active-1"
    assert mapped.identity_action == "hold"
    assert mapped.identity_reason == "energy-gap margin review"
    assert mapped.geometry_gradient_norm == 0.3
    assert observer_inputs_from_products().to_dict() == {
        "active_sensing_id": None,
        "identity_action": None,
        "identity_reason": None,
        "geometry_gradient_norm": None,
    }


def test_adaptive_fim_observer_and_proposer_ports_remain_unapplied() -> None:
    """Map proposal telemetry and scalar values without applying a controller."""
    witness = FIMWitness.from_counts(
        leakage_events=100,
        retention_events=800,
        shots=1024,
        source="synthetic",
    )
    step = propose_count_aware_lambda(
        4.0,
        witness,
        AdaptiveFIMConfig(target_leakage=0.05, step_gain=4.0),
    )
    observer = observer_record_from_step(step, policy_id="ci_dry_run_only")

    mapped = observer_inputs_from_products(adaptive_fim=observer)
    proposal = adaptive_fim_proposal_port(step)

    assert mapped.adaptive_fim_id == "adaptive_fim:ci_dry_run_only:0"
    assert mapped.adaptive_fim_action == "decrease"
    assert mapped.adaptive_fim_lambda_out == step.lambda_out
    assert mapped.to_dict()["adaptive_fim_action"] == "decrease"
    assert proposal.parameters == (step.lambda_out,)
    assert proposal.update == (step.lambda_out - step.lambda_in,)
    assert proposal.gain_scale == 1.0


def test_adaptive_fim_observer_fields_fail_closed() -> None:
    """Reject partial, invalid, blank, non-finite, and negative proposal telemetry."""
    with np.testing.assert_raises_regex(ValueError, "adaptive_fim_action"):
        ObserverInputs(adaptive_fim_action="increase")
    with np.testing.assert_raises_regex(ValueError, "supplied together"):
        ObserverInputs(adaptive_fim_id="id")
    with np.testing.assert_raises_regex(ValueError, "non-empty"):
        ObserverInputs(
            adaptive_fim_id="",
            adaptive_fim_action="hold",
            adaptive_fim_lambda_out=1.0,
        )
    with np.testing.assert_raises_regex(ValueError, "finite"):
        ObserverInputs(
            adaptive_fim_id="id",
            adaptive_fim_action="hold",
            adaptive_fim_lambda_out=float("nan"),
        )
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        ObserverInputs(
            adaptive_fim_id="id",
            adaptive_fim_action="hold",
            adaptive_fim_lambda_out=-1.0,
        )
