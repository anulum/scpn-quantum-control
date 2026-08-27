# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control-stack executable adapter tests
"""Tests for policy-gated adapters over the existing control stack."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control.realtime_feedback import (
    FeedbackStep,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.control_stack_runtime_adapters import (
    CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA,
    PolicyGatedAdapterError,
    decide_pulse_compose_boundary,
    run_cosimulation_partition_adapter,
    run_qaoa_mpc_adapter,
    run_realtime_feedback_adapter,
)


class _CountingFeedbackPort:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, n_steps: int, seed: int | None = None) -> list[FeedbackStep]:
        self.calls += 1
        return []


class _ActionPort:
    def __init__(self, actions: NDArray[np.int64]) -> None:
        self.actions = actions
        self.calls = 0

    def optimize(self, seed: int | None = None) -> NDArray[np.int64]:
        self.calls += 1
        return self.actions


def test_realtime_adapter_uses_existing_controller() -> None:
    """Run feedback through the existing simulation controller."""
    controller = RealtimeSyncFeedbackController(
        np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        np.array([0.1, -0.1], dtype=np.float64),
    )
    result = run_realtime_feedback_adapter(
        controller,
        policy=ClosedLoopExecutionPolicy(),
        n_rounds=2,
        seed=17,
    )

    assert result.decision.authorised is True
    assert result.decision.mode.value == "simulation"
    assert len(result.steps) == 2
    assert result.steps == tuple(controller.history)
    assert result.schema == CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA


def test_absent_or_refused_policy_never_calls_port() -> None:
    """Keep the execution port untouched when policy admission fails."""
    port = _CountingFeedbackPort()
    with pytest.raises(PolicyGatedAdapterError, match="ExecutionPolicy is required"):
        run_realtime_feedback_adapter(port, policy=None, n_rounds=2)
    assert port.calls == 0

    policy = ClosedLoopExecutionPolicy(round_budget=1)
    with pytest.raises(PolicyGatedAdapterError, match="exceed the policy budget"):
        run_realtime_feedback_adapter(port, policy=policy, n_rounds=2)
    assert port.calls == 0


def test_hardware_authorisation_is_still_local_adapter_refused() -> None:
    """Refuse hardware routing even when a policy admits its backend."""
    port = _CountingFeedbackPort()
    policy = ClosedLoopExecutionPolicy(
        allow_hardware=True,
        live_ticket="owner-ticket",
        backend_allowlist=("provider",),
    )
    with pytest.raises(PolicyGatedAdapterError, match="local-simulator only"):
        run_realtime_feedback_adapter(
            port,
            policy=policy,
            n_rounds=2,
            backend="provider",
        )
    assert port.calls == 0


def test_qaoa_mpc_adapter_uses_existing_optimizer() -> None:
    """Run the ambient optimizer through the policy-gated adapter."""
    controller = QAOA_MPC(
        np.array([[1.0]], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        horizon=1,
        p_layers=1,
    )
    result = run_qaoa_mpc_adapter(
        controller,
        policy=ClosedLoopExecutionPolicy(),
        seed=5,
    )

    assert result.actions in {(0,), (1,)}
    assert result.decision.authorised is True


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.array([], dtype=np.int64), "non-empty 1-D"),
        (np.array([[0, 1]], dtype=np.int64), "non-empty 1-D"),
        (np.array([0, 2], dtype=np.int64), "must be binary"),
    ],
)
def test_qaoa_adapter_rejects_invalid_ambient_schedule(
    actions: NDArray[np.int64], message: str
) -> None:
    """Reject empty, non-vector, and non-binary optimizer schedules."""
    port = _ActionPort(actions)
    with pytest.raises(ValueError, match=message):
        run_qaoa_mpc_adapter(port, policy=ClosedLoopExecutionPolicy())
    assert port.calls == 1


def test_cosimulation_partition_adapter_maps_ambient_result() -> None:
    """Map the ambient co-simulation result into bounded telemetry."""
    K = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    omega = np.array([0.1, -0.1], dtype=np.float64)
    adapted = run_cosimulation_partition_adapter(
        K,
        omega,
        policy=ClosedLoopExecutionPolicy(),
        dt=0.02,
        n_steps=2,
        max_quantum_nodes=1,
        seed=3,
    )

    assert adapted.decision.authorised is True
    assert adapted.telemetry.samples == 3
    assert adapted.telemetry.n_quantum + adapted.telemetry.n_classical == 2
    assert adapted.telemetry.final_global_order == pytest.approx(adapted.result.global_order[-1])
    assert "not exact, not hardware" in adapted.telemetry.claim_boundary


def test_pulse_boundary_is_explicitly_fail_closed_to_execution_adapter() -> None:
    """Keep pulse composition outside the executable adapter boundary."""
    decision = decide_pulse_compose_boundary(policy=ClosedLoopExecutionPolicy())

    assert decision.allowed is False
    assert decision.dependency == "pulse-boundary/runtime-adapter"
    assert "outside control-stack adapter ownership" in decision.reason
    assert decision.schema == CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA


def test_result_contract_rejects_stale_schema_and_claim_drift() -> None:
    """Reject stale schemas and altered result claim boundaries."""
    result = run_realtime_feedback_adapter(
        _CountingFeedbackPort(),
        policy=ClosedLoopExecutionPolicy(),
        n_rounds=1,
    )

    with pytest.raises(ValueError, match="runtime-adapter schema"):
        replace(result, schema="control_stack_runtime_adapters.v1")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(result, claim_boundary="altered")
