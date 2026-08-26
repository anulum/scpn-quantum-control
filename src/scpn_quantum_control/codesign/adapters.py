# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design adapters over existing product ports
"""Co-design adapters for control ports and sensing, identity, geometry, and FIM observers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ..active_sensing_product import ActiveSensingObserverRecord
from ..analysis.adaptive_fim_feedback import AdaptiveFIMObserverRecord, AdaptiveFIMStep
from ..control.closed_loop_analysis import ClosedLoopExecutionPolicy
from ..control_stack_runtime_adapters import (
    QaoaMpcPort,
    RealtimeFeedbackPort,
    run_cosimulation_partition_adapter,
    run_qaoa_mpc_adapter,
    run_realtime_feedback_adapter,
)
from ..identity_observer_product import IdentitySafetyDecision
from ..ssgf_geometry_gradient_product import SsgfGeometryObserverRecord
from .contracts import (
    CODESIGN_CLAIM_BOUNDARY,
    CODESIGN_SCHEMA,
    ControllerProposal,
    ObserverInputs,
)


@dataclass(frozen=True, slots=True)
class ControlAdapterEvidence:
    """Compact evidence from an existing policy-gated control port."""

    adapter_id: str
    authorised: bool
    values: tuple[float, ...]
    samples: int
    hardware_execution: bool = False
    schema: str = CODESIGN_SCHEMA
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready adapter evidence mapping."""
        return asdict(self)


def observer_inputs_from_products(
    *,
    active_sensing: ActiveSensingObserverRecord | None = None,
    identity: IdentitySafetyDecision | None = None,
    geometry: SsgfGeometryObserverRecord | None = None,
    adaptive_fim: AdaptiveFIMObserverRecord | None = None,
) -> ObserverInputs:
    """Map completed observer products into bounded co-design telemetry."""
    return ObserverInputs(
        active_sensing_id=(None if active_sensing is None else active_sensing.observer_id),
        identity_action=None if identity is None else identity.action,
        identity_reason=None if identity is None else identity.reason,
        geometry_gradient_norm=(None if geometry is None else geometry.gradient_norm),
        adaptive_fim_id=(None if adaptive_fim is None else adaptive_fim.observer_id),
        adaptive_fim_action=(None if adaptive_fim is None else adaptive_fim.action),
        adaptive_fim_lambda_out=(None if adaptive_fim is None else adaptive_fim.lambda_out),
    )


def adaptive_fim_proposal_port(step: AdaptiveFIMStep) -> ControllerProposal:
    """Map one adaptive-FIM proposal to the unapplied controller port.

    The adapter does not pass a safety envelope or apply the proposal. A hold is
    represented by a zero update; ``gain_scale=1`` avoids inventing a controller gain.
    """
    return ControllerProposal(
        parameters=(step.lambda_out,),
        update=(step.lambda_out - step.lambda_in,),
        gain_scale=1.0,
    )


def consume_realtime_feedback_port(
    controller: RealtimeFeedbackPort,
    *,
    policy: ClosedLoopExecutionPolicy,
    n_rounds: int,
    seed: int | None = None,
) -> ControlAdapterEvidence:
    """Consume the existing realtime-feedback port under execution policy."""
    result = run_realtime_feedback_adapter(
        controller,
        policy=policy,
        n_rounds=n_rounds,
        seed=seed,
    )
    return ControlAdapterEvidence(
        adapter_id="control_stack.realtime_feedback",
        authorised=result.decision.authorised,
        values=tuple(float(step.r_live) for step in result.steps),
        samples=len(result.steps),
    )


def consume_qaoa_mpc_port(
    controller: QaoaMpcPort,
    *,
    policy: ClosedLoopExecutionPolicy,
    seed: int | None = None,
) -> ControlAdapterEvidence:
    """Consume the existing abstract QAOA-MPC port under execution policy."""
    result = run_qaoa_mpc_adapter(controller, policy=policy, seed=seed)
    return ControlAdapterEvidence(
        adapter_id="control_stack.qaoa_mpc",
        authorised=result.decision.authorised,
        values=tuple(float(action) for action in result.actions),
        samples=len(result.actions),
    )


def consume_cosimulation_port(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    policy: ClosedLoopExecutionPolicy,
    dt: float,
    n_steps: int,
    max_quantum_nodes: int = 8,
    seed: int | None = None,
) -> ControlAdapterEvidence:
    """Consume the existing policy-gated co-simulation partition."""
    result = run_cosimulation_partition_adapter(
        coupling,
        omega,
        policy=policy,
        dt=dt,
        n_steps=n_steps,
        max_quantum_nodes=max_quantum_nodes,
        seed=seed,
    )
    telemetry = result.telemetry
    return ControlAdapterEvidence(
        adapter_id="control_stack.cosimulation_partition",
        authorised=result.decision.authorised,
        values=(
            telemetry.final_quantum_order,
            telemetry.final_classical_order,
            telemetry.final_global_order,
        ),
        samples=telemetry.samples,
    )


__all__ = [
    "ControlAdapterEvidence",
    "adaptive_fim_proposal_port",
    "consume_cosimulation_port",
    "consume_qaoa_mpc_port",
    "consume_realtime_feedback_port",
    "observer_inputs_from_products",
]
