# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned control and topology co-design instrument data
"""Project control proposals, constraints and simulated traces for read-only inspection.

Every value comes from an existing control owner run under the closed-loop
execution policy in local simulation:

- tracking costs from the QAOA-MPC cost Hamiltonian, where the target keeps its
  sign and direction;
- constrained coupling proposals from the co-design session, which recomputes
  feasibility itself and adopts a plan only when it verifies;
- replayed latency samples judged by the closed-loop latency budget;
- a quantum/classical partition trace whose timestamps are the integrator's own.

Nothing here actuates hardware. A proposal is never applied: no runtime
actuation evidence exists, and observed hardware telemetry stays unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..control.closed_loop_analysis import (
    ClosedLoopExecutionPolicy,
    ClosedLoopLatencyBudget,
    measure_closed_loop_latency_budget,
)
from ..control.qaoa_mpc import QAOA_MPC
from ..control.realtime_feedback import RealtimeFeedbackConfig, RealtimeSyncFeedbackController
from ..control_stack_runtime_adapters import (
    CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY,
    ConstrainedCodesignSession,
    ConstrainedProposal,
    OptimiserOutcome,
    PolicyGatedAdapterError,
    run_cosimulation_partition_adapter,
)
from .committed_data import run_committed_data_cli

CONTROL_PROPOSAL_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_control_proposal_instrument_v1"
DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data/studio/control_proposal_instrument_20261109.json"
)
APPLIED_STATUS_RULE: Final[str] = (
    "A proposal counts as applied only with explicit runtime actuation evidence; "
    "this local-simulation instrument has none, so every proposal is unapplied"
)

_EDGES: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))
_EDGE_CAPS: Final[NDArray[np.float64]] = np.array([0.6, 0.5, 0.4])
_DESIRED: Final[NDArray[np.float64]] = np.array([0.9, 0.3, 0.2])
_TOTAL_BUDGET: Final[float] = 1.2
_DISTURBANCES: Final[tuple[tuple[float, ...], ...]] = ((0.95, 1.0, 1.0),)
_POLICY: Final[ClosedLoopExecutionPolicy] = ClosedLoopExecutionPolicy()
_BINDING_SLACK: Final[float] = 1.0e-9
"""A cap within this distance of its limit, in any checked case, is shown as binding."""


def mpc_action_costs(b_matrix: NDArray[np.float64], target: NDArray[np.float64]) -> list[float]:
    """Return the core cost of every binary action of a one-step QAOA-MPC horizon.

    Parameters
    ----------
    b_matrix
        Actuation matrix, shape ``(n_state, n_coils)``.
    target
        Target state vector, shape ``(n_state,)``; its sign and direction count.

    Returns
    -------
    list[float]
        ``[C(u=0), C(u=1)]`` read from the diagonal of the controller's own Ising
        cost Hamiltonian; no cost is recomputed here.

    """
    hamiltonian = QAOA_MPC(b_matrix, target, horizon=1, p_layers=1).build_cost_hamiltonian()
    diagonal = np.real(np.asarray(hamiltonian.to_matrix(sparse=True).diagonal()))
    return [float(value) for value in diagonal]


def _mpc_case(case_id: str, b_matrix: list[list[float]], target: list[float]) -> dict[str, object]:
    """Describe one tracking target and the controller's proposed action."""
    costs = mpc_action_costs(np.array(b_matrix), np.array(target))
    return {
        "case_id": case_id,
        "b_matrix": b_matrix,
        "target": target,
        "action_costs": costs,
        "proposed_action": int(np.argmin(costs)),
        "applied": False,
        "objective_source": "QAOA_MPC.build_cost_hamiltonian diagonal, horizon 1",
    }


def _objective(plan: NDArray[np.float64]) -> float:
    """Squared distance of the coupling plan from the desired couplings."""
    return float(np.sum((plan - _DESIRED) ** 2))


def _constraints(
    plan: NDArray[np.float64], disturbance: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Per-edge caps, scaled by a held-out disturbance, then the total budget."""
    caps = _EDGE_CAPS if disturbance is None else _EDGE_CAPS * disturbance
    return np.concatenate([plan - caps, [float(np.sum(plan)) - _TOTAL_BUDGET]])


@dataclass(frozen=True)
class _FixedPlanOptimiser:
    """Optimiser port returning one declared plan; the session verifies it independently."""

    plan: tuple[float, ...]
    converged: bool = True
    evaluations: int = 1

    def optimize(self, start: NDArray[np.float64]) -> OptimiserOutcome:
        """Return the declared plan whatever the starting point."""
        return OptimiserOutcome(np.array(self.plan), self.converged, self.evaluations)


def _proposal_row(label: str, proposal: ConstrainedProposal) -> dict[str, object]:
    """Project a verified session proposal, marking which edges bind or fail."""
    nominal = proposal.constraint_values[0] if proposal.constraint_values else ()
    worst = (
        [max(row[index] for row in proposal.constraint_values) for index in range(len(nominal))]
        if nominal
        else []
    )
    return {
        "label": label,
        "status": proposal.status,
        "reason": proposal.reason,
        "decision": {
            "authorised": proposal.decision.authorised,
            "mode": proposal.decision.mode.value,
            "reason": proposal.decision.reason,
        },
        "plan": None if proposal.plan is None else list(proposal.plan),
        "objective": proposal.objective,
        "constraint_values": [list(row) for row in proposal.constraint_values],
        "max_violation": proposal.max_violation,
        "evaluations": proposal.evaluations,
        "adopted_plan": None if proposal.adopted_plan is None else list(proposal.adopted_plan),
        "binding_edges": [
            index for index in range(len(_EDGES)) if worst and worst[index] >= -_BINDING_SLACK
        ],
        "violated_edges": [
            index for index in range(len(_EDGES)) if worst and worst[index] > proposal.tolerance
        ],
        "hardware_actuation": proposal.hardware_actuation,
        "applied": False,
    }


def _attempt(
    session: ConstrainedCodesignSession, label: str, optimiser: _FixedPlanOptimiser
) -> dict[str, object]:
    """Propose through the session; a policy refusal is recorded as a veto, not raised."""
    try:
        return _proposal_row(label, session.propose(optimiser))
    except PolicyGatedAdapterError as error:
        return {
            "label": label,
            "status": "vetoed",
            "reason": str(error),
            "decision": None,
            "plan": None,
            "objective": None,
            "constraint_values": [],
            "max_violation": None,
            "evaluations": 0,
            "adopted_plan": None,
            "binding_edges": [],
            "violated_edges": [],
            "hardware_actuation": False,
            "applied": False,
        }


def build_codesign_section() -> dict[str, object]:
    """Run the constrained co-design session on a three-edge coupling graph.

    Returns
    -------
    dict[str, object]
        The graph, domain, constraint rows and, in order, an accepted, an
        infeasible and an out-of-domain proposal plus a policy veto. Only the
        accepted proposal changes the adopted plan; none is applied.

    """
    session = ConstrainedCodesignSession(
        _objective,
        _constraints,
        lower=np.zeros(3),
        upper=np.ones(3),
        policy=_POLICY,
        held_out_disturbances=[np.array(item) for item in _DISTURBANCES],
    )
    attempts = (
        ("capped plan", _FixedPlanOptimiser((0.57, 0.3, 0.2))),
        ("desired plan, edge 0-1 over its cap", _FixedPlanOptimiser(tuple(_DESIRED.tolist()))),
        ("plan outside the tested domain", _FixedPlanOptimiser((0.5, -0.1, 0.2))),
    )
    hardware = ClosedLoopExecutionPolicy(
        allow_hardware=True, live_ticket="no-live-session", backend_allowlist=("local-proxy",)
    )
    vetoed = ConstrainedCodesignSession(
        _objective, _constraints, lower=np.zeros(3), upper=np.ones(3), policy=hardware
    )
    proposals = [_attempt(session, label, port) for label, port in attempts]
    proposals.append(_attempt(vetoed, "hardware-mode request", attempts[0][1]))
    return {
        "graph": {"nodes": 3, "edges": [list(edge) for edge in _EDGES]},
        "plan_meaning": "coupling weight of each graph edge, in edge order",
        "objective": "sum over edges of (weight - desired)^2",
        "desired": _DESIRED.tolist(),
        "edge_caps": _EDGE_CAPS.tolist(),
        "total_budget": _TOTAL_BUDGET,
        "constraint_rows": [f"edge {a}-{b} cap" for a, b in _EDGES] + ["total coupling budget"],
        "domain": [list(bounds) for bounds in session.domain],
        "tolerance": 0.0,
        "disturbances": [list(item) for item in _DISTURBANCES],
        "proposals": proposals,
    }


def build_latency_section() -> list[dict[str, object]]:
    """Judge two replayed latency series against one closed-loop budget.

    Returns
    -------
    list[dict[str, object]]
        One passing and one failing verdict. The samples are declared replay
        inputs, not measurements of this host; a failing verdict refuses any
        latency or stability claim.

    """
    budget = ClosedLoopLatencyBudget(max_round_latency_s=0.010, p95_round_latency_s=0.008)
    rows: list[dict[str, object]] = []
    for case_id, samples in (
        ("within-budget replay", (0.002, 0.003, 0.004, 0.003)),
        ("over-budget replay", (0.004, 0.012, 0.009, 0.015)),
    ):
        controller = RealtimeSyncFeedbackController(
            np.array([[0.0, 0.5], [0.5, 0.0]]),
            np.array([1.0, 1.2]),
            config=RealtimeFeedbackConfig(measurement_shots=32),
        )
        report = measure_closed_loop_latency_budget(
            controller,
            len(samples),
            budget=budget,
            policy=_POLICY,
            seed=5,
            observed_round_latencies_s=samples,
        )
        payload = report.to_dict()
        rows.append(
            {
                "case_id": case_id,
                "passes": payload["passes"],
                "blockers": payload["blockers"],
                "round_latencies_s": payload["round_latencies_s"],
                "max_round_latency_s": payload["max_round_latency_s"],
                "p95_round_latency_s": payload["p95_round_latency_s"],
                "budget": payload["budget"],
                "clock_source": payload["clock_source"],
                "claim_boundary": payload["claim_boundary"],
            }
        )
    return rows


def build_trace_section() -> dict[str, object]:
    """Run the policy-gated quantum/classical partition and keep its own timestamps.

    Returns
    -------
    dict[str, object]
        Integrator sample times (sample ``s`` at ``s * dt``), the order
        parameters at those times and the source claim boundary. The trace is
        simulated; observed hardware telemetry is recorded as unavailable.

    """
    coupling = np.array([[0.0, 0.6, 0.2], [0.6, 0.0, 0.4], [0.2, 0.4, 0.0]])
    omega = np.array([1.0, 1.3, 0.8])
    result = run_cosimulation_partition_adapter(
        coupling, omega, policy=_POLICY, dt=0.3, n_steps=4, max_quantum_nodes=2, seed=3
    )
    trace = result.result
    return {
        "source": "run_cosimulation_partition_adapter (local mean-field partition)",
        "dt": 0.3,
        "n_steps": 4,
        "times": trace.times.tolist(),
        "global_order": trace.global_order.tolist(),
        "quantum_order": trace.quantum_order.tolist(),
        "classical_order": trace.classical_order.tolist(),
        "simulated": True,
        "claim_boundary": result.telemetry.claim_boundary,
        "observed_hardware_telemetry": None,
    }


def build_control_proposal_instrument() -> dict[str, object]:
    """Build the committed, source-owned control instrument data.

    Returns
    -------
    dict[str, object]
        Signed and rotated tracking targets, constrained co-design proposals,
        replayed latency verdicts and a simulated trace, all unapplied.

    """
    return {
        "schema": CONTROL_PROPOSAL_INSTRUMENT_SCHEMA,
        "artifact_id": "studio-control-proposal-instrument-20261109",
        "generated_by": (
            "python -m scpn_quantum_control.studio.control_proposal_instrument_artifact --write"
        ),
        "claim_boundary": CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY,
        "applied_status_rule": APPLIED_STATUS_RULE,
        "mpc_targets": [
            _mpc_case("signed target", [[1.0]], [-1.0]),
            _mpc_case("sign-flipped target", [[1.0]], [1.0]),
            _mpc_case("rotated target", [[1.0, 0.0], [0.0, 1.0]], [-1.0, 1.0]),
        ],
        "codesign": build_codesign_section(),
        "latency": build_latency_section(),
        "trace": build_trace_section(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed control instrument data.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_control_proposal_instrument,
        default_path=DEFAULT_PATH,
        label="control proposal instrument data",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
