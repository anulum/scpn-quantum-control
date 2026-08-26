# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded-director bounded L16 director product
"""Policy-gated L16 indicators and conservative co-design safety routing."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ..codesign.contracts import ObserverInputs
from ..control.closed_loop_analysis import (
    ClosedLoopExecutionPolicy,
    ExecutionMode,
    evaluate_closed_loop_policy,
)
from ..governed_route_matrix import get_governed_route
from .director_contracts import (
    L16DirectorEvidence,
    L16IndicatorCertificate,
    L16RouteEvidence,
    L16ScenarioSpec,
)
from .quantum_director import L16Result, compute_l16_lyapunov

_INDICATOR_TOLERANCE = 1e-8
_ACTION_MAP = {"continue": "allow", "adjust": "hold", "halt": "abort"}


class L16DirectorPolicyError(RuntimeError):
    """Raised when the closed-loop execution policy refuses bounded L16 evaluation."""


def frozen_l16_scenarios() -> tuple[L16ScenarioSpec, ...]:
    """Return the three ordered small-system L16 director scenarios."""
    return (
        L16ScenarioSpec("paper27_baseline", 4, 1.0, 1.0, 0.5),
        L16ScenarioSpec("susceptibility_probe", 3, 0.3, 0.1, 0.5),
        L16ScenarioSpec("weak_coupling_probe", 3, 0.01, 1.0, 0.5),
    )


def observer_inputs_from_l16(
    action: str,
    *,
    reason: str = "",
) -> ObserverInputs:
    """Map a legacy L16 action into the conservative observer interlock contract."""
    if action not in _ACTION_MAP:
        raise ValueError("L16 action must be continue, adjust, or halt")
    mapped_reason = reason.strip() or (
        "L16 heuristic requested continue"
        if action == "continue"
        else f"L16 heuristic requested {action}; conservative controller interlock"
    )
    return ObserverInputs(l16_action=action, l16_reason=mapped_reason)


def run_l16_indicator_scenario(
    scenario: L16ScenarioSpec,
    *,
    policy: ClosedLoopExecutionPolicy,
    backend: str | None = None,
) -> L16IndicatorCertificate:
    """Execute and replay one frozen scenario under the shared execution policy."""
    decision = evaluate_closed_loop_policy(policy, backend=backend, requested_rounds=1)
    if not decision.authorised:
        raise L16DirectorPolicyError(f"execution policy refused L16 evaluation: {decision.reason}")
    if decision.mode is not ExecutionMode.SIMULATION:
        raise L16DirectorPolicyError(
            "bounded L16 director evidence is local-simulator only; hardware mode is refused"
        )
    coupling, omega = _scenario_arrays(scenario)
    first = compute_l16_lyapunov(coupling, omega, t=scenario.evolution_time)
    replay = compute_l16_lyapunov(coupling, omega, t=scenario.evolution_time)
    first_payload = asdict(first)
    replay_payload = asdict(replay)
    route = get_governed_route("adapter:l16.local_indicator")
    return L16IndicatorCertificate(
        scenario=scenario,
        loschmidt_echo=float(first.loschmidt_echo),
        energy_variance=float(first.energy_variance),
        fidelity_susceptibility=float(first.fidelity_susceptibility),
        order_parameter=float(first.order_parameter),
        heuristic_score=float(first.stability_score),
        heuristic_action=first.action,
        codesign_action=_ACTION_MAP[first.action],
        informative_indicators=informative_l16_indicators(first),
        policy_authorised=decision.authorised,
        deterministic_replay=first_payload == replay_payload,
        route_id=route.route_id,
    )


def run_l16_director_suite(
    *,
    policy: ClosedLoopExecutionPolicy | None = None,
) -> L16DirectorEvidence:
    """Run complete bounded evidence and retain permanent promotion blockers."""
    selected_policy = policy or ClosedLoopExecutionPolicy(round_budget=3)
    certificates = tuple(
        run_l16_indicator_scenario(scenario, policy=selected_policy)
        for scenario in frozen_l16_scenarios()
    )
    route_rows = tuple(
        _route_evidence(route_id)
        for route_id in (
            "adapter:l16.local_indicator",
            "adapter:l16.autonomous_hardware_control",
        )
    )
    blockers = l16_promotion_blockers(certificates)
    return L16DirectorEvidence(
        certificates=certificates,
        routes=route_rows,
        promotion_blockers=blockers,
    )


def _scenario_arrays(
    scenario: L16ScenarioSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coupling = np.asarray(
        build_knm_paper27(L=scenario.oscillators) * scenario.coupling_scale,
        dtype=np.float64,
    )
    omega = np.asarray(
        OMEGA_N_16[: scenario.oscillators] * scenario.frequency_scale,
        dtype=np.float64,
    )
    return coupling, omega


def informative_l16_indicators(result: L16Result) -> tuple[str, ...]:
    """Name raw indicators that differ nontrivially from their invariant baseline."""
    indicators: list[str] = []
    if abs(1.0 - result.loschmidt_echo) > _INDICATOR_TOLERANCE:
        indicators.append("loschmidt_echo")
    if result.energy_variance > _INDICATOR_TOLERANCE:
        indicators.append("energy_variance")
    if result.fidelity_susceptibility > _INDICATOR_TOLERANCE:
        indicators.append("fidelity_susceptibility")
    if abs(1.0 - result.order_parameter) > _INDICATOR_TOLERANCE:
        indicators.append("order_parameter")
    return tuple(indicators)


def l16_promotion_blockers(
    certificates: tuple[L16IndicatorCertificate, ...],
) -> tuple[str, ...]:
    """Return fixed claim boundaries plus findings from the supplied real certificates."""
    blockers = [
        "weighted composite is a heuristic policy, not a Lyapunov or PCS certificate",
        "no provider, QPU, plant, or realtime-hardware execution",
        "no closed-loop stability theorem or causal diagnosis",
    ]
    if len({item.heuristic_action for item in certificates}) == 1:
        blockers.append("frozen real scenarios did not establish action diversity")
    if any(len(item.informative_indicators) < 2 for item in certificates):
        blockers.append("at least one scenario has fewer than two nontrivial raw indicators")
    return tuple(blockers)


def _route_evidence(route_id: str) -> L16RouteEvidence:
    route = get_governed_route(route_id)
    return L16RouteEvidence(
        route_id=route.route_id,
        closure_status=route.closure_status,
        closure_reason=route.closure_reason,
    )


__all__ = [
    "L16DirectorPolicyError",
    "frozen_l16_scenarios",
    "informative_l16_indicators",
    "l16_promotion_blockers",
    "observer_inputs_from_l16",
    "run_l16_director_suite",
    "run_l16_indicator_scenario",
]
