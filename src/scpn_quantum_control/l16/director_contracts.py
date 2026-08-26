# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded-director L16 director contracts
"""Immutable contracts for bounded L16 indicator evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

L16_DIRECTOR_SCHEMA: Final[str] = "l16_director_product.v2"
L16_DIRECTOR_CLAIM_BOUNDARY: Final[str] = (
    "bounded exact-simulator L16 indicator and heuristic safety-routing evidence; "
    "no classical or quantum Lyapunov-exponent proof, PCS certificate, causal "
    "diagnosis, stability guarantee, autonomous actuation, provider, QPU, or "
    "production-control claim"
)


@dataclass(frozen=True, slots=True)
class L16ScenarioSpec:
    """Frozen small-system scenario for one L16 indicator evaluation."""

    scenario_id: str
    oscillators: int
    coupling_scale: float
    frequency_scale: float
    evolution_time: float

    def __post_init__(self) -> None:
        """Validate the bounded exact-simulator scenario."""
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if (
            isinstance(self.oscillators, bool)
            or not isinstance(self.oscillators, int)
            or not 1 <= self.oscillators <= 6
        ):
            raise ValueError("oscillators must be an integer in [1, 6]")
        _non_negative_finite("coupling_scale", self.coupling_scale)
        _positive_finite("frequency_scale", self.frequency_scale)
        _non_negative_finite("evolution_time", self.evolution_time)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready scenario mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class L16IndicatorCertificate:
    """Validated raw indicators and their conservative safety interpretation."""

    scenario: L16ScenarioSpec
    loschmidt_echo: float
    energy_variance: float
    fidelity_susceptibility: float
    order_parameter: float
    heuristic_score: float
    heuristic_action: str
    codesign_action: str
    informative_indicators: tuple[str, ...]
    policy_authorised: bool
    deterministic_replay: bool
    route_id: str
    hardware_execution: bool = False
    claim_boundary: str = L16_DIRECTOR_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate indicators, action mapping, and simulator-only provenance."""
        _bounded_with_roundoff("loschmidt_echo", self.loschmidt_echo)
        _non_negative_finite("energy_variance", self.energy_variance)
        _non_negative_finite("fidelity_susceptibility", self.fidelity_susceptibility)
        _bounded_with_roundoff("order_parameter", self.order_parameter)
        _bounded_with_roundoff("heuristic_score", self.heuristic_score)
        expected_actions = {
            "continue": "allow",
            "adjust": "hold",
            "halt": "abort",
        }
        if self.heuristic_action not in expected_actions:
            raise ValueError("heuristic_action must be continue, adjust, or halt")
        if self.codesign_action != expected_actions[self.heuristic_action]:
            raise ValueError("codesign_action disagrees with the conservative L16 mapping")
        allowed_indicators = {
            "loschmidt_echo",
            "energy_variance",
            "fidelity_susceptibility",
            "order_parameter",
        }
        if len(set(self.informative_indicators)) != len(self.informative_indicators):
            raise ValueError("informative_indicators must be unique")
        if not set(self.informative_indicators) <= allowed_indicators:
            raise ValueError("informative_indicators contains an unknown indicator")
        if not self.route_id.strip():
            raise ValueError("route_id must be non-empty")
        if self.hardware_execution:
            raise ValueError("bounded L16 director certificates must remain simulator-only")
        if self.claim_boundary != L16_DIRECTOR_CLAIM_BOUNDARY:
            raise ValueError("bounded L16 director certificate claim boundary is fixed")

    @property
    def passed(self) -> bool:
        """Return whether bounded execution and exact replay both passed."""
        return self.policy_authorised and self.deterministic_replay and not self.hardware_execution

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate mapping."""
        return {
            "scenario": self.scenario.to_dict(),
            "loschmidt_echo": self.loschmidt_echo,
            "energy_variance": self.energy_variance,
            "fidelity_susceptibility": self.fidelity_susceptibility,
            "order_parameter": self.order_parameter,
            "heuristic_score": self.heuristic_score,
            "heuristic_action": self.heuristic_action,
            "codesign_action": self.codesign_action,
            "informative_indicators": list(self.informative_indicators),
            "policy_authorised": self.policy_authorised,
            "deterministic_replay": self.deterministic_replay,
            "route_id": self.route_id,
            "hardware_execution": self.hardware_execution,
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class L16RouteEvidence:
    """One governed route status retained in bounded L16 director evidence."""

    route_id: str
    closure_status: str
    closure_reason: str

    def __post_init__(self) -> None:
        """Require complete supported or permanent-boundary route evidence."""
        if not self.route_id.strip():
            raise ValueError("route_id must be non-empty")
        if self.closure_status not in {"supported", "permanent_boundary"}:
            raise ValueError("L16 route status must be supported or permanent_boundary")
        if self.closure_status == "supported" and self.closure_reason:
            raise ValueError("supported L16 routes cannot carry a closure_reason")
        if self.closure_status == "permanent_boundary" and not self.closure_reason.strip():
            raise ValueError("permanent L16 routes require a closure_reason")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready route mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class L16DirectorEvidence:
    """Complete functional L16 director evidence without stability promotion."""

    certificates: tuple[L16IndicatorCertificate, ...]
    routes: tuple[L16RouteEvidence, ...]
    promotion_blockers: tuple[str, ...]
    schema: str = L16_DIRECTOR_SCHEMA
    claim_boundary: str = L16_DIRECTOR_CLAIM_BOUNDARY
    provider_execution: bool = False
    hardware_execution: bool = False
    promotion_ready: bool = False

    def __post_init__(self) -> None:
        """Require the complete frozen suite and permanent promotion boundary."""
        scenario_ids = tuple(item.scenario.scenario_id for item in self.certificates)
        if scenario_ids != (
            "paper27_baseline",
            "susceptibility_probe",
            "weak_coupling_probe",
        ):
            raise ValueError("certificates must retain the frozen L16 scenario order")
        route_ids = tuple(route.route_id for route in self.routes)
        if route_ids != (
            "adapter:l16.local_indicator",
            "adapter:l16.autonomous_hardware_control",
        ):
            raise ValueError("routes must retain ordered local and autonomous-hardware L16 cells")
        if not self.promotion_blockers or any(
            not item.strip() for item in self.promotion_blockers
        ):
            raise ValueError("promotion_blockers must be complete and non-empty")
        if (
            self.schema != L16_DIRECTOR_SCHEMA
            or self.claim_boundary != L16_DIRECTOR_CLAIM_BOUNDARY
        ):
            raise ValueError("L16 director evidence schema and claim boundary are fixed")
        if self.provider_execution or self.hardware_execution or self.promotion_ready:
            raise ValueError("L16 director evidence cannot promote or record external execution")

    @property
    def functional_passed(self) -> bool:
        """Return whether every bounded certificate and route gate passed."""
        routes_pass = all(
            route.closure_status
            == (
                "supported"
                if route.route_id == "adapter:l16.local_indicator"
                else "permanent_boundary"
            )
            for route in self.routes
        )
        return all(item.passed for item in self.certificates) and routes_pass

    @property
    def action_diversity(self) -> bool:
        """Return whether the frozen real scenarios produced multiple actions."""
        return len({item.heuristic_action for item in self.certificates}) > 1

    def to_payload(self) -> dict[str, object]:
        """Return the digestable JSON payload without its integrity digest."""
        return {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "functional_passed": self.functional_passed,
            "promotion_ready": self.promotion_ready,
            "promotion_blockers": list(self.promotion_blockers),
            "action_diversity": self.action_diversity,
            "provider_execution": self.provider_execution,
            "hardware_execution": self.hardware_execution,
            "certificates": [item.to_dict() for item in self.certificates],
            "routes": [item.to_dict() for item in self.routes],
        }


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_finite(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _non_negative_finite(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _bounded_with_roundoff(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if not -1e-9 <= scalar <= 1.0 + 1e-9:
        raise ValueError(f"{name} must lie in [0, 1] within numerical roundoff")
    return scalar


__all__ = [
    "L16_DIRECTOR_CLAIM_BOUNDARY",
    "L16_DIRECTOR_SCHEMA",
    "L16DirectorEvidence",
    "L16IndicatorCertificate",
    "L16RouteEvidence",
    "L16ScenarioSpec",
]
