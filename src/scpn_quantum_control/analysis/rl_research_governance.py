# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — RL research governance
"""Fail-closed governance for witness-search and pulse-optimisation research.

RL-governance keeps reinforcement-learning-adjacent routes in a research extra.  The
existing witness discovery is a seeded static candidate search, not a Gym
environment or a trained production policy.  Its dense composite witness score
is therefore named explicitly and evaluated through deterministic replay over
multiple seeds. The pulse optimiser remains unimplemented and blocked behind
the separately governed pulse-execution boundary.

Nothing in this module enables provider submission, hardware execution,
production control, policy deployment, or a scientific performance claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from operator import index
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .witness_discovery import (
    WitnessDiscoverySpec,
    discover_kuramoto_witnesses,
)

FloatArray: TypeAlias = NDArray[np.float64]

RL_RESEARCH_GOVERNANCE_SCHEMA = "scpn.rl-research-governance.v1"
RL_RESEARCH_CLAIM_BOUNDARY = (
    "local seeded research replay only; no product policy, autonomous control, "
    "provider submission, QPU or pulse execution, hardware validity, advantage, "
    "optimal-policy, realtime, clinical, or publication claim"
)
RL_DENSE_REWARD_CONTRACT = "witness_score_dense_composite_v1"
RL_ENVIRONMENT_API = "not_applicable_static_candidate_search"
DEFAULT_RL_RESEARCH_SEEDS = (104729, 130363, 155921)


class RLResearchLane(str, Enum):
    """Governed RL-adjacent route."""

    WITNESS_DISCOVERY = "witness_discovery"
    PULSE_OPTIMISATION = "pulse_optimisation"


class RLResearchGovernanceError(RuntimeError):
    """Raised when an RL-adjacent route lacks its research gates."""


@dataclass(frozen=True, slots=True)
class RLResearchPolicy:
    """Explicit research-only enablement and reproducibility budget.

    Parameters
    ----------
    enabled
        Opt-in research flag. The default is ``False``.
    preregistration_id
        Stable identifier for the protocol fixed before the run. An enabled
        route without this value is refused.
    seeds
        At least three distinct non-negative seeds used for the evaluation
        suite.
    max_episodes
        Maximum witness-search iterations per seed. The legacy API calls these
        iterations ``episodes``; no Gym episode contract is implied.
    max_evaluations_per_seed
        Upper bound on candidate evaluations for each seed.
    deterministic_evaluation
        Must remain ``True``.
    evaluation_exploration_noise
        Must remain exactly zero for deterministic evaluation.
    allow_hardware
        Must remain ``False``.
    allow_production_control
        Must remain ``False``.
    reward_contract
        Frozen dense composite score identifier. The score combines final
        order, correlations, Fiedler value, witness margin, and novelty; it can
        be gamed if reported without its components and is not a sparse task
        reward or operational utility.

    Notes
    -----
    This policy does not design a Gym environment. Consequently the Gym
    ``step`` tuple is not applicable; a future environment must separately
    implement ``obs, reward, terminated, truncated, info``.

    """

    enabled: bool = False
    preregistration_id: str = ""
    seeds: tuple[int, ...] = DEFAULT_RL_RESEARCH_SEEDS
    max_episodes: int = 5
    max_evaluations_per_seed: int = 32
    deterministic_evaluation: bool = True
    evaluation_exploration_noise: float = 0.0
    allow_hardware: bool = False
    allow_production_control: bool = False
    reward_contract: str = RL_DENSE_REWARD_CONTRACT

    def __post_init__(self) -> None:
        """Validate immutable safety, seed, and evaluation invariants."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean research flag")
        if not isinstance(self.preregistration_id, str):
            raise ValueError("preregistration_id must be a string")
        object.__setattr__(self, "preregistration_id", self.preregistration_id.strip())
        normalized_seeds = tuple(
            _positive_index(seed, "seed", allow_zero=True) for seed in self.seeds
        )
        if len(normalized_seeds) < 3:
            raise ValueError("seeds must contain at least three evaluation seeds")
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be distinct")
        object.__setattr__(self, "seeds", normalized_seeds)
        object.__setattr__(
            self, "max_episodes", _positive_index(self.max_episodes, "max_episodes")
        )
        object.__setattr__(
            self,
            "max_evaluations_per_seed",
            _positive_index(self.max_evaluations_per_seed, "max_evaluations_per_seed"),
        )
        if isinstance(self.evaluation_exploration_noise, bool):
            raise ValueError("evaluation_exploration_noise must be exactly 0.0")
        noise = float(self.evaluation_exploration_noise)
        if not np.isfinite(noise) or noise != 0.0:
            raise ValueError("evaluation_exploration_noise must be exactly 0.0")
        object.__setattr__(self, "evaluation_exploration_noise", noise)
        if (
            not isinstance(self.deterministic_evaluation, bool)
            or not self.deterministic_evaluation
        ):
            raise ValueError("deterministic_evaluation must remain enabled")
        if not isinstance(self.allow_hardware, bool) or self.allow_hardware:
            raise ValueError("allow_hardware must remain False")
        if not isinstance(self.allow_production_control, bool) or self.allow_production_control:
            raise ValueError("allow_production_control must remain False")
        if self.reward_contract != RL_DENSE_REWARD_CONTRACT:
            raise ValueError(f"reward_contract must be {RL_DENSE_REWARD_CONTRACT!r}")

    @property
    def policy_id(self) -> str:
        """Return a stable digest-bound identifier for this policy."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"rl-policy-{hashlib.sha256(canonical).hexdigest()[:16]}"

    def as_dict(self) -> dict[str, Any]:
        """Return the policy as deterministic JSON-ready primitives."""
        return {
            "enabled": self.enabled,
            "preregistration_id": self.preregistration_id,
            "seeds": list(self.seeds),
            "max_episodes": self.max_episodes,
            "max_evaluations_per_seed": self.max_evaluations_per_seed,
            "deterministic_evaluation": self.deterministic_evaluation,
            "evaluation_exploration_noise": self.evaluation_exploration_noise,
            "allow_hardware": self.allow_hardware,
            "allow_production_control": self.allow_production_control,
            "reward_contract": self.reward_contract,
            "environment_api": RL_ENVIRONMENT_API,
        }


@dataclass(frozen=True, slots=True)
class RLResearchDecision:
    """Fail-closed admission result for one route and optional search spec."""

    lane: RLResearchLane
    allowed: bool
    blockers: tuple[str, ...]
    estimated_evaluations_per_seed: int
    policy_id: str
    reason: str

    def __post_init__(self) -> None:
        """Require exact consistency between blockers and admission."""
        if self.allowed is bool(self.blockers):
            raise ValueError("allowed must equal the absence of blockers")
        if self.estimated_evaluations_per_seed < 0:
            raise ValueError("estimated_evaluations_per_seed must be non-negative")
        if not self.policy_id.strip() or not self.reason.strip():
            raise ValueError("policy_id and reason must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready admission evidence."""
        return {
            "lane": self.lane.value,
            "allowed": self.allowed,
            "blockers": list(self.blockers),
            "estimated_evaluations_per_seed": self.estimated_evaluations_per_seed,
            "policy_id": self.policy_id,
            "reason": self.reason,
            "claim_boundary": RL_RESEARCH_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RLSeedEvaluation:
    """Deterministic replay evidence for one preregistered seed."""

    seed: int
    evaluation_count: int
    best_score: float
    best_candidate: tuple[tuple[str, float], ...]
    replay_identical: bool

    def __post_init__(self) -> None:
        """Reject malformed or invent-green seed evidence."""
        _positive_index(self.seed, "seed", allow_zero=True)
        _positive_index(self.evaluation_count, "evaluation_count")
        if not np.isfinite(self.best_score):
            raise ValueError("best_score must be finite")
        if not self.best_candidate or any(
            not name.strip() or not np.isfinite(value) for name, value in self.best_candidate
        ):
            raise ValueError("best_candidate must contain finite named values")
        names = [name for name, _value in self.best_candidate]
        if len(names) != len(set(names)):
            raise ValueError("best_candidate names must be unique")
        if not isinstance(self.replay_identical, bool) or not self.replay_identical:
            raise ValueError("seed evaluation replay must be byte-identical")

    def as_dict(self) -> dict[str, Any]:
        """Return the seed result as JSON-ready primitives."""
        return {
            "seed": self.seed,
            "evaluation_count": self.evaluation_count,
            "best_score": self.best_score,
            "best_candidate": dict(self.best_candidate),
            "replay_identical": self.replay_identical,
        }


@dataclass(frozen=True, slots=True)
class RLSeedSuiteReport:
    """Multi-seed deterministic research evidence with a content digest."""

    schema: str
    policy_id: str
    preregistration_id: str
    decision: RLResearchDecision
    seed_results: tuple[RLSeedEvaluation, ...]
    content_digest: str

    def __post_init__(self) -> None:
        """Validate report identity, seed uniqueness, and digest shape."""
        if self.schema != RL_RESEARCH_GOVERNANCE_SCHEMA:
            raise ValueError("schema must match RL_RESEARCH_GOVERNANCE_SCHEMA")
        if self.policy_id != self.decision.policy_id:
            raise ValueError("policy_id must match the admission decision")
        if not self.preregistration_id.strip():
            raise ValueError("preregistration_id must be non-empty")
        seeds = [result.seed for result in self.seed_results]
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed_results must have unique seeds")
        if len(self.content_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_digest
        ):
            raise ValueError("content_digest must be a lowercase SHA-256 hex digest")

    @property
    def passed(self) -> bool:
        """Return whether admission and every deterministic seed replay passed."""
        return (
            self.decision.allowed
            and bool(self.seed_results)
            and all(result.replay_identical for result in self.seed_results)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the complete report as deterministic JSON-ready data."""
        scores = [result.best_score for result in self.seed_results]
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "preregistration_id": self.preregistration_id,
            "passed": self.passed,
            "decision": self.decision.as_dict(),
            "seed_count": len(self.seed_results),
            "seed_results": [result.as_dict() for result in self.seed_results],
            "best_score_mean": float(np.mean(scores)) if scores else None,
            "best_score_population_std": float(np.std(scores)) if scores else None,
            "reward_contract": RL_DENSE_REWARD_CONTRACT,
            "environment_api": RL_ENVIRONMENT_API,
            "evaluation_exploration_noise": 0.0,
            "registry_grants_production_control": False,
            "registry_grants_hardware_execution": False,
            "claim_boundary": RL_RESEARCH_CLAIM_BOUNDARY,
            "content_digest": self.content_digest,
        }


def _positive_index(value: Any, name: str, *, allow_zero: bool = False) -> int:
    """Normalize an integer-like value with a strict positive/zero policy."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < 0 or (normalized == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(normalized)


def estimate_witness_evaluation_budget(spec: WitnessDiscoverySpec) -> int:
    """Return a conservative candidate-evaluation upper bound for one seed.

    The initial Latin-hypercube candidates consume ``n_initial`` evaluations.
    Each iteration proposes at most ``max(batch_size - 1, 1)`` Bayesian rows
    plus one seeded bandit row.
    """
    per_iteration = max(spec.batch_size - 1, 1) + 1
    return spec.n_initial + spec.n_iterations * per_iteration


def assess_rl_research(
    policy: RLResearchPolicy | None,
    lane: RLResearchLane,
    *,
    spec: WitnessDiscoverySpec | None = None,
) -> RLResearchDecision:
    """Return a fail-closed admission decision for an RL-adjacent route.

    Parameters
    ----------
    policy
        Explicit policy. ``None`` resolves to the disabled default.
    lane
        Witness discovery or pulse optimisation.
    spec
        Search specification used for budget checks. It is ignored for the
        pulse route, which is blocked in the current implementation.

    """
    if not isinstance(lane, RLResearchLane):
        raise ValueError("lane must be an RLResearchLane")
    current = policy if policy is not None else RLResearchPolicy()
    blockers: list[str] = []
    estimated = 0
    if not current.enabled:
        blockers.append("research_extra_disabled")
    if not current.preregistration_id:
        blockers.append("preregistration_id_missing")
    if lane is RLResearchLane.PULSE_OPTIMISATION:
        blockers.extend(("rl_pulse_optimizer_unimplemented", "pulse_boundary_open"))
    else:
        current_spec = spec if spec is not None else WitnessDiscoverySpec()
        estimated = estimate_witness_evaluation_budget(current_spec)
        if current_spec.n_iterations > current.max_episodes:
            blockers.append("episode_budget_exceeded")
        if estimated > current.max_evaluations_per_seed:
            blockers.append("evaluation_budget_exceeded")
    allowed = not blockers
    reason = (
        "local preregistered witness-discovery research admitted"
        if allowed
        else "RL research refused: " + ", ".join(blockers)
    )
    return RLResearchDecision(
        lane=lane,
        allowed=allowed,
        blockers=tuple(blockers),
        estimated_evaluations_per_seed=estimated,
        policy_id=current.policy_id,
        reason=reason,
    )


def assert_rl_research_allowed(
    policy: RLResearchPolicy | None,
    lane: RLResearchLane,
    *,
    spec: WitnessDiscoverySpec | None = None,
) -> RLResearchDecision:
    """Return an allowed decision or raise :class:`RLResearchGovernanceError`."""
    decision = assess_rl_research(policy, lane, spec=spec)
    if not decision.allowed:
        raise RLResearchGovernanceError(decision.reason)
    return decision


def build_witness_seed_suite(
    policy: RLResearchPolicy,
    template: WitnessDiscoverySpec,
) -> tuple[WitnessDiscoverySpec, ...]:
    """Build one budget-checked witness specification per policy seed."""
    assert_rl_research_allowed(policy, RLResearchLane.WITNESS_DISCOVERY, spec=template)
    return tuple(replace(template, seed=seed) for seed in policy.seeds)


def run_governed_witness_seed_suite(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    policy: RLResearchPolicy,
    template: WitnessDiscoverySpec,
    theta0: FloatArray | None = None,
    prefer_rust: bool = False,
) -> RLSeedSuiteReport:
    """Run and replay each preregistered seed without hardware or deployment.

    Each full seeded search is executed twice. Byte-identical serialized traces
    are required before a seed result can be constructed. This is reproducible
    software evidence across multiple seeds, not statistical significance for
    an operational policy or a claim that the dense score is ungameable.
    """
    specs = build_witness_seed_suite(policy, template)
    decision = assert_rl_research_allowed(policy, RLResearchLane.WITNESS_DISCOVERY, spec=template)
    rows: list[RLSeedEvaluation] = []
    for seeded_spec in specs:
        first = discover_kuramoto_witnesses(
            K_nm,
            omega,
            theta0=theta0,
            spec=seeded_spec,
            prefer_rust=prefer_rust,
        )
        replay = discover_kuramoto_witnesses(
            K_nm,
            omega,
            theta0=theta0,
            spec=seeded_spec,
            prefer_rust=prefer_rust,
        )
        candidate = tuple(sorted(first.best.candidate.to_metadata().items()))
        rows.append(
            RLSeedEvaluation(
                seed=seeded_spec.seed,
                evaluation_count=len(first.evaluations),
                best_score=float(first.best.score),
                best_candidate=candidate,
                replay_identical=first.to_json() == replay.to_json(),
            )
        )
    payload = {
        "schema": RL_RESEARCH_GOVERNANCE_SCHEMA,
        "policy_id": policy.policy_id,
        "preregistration_id": policy.preregistration_id,
        "decision": decision.as_dict(),
        "seed_results": [row.as_dict() for row in rows],
        "reward_contract": RL_DENSE_REWARD_CONTRACT,
        "environment_api": RL_ENVIRONMENT_API,
        "claim_boundary": RL_RESEARCH_CLAIM_BOUNDARY,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return RLSeedSuiteReport(
        schema=RL_RESEARCH_GOVERNANCE_SCHEMA,
        policy_id=policy.policy_id,
        preregistration_id=policy.preregistration_id,
        decision=decision,
        seed_results=tuple(rows),
        content_digest=hashlib.sha256(canonical).hexdigest(),
    )


def build_rl_research_evidence_report() -> RLSeedSuiteReport:
    """Run the frozen credential-free three-seed governance fixture."""
    policy = RLResearchPolicy(
        enabled=True,
        preregistration_id="RL-GOVERNANCE-LOCAL-WITNESS-REPLAY-v1",
        seeds=DEFAULT_RL_RESEARCH_SEEDS,
        max_episodes=1,
        max_evaluations_per_seed=5,
    )
    template = WitnessDiscoverySpec(
        n_steps=8,
        n_initial=3,
        n_iterations=1,
        batch_size=2,
        pool_size=8,
        seed=0,
        metadata={"protocol": policy.preregistration_id, "purpose": "local_replay"},
    )
    K_nm = np.array(
        [[0.0, 0.8, 0.3], [0.8, 0.0, 0.6], [0.3, 0.6, 0.0]],
        dtype=np.float64,
    )
    omega = np.array([-0.15, 0.0, 0.15], dtype=np.float64)
    theta0 = np.array([-0.2, 0.1, 0.4], dtype=np.float64)
    return run_governed_witness_seed_suite(
        K_nm,
        omega,
        policy=policy,
        template=template,
        theta0=theta0,
        prefer_rust=False,
    )


def render_rl_research_evidence_markdown(
    report: RLSeedSuiteReport | None = None,
) -> str:
    """Render deterministic human-readable RL governance evidence."""
    current = report if report is not None else build_rl_research_evidence_report()
    lines = [
        "# RL research-governance evidence",
        "",
        f"- Schema: `{current.schema}`",
        f"- Policy: `{current.policy_id}`",
        f"- Preregistration: `{current.preregistration_id}`",
        f"- Content digest: `{current.content_digest}`",
        f"- Result: **{'PASS' if current.passed else 'FAIL'}**",
        f"- Environment API: `{RL_ENVIRONMENT_API}` (no Gym environment is introduced)",
        f"- Reward: `{RL_DENSE_REWARD_CONTRACT}` (dense composite; component gaming remains a research risk)",
        f"- Boundary: {RL_RESEARCH_CLAIM_BOUNDARY}.",
        "",
        "| Seed | Evaluations | Best composite score | Replay identical | Best candidate |",
        "|---:|---:|---:|---|---|",
    ]
    for row in current.seed_results:
        candidate = ", ".join(f"{name}={value:.12g}" for name, value in row.best_candidate)
        lines.append(
            f"| {row.seed} | {row.evaluation_count} | {row.best_score:.12g} | "
            f"{str(row.replay_identical).lower()} | {candidate} |"
        )
    lines.extend(("", "The score is fixture-local and carries no policy-performance claim.", ""))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_RL_RESEARCH_SEEDS",
    "RL_DENSE_REWARD_CONTRACT",
    "RL_ENVIRONMENT_API",
    "RL_RESEARCH_CLAIM_BOUNDARY",
    "RL_RESEARCH_GOVERNANCE_SCHEMA",
    "RLResearchDecision",
    "RLResearchGovernanceError",
    "RLResearchLane",
    "RLResearchPolicy",
    "RLSeedEvaluation",
    "RLSeedSuiteReport",
    "assert_rl_research_allowed",
    "assess_rl_research",
    "build_rl_research_evidence_report",
    "build_witness_seed_suite",
    "estimate_witness_evaluation_budget",
    "render_rl_research_evidence_markdown",
    "run_governed_witness_seed_suite",
]
