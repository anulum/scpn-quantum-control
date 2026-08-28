# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — RL-governance RL research-governance tests
"""Tests for disabled-by-default, preregistered RL-adjacent research."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.rl_research_governance as governance
from scpn_quantum_control.analysis import (
    DEFAULT_RL_RESEARCH_SEEDS,
    RL_DENSE_REWARD_CONTRACT,
    RL_ENVIRONMENT_API,
    RL_RESEARCH_CLAIM_BOUNDARY,
    RL_RESEARCH_GOVERNANCE_SCHEMA,
    RLDiscoveryAgent,
    RLPulseOptimizer,
    RLResearchDecision,
    RLResearchGovernanceError,
    RLResearchLane,
    RLResearchPolicy,
    RLSeedEvaluation,
    RLSeedSuiteReport,
    WitnessDiscoverySpec,
    assert_rl_research_allowed,
    assess_rl_research,
    build_rl_research_evidence_report,
    build_witness_seed_suite,
    estimate_witness_evaluation_budget,
    render_rl_research_evidence_markdown,
    run_governed_witness_seed_suite,
)

ROOT = Path(__file__).resolve().parents[1]


def _policy(**changes: object) -> RLResearchPolicy:
    """Return a small enabled policy for local tests."""
    values: dict[str, object] = {
        "enabled": True,
        "preregistration_id": "test-prereg-v1",
        "seeds": (3, 5, 7),
        "max_episodes": 1,
        "max_evaluations_per_seed": 5,
    }
    values.update(changes)
    return RLResearchPolicy(**values)  # type: ignore[arg-type]


def _spec(**changes: object) -> WitnessDiscoverySpec:
    """Return a five-evaluation deterministic search specification."""
    values: dict[str, object] = {
        "n_steps": 5,
        "n_initial": 3,
        "n_iterations": 1,
        "batch_size": 2,
        "pool_size": 8,
        "seed": 0,
    }
    values.update(changes)
    return WitnessDiscoverySpec(**values)  # type: ignore[arg-type]


def _problem() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a tiny local Kuramoto fixture."""
    K_nm = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.6], [0.3, 0.6, 0.0]], dtype=float)
    return K_nm, np.array([-0.15, 0.0, 0.15]), np.array([-0.2, 0.1, 0.4])


def _load_runner() -> ModuleType:
    """Load the evidence runner without changing package paths."""
    path = ROOT / "scripts/run_rl_research_governance_evidence.py"
    spec = importlib.util.spec_from_file_location("rl_governance_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_policy_is_disabled_and_non_operational() -> None:
    """No caller gets research, hardware, or control authority by default."""
    policy = RLResearchPolicy()
    payload = policy.as_dict()
    decision = assess_rl_research(None, RLResearchLane.WITNESS_DISCOVERY)

    assert not policy.enabled
    assert policy.seeds == DEFAULT_RL_RESEARCH_SEEDS
    assert payload["reward_contract"] == RL_DENSE_REWARD_CONTRACT
    assert payload["environment_api"] == RL_ENVIRONMENT_API
    assert payload["allow_hardware"] is False
    assert payload["allow_production_control"] is False
    assert policy.policy_id.startswith("rl-policy-")
    assert not decision.allowed
    assert decision.blockers == (
        "research_extra_disabled",
        "preregistration_id_missing",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"enabled": 1}, "enabled must be a boolean"),
        ({"preregistration_id": 1}, "preregistration_id must be a string"),
        ({"seeds": (1, 2)}, "at least three"),
        ({"seeds": (1, 1, 2)}, "distinct"),
        ({"seeds": (True, 2, 3)}, "seed must be an integer"),
        ({"seeds": (-1, 2, 3)}, "seed must be non-negative"),
        ({"max_episodes": 0}, "max_episodes must be positive"),
        ({"max_evaluations_per_seed": 1.5}, "must be an integer"),
        ({"evaluation_exploration_noise": float("nan")}, "exactly 0.0"),
        ({"evaluation_exploration_noise": 0.1}, "exactly 0.0"),
        ({"evaluation_exploration_noise": False}, "exactly 0.0"),
        ({"deterministic_evaluation": False}, "must remain enabled"),
        ({"deterministic_evaluation": 1}, "must remain enabled"),
        ({"allow_hardware": True}, "must remain False"),
        ({"allow_hardware": 1}, "must remain False"),
        ({"allow_production_control": True}, "must remain False"),
        ({"allow_production_control": 1}, "must remain False"),
        ({"reward_contract": "custom"}, "reward_contract must be"),
    ],
)
def test_policy_rejects_unsafe_or_non_reproducible_configuration(
    changes: dict[str, object], message: str
) -> None:
    """Safety, integer, seed, noise, and reward invariants fail closed."""
    with pytest.raises(ValueError, match=message):
        _policy(**changes)


def test_preregistration_is_normalized_but_required_for_admission() -> None:
    """Whitespace is stripped and an empty enabled identifier remains blocked."""
    normalized = _policy(preregistration_id="  stable-id  ")
    missing = _policy(preregistration_id="  ")

    assert normalized.preregistration_id == "stable-id"
    decision = assess_rl_research(missing, RLResearchLane.WITNESS_DISCOVERY, spec=_spec())
    assert decision.blockers == ("preregistration_id_missing",)
    with pytest.raises(RLResearchGovernanceError, match="preregistration_id_missing"):
        assert_rl_research_allowed(missing, RLResearchLane.WITNESS_DISCOVERY, spec=_spec())


def test_evaluation_budget_formula_handles_unit_and_larger_batches() -> None:
    """The conservative budget includes one bandit row per iteration."""
    assert estimate_witness_evaluation_budget(_spec(batch_size=1)) == 5
    assert estimate_witness_evaluation_budget(_spec(batch_size=4)) == 7


def test_assessment_names_episode_evaluation_and_pulse_blockers() -> None:
    """Budget overflow and the pulse-execution boundary remain explicit."""
    budget = assess_rl_research(
        _policy(max_episodes=1, max_evaluations_per_seed=4),
        RLResearchLane.WITNESS_DISCOVERY,
        spec=_spec(n_iterations=2),
    )
    pulse = assess_rl_research(_policy(), RLResearchLane.PULSE_OPTIMISATION)

    assert budget.blockers == ("episode_budget_exceeded", "evaluation_budget_exceeded")
    assert not budget.allowed
    assert pulse.estimated_evaluations_per_seed == 0
    assert pulse.blockers == (
        "rl_pulse_optimizer_unimplemented",
        "pulse_boundary_open",
    )
    assert RL_RESEARCH_CLAIM_BOUNDARY in pulse.as_dict()["claim_boundary"]
    with pytest.raises(ValueError, match="lane must be"):
        assess_rl_research(_policy(), "witness")  # type: ignore[arg-type]


def test_allowed_decision_and_seed_suite_are_exact() -> None:
    """An enabled preregistered route expands only its fixed seed tuple."""
    policy = _policy()
    spec = _spec()
    decision = assert_rl_research_allowed(policy, RLResearchLane.WITNESS_DISCOVERY, spec=spec)
    suite = build_witness_seed_suite(policy, spec)

    assert decision.allowed
    assert decision.blockers == ()
    assert decision.estimated_evaluations_per_seed == 5
    assert decision.reason == "local preregistered witness-discovery research admitted"
    assert tuple(item.seed for item in suite) == policy.seeds
    assert all(item.n_iterations == 1 for item in suite)


def test_decision_record_rejects_internal_inconsistency() -> None:
    """Admission, counts, and display fields cannot contradict each other."""
    valid = RLResearchDecision(
        lane=RLResearchLane.WITNESS_DISCOVERY,
        allowed=True,
        blockers=(),
        estimated_evaluations_per_seed=1,
        policy_id="p",
        reason="allowed",
    )
    with pytest.raises(ValueError, match="absence of blockers"):
        replace(valid, allowed=False)
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid, estimated_evaluations_per_seed=-1)
    with pytest.raises(ValueError, match="must be non-empty"):
        replace(valid, reason="")


def test_seed_evaluation_rejects_malformed_or_non_reproducible_rows() -> None:
    """Seed evidence requires finite candidates and byte-identical replay."""
    row = RLSeedEvaluation(1, 2, 0.5, (("x", 1.0),), True)
    assert row.as_dict()["best_candidate"] == {"x": 1.0}
    with pytest.raises(ValueError, match="evaluation_count must be positive"):
        replace(row, evaluation_count=0)
    with pytest.raises(ValueError, match="best_score must be finite"):
        replace(row, best_score=float("nan"))
    with pytest.raises(ValueError, match="finite named"):
        replace(row, best_candidate=(("", 1.0),))
    with pytest.raises(ValueError, match="names must be unique"):
        replace(row, best_candidate=(("x", 1.0), ("x", 2.0)))
    with pytest.raises(ValueError, match="byte-identical"):
        replace(row, replay_identical=False)
    with pytest.raises(ValueError, match="byte-identical"):
        replace(row, replay_identical=1)  # type: ignore[arg-type]


def test_governed_seed_suite_replays_three_seeds_exactly() -> None:
    """The real local search is deterministic across every fixed seed."""
    K_nm, omega, theta0 = _problem()
    report = run_governed_witness_seed_suite(
        K_nm,
        omega,
        policy=_policy(),
        template=_spec(),
        theta0=theta0,
        prefer_rust=False,
    )
    payload = report.as_dict()

    assert report.passed
    assert len(report.seed_results) == 3
    assert all(row.evaluation_count == 5 for row in report.seed_results)
    assert all(row.replay_identical for row in report.seed_results)
    assert payload["seed_count"] == 3
    assert payload["best_score_mean"] is not None
    assert payload["best_score_population_std"] is not None
    assert payload["evaluation_exploration_noise"] == 0.0
    assert payload["registry_grants_production_control"] is False
    assert payload["registry_grants_hardware_execution"] is False


def test_empty_suite_report_is_not_invent_green() -> None:
    """Aggregation exposes null statistics when no seed evidence exists."""
    decision = RLResearchDecision(RLResearchLane.WITNESS_DISCOVERY, True, (), 1, "p", "allowed")
    report = RLSeedSuiteReport(RL_RESEARCH_GOVERNANCE_SCHEMA, "p", "pre", decision, (), "0" * 64)
    payload = report.as_dict()
    assert not report.passed
    assert payload["best_score_mean"] is None
    assert payload["best_score_population_std"] is None


def test_suite_report_rejects_inconsistent_identity_and_digest() -> None:
    """A suite report cannot detach itself from its decision or evidence identity."""
    decision = RLResearchDecision(RLResearchLane.WITNESS_DISCOVERY, True, (), 1, "p", "allowed")
    row = RLSeedEvaluation(1, 1, 0.5, (("x", 1.0),), True)
    report = RLSeedSuiteReport(
        RL_RESEARCH_GOVERNANCE_SCHEMA, "p", "pre", decision, (row,), "0" * 64
    )

    with pytest.raises(ValueError, match="schema must match"):
        replace(report, schema="wrong")
    with pytest.raises(ValueError, match="policy_id must match"):
        replace(report, policy_id="other")
    with pytest.raises(ValueError, match="preregistration_id must be non-empty"):
        replace(report, preregistration_id=" ")
    with pytest.raises(ValueError, match="unique seeds"):
        replace(report, seed_results=(row, row))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(report, content_digest="F" * 64)


def test_frozen_evidence_and_markdown_are_stable_and_non_promotional() -> None:
    """The governed fixture is digest-bound and says what it cannot prove."""
    first = build_rl_research_evidence_report()
    second = build_rl_research_evidence_report()
    markdown = render_rl_research_evidence_markdown(first)
    implicit = render_rl_research_evidence_markdown()

    assert first.passed
    assert first.content_digest == second.content_digest
    assert len(first.content_digest) == 64
    assert markdown == implicit
    assert markdown.endswith("\n")
    assert "no Gym environment is introduced" in markdown
    assert "dense composite" in markdown
    assert "no policy-performance claim" in markdown


def test_discovery_agent_requires_policy_after_real_problem_validation() -> None:
    """Configured discovery stays disabled until explicit research admission."""
    K_nm, omega, theta0 = _problem()
    disabled = RLDiscoveryAgent(K_nm=K_nm, omega=omega, theta0=theta0, spec=_spec())
    with pytest.raises(RLResearchGovernanceError, match="research_extra_disabled"):
        asyncio.run(disabled.run_discovery_loop())

    enabled = RLDiscoveryAgent(
        K_nm=K_nm,
        omega=omega,
        theta0=theta0,
        spec=_spec(),
        policy=_policy(),
    )
    assert asyncio.run(enabled.run_discovery_loop()).spec.seed == 0


def test_pulse_optimizer_is_disabled_and_bl58_blocked() -> None:
    """Neither default nor enabled research policy can execute pulse RL."""
    disabled = RLPulseOptimizer(object(), episodes=1)
    enabled = RLPulseOptimizer(object(), episodes=1, policy=_policy())
    with pytest.raises(RLResearchGovernanceError, match="research_extra_disabled"):
        asyncio.run(disabled.optimize_pulses())
    with pytest.raises(RLResearchGovernanceError, match="rl_pulse_optimizer_unimplemented"):
        asyncio.run(enabled.optimize_pulses())
    with pytest.raises(NotImplementedError, match="No RL pulse"):
        enabled.save_results("unused.json")


def test_pulse_defence_in_depth_fallback_is_unimplemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future accidental admission still cannot invent a pulse optimiser."""
    monkeypatch.setattr(
        "scpn_quantum_control.analysis.rl_pulse_optimizer.assert_rl_research_allowed",
        lambda *_args, **_kwargs: None,
    )
    optimizer = RLPulseOptimizer(object(), episodes=1, policy=_policy())
    with pytest.raises(AssertionError, match="unreachable"):
        asyncio.run(optimizer.optimize_pulses())


def test_runner_writes_checks_and_rejects_missing_or_stale_files(tmp_path: Path) -> None:
    """Evidence runner uses one canonical byte representation in both modes."""
    runner = _load_runner()
    json_path = tmp_path / "nested/evidence.json"
    markdown_path = tmp_path / "nested/evidence.md"
    with pytest.raises(SystemExit, match="missing RL research evidence"):
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"])
    assert runner.main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    expected_json, expected_markdown = runner._expected_bytes()
    assert json_path.read_bytes() == expected_json
    assert markdown_path.read_bytes() == expected_markdown
    assert (
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"]) == 0
    )
    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale RL research evidence"):
        runner.main(["--json", str(json_path), "--markdown", str(markdown_path), "--check"])


def test_private_positive_index_rejects_bool_and_non_integer() -> None:
    """Integer normalization never accepts bool or lossy float conversion."""
    with pytest.raises(ValueError, match="must be an integer"):
        governance._positive_index(True, "x")
    with pytest.raises(ValueError, match="must be an integer"):
        governance._positive_index(1.5, "x")
