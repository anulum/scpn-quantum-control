# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for closed-loop control analysis
"""Tests for control/closed_loop_analysis.py: response classification and policy gate."""

import warnings

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_kuramoto_ring
from scpn_quantum_control.control.closed_loop_analysis import (
    ClosedLoopExecutionPolicy,
    ClosedLoopLatencyBudget,
    ExecutionMode,
    ResponseClass,
    analyse_closed_loop_response,
    build_closed_loop_publication_package,
    evaluate_closed_loop_policy,
    measure_closed_loop_latency_budget,
    run_closed_loop_control,
)
from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)

_TARGET = 0.75
_ROUNDS = 40


# --------------------------------------------------------------------------- #
# Response classification
# --------------------------------------------------------------------------- #
def test_converged_response():
    response = _TARGET - 0.5 * np.exp(-0.3 * np.arange(_ROUNDS))
    verdict, perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.05)
    assert verdict is ResponseClass.CONVERGED
    assert perf.settling_round is not None
    assert perf.steady_state_error <= 0.05


def test_limit_cycle_response():
    response = _TARGET + 0.15 * np.sin(0.6 * np.arange(_ROUNDS))
    verdict, perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.05)
    assert verdict is ResponseClass.LIMIT_CYCLE
    assert perf.oscillation_amplitude > 0.05
    assert perf.error_sign_changes >= 4


def test_diverged_response():
    response = 0.75 - 0.012 * np.arange(_ROUNDS)
    verdict, _perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.05)
    assert verdict is ResponseClass.DIVERGED


def test_unsettled_response():
    response = _TARGET - 0.5 + 0.008 * np.arange(_ROUNDS)
    verdict, _perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.02)
    assert verdict is ResponseClass.UNSETTLED


def test_transient_oscillation_then_settle_is_converged():
    response = _TARGET + np.concatenate([0.3 * np.sin(np.arange(12)), np.zeros(28)])
    verdict, _perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.03)
    assert verdict is ResponseClass.CONVERGED


def test_metrics_on_known_signal():
    # Constant offset of 0.1 below target for 10 rounds.
    response = np.full(10, _TARGET - 0.1)
    _verdict, perf = analyse_closed_loop_response(response, _TARGET, tolerance=0.2)
    assert perf.integral_absolute_error == pytest.approx(1.0)
    assert perf.steady_state_error == pytest.approx(0.1)
    assert perf.final_value == pytest.approx(_TARGET - 0.1)
    assert perf.settling_round == 0  # already inside the 0.2 band from round 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response": np.array([0.5]), "target": 0.5, "tolerance": 0.1},
        {"response": np.zeros((2, 2)), "target": 0.5, "tolerance": 0.1},
        {"response": np.zeros(5), "target": 0.5, "tolerance": 0.0},
        {"response": np.zeros(5), "target": 0.5, "tolerance": 0.1, "settle_window": 0},
    ],
)
def test_analyse_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        analyse_closed_loop_response(**kwargs)


# --------------------------------------------------------------------------- #
# Execution policy gate
# --------------------------------------------------------------------------- #
def test_policy_defaults_to_simulation():
    decision = evaluate_closed_loop_policy(ClosedLoopExecutionPolicy(), requested_rounds=10)
    assert decision.authorised
    assert decision.mode is ExecutionMode.SIMULATION


def test_policy_authorises_hardware_with_ticket_and_backend():
    policy = ClosedLoopExecutionPolicy(
        allow_hardware=True, live_ticket="TICKET-1", backend_allowlist=("ibm_heron",)
    )
    decision = evaluate_closed_loop_policy(policy, backend="ibm_heron", requested_rounds=10)
    assert decision.authorised
    assert decision.mode is ExecutionMode.HARDWARE


def test_policy_refuses_hardware_without_ticket():
    policy = ClosedLoopExecutionPolicy(allow_hardware=True, backend_allowlist=("ibm_heron",))
    decision = evaluate_closed_loop_policy(policy, backend="ibm_heron", requested_rounds=10)
    assert not decision.authorised
    assert decision.mode is ExecutionMode.SIMULATION


def test_policy_refuses_unlisted_backend():
    policy = ClosedLoopExecutionPolicy(
        allow_hardware=True, live_ticket="TICKET-1", backend_allowlist=("ibm_heron",)
    )
    decision = evaluate_closed_loop_policy(policy, backend="rogue", requested_rounds=10)
    assert not decision.authorised


def test_policy_blocks_over_budget():
    policy = ClosedLoopExecutionPolicy(round_budget=5)
    decision = evaluate_closed_loop_policy(policy, requested_rounds=10)
    assert not decision.authorised
    assert "budget" in decision.reason


def test_policy_rejects_bad_budget():
    with pytest.raises(ValueError):
        ClosedLoopExecutionPolicy(round_budget=0)


def test_policy_rejects_bad_round_request():
    with pytest.raises(ValueError):
        evaluate_closed_loop_policy(ClosedLoopExecutionPolicy(), requested_rounds=0)


# --------------------------------------------------------------------------- #
# Controller integration
# --------------------------------------------------------------------------- #
def _controller() -> RealtimeSyncFeedbackController:
    K, omega = build_kuramoto_ring(4, coupling=0.6, rng_seed=0)
    return RealtimeSyncFeedbackController(K, omega, config=RealtimeFeedbackConfig(target_r=0.6))


def test_run_closed_loop_control_is_replayable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = run_closed_loop_control(_controller(), 20, seed=7)
        second = run_closed_loop_control(_controller(), 20, seed=7)
    assert np.array_equal(first.response, second.response)
    assert np.array_equal(first.feedback_signal, second.feedback_signal)


def test_run_closed_loop_control_evidence_structure():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evidence = run_closed_loop_control(_controller(), 20, seed=3)
    assert evidence.response.shape == (20,)
    assert evidence.decision.mode is ExecutionMode.SIMULATION
    assert evidence.classification in set(ResponseClass)
    assert np.isfinite(evidence.performance.integral_absolute_error)
    assert "claim_boundary" in evidence.provenance
    assert evidence.target == pytest.approx(0.6)


def test_run_closed_loop_control_rejects_short_horizon():
    with pytest.raises(ValueError):
        run_closed_loop_control(_controller(), 1)


# --------------------------------------------------------------------------- #
# Latency budget and publication scaffold
# --------------------------------------------------------------------------- #
def test_measure_closed_loop_latency_budget_accepts_replay_samples():
    budget = ClosedLoopLatencyBudget(
        max_round_latency_s=0.010,
        p95_round_latency_s=0.008,
        p99_round_latency_s=0.009,
        max_total_latency_s=0.050,
    )

    report = measure_closed_loop_latency_budget(
        _controller(),
        4,
        budget=budget,
        seed=11,
        observed_round_latencies_s=(0.001, 0.002, 0.003, 0.004),
    )

    assert report.passes
    assert report.blockers == ()
    assert report.classification == "software_in_loop_latency"
    assert report.samples == 4
    assert report.max_round_latency_s == pytest.approx(0.004)
    assert report.total_latency_s == pytest.approx(0.010)
    assert report.control_evidence.decision.mode is ExecutionMode.SIMULATION
    assert report.to_dict()["claim_boundary"].startswith("software-in-the-loop")


def test_measure_closed_loop_latency_budget_fails_closed_on_budget_breach():
    budget = ClosedLoopLatencyBudget(max_round_latency_s=0.002, max_total_latency_s=0.010)

    report = measure_closed_loop_latency_budget(
        _controller(),
        3,
        budget=budget,
        observed_round_latencies_s=(0.001, 0.003, 0.001),
    )

    assert not report.passes
    assert any("max round latency" in blocker for blocker in report.blockers)


def test_measure_closed_loop_latency_budget_records_policy_block_without_hardware_submit():
    policy = ClosedLoopExecutionPolicy(
        allow_hardware=True,
        live_ticket=None,
        backend_allowlist=("ibm_heron",),
    )

    report = measure_closed_loop_latency_budget(
        _controller(),
        3,
        policy=policy,
        backend="ibm_heron",
        observed_round_latencies_s=(0.001, 0.001, 0.001),
    )

    assert not report.passes
    assert report.control_evidence.decision.mode is ExecutionMode.SIMULATION
    assert any("not authorised" in blocker for blocker in report.blockers)


def test_measure_closed_loop_latency_budget_rejects_bad_budget_and_samples():
    with pytest.raises(ValueError, match="max_round_latency_s"):
        ClosedLoopLatencyBudget(max_round_latency_s=0.0)

    with pytest.raises(ValueError, match="observed_round_latencies_s"):
        measure_closed_loop_latency_budget(
            _controller(),
            3,
            observed_round_latencies_s=(0.001, 0.002),
        )


def test_closed_loop_publication_package_separates_evidence_classes():
    latency_report = measure_closed_loop_latency_budget(
        _controller(),
        4,
        observed_round_latencies_s=(0.001, 0.0015, 0.0012, 0.0013),
    )

    package = build_closed_loop_publication_package(latency_report=latency_report)
    payload = package.to_dict()
    markdown = package.to_markdown()

    assert package.title == "Closed-Loop Quantum Control Evidence Package"
    assert "software_in_loop_simulation" in payload["evidence_classes"]
    assert "provider_prepared_dynamic_circuit" in payload["evidence_classes"]
    assert "live_closed_loop_qpu" in payload["evidence_classes"]
    assert payload["benchmark_rows"][0]["classification"] == "software_in_loop_latency"
    assert payload["claim_ledger_rows"][0]["promotion_status"] == "unpromoted"
    assert "not live closed-loop QPU evidence" in markdown
