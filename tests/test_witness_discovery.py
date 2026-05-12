# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Automated witness discovery tests
"""Tests for automated Kuramoto witness discovery."""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import pytest

from scpn_quantum_control.analysis import (
    RLDiscoveryAgent,
    WitnessCandidate,
    WitnessDiscoveryResult,
    WitnessDiscoverySpec,
    WitnessSearchMode,
    discover_kuramoto_witnesses,
    score_witness_candidates,
)
from scpn_quantum_control.analysis.witness_discovery import (
    _candidate_features_numpy,
    _correlation_witness_from_mean,
    _nearest_distance,
    _rbf_surrogate_predict,
    _rl_bandit_candidates,
    _select_unique_candidates,
)


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_nm = np.array(
        [
            [0.0, 0.5, 0.2, 0.0],
            [0.5, 0.0, 0.4, 0.1],
            [0.2, 0.4, 0.0, 0.3],
            [0.0, 0.1, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.0, 0.35, 0.7, 1.05], dtype=np.float64)
    theta0 = np.array([0.0, 0.7, 1.4, 2.8], dtype=np.float64)
    return K_nm, omega, theta0


def _small_spec(seed: int = 123) -> WitnessDiscoverySpec:
    return WitnessDiscoverySpec(
        dt=0.025,
        n_steps=32,
        n_initial=5,
        n_iterations=3,
        batch_size=3,
        pool_size=20,
        seed=seed,
        coupling_bounds=(0.0, 2.2),
        omega_bounds=(0.6, 1.4),
        phase_bias_bounds=(-0.2, 0.2),
        correlation_threshold=0.25,
        fiedler_threshold=0.2,
    )


def test_fixed_candidate_scoring_uses_witness_results_and_metadata() -> None:
    K_nm, omega, theta0 = _problem()
    candidates = [
        WitnessCandidate(0.1, 1.0, 0.0),
        WitnessCandidate(1.8, 0.8, 0.05),
    ]

    evaluations = score_witness_candidates(
        K_nm,
        omega,
        candidates,
        theta0=theta0,
        spec=_small_spec(),
        prefer_rust=False,
    )

    assert len(evaluations) == 2
    assert all(item.backend == "numpy:kuramoto_witness_candidate_features" for item in evaluations)
    assert all(item.correlation_witness.witness_name == "correlation" for item in evaluations)
    assert all(item.fiedler_witness.witness_name == "fiedler" for item in evaluations)
    assert all(0.0 <= item.final_r <= 1.0 for item in evaluations)
    assert evaluations[0].to_metadata()["candidate"]["coupling_scale"] == 0.1


def test_candidate_scoring_rejects_empty_candidate_batches() -> None:
    K_nm, omega, theta0 = _problem()

    with pytest.raises(ValueError, match="at least one candidate"):
        score_witness_candidates(K_nm, omega, [], theta0=theta0, spec=_small_spec())


def test_single_oscillator_candidate_features_use_trivial_correlation() -> None:
    final_r, mean_corr, final_theta = _candidate_features_numpy(
        np.array([0.1], dtype=np.float64),
        np.array([0.2], dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        dt=0.05,
        n_steps=3,
    )

    assert final_r[0] == pytest.approx(1.0)
    assert mean_corr[0] == pytest.approx(1.0)
    assert final_theta.shape == (1, 1)


def test_discovery_loop_is_deterministic_and_uses_bayesian_and_rl_sources() -> None:
    K_nm, omega, theta0 = _problem()
    spec = _small_spec(seed=77)

    result_1 = discover_kuramoto_witnesses(
        K_nm, omega, theta0=theta0, spec=spec, prefer_rust=False
    )
    result_2 = discover_kuramoto_witnesses(
        K_nm, omega, theta0=theta0, spec=spec, prefer_rust=False
    )

    assert isinstance(result_1, WitnessDiscoveryResult)
    assert result_1.best.score == pytest.approx(result_2.best.score)
    assert result_1.best.candidate.to_metadata() == result_2.best.candidate.to_metadata()
    sources = {item.source for item in result_1.evaluations}
    assert WitnessSearchMode.INITIAL in sources
    assert WitnessSearchMode.BAYESIAN_UCB in sources
    assert WitnessSearchMode.RL_BANDIT in sources
    assert result_1.best in result_1.ranked(limit=1)
    assert result_1.ranked()[0].score >= result_1.ranked()[-1].score


def test_discovery_metadata_is_json_safe_and_immutable() -> None:
    spec = WitnessDiscoverySpec(metadata={"dataset": "synthetic", "seed": 12})

    assert spec.metadata["dataset"] == "synthetic"
    with pytest.raises(TypeError):
        spec.metadata["dataset"] = "mutated"
    with pytest.raises(TypeError, match="JSON"):
        WitnessDiscoverySpec(metadata={"bad": object()})
    with pytest.raises(TypeError, match="keys"):
        WitnessDiscoverySpec(metadata={1: "not-a-string"})


def test_preferred_rust_feature_path_matches_numpy_when_available() -> None:
    K_nm, omega, theta0 = _problem()
    spec = _small_spec()
    candidates = np.array(
        [
            [0.2, 1.0, 0.0],
            [1.4, 0.9, 0.1],
            [2.0, 0.7, -0.1],
        ],
        dtype=np.float64,
    )

    numpy_r, numpy_corr, numpy_theta = _candidate_features_numpy(
        theta0,
        omega,
        K_nm,
        candidates,
        spec.dt,
        spec.n_steps,
    )

    try:
        import scpn_quantum_engine as engine
    except ImportError:
        pytest.skip("Rust engine not installed")

    if not hasattr(engine, "kuramoto_witness_candidate_features"):
        pytest.skip("Rust witness discovery kernel not installed")
    rust_r, rust_corr, rust_theta = engine.kuramoto_witness_candidate_features(
        theta0,
        omega,
        K_nm,
        candidates,
        spec.dt,
        spec.n_steps,
    )

    np.testing.assert_allclose(rust_r, numpy_r, atol=1e-12)
    np.testing.assert_allclose(rust_corr, numpy_corr, atol=1e-12)
    np.testing.assert_allclose(rust_theta, numpy_theta, atol=1e-12)


def test_preferred_rust_feature_path_records_backend_provenance(monkeypatch) -> None:
    K_nm, omega, theta0 = _problem()
    spec = _small_spec()

    class FakeEngine:
        @staticmethod
        def kuramoto_witness_candidate_features(theta0, omega, K_nm, candidates, dt, n_steps):
            n_candidates = candidates.shape[0]
            return (
                np.full(n_candidates, 0.5, dtype=np.float64),
                np.full(n_candidates, 0.25, dtype=np.float64),
                np.tile(theta0, (n_candidates, 1)),
            )

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", FakeEngine)

    evaluations = score_witness_candidates(
        K_nm,
        omega,
        [WitnessCandidate(0.2, 1.0, 0.0)],
        theta0=theta0,
        spec=spec,
        prefer_rust=True,
    )

    assert evaluations[0].backend == "rust:kuramoto_witness_candidate_features"
    assert evaluations[0].final_r == pytest.approx(0.5)
    assert evaluations[0].mean_correlation == pytest.approx(0.25)


def test_bayesian_surrogate_prefers_near_high_scoring_region() -> None:
    X_train = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_train = np.array([0.1, 1.0, 0.2], dtype=np.float64)
    X_pool = np.array(
        [
            [1.05, 1.0, 0.0],
            [0.05, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    mean, std = _rbf_surrogate_predict(X_train, y_train, X_pool)

    assert mean[0] > mean[1]
    assert np.all(std >= 0.0)


def test_bayesian_surrogate_and_novelty_have_prior_for_single_training_point() -> None:
    X_train = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    y_train = np.array([0.7], dtype=np.float64)
    X_pool = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)

    mean, std = _rbf_surrogate_predict(X_train, y_train, X_pool)
    novelty = _nearest_distance(X_train, X_pool)

    np.testing.assert_allclose(mean, np.zeros(2))
    np.testing.assert_allclose(std, np.ones(2))
    assert novelty[0] == pytest.approx(0.0)
    assert novelty[1] > novelty[0]


def test_bayesian_surrogate_falls_back_to_pseudoinverse_for_singular_solver(
    monkeypatch,
) -> None:
    X_train = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_train = np.array([0.2, 0.8], dtype=np.float64)
    X_pool = np.array([[0.5, 1.0, 0.0]], dtype=np.float64)

    def raise_singular(*args, **kwargs):
        raise np.linalg.LinAlgError("forced singular solve")

    monkeypatch.setattr(np.linalg, "solve", raise_singular)

    mean, std = _rbf_surrogate_predict(X_train, y_train, X_pool)

    assert np.isfinite(mean[0])
    assert std[0] >= 0.0


def test_rl_bandit_zero_count_and_duplicate_filtering_are_stable() -> None:
    spec = _small_spec(seed=17)
    best = WitnessCandidate(1.0, 1.0, 0.0)
    empty = _rl_bandit_candidates(spec, best, 0, np.random.default_rng(17))

    assert empty.shape == (0, 3)

    evaluated = score_witness_candidates(
        *_problem()[:2],
        candidates=[best],
        theta0=_problem()[2],
        spec=spec,
        prefer_rust=False,
    )
    pool = np.array([[1.0, 1.0, 0.0], [1.2, 1.1, 0.1]], dtype=np.float64)
    acquisition = np.array([10.0, 1.0], dtype=np.float64)

    selected, selected_acq = _select_unique_candidates(pool, acquisition, evaluated, 2)

    np.testing.assert_allclose(selected, [[1.2, 1.1, 0.1]])
    np.testing.assert_allclose(selected_acq, [1.0])

    all_seen, all_seen_acq = _select_unique_candidates(
        np.array([[1.0, 1.0, 0.0]], dtype=np.float64),
        np.array([5.0], dtype=np.float64),
        evaluated,
        1,
    )
    assert all_seen.shape == (0, 3)
    assert all_seen_acq.shape == (0,)


def test_nearest_distance_empty_training_set_returns_unit_novelty() -> None:
    novelty = _nearest_distance(
        np.zeros((0, 3), dtype=np.float64),
        np.array([[0.1, 1.0, 0.0], [0.2, 1.1, 0.1]], dtype=np.float64),
    )

    np.testing.assert_allclose(novelty, np.ones(2))


def test_correlation_witness_encodes_threshold_crossing_direction() -> None:
    unsynchronised = _correlation_witness_from_mean(0.2, n_qubits=4, threshold=0.5)
    synchronised = _correlation_witness_from_mean(0.8, n_qubits=4, threshold=0.5)

    assert unsynchronised.is_synchronized is False
    assert unsynchronised.expectation_value == pytest.approx(0.3)
    assert synchronised.is_synchronized is True
    assert synchronised.expectation_value == pytest.approx(-0.3)


def test_discovery_result_serialises_to_json() -> None:
    K_nm, omega, theta0 = _problem()
    result = discover_kuramoto_witnesses(
        K_nm,
        omega,
        theta0=theta0,
        spec=_small_spec(seed=5),
        prefer_rust=False,
    )

    payload = json.loads(result.to_json())

    assert payload["n_oscillators"] == 4
    assert payload["best"]["score"] == pytest.approx(result.best.score)
    assert len(payload["evaluations"]) == len(result.evaluations)


def test_rl_discovery_agent_requires_real_problem_and_runs_when_configured() -> None:
    agent = RLDiscoveryAgent(n_episodes=1)
    with pytest.raises(NotImplementedError, match="requires K_nm and omega"):
        asyncio.run(agent.run_discovery_loop())

    K_nm, omega, theta0 = _problem()
    configured = RLDiscoveryAgent(
        n_episodes=1,
        K_nm=K_nm,
        omega=omega,
        theta0=theta0,
        spec=_small_spec(seed=9),
    )
    result = asyncio.run(configured.run_discovery_loop())

    assert isinstance(result, WitnessDiscoveryResult)
    assert configured.get_next_params() == result.best.candidate.to_metadata()
    assert configured.discovered_phases[0]["score"] == pytest.approx(result.best.score)
    with pytest.raises(NotImplementedError, match="External reward mutation"):
        configured.update_reward({"score": 1.0})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"runner": object()}, "runner"),
        ({"observables": ["correlation"]}, "observables"),
        ({"reward_function": "custom_reward"}, "reward_function"),
        ({"n_episodes": 0}, "n_episodes"),
    ],
)
def test_rl_discovery_agent_rejects_unwired_compatibility_parameters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RLDiscoveryAgent(**kwargs)


def test_invalid_inputs_rejected_before_search() -> None:
    K_nm, omega, theta0 = _problem()

    result = discover_kuramoto_witnesses(K_nm, omega, spec=_small_spec(), prefer_rust=False)
    assert result.n_oscillators == omega.size

    with pytest.raises(ValueError, match="symmetric"):
        discover_kuramoto_witnesses(K_nm + np.triu(np.ones_like(K_nm), 1), omega)
    with pytest.raises(ValueError, match="theta0"):
        discover_kuramoto_witnesses(K_nm, omega, theta0=theta0[:2])
    with pytest.raises(ValueError, match="coupling_bounds"):
        WitnessDiscoverySpec(coupling_bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match="coupling_scale"):
        WitnessCandidate(-0.1, 1.0, 0.0)
    with pytest.raises(ValueError, match="phase_bias"):
        WitnessCandidate(0.1, 1.0, float("nan"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"K_nm": np.ones((2, 3)), "omega": np.ones(2)}, "square"),
        ({"K_nm": np.eye(2), "omega": np.ones(3)}, "omega"),
        ({"K_nm": np.array([[0.0, np.nan], [np.nan, 0.0]]), "omega": np.ones(2)}, "K_nm"),
        ({"K_nm": np.eye(2), "omega": np.array([0.0, np.inf])}, "omega"),
        ({"K_nm": np.eye(2), "omega": np.ones(2), "theta0": np.array([0.0, np.nan])}, "theta0"),
    ],
)
def test_problem_validation_rejects_non_physical_inputs(kwargs, match) -> None:
    theta0 = kwargs.pop("theta0", None)

    with pytest.raises(ValueError, match=match):
        discover_kuramoto_witnesses(**kwargs, theta0=theta0, spec=_small_spec(), prefer_rust=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt": 0.0}, "dt"),
        ({"n_steps": 0}, "n_steps"),
        ({"pool_size": 2, "batch_size": 3}, "pool_size"),
        ({"correlation_threshold": 3.0}, "correlation_threshold"),
        ({"fiedler_threshold": -0.1}, "fiedler_threshold"),
        ({"final_r_weight": -0.1}, "final_r_weight"),
        ({"rl_epsilon": 1.1}, "rl_epsilon"),
    ],
)
def test_discovery_spec_rejects_invalid_search_contract(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        WitnessDiscoverySpec(**kwargs)
