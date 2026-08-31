# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the KYMA v2 probe orchestration
"""Tests for seed orchestration and the frozen pass/fail contract.

Uses a tiny config and a few epochs (monkeypatched) — the research budget is
exercised by the runner, not the test suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import probe, task  # noqa: E402


def _tiny_cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(
        g_sync=0.5,
        steps=20,
        dt=0.1,
        k_bridge=0.8,
        trials_per_single=2,
        trials_per_conjunction=2,
        test_trials=24,
    )


@pytest.fixture(autouse=True)
def _fast_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce both training budgets for every direct orchestration test."""
    monkeypatch.setattr(probe, "STUDENT_EPOCHS", 40)
    monkeypatch.setattr(probe, "MLP_EPOCHS", 40)


def _seed_result(
    seed: int,
    student_accuracy: float,
    mlp_accuracy: float,
    chance_accuracy: float,
) -> probe.SeedResultV2:
    """Build a deterministic v2 result for aggregate-contract tests."""
    return probe.SeedResultV2(
        seed=seed,
        student_accuracy=student_accuracy,
        mlp_accuracy=mlp_accuracy,
        chance_accuracy=chance_accuracy,
        student_params=400,
        mlp_params=396,
        mlp_hidden=12,
        student_j_per_task=0.2,
        mlp_j_per_task=0.1,
        steps=_tiny_cfg().steps,
    )


def test_run_seed_returns_populated_result() -> None:
    """Return all registered metrics for one deterministic seed."""
    r = probe.run_seed(0, _tiny_cfg())
    assert r.seed == 0
    for acc in (r.student_accuracy, r.mlp_accuracy, r.chance_accuracy):
        assert 0.0 <= acc <= 1.0
    assert r.student_params > 0 and r.mlp_params > 0
    assert r.student_j_per_task > 0 and r.mlp_j_per_task > 0


def test_run_probe_aggregates_and_scores_contract() -> None:
    """Aggregate seed metrics under the frozen v2 contract."""
    result = probe.run_probe((0, 1), _tiny_cfg())
    assert result["verdict"] in {"PASS", "NEGATIVE"}
    assert set(result["student_accuracy"]) == {"mean", "sd"}
    assert result["margin_over_mlp_pp"] == pytest.approx(
        result["student_accuracy"]["mean"] - result["mlp_accuracy"]["mean"]
    )
    assert len(result["per_seed"]) == 2
    assert result["thresholds"]["pass_accuracy"] == probe.PASS_ACCURACY


def test_pass_requires_all_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require every preregistered condition for a passing verdict."""
    # A verdict of PASS must satisfy accuracy, margin, and chance simultaneously.
    result = probe.run_probe((0,), _tiny_cfg())
    s = result["student_accuracy"]["mean"]
    m = result["margin_over_mlp_pp"]
    c = result["chance_accuracy"]["mean"]
    expected = s >= probe.PASS_ACCURACY and m >= probe.PASS_MARGIN_PP and s > c
    assert (result["verdict"] == "PASS") == expected


@pytest.mark.parametrize(
    ("student_accuracy", "mlp_accuracy", "chance_accuracy", "verdict"),
    [
        (0.70, 0.40, 0.20, "PASS"),
        (0.59, 0.30, 0.20, "NEGATIVE"),
        (0.70, 0.55, 0.20, "NEGATIVE"),
        (0.70, 0.40, 0.70, "NEGATIVE"),
    ],
)
def test_run_probe_guards_every_frozen_verdict_branch(
    student_accuracy: float,
    mlp_accuracy: float,
    chance_accuracy: float,
    verdict: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard accuracy, MLP margin, and chance separation independently."""

    def deterministic_seed(seed: int, _config: task.ProbeConfigV2) -> probe.SeedResultV2:
        return _seed_result(seed, student_accuracy, mlp_accuracy, chance_accuracy)

    monkeypatch.setattr(probe, "run_seed", deterministic_seed)
    assert probe.run_probe((0,), _tiny_cfg())["verdict"] == verdict
