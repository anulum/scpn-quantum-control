# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2.1 supplementary-rigor orchestration
"""Tests for the v2.1 rigor orchestration (ablations, baselines, convergence, LOO).

Uses a tiny config and a few epochs (monkeypatched) — the research budget is
exercised by the runner, not the suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import rigor, task  # noqa: E402


def _tiny_cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(
        g_sync=0.5,
        steps=20,
        dt=0.1,
        k_bridge=0.8,
        trials_per_single=3,
        trials_per_conjunction=3,
        test_trials=24,
    )


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rigor, "SUBSTRATE_EPOCHS", 30)
    monkeypatch.setattr(rigor, "MLP_EPOCHS", 30)


def test_run_ablations_structure() -> None:
    out = rigor.run_ablations(_tiny_cfg(), seeds=(0,))
    for key in ("A1_no_gating", "A2_separable_readout"):
        assert key in out
        assert "meets_prediction" in out[key]
        assert isinstance(out[key]["margin_pp"], float)
    # A1 reports a shared-substrate vs MLP comparison
    assert 0.0 <= out["A1_no_gating"]["shared_substrate"]["mean"] <= 1.0


def test_run_stronger_baselines_structure() -> None:
    out = rigor.run_stronger_baselines(_tiny_cfg(), seeds=(0,), gnn_hidden=8)
    for key in ("substrate", "over_param_mlp", "deep_mlp", "gnn"):
        assert key in out
    # over-parameterised MLP really has ~4x the substrate budget
    assert out["over_param_mlp"]["params"] > 3 * 336
    assert "meets_prediction" in out["gnn"]


def test_mlp_convergence_reports_trajectory_and_train_accuracy() -> None:
    out = rigor.mlp_convergence(_tiny_cfg(), seed=0, probes=3)
    assert len(out["loss_trajectory"]) == len(out["checkpoint_epochs"]) == 3
    assert 0.0 <= out["train_accuracy"] <= 1.0
    assert isinstance(out["plateaued"], bool)


def test_leave_one_out_covers_all_six_splits() -> None:
    out = rigor.run_leave_one_out(base_config=_tiny_cfg(), seeds=(0,))
    assert len(out["per_split"]) == 6
    for split in out["per_split"]:
        assert not (set(task.PAIRS[split["held_out"][0]]) & set(task.PAIRS[split["held_out"][1]]))
    assert out["substrate_wins"].endswith("/6")
    assert isinstance(out["mean_margin_pp"], float)
