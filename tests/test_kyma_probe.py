# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the KYMA probe orchestration
"""End-to-end probe orchestration on a tiny configuration.

Uses a deliberately tiny config (few osc -steps, few trials, few epochs) so the
full train → evaluate → J/task → aggregate path runs quickly while exercising
``run_seed`` and ``run_probe`` and the frozen pass/fail contract wiring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")  # KYMA probe requires the optional [jax] extra, absent from the CI lane


from scpn_quantum_control.benchmarks.kyma import probe
from scpn_quantum_control.benchmarks.kyma.task import ProbeConfig

_TINY = ProbeConfig(steps=5, trials_per_single=2, trials_per_conjunction=2, test_trials=8)


def _fast(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(probe, "SUBSTRATE_EPOCHS", 5)
    monkeypatch.setattr(probe, "MLP_EPOCHS", 5)


def test_run_seed_reports_all_fields(monkeypatch) -> None:  # noqa: ANN001
    _fast(monkeypatch)
    result = probe.run_seed(0, _TINY)
    assert result.seed == 0
    assert 0.0 <= result.substrate_accuracy <= 1.0
    assert 0.0 <= result.mlp_accuracy <= 1.0
    assert result.substrate_params == 312
    # MLP parameter count is matched within ±10 %.
    assert abs(result.mlp_params - result.substrate_params) / result.substrate_params <= 0.10
    assert result.substrate_j_per_task > 0.0
    assert result.mlp_j_per_task > 0.0
    assert result.substrate_steps == _TINY.steps


def test_run_probe_evaluates_frozen_contract(monkeypatch) -> None:  # noqa: ANN001
    _fast(monkeypatch)
    out = probe.run_probe((0, 1), _TINY)
    assert out["verdict"] in {"PASS", "NEGATIVE"}
    assert set(out["substrate_accuracy"]) == {"mean", "sd"}
    assert out["thresholds"]["pass_accuracy"] == 0.70
    assert out["thresholds"]["pass_margin_pp"] == 0.25
    assert out["margin_over_mlp_pp"] == (
        out["substrate_accuracy"]["mean"] - out["mlp_accuracy"]["mean"]
    )
    assert len(out["per_seed"]) == 2
    assert out["param_counts"]["substrate"] == 312


def test_verdict_pass_requires_all_conditions(monkeypatch) -> None:  # noqa: ANN001
    # A hand-built result where the substrate beats the bar → PASS wiring.
    _fast(monkeypatch)
    out = probe.run_probe((0,), _TINY)
    sub = out["substrate_accuracy"]["mean"]
    mlp = out["mlp_accuracy"]["mean"]
    expected = (
        sub >= probe.PASS_ACCURACY
        and (sub - mlp) >= probe.PASS_MARGIN_PP
        and sub > out["chance_accuracy"]["mean"]
    )
    assert (out["verdict"] == "PASS") == expected
