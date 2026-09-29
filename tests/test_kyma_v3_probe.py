# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KYMA v3 probe orchestration
"""Tests for the KYMA v3 probe orchestration and frozen contract."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v3 import probe


def _result(seed: int, substrate: float, scores: dict[str, float]) -> probe.SeedResult:
    return probe.SeedResult(
        seed=seed,
        substrate_train_accuracy=1.0,
        substrate_test_accuracy=substrate,
        substrate_seconds_per_item=1e-4,
        baseline_train_accuracy=dict.fromkeys(scores, 1.0),
        baseline_test_accuracy=scores,
        baseline_seconds_per_item=dict.fromkeys(scores, 1e-5),
    )


_LOW = {
    "mlp": 0.30,
    "gnn": 0.40,
    "transformer": 0.35,
    "sequential": 0.30,
    "mlp_large": 0.45,
    "gnn_large": 0.50,
    "transformer_large": 0.30,
}


def test_contract_and_diagnostic_specs() -> None:
    """Check that contract and diagnostic specs."""
    specs = {spec.name: spec for spec in probe.baseline_specs()}
    assert [name for name, spec in specs.items() if spec.contract] == ["mlp", "gnn", "transformer"]
    assert (specs["mlp"].width, specs["mlp"].params) == (5, 109)
    assert (specs["gnn"].width, specs["gnn"].params) == (4, 103)
    assert (specs["transformer"].width, specs["transformer"].params) == (3, 100)
    assert (specs["sequential"].width, specs["sequential"].params) == (2, 96)
    for spec in specs.values():
        if spec.contract:
            assert abs(spec.params - 108) <= 10.8
    assert specs["mlp_large"].width == 64
    assert specs["gnn_large"].width == 16
    assert specs["transformer_large"].width == 16


def test_hand_gates_are_realisable_on_every_item() -> None:
    """Check that hand gates are realisable on every item."""
    assert probe.realisability_accuracy() == 1.0


def test_verdict_passes_a_clear_substrate_win() -> None:
    """Check that verdict passes a clear substrate win."""
    results = [_result(seed, 0.95, _LOW) for seed in range(5)]
    outcome = probe.verdict(results, chance=0.25)
    assert outcome["pass"] is True
    assert outcome["best_contract_baseline"] == "gnn"
    assert outcome["margin_over_best_contract"] == pytest.approx(0.55)
    assert outcome["best_diagnostic"] == "gnn_large"
    assert outcome["staged_model_within_attribution_margin"] == {"gnn": False, "sequential": False}


def test_verdict_fails_when_a_contract_baseline_is_within_the_margin() -> None:
    """Check that verdict fails when a contract baseline is within the margin."""
    close = dict(_LOW, transformer=0.90)
    outcome = probe.verdict([_result(seed, 0.95, close) for seed in range(5)], chance=0.25)
    assert outcome["pass"] is False
    assert outcome["best_contract_baseline"] == "transformer"


def test_verdict_fails_when_the_substrate_is_not_clearly_above_chance() -> None:
    """Check that verdict fails when the substrate is not clearly above chance."""
    scattered = [
        _result(seed, score, dict.fromkeys(_LOW, 0.0))
        for seed, score in enumerate([0.0, 0.0, 0.0, 0.0, 0.8])
    ]
    outcome = probe.verdict(scattered, chance=0.25)
    assert outcome["pass"] is False


def test_attribution_flags_a_staged_model_near_the_substrate() -> None:
    """Check that attribution flags a staged model near the substrate."""
    near = dict(_LOW, sequential=0.90)
    outcome = probe.verdict([_result(seed, 0.95, near) for seed in range(5)], chance=0.25)
    assert outcome["pass"] is True
    assert outcome["staged_model_within_attribution_margin"]["sequential"] is True


@pytest.mark.parametrize("power", [None, 10.0])
def test_run_probe_emits_the_artefact(power: float | None) -> None:
    """Check that run probe emits the artefact."""
    artefact = probe.run_probe(seeds=(0,), epochs=1, nominal_power_w=power)
    assert artefact["schema"] == "kyma_v3_probe_result_v1"
    assert artefact["realisability_accuracy"] == 1.0
    assert artefact["substrate_params"] == 108
    assert artefact["design"]["ambiguous_fraction_by_query"] == (1.0, 0.0, 0.0)
    assert len(artefact["per_seed"]) == 1
    assert set(artefact["seconds_per_item"]) == {"substrate", *_LOW}
    assert artefact["verdict"]["chance_floor"] == 0.25
    if power is None:
        assert artefact["energy_proxy"] is None
    else:
        assert artefact["energy_proxy"]["nominal_power_w"] == 10.0
        assert set(artefact["energy_proxy"]["joules_per_item"]) == {"substrate", *_LOW}
