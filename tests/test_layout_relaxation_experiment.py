# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KT-4 relaxation seed-sweep experiment
"""Multi-angle tests for benchmarks/layout_relaxation_experiment.py.

Dimensions: the preregistered instance set, input validation (fail-closed on
pinned search configurations and empty sweeps), the stubbed end-to-end run
(budget binding, outcome extraction, honest labelling), every verdict branch
via a monkeypatched comparison run, aggregation arithmetic against
``statistics``, and artifact serialisation. All providers are injected or
stubbed, so nothing here touches transpilation or the NumPy coverage tracer.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import layout_relaxation_experiment as experiment
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness
from scpn_quantum_control.benchmarks.layout_method_comparison import (
    LayoutComparisonArtifact,
    LayoutComparisonConfig,
    MethodRow,
    RoutedLayoutMetrics,
)
from scpn_quantum_control.benchmarks.layout_relaxation_experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    PREREGISTRATION_REFERENCE,
    InstanceOutcome,
    RelaxationExperimentInstance,
    _verdict,
    preregistered_instances,
    run_layout_relaxation_experiment,
)
from scpn_quantum_control.hardware.kuramoto_layout_optimiser import LayoutSearchConfig
from scpn_quantum_control.hardware.kuramoto_layout_relaxation import (
    RESEARCH_LABEL,
    SinkhornRelaxationConfig,
)

_N = 3
_K = np.ones((_N, _N)) - np.eye(_N)
_OMEGA = np.array([0.1, 0.2, 0.3])

#: Two well-connected triangles bridged at (2, 4): DynQ picks the low-error one.
_GATE_ERRORS = {
    (0, 1): 0.001,
    (1, 2): 0.002,
    (0, 2): 0.001,
    (2, 4): 0.08,
    (4, 5): 0.02,
    (5, 6): 0.03,
    (4, 6): 0.02,
}
_READOUT = {qubit: 0.01 for qubit in range(7)}

_INSTANCES = (
    RelaxationExperimentInstance("two_cluster_seed3", 3, "dynq_region"),
    RelaxationExperimentInstance("full_device_seed3", 3, "full_device"),
)


def _ready_host(ready: bool) -> HostReadiness:
    return HostReadiness(
        ready=ready,
        reserved_core=0,
        governor="performance" if ready else "powersave",
        governor_is_stable=ready,
        frequency_mhz=3000.0,
        load_average=(0.1, 0.1, 0.1),
        load_is_low=True,
        blockers=() if ready else ("governor is powersave",),
    )


def _stub_metrics(
    K: Any,
    omega: Any,
    gate_errors: Any,
    *,
    t: float,
    reps: int,
    initial_layout: tuple[int, ...] | None,
    layout_method: str | None,
    optimization_level: int,
    seed: int,
) -> tuple[RoutedLayoutMetrics, tuple[int, ...]]:
    if initial_layout is not None:
        return RoutedLayoutMetrics(10, 6, 0.9), tuple(initial_layout)
    return RoutedLayoutMetrics(8, 5, 0.4), (4, 5, 6)


def _stub_r(K: Any, omega: Any, *, t: float, reps: int) -> float:
    return 0.5


def _stub_depth(
    layout: tuple[int, ...],
    K: Any,
    omega: Any,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
) -> int:
    return int(sum(layout))


def _run_stubbed(
    *,
    ready: bool = True,
    instances: tuple[RelaxationExperimentInstance, ...] | None = _INSTANCES,
    base_config: LayoutComparisonConfig | None = None,
) -> Any:
    return run_layout_relaxation_experiment(
        _GATE_ERRORS,
        _K,
        _OMEGA,
        readout_errors=_READOUT,
        base_config=base_config,
        instances=instances,
        host_readiness=_ready_host(ready),
        r_provider=_stub_r,
        metrics_provider=_stub_metrics,
        depth_provider=_stub_depth,
    )


class TestPreregisteredInstances:
    def test_default_set_is_sweep_plus_full_device(self) -> None:
        instances = preregistered_instances()
        assert len(instances) == 11
        for seed, instance in enumerate(instances[:10]):
            assert instance == RelaxationExperimentInstance(
                f"two_cluster_seed{seed}", seed, "dynq_region"
            )
        assert instances[-1] == RelaxationExperimentInstance("full_device_seed0", 0, "full_device")

    def test_custom_seeds_and_full_device_seed(self) -> None:
        instances = preregistered_instances(seeds=(4, 8), full_device_seed=2)
        assert [instance.seed for instance in instances] == [4, 8, 2]
        assert instances[-1].candidate_region == "full_device"

    def test_empty_seeds_rejected(self) -> None:
        with pytest.raises(ValueError, match="seeds must not be empty"):
            preregistered_instances(seeds=())

    def test_instance_to_dict(self) -> None:
        instance = RelaxationExperimentInstance("two_cluster_seed1", 1, "dynq_region")
        assert instance.to_dict() == {
            "label": "two_cluster_seed1",
            "seed": 1,
            "candidate_region": "dynq_region",
        }


class TestValidation:
    def test_pinned_search_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be None"):
            _run_stubbed(base_config=LayoutComparisonConfig(search=LayoutSearchConfig()))

    def test_pinned_relaxation_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be None"):
            _run_stubbed(base_config=LayoutComparisonConfig(relaxation=SinkhornRelaxationConfig()))

    def test_empty_instances_rejected(self) -> None:
        with pytest.raises(ValueError, match="instances must not be empty"):
            _run_stubbed(instances=())


class TestStubbedRun:
    def test_outcomes_follow_instances(self) -> None:
        artifact = _run_stubbed()
        assert [outcome.label for outcome in artifact.outcomes] == [
            "two_cluster_seed3",
            "full_device_seed3",
        ]
        assert [outcome.candidate_region for outcome in artifact.outcomes] == [
            "dynq_region",
            "full_device",
        ]
        assert all(outcome.seed == 3 for outcome in artifact.outcomes)

    def test_budget_binding_extracted_per_instance(self) -> None:
        artifact = _run_stubbed()
        for outcome in artifact.outcomes:
            assert outcome.budget >= 1
            assert 1 <= outcome.relaxation_true_evaluations <= outcome.budget
            assert outcome.cost_delta == pytest.approx(
                outcome.relaxation_cost - outcome.baseline_cost
            )

    def test_aggregate_matches_outcomes(self) -> None:
        artifact = _run_stubbed()
        baseline = [outcome.baseline_cost for outcome in artifact.outcomes]
        relaxation = [outcome.relaxation_cost for outcome in artifact.outcomes]
        assert artifact.baseline_mean_cost == pytest.approx(mean(baseline))
        assert artifact.relaxation_mean_cost == pytest.approx(mean(relaxation))
        assert artifact.baseline_cost_std == pytest.approx(pstdev(baseline))
        assert artifact.relaxation_cost_std == pytest.approx(pstdev(relaxation))
        assert artifact.wins + artifact.ties + artifact.losses == len(artifact.outcomes)

    def test_research_labelling_and_references(self) -> None:
        artifact = _run_stubbed()
        assert artifact.research_label == RESEARCH_LABEL
        assert artifact.preregistration == PREREGISTRATION_REFERENCE
        assert RESEARCH_LABEL in artifact.notes
        assert any("not a hardware measurement" in note for note in artifact.notes)
        assert any("budget match" in note for note in artifact.notes)

    def test_config_records_base_and_instances(self) -> None:
        artifact = _run_stubbed()
        assert artifact.config["base"]["include_relaxation"] is False
        assert [entry["label"] for entry in artifact.config["instances"]] == [
            "two_cluster_seed3",
            "full_device_seed3",
        ]
        assert sorted(artifact.provenance) == ["command", "dependencies", "git_commit"]

    def test_isolated_host_grades_timings(self) -> None:
        artifact = _run_stubbed(ready=True)
        assert artifact.timing_grade == "isolated_measured"
        assert not any("advisory" in note for note in artifact.notes)

    def test_shared_host_labels_timings_advisory(self) -> None:
        artifact = _run_stubbed(ready=False)
        assert artifact.timing_grade == "advisory_shared_host"
        assert any("advisory" in note for note in artifact.notes)
        assert artifact.host["ready"] is False

    def test_to_dict_and_table(self) -> None:
        artifact = _run_stubbed()
        payload = artifact.to_dict()
        assert payload["schema_version"] == EXPERIMENT_SCHEMA_VERSION
        assert payload["preregistration"] == PREREGISTRATION_REFERENCE
        assert len(payload["outcomes"]) == 2
        assert isinstance(payload["null_hypothesis_rejected"], bool)
        table = artifact.render_markdown_table()
        assert table.splitlines()[0].startswith("| Instance ")
        assert "two_cluster_seed3" in table
        assert any(word in table for word in ("win", "tie", "loss"))

    def test_live_host_captured_when_not_injected(self) -> None:
        artifact = run_layout_relaxation_experiment(
            _GATE_ERRORS,
            _K,
            _OMEGA,
            readout_errors=_READOUT,
            instances=_INSTANCES[:1],
            r_provider=_stub_r,
            metrics_provider=_stub_metrics,
            depth_provider=_stub_depth,
        )
        assert artifact.timing_grade in {"isolated_measured", "advisory_shared_host"}
        assert "ready" in artifact.host


def _fake_comparison(
    baseline_cost: float, relaxation_cost: float, budget: int
) -> LayoutComparisonArtifact:
    def row(method: str, depth: int, success: float) -> MethodRow:
        return MethodRow(
            method=method,
            layout=(0, 1, 2),
            routed_depth=depth,
            two_qubit_gates=6,
            estimated_success_probability=success,
            r_ideal=0.5,
            r_noisy_proxy=0.5 * success,
            selection_time_s=0.01,
            notes=("model",),
        )

    return LayoutComparisonArtifact(
        rows=(
            row("dynq", 10, 0.9),
            row("dynq+kuramoto_opt", 9, 0.91),
            row("sabre", 8, 0.4),
            row("dynq+sinkhorn_relaxation", 9, 0.91),
        ),
        r_ideal=0.5,
        timing_grade="isolated_measured",
        host={"ready": True},
        config={"seed": 0},
        provenance={
            "optimiser": {
                "best_layout": [0, 1, 2],
                "best_cost": {"total": baseline_cost},
                "n_evaluations": budget,
            },
            "relaxation": {
                "best_layout": [0, 1, 2],
                "best_cost": {"total": relaxation_cost},
                "n_true_evaluations": budget,
                "budget": budget,
            },
        },
        notes=("model",),
    )


class TestVerdictBranches:
    """Every aggregate verdict, via a monkeypatched comparison run."""

    @staticmethod
    def _run_with_costs(
        monkeypatch: pytest.MonkeyPatch,
        costs: list[tuple[float, float]],
        captured: list[dict[str, Any]] | None = None,
    ) -> Any:
        results = iter(costs)

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            if captured is not None:
                captured.append(kwargs)
            baseline_cost, relaxation_cost = next(results)
            return _fake_comparison(baseline_cost, relaxation_cost, budget=7)

        monkeypatch.setattr(experiment, "run_layout_method_comparison", fake_run)
        instances = tuple(
            RelaxationExperimentInstance(f"two_cluster_seed{index}", index, "dynq_region")
            for index in range(len(costs))
        )
        return run_layout_relaxation_experiment(
            _GATE_ERRORS,
            _K,
            _OMEGA,
            instances=instances,
            host_readiness=_ready_host(True),
        )

    def test_all_ties_keep_null_hypothesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = self._run_with_costs(monkeypatch, [(1.0, 1.0), (2.0, 2.0)])
        assert (artifact.wins, artifact.ties, artifact.losses) == (0, 2, 0)
        assert artifact.null_hypothesis_rejected is False
        assert artifact.verdict.startswith("no_gain")

    def test_consistent_wins_reject_null_hypothesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = self._run_with_costs(monkeypatch, [(1.0, 0.9), (2.0, 1.8)])
        assert (artifact.wins, artifact.ties, artifact.losses) == (2, 0, 0)
        assert artifact.null_hypothesis_rejected is True
        assert artifact.verdict.startswith("consistent_gain")

    def test_mean_gain_with_a_loss_is_inconsistent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = self._run_with_costs(monkeypatch, [(1.0, 0.5), (2.0, 2.1)])
        assert (artifact.wins, artifact.ties, artifact.losses) == (1, 0, 1)
        assert artifact.null_hypothesis_rejected is True
        assert artifact.verdict.startswith("inconsistent_gain")

    def test_balanced_wins_and_losses_keep_null_hypothesis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_with_costs(monkeypatch, [(1.0, 0.9), (2.0, 2.1)])
        assert (artifact.wins, artifact.ties, artifact.losses) == (1, 0, 1)
        assert artifact.null_hypothesis_rejected is False
        assert artifact.verdict.startswith("no_gain")

    def test_default_instances_used_when_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        results = [(1.0, 1.0)] * 11

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            captured.append(kwargs)
            baseline_cost, relaxation_cost = results[len(captured) - 1]
            return _fake_comparison(baseline_cost, relaxation_cost, budget=7)

        monkeypatch.setattr(experiment, "run_layout_method_comparison", fake_run)
        artifact = run_layout_relaxation_experiment(
            _GATE_ERRORS, _K, _OMEGA, host_readiness=_ready_host(True)
        )
        assert len(artifact.outcomes) == 11
        assert [kwargs["config"].seed for kwargs in captured] == [*range(10), 0]
        assert captured[-1]["config"].candidate_region == "full_device"
        assert all(kwargs["config"].include_relaxation for kwargs in captured)

    def test_table_marks_wins_and_losses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = self._run_with_costs(monkeypatch, [(1.0, 0.5), (2.0, 2.1)])
        table = artifact.render_markdown_table()
        assert "| win |" in table
        assert "| loss |" in table

    def test_readiness_shared_across_instances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        self._run_with_costs(monkeypatch, [(1.0, 1.0), (2.0, 2.0)], captured)
        readiness_objects = {id(kwargs["host_readiness"]) for kwargs in captured}
        assert len(readiness_objects) == 1


class TestVerdictFunction:
    def test_verdict_strings_cover_every_branch(self) -> None:
        assert _verdict(0, 3, False).startswith("no_gain")
        assert _verdict(3, 3, True).startswith("consistent_gain")
        assert _verdict(2, 3, True).startswith("inconsistent_gain")


class TestInstanceOutcomeSerialisation:
    def test_to_dict_round_trip(self) -> None:
        outcome = InstanceOutcome(
            label="two_cluster_seed0",
            seed=0,
            candidate_region="dynq_region",
            budget=7,
            baseline_cost=1.0,
            relaxation_cost=0.9,
            cost_delta=-0.1,
            baseline_layout=(0, 1, 2),
            relaxation_layout=(2, 1, 0),
            relaxation_true_evaluations=5,
            baseline_depth=9,
            relaxation_depth=8,
            baseline_success_probability=0.91,
            relaxation_success_probability=0.92,
        )
        payload = outcome.to_dict()
        assert payload["baseline_layout"] == [0, 1, 2]
        assert payload["relaxation_layout"] == [2, 1, 0]
        assert payload["cost_delta"] == pytest.approx(-0.1)
        assert payload["budget"] == 7
