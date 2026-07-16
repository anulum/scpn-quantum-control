# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KT-4 preregistered relaxation seed-sweep experiment
"""RESEARCH: the KT-4 preregistered seed-sweep experiment.

**Research label.** Runs the comparison protocol preregistered in
``docs/layout_relaxation_preregistration.md``
§4: on every preregistered instance, the annealed Sinkhorn relaxation
(:func:`~scpn_quantum_control.hardware.kuramoto_layout_relaxation.relax_kuramoto_layout`)
competes against the KT-3 discrete optimiser at a **matched budget of true
seeded KT-2 cost evaluations** (the binding is enforced inside
:func:`~scpn_quantum_control.benchmarks.layout_method_comparison.run_layout_method_comparison`).

The preregistered instance set is the two-cluster topology under seeds 0..9
(both arms searching the DynQ region) plus one full-device instance where the
candidate set is at least twice the logical width (``m ≥ 2n``, relocations
dominate). The decision metric is the **mean best true cost** across the
instances; the surrogate never enters the comparison.

Honest labelling
----------------
* The best true cost is the seeded KT-2 layout cost — a routed-depth
  measurement combined with Trotter-error and infidelity *models*; it is not
  a hardware measurement.
* The verdict is computed against the preregistered null hypothesis: *the
  relaxation does not beat the discrete baseline's mean best cost at matched
  budget*. "No gain" is a valid, publishable outcome.
* Nothing in this artifact promotes the relaxation beyond a research
  observation; promotion requires the owner-gated KT-5 isolated benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import mean, pstdev
from typing import Any

from ..hardware.kuramoto_layout_cost import DepthProvider, routed_layout_depth
from ..hardware.kuramoto_layout_relaxation import RESEARCH_LABEL
from .decisive_run_harness import command_line, dependency_versions, git_commit
from .isolated_host_readiness import HostReadiness, capture_host_readiness
from .layout_method_comparison import (
    FloatArray,
    GateErrors,
    LayoutComparisonArtifact,
    LayoutComparisonConfig,
    MetricsProvider,
    RProvider,
    ideal_xy_order_parameter,
    routed_layout_metrics,
    run_layout_method_comparison,
)

EXPERIMENT_SCHEMA_VERSION = "1.0"

#: Repository-relative path of the preregistered protocol this run executes.
PREREGISTRATION_REFERENCE = "docs/layout_relaxation_preregistration.md"

#: Preregistered seed sweep (design doc §4).
_SWEEP_SEEDS: tuple[int, ...] = tuple(range(10))

_TRUE_COST_NOTE = (
    "best true cost is the seeded KT-2 layout cost: a routed-depth measurement "
    "combined with Trotter-error and infidelity models, not a hardware measurement"
)
_BUDGET_NOTE = (
    "the relaxation's true-cost budget is bound per instance to the discrete "
    "optimiser's n_evaluations (the preregistered budget match)"
)


@dataclass(frozen=True)
class RelaxationExperimentInstance:
    """One preregistered experiment instance.

    Parameters
    ----------
    label
        Human-readable instance name carried into the artifact.
    seed
        Seed shared by DynQ, both search arms, and the transpiler.
    candidate_region
        Candidate set searched by both arms: ``"dynq_region"`` or
        ``"full_device"``.
    """

    label: str
    seed: int
    candidate_region: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the instance."""
        return {
            "label": self.label,
            "seed": self.seed,
            "candidate_region": self.candidate_region,
        }


def preregistered_instances(
    seeds: tuple[int, ...] = _SWEEP_SEEDS,
    full_device_seed: int = 0,
) -> tuple[RelaxationExperimentInstance, ...]:
    """Return the preregistered instance set of the KT-4 protocol.

    Parameters
    ----------
    seeds
        Seed sweep for the DynQ-region instances (design default ``0..9``).
    full_device_seed
        Seed of the single full-device (``m ≥ 2n``) instance.

    Returns
    -------
    tuple of RelaxationExperimentInstance
        The sweep instances followed by the full-device instance.

    Raises
    ------
    ValueError
        If ``seeds`` is empty — an empty sweep has no preregistered meaning.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    instances = [
        RelaxationExperimentInstance(f"two_cluster_seed{seed}", seed, "dynq_region")
        for seed in seeds
    ]
    instances.append(
        RelaxationExperimentInstance(
            f"full_device_seed{full_device_seed}", full_device_seed, "full_device"
        )
    )
    return tuple(instances)


@dataclass(frozen=True)
class InstanceOutcome:
    """Decision-relevant numbers extracted from one comparison run."""

    label: str
    seed: int
    candidate_region: str
    budget: int
    baseline_cost: float
    relaxation_cost: float
    cost_delta: float
    baseline_layout: tuple[int, ...]
    relaxation_layout: tuple[int, ...]
    relaxation_true_evaluations: int
    baseline_depth: int
    relaxation_depth: int
    baseline_success_probability: float
    relaxation_success_probability: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the outcome."""
        return {
            "label": self.label,
            "seed": self.seed,
            "candidate_region": self.candidate_region,
            "budget": self.budget,
            "baseline_cost": self.baseline_cost,
            "relaxation_cost": self.relaxation_cost,
            "cost_delta": self.cost_delta,
            "baseline_layout": list(self.baseline_layout),
            "relaxation_layout": list(self.relaxation_layout),
            "relaxation_true_evaluations": self.relaxation_true_evaluations,
            "baseline_depth": self.baseline_depth,
            "relaxation_depth": self.relaxation_depth,
            "baseline_success_probability": self.baseline_success_probability,
            "relaxation_success_probability": self.relaxation_success_probability,
        }


@dataclass(frozen=True)
class RelaxationExperimentArtifact:
    """Aggregated experiment result with the preregistered verdict."""

    outcomes: tuple[InstanceOutcome, ...]
    baseline_mean_cost: float
    relaxation_mean_cost: float
    baseline_cost_std: float
    relaxation_cost_std: float
    wins: int
    ties: int
    losses: int
    null_hypothesis_rejected: bool
    verdict: str
    timing_grade: str
    host: dict[str, Any]
    config: dict[str, Any]
    provenance: dict[str, Any]
    notes: tuple[str, ...]
    research_label: str = RESEARCH_LABEL
    preregistration: str = PREREGISTRATION_REFERENCE
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the artifact."""
        return {
            "schema_version": self.schema_version,
            "research_label": self.research_label,
            "preregistration": self.preregistration,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "baseline_mean_cost": self.baseline_mean_cost,
            "relaxation_mean_cost": self.relaxation_mean_cost,
            "baseline_cost_std": self.baseline_cost_std,
            "relaxation_cost_std": self.relaxation_cost_std,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "null_hypothesis_rejected": self.null_hypothesis_rejected,
            "verdict": self.verdict,
            "timing_grade": self.timing_grade,
            "host": self.host,
            "config": self.config,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }

    def render_markdown_table(self) -> str:
        """Render the per-instance outcomes as a GitHub-flavoured table."""
        lines = [
            "| Instance | Candidates | Budget | Baseline cost | Relaxation cost "
            "| Δ (relax − base) | Outcome |",
            "|---|---|---|---|---|---|---|",
        ]
        for outcome in self.outcomes:
            if outcome.cost_delta < 0.0:
                result = "win"
            elif outcome.cost_delta > 0.0:
                result = "loss"
            else:
                result = "tie"
            lines.append(
                f"| {outcome.label} | {outcome.candidate_region} | {outcome.budget} "
                f"| {outcome.baseline_cost:.6f} | {outcome.relaxation_cost:.6f} "
                f"| {outcome.cost_delta:+.6f} | {result} |"
            )
        return "\n".join(lines)


def _instance_outcome(
    instance: RelaxationExperimentInstance, comparison: LayoutComparisonArtifact
) -> InstanceOutcome:
    """Extract the decision-relevant numbers from one comparison artifact."""
    rows = {row.method: row for row in comparison.rows}
    baseline_row = rows["dynq+kuramoto_opt"]
    relaxation_row = rows["dynq+sinkhorn_relaxation"]
    optimiser = comparison.provenance["optimiser"]
    relaxation = comparison.provenance["relaxation"]
    baseline_cost = float(optimiser["best_cost"]["total"])
    relaxation_cost = float(relaxation["best_cost"]["total"])
    return InstanceOutcome(
        label=instance.label,
        seed=instance.seed,
        candidate_region=instance.candidate_region,
        budget=int(relaxation["budget"]),
        baseline_cost=baseline_cost,
        relaxation_cost=relaxation_cost,
        cost_delta=relaxation_cost - baseline_cost,
        baseline_layout=tuple(optimiser["best_layout"]),
        relaxation_layout=tuple(relaxation["best_layout"]),
        relaxation_true_evaluations=int(relaxation["n_true_evaluations"]),
        baseline_depth=baseline_row.routed_depth,
        relaxation_depth=relaxation_row.routed_depth,
        baseline_success_probability=baseline_row.estimated_success_probability,
        relaxation_success_probability=relaxation_row.estimated_success_probability,
    )


def _verdict(wins: int, n_instances: int, null_hypothesis_rejected: bool) -> str:
    """Return the preregistered verdict string for the aggregate outcome."""
    if not null_hypothesis_rejected:
        return (
            "no_gain: the relaxation does not beat the discrete baseline's mean "
            "best true cost at matched budget (the preregistered null hypothesis stands)"
        )
    if wins == n_instances:
        return (
            "consistent_gain: the relaxation beats the discrete baseline on every "
            "instance at matched budget (research observation; promotion requires KT-5)"
        )
    return (
        "inconsistent_gain: the relaxation beats the discrete baseline on mean best "
        "true cost but not on every instance (research observation; promotion requires KT-5)"
    )


def run_layout_relaxation_experiment(
    gate_errors: GateErrors,
    K: FloatArray,
    omega: FloatArray,
    *,
    readout_errors: dict[int, float] | None = None,
    base_config: LayoutComparisonConfig | None = None,
    instances: tuple[RelaxationExperimentInstance, ...] | None = None,
    host_readiness: HostReadiness | None = None,
    r_provider: RProvider = ideal_xy_order_parameter,
    metrics_provider: MetricsProvider = routed_layout_metrics,
    depth_provider: DepthProvider = routed_layout_depth,
) -> RelaxationExperimentArtifact:
    """Run the preregistered KT-4 seed-sweep experiment.

    Every instance runs the full layout-method comparison with the research
    row enabled; the per-instance seed and candidate region come from the
    instance, so ``base_config`` must leave both search configurations unset
    (they are derived per instance from the swept seed).

    Parameters
    ----------
    gate_errors
        Per-edge two-qubit calibration errors defining the coupling map.
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    readout_errors
        Optional per-qubit readout errors for the DynQ baseline.
    base_config
        Shared run settings (``t``, ``reps``, ``order``, levels, DynQ knobs);
        ``None`` selects the :class:`~.layout_method_comparison.LayoutComparisonConfig`
        defaults. Its ``seed``, ``candidate_region``, and
        ``include_relaxation`` fields are overridden per instance.
    instances
        Experiment instances; ``None`` selects :func:`preregistered_instances`.
    host_readiness
        Pre-captured host-isolation verdict shared by every instance; when
        ``None`` the live host is assessed once via
        :func:`~.isolated_host_readiness.capture_host_readiness`.
    r_provider, metrics_provider, depth_provider
        Injectable providers forwarded to every comparison run.

    Returns
    -------
    RelaxationExperimentArtifact
        Per-instance outcomes, the aggregate statistics, and the verdict
        against the preregistered null hypothesis.

    Raises
    ------
    ValueError
        If ``base_config`` pins an explicit search or relaxation
        configuration (the sweep could then not bind the per-instance seed),
        or ``instances`` is empty, or any comparison run fails closed.
    """
    base = base_config or LayoutComparisonConfig()
    if base.search is not None or base.relaxation is not None:
        raise ValueError(
            "base_config.search and base_config.relaxation must be None: "
            "the sweep binds each instance's seed into both search arms"
        )
    if instances is None:
        instances = preregistered_instances()
    if not instances:
        raise ValueError("instances must not be empty")

    readiness = host_readiness or capture_host_readiness(base.reserved_core)

    outcomes: list[InstanceOutcome] = []
    for instance in instances:
        config = replace(
            base,
            seed=instance.seed,
            candidate_region=instance.candidate_region,
            include_relaxation=True,
        )
        comparison = run_layout_method_comparison(
            gate_errors,
            K,
            omega,
            readout_errors=readout_errors,
            config=config,
            host_readiness=readiness,
            r_provider=r_provider,
            metrics_provider=metrics_provider,
            depth_provider=depth_provider,
        )
        outcomes.append(_instance_outcome(instance, comparison))

    baseline_costs = [outcome.baseline_cost for outcome in outcomes]
    relaxation_costs = [outcome.relaxation_cost for outcome in outcomes]
    baseline_mean = mean(baseline_costs)
    relaxation_mean = mean(relaxation_costs)
    wins = sum(1 for outcome in outcomes if outcome.cost_delta < 0.0)
    losses = sum(1 for outcome in outcomes if outcome.cost_delta > 0.0)
    ties = len(outcomes) - wins - losses
    null_hypothesis_rejected = relaxation_mean < baseline_mean

    timing_grade = "isolated_measured" if readiness.ready else "advisory_shared_host"
    notes = [RESEARCH_LABEL, _TRUE_COST_NOTE, _BUDGET_NOTE]
    if not readiness.ready:
        notes.append("selection wall-times measured on a shared host: advisory only")

    return RelaxationExperimentArtifact(
        outcomes=tuple(outcomes),
        baseline_mean_cost=baseline_mean,
        relaxation_mean_cost=relaxation_mean,
        baseline_cost_std=pstdev(baseline_costs),
        relaxation_cost_std=pstdev(relaxation_costs),
        wins=wins,
        ties=ties,
        losses=losses,
        null_hypothesis_rejected=null_hypothesis_rejected,
        verdict=_verdict(wins, len(outcomes), null_hypothesis_rejected),
        timing_grade=timing_grade,
        host=asdict(readiness),
        config={
            "base": base.to_dict(),
            "instances": [instance.to_dict() for instance in instances],
        },
        provenance={
            "git_commit": git_commit(),
            "command": command_line(),
            "dependencies": dependency_versions(),
        },
        notes=tuple(notes),
    )
