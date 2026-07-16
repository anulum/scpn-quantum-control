# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Reproducible closed-loop publication artifact run (RC-2)
"""Reproducible software-in-the-loop closed-loop publication artifact.

Packages the existing software-in-the-loop closed-loop surfaces into one reproducible,
honestly-labelled artifact:

* the **measured** software-in-the-loop latency report from
  :func:`~scpn_quantum_control.control.closed_loop_analysis.measure_closed_loop_latency_budget`
  (local controller/simulator wall-clock only — the latency budget is a
  *software budget/telemetry surface, not a hardware measurement*);
* the **publication scaffold** from
  :func:`~scpn_quantum_control.control.closed_loop_analysis.build_closed_loop_publication_package`
  whose claim ledger keeps provider-prepared and live-QPU claims blocked until
  real hardware artefacts exist;
* the **dynamic-circuit templates** (the monitored feedback circuit with real
  mid-circuit measurement + ``if_test`` conditionals, and its matched
  open-loop control) exported as OpenQASM 3 with content digests — these are
  *exportable but un-run*: no provider job IDs, no counts, no calibration
  snapshots;
* **provenance** (git commit, command line, dependency versions) and the
  host-isolation verdict grading the measured wall-clock samples.

Fail-closed honesty: nothing in the artifact promotes a hardware claim; the
claim ledger rows stay ``unpromoted``/``blocked`` exactly as the underlying
publication scaffold reports them.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Protocol, cast

from ..bridge.knm_hamiltonian import build_kuramoto_ring
from ..control.closed_loop_analysis import (
    ClosedLoopLatencyBudget,
    ClosedLoopLatencyReport,
    build_closed_loop_publication_package,
    measure_closed_loop_latency_budget,
)
from ..control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
    build_monitored_feedback_circuit,
    build_open_loop_feedback_control_circuit,
)
from .decisive_run_harness import command_line, dependency_versions, git_commit
from .isolated_host_readiness import HostReadiness, capture_host_readiness

SCHEMA_VERSION = "1.0"

_NON_HARDWARE_NOTE = (
    "the latency budget is a software budget/telemetry surface measured on a "
    "local statevector simulator; it is not a hardware measurement and makes "
    "no hardware latency claim"
)
_TEMPLATES_NOTE = (
    "dynamic-circuit templates are exportable but un-run: no provider job IDs, "
    "no raw counts, no calibration snapshots"
)


class FeedbackControllerLike(Protocol):
    """Structural API the closed-loop latency measurement needs."""

    config: RealtimeFeedbackConfig

    def reset(self) -> None:
        """Reset the controller state."""
        ...

    def step(self, seed: int | None = None) -> Any:
        """Run one feedback round and return its step record."""
        ...

    def run(self, n_steps: int, seed: int | None = None) -> list[Any]:
        """Run ``n_steps`` feedback rounds and return their step records."""
        ...


class LatencyMeasurer(Protocol):
    """Callable measuring the software-in-the-loop latency report."""

    def __call__(
        self,
        controller: FeedbackControllerLike,
        n_rounds: int,
        *,
        budget: ClosedLoopLatencyBudget | None,
        seed: int | None,
    ) -> ClosedLoopLatencyReport:
        """Return the latency report for ``n_rounds`` feedback rounds."""
        ...


class ControllerFactory(Protocol):
    """Callable building the feedback controller for the measured run."""

    def __call__(
        self,
        K: Any,
        omega: Any,
        *,
        config: RealtimeFeedbackConfig,
        trotter_order: int,
    ) -> FeedbackControllerLike:
        """Return a controller for the ring problem."""
        ...


@dataclass(frozen=True)
class ClosedLoopRunConfig:
    """Configuration of the reproducible closed-loop publication run.

    Parameters
    ----------
    n_oscillators
        Ring size of the Kuramoto problem driving the controller.
    coupling
        Nearest-neighbour ring coupling strength.
    target_r
        Order-parameter setpoint for the feedback controller.
    n_rounds
        Measured software-in-the-loop feedback rounds (at least two).
    dynamic_circuit_rounds
        Feedback rounds in the exported dynamic-circuit templates.
    seed
        Seed for the ring frequencies and the controller run.
    trotter_order
        Product-formula order of the controller and template circuits.
    reserved_core
        CPU core whose isolation state grades the measured wall-clock samples.
    budget
        Latency budget; ``None`` selects the
        :class:`~scpn_quantum_control.control.closed_loop_analysis.ClosedLoopLatencyBudget`
        defaults.
    """

    n_oscillators: int = 4
    coupling: float = 0.6
    target_r: float = 0.6
    n_rounds: int = 32
    dynamic_circuit_rounds: int = 3
    seed: int = 0
    trotter_order: int = 1
    reserved_core: int = 0
    budget: ClosedLoopLatencyBudget | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If the ring is smaller than two oscillators, the coupling or
            target is not finite and positive, fewer than two rounds are
            requested, or the template round count is not positive.
        """
        if self.n_oscillators < 2:
            raise ValueError("n_oscillators must be at least two")
        if not isfinite(self.coupling) or self.coupling <= 0.0:
            raise ValueError("coupling must be finite and positive")
        if not isfinite(self.target_r) or not 0.0 < self.target_r <= 1.0:
            raise ValueError("target_r must lie in (0, 1]")
        if self.n_rounds < 2:
            raise ValueError("n_rounds must be at least two for a latency verdict")
        if self.dynamic_circuit_rounds < 1:
            raise ValueError("dynamic_circuit_rounds must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""
        return {
            "n_oscillators": self.n_oscillators,
            "coupling": self.coupling,
            "target_r": self.target_r,
            "n_rounds": self.n_rounds,
            "dynamic_circuit_rounds": self.dynamic_circuit_rounds,
            "seed": self.seed,
            "trotter_order": self.trotter_order,
            "reserved_core": self.reserved_core,
            "budget": None if self.budget is None else asdict(self.budget),
        }


@dataclass(frozen=True)
class ClosedLoopPublicationArtifact:
    """Reproducible software-in-the-loop publication artifact with honest labelling."""

    package: dict[str, Any]
    latency_report: dict[str, Any]
    dynamic_circuit_templates: dict[str, Any]
    timing_grade: str
    host: dict[str, Any]
    config: dict[str, Any]
    provenance: dict[str, Any]
    notes: tuple[str, ...]
    package_markdown: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the artifact."""
        return {
            "schema_version": self.schema_version,
            "package": self.package,
            "latency_report": self.latency_report,
            "dynamic_circuit_templates": self.dynamic_circuit_templates,
            "timing_grade": self.timing_grade,
            "host": self.host,
            "config": self.config,
            "provenance": self.provenance,
            "notes": list(self.notes),
            "package_markdown": self.package_markdown,
        }


def _qasm3_template_entry(circuit: Any, template_id: str) -> dict[str, Any]:
    """Export one circuit as an OpenQASM 3 template entry with a digest."""
    from qiskit import qasm3

    program = qasm3.dumps(circuit)
    conditional_blocks = sum(
        1 for instruction in circuit.data if instruction.operation.name == "if_else"
    )
    return {
        "id": template_id,
        "format": "openqasm3",
        "n_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "conditional_blocks": conditional_blocks,
        "sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "program": program,
    }


def dynamic_circuit_templates(config: ClosedLoopRunConfig) -> dict[str, Any]:
    """Export the monitored-feedback and open-loop dynamic-circuit templates.

    Both templates share the ring problem and round count. The monitored
    template carries real mid-circuit measurement and ``if_test`` conditional
    corrections; the open-loop template is its matched control (same schedule,
    no feedback). They are exportable but un-run — the entry notes say so.

    Parameters
    ----------
    config
        Run configuration fixing the ring, rounds, and Trotter order.

    Returns
    -------
    dict of str to Any
        Mapping with the claim note and one OpenQASM 3 entry per template.
    """
    K, omega = build_kuramoto_ring(
        config.n_oscillators, coupling=config.coupling, rng_seed=config.seed
    )
    feedback_config = RealtimeFeedbackConfig(target_r=config.target_r)
    monitored = build_monitored_feedback_circuit(
        K,
        omega,
        config=feedback_config,
        n_rounds=config.dynamic_circuit_rounds,
        trotter_order=config.trotter_order,
    )
    open_loop = build_open_loop_feedback_control_circuit(
        K,
        omega,
        config=feedback_config,
        n_rounds=config.dynamic_circuit_rounds,
        trotter_order=config.trotter_order,
    )
    return {
        "claim_note": _TEMPLATES_NOTE,
        "monitored_feedback": _qasm3_template_entry(monitored, "monitored_feedback"),
        "open_loop_control": _qasm3_template_entry(open_loop, "open_loop_control"),
    }


def _default_latency_measurer(
    controller: FeedbackControllerLike,
    n_rounds: int,
    *,
    budget: ClosedLoopLatencyBudget | None,
    seed: int | None,
) -> ClosedLoopLatencyReport:
    """Measure the software-in-the-loop latency report with the live wall clock."""
    return measure_closed_loop_latency_budget(
        cast(RealtimeSyncFeedbackController, controller), n_rounds, budget=budget, seed=seed
    )


def _default_controller_factory(
    K: Any,
    omega: Any,
    *,
    config: RealtimeFeedbackConfig,
    trotter_order: int,
) -> RealtimeSyncFeedbackController:
    """Build the real statevector-simulator feedback controller."""
    return RealtimeSyncFeedbackController(K, omega, config=config, trotter_order=trotter_order)


def run_closed_loop_publication(
    config: ClosedLoopRunConfig | None = None,
    *,
    host_readiness: HostReadiness | None = None,
    latency_measurer: LatencyMeasurer = _default_latency_measurer,
    controller_factory: ControllerFactory = _default_controller_factory,
) -> ClosedLoopPublicationArtifact:
    """Run the reproducible software-in-the-loop closed-loop publication package.

    Parameters
    ----------
    config
        Run configuration; ``None`` selects :class:`ClosedLoopRunConfig`
        defaults.
    host_readiness
        Pre-captured host-isolation verdict; when ``None`` the live host is
        assessed via :func:`~.isolated_host_readiness.capture_host_readiness`.
    latency_measurer
        software-in-the-loop latency callable (injectable for tests); defaults to the live
        wall-clock measurement.
    controller_factory
        Controller builder (injectable for tests); defaults to the real
        statevector-simulator controller.

    Returns
    -------
    ClosedLoopPublicationArtifact
        The publication scaffold, the measured latency report, the exported
        dynamic-circuit templates, provenance, and the honest labels.
    """
    config = config or ClosedLoopRunConfig()

    K, omega = build_kuramoto_ring(
        config.n_oscillators, coupling=config.coupling, rng_seed=config.seed
    )
    controller = controller_factory(
        K,
        omega,
        config=RealtimeFeedbackConfig(target_r=config.target_r),
        trotter_order=config.trotter_order,
    )
    report = latency_measurer(controller, config.n_rounds, budget=config.budget, seed=config.seed)
    package = build_closed_loop_publication_package(latency_report=report)
    templates = dynamic_circuit_templates(config)

    readiness = host_readiness or capture_host_readiness(config.reserved_core)
    timing_grade = "isolated_measured" if readiness.ready else "advisory_shared_host"
    notes = [_NON_HARDWARE_NOTE, _TEMPLATES_NOTE]
    if not readiness.ready:
        notes.append("wall-clock samples measured on a shared host: advisory only")

    return ClosedLoopPublicationArtifact(
        package=package.to_dict(),
        latency_report=report.to_dict(),
        dynamic_circuit_templates=templates,
        timing_grade=timing_grade,
        host=asdict(readiness),
        config=config.to_dict(),
        provenance={
            "git_commit": git_commit(),
            "command": command_line(),
            "dependencies": dependency_versions(),
        },
        notes=tuple(notes),
        package_markdown=package.to_markdown(),
    )
