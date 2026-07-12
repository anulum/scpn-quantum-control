# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable core module
# scpn-quantum-control -- stable core contracts
"""Stable first-path contracts for SCPN quantum-control workflows.

The contracts in this module are intentionally small. They define the durable
shape that higher-level compilers, backend adapters, benchmark harnesses, and
hardware result-pack replay paths can share without depending on low-level
module layout.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .kuramoto_core import KuramotoProblem

ProblemKind = Literal["kuramoto_xy"]
BackendKind = Literal[
    "classical_reference",
    "qiskit",
    "qutip",
    "pennylane",
    "pulser_surrogate",
    "hardware_replay",
]
Capability = Literal[
    "order_parameter",
    "parity",
    "mitigation_replay",
    "fim",
    "control",
    "hamiltonian_dynamics",
    "lindblad",
    "pulse_schedule",
    "autodiff",
    "analog_surrogate",
]
STABLE_CORE_CAPABILITY_SCHEMA = "stable_core_backend_capability_matrix_v1"
ExperimentObjective = Literal[
    "order_parameter",
    "parity_leakage",
    "fim_metric",
    "control_cost",
    "mitigation_replay",
]
ResultStatus = Literal["succeeded", "blocked", "failed"]


def _metadata_copy(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return immutable metadata with string keys."""
    if metadata is None:
        return MappingProxyType({})
    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        copied[key] = value
    return MappingProxyType(copied)


def _empty_metadata() -> Mapping[str, Any]:
    """Return a fresh immutable empty metadata mapping for dataclass defaults."""
    return MappingProxyType({})


def _normalise_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Return a finite square coupling matrix."""
    if not matrix:
        raise ValueError("coupling_matrix must not be empty")
    width = len(matrix)
    rows: list[tuple[float, ...]] = []
    for row in matrix:
        if len(row) != width:
            raise ValueError("coupling_matrix must be square")
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Problem:
    """Stable domain problem contract."""

    problem_id: str
    kind: ProblemKind
    n_qubits: int
    coupling_matrix: tuple[tuple[float, ...], ...]
    omega: tuple[float, ...]
    initial_state: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate stable problem invariants."""
        if not self.problem_id:
            raise ValueError("problem_id must not be empty")
        if self.kind != "kuramoto_xy":
            raise ValueError(f"unsupported problem kind: {self.kind!r}")
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        matrix = _normalise_matrix(self.coupling_matrix)
        if len(matrix) != self.n_qubits:
            raise ValueError("coupling_matrix dimension must match n_qubits")
        omega = tuple(float(value) for value in self.omega)
        if len(omega) != self.n_qubits:
            raise ValueError("omega length must match n_qubits")
        if self.initial_state is not None and (
            len(self.initial_state) != self.n_qubits
            or any(bit not in {"0", "1"} for bit in self.initial_state)
        ):
            raise ValueError("initial_state must be a computational-basis bitstring")
        object.__setattr__(self, "coupling_matrix", matrix)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible problem payload."""
        return {
            "problem_id": self.problem_id,
            "kind": self.kind,
            "n_qubits": self.n_qubits,
            "coupling_matrix": self.coupling_matrix,
            "omega": self.omega,
            "initial_state": self.initial_state,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Backend:
    """Stable backend capability contract."""

    backend_id: str
    kind: BackendKind
    capabilities: tuple[str, ...]
    hardware_submission_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate stable backend invariants."""
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")
        if not self.capabilities:
            raise ValueError("backend capabilities must not be empty")
        if self.hardware_submission_allowed and self.kind != "qiskit":
            raise ValueError("hardware submission is only allowed for explicit qiskit backends")
        object.__setattr__(self, "capabilities", tuple(str(item) for item in self.capabilities))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    def supports(self, capability: str) -> bool:
        """Return whether the backend declares one capability."""
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible backend payload."""
        return {
            "backend_id": self.backend_id,
            "kind": self.kind,
            "capabilities": self.capabilities,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Experiment:
    """Stable experiment contract tying a problem to a backend and objective."""

    experiment_id: str
    problem: Problem
    backend: Backend
    objective: ExperimentObjective
    seed: int
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate stable experiment invariants."""
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        required = {
            "order_parameter": "order_parameter",
            "parity_leakage": "parity",
            "fim_metric": "fim",
            "control_cost": "control",
            "mitigation_replay": "mitigation_replay",
        }[self.objective]
        if not self.backend.supports(required):
            raise ValueError(
                f"backend {self.backend.backend_id!r} does not support objective "
                f"{self.objective!r}"
            )
        if self.backend.hardware_submission_allowed and "preregistration_id" not in self.metadata:
            raise ValueError("hardware submission experiments require preregistration_id metadata")
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible experiment payload."""
        return {
            "experiment_id": self.experiment_id,
            "problem": self.problem.to_dict(),
            "backend": self.backend.to_dict(),
            "objective": self.objective,
            "seed": self.seed,
            "shots": self.shots,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Result:
    """Stable result contract for experiment outputs."""

    experiment_id: str
    backend_id: str
    status: ResultStatus
    observables: Mapping[str, float]
    artifacts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        """Validate stable result invariants."""
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")
        if self.status == "succeeded" and not self.observables:
            raise ValueError("succeeded results must contain observables")
        if self.status in {"blocked", "failed"} and not self.blockers:
            raise ValueError("blocked or failed results must contain blockers")
        normalised_observables = {
            str(key): float(value) for key, value in self.observables.items()
        }
        object.__setattr__(self, "observables", MappingProxyType(normalised_observables))
        object.__setattr__(self, "artifacts", tuple(str(item) for item in self.artifacts))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result payload."""
        return {
            "experiment_id": self.experiment_id,
            "backend_id": self.backend_id,
            "status": self.status,
            "observables": dict(self.observables),
            "artifacts": self.artifacts,
            "blockers": self.blockers,
            "metadata": dict(self.metadata),
        }


def build_problem(
    *,
    problem_id: str,
    coupling_matrix: tuple[tuple[float, ...], ...],
    omega: tuple[float, ...],
    initial_state: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Problem:
    """Build a stable Kuramoto/XY problem contract."""
    return Problem(
        problem_id=problem_id,
        kind="kuramoto_xy",
        n_qubits=len(omega),
        coupling_matrix=coupling_matrix,
        omega=omega,
        initial_state=initial_state,
        metadata=_metadata_copy(metadata),
    )


def build_backend(
    *,
    backend_id: str,
    kind: BackendKind,
    capabilities: tuple[str, ...],
    hardware_submission_allowed: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build a stable backend capability contract."""
    return Backend(
        backend_id=backend_id,
        kind=kind,
        capabilities=capabilities,
        hardware_submission_allowed=hardware_submission_allowed,
        metadata=_metadata_copy(metadata),
    )


def classical_reference_backend(
    backend_id: str = "classical-reference",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable no-QPU classical-reference backend profile."""
    return build_backend(
        backend_id=backend_id,
        kind="classical_reference",
        capabilities=("order_parameter", "parity", "fim", "control"),
        metadata=metadata,
    )


def hardware_replay_backend(
    backend_id: str = "hardware-replay",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable no-submit hardware-result replay backend profile."""
    return build_backend(
        backend_id=backend_id,
        kind="hardware_replay",
        capabilities=("order_parameter", "parity", "mitigation_replay"),
        metadata=metadata,
    )


def qiskit_backend(
    backend_id: str = "qiskit-runtime",
    *,
    hardware_submission_allowed: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable Qiskit backend profile.

    Hardware submission remains disabled by default. Callers must opt in and
    still provide experiment preregistration metadata before a hardware-enabled
    experiment can be built.
    """
    return build_backend(
        backend_id=backend_id,
        kind="qiskit",
        capabilities=("order_parameter", "parity", "mitigation_replay"),
        hardware_submission_allowed=hardware_submission_allowed,
        metadata=metadata,
    )


def qutip_backend(
    backend_id: str = "qutip-dynamics",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable QuTiP/open-system dynamics backend profile."""
    return build_backend(
        backend_id=backend_id,
        kind="qutip",
        capabilities=("order_parameter", "hamiltonian_dynamics", "lindblad"),
        metadata=metadata,
    )


def pennylane_backend(
    backend_id: str = "pennylane-autodiff",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable PennyLane/autodiff backend profile."""
    return build_backend(
        backend_id=backend_id,
        kind="pennylane",
        capabilities=("order_parameter", "parity", "control", "autodiff"),
        metadata=metadata,
    )


def pulser_surrogate_backend(
    backend_id: str = "pulser-surrogate",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Backend:
    """Build the stable Pulser-surrogate analog backend profile."""
    return build_backend(
        backend_id=backend_id,
        kind="pulser_surrogate",
        capabilities=("order_parameter", "analog_surrogate", "pulse_schedule"),
        metadata=metadata,
    )


def backend_capability_matrix() -> tuple[dict[str, Any], ...]:
    """Return the stable backend capability matrix.

    The matrix is intentionally descriptive. It records declared capability
    profiles and hardware-submission boundaries; it does not imply that every
    adapter implementation is complete.
    """
    rows: list[dict[str, Any]] = []
    for backend in (
        classical_reference_backend(),
        hardware_replay_backend(),
        qiskit_backend(),
        qutip_backend(),
        pennylane_backend(),
        pulser_surrogate_backend(),
    ):
        rows.append(
            {
                "backend_id": backend.backend_id,
                "kind": backend.kind,
                "capabilities": backend.capabilities,
                "hardware_submission_allowed": backend.hardware_submission_allowed,
                "claim_boundary": (
                    "Capability profile only; adapter implementation, dependency checks, "
                    "and evidence gates remain required before execution claims."
                ),
            }
        )
    return tuple(rows)


def stable_core_capability_payload() -> dict[str, Any]:
    """Return a JSON-compatible stable core capability payload."""
    payload = {
        "schema": STABLE_CORE_CAPABILITY_SCHEMA,
        "hardware_submission": False,
        "rows": [dict(row) for row in backend_capability_matrix()],
        "claim_boundary": (
            "This payload records stable backend capability profiles only. "
            "It does not prove adapter implementation, dependency availability, "
            "or hardware execution readiness."
        ),
    }
    normalised: dict[str, Any] = json.loads(json.dumps(payload, sort_keys=True))
    return normalised


def normalised_stable_core_json(data: dict[str, Any]) -> str:
    """Return deterministic JSON text for stable core capability artifacts."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def stable_core_capability_markdown(data: dict[str, Any]) -> str:
    """Render a public backend capability matrix summary."""
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- stable core backend capability matrix -->",
        "",
        "# Stable Core Backend Capability Matrix",
        "",
        "This generated page records stable backend capability profiles.",
        "",
        "| Backend | Kind | Capabilities | Hardware submission | Claim boundary |",
        "|---|---|---|---|---|",
    ]
    for row in data["rows"]:
        lines.append(
            "| `{backend}` | `{kind}` | {capabilities} | `{submission}` | {boundary} |".format(
                backend=row["backend_id"],
                kind=row["kind"],
                capabilities=", ".join(row["capabilities"]),
                submission=row["hardware_submission_allowed"],
                boundary=row["claim_boundary"],
            )
        )
    lines.extend(
        [
            "",
            "## Payload boundary",
            "",
            str(data["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def write_stable_core_capability_artifacts(
    *,
    json_path: Path,
    doc_path: Path,
) -> dict[str, str]:
    """Write deterministic stable core capability artifacts and return digests."""
    payload = stable_core_capability_payload()
    json_text = normalised_stable_core_json(payload)
    doc_text = stable_core_capability_markdown(payload)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    doc_path.write_text(doc_text, encoding="utf-8")
    return {
        "json_sha256": hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
        "doc_sha256": hashlib.sha256(doc_text.encode("utf-8")).hexdigest(),
    }


def build_experiment(
    *,
    experiment_id: str,
    problem: Problem,
    backend: Backend,
    objective: ExperimentObjective,
    seed: int,
    shots: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Experiment:
    """Build a stable experiment contract."""
    return Experiment(
        experiment_id=experiment_id,
        problem=problem,
        backend=backend,
        objective=objective,
        seed=seed,
        shots=shots,
        metadata=_metadata_copy(metadata),
    )


def build_result(
    *,
    experiment_id: str,
    backend_id: str,
    status: ResultStatus,
    observables: Mapping[str, float],
    artifacts: tuple[str, ...] = (),
    blockers: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Result:
    """Build a stable result contract."""
    return Result(
        experiment_id=experiment_id,
        backend_id=backend_id,
        status=status,
        observables=observables,
        artifacts=artifacts,
        blockers=blockers,
        metadata=_metadata_copy(metadata),
    )


def problem_from_kuramoto(
    kuramoto_problem: KuramotoProblem,
    *,
    problem_id: str,
    initial_state: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Problem:
    """Convert a Kuramoto facade problem into the stable problem contract."""
    merged_metadata = dict(kuramoto_problem.metadata)
    if metadata:
        merged_metadata.update(metadata)
    return build_problem(
        problem_id=problem_id,
        coupling_matrix=tuple(
            tuple(float(value) for value in row) for row in kuramoto_problem.K_nm.tolist()
        ),
        omega=tuple(float(value) for value in kuramoto_problem.omega.tolist()),
        initial_state=initial_state,
        metadata=merged_metadata,
    )


def problem_to_kuramoto(problem: Problem) -> KuramotoProblem:
    """Convert a stable problem contract into the Kuramoto facade problem."""
    from .kuramoto_core import build_kuramoto_problem

    return build_kuramoto_problem(
        np.asarray(problem.coupling_matrix, dtype=float),
        np.asarray(problem.omega, dtype=float),
        metadata=problem.metadata,
    )
