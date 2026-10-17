# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable algorithm design inputs
"""Typed dimensionless design objectives and declared backend snapshots."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ..canonical_encoding import canonical_digest
from ..hardware.hal import BackendProfile


@dataclass(frozen=True, slots=True)
class AlgorithmDesignProblem:
    """Bind a bounded Pauli objective to exact shared parameter identities.

    Parameters
    ----------
    source_ref
        SHA-256 of the source scientific problem; never an inferred model identity.
    observables
        One to 64 distinct, equal-width Pauli labels with finite real coefficients.
        Qiskit's rightmost character denotes logical qubit zero. Coefficients and
        the summed expectation are dimensionless in this supported profile.
    parameters
        One to 64 immutable Qiskit parameters, in caller evaluation order.
    parameter_values
        Finite real evaluation point in radians, matching ``parameters``.
    parameter_uses
        Required occurrence count of each parameter across gate expressions.
        Names alone do not identify parameters; their native UUIDs must match.
    symmetries
        Optional equal-width Pauli stabilisers and required eigenvalues, -1 or 1.
        These constrain the candidate state at the declared evaluation point.

    """

    source_ref: str
    observables: tuple[tuple[str, float], ...]
    parameters: tuple[Parameter, ...]
    parameter_values: tuple[float, ...]
    parameter_uses: tuple[int, ...]
    symmetries: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        """Refuse ambiguous identities, mutable containers and unbounded inputs."""
        if (
            not isinstance(self.source_ref, str)
            or len(self.source_ref) != 64
            or any(c not in "0123456789abcdef" for c in self.source_ref)
        ):
            raise ValueError("source_ref must be a lowercase SHA-256")
        if not isinstance(self.observables, tuple) or not 1 <= len(self.observables) <= 64:
            raise ValueError("observables must contain 1..64 immutable Pauli terms")
        if any(not isinstance(term, tuple) or len(term) != 2 for term in self.observables):
            raise ValueError("observable entries must be immutable label/coefficient pairs")
        if not isinstance(self.observables[0][0], str):
            raise ValueError("observable label must be text")
        width = len(self.observables[0][0])
        for label, coefficient in self.observables:
            _pauli(label, width)
            _finite(coefficient, "coefficient")
        if len({label for label, _ in self.observables}) != len(self.observables):
            raise ValueError("observable labels must be unique")
        if (
            not isinstance(self.parameters, tuple)
            or not 1 <= len(self.parameters) <= 64
            or any(not isinstance(p, Parameter) for p in self.parameters)
            or len(set(self.parameters)) != len(self.parameters)
            or len({p.name for p in self.parameters}) != len(self.parameters)
        ):
            raise ValueError("parameters must contain unique native identities and names")
        if (
            not isinstance(self.parameter_values, tuple)
            or not isinstance(self.parameter_uses, tuple)
            or len(self.parameter_values) != len(self.parameters)
            or len(self.parameter_uses) != len(self.parameters)
        ):
            raise ValueError("parameter values and uses must match parameter order")
        for value in self.parameter_values:
            _finite(value, "parameter value")
        if any(type(count) is not int or not 1 <= count <= 4096 for count in self.parameter_uses):
            raise ValueError("parameter uses must be positive bounded integers")
        if not isinstance(self.symmetries, tuple):
            raise ValueError("symmetries must be immutable")
        if any(not isinstance(term, tuple) or len(term) != 2 for term in self.symmetries):
            raise ValueError("symmetry entries must be immutable label/eigenvalue pairs")
        for label, sign in self.symmetries:
            _pauli(label, width)
            if type(sign) is not int or sign not in (-1, 1):
                raise ValueError("symmetry eigenvalue must be -1 or 1")

    @property
    def sha256(self) -> str:
        """Content identity including coefficients, UUIDs, sharing and constraints."""
        return canonical_digest(
            "algorithm_design_problem.v1",
            {
                "source_ref": self.source_ref,
                "observables": self.observables,
                "parameters": tuple((p.name, str(p.uuid)) for p in self.parameters),
                "parameter_values": self.parameter_values,
                "parameter_uses": self.parameter_uses,
                "symmetries": self.symmetries,
            },
        )


@dataclass(frozen=True, slots=True)
class DesignBackendSnapshot:
    """Freeze target declarations used for offline feasibility, without submission.

    Parameters
    ----------
    profile
        Existing immutable HAL route declaration, never observed hardware proof.
    snapshot_ref
        Explicit calibration/snapshot identity supplied by its actual owner.
    basis_gates
        Nonempty unique subset of rx, ry, rz, cx and cz for static native lowering.
    coupling_edges
        Immutable directed physical two-qubit edges, including reverse edges only
        where declared. An empty tuple admits no two-qubit gates.

    """

    profile: BackendProfile
    snapshot_ref: str
    basis_gates: tuple[str, ...]
    coupling_edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Validate the exact declared target without fetching provider metadata."""
        if not isinstance(self.profile, BackendProfile):
            raise ValueError("profile must be an existing HAL BackendProfile")
        if not isinstance(self.snapshot_ref, str) or not self.snapshot_ref.strip():
            raise ValueError("snapshot_ref must identify the actual declaration")
        if (
            not isinstance(self.basis_gates, tuple)
            or not self.basis_gates
            or len(set(self.basis_gates)) != len(self.basis_gates)
            or any(gate not in ("rx", "ry", "rz", "cx", "cz") for gate in self.basis_gates)
        ):
            raise ValueError("basis_gates must be a unique supported static basis")
        if not isinstance(self.coupling_edges, tuple):
            raise ValueError("coupling_edges must be immutable")
        for edge in self.coupling_edges:
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or any(type(q) is not int or not 0 <= q < 8 for q in edge)
                or edge[0] == edge[1]
            ):
                raise ValueError("coupling edge must contain distinct bounded qubits")

    @property
    def sha256(self) -> str:
        """Hash every HAL field, calibration reference, basis and connectivity."""
        return canonical_digest("algorithm_design_backend.v1", asdict(self))


@dataclass(frozen=True, slots=True)
class DesignCandidate:
    """One native ansatz input, copied before evaluation and absent from the plan.

    Parameters
    ----------
    candidate_id
        Nonempty comparison identity, unique within a selection request.
    circuit
        Native static parameterised circuit, at most eight qubits/4096 operations.
        The caller retains ownership; the planner never mutates this object.
    layout
        Logical-to-physical complete qubit bijection in the same bounded width.
    measurement_terms
        Complete logical Pauli labels, matched exactly against the problem.

    """

    candidate_id: str
    circuit: QuantumCircuit
    layout: tuple[int, ...]
    measurement_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate bounded circuit metadata before any dense calculation."""
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be nonempty")
        if not isinstance(self.circuit, QuantumCircuit):
            raise ValueError("candidate circuit must be native Qiskit")
        n = self.circuit.num_qubits
        if not 1 <= n <= 8 or len(self.circuit.data) > 4096:
            raise ValueError("candidate exceeds the eight-qubit/4096-operation profile")
        if (
            not isinstance(self.layout, tuple)
            or any(type(q) is not int for q in self.layout)
            or sorted(self.layout) != list(range(n))
        ):
            raise ValueError("layout must be a complete qubit bijection")
        if not isinstance(self.measurement_terms, tuple):
            raise ValueError("measurement_terms must be immutable")


@dataclass(frozen=True, slots=True)
class DesignObjectives:
    """Common weighted minimisation and hard feasibility limits for all candidates.

    Parameters
    ----------
    gate_weight, measurement_weight, trainability_weight
        Nonnegative finite weights in the score: expectation + weighted gate count
        + weighted QWC group count - weighted local finite-difference gradient norm.
    max_gates
        Maximum lowered gate count, a positive integer no larger than 4096.
    minimum_gradient_norm
        Nonnegative local diagnostic threshold, never a global trainability claim.
    diagnostic_step
        Central difference step in radians, in [1e-6, 1e-2]; both h and h/2 run.

    """

    gate_weight: float = 0.0
    measurement_weight: float = 0.0
    trainability_weight: float = 0.0
    max_gates: int = 4096
    minimum_gradient_norm: float = 0.0
    diagnostic_step: float = 1e-4

    def __post_init__(self) -> None:
        """Reject invalid limits without clamping or changing the objective."""
        for name in (
            "gate_weight",
            "measurement_weight",
            "trainability_weight",
            "minimum_gradient_norm",
        ):
            value = getattr(self, name)
            _finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if type(self.max_gates) is not int or not 1 <= self.max_gates <= 4096:
            raise ValueError("max_gates must be a bounded positive integer")
        _finite(self.diagnostic_step, "diagnostic_step")
        if not 1e-6 <= self.diagnostic_step <= 1e-2:
            raise ValueError("diagnostic_step must lie in [1e-6, 1e-2]")


def _finite(value: float, name: str) -> None:
    """Require a real finite native scalar for the named input."""
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite real scalar")


def _pauli(label: str, width: int) -> None:
    """Require one bounded equal-width Pauli label in logical qubit order."""
    if (
        not isinstance(label, str)
        or not 1 <= width <= 8
        or len(label) != width
        or any(char not in "IXYZ" for char in label)
    ):
        raise ValueError("Pauli label must have 1..8 equal-width I/X/Y/Z characters")
