# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded algorithm design selection
"""Compare ansatzes through existing native planners and numerical owners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import ParameterExpression
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..canonical_encoding import canonical_digest
from ..compiler.circuit_pass_records import CircuitIR
from ..compiler.circuit_source import snapshot_circuit
from ..differentiable_finite_difference import finite_difference_gradient
from ..execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from ..execution_reservations import reserve_execution_memory
from .algorithm_design_contracts import (
    AlgorithmDesignProblem,
    DesignBackendSnapshot,
    DesignCandidate,
    DesignObjectives,
)


@dataclass(frozen=True, slots=True)
class DesignEvaluation:
    """One supported comparison or an explicit source-backed refusal.

    Parameters
    ----------
    candidate_id
        Original alternative identity.
    reasons
        Empty for an eligible alternative, otherwise exact admission failures.
    objective_value
        Dimensionless expectation at the shared evaluation point, or unavailable.
    gradient_norm, gradient_refinement
        Local central-difference norm and h versus h/2 norm difference. The latter
        is an empirical diagnostic, not a certified derivative error bound.
    gate_count
        Actual lowered native gate count; zero if lowering was refused.
    measurement_groups
        Native qubit-wise commuting groups in original logical Pauli labels.
    estimated_numeric_bytes
        Conservative live numeric buffer estimate admitted by the existing gate.
    score
        Common weighted objective score, available only for eligible alternatives.
    circuit_ir
        Immutable bound native circuit snapshot, or unavailable before evaluation.
    ansatz_qpy_hex
        Immutable original parameterised native QPY bytes, hex encoded. Retains
        native parameter UUIDs/expressions; never loaded from untrusted input here.

    """

    candidate_id: str
    reasons: tuple[str, ...] = ()
    objective_value: float | None = None
    gradient_norm: float | None = None
    gradient_refinement: float | None = None
    gate_count: int = 0
    measurement_groups: tuple[tuple[str, ...], ...] = ()
    estimated_numeric_bytes: int = 0
    score: float | None = None
    circuit_ir: CircuitIR | None = None
    ansatz_qpy_hex: str = ""


@dataclass(frozen=True, slots=True)
class AlgorithmDesignPlan:
    """Immutable selected ansatz, logical layout, objective and target binding.

    Parameters
    ----------
    problem_sha256, backend_sha256
        Complete input semantic identities, checked again before reuse.
    evaluation
        Selected original comparison, including bound circuit and parameterised QPY.
    layout
        Selected logical-to-physical bijection.
    objectives
        Exact common weights, feasibility limits and diagnostic displacement.

    """

    problem_sha256: str
    backend_sha256: str
    evaluation: DesignEvaluation
    layout: tuple[int, ...]
    objectives: DesignObjectives

    @property
    def candidate_id(self) -> str:
        """Original identity of the selected alternative."""
        return self.evaluation.candidate_id

    @property
    def sha256(self) -> str:
        """Canonical identity of every immutable selection field."""
        return canonical_digest("algorithm_design_plan.v1", asdict(self))

    def require_current(
        self, problem: AlgorithmDesignProblem, backend: DesignBackendSnapshot
    ) -> None:
        """Refuse reuse against a changed scientific problem or target declaration.

        Parameters
        ----------
        problem, backend
            Current exact input contracts, including parameter and calibration identity.

        Raises
        ------
        ValueError
            If either binding has changed; the prior plan is left untouched.

        """
        if problem.sha256 != self.problem_sha256:
            raise ValueError("selected design problem binding is stale")
        if backend.sha256 != self.backend_sha256:
            raise ValueError("selected design backend binding is stale")


@dataclass(frozen=True, slots=True)
class AlgorithmDesignSelection:
    """Complete alternative comparisons and an optional local design selection.

    Parameters
    ----------
    evaluations
        All alternatives in request order, retaining every refused candidate.
    selected
        Lowest eligible score, stable by candidate ID on exact ties; absent when
        all alternatives refuse. This record grants no execution authority.

    """

    evaluations: tuple[DesignEvaluation, ...]
    selected: AlgorithmDesignPlan | None


def select_algorithm_design(
    problem: AlgorithmDesignProblem,
    candidates: tuple[DesignCandidate, ...],
    backend: DesignBackendSnapshot,
    *,
    objectives: DesignObjectives | None = None,
) -> AlgorithmDesignSelection:
    """Evaluate one bounded native candidate set against a common objective.

    Parameters
    ----------
    problem
        Exact dimensionless Pauli objective, parameter identity and constraints.
    candidates
        One to 16 alternatives; input circuits are copied and never modified.
    backend
        Declared target snapshot, basis and directed connectivity. All numerical
        evaluation remains on the local native statevector reference.
    objectives
        Shared score weights, gate ceiling and local diagnostic limits.

    Returns
    -------
    AlgorithmDesignSelection
        All observed comparisons/refusals and immutable selection, when eligible.

    Raises
    ------
    ValueError
        Request types, candidate count or identities violate the comparison contract.
    DenseAllocationError
        The existing memory owner refuses the bounded numeric reference allocation.

    Notes
    -----
    No cloud call, file write, optimiser training, pulse/analog execution or new
    numerical engine occurs. Refinement differences and symmetry checks apply to
    this evaluation point only. No physical fidelity or performance claim follows.

    """
    objectives = DesignObjectives() if objectives is None else objectives
    if (
        not isinstance(problem, AlgorithmDesignProblem)
        or not isinstance(backend, DesignBackendSnapshot)
        or not isinstance(objectives, DesignObjectives)
    ):
        raise ValueError("design request requires typed problem, backend and objectives")
    if (
        not isinstance(candidates, tuple)
        or not 1 <= len(candidates) <= 16
        or any(not isinstance(c, DesignCandidate) for c in candidates)
    ):
        raise ValueError("candidates must contain 1..16 typed alternatives")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("candidate identities must be unique")
    evaluations = tuple(
        _evaluate(problem, candidate, backend, objectives) for candidate in candidates
    )
    eligible = [
        (row.score, row.candidate_id, i)
        for i, row in enumerate(evaluations)
        if row.score is not None
    ]
    if not eligible:
        return AlgorithmDesignSelection(evaluations, None)
    _, _, index = min(eligible)
    return AlgorithmDesignSelection(
        evaluations,
        AlgorithmDesignPlan(
            problem.sha256,
            backend.sha256,
            evaluations[index],
            candidates[index].layout,
            objectives,
        ),
    )


def _evaluate(
    problem: AlgorithmDesignProblem,
    candidate: DesignCandidate,
    backend: DesignBackendSnapshot,
    objectives: DesignObjectives,
) -> DesignEvaluation:
    """Admit the original contracts before calling existing native numerical owners."""
    reasons = _admission_reasons(problem, candidate, backend)
    if reasons:
        return DesignEvaluation(candidate.candidate_id, reasons)
    circuit = candidate.circuit.copy()
    # Copy the symbolic source before native lowering; the selected plan stores
    # original shared parameter identities, not only a bound numeric circuit.
    stream = BytesIO()
    qpy.dump(circuit, stream)
    try:
        mapped = QuantumCircuit(circuit.num_qubits).compose(circuit, qubits=candidate.layout)
        lowered = transpile(
            mapped, basis_gates=list(backend.basis_gates), optimization_level=0, seed_transpiler=0
        )
        gates = len(lowered.data)
        for instruction in lowered.data:
            if len(instruction.qubits) == 2:
                edge = tuple(lowered.find_bit(q).index for q in instruction.qubits)
                if edge not in backend.coupling_edges:
                    reasons += (
                        f"hardware connectivity refuses lowered {instruction.operation.name}{edge}",
                    )
        if gates > objectives.max_gates:
            reasons += ("gate cost exceeds the shared max_gates constraint",)
        if reasons:
            return DesignEvaluation(candidate.candidate_id, reasons, gate_count=gates)
        labels = {
            _mapped_label(label, candidate.layout): label for label, _ in problem.observables
        }
        observable = SparsePauliOp.from_list(
            [
                (_mapped_label(label, candidate.layout), coefficient)
                for label, coefficient in problem.observables
            ]
        )
        groups = tuple(
            sorted(
                tuple(sorted(labels[str(label)] for label in group.paulis.to_labels()))
                for group in observable.group_commuting(qubit_wise=True)
            )
        )
        memory = ExecutionMemoryPlan(
            (
                ExecutionBuffer.hilbert(
                    "design_reference",
                    "intermediate",
                    circuit.num_qubits,
                    rank=2,
                    count=4,
                ),
            )
        )
        with reserve_execution_memory(memory) as reservation:

            def value(values: NDArray[np.float64]) -> float:
                """Evaluate the original observable with native shared-parameter binding."""
                reservation.checkpoint()
                bound = lowered.assign_parameters(
                    dict(zip(problem.parameters, values, strict=True))
                )
                result = float(
                    Statevector.from_instruction(bound).expectation_value(observable).real
                )
                reservation.checkpoint()
                return result

            point = np.array(problem.parameter_values, dtype=np.float64)
            expectation = value(point)
            coarse = finite_difference_gradient(value, point, step=objectives.diagnostic_step)
            fine = finite_difference_gradient(value, point, step=objectives.diagnostic_step / 2)
            norm = float(np.linalg.norm(fine))
            refinement = float(np.linalg.norm(fine - coarse))
            bound = lowered.assign_parameters(dict(zip(problem.parameters, point, strict=True)))
            ir = snapshot_circuit(bound)
            state = Statevector.from_instruction(bound)
            for label, sign in problem.symmetries:
                symmetry = SparsePauliOp.from_list([(_mapped_label(label, candidate.layout), 1.0)])
                if abs(complex(state.expectation_value(symmetry)) - sign) > 1e-10:
                    reasons += (
                        f"symmetry {label} violates the required eigenvalue at this point",
                    )
            if norm - refinement < objectives.minimum_gradient_norm:
                reasons += (
                    "local trainability diagnostic does not meet the minimum gradient norm",
                )
        score = (
            expectation
            + objectives.gate_weight * gates
            + objectives.measurement_weight * len(groups)
            - objectives.trainability_weight * norm
        )
        if not isfinite(score):
            reasons += ("weighted design score is not finite",)
        return DesignEvaluation(
            candidate.candidate_id,
            reasons,
            expectation,
            norm,
            refinement,
            gates,
            groups,
            memory.bytes_required,
            None if reasons else score,
            ir,
            stream.getvalue().hex(),
        )
    except (QiskitError, ValueError, OverflowError) as exc:
        return DesignEvaluation(candidate.candidate_id, (f"native static design refused: {exc}",))


def _admission_reasons(
    problem: AlgorithmDesignProblem,
    candidate: DesignCandidate,
    backend: DesignBackendSnapshot,
) -> tuple[str, ...]:
    """Keep objective, sharing, effects and target meaning intact before evaluation."""
    circuit = candidate.circuit
    if len(circuit.data) > 4096:
        return ("candidate exceeds the 4096-operation design profile",)
    if circuit.metadata:
        return ("arbitrary circuit metadata needs an explicit design companion",)
    width = len(problem.observables[0][0])
    if circuit.num_qubits != width:
        return ("candidate width changes the objective domain",)
    if sorted(candidate.measurement_terms) != sorted(label for label, _ in problem.observables):
        return ("candidate drops, duplicates or substitutes an observable",)
    if set(circuit.parameters) != set(problem.parameters):
        return ("candidate changes parameter identity or shared parameter binding",)
    uses = tuple(
        sum(
            p in expression.parameters
            for instruction in circuit.data
            for expression in instruction.operation.params
            if isinstance(expression, ParameterExpression)
        )
        for p in problem.parameters
    )
    if uses != problem.parameter_uses:
        return ("candidate breaks required shared parameter occurrence counts",)
    if circuit.num_clbits or any(
        instruction.operation.name
        not in ("rx", "ry", "rz", "cx", "cz", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "id")
        for instruction in circuit.data
    ):
        return ("candidate uses unsupported measurement, pulse, analog or dynamic effects",)
    modality = backend.profile.modality
    if "gate_model" not in modality and "statevector_simulator" not in modality:
        return (f"target modality {modality} is not an admitted gate-model design",)
    capacity = backend.profile.capabilities.max_qubits
    if capacity is None or isinstance(capacity, bool) or width > capacity:
        return ("hardware feasibility requires sufficient declared qubit capacity",)
    return ()


def _mapped_label(label: str, layout: tuple[int, ...]) -> str:
    """Map logical Pauli factors onto the explicitly selected physical layout."""
    result = ["I"] * len(layout)
    for logical, physical in enumerate(layout):
        result[len(layout) - 1 - physical] = label[len(layout) - 1 - logical]
    return "".join(result)
