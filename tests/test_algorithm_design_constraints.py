# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public algorithm design qualification
"""Compare real native circuits against independent two-qubit expectations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import cos, sin
from typing import cast

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from scpn_quantum_control.codesign import (
    AlgorithmDesignProblem,
    AlgorithmDesignSelection,
    CoDesignLoop,
    DesignBackendSnapshot,
    DesignCandidate,
    DesignObjectives,
)
from scpn_quantum_control.hardware.hal import BackendCapabilities, BackendProfile
from scpn_quantum_control.hardware.hal_qiskit import _reviewed_qpy_load_circuits


def _inputs() -> tuple[AlgorithmDesignProblem, DesignBackendSnapshot, tuple[DesignCandidate, ...]]:
    """Create two layouts sharing one parameter and one dimensionless objective.

    Returns
    -------
    tuple
        Problem, declared local backend snapshot and two native circuit inputs.

    """
    theta = Parameter("theta")
    problem = AlgorithmDesignProblem(
        source_ref="a" * 64,
        observables=(("ZI", 1.0), ("IZ", 0.5)),
        parameters=(theta,),
        parameter_values=(1.1,),
        parameter_uses=(2,),
    )
    profile = BackendProfile(
        "design_local",
        "local",
        "direct",
        "gate_model",
        "qiskit",
        ("qiskit",),
        BackendCapabilities(True, True, True, False, False, False, max_qubits=2),
    )
    backend = DesignBackendSnapshot(
        profile, "synthetic-calibration-1", ("ry", "cx"), ((0, 1), (1, 0))
    )
    plain = QuantumCircuit(2)
    plain.ry(theta, 0)
    plain.ry(theta, 1)
    entangled = QuantumCircuit(2)
    entangled.ry(theta, 1)
    entangled.ry(theta, 0)
    entangled.cx(0, 1)
    return (
        problem,
        backend,
        (
            DesignCandidate("plain", plain, (0, 1), ("ZI", "IZ")),
            DesignCandidate("entangled", entangled, (1, 0), ("ZI", "IZ")),
        ),
    )


def test_algorithm_design_constraints_01() -> None:
    """Score both layouts using the same independently enumerated objective."""
    problem, backend, candidates = _inputs()
    result = CoDesignLoop.design(
        problem,
        candidates,
        backend,
        objectives=DesignObjectives(
            gate_weight=0.1, measurement_weight=0.05, trainability_weight=0.01
        ),
    )
    c, s = cos(1.1 / 2), sin(1.1 / 2)
    amplitudes = (c * c, s * c, c * s, s * s)
    energies = []
    for entangled in (False, True):
        value = 0.0
        for basis, amplitude in enumerate(amplitudes):
            q0, q1 = basis & 1, basis >> 1
            if entangled:
                q1 ^= q0
            value += amplitude**2 * ((-1) ** q1 + 0.5 * (-1) ** q0)
        energies.append(value)
    assert tuple(row.objective_value for row in result.evaluations) == pytest.approx(
        energies, abs=1e-12
    )
    assert tuple(row.gate_count for row in result.evaluations) == (2, 3)
    assert all(row.measurement_groups == (("IZ", "ZI"),) for row in result.evaluations)
    assert result.selected is not None
    assert result.selected.candidate_id == "entangled"
    result.selected.require_current(problem, backend)


def test_algorithm_design_constraints_02() -> None:
    """Refuse lost observables and broken parameter sharing without source writes."""
    problem, backend, candidates = _inputs()
    original = str(candidates[0].circuit)
    broken = QuantumCircuit(2)
    broken.ry(problem.parameters[0], 0)
    broken.ry(Parameter("other"), 1)
    result = CoDesignLoop.design(
        problem,
        (
            replace(candidates[0], measurement_terms=("ZI",)),
            replace(candidates[1], circuit=broken),
        ),
        backend,
    )
    assert result.selected is None
    assert "observable" in " ".join(result.evaluations[0].reasons)
    assert "parameter" in " ".join(result.evaluations[1].reasons)
    assert str(candidates[0].circuit) == original
    recovered = CoDesignLoop.design(problem, candidates, backend)
    assert recovered.selected is not None


def test_algorithm_design_constraints_03() -> None:
    """Invalidate an immutable selected plan after backend or problem changes."""
    problem, backend, candidates = _inputs()
    selected = CoDesignLoop.design(problem, candidates, backend).selected
    assert selected is not None
    digest = selected.sha256
    with pytest.raises(ValueError, match="backend"):
        selected.require_current(problem, replace(backend, snapshot_ref="synthetic-calibration-2"))
    with pytest.raises(ValueError, match="problem"):
        selected.require_current(replace(problem, parameter_values=(0.3,)), backend)
    next(c for c in candidates if c.candidate_id == selected.candidate_id).circuit.x(0)
    assert selected.sha256 == digest
    selected.require_current(problem, backend)


def test_shared_parameter_diagnostics_match_analytic_derivative_and_refine() -> None:
    """Retain native sharing and the expected quadratic central-difference regime."""
    problem, backend, candidates = _inputs()
    coarse = CoDesignLoop.design(
        problem, candidates[1:], backend, objectives=DesignObjectives(diagnostic_step=1e-2)
    )
    fine = CoDesignLoop.design(
        problem, candidates[1:], backend, objectives=DesignObjectives(diagnostic_step=5e-3)
    )
    row = fine.evaluations[0]
    expected = abs(sin(1.1) * (2 * cos(1.1) + 0.5))
    assert row.gradient_norm == pytest.approx(expected, abs=2e-5)
    assert row.gradient_refinement is not None
    assert coarse.evaluations[0].gradient_refinement is not None
    assert 3.9 < coarse.evaluations[0].gradient_refinement / row.gradient_refinement < 4.1
    restored = _reviewed_qpy_load_circuits(bytes.fromhex(row.ansatz_qpy_hex))[0]
    assert set(restored.parameters) == set(problem.parameters)
    assert restored.data[-1].operation.name == "cx"
    assert restored.find_bit(restored.data[-1].qubits[0]).index == 0


@pytest.mark.parametrize(
    "kind",
    [
        "count",
        "sharing",
        "effect",
        "capacity",
        "unknown_capacity",
        "analog",
        "connectivity",
        "gate_limit",
        "trainability",
        "symmetry",
        "native_basis",
        "score_overflow",
    ],
)
def test_infeasible_alternatives_retain_reasons_and_recover(kind: str) -> None:
    """Keep actual physical, semantic and numerical refusals separate from recovery.

    Parameters
    ----------
    kind
        Public candidate, objective or declared-backend constraint to violate.

    """
    problem, backend, candidates = _inputs()
    candidate = candidates[1]
    objectives = DesignObjectives()
    if kind == "count":
        candidate = replace(candidate, circuit=QuantumCircuit(1), layout=(0,))
    elif kind == "sharing":
        problem = replace(problem, parameter_uses=(1,))
    elif kind == "effect":
        circuit = candidate.circuit.copy()
        circuit.measure_all()
        candidate = replace(candidate, circuit=circuit)
    elif kind in ("capacity", "unknown_capacity"):
        capacity = 1 if kind == "capacity" else None
        backend = replace(
            backend,
            profile=replace(
                backend.profile,
                capabilities=replace(backend.profile.capabilities, max_qubits=capacity),
            ),
        )
    elif kind == "analog":
        backend = replace(
            backend, profile=replace(backend.profile, modality="neutral_atom_analog")
        )
    elif kind == "connectivity":
        backend = replace(backend, coupling_edges=())
    elif kind == "gate_limit":
        objectives = replace(objectives, max_gates=1)
    elif kind == "trainability":
        objectives = replace(objectives, minimum_gradient_norm=10.0)
    elif kind == "symmetry":
        problem = replace(problem, symmetries=(("ZZ", 1),))
    elif kind == "native_basis":
        backend = replace(backend, basis_gates=("cx",))
    else:
        objectives = replace(objectives, gate_weight=1e308)
    rejected = CoDesignLoop.design(problem, (candidate,), backend, objectives=objectives)
    assert rejected.selected is None
    assert rejected.evaluations[0].reasons
    good_problem, good_backend, good_candidates = _inputs()
    assert CoDesignLoop.design(good_problem, good_candidates, good_backend).selected is not None


def test_symmetry_preserving_state_and_deterministic_cost_tradeoff() -> None:
    """Accept the declared stabiliser and change selection only through common weights."""
    problem, backend, candidates = _inputs()
    symmetric = replace(problem, parameter_values=(0.0,), symmetries=(("ZZ", 1),))
    result = CoDesignLoop.design(symmetric, candidates, backend)
    assert result.selected is not None
    assert not any(row.reasons for row in result.evaluations)
    cheaper = CoDesignLoop.design(
        problem, candidates, backend, objectives=DesignObjectives(gate_weight=1.0)
    )
    assert cheaper.selected is not None
    assert cheaper.selected.candidate_id == "plain"
    tied = CoDesignLoop.design(
        problem,
        (replace(candidates[0], candidate_id="z"), replace(candidates[0], candidate_id="a")),
        backend,
    )
    assert tied.selected is not None
    assert tied.selected.candidate_id == "a"


@pytest.mark.parametrize(
    "kind", ["problem", "backend", "objectives", "empty", "duplicate", "untyped", "too_many"]
)
def test_request_admission_is_typed_and_bounded(kind: str) -> None:
    """Refuse malformed whole requests before candidate evaluation.

    Parameters
    ----------
    kind
        Contract type or original candidate-set bound to violate.

    """
    problem, backend, candidates = _inputs()
    kwargs: dict[str, object] = {"problem": problem, "backend": backend, "candidates": candidates}
    if kind in ("problem", "backend", "objectives"):
        kwargs[kind] = object()
    elif kind == "empty":
        kwargs["candidates"] = ()
    elif kind == "duplicate":
        kwargs["candidates"] = (candidates[0], candidates[0])
    elif kind == "untyped":
        kwargs["candidates"] = (object(),)
    else:
        kwargs["candidates"] = candidates * 9
    with pytest.raises(ValueError):
        # This is an intentional runtime boundary call with heterogeneous invalid
        # keyword values, kept outside the statically supported request domain.
        call = cast("Callable[..., AlgorithmDesignSelection]", CoDesignLoop.design)
        call(**kwargs)


def test_mutated_candidate_rechecks_operation_admission_before_native_lowering() -> None:
    """Reject a caller-owned circuit that grew after candidate construction."""
    problem, backend, candidates = _inputs()
    candidate = candidates[0]
    for _ in range(4095):
        candidate.circuit.h(0)
    result = CoDesignLoop.design(problem, (candidate,), backend)
    assert result.selected is None
    assert result.evaluations[0].reasons == (
        "candidate exceeds the 4096-operation design profile",
    )
    assert result.evaluations[0].ansatz_qpy_hex == ""


def test_unqualified_circuit_metadata_is_not_copied_into_selected_plan() -> None:
    """Refuse arbitrary metadata before native QPY serialisation can carry it onward."""
    problem, backend, candidates = _inputs()
    candidates[0].circuit.metadata = {"unqualified_private_note": "sentinel"}
    result = CoDesignLoop.design(problem, candidates[:1], backend)
    assert result.selected is None
    assert result.evaluations[0].ansatz_qpy_hex == ""
    assert "explicit design companion" in result.evaluations[0].reasons[0]
