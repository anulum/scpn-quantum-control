# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific problem parameter integration tests
"""Bind scientific inputs through real problem, force and workspace owners."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as qc
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force
from scpn_quantum_control.studio_workspace import (
    canonical_digest,
    validate_parameter_binding,
)


def test_scientific_problem_parameters_01() -> None:
    """Equal supplied matrices with different force conventions stay distinct."""
    coupling = np.array([[0.0, 0.8], [0.8, 0.0]])
    frequencies = np.array([0.2, -0.3])
    phases = np.array([0.0, np.pi / 2])
    problem = qc.build_kuramoto_problem(coupling, frequencies)
    units = qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1")
    bindings = [
        qc.ScientificProblemParameters(
            problem,
            qc.ScientificDesign(
                model="phase_kuramoto",
                normalisation=convention,
                coordinate_space="logical",
                units=units,
                topology=((0, 1),),
                initial_state=phases,
                observable="phase_order_parameter",
                observable_weights=np.ones(2),
                objective=qc.DesignObjective("simulate", None, "1"),
            ),
        )
        for convention in ("pairwise_sum", "population_mean")
    ]
    assert bindings[0].identity != bindings[1].identity
    velocities = [
        frequencies + networked_kuramoto_force(phases, binding.effective_problem().K_nm)
        for binding in bindings
    ]
    np.testing.assert_allclose(velocities[0], [1.0, -1.1], atol=1e-12)
    np.testing.assert_allclose(velocities[1], [0.6, -0.7], atol=1e-12)
    np.testing.assert_array_equal(problem.K_nm, coupling)


def test_scientific_problem_parameters_02() -> None:
    """Public scientific inputs refuse bad matrices, dimensions and nonfinites."""
    for coupling, frequencies in (
        (np.zeros((2, 3)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros(3)),
        (np.array([[0.0, np.inf], [np.inf, 0.0]]), np.zeros(2)),
        (np.zeros((2, 2)), np.array([0.0, np.nan])),
    ):
        with pytest.raises(ValueError):
            qc.build_kuramoto_problem(coupling, frequencies)
    problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="initial_state"):
        qc.ScientificProblemParameters(
            problem,
            qc.ScientificDesign(
                model="phase_kuramoto",
                normalisation="pairwise_sum",
                coordinate_space="logical",
                units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
                topology=(),
                initial_state=np.zeros(3),
                observable="phase_order_parameter",
                observable_weights=np.ones(2),
                objective=qc.DesignObjective("simulate", None, "1"),
            ),
        )


def test_scientific_problem_parameters_03() -> None:
    """Zero coupling leaves the analytic independent-oscillator angular rates."""
    frequencies = np.array([0.3, -0.2, 0.7])
    phases = np.array([-0.4, 0.5, 1.2])
    problem = qc.build_kuramoto_problem(np.zeros((3, 3)), frequencies)
    parameters = qc.ScientificProblemParameters(
        problem,
        qc.ScientificDesign(
            model="phase_kuramoto",
            normalisation="population_mean",
            coordinate_space="physical",
            units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
            topology=(),
            initial_state=phases,
            observable="phase_order_parameter",
            observable_weights=np.ones(3),
            objective=qc.DesignObjective("simulate", None, "1"),
        ),
    )
    effective = parameters.effective_problem()
    for time in (0.0, 0.25, 1.0):
        actual = effective.omega + networked_kuramoto_force(
            phases + frequencies * time, effective.K_nm
        )
        np.testing.assert_array_equal(actual, frequencies)
    specs = parameters.parameter_specs()
    assert [spec.body["key"] for spec in specs][:2] == ["omega", "K_nm"]
    assert [spec.body["unit"] for spec in specs][:2] == ["rad/s", "rad/s"]
    np.testing.assert_array_equal(problem.omega, frequencies)


def _phase_design() -> qc.ScientificDesign:
    """Supply explicit phase declarations with no inferred convention or units."""
    return qc.ScientificDesign(
        model="phase_kuramoto",
        normalisation="pairwise_sum",
        coordinate_space="logical",
        units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
        topology=((0, 1),),
        initial_state=np.array([-0.0, 0.2]),
        observable="phase_order_parameter",
        observable_weights=np.ones(2),
        objective=qc.DesignObjective("simulate", None, "1"),
    )


def test_parameters_snapshot_original_problem_and_exported_documents() -> None:
    """Mutating incoming raw arrays or exported documents cannot rebind identity."""
    original = qc.build_kuramoto_problem(
        np.array([[0.0, 0.3], [0.3, 0.0]]), np.array([0.1, -0.2]), metadata={"source": "supplied"}
    )
    raw_before = original.to_metadata()
    parameters = qc.ScientificProblemParameters(original, _phase_design())
    identity = parameters.identity
    assert parameters.problem is not original
    for values in (original.K_nm, original.omega):
        values.setflags(write=True)
        values[...] = 99.0
    np.testing.assert_array_equal(parameters.problem.K_nm, [[0.0, 0.3], [0.3, 0.0]])
    np.testing.assert_array_equal(parameters.problem.omega, [0.1, -0.2])
    for values in (parameters.problem.K_nm, parameters.problem.omega):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    assert parameters.problem.to_metadata() == raw_before
    document = parameters.to_dict()
    body = cast(dict[str, object], document["body"])
    assert (
        canonical_digest(parameters.schema, {"body": body, "extensions": document["extensions"]})
        == identity
    )
    body["model"] = "changed"
    document["schema"] = "scientific_problem.v99"
    exported = parameters.parameter_values()
    cast(dict[str, object], exported["omega"])["values"] = ["0000000000000000"]
    assert parameters.identity == identity
    assert parameters.to_dict()["schema"] == "scientific_problem.v1"
    assert parameters.effective_problem().metadata["scientific_problem_identity"] == identity


def test_existing_parameter_consumers_accept_exact_payloads_and_refuse_unit_edits() -> None:
    """The existing schema/binding owners consume every exported typed parameter."""
    design = replace(
        _phase_design(),
        trainable=("omega", "K_nm", "history_states"),
        history_times=np.array([-0.5, 0.0]),
        history_states=np.array([[0.3, -0.4], [-0.0, 0.2]]),
    )
    parameters = qc.ScientificProblemParameters(
        qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), design
    )
    values = parameters.parameter_values()
    specs = parameters.parameter_specs()
    keys = [str(spec.body["key"]) for spec in specs]
    assert keys == [
        "omega",
        "K_nm",
        "initial_state_real",
        "observable_weights",
        "history_times",
        "history_states",
    ]
    for spec in specs:
        key = str(spec.body["key"])
        validate_parameter_binding(spec, values[key], str(spec.body["unit"]))
        assert spec.body["trainable"] == (key in design.trainable)
        assert spec.body["default_source"] == f"scientific_problem.v1:{parameters.identity}#{key}"
        with pytest.raises(ValueError, match="unit mismatch"):
            validate_parameter_binding(spec, values[key], "undeclared")
    initial = cast(dict[str, object], values["initial_state_real"])
    assert initial["values"] == ["8000000000000000", "3fc999999999999a"]
    before = parameters.identity
    with pytest.raises(ValueError, match="shape or unit mismatch"):
        validate_parameter_binding(
            specs[0], {"dtype": "float64", "shape": [1], "values": ["0000000000000000"]}, "rad/s"
        )
    assert parameters.identity == before
    np.testing.assert_array_equal(parameters.design.history_times, [-0.5, 0.0])


def test_identity_includes_units_graph_coordinates_state_objective_and_trainability() -> None:
    """Every scientific declaration affecting interpretation participates in identity."""
    problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2))
    base = _phase_design()
    designs = [
        base,
        replace(base, units=qc.ScientificUnits("1", "1", "1", "rad", "1")),
        replace(base, coordinate_space="physical"),
        replace(base, topology=()),
        replace(base, initial_state=np.array([0.0, 0.2])),
        replace(base, observable_weights=np.array([1.0, 2.0])),
        replace(base, objective=qc.DesignObjective("synchronise", 0.8, "1")),
        replace(base, trainable=("omega",)),
        replace(base, history_times=np.array([0.0]), history_states=np.array([[-0.0, 0.2]])),
    ]
    identities = [qc.ScientificProblemParameters(problem, design).identity for design in designs]
    assert len(set(identities)) == len(designs)
    for design, identity in zip(designs, identities, strict=True):
        assert qc.ScientificProblemParameters(problem, design).identity == identity


def test_quantum_identity_schemas_and_real_hamiltonian_compile() -> None:
    """Declared XY inputs reach the actual compiler and independent Pauli oracle."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    frequencies = np.array([0.1, -0.2])
    problem = qc.build_kuramoto_problem(coupling, frequencies)
    amplitudes = np.array([1.0, 1j, 0.0, 0.0]) / np.sqrt(2)
    design = replace(
        _phase_design(),
        model="quantum_xy",
        units=qc.ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
        observable="spin_z",
        initial_state=amplitudes,
        objective=qc.DesignObjective("minimise_gate_cost", None, "gate"),
        trainable=("initial_state_imag",),
    )
    parameters = qc.ScientificProblemParameters(problem, design)
    operator = parameters.compile_hamiltonian()
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1j], [1j, 0.0]])
    z = np.diag([1.0, -1.0])
    expected = (
        -0.3 * (np.kron(x, x) + np.kron(y, y))
        - 0.1 * np.kron(np.eye(2), z)
        + 0.2 * np.kron(z, np.eye(2))
    )
    np.testing.assert_allclose(operator.to_matrix(), expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        qc.compile_dense_hamiltonian(parameters.effective_problem()),
        expected,
        atol=1e-12,
        rtol=0.0,
    )
    specs = parameters.parameter_specs()
    assert [spec.body["key"] for spec in specs][-1] == "initial_state_imag"
    assert specs[-1].body["trainable"] is True
    values = parameters.parameter_values()
    for spec in specs:
        validate_parameter_binding(spec, values[str(spec.body["key"])], str(spec.body["unit"]))
    assert (
        qc.ScientificProblemParameters(
            problem, replace(design, initial_state=amplitudes.conjugate())
        ).identity
        != parameters.identity
    )
    assert parameters.problem.to_metadata() == problem.to_metadata()


def test_phase_compilation_requires_explicit_spin_model() -> None:
    """A valid phase declaration cannot silently enter the spin compiler."""
    problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2))
    parameters = qc.ScientificProblemParameters(problem, _phase_design())
    before = parameters.identity
    with pytest.raises(ValueError, match="no implicit quantum-spin equivalence"):
        parameters.compile_hamiltonian()
    assert parameters.identity == before
    assert parameters.problem.to_metadata() == problem.to_metadata()


def test_mutable_numpy_headers_cannot_rebind_the_captured_companion() -> None:
    """Public array shape/dtype edits affect a read snapshot, not saved identity."""
    problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2))
    design = _phase_design()
    parameters = qc.ScientificProblemParameters(problem, design)
    original = parameters.to_dict()
    identity = parameters.identity
    design.initial_state.shape = (1, 2)
    cast(Any, design.observable_weights).dtype = np.int64
    read_problem, read_design = parameters.problem, parameters.design
    read_problem.K_nm.shape = (4,)
    cast(Any, read_problem.omega).dtype = np.int64
    read_design.initial_state.shape = (1, 2)
    cast(Any, read_design.observable_weights).dtype = np.int64
    assert parameters.to_dict() == original and parameters.identity == identity
    assert parameters.problem.K_nm.shape == (2, 2)
    assert parameters.problem.omega.dtype == np.float64
    assert parameters.design.initial_state.shape == (2,)
    assert parameters.design.observable_weights.dtype == np.float64
    np.testing.assert_array_equal(parameters.effective_problem().K_nm, np.zeros((2, 2)))


def test_scientific_binding_requires_actual_typed_problem_and_design() -> None:
    """Wrong public record types refuse before producing a qualified companion."""
    with pytest.raises(ValueError, match="KuramotoProblem"):
        qc.ScientificProblemParameters(cast(Any, {}), _phase_design())
    with pytest.raises(ValueError, match="ScientificDesign"):
        qc.ScientificProblemParameters(
            qc.build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), cast(Any, {})
        )
