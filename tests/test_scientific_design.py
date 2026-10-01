# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific design input tests
"""Exercise immutable supplied designs through public problem validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import (
    DesignObjective,
    ScientificDesign,
    ScientificUnits,
    build_kuramoto_problem,
    validate_scientific_design,
)


def _phase() -> ScientificDesign:
    """Build a supplied two-node phase declaration for public validation."""
    return ScientificDesign(
        model="phase_kuramoto",
        normalisation="pairwise_sum",
        coordinate_space="logical",
        units=ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
        topology=((0, 1),),
        initial_state=np.array([0.0, 0.2]),
        observable="phase_order_parameter",
        observable_weights=np.ones(2),
        objective=DesignObjective("simulate", None, "1"),
    )


def test_design_snapshots_arrays_graph_and_trainable_names() -> None:
    """Caller mutation and write-flag changes cannot alter the supplied design."""
    phases = np.array([0.0, 0.2])
    weights = np.ones(2)
    times = np.array([-1.0, 0.0])
    states = np.array([[0.1, 0.3], [0.0, 0.2]])
    graph = [[0, 1]]
    trainable = ["omega"]
    design = replace(
        _phase(),
        initial_state=phases,
        observable_weights=weights,
        topology=cast(Any, graph),
        trainable=cast(Any, trainable),
        history_times=times,
        history_states=states,
    )
    validate_scientific_design(build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), design)
    for values in (phases, weights, times, states):
        values[...] = 99.0
    graph[0][1] = 0
    trainable.append("K_nm")
    np.testing.assert_array_equal(design.initial_state, [0.0, 0.2])
    np.testing.assert_array_equal(design.observable_weights, [1.0, 1.0])
    np.testing.assert_array_equal(design.history_times, [-1.0, 0.0])
    np.testing.assert_array_equal(design.history_states, [[0.1, 0.3], [0.0, 0.2]])
    assert design.topology == ((0, 1),) and design.trainable == ("omega",)
    for snapshot in (
        design.initial_state,
        design.observable_weights,
        design.history_times,
        design.history_states,
    ):
        assert snapshot is not None
        with pytest.raises(ValueError):
            snapshot.setflags(write=True)


@pytest.mark.parametrize(
    "field,values",
    [
        ("initial_state", [[0.0], [0.0, 0.1]]),
        ("initial_state", [False, True]),
        ("initial_state", ["0", "1"]),
        ("initial_state", [object(), object()]),
        ("initial_state", np.array([0.0 + 1j, 0.0])),
        ("initial_state", [0.0, np.nan]),
        ("observable_weights", [np.inf, 1.0]),
        ("history_times", [0.0, np.nan]),
        ("history_states", [[0.0, np.inf]]),
        ("topology", (1,)),
        ("trainable", ([],)),
    ],
)
def test_design_rejects_unusable_numeric_inputs(field: str, values: object) -> None:
    """Malformed, nonfinite or coerced design values refuse before binding."""
    with pytest.raises(ValueError):
        replace(_phase(), **{field: cast(Any, values)})


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model", "unspecified", "model"),
        ("normalisation", "implicit", "normalisation"),
        ("coordinate_space", "unknown", "coordinate_space"),
        ("units", None, "units"),
        ("units", ScientificUnits("s", "Hz", "rad/s", "rad", "1"), "units"),
        ("units", ScientificUnits("1", "rad/s", "1", "rad", "1"), "units"),
        ("units", ScientificUnits("s", "rad/s", "rad/s", "rad", "rad"), "units"),
        ("units", ScientificUnits("s", "rad/s", "rad/s", "1", "1"), "units.state"),
        ("observable", "spin_z", "observable"),
        ("observable_weights", np.ones(3), "observable_weights"),
        ("observable_weights", np.array([-1.0, 1.0]), "observable_weights"),
        ("observable_weights", np.zeros(2), "observable_weights"),
        ("initial_state", np.zeros((1, 2)), "initial_state"),
        ("topology", ((0,),), "topology"),
        ("topology", ((False, 1),), "topology"),
        ("topology", ((0.0, 1),), "topology"),
        ("topology", ((1, 0),), "topology"),
        ("topology", ((0, 2),), "topology"),
        ("topology", ((0, 1), (0, 1)), "topology"),
        ("topology", (), "topology"),
        ("history_times", np.array([0.0]), "supplied together"),
        ("history_states", np.array([[0.0, 0.2]]), "supplied together"),
        ("objective", None, "objective"),
        ("objective", DesignObjective(cast(Any, "unknown"), None, "1"), "kind"),
        ("objective", DesignObjective("simulate", None, "rad"), "unit"),
        ("objective", DesignObjective("simulate", 0.2, "1"), "no target"),
        ("objective", DesignObjective("synchronise", None, "1"), "synchronise"),
        ("objective", DesignObjective("synchronise", 1.01, "1"), "synchronise"),
        ("objective", DesignObjective("maximise_observable", float("inf"), "1"), "finite"),
        ("objective", DesignObjective("maximise_observable", cast(Any, True), "1"), "finite"),
        ("objective", DesignObjective("maximise_observable", cast(Any, "1"), "1"), "finite"),
        ("objective", DesignObjective("minimise_gate_cost", None, "gate"), "quantum"),
        ("trainable", ("missing",), "trainable"),
        ("trainable", ("omega", "omega"), "trainable"),
        ("trainable", ("initial_state_imag",), "trainable"),
        ("trainable", ("history_states",), "trainable"),
    ],
)
def test_bound_phase_design_rejects_ambiguous_contracts(
    field: str, value: object, message: str
) -> None:
    """Actual public binding refuses malformed declarations without source edits."""
    coupling = np.array([[0.0, 0.4], [0.4, 0.0]])
    problem = build_kuramoto_problem(coupling, np.zeros(2))
    design = replace(_phase(), **{field: cast(Any, value)})
    with pytest.raises(ValueError, match=message):
        validate_scientific_design(problem, design)
    np.testing.assert_array_equal(problem.K_nm, coupling)


@pytest.mark.parametrize(
    "times,states,message",
    [
        (np.zeros((1, 1)), np.array([[0.0, 0.2]]), "shapes"),
        (np.empty(0), np.empty((0, 2)), "shapes"),
        (np.array([0.0]), np.zeros((1, 3)), "shapes"),
        (np.array([-1.0]), np.array([[0.0, 0.2]]), "end at zero"),
        (np.array([0.0, 0.0]), np.array([[0.0, 0.2], [0.0, 0.2]]), "increase strictly"),
        (np.array([1.0, 0.0]), np.array([[0.0, 0.2], [0.0, 0.2]]), "increase strictly"),
        (np.array([0.0]), np.array([[0.0, 0.3]]), "end at initial_state"),
    ],
)
def test_bound_phase_history_rejects_inconsistent_time_and_state(
    times: np.ndarray[Any, Any], states: np.ndarray[Any, Any], message: str
) -> None:
    """History is a consistent initial-state declaration, not a saved trajectory."""
    design = replace(_phase(), history_times=times, history_states=states)
    with pytest.raises(ValueError, match=message):
        validate_scientific_design(build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), design)


def test_topology_order_and_untyped_binding_refuse() -> None:
    """Graph identity uses canonical ordered edges and an actual typed design."""
    problem = build_kuramoto_problem(np.zeros((3, 3)), np.zeros(3))
    design = replace(
        _phase(),
        topology=((1, 2), (0, 1)),
        initial_state=np.zeros(3),
        observable_weights=np.ones(3),
    )
    with pytest.raises(ValueError, match="sorted"):
        validate_scientific_design(problem, design)
    with pytest.raises(ValueError, match="ScientificDesign"):
        validate_scientific_design(problem, cast(Any, {}))


def test_dimensionless_phase_and_objective_are_explicit() -> None:
    """Dimensionless time/rates and valid phase targets retain their declarations."""
    problem = build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2))
    for objective in (
        DesignObjective("synchronise", 0.8, "1"),
        DesignObjective("maximise_observable", 0.6, "1"),
    ):
        design = replace(
            _phase(), units=ScientificUnits("1", "1", "1", "rad", "1"), objective=objective
        )
        validate_scientific_design(problem, design)
        assert design.units.time == "1" and design.objective is objective


def test_quantum_complex_snapshot_and_gate_objective() -> None:
    """Quantum amplitudes are finite immutable complex data, separate from phase."""
    initial = np.array([1.0, 1j, 0.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    design = replace(
        _phase(),
        model="quantum_xy",
        units=ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
        observable="spin_z",
        initial_state=initial,
        objective=DesignObjective("minimise_gate_cost", 5.0, "gate"),
        trainable=("initial_state_imag",),
    )
    validate_scientific_design(build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), design)
    initial[:] = 0.0
    assert design.initial_state[1] == pytest.approx(1j / np.sqrt(2))
    with pytest.raises(ValueError):
        design.initial_state.setflags(write=True)
    with pytest.raises(ValueError, match="finite"):
        replace(design, initial_state=np.array([1.0, complex(0.0, np.inf), 0.0, 0.0]))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("normalisation", "population_mean", "pairwise_sum"),
        ("initial_state", np.zeros(3, dtype=np.complex128), "initial_state"),
        ("initial_state", np.ones(4, dtype=np.complex128), "unit norm"),
        ("objective", DesignObjective("synchronise", 0.5, "1"), "phase-order"),
        ("objective", DesignObjective("minimise_gate_cost", -1.0, "gate"), "nonnegative"),
        ("objective", DesignObjective("minimise_gate_cost", 0.5, "gate"), "integer"),
    ],
)
def test_quantum_binding_refuses_phase_conventions_and_invalid_amplitudes(
    field: str, value: object, message: str
) -> None:
    """Actual spin binding refuses missing quantum constraints or phase claims."""
    design = replace(
        _phase(),
        model="quantum_xy",
        units=ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
        observable="spin_z",
        initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
    )
    with pytest.raises(ValueError, match=message):
        validate_scientific_design(
            build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)),
            replace(design, **{field: cast(Any, value)}),
        )


def test_quantum_phase_history_refuses_before_compilation() -> None:
    """Unsupported phase history cannot become an accepted quantum declaration."""
    design = replace(
        _phase(),
        model="quantum_xy",
        units=ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
        observable="spin_z",
        initial_state=np.array([1.0, 0.0, 0.0, 0.0]),
        history_times=np.array([0.0]),
        history_states=np.zeros((1, 2)),
    )
    with pytest.raises(ValueError, match="history is unsupported"):
        validate_scientific_design(build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2)), design)


def test_quantum_binding_preserves_original_problem_dimension_support() -> None:
    """Scientific metadata introduces no arbitrary new qubit-count ceiling."""
    initial = np.zeros(2**17, dtype=np.complex128)
    initial[0] = 1.0
    design = replace(
        _phase(),
        model="quantum_xy",
        units=ScientificUnits("s", "rad/s", "rad/s", "1", "1"),
        observable="spin_z",
        topology=(),
        observable_weights=np.ones(17),
        initial_state=initial,
    )
    validate_scientific_design(build_kuramoto_problem(np.zeros((17, 17)), np.zeros(17)), design)
    assert design.initial_state.shape == (2**17,)
