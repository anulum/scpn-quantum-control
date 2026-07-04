# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for topological (Hodge-Laplacian) synchronisation
"""Module-specific tests for :mod:`topological_synchronisation`.

The contracts: the boundary operators satisfy ``B_1 B_2 = 0`` and the harmonic dimension is the first
Betti number (zero for a filled triangle, one for a hollow cycle); the Hodge decomposition is exact and
orthogonal, with a divergence-free curl part, a curl-free gradient part, and a harmonic part in the
kernel of the Hodge Laplacian; a purely harmonic edge signal is a fixed point of the free flow; the
topological Kuramoto flow synchronises the edge phases under coupling; and the input contract is
enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oscillatools.accel.topological_synchronisation import (
    HodgeComponents,
    HodgeStructure,
    TopologicalKuramotoTrajectory,
    hodge_decomposition,
    integrate_topological_kuramoto,
    simplicial_hodge_structure,
    topological_kuramoto_field,
    topological_order_parameter,
)

_FILLED_EDGES = np.array([[0, 1], [0, 2], [1, 2]])
_FILLED_TRIANGLES = np.array([[0, 1, 2]])
_CYCLE_EDGES = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
_NO_TRIANGLES = np.zeros((0, 3), dtype=np.int_)


def _filled() -> HodgeStructure:
    return simplicial_hodge_structure(3, _FILLED_EDGES, _FILLED_TRIANGLES)


def _cycle() -> HodgeStructure:
    return simplicial_hodge_structure(4, _CYCLE_EDGES, _NO_TRIANGLES)


def test_boundary_of_a_boundary_is_zero() -> None:
    structure = _filled()
    assert np.max(np.abs(structure.node_boundary @ structure.edge_boundary)) == pytest.approx(0.0)


def test_betti_number_counts_independent_cycles() -> None:
    # a filled triangle has no hole; a hollow 4-cycle has one independent cycle
    assert _filled().betti_number == 0
    assert _cycle().betti_number == 1


def test_hodge_laplacian_is_the_sum_of_down_and_up_and_is_positive_semidefinite() -> None:
    structure = _filled()
    assert structure.hodge_laplacian == pytest.approx(
        structure.down_laplacian + structure.up_laplacian
    )
    assert np.min(np.linalg.eigvalsh(structure.hodge_laplacian)) > -1e-9


def test_hodge_decomposition_is_exact_and_orthogonal() -> None:
    structure = _filled()
    signal = np.random.default_rng(0).standard_normal(3)
    components = hodge_decomposition(signal, structure)
    assert isinstance(components, HodgeComponents)
    assert components.gradient + components.curl + components.harmonic == pytest.approx(signal)
    assert components.gradient @ components.curl == pytest.approx(0.0, abs=1e-12)
    assert components.gradient @ components.harmonic == pytest.approx(0.0, abs=1e-12)
    assert components.curl @ components.harmonic == pytest.approx(0.0, abs=1e-12)
    # the harmonic part lies in the kernel of the Hodge Laplacian
    assert structure.hodge_laplacian @ components.harmonic == pytest.approx(np.zeros(3), abs=1e-12)
    # the curl part is divergence-free and the gradient part is curl-free
    assert structure.node_boundary @ components.curl == pytest.approx(np.zeros(3), abs=1e-12)
    assert structure.edge_boundary.T @ components.gradient == pytest.approx(np.zeros(1), abs=1e-12)


def test_harmonic_signal_is_a_fixed_point_of_the_free_flow() -> None:
    structure = _cycle()
    harmonic = hodge_decomposition(np.random.default_rng(1).standard_normal(4), structure).harmonic
    # B_1 (harmonic) = 0 and B_2^T (harmonic) = 0, so the coupling vanishes
    field = topological_kuramoto_field(
        harmonic, np.zeros(4), structure, down_coupling=2.0, up_coupling=2.0
    )
    assert field == pytest.approx(np.zeros(4), abs=1e-12)


def test_field_matches_the_topological_kuramoto_formula() -> None:
    structure = _filled()
    phases = np.array([0.3, -0.7, 1.1])
    frequencies = np.array([0.1, 0.2, -0.1])
    expected = (
        frequencies
        - 1.5 * structure.node_boundary.T @ np.sin(structure.node_boundary @ phases)
        - 0.8 * structure.edge_boundary @ np.sin(structure.edge_boundary.T @ phases)
    )
    field = topological_kuramoto_field(
        phases, frequencies, structure, down_coupling=1.5, up_coupling=0.8
    )
    assert field == pytest.approx(expected)


def test_coupling_synchronises_the_edge_phases() -> None:
    structure = _filled()
    # a perturbation of the coherent state: the filled triangle has no harmonic mode, so the
    # Hodge Laplacian is positive definite and the coupling restores coherence (avoiding the
    # winding fixed points that large random phases can fall into)
    initial = 0.4 * np.random.default_rng(2).standard_normal(3)
    frequencies = np.zeros(3)
    uncoupled = integrate_topological_kuramoto(
        initial, frequencies, structure, 0.05, 600, down_coupling=0.0, up_coupling=0.0
    )
    coupled = integrate_topological_kuramoto(
        initial, frequencies, structure, 0.05, 600, down_coupling=3.0, up_coupling=3.0
    )
    assert isinstance(coupled, TopologicalKuramotoTrajectory)
    assert uncoupled.final_phases == pytest.approx(initial)  # no coupling, no drift
    assert topological_order_parameter(coupled.final_phases) > 0.99


@pytest.mark.parametrize(
    ("call", "args", "message"),
    [
        ("structure", (1, _FILLED_EDGES, _FILLED_TRIANGLES), "n_nodes must be at least two"),
        ("structure", (3, np.zeros((0, 2), dtype=np.int_), _NO_TRIANGLES), "non-empty"),
        ("structure", (3, np.array([[0, 5]]), _NO_TRIANGLES), "valid node indices"),
        ("structure", (3, np.array([[1, 0]]), _NO_TRIANGLES), "low-to-high"),
        ("structure", (3, np.array([[0, 1], [0, 1]]), _NO_TRIANGLES), "distinct"),
        ("structure", (3, _FILLED_EDGES, np.array([[2, 1, 0]])), "low-to-high"),
        (
            "structure",
            (4, np.array([[0, 1], [1, 2]]), np.array([[0, 1, 2]])),
            "missing from edges",
        ),
        ("decomposition", (np.zeros(5),), "edge_signal must have length"),
        ("order", (np.zeros((2, 2)),), "non-empty one-dimensional"),
        ("field", (np.zeros(5),), "phases must have length"),
        ("field", (None, np.zeros(5)), "natural_frequencies must have length"),
        ("field", (np.full(3, np.nan),), "must be finite"),
        ("field-coupling", (), "down_coupling and up_coupling must be finite"),
        ("integrate", ("dt", 0.0), "dt must be positive"),
        ("integrate", ("n_steps", 0), "n_steps must be positive"),
        ("integrate-coupling", (), "down_coupling and up_coupling must be finite"),
    ],
)
def test_validation_errors(call: str, args: tuple[Any, ...], message: str) -> None:
    structure = _filled()
    with pytest.raises(ValueError, match=message):
        if call == "structure":
            simplicial_hodge_structure(*args)
        elif call == "decomposition":
            hodge_decomposition(args[0], structure)
        elif call == "order":
            topological_order_parameter(args[0])
        elif call == "field":
            phases = np.zeros(3) if args[0] is None else args[0]
            frequencies = args[1] if len(args) > 1 else np.zeros(3)
            topological_kuramoto_field(
                phases, frequencies, structure, down_coupling=1.0, up_coupling=1.0
            )
        elif call == "field-coupling":
            topological_kuramoto_field(
                np.zeros(3), np.zeros(3), structure, down_coupling=np.inf, up_coupling=1.0
            )
        elif call == "integrate-coupling":
            integrate_topological_kuramoto(
                np.zeros(3),
                np.zeros(3),
                structure,
                0.05,
                10,
                down_coupling=1.0,
                up_coupling=np.inf,
            )
        else:
            settings: dict[str, Any] = {"dt": 0.05, "n_steps": 10}
            settings[args[0]] = args[1]
            integrate_topological_kuramoto(
                np.zeros(3),
                np.zeros(3),
                structure,
                settings["dt"],
                settings["n_steps"],
                down_coupling=1.0,
                up_coupling=1.0,
            )
