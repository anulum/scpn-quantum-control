# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — symmetry, chimera and topology instrument oracles and refusals
"""Check the topology instrument against independent oracles, its CLI and its refusals."""

from __future__ import annotations

import json
import runpy
import sys
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.chimera_control import generate_two_population_chimera
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem, compile_hamiltonian
from scpn_quantum_control.studio import topology_instrument_artifact as artifact
from scpn_quantum_control.studio import topology_instrument_templates as templates
from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    TopologyConstraintLedger,
)

ComplexArray = NDArray[np.complex128]


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Build the instrument once from its owners.

    Returns
    -------
    dict
        The fresh artefact payload.

    """
    return artifact.build_topology_instrument_artifact()


def _xy_order(state: ComplexArray, n: int) -> float:
    """Order parameter from explicit X and Y expectations, without the solver."""
    total = 0.0 + 0.0j
    for qubit in range(n):
        flipped = np.arange(len(state)) ^ (1 << qubit)
        bits = (np.arange(len(state)) >> qubit) & 1
        exp_x = np.vdot(state, state[flipped]).real
        exp_y = np.vdot(state, np.where(bits == 1, 1j, -1j) * state[flipped]).real
        total += exp_x + 1j * exp_y
    return abs(total / n)


def _oracle_state(coupling: NDArray[np.float64], omega: NDArray[np.float64]) -> ComplexArray:
    """Evolve the recorded product state through the independent Pauli-sum matrix."""
    circuit = QuantumCircuit(len(omega))
    for qubit, angle in enumerate(templates.INITIAL_ANGLES):
        circuit.ry(angle, qubit)
    start = Statevector.from_instruction(circuit).data
    matrix = compile_hamiltonian(build_kuramoto_problem(coupling, omega)).to_matrix()
    evolved: ComplexArray = scipy.linalg.expm(-1j * templates.EVOLUTION_TIME * matrix) @ start
    return evolved


def test_topology_symmetry_instrument_01_relabelling_keeps_the_order_parameter(
    built: dict[str, Any],
) -> None:
    """The owner's order parameter is invariant under a consistent node relabelling.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    row = built["templates"]["relabelling"]
    coupling, omega = np.array(row["coupling"]), np.array(row["omega"])
    oracle = _xy_order(_oracle_state(coupling, omega), len(omega))
    assert row["original"]["magnitude"] == pytest.approx(oracle, abs=1e-12)
    assert row["relabelled"]["magnitude"] == pytest.approx(oracle, abs=1e-12)
    assert row["magnitude_difference"] <= row["tolerance"]
    assert row["phase_difference"] <= row["tolerance"]
    assert sorted(row["permutation"]) == list(range(len(omega)))
    permute = np.eye(len(omega))[row["permutation"]]
    with pytest.raises(ValueError, match="K_nm must be symmetric"):
        build_kuramoto_problem(permute @ coupling, omega)


def test_topology_symmetry_instrument_02_finite_sectors_match_the_dense_reference(
    built: dict[str, Any],
) -> None:
    """Sector spectra agree with the dense spectrum; parity weights are conserved.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    row = built["templates"]["sectors"]
    spectra, n = row["spectra"], len(row["omega"])
    reference = np.linalg.eigvalsh(
        compile_hamiltonian(
            build_kuramoto_problem(np.array(row["coupling"]), np.zeros(n))
        ).to_matrix()
    )
    assert np.allclose(spectra["dense_eigenvalues"], reference, atol=1e-12)
    assert spectra["parity_max_deviation"] <= row["tolerance"]
    assert spectra["magnetisation_max_deviation"] <= row["tolerance"]
    assert spectra["parity_dimensions"] == {"even": 2 ** (n - 1), "odd": 2 ** (n - 1)}
    assert spectra["magnetisation_dimensions"] == {
        str(n - 2 * ones): comb(n, ones) for ones in range(n + 1)
    }
    assert spectra["ground_energy"] == pytest.approx(min(reference), abs=1e-12)
    popcount_odd = np.array([bin(k).count("1") % 2 for k in range(2**n)], dtype=bool)
    for name, start in (
        ("even_basis_state", Statevector.from_label(templates.EVEN_BASIS_STATE).data),
        ("mixed_product_state", _oracle_state(np.zeros((n, n)), np.zeros(n))),
    ):
        weights = row["weights"][name]
        odd = float(np.sum(np.abs(start[popcount_odd]) ** 2))
        assert weights["before"]["odd"] == pytest.approx(odd, abs=1e-12)
        assert weights["after"]["odd"] == pytest.approx(odd, abs=1e-12)
        assert weights["after"]["even"] + weights["after"]["odd"] == pytest.approx(1.0)
        assert weights["after"]["leakage_from_even"] == pytest.approx(odd, abs=1e-12)


def test_topology_symmetry_instrument_03_kinks_and_discrete_edits_carry_no_derivative(
    built: dict[str, Any],
) -> None:
    """Only the interior point has a JVP; kinks and edits keep the owner's refusal.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    row = built["templates"]["sensitivity"]
    cases = {case["case"]: case for case in row["cases"]}
    interior = cases["interior point"]
    assert interior["derivative_supported"] is True and interior["refusal"] is None
    ledger = TopologyConstraintLedger(bounds=CouplingGraphBounds(0.0, 1.0))
    point, tangent, step = np.array(interior["matrix"]), np.array(row["tangent"]), 1e-7
    central = (ledger.project(point + step * tangent) - ledger.project(point - step * tangent)) / (
        2 * step
    )
    assert np.allclose(interior["projected_tangent"], central, atol=1e-7)
    clipped = np.array(cases["coupling at the clipping bound"]["matrix"])
    edge = np.zeros_like(clipped)
    edge[0, 2] = edge[2, 0] = 1.0
    right = (ledger.project(clipped + step * edge) - ledger.project(clipped)) / step
    left = (ledger.project(clipped) - ledger.project(clipped - step * edge)) / step
    assert right[0, 2] == 0.0 and left[0, 2] == pytest.approx(1.0)
    expected = {
        "coupling at the clipping bound": "uniform_bounds",
        "edge removed": "sign_policy, uniform_bounds",
        "connectivity threshold": "algebraic_connectivity_threshold",
    }
    for name, blockers in expected.items():
        case = cases[name]
        assert case["derivative_supported"] is False
        assert case["projected_tangent"] is None
        assert case["refusal"] == f"projection derivative is unsupported for: {blockers}"
        assert any(r["status"] == "unsupported" for r in case["rows"])


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {k for item in value.values() for k in _keys(item)}
    if isinstance(value, list):
        return {k for item in value for k in _keys(item)}
    return set()


def test_topology_symmetry_instrument_04_no_layout_enters_the_physics(
    built: dict[str, Any],
) -> None:
    """The data carries no node positions; the layout boundary says so.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    keys = _keys(built["templates"])
    assert not {key for key in keys if "layout" in key or "position" in key}
    assert "carry no physical meaning" in built["layout_boundary"]


def test_chimera_clusters_and_persistence_agree_with_independent_formulas(
    built: dict[str, Any],
) -> None:
    """Cluster order parameters and exact persistence match direct computations.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    chimera = built["templates"]["chimera"]
    run = generate_two_population_chimera(templates.CHIMERA_CONFIG)
    nodes = [cluster["nodes"] for cluster in chimera["clusters"]]
    assert sorted(node for group in nodes for node in group) == list(range(run.phases.shape[1]))
    for cluster in chimera["clusters"]:
        series = np.abs(np.mean(np.exp(1j * run.settled_phases[:, cluster["nodes"]]), axis=1))
        assert cluster["mean_order_parameter"] == pytest.approx(np.mean(series), abs=1e-12)
    coherent, drifting = (cluster["mean_order_parameter"] for cluster in chimera["clusters"])
    assert coherent > drifting
    persistence = built["templates"]["persistence"]
    exact = persistence["exact_example"]
    h0, h1 = exact["summary"]["H0"], exact["summary"]["H1"]
    assert h0["essential"] == 1 and h0["persistent_finite"] == templates.CIRCLE_POINTS - 1
    assert np.allclose(h0["finite_deaths"], exact["analytic_spacing"], atol=1e-12)
    assert h1["persistent_finite"] == 1
    assert persistence["chimera_final_phases"]["H0"]["pairs"] == len(chimera["final_phases"])


def test_the_build_leaves_and_guards_the_global_random_state(
    built: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded statement holds, and a template that draws globally is refused.

    Parameters
    ----------
    built
        Fresh artefact payload.
    monkeypatch
        Replaces one template with one that consumes the global generator.

    """
    assert built["global_random_state_unchanged"] is True
    monkeypatch.setattr(artifact, "sensitivity_template", lambda: {"draw": np.random.random()})
    with pytest.raises(RuntimeError, match="changed the caller's global random state"):
        artifact.build_topology_instrument_artifact()


def test_committed_data_and_command_line_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse changed data through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert artifact.main(["--check"]) == 0
    target = tmp_path / "topology.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["templates"]["relabelling"]["permutation"] = [0, 1, 2, 3]
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["topology", "--check"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module(
            "scpn_quantum_control.studio.topology_instrument_artifact", run_name="__main__"
        )
    assert exit_status.value.code == 0
