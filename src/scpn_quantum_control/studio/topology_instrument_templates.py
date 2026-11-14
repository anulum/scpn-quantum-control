# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned symmetry, chimera and topology templates
"""Build one bounded template per symmetry, chimera and topology owner.

Every function calls a public owner on a small finite example and returns
JSON-ready source values. Exact examples are labelled exact, finite summaries
are labelled finite, and every refusal is the owner's own message. Nothing
here simulates, scores or differentiates on its own account.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..analysis.magnetisation_sectors import eigh_by_magnetisation
from ..analysis.symmetry_sectors import eigh_by_sector
from ..chimera_control import (
    CHIMERA_CONTROL_CLAIM_BOUNDARY,
    SyntheticChimeraConfig,
    generate_two_population_chimera,
    measure_multiscale_order_parameters,
)
from ..dla_topology_control import (
    DLA_TOPOLOGY_CLAIM_BOUNDARY,
    ParitySector,
    ParitySectorProjector,
    UnsupportedDifferentiableConstraintError,
    topology_projection_jvp,
    topology_projection_support,
)
from ..kuramoto_core import (
    KuramotoProblem,
    build_kuramoto_problem,
    compile_dense_hamiltonian,
    measure_order_parameter,
)
from ..phase.synchronisation_witness import (
    geodesic_phase_distance_matrix,
    vietoris_rips_persistence,
)
from ..topology_control.constraints import CouplingGraphBounds, TopologyConstraintLedger

FloatArray = NDArray[np.float64]

RELABEL_COUPLING: Final = (
    (0.0, 0.8, 0.0, 0.3),
    (0.8, 0.0, 0.5, 0.0),
    (0.0, 0.5, 0.0, 0.9),
    (0.3, 0.0, 0.9, 0.0),
)
RELABEL_OMEGA: Final = (-0.4, 0.1, 0.25, 0.6)
INITIAL_ANGLES: Final = (0.3, 1.1, 1.9, 2.4)
RELABEL_PERMUTATION: Final = (2, 0, 3, 1)
EVOLUTION_TIME: Final = 0.7
RELABEL_TOLERANCE: Final = 1.0e-12
RING_COUPLING: Final = (
    (0.0, 1.0, 0.0, 1.0),
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 1.0),
    (1.0, 0.0, 1.0, 0.0),
)
RING_OMEGA: Final = (0.0, 0.0, 0.0, 0.0)
EVEN_BASIS_STATE: Final = "0011"
SECTOR_TOLERANCE: Final = 1.0e-10
CHIMERA_CONFIG: Final = SyntheticChimeraConfig(population_size=8, steps=400, settle_steps=100)
CIRCLE_POINTS: Final = 6
PERSISTENT_LIFETIME: Final = 1.0e-9
SENSITIVITY_POINT: Final = ((0.0, 0.4, 0.6), (0.4, 0.0, 0.3), (0.6, 0.3, 0.0))
SENSITIVITY_TANGENT: Final = ((0.0, 1.0, 0.0), (1.0, 0.0, -1.0), (0.0, -1.0, 0.0))
LEDGER_BOUNDS: Final = (0.0, 1.0)
CONNECTIVITY_THRESHOLD: Final = 0.5
EXACT_EVOLUTION: Final = "exact dense matrix exponential of the compiled XY Hamiltonian"


def _matrix(rows: tuple[tuple[float, ...], ...]) -> FloatArray:
    return np.array(rows, dtype=np.float64)


def _product_state(angles: FloatArray) -> Statevector:
    circuit = QuantumCircuit(len(angles))
    for qubit, angle in enumerate(angles):
        circuit.ry(float(angle), qubit)
    return Statevector.from_instruction(circuit)


def _evolve(problem: KuramotoProblem, state: Statevector) -> Statevector:
    hamiltonian = compile_dense_hamiltonian(problem)
    propagator = scipy.linalg.expm(-1j * EVOLUTION_TIME * hamiltonian)
    return Statevector(propagator @ state.data)


def _order(problem: KuramotoProblem, angles: FloatArray) -> dict[str, float]:
    magnitude, phase = measure_order_parameter(problem, _evolve(problem, _product_state(angles)))
    return {"magnitude": float(magnitude), "phase": float(phase)}


def relabelling_template() -> dict[str, object]:
    """Measure the order parameter before and after a relabelling of the nodes.

    Returns
    -------
    dict[str, object]
        The coupling, frequencies, initial angles and permutation, the owner's
        order parameter for both labellings after the same exact evolution,
        and their differences against the recorded tolerance.

    """
    coupling, omega = _matrix(RELABEL_COUPLING), np.array(RELABEL_OMEGA)
    angles, order = np.array(INITIAL_ANGLES), np.array(RELABEL_PERMUTATION)
    permute = np.eye(len(order))[order]
    original = _order(build_kuramoto_problem(coupling, omega), angles)
    relabelled = _order(
        build_kuramoto_problem(permute @ coupling @ permute.T, permute @ omega), permute @ angles
    )
    phase_change = np.angle(np.exp(1j * (relabelled["phase"] - original["phase"])))
    return {
        "owner": "scpn_quantum_control.kuramoto_core",
        "coupling": coupling.tolist(),
        "omega": omega.tolist(),
        "initial_angles": angles.tolist(),
        "permutation": order.tolist(),
        "evolution": {"method": EXACT_EVOLUTION, "time": EVOLUTION_TIME},
        "original": original,
        "relabelled": relabelled,
        "magnitude_difference": abs(relabelled["magnitude"] - original["magnitude"]),
        "phase_difference": abs(float(phase_change)),
        "tolerance": RELABEL_TOLERANCE,
    }


def _sector_spectra(coupling: FloatArray, omega: FloatArray) -> dict[str, object]:
    dense = np.linalg.eigvalsh(compile_dense_hamiltonian(build_kuramoto_problem(coupling, omega)))
    parity = eigh_by_sector(coupling, omega)
    magnetisation = eigh_by_magnetisation(coupling, omega)
    by_magnetisation = magnetisation["results"]
    return {
        "dense_eigenvalues": dense.tolist(),
        "parity_dimensions": {
            "even": len(parity["eigvals_even"]),
            "odd": len(parity["eigvals_odd"]),
        },
        "magnetisation_dimensions": {
            str(sector): len(by_magnetisation[sector]["eigvals"])
            for sector in sorted(by_magnetisation)
        },
        "parity_max_deviation": float(np.max(np.abs(parity["eigvals_all"] - dense))),
        "magnetisation_max_deviation": float(
            np.max(np.abs(np.sort(magnetisation["eigvals_all"]) - dense))
        ),
        "ground_energy": float(parity["ground_energy"]),
        "ground_parity": ("even", "odd")[int(parity["ground_parity"])],
    }


def _parity_weights(state: Statevector) -> dict[str, float]:
    even = ParitySectorProjector(n_qubits=state.num_qubits, sector=ParitySector.EVEN)
    odd = ParitySectorProjector(n_qubits=state.num_qubits, sector=ParitySector.ODD)
    leakage = even.leakage_value_and_gradient(state.data)
    return {
        "even": float(np.vdot(even.project(state.data), even.project(state.data)).real),
        "odd": float(np.vdot(odd.project(state.data), odd.project(state.data)).real),
        "leakage_from_even": float(leakage.value),
    }


def sector_template() -> dict[str, object]:
    """Compare sector spectra with the dense spectrum and track parity weights.

    Returns
    -------
    dict[str, object]
        The uniform-ring spectra by parity and magnetisation sector against the
        dense reference, and parity weights and leakage of an even and a mixed
        initial state before and after the same exact evolution.

    """
    coupling, omega = _matrix(RING_COUPLING), np.array(RING_OMEGA)
    problem = build_kuramoto_problem(coupling, omega)
    states = {
        "even_basis_state": Statevector.from_label(EVEN_BASIS_STATE),
        "mixed_product_state": _product_state(np.array(INITIAL_ANGLES)),
    }
    return {
        "owners": [
            "scpn_quantum_control.analysis.symmetry_sectors",
            "scpn_quantum_control.analysis.magnetisation_sectors",
            "scpn_quantum_control.dla_topology_control",
        ],
        "coupling": coupling.tolist(),
        "omega": omega.tolist(),
        "spectra": _sector_spectra(coupling, omega),
        "tolerance": SECTOR_TOLERANCE,
        "weights": {
            name: {
                "before": _parity_weights(state),
                "after": _parity_weights(_evolve(problem, state)),
            }
            for name, state in states.items()
        },
        "evolution": {"method": EXACT_EVOLUTION, "time": EVOLUTION_TIME},
        "claim_boundary": DLA_TOPOLOGY_CLAIM_BOUNDARY,
    }


def _final_phases() -> FloatArray:
    run = generate_two_population_chimera(CHIMERA_CONFIG)
    final: FloatArray = np.mod(run.settled_phases[-1], 2.0 * np.pi)
    return final


def chimera_template() -> dict[str, object]:
    """Report the clusters and order parameters of the finite synthetic chimera.

    Returns
    -------
    dict[str, object]
        The configuration and source, each population (cluster) with its mean
        order parameter, the chimera and metastability indices, the mean global
        order parameter and the settled final phases.

    """
    run = generate_two_population_chimera(CHIMERA_CONFIG)
    report = measure_multiscale_order_parameters(run.settled_phases, run.hierarchy)
    population = report.levels[0]
    communities = run.hierarchy.levels[0].communities
    return {
        "owner": "scpn_quantum_control.chimera_control",
        "source": run.source,
        "config": {
            "regime": CHIMERA_CONFIG.regime.value,
            "population_size": CHIMERA_CONFIG.population_size,
            "steps": CHIMERA_CONFIG.steps,
            "settle_steps": CHIMERA_CONFIG.settle_steps,
            "dt": CHIMERA_CONFIG.dt,
            "seed": CHIMERA_CONFIG.seed,
        },
        "clusters": [
            {"nodes": list(nodes), "mean_order_parameter": float(mean)}
            for nodes, mean in zip(communities, population.mean_by_community, strict=True)
        ],
        "chimera_index": float(population.chimera_index),
        "metastability": float(population.community_metastability),
        "global_order_parameter_mean": float(np.mean(report.global_order_parameter)),
        "final_phases": _final_phases().tolist(),
        "claim_boundary": CHIMERA_CONTROL_CLAIM_BOUNDARY,
    }


def _persistence(phases: FloatArray) -> dict[str, object]:
    pairs = vietoris_rips_persistence(geodesic_phase_distance_matrix(phases), max_dimension=1)
    summary: dict[str, object] = {}
    for dimension, rows in sorted(pairs.items()):
        finite = rows[np.isfinite(rows[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0]
        lasting = lifetimes[lifetimes > PERSISTENT_LIFETIME]
        summary[f"H{dimension}"] = {
            "pairs": len(rows),
            "essential": int(np.sum(~np.isfinite(rows[:, 1]))),
            "persistent_finite": len(lasting),
            "longest_finite_lifetime": float(np.max(lasting)) if len(lasting) else 0.0,
            "finite_deaths": sorted(float(death) for death in finite[:, 1]),
        }
    return summary


def persistence_template() -> dict[str, object]:
    """Summarise exact Vietoris–Rips persistence of finite phase samples.

    Returns
    -------
    dict[str, object]
        An exact example (evenly spaced phases with its analytic spacing) and
        the summary of the chimera's final phases, each labelled finite.

    """
    circle = 2.0 * np.pi * np.arange(CIRCLE_POINTS) / CIRCLE_POINTS
    return {
        "owner": "scpn_quantum_control.phase.synchronisation_witness",
        "method": "exact GF(2) Vietoris-Rips persistence of geodesic phase distances",
        "lifetime_threshold": PERSISTENT_LIFETIME,
        "exact_example": {
            "phases": circle.tolist(),
            "analytic_spacing": 2.0 * np.pi / CIRCLE_POINTS,
            "summary": _persistence(circle),
        },
        "chimera_final_phases": _persistence(_final_phases()),
        "boundary": (
            "finite summaries of sampled phases; not a statement about the continuous "
            "dynamics, and not differentiated"
        ),
    }


def _support_case(
    name: str, ledger: TopologyConstraintLedger, point: FloatArray
) -> dict[str, object]:
    report = topology_projection_support(ledger, point)
    case: dict[str, object] = {
        "case": name,
        "matrix": point.tolist(),
        "derivative_supported": report.derivative_supported,
        "rows": [
            {
                "capability": row.capability,
                "status": row.status,
                "differentiability": row.differentiability.value,
                "evidence": row.evidence,
            }
            for row in report.rows
        ],
    }
    try:
        jvp = topology_projection_jvp(ledger, point, _matrix(SENSITIVITY_TANGENT))
    except UnsupportedDifferentiableConstraintError as refusal:
        return {**case, "projected_tangent": None, "refusal": str(refusal)}
    return {**case, "projected_tangent": jvp.projected_tangent.tolist(), "refusal": None}


def sensitivity_template() -> dict[str, object]:
    """Classify fixed-active-set derivatives at interior, kink and discrete points.

    Returns
    -------
    dict[str, object]
        The ledger policy, the tangent and four cases: an interior point with
        its local JVP, a coupling at the clipping bound, a removed edge and a
        connectivity threshold, each with the owner's support rows and refusal.

    """
    lower, upper = LEDGER_BOUNDS
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(lower, upper), sign_policy="nonnegative"
    )
    interior = _matrix(SENSITIVITY_POINT)
    clipped, removed = interior.copy(), interior.copy()
    clipped[0, 2] = clipped[2, 0] = upper
    removed[0, 2] = removed[2, 0] = lower
    threshold = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(lower, upper),
        sign_policy="nonnegative",
        algebraic_connectivity_min=CONNECTIVITY_THRESHOLD,
    )
    return {
        "owner": "scpn_quantum_control.dla_topology_control.projection",
        "ledger": {"bounds": [lower, upper], "sign_policy": "nonnegative"},
        "tangent": _matrix(SENSITIVITY_TANGENT).tolist(),
        "cases": [
            _support_case("interior point", ledger, interior),
            _support_case("coupling at the clipping bound", ledger, clipped),
            _support_case("edge removed", ledger, removed),
            _support_case("connectivity threshold", threshold, interior),
        ],
        "claim_boundary": DLA_TOPOLOGY_CLAIM_BOUNDARY,
    }


__all__ = [
    "chimera_template",
    "persistence_template",
    "relabelling_template",
    "sector_template",
    "sensitivity_template",
]
