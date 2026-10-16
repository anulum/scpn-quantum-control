# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual open-system ensemble conformance
"""Qualify original public solvers against independent population references.

The source lowering operator is |1><0|. For H=0 and initial |0>, the surviving
population is exp(-gamma*t); R is not a substitute for this observable.
The fixed study retains all200 replicates, including failed interval coverage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import scipy
from scipy.linalg import expm
from scipy.stats import binomtest

from scpn_quantum_control.open_system_mcwf_product import decide_open_system_path
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
from scpn_quantum_control.phase.lindblad_engine import LindbladSyncEngine
from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory


def _populations(seed: int, count: int, dt: float) -> tuple[list[int], list[float], list[int]]:
    """Capture actual public trajectories with original child-seed ordering."""
    seeds = [int(value) for value in np.random.default_rng(seed).integers(0, 2**31, size=count)]
    populations: list[float] = []
    jumps: list[int] = []
    for child_seed in seeds:
        result = mcwf_trajectory(
            np.zeros((1, 1)),
            np.zeros(1),
            gamma_amp=0.7,
            gamma_deph=0.0,
            t_max=1.0,
            dt=dt,
            seed=child_seed,
            max_dense_gib=0.001,
        )
        state = np.asarray(result["psi_final"], dtype=np.complex128)
        jump_count = int(result["n_jumps"])
        population = float(abs(state[0]) ** 2)
        assert jump_count in (0, 1)
        assert abs(population - (1 - jump_count)) < 1e-12
        assert abs(float(np.vdot(state, state).real) - 1.0) < 1e-12
        populations.append(population)
        jumps.append(jump_count)
    return seeds, populations, jumps


def test_open_system_ensemble_conformance_01(tmp_path: Path) -> None:
    """Compare actual step refinements and density outputs with analytic amplitude decay."""
    target = math.exp(-0.7)
    rows: list[dict[str, object]] = []
    for steps in (16, 32, 64):
        density = LindbladKuramotoSolver(
            1, np.zeros((1, 1)), np.zeros(1), gamma_amp=0.7, max_dense_gib=0.001
        ).run(1.0, 1.0 / steps, method="DOP853", atol=1e-13, rtol=1e-11)
        rho = np.asarray(density["rho_final"], dtype=np.complex128)
        np.testing.assert_allclose(rho[0, 0].real, target, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
        assert float(np.linalg.eigvalsh(rho).min()) >= -1e-12
        seeds, populations, jumps = _populations(1701, 256, 1.0 / steps)
        survivors = sum(1 - value for value in jumps)
        interval = binomtest(survivors, 256).proportion_ci(
            confidence_level=1.0 - 0.05 / 3.0, method="exact"
        )
        rows.append(
            {
                "steps": steps,
                "density_population": float(rho[0, 0].real),
                "trajectory_seeds": seeds,
                "populations": populations,
                "jumps": jumps,
                "interval": [interval.low, interval.high],
                "analytic_population": target,
                "scope": "Analytic one-qubit population; density dt is output spacing, not RK step.",
            }
        )
        (tmp_path / "step_refinements.json").write_text(json.dumps(rows, indent=2))
        assert interval.low <= target <= interval.high


def test_open_system_ensemble_conformance_02(tmp_path: Path) -> None:
    """Retain fixed200-by1000 actual trajectories and evaluate exact interval coverage."""
    target = math.exp(-0.7)
    protocol = {
        "ensemble_seeds": list(range(200)),
        "trajectories_per_replicate": 1000,
        "sample_refinements": [64, 256, 1000],
        "gamma_amp": 0.7,
        "gamma_deph": 0.0,
        "T": 1.0,
        "dt": 1.0,
        "initial_state": "|0>",
        "lowering_operator": "|1><0|",
        "estimand": "surviving population of |0>",
        "interval": "SciPy exact Clopper-Pearson",
        "confidence_level": 0.95,
        "acceptance": "Exact95% interval for observed coverage must not lie wholly below0.95; no universal calibration claim.",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "step_scope": "This eigenstate population has no finite-step jump-location bias; general noncommuting dynamics do not inherit that claim.",
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol, indent=2))
    covered = 0
    squared_errors: dict[int, list[float]] = {64: [], 256: [], 1000: []}
    with (tmp_path / "replicates.jsonl").open("w") as raw:
        for seed in range(200):
            children, populations, jumps = _populations(seed, 1000, 1.0)
            survivors = sum(1 - value for value in jumps)
            interval = binomtest(survivors, 1000).proportion_ci(
                confidence_level=0.95, method="exact"
            )
            includes_target = bool(interval.low <= target <= interval.high)
            covered += includes_target
            for count in squared_errors:
                squared_errors[count].append((float(np.mean(populations[:count])) - target) ** 2)
            raw.write(
                json.dumps(
                    {
                        "seed": seed,
                        "trajectory_seeds": children,
                        "populations": populations,
                        "jumps": jumps,
                        "survivors": survivors,
                        "interval": [interval.low, interval.high],
                        "covers": includes_target,
                    }
                )
                + "\n"
            )
            raw.flush()
    coverage_interval = binomtest(covered, 200).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    mse = {str(count): float(np.mean(errors)) for count, errors in squared_errors.items()}
    report = {
        "replicates": 200,
        "covered": covered,
        "observed_coverage": covered / 200,
        "nominal_coverage": 0.95,
        "coverage_interval": [coverage_interval.low, coverage_interval.high],
        "analytic_population": target,
        "mean_squared_error_by_sample_count": mse,
        "claim_boundary": "Fixed synthetic Markovian population study only; sampling coverage is separate from time error or hardware fidelity.",
    }
    (tmp_path / "summary.json").write_text(json.dumps(report, indent=2))
    records = [
        json.loads(line) for line in (tmp_path / "replicates.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in records] == list(range(200))
    assert all(len(row["populations"]) == len(row["trajectory_seeds"]) == 1000 for row in records)
    assert coverage_interval.high >= 0.95
    assert mse["64"] > mse["256"] > mse["1000"]


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf")])
def test_open_system_ensemble_conformance_03(rate: float) -> None:
    """Refuse invalid rates through all three actual model entry points."""
    coupling = np.zeros((1, 1))
    frequencies = np.zeros(1)
    with pytest.raises(ValueError, match="gamma"):
        mcwf_trajectory(coupling, frequencies, gamma_amp=rate, seed=7)
    with pytest.raises(ValueError, match="gamma"):
        LindbladKuramotoSolver(1, coupling, frequencies, gamma_amp=rate)
    with pytest.raises(ValueError, match="gamma"):
        LindbladSyncEngine(coupling, frequencies, gamma=rate)


def test_open_system_ensemble_conformance_04() -> None:
    """Preserve caller RNG/input state and the original unqualified-process refusals."""
    before = np.random.get_state(legacy=True)
    assert isinstance(before, tuple)
    coupling = np.zeros((1, 1))
    frequencies = np.zeros(1)
    first = mcwf_trajectory(coupling, frequencies, gamma_amp=0.7, t_max=0.1, seed=7)
    second = mcwf_trajectory(coupling, frequencies, gamma_amp=0.7, t_max=0.1, seed=7)
    np.testing.assert_array_equal(first["psi_final"], second["psi_final"])
    after = np.random.get_state(legacy=True)
    assert isinstance(after, tuple)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(coupling, np.zeros((1, 1)))
    np.testing.assert_array_equal(frequencies, np.zeros(1))
    for decision in (
        decide_open_system_path("mcwf_ensemble", unseeded_variance_claim=True),
        decide_open_system_path("mcwf_ensemble", invent_green_non_markovian=True),
        decide_open_system_path("mcwf_ensemble", invent_green_non_cp=True),
        decide_open_system_path("mcwf_ensemble", invent_green_hardware_noise=True),
    ):
        assert not decision.allowed
    with pytest.raises((ValueError, TypeError)):
        from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_probe

        materialise_mcwf_ensemble_probe(seed=cast(int, None), n_trajectories=1)


def test_transfer_ensemble_has_nontrivial_population_and_same_density_generator() -> None:
    """Compare an excited transfer state with an independent small Liouvillian exponential."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    frequencies = np.zeros(2)
    initial = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    rho0 = np.outer(initial, initial.conj())
    engine = LindbladSyncEngine(coupling, frequencies, gamma=0.7)
    density = engine.evolve(
        1.0,
        64,
        method="density_matrix",
        initial_state=rho0,
        density_atol=1e-13,
        density_rtol=1e-11,
    )
    hamiltonian = engine.H_dense
    assert hamiltonian is not None
    identity = np.eye(4)
    generator = -1j * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
    for source, target in ((1, 2), (2, 1)):
        jump = np.zeros((4, 4), dtype=np.complex128)
        jump[target, source] = 1.0
        damping = jump.conj().T @ jump
        generator += 0.7 * (
            np.kron(jump, jump.conj())
            - 0.5 * (np.kron(damping, identity) + np.kron(identity, damping.T))
        )
    reference = (expm(generator) @ rho0.ravel()).reshape(4, 4)
    np.testing.assert_allclose(density["final_state"], reference, atol=1e-10, rtol=1e-8)
    trajectory = engine.evolve(1.0, 64, n_traj=1024, initial_state=initial, seed=1701)
    final = np.asarray(trajectory["final_state"], dtype=np.complex128)
    assert 0.1 < final[1, 1].real < 0.9
    np.testing.assert_allclose(final, reference, atol=0.06)
    np.testing.assert_allclose(np.trace(final), 1.0, atol=1e-12)
    np.testing.assert_allclose(final, final.conj().T, atol=1e-12)
    assert float(np.linalg.eigvalsh(final).min()) >= -1e-12
