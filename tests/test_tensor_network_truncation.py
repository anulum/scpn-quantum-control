# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor-network truncation contracts
"""Independent state and observable references for the public MPS backend."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.mps_evolution import (
    dmrg_ground_state,
    export_mps_state,
    is_quimb_available,
    tebd_evolution,
)


@pytest.fixture
def quimb_runtime() -> ModuleType:
    """Require the genuine optional runtime in its owning CI cohort.

    Returns
    -------
    ModuleType
        Installed Quimb module; the required native job verifies its pin before
        running this file. Base profiles cannot qualify positive runtime cases.

    """
    return cast(ModuleType, pytest.importorskip("quimb"))


def test_tensor_network_truncation_01(quimb_runtime: ModuleType) -> None:
    """Keep a product state exact with bond one and compressed observables.

    Parameters
    ----------
    quimb_runtime
        Genuine tensor-network runtime admitted by the owning CI job.

    """
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    original = initial.tobytes()
    result = tebd_evolution(
        np.zeros((2, 2)),
        np.zeros(2),
        t_max=0.1,
        dt=0.05,
        bond_dim=1,
        cutoff=0.0,
        initial_state=initial,
    )
    assert result["bond_dims_final"] == [1]
    diagnostics = result["diagnostics"]
    assert diagnostics.discarded_weight == 0.0
    assert diagnostics.dense_result_exported is False
    assert diagnostics.contraction_strategy == "canonical nearest-neighbour TEBD sweeps"
    for observation in diagnostics.observations:
        assert observation.bond_dims == (1,)
        np.testing.assert_allclose(observation.local_z, [1.0, 1.0], atol=1e-12, rtol=0)
        assert observation.norm == pytest.approx(1.0, abs=1e-12)
    assert initial.tobytes() == original


def test_tensor_network_truncation_02(quimb_runtime: ModuleType) -> None:
    """Require bond two for exact Bell preparation and preserve refusals.

    Parameters
    ----------
    quimb_runtime
        Genuine tensor-network runtime admitted by the owning CI job.

    """
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    original = bell.tobytes()
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        tebd_evolution(
            np.zeros((2, 2)),
            np.zeros(2),
            t_max=0.1,
            dt=0.05,
            bond_dim=1,
            cutoff=0.0,
            initial_state=bell,
        )
    result = tebd_evolution(
        np.zeros((2, 2)),
        np.zeros(2),
        t_max=0.1,
        dt=0.05,
        bond_dim=2,
        cutoff=0.0,
        initial_state=bell,
    )
    assert result["bond_dims_final"] == [2]
    np.testing.assert_allclose(
        result["mps_final"].to_dense().reshape(-1), bell, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result["diagnostics"].observations[-1].local_z, [0.0, 0.0], rtol=0.0, atol=1e-12
    )
    assert bell.tobytes() == original


def test_tensor_network_truncation_03(quimb_runtime: ModuleType) -> None:
    """Resolve an entangling XY trajectory against its analytic solution.

    Parameters
    ----------
    quimb_runtime
        Genuine tensor-network runtime admitted by the owning CI job.

    """
    coupling = np.array([[0.0, 2.0], [2.0, 0.0]])
    initial = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    results = [
        tebd_evolution(
            coupling,
            np.zeros(2),
            t_max=1.0,
            dt=0.05,
            bond_dim=bond,
            cutoff=0.0,
            initial_state=initial,
        )
        for bond in [1, 2]
    ]
    exact = np.array([0.0, np.cos(1.0), 1j * np.sin(1.0), 0.0])
    expected_z = np.array([np.cos(2.0), -np.cos(2.0)])
    high = results[1]
    np.testing.assert_allclose(
        high["mps_final"].to_dense().reshape(-1), exact, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        high["diagnostics"].observations[-1].local_z, expected_z, rtol=1e-8, atol=1e-10
    )
    errors = [
        float(
            np.max(np.abs(np.array(result["diagnostics"].observations[-1].local_z) - expected_z))
        )
        for result in results
    ]
    assert errors[0] > 0.1
    assert errors[1] < 1e-8
    assert results[0]["diagnostics"].discarded_weight > 0.0
    assert high["diagnostics"].discarded_weight < 1e-20
    assert "not a global" in high["diagnostics"].discarded_weight_scope
    assert high["diagnostics"].observations[0].bond_dims == (1,)
    assert high["diagnostics"].observations[-1].bond_dims == (2,)


def test_tensor_network_truncation_04() -> None:
    """Refuse an insufficient compressed memory cap before any runtime work."""
    coupling = np.zeros((2, 2))
    frequencies = np.zeros(2)
    before = (coupling.tobytes(), frequencies.tobytes())
    with pytest.raises(MemoryError, match="MPS"):
        tebd_evolution(coupling, frequencies, max_memory_bytes=1)
    with pytest.raises(MemoryError, match="MPS"):
        dmrg_ground_state(coupling, frequencies, max_memory_bytes=1)
    assert (coupling.tobytes(), frequencies.tobytes()) == before
    if is_quimb_available():
        recovered = tebd_evolution(
            coupling,
            frequencies,
            t_max=0.1,
            dt=0.05,
            bond_dim=1,
            max_memory_bytes=1024 * 1024,
        )
        assert recovered["diagnostics"].memory_budget_bytes <= 1024 * 1024
    else:
        with pytest.raises(ImportError, match="quimb not installed"):
            tebd_evolution(coupling, frequencies, max_memory_bytes=1024 * 1024)


@pytest.mark.parametrize(
    "field,value",
    [
        ("t_max", -1.0),
        ("t_max", np.nan),
        ("t_max", np.inf),
        ("t_max", True),
        ("dt", 0.0),
        ("dt", -1.0),
        ("dt", np.nan),
        ("dt", np.inf),
        ("dt", True),
        ("cutoff", -1.0),
        ("cutoff", np.nan),
        ("cutoff", np.inf),
        ("cutoff", True),
    ],
)
def test_tebd_invalid_grid_and_cutoff_refuse(field: str, value: float) -> None:
    """Refuse malformed time or truncation settings before solver allocation.

    Parameters
    ----------
    field
        Invalid public setting under test.
    value
        Deliberately invalid scalar, including boolean metadata.

    """
    coupling, frequencies = np.zeros((2, 2)), np.zeros(2)
    with pytest.raises(ValueError, match=field):
        if field == "t_max":
            tebd_evolution(coupling, frequencies, t_max=value)
        elif field == "dt":
            tebd_evolution(coupling, frequencies, dt=value)
        else:
            tebd_evolution(coupling, frequencies, cutoff=value)
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))
    np.testing.assert_array_equal(frequencies, np.zeros(2))


@pytest.mark.parametrize("order", [0, 1, 3, True, 4.0])
def test_tebd_invalid_order_refuses(order: int) -> None:
    """Reject unsupported Suzuki orders before starting an MPS run.

    Parameters
    ----------
    order
        Unsupported or non-integral order metadata.

    """
    with pytest.raises(ValueError, match="order"):
        tebd_evolution(np.zeros((2, 2)), np.zeros(2), order=order)


def test_tebd_excessive_grid_refuses_without_array_allocation() -> None:
    """Refuse non-finite step counts and bounded but excessive result buffers."""
    coupling, frequencies = np.zeros((2, 2)), np.zeros(2)
    with pytest.raises(ValueError, match="time grid"):
        tebd_evolution(coupling, frequencies, t_max=1e308, dt=1e-300)
    with pytest.raises(MemoryError, match="observation"):
        tebd_evolution(coupling, frequencies, t_max=1e20, dt=1.0, max_memory_bytes=1024 * 1024)


@pytest.mark.parametrize("sweeps", [0, -1, False, 1.5])
def test_dmrg_invalid_sweep_count_refuses(sweeps: int) -> None:
    """Reject an invalid DMRG stopping budget before constructing its state.

    Parameters
    ----------
    sweeps
        Deliberately malformed sweep-count metadata.

    """
    with pytest.raises(ValueError, match="max_sweeps"):
        dmrg_ground_state(np.zeros((2, 2)), np.zeros(2), max_sweeps=sweeps)


@pytest.mark.parametrize("cutoff", [-1.0, np.nan, np.inf, True])
def test_dmrg_invalid_cutoff_refuses(cutoff: float) -> None:
    """Refuse malformed DMRG truncation settings rather than coercing them.

    Parameters
    ----------
    cutoff
        Invalid finite-threshold or boolean metadata.

    """
    with pytest.raises(ValueError, match="cutoff"):
        dmrg_ground_state(np.zeros((2, 2)), np.zeros(2), cutoff=cutoff)


@pytest.mark.parametrize("coupling", [np.zeros(2), np.zeros((2, 3))])
def test_public_solver_shape_errors_preserve_model(coupling: np.ndarray) -> None:
    """Both public solvers reject non-square couplings before any runtime work.

    Parameters
    ----------
    coupling
        Deliberately malformed model array.

    """
    original = coupling.tobytes()
    with pytest.raises(ValueError, match="square"):
        tebd_evolution(coupling, np.zeros(2))
    with pytest.raises(ValueError, match="square"):
        dmrg_ground_state(coupling, np.zeros(2))
    assert coupling.tobytes() == original


def test_zero_duration_and_explicit_dense_export(quimb_runtime: ModuleType) -> None:
    """Return one initial observation and budget every explicit dense export.

    Parameters
    ----------
    quimb_runtime
        Genuine Quimb runtime admitted by the owning CI job.

    """
    result = tebd_evolution(np.zeros((2, 2)), np.zeros(2), t_max=0.0, bond_dim=1)
    np.testing.assert_array_equal(result["times"], [0.0])
    assert result["diagnostics"].truncation_events == 0
    before = export_mps_state(result["mps_final"], max_memory_bytes=128)
    np.testing.assert_allclose(before, [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
    with pytest.raises(MemoryError, match="explicit dense"):
        export_mps_state(result["mps_final"], max_memory_bytes=127)
    for invalid in [0, -1, True]:
        with pytest.raises(ValueError, match="positive integer"):
            export_mps_state(result["mps_final"], max_memory_bytes=invalid)
    np.testing.assert_allclose(export_mps_state(result["mps_final"]), before, rtol=0.0, atol=1e-12)


def test_two_site_ground_state_matches_analytic_energy(quimb_runtime: ModuleType) -> None:
    """Keep DMRG energy and its final state tied to the same independent model.

    Parameters
    ----------
    quimb_runtime
        Genuine Quimb runtime admitted by the owning CI job.

    """
    coupling = np.array([[0.0, 2.0], [2.0, 0.0]])
    result = dmrg_ground_state(coupling, np.zeros(2), bond_dim=2, cutoff=0.0, max_sweeps=8)
    assert result["energy"] == pytest.approx(-1.0, abs=1e-12)
    state = export_mps_state(result["mps"])
    exchange = np.zeros((4, 4))
    exchange[1, 2] = exchange[2, 1] = -1.0
    assert np.real(np.vdot(state, exchange @ state)) == pytest.approx(result["energy"], abs=1e-12)
    assert result["diagnostics"].discarded_weight is None
    assert "does not report" in result["diagnostics"].discarded_weight_scope


def _four_site_reference() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Build the independent spin-half Pauli Hamiltonian in MPS site order."""
    n = 4
    coupling = np.zeros((n, n))
    for site in range(n - 1):
        coupling[site, site + 1] = coupling[site + 1, site] = 1.4
    frequencies = np.array([0.0, 0.2, -0.1, 0.3])
    initial = np.zeros(16, dtype=np.complex128)
    initial[5] = 1.0  # |0101>, site zero is the most significant bit.
    paulis = [
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.diag(np.array([1, -1], dtype=np.complex128)),
    ]
    identity = np.eye(2, dtype=np.complex128)
    hamiltonian = np.zeros((16, 16), dtype=np.complex128)
    for site in range(n):
        term = np.array([[1.0]], dtype=np.complex128)
        for position in range(n):
            term = np.asarray(
                np.kron(term, paulis[2] if position == site else identity), dtype=np.complex128
            )
        hamiltonian -= frequencies[site] * term / 2
    for site in range(n - 1):
        for pauli in paulis[:2]:
            term = np.array([[1.0]], dtype=np.complex128)
            for position in range(n):
                term = np.asarray(
                    np.kron(term, pauli if position in (site, site + 1) else identity),
                    dtype=np.complex128,
                )
            hamiltonian -= coupling[site, site + 1] * term / 4
    return coupling, frequencies, initial, hamiltonian


def _aer_mps_reference(
    initial: NDArray[np.complex128],
    hamiltonian: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Execute the matched four-site Hamiltonian on actual Aer MPS."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HamiltonianGate
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(4)
    circuit.initialize(initial)
    circuit.append(HamiltonianGate(hamiltonian, time=1.0), range(4))
    circuit.save_statevector()
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=4,
        matrix_product_state_truncation_threshold=0.0,
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=1, seed_simulator=0).result()
    assert result.success
    assert result.results[0].metadata["method"] == "matrix_product_state"
    return np.asarray(result.data(0)["statevector"], dtype=np.complex128)


def test_actual_aer_mps_matches_independent_four_site_reference() -> None:
    """Verify the actual matched Aer candidate against a separate dense solver."""
    from scipy.linalg import expm

    _, _, initial, hamiltonian = _four_site_reference()
    expected = expm(-1j * hamiltonian) @ initial
    np.testing.assert_allclose(
        _aer_mps_reference(initial, hamiltonian), expected, rtol=1e-8, atol=1e-10
    )


def test_entanglement_growth_convergence_and_loss_of_storage_advantage(
    quimb_runtime: ModuleType,
) -> None:
    """Measure bond and timestep convergence with an actual matched Aer baseline.

    Parameters
    ----------
    quimb_runtime
        Genuine Quimb runtime admitted by the owning CI job.

    """
    from scipy.linalg import expm

    coupling, frequencies, initial, hamiltonian = _four_site_reference()
    exact = expm(-1j * hamiltonian) @ initial
    aer = _aer_mps_reference(initial, hamiltonian)
    np.testing.assert_allclose(aer, exact, rtol=1e-8, atol=1e-10)
    errors: list[float] = []
    for steps in [16, 32, 64]:
        result = tebd_evolution(
            coupling,
            frequencies,
            t_max=1.0,
            dt=1.0 / steps,
            bond_dim=4,
            cutoff=0.0,
            order=4,
            initial_state=initial,
        )
        state = export_mps_state(result["mps_final"], max_memory_bytes=512)
        errors.append(float(np.linalg.norm(state - exact)))
    assert errors[1] < errors[0] / 4
    assert errors[2] < errors[1] / 4
    np.testing.assert_allclose(state, exact, rtol=1e-8, atol=1e-10)
    rows: list[dict[str, object]] = []
    for bond in [1, 2, 4]:
        result = tebd_evolution(
            coupling,
            frequencies,
            t_max=1.0,
            dt=1.0 / 64,
            bond_dim=bond,
            cutoff=0.0,
            order=4,
            initial_state=initial,
        )
        state = export_mps_state(result["mps_final"], max_memory_bytes=512)
        diagnostics = result["diagnostics"]
        observation = diagnostics.observations[-1]
        rows.append(
            {
                "bond_cap": bond,
                "actual_bonds": observation.bond_dims,
                "tensor_bytes": observation.stored_bytes,
                "dense_state_bytes": exact.nbytes,
                "state_error": float(np.linalg.norm(state - exact)),
                "discarded_weight": diagnostics.discarded_weight,
                "discarded_weight_scope": diagnostics.discarded_weight_scope,
                "observables": observation.local_z,
            }
        )
        if bond == 4:
            assert 4 in observation.bond_dims
            assert observation.stored_bytes > exact.nbytes
            np.testing.assert_allclose(state, aer, rtol=1e-8, atol=1e-10)
        else:
            assert np.linalg.norm(state - exact) > 1e-4
    print(
        json.dumps(
            {
                "model": "nearest-neighbour spin-half XY, site zero most significant",
                "quimb_version": quimb_runtime.__version__,
                "T": 1.0,
                "timestep_errors_16_32_64": errors,
                "rows": rows,
                "claim_boundary": "small-system representation/error study; no speed or quantum advantage claim",
            },
            sort_keys=True,
        )
    )


def test_runtime_cost_against_previous_public_owner(
    quimb_runtime: ModuleType,
    tmp_path: Path,
) -> None:
    """Compare actual old/new solver work in the same functional runtime.

    Parameters
    ----------
    quimb_runtime
        Genuine Quimb runtime admitted by the owning CI job.
    tmp_path
        Scoped temporary directory holding the immutable published baseline.

    Notes
    -----
    These are non-isolated functional cost observations. Process/library caches
    are already warm, so neither first invocation is labelled a cold-runtime
    measurement. No production speed or hardware advantage claim is made.

    """
    baseline_sha = "8c0a8fce8bebf03fe54c268b09ece6467f200289"
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{baseline_sha}:src/scpn_quantum_control/phase/mps_evolution.py",
        ]
    )
    baseline_path = tmp_path / "published_mps_owner.py"
    baseline_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(
        "scpn_quantum_control.phase._published_mps_cost_baseline",
        baseline_path,
    )
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    coupling = np.array([[0.0, 0.8], [0.8, 0.0]])
    frequencies = np.array([0.4, -0.2])
    load = getattr(os, "getloadavg", None)
    affinity = getattr(os, "sched_getaffinity", None)
    load_before = load() if load is not None else None
    samples: dict[str, list[float]] = {"baseline": [], "current": []}
    for _ in range(4):
        started = time.perf_counter()
        old = baseline.tebd_evolution(
            coupling,
            frequencies,
            t_max=0.1,
            dt=0.05,
            bond_dim=2,
        )
        samples["baseline"].append(1000 * (time.perf_counter() - started))
        started = time.perf_counter()
        current = tebd_evolution(
            coupling,
            frequencies,
            t_max=0.1,
            dt=0.05,
            bond_dim=2,
        )
        samples["current"].append(1000 * (time.perf_counter() - started))
        np.testing.assert_allclose(current["R"], old["R"], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            export_mps_state(current["mps_final"]),
            old["mps_final"].to_dense().reshape(-1),
            rtol=0.0,
            atol=1e-12,
        )
    print(
        json.dumps(
            {
                "baseline_source_sha": baseline_sha,
                "baseline_source_sha256": hashlib.sha256(source).hexdigest(),
                "quimb_version": quimb_runtime.__version__,
                "cpu_affinity": sorted(affinity(0)) if affinity is not None else None,
                "host_load_before": load_before,
                "host_load_after": load() if load is not None else None,
                "invocation_ms": samples,
                "method": "same model, float64/complex128, alternating old/current, process caches warm",
                "isolation": "non-isolated functional observation; no production speed claim",
            },
            sort_keys=True,
        )
    )


def test_dense_preparation_shares_the_caller_memory_ceiling(quimb_runtime: ModuleType) -> None:
    """Refuse dense preparation that exceeds the remaining compressed-run cap.

    Parameters
    ----------
    quimb_runtime
        Genuine Quimb runtime admitted by the owning CI job.

    """
    from scpn_quantum_control.phase.mps_diagnostics import require_mps_allocation

    coupling, frequencies = np.zeros((2, 2)), np.zeros(2)
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    workspace = require_mps_allocation(2, 1, observations=3).workspace_bytes
    original = initial.tobytes()
    with pytest.raises(MemoryError, match="remaining memory cap"):
        tebd_evolution(
            coupling,
            frequencies,
            t_max=0.1,
            dt=0.05,
            bond_dim=1,
            initial_state=initial,
            max_memory_bytes=workspace,
        )
    with pytest.raises(MemoryError, match="dense initial state"):
        tebd_evolution(
            coupling,
            frequencies,
            t_max=0.1,
            dt=0.05,
            bond_dim=1,
            initial_state=initial,
            max_memory_bytes=workspace + 127,
        )
    result = tebd_evolution(
        coupling,
        frequencies,
        t_max=0.1,
        dt=0.05,
        bond_dim=1,
        initial_state=initial,
        max_memory_bytes=workspace + 128,
    )
    assert result["diagnostics"].memory_budget_bytes == workspace + 128
    assert initial.tobytes() == original
