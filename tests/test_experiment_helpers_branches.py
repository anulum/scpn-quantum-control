# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the hardware experiment helpers
"""Guard tests for the hardware experiment reduction helpers.

Each test drives one missing-data guard: absent counts or expectation values on
a job result and the count-consuming per-qubit, QAOA-cost and correlator
reductions.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware._experiment_helpers import (
    _build_evo_base,
    _build_xyz_circuits,
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _R_from_xyz,
    _require_counts,
    _require_expectations,
)
from scpn_quantum_control.hardware.runner import JobResult


def _job_result() -> JobResult:
    """Return a result with neither sampler nor estimator payload."""
    return JobResult(
        job_id="job1",
        backend_name="sim",
        experiment_name="exp",
        metadata={},
    )


def test_require_counts_rejects_absent_counts() -> None:
    """A job result without counts is rejected."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _require_counts(_job_result())


def test_require_counts_returns_sampler_payload() -> None:
    """Return the exact counts mapping carried by a sampler result."""
    counts = {"0": 7, "1": 1}
    result = JobResult(
        job_id="job-counts",
        backend_name="sim",
        experiment_name="counts",
        counts=counts,
        metadata={},
    )

    assert _require_counts(result) is counts


def test_require_expectations_rejects_absent_values() -> None:
    """A job result without expectation values is rejected."""
    with pytest.raises(ValueError, match="expectation values are required"):
        _require_expectations(_job_result())


def test_require_expectations_returns_estimator_payload() -> None:
    """Return the exact expectation array carried by an estimator result."""
    expectations = np.array([0.25, -0.5])
    result = JobResult(
        job_id="job-expectations",
        backend_name="sim",
        experiment_name="expectations",
        expectation_values=expectations,
        metadata={},
    )

    assert _require_expectations(result) is expectations


def test_build_evolution_base_supports_both_trotter_syntheses() -> None:
    """Build real first- and second-order Qiskit evolution circuits."""
    coupling = build_knm_paper27(L=2)
    frequencies = OMEGA_N_16[:2]

    lie = _build_evo_base(2, coupling, frequencies, 0.1, 2)
    suzuki = _build_evo_base(2, coupling, frequencies, 0.1, 2, trotter_order=2)

    assert lie.num_qubits == suzuki.num_qubits == 2
    assert lie.count_ops()["PauliEvolution"] == 1
    assert suzuki.count_ops()["PauliEvolution"] == 1


def test_build_xyz_circuits_adds_each_measurement_basis() -> None:
    """Copy a real base circuit into Z-, X-, and Y-basis measurements."""
    base = QuantumCircuit(2)
    base.ry(0.2, 0)

    z_circuit, x_circuit, y_circuit = _build_xyz_circuits(base, 2)

    assert z_circuit.num_clbits == x_circuit.num_clbits == y_circuit.num_clbits == 2
    assert "h" not in z_circuit.count_ops()
    assert x_circuit.count_ops()["h"] == 2
    assert y_circuit.count_ops()["sdg"] == 2


def test_expectation_per_qubit_requires_counts() -> None:
    """The per-qubit reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _expectation_per_qubit(None, 2)


def test_expectation_per_qubit_handles_spaced_and_short_bitstrings() -> None:
    """Reduce spaced count keys without reading beyond their available bits."""
    expectations, deviations = _expectation_per_qubit({"0 0": 3, "1": 1}, 3)

    np.testing.assert_allclose(expectations, [0.5, 0.75, 0.0])
    assert np.all(np.isfinite(deviations))


def test_R_from_xyz_returns_all_means_and_uncertainties() -> None:
    """Reconstruct a finite order parameter from real XYZ count mappings."""
    z_counts = {"00": 2, "11": 2}
    x_counts = {"00": 4}
    y_counts = {"00": 2, "11": 2}

    result = _R_from_xyz(z_counts, x_counts, y_counts, 2)

    assert len(result) == 8
    assert result[0] == pytest.approx(1.0)
    assert all(np.all(np.isfinite(values)) for values in result[2:])


def test_qaoa_cost_requires_counts() -> None:
    """The QAOA-cost reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _qaoa_cost_from_counts(None, SparsePauliOp("Z"), 1)


def test_qaoa_cost_evaluates_diagonal_and_rejects_xy_terms() -> None:
    """Evaluate Z/identity terms while zeroing non-diagonal X/Y terms."""
    hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 2.0), ("YI", 3.0), ("II", 0.5)])

    cost = _qaoa_cost_from_counts({"0 0": 3, "0 1": 1}, hamiltonian, 2)

    assert cost == pytest.approx(1.0)


def test_correlator_requires_counts() -> None:
    """The correlator reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _correlator_from_counts(None, 0, 1)


def test_correlator_handles_same_different_and_empty_counts() -> None:
    """Compute a signed marginal correlator and the empty-count sentinel."""
    assert _correlator_from_counts({"00": 3, "01": 1}, 0, 1) == pytest.approx(0.5)
    assert _correlator_from_counts({}, 0, 1) == 0.0
