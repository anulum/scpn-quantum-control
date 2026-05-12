#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

import numpy as np
import pytest

from scpn_quantum_control.analysis.dla_parity_witness import DLAParityWitness
from scpn_quantum_control.analysis.integrated_information_phi import IntegratedInformationPhi
from scpn_quantum_control.analysis.logical_sync_witness import LogicalSyncWitness
from scpn_quantum_control.analysis.quantum_fisher_information import QuantumFisherInformation
from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter
from scpn_quantum_control.analysis.thermodynamic_witness import ThermodynamicWitness


def test_all_observables_with_real_counts():
    # Realistic 8-qubit measurement counts (example from actual circuit)
    counts = {"00000000": 4523, "11111111": 3187, "01010101": 1740, "10101010": 550}

    dla = DLAParityWitness()
    sync = SyncOrderParameter()
    phi = IntegratedInformationPhi()
    qfi = QuantumFisherInformation()
    thermo = ThermodynamicWitness()
    logical = LogicalSyncWitness()

    dla_res = dla(counts)
    sync_res = sync(counts)
    phi_res = phi(counts, allow_entropy_proxy=True)
    qfi_res = qfi(counts, sync_order=0.95, dla_asymmetry=8.0, allow_proxy_estimate=True)
    thermo_res = thermo(counts, work_samples_joule=[1.19, 1.20, 1.21])
    logical_res = logical(counts, logical_fidelity=0.92)

    print("✅ DLAParityWitness:", dla_res)
    print("✅ SyncOrderParameter:", sync_res)
    print("✅ IntegratedInformationPhi:", phi_res)
    print("✅ QuantumFisherInformation:", qfi_res)
    print("✅ ThermodynamicWitness:", thermo_res)
    print("✅ LogicalSyncWitness:", logical_res)

    assert 0.0 <= sync_res["sync_order"] <= 1.0
    assert abs(dla_res["dla_asymmetry"]) <= 100  # ±100% is the physical maximum
    assert phi_res["is_integrated_information"] == 0.0
    assert "phi" not in phi_res
    assert qfi_res["qfi_available"] == 0.0
    assert qfi_res["is_quantum_fisher_information"] == 0.0
    assert "qfi" not in qfi_res

    print("\nAll observables passed integration test with real circuit counts.")


def test_thermodynamic_witness_requires_explicit_work_input():
    witness = ThermodynamicWitness()

    with pytest.raises(ValueError, match="work_samples_joule"):
        witness({"0": 10, "1": 6})


def test_thermodynamic_witness_reports_sample_statistics_and_jarzynski_residual():
    witness = ThermodynamicWitness()

    result = witness(
        {"0": 10, "1": 6},
        work_samples_joule=[1.0e-21, 1.2e-21, 0.8e-21],
        beta_per_joule=2.5e20,
        delta_free_energy_joule=0.9e-21,
    )

    assert result["mean_work_joule"] == pytest.approx(1.0e-21)
    assert result["work_variance_joule2"] > 0.0
    assert result["dissipated_work_joule"] == pytest.approx(0.1e-21)
    assert "jarzynski_delta_free_energy_joule" in result
    assert "work" not in result


def test_integrated_information_does_not_report_entropy_proxy_as_phi():
    witness = IntegratedInformationPhi()

    with pytest.raises(NotImplementedError, match="integrated information"):
        witness({"00": 5, "11": 5})


def test_integrated_information_entropy_proxy_is_explicitly_labelled():
    witness = IntegratedInformationPhi()

    result = witness({"00": 5, "11": 5}, allow_entropy_proxy=True)

    assert result["phi_available"] == 0.0
    assert result["entropy_proxy"] > 0.0
    assert result["is_integrated_information"] == 0.0
    assert "phi" not in result


def test_integrated_information_routes_real_hamiltonian_inputs_to_quantum_phi_engine():
    witness = IntegratedInformationPhi()
    coupling_matrix = np.array(
        [
            [0.0, 0.4, 0.2],
            [0.4, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        dtype=float,
    )
    natural_frequencies = np.array([0.1, 0.2, 0.3], dtype=float)

    result = witness(
        coupling_matrix=coupling_matrix,
        natural_frequencies=natural_frequencies,
    )

    assert result["phi_available"] == 1.0
    assert result["is_integrated_information"] == 1.0
    assert result["phi"] >= 0.0
    assert result["phi_max"] >= result["phi"]
    assert result["n_qubits"] == 3.0
    assert result["n_bipartitions"] > 0.0
    assert "entropy_proxy" not in result


def test_integrated_information_rejects_partial_or_invalid_hamiltonian_inputs():
    witness = IntegratedInformationPhi()

    with pytest.raises(ValueError, match="coupling_matrix and natural_frequencies"):
        witness(coupling_matrix=np.eye(2))
    with pytest.raises(ValueError, match="square"):
        witness(coupling_matrix=np.ones((2, 3)), natural_frequencies=np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        witness(coupling_matrix=np.eye(2), natural_frequencies=np.array([0.0, np.nan]))


def test_quantum_fisher_information_refuses_proxy_by_default():
    qfi = QuantumFisherInformation()

    with pytest.raises(NotImplementedError, match="coupling_matrix"):
        qfi({"00": 5, "11": 5}, sync_order=0.95, dla_asymmetry=8.0)


def test_quantum_fisher_information_proxy_is_explicitly_labelled():
    qfi = QuantumFisherInformation()

    result = qfi({"00": 5, "11": 5}, sync_order=0.95, dla_asymmetry=8.0, allow_proxy_estimate=True)

    assert result["qfi_available"] == 0.0
    assert result["is_quantum_fisher_information"] == 0.0
    assert result["qfi_proxy"] > 0.0
    assert "qfi" not in result


def test_quantum_fisher_information_routes_real_hamiltonian_inputs_to_qfi_engine():
    import numpy as np

    qfi = QuantumFisherInformation()
    coupling_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    natural_frequencies = np.array([0.2, -0.2])

    result = qfi(
        coupling_matrix=coupling_matrix,
        natural_frequencies=natural_frequencies,
        coupling_pairs=[(0, 1)],
        n_measurements=5000,
    )

    assert result["qfi_available"] == 1.0
    assert result["is_quantum_fisher_information"] == 1.0
    assert result["qfi"] > 0.0
    assert result["qfi_matrix_shape_0"] == 1.0
    assert result["qfi_matrix_shape_1"] == 1.0
    assert result["precision_bound_for_measurement_budget"] > 0.0
    assert result["n_measurements"] == 5000.0


def test_logical_sync_witness_refuses_fidelity_only_proxy_by_default():
    logical = LogicalSyncWitness()

    with pytest.raises(NotImplementedError, match="counts or probabilities"):
        logical(logical_fidelity=0.92)


def test_logical_sync_witness_fidelity_proxy_is_explicitly_labelled():
    logical = LogicalSyncWitness()

    result = logical(logical_fidelity=0.92, allow_fidelity_proxy=True)

    assert result["logical_sync_available"] == 0.0
    assert result["is_logical_sync_witness"] == 0.0
    assert result["logical_fidelity_proxy"] == pytest.approx(0.92)
    assert "logical_sync_order" not in result
    assert "passes" not in result


if __name__ == "__main__":
    test_all_observables_with_real_counts()
