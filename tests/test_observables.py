#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

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
    qfi_res = qfi(counts, sync_order=0.95, dla_asymmetry=8.0)
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


if __name__ == "__main__":
    test_all_observables_with_real_counts()
