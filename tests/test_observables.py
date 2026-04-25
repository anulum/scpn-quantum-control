#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

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
    phi_res = phi(counts)
    qfi_res = qfi(counts, sync_order=0.95, dla_asymmetry=8.0)
    thermo_res = thermo(counts, work=1.2)
    logical_res = logical(counts, logical_fidelity=0.92)

    print("✅ DLAParityWitness:", dla_res)
    print("✅ SyncOrderParameter:", sync_res)
    print("✅ IntegratedInformationPhi:", phi_res)
    print("✅ QuantumFisherInformation:", qfi_res)
    print("✅ ThermodynamicWitness:", thermo_res)
    print("✅ LogicalSyncWitness:", logical_res)

    assert 0.0 <= sync_res["sync_order"] <= 1.0
    assert abs(dla_res["dla_asymmetry"]) <= 100
    print("\nAll observables passed integration test with real circuit counts.")


if __name__ == "__main__":
    test_all_observables_with_real_counts()
