# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Surface Code Upde
"""Tests for surface-code protected UPDE simulation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import omega_for_oscillators
from scpn_quantum_control.qec.surface_code_upde import SurfaceCodeSpec, SurfaceCodeUPDE


class TestSurfaceCodeSpec:
    def test_d3(self) -> None:
        spec = SurfaceCodeSpec.from_distance(3)
        assert spec.n_data == 9
        assert spec.n_ancilla == 8
        assert spec.n_physical == 17

    def test_d5(self) -> None:
        spec = SurfaceCodeSpec.from_distance(5)
        assert spec.n_data == 25
        assert spec.n_ancilla == 24
        assert spec.n_physical == 49

    def test_even_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            SurfaceCodeSpec.from_distance(4)

    def test_distance_1_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd >= 3"):
            SurfaceCodeSpec.from_distance(1)


class TestSurfaceCodeUPDE:
    def test_init_4osc_d3(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=4, code_distance=3)
        assert sc.total_qubits == 4 * 17
        assert sc.spec.distance == 3

    def test_physical_qubit_budget(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=4, code_distance=3)
        budget = sc.physical_qubit_budget()
        assert budget["total_physical"] == 68
        assert budget["correctable_errors"] == 1
        assert budget["data_per_osc"] == 9

    def test_build_step_circuit(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        qc = sc.build_step_circuit(dt=0.1)
        assert qc.num_qubits == 2 * 17
        assert qc.size() > 0

    def test_encode_produces_valid_circuit(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(sc.total_qubits)
        sc.encode_logical(0, qc)
        assert qc.size() > 0

    def test_logical_rz(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(sc.total_qubits)
        sc.logical_rz(0, 0.5, qc)
        rz_count = sum(1 for g in qc if g.operation.name == "rz")
        assert rz_count == sc.spec.n_data

    def test_logical_zz(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(sc.total_qubits)
        sc.logical_zz(0, 1, 0.3, qc)
        rzz_count = sum(1 for g in qc if g.operation.name == "rzz")
        assert rzz_count == sc.spec.n_data

    def test_x_syndrome_extract(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(sc.total_qubits)
        sc.x_syndrome_extract(0, qc)
        assert qc.size() > 0

    def test_z_syndrome_extract(self) -> None:
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(sc.total_qubits)
        sc.z_syndrome_extract(0, qc)
        assert qc.size() > 0

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_syndrome_scaffold_connects_four_neighbours(self, distance: int) -> None:
        """Every valid-distance ancilla proxy connects to four data qubits."""
        from qiskit import QuantumCircuit

        sc = SurfaceCodeUPDE(n_osc=2, code_distance=distance)
        qc = QuantumCircuit(sc.total_qubits)

        sc.x_syndrome_extract(0, qc)
        sc.z_syndrome_extract(0, qc)

        operations = qc.count_ops()
        assert operations.get("cx", 0) == 4 * sc.spec.n_ancilla
        assert operations.get("h", 0) == 2 * ((sc.spec.n_ancilla + 1) // 2)
        assert qc.num_clbits == 0

    def test_budget_scaling(self) -> None:
        for d in [3, 5, 7]:
            sc = SurfaceCodeUPDE(n_osc=4, code_distance=d)
            budget = sc.physical_qubit_budget()
            assert budget["total_physical"] == 4 * (2 * d * d - 1)
            assert budget["correctable_errors"] == (d - 1) // 2

    def test_n_osc_1_rejected(self) -> None:
        with pytest.raises(ValueError, match="Need >= 2"):
            SurfaceCodeUPDE(n_osc=1)

    def test_custom_K_omega(self) -> None:
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 2.0])
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3, K=K, omega=omega)
        qc = sc.build_step_circuit(dt=0.1)
        assert qc.num_qubits == 34

    def test_default_frequencies_extend_beyond_canonical_table(self) -> None:
        """The default dependency supplies one deterministic frequency per patch."""
        sc = SurfaceCodeUPDE(n_osc=17, code_distance=3)

        np.testing.assert_array_equal(sc.omega, omega_for_oscillators(17))
        assert sc.omega.shape == (17,)

    def test_zero_coupling_omits_rzz_proxy_layer(self) -> None:
        """The public circuit builder skips coupling proxies below its threshold."""
        K = np.zeros((2, 2), dtype=np.float64)
        omega = np.array([1.0, 2.0], dtype=np.float64)
        sc = SurfaceCodeUPDE(n_osc=2, code_distance=3, K=K, omega=omega)

        qc = sc.build_step_circuit(dt=0.1)

        assert all(instruction.operation.name != "rzz" for instruction in qc)
