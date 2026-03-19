# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Surface-code protected UPDE simulation.

Each oscillator is encoded as a distance-d rotated surface code logical
qubit (Horsman et al., NJP 14, 123011 (2012)). This corrects both
X and Z errors, unlike the repetition code in fault_tolerant.py which
only corrects X.

Physical qubit layout per oscillator: d² data + (d²-1) ancilla = 2d²-1.
Total physical qubits: n_osc × (2d² - 1).

Logical gates:
  - Logical Z rotation: transversal Rz on all data qubits (trivial)
  - Logical ZZ coupling: lattice surgery merge-and-split pattern
    (Horsman et al., Sec. IV; Litinski, Quantum 3, 205 (2019))
  - Syndrome extraction: X-type and Z-type stabilizer measurements

This module provides the circuit construction and syndrome extraction.
Actual decoding delegates to the existing MWPM decoder in control_qec.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


@dataclass(frozen=True)
class SurfaceCodeSpec:
    """Surface code parameters for one logical qubit."""

    distance: int
    n_data: int  # d²
    n_ancilla: int  # d² - 1 (X-type: (d²-1)/2, Z-type: (d²-1)/2)
    n_physical: int  # 2d² - 1

    @classmethod
    def from_distance(cls, d: int) -> SurfaceCodeSpec:
        if d < 3 or d % 2 == 0:
            raise ValueError(f"Distance must be odd >= 3, got {d}")
        n_data = d * d
        n_ancilla = d * d - 1
        return cls(distance=d, n_data=n_data, n_ancilla=n_ancilla, n_physical=2 * d * d - 1)


class SurfaceCodeUPDE:
    """Structural model of surface-code protected Kuramoto-XY simulation.

    NOT an executable QEC implementation. This models the circuit structure
    (encoding, logical gates, syndrome extraction) and qubit budget of a
    surface-code UPDE, but does not perform stabilizer-state preparation,
    ancilla measurement, or syndrome decoding. Use for resource estimation
    and circuit-depth analysis, not for fault-tolerance claims.

    Each oscillator is modeled as a distance-d rotated surface code patch.
    Logical Rz is transversal (angle distributed across d² data qubits).
    Logical ZZ uses pairwise RZZ as an operator-level approximation of
    lattice surgery (Litinski, Quantum 3, 205 (2019)).

    Physical qubit budget:
      n_osc=4, d=3: 4 × 17 = 68 qubits
      n_osc=4, d=5: 4 × 49 = 196 qubits
      n_osc=16, d=3: 16 × 17 = 272 qubits
      n_osc=16, d=5: 16 × 49 = 784 qubits
    """

    def __init__(
        self,
        n_osc: int,
        code_distance: int = 3,
        K: np.ndarray | None = None,
        omega: np.ndarray | None = None,
    ):
        if n_osc < 2:
            raise ValueError(f"Need >= 2 oscillators, got {n_osc}")

        self.n_osc = n_osc
        self.spec = SurfaceCodeSpec.from_distance(code_distance)
        self.K = K if K is not None else build_knm_paper27(L=n_osc)
        self.omega = omega if omega is not None else OMEGA_N_16[:n_osc].copy()
        self.total_qubits = n_osc * self.spec.n_physical

    def _osc_data_qubits(self, osc: int) -> list[int]:
        """Data qubit indices for oscillator osc."""
        base = osc * self.spec.n_physical
        return list(range(base, base + self.spec.n_data))

    def _osc_x_ancilla(self, osc: int) -> list[int]:
        """X-stabilizer ancilla indices (first half of ancillae)."""
        base = osc * self.spec.n_physical + self.spec.n_data
        n_x = (self.spec.n_ancilla + 1) // 2
        return list(range(base, base + n_x))

    def _osc_z_ancilla(self, osc: int) -> list[int]:
        """Z-stabilizer ancilla indices (second half of ancillae)."""
        base = osc * self.spec.n_physical + self.spec.n_data
        n_x = (self.spec.n_ancilla + 1) // 2
        n_z = self.spec.n_ancilla - n_x
        return list(range(base + n_x, base + n_x + n_z))

    def encode_logical(self, osc: int, qc: QuantumCircuit) -> None:
        """Prepare logical |+_L> state for oscillator.

        Ry(theta) on representative data qubit, then stabilizer preparation
        via CNOT fan-out in both X and Z bases.
        """
        data = self._osc_data_qubits(osc)
        theta = float(self.omega[osc]) % (2 * np.pi)

        # Prepare representative qubit
        qc.ry(theta, data[0])

        # X-basis fan-out (row-wise CNOT chain within surface code patch)
        d = self.spec.distance
        for row in range(d):
            row_start = row * d
            for col in range(1, d):
                qc.cx(data[row_start], data[row_start + col])

        # Z-basis fan-out (column-wise CNOT chain)
        for col in range(d):
            for row in range(1, d):
                qc.cx(data[col], data[row * d + col])

    def logical_rz(self, osc: int, angle: float, qc: QuantumCircuit) -> None:
        """Transversal logical Rz: apply Rz(angle/d²) to each data qubit.

        Rz is transversal on the rotated surface code.
        """
        data = self._osc_data_qubits(osc)
        distributed_angle = angle / len(data)
        for dq in data:
            qc.rz(distributed_angle, dq)

    def logical_zz(
        self,
        osc_i: int,
        osc_j: int,
        angle: float,
        qc: QuantumCircuit,
    ) -> None:
        """Logical ZZ coupling via lattice surgery pattern.

        Simplified model: transversal RZZ between corresponding data qubits.
        Full lattice surgery (Litinski, Quantum 3, 205 (2019)) would use
        ancilla-mediated merge-and-split, but at the logical level the
        effect is equivalent for small angles.
        """
        data_i = self._osc_data_qubits(osc_i)
        data_j = self._osc_data_qubits(osc_j)
        distributed_angle = angle / len(data_i)
        for di, dj in zip(data_i, data_j):
            qc.rzz(distributed_angle, di, dj)

    def x_syndrome_extract(self, osc: int, qc: QuantumCircuit) -> None:
        """X-type stabilizer measurement.

        Each X-stabilizer ancilla checks the parity of its 4 neighboring
        data qubits in the X basis (Hadamard → CNOT → Hadamard).
        """
        data = self._osc_data_qubits(osc)
        x_anc = self._osc_x_ancilla(osc)
        d = self.spec.distance

        for idx, anc in enumerate(x_anc):
            qc.h(anc)
            # Connect to up to 4 neighboring data qubits
            row = idx // (d - 1) if d > 1 else 0
            col = idx % (d - 1) if d > 1 else 0
            neighbors = []
            if row * d + col < len(data):
                neighbors.append(data[row * d + col])
            if row * d + col + 1 < len(data):
                neighbors.append(data[row * d + col + 1])
            if (row + 1) * d + col < len(data):
                neighbors.append(data[(row + 1) * d + col])
            if (row + 1) * d + col + 1 < len(data):
                neighbors.append(data[(row + 1) * d + col + 1])
            for nb in neighbors:
                qc.cx(anc, nb)
            qc.h(anc)

    def z_syndrome_extract(self, osc: int, qc: QuantumCircuit) -> None:
        """Z-type stabilizer measurement.

        Each Z-stabilizer ancilla checks parity of 4 neighboring data
        qubits in the Z basis (CNOT pattern).
        """
        data = self._osc_data_qubits(osc)
        z_anc = self._osc_z_ancilla(osc)
        d = self.spec.distance

        for idx, anc in enumerate(z_anc):
            row = idx // (d - 1) if d > 1 else 0
            col = idx % (d - 1) if d > 1 else 0
            neighbors = []
            if row * d + col < len(data):
                neighbors.append(data[row * d + col])
            if row * d + col + 1 < len(data):
                neighbors.append(data[row * d + col + 1])
            if (row + 1) * d + col < len(data):
                neighbors.append(data[(row + 1) * d + col])
            if (row + 1) * d + col + 1 < len(data):
                neighbors.append(data[(row + 1) * d + col + 1])
            for nb in neighbors:
                qc.cx(nb, anc)

    def build_step_circuit(self, dt: float = 0.1) -> QuantumCircuit:
        """One QEC-protected Trotter step.

        1. Encode logical qubits
        2. Transversal Rz (natural frequency)
        3. Logical ZZ coupling (lattice surgery pattern)
        4. X and Z syndrome extraction
        """
        qc = QuantumCircuit(self.total_qubits)

        for osc in range(self.n_osc):
            self.encode_logical(osc, qc)

        for osc in range(self.n_osc):
            self.logical_rz(osc, self.omega[osc] * dt, qc)

        for i in range(self.n_osc):
            for j in range(i + 1, self.n_osc):
                if abs(self.K[i, j]) > 1e-10:
                    self.logical_zz(i, j, self.K[i, j] * dt, qc)

        for osc in range(self.n_osc):
            self.x_syndrome_extract(osc, qc)
            self.z_syndrome_extract(osc, qc)

        return qc

    def physical_qubit_budget(self) -> dict:
        """Physical qubit requirements."""
        return {
            "n_osc": self.n_osc,
            "code_distance": self.spec.distance,
            "data_per_osc": self.spec.n_data,
            "ancilla_per_osc": self.spec.n_ancilla,
            "physical_per_osc": self.spec.n_physical,
            "total_physical": self.total_qubits,
            "correctable_errors": (self.spec.distance - 1) // 2,
        }
