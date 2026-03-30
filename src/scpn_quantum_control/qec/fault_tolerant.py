# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Fault Tolerant
"""Repetition-code UPDE simulation with bit-flip protected logical qubits.

Proof-of-concept for QEC-protected Kuramoto dynamics. Uses distance-d
repetition code (bit-flip only) per oscillator. Does NOT correct phase
errors — use SurfaceCodeUPDE for full X+Z protection. Validates approach
on statevector; not executable on current hardware at useful noise levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


@dataclass
class LogicalQubit:
    """Repetition-code logical qubit."""

    code_distance: int
    phase_angle: float

    @property
    def data_qubits(self) -> int:
        return self.code_distance


class RepetitionCodeUPDE:
    """Repetition-code Kuramoto evolution (bit-flip protection only).

    Each of n_osc oscillators is encoded into d physical qubits.
    Layout per oscillator: [d data qubits | d-1 ancilla qubits].
    Total physical qubits = n_osc * (2d - 1).
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
        if code_distance < 1 or code_distance % 2 == 0:
            raise ValueError(f"code_distance must be odd positive integer, got {code_distance}")

        self.n_osc = n_osc
        self.d = code_distance
        self.K = K if K is not None else build_knm_paper27(L=n_osc)
        self.omega = omega if omega is not None else OMEGA_N_16[:n_osc].copy()

        self.data_per_osc = self.d
        self.ancilla_per_osc = self.d - 1
        self.qubits_per_osc = self.data_per_osc + self.ancilla_per_osc
        self.total_qubits = n_osc * self.qubits_per_osc

        self.logical_qubits = [
            LogicalQubit(code_distance, float(self.omega[i]) % (2 * np.pi)) for i in range(n_osc)
        ]

    def _osc_data_range(self, osc: int) -> range:
        start = osc * self.qubits_per_osc
        return range(start, start + self.data_per_osc)

    def _osc_ancilla_range(self, osc: int) -> range:
        start = osc * self.qubits_per_osc + self.data_per_osc
        return range(start, start + self.ancilla_per_osc)

    def encode_logical(self, osc: int, qc: QuantumCircuit) -> None:
        """Repetition-code encoding: Ry(theta) on first data qubit, CNOT fan-out."""
        data = list(self._osc_data_range(osc))
        theta = self.logical_qubits[osc].phase_angle
        qc.ry(theta, data[0])
        for i in range(1, len(data)):
            qc.cx(data[0], data[i])

    def transversal_zz(
        self,
        osc_i: int,
        osc_j: int,
        angle: float,
        qc: QuantumCircuit,
    ) -> None:
        """Transversal RZZ between logical qubits i and j.

        Applies d pairwise RZZ gates between corresponding data qubits.
        """
        data_i = list(self._osc_data_range(osc_i))
        data_j = list(self._osc_data_range(osc_j))
        for di, dj in zip(data_i, data_j):
            qc.rzz(angle / self.d, di, dj)

    def syndrome_extract(self, osc: int, qc: QuantumCircuit) -> None:
        """Parity checks between adjacent data qubits via ancillae."""
        data = list(self._osc_data_range(osc))
        ancilla = list(self._osc_ancilla_range(osc))
        for k in range(len(ancilla)):
            qc.cx(data[k], ancilla[k])
            qc.cx(data[k + 1], ancilla[k])

    def build_step_circuit(self, dt: float = 0.1) -> QuantumCircuit:
        """One Trotter step: encode -> Z rotations -> ZZ coupling -> syndrome."""
        qc = QuantumCircuit(self.total_qubits)

        for osc in range(self.n_osc):
            self.encode_logical(osc, qc)

        for osc in range(self.n_osc):
            for dq in self._osc_data_range(osc):
                qc.rz(self.omega[osc] * dt / self.d, dq)

        for i in range(self.n_osc):
            for j in range(i + 1, self.n_osc):
                if abs(self.K[i, j]) > 1e-10:
                    self.transversal_zz(i, j, self.K[i, j] * dt, qc)

        for osc in range(self.n_osc):
            self.syndrome_extract(osc, qc)

        return qc

    def step_with_qec(self, dt: float = 0.1) -> dict:
        """Execute one QEC-protected Trotter step and extract syndromes."""
        qc = self.build_step_circuit(dt)
        sv = Statevector.from_instruction(qc)

        syndromes: list[list[int]] = []
        for osc in range(self.n_osc):
            ancilla = list(self._osc_ancilla_range(osc))
            osc_syn = []
            for a in ancilla:
                p1 = float(sv.probabilities([a])[1])
                osc_syn.append(1 if p1 > 0.5 else 0)
            syndromes.append(osc_syn)

        return {
            "syndromes": syndromes,
            "errors_detected": sum(sum(s) for s in syndromes),
            "total_qubits": self.total_qubits,
            "n_osc": self.n_osc,
            "code_distance": self.d,
        }

    def physical_qubit_count(self) -> int:
        """Total physical qubits: n_osc * (d + d - 1)."""
        return self.total_qubits


# Backwards compatibility
FaultTolerantUPDE = RepetitionCodeUPDE
