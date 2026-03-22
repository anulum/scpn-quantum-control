# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Google Cirq backend adapter.

Translates the Kuramoto-XY Hamiltonian into Cirq circuits for
execution on Google Sycamore/Weber processors or Cirq simulators.

Usage:
    from scpn_quantum_control.hardware.cirq_adapter import CirqRunner
    runner = CirqRunner(K, omega)
    result = runner.run_trotter(t=1.0, reps=5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cirq  # type: ignore[import-untyped,import-not-found]

    _CIRQ_AVAILABLE = True
except ImportError:
    _CIRQ_AVAILABLE = False
    cirq = None  # type: ignore[assignment]


@dataclass
class CirqResult:
    """Result from Cirq execution."""

    energy: float
    n_qubits: int
    device_name: str


def is_cirq_available() -> bool:
    """Check if Cirq is installed."""
    return _CIRQ_AVAILABLE


class CirqRunner:
    """Cirq backend for Kuramoto-XY simulation.

    Builds Trotter circuits using Cirq gates (XX, YY, ZZ decomposition)
    and runs on Cirq simulators or Google hardware.
    """

    def __init__(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        device: str = "simulator",
    ):
        if not _CIRQ_AVAILABLE:
            raise ImportError("Cirq not installed: pip install cirq-core")

        self.K = K
        self.omega = omega
        self.n = K.shape[0]
        self.device_name = device
        self.qubits = cirq.LineQubit.range(self.n)

    def _build_trotter_step(self, dt: float) -> cirq.Circuit:
        """One Trotter step: XX+YY coupling + Z field."""
        ops: list = []
        n = self.n

        # XX+YY coupling gates
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.K[i, j]) > 1e-15:
                    angle = float(self.K[i, j]) * dt
                    ops.append(
                        cirq.XXPowGate(exponent=angle / np.pi)(self.qubits[i], self.qubits[j])
                    )
                    ops.append(
                        cirq.YYPowGate(exponent=angle / np.pi)(self.qubits[i], self.qubits[j])
                    )

        # Z field terms
        for i in range(n):
            if abs(self.omega[i]) > 1e-15:
                angle = float(self.omega[i]) * dt / 2.0
                ops.append(cirq.rz(2 * angle)(self.qubits[i]))

        return cirq.Circuit(ops)

    def run_trotter(
        self,
        t: float = 1.0,
        reps: int = 5,
    ) -> CirqResult:
        """Run Trotter evolution on Cirq simulator."""
        dt = t / reps
        step = self._build_trotter_step(dt)
        circuit = cirq.Circuit()
        for _ in range(reps):
            circuit += step

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        sv = result.final_state_vector

        # Energy expectation
        energy = self._compute_energy(sv)

        return CirqResult(
            energy=energy,
            n_qubits=self.n,
            device_name=self.device_name,
        )

    def _compute_energy(self, sv: np.ndarray) -> float:
        """Compute <H> from statevector."""
        n = self.n
        dim = 2**n
        energy = 0.0

        # Z terms: -ω_i/2 × <Z_i>
        for i in range(n):
            z_exp = 0.0
            for idx in range(dim):
                bit = (idx >> (n - 1 - i)) & 1
                z_exp += (1 - 2 * bit) * abs(sv[idx]) ** 2
            energy -= self.omega[i] / 2.0 * z_exp

        return float(energy)
