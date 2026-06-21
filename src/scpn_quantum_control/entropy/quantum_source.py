# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum entropy sources
"""Quantum measurement entropy sources for the QRNG streaming harness.

Three measurement circuits supply raw entropy, sampled on the Qiskit Aer
simulator (the exact stabilizer backend for the Clifford sources, statevector
for ``phase_estimation``):

- ``xy_measurement`` — a register of qubits each rotated to ``|+>`` by a
  Hadamard and measured in the computational basis; every shot yields one
  unbiased bit per qubit.
- ``bell_pair`` — qubit pairs prepared in the Bell state ``|Phi+>``; one member
  of each pair is read out (its marginal is maximally mixed).
- ``phase_estimation`` — a Hadamard test with a controlled-phase (``pi/2``)
  kickback whose control-qubit marginal is uniform.

Bias from device imperfections is removed downstream by Von Neumann debiasing in
:mod:`~scpn_quantum_control.entropy.qrng_stream`; the raw simulator output is
already unbiased, and the debiaser keeps the contract honest on biased hardware.

Reference for the entropy model: Bell, Pironio, Christensen et al.,
*Device-independent randomness from a single measurement on a Bell state*,
Physical Review Letters 121, 100403 (2018).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

QuantumSourceKind = Literal["xy_measurement", "bell_pair", "phase_estimation"]


class EntropyBackend(Protocol):
    """Backend boundary that turns a circuit into per-shot measurement bits."""

    def sample_bits(self, n_bits: int) -> NDArray[np.int8]:
        """Return exactly ``n_bits`` raw measurement bits."""
        ...


class AerQuantumEntropySource:
    """Raw quantum entropy from Qiskit Aer measurement circuits.

    The circuit is sized to ``register_qubits`` measured qubits; each Aer shot
    contributes ``register_qubits`` bits, so a request for ``n_bits`` runs
    ``ceil(n_bits / register_qubits)`` shots and trims to length.
    """

    def __init__(
        self,
        kind: QuantumSourceKind = "xy_measurement",
        *,
        register_qubits: int = 24,
        seed: int | None = None,
    ) -> None:
        if kind not in ("xy_measurement", "bell_pair", "phase_estimation"):
            raise ValueError(f"unknown quantum source kind: {kind!r}")
        if not isinstance(register_qubits, int) or register_qubits < 1:
            raise ValueError("register_qubits must be a positive integer")
        # xy_measurement and bell_pair are Clifford circuits and use the exact
        # stabilizer simulator, which scales to large registers. phase_estimation
        # uses a controlled-phase (non-Clifford) kickback and must be simulated by
        # statevector, so its register is capped to keep 2*register qubits tractable.
        self._kind: QuantumSourceKind = kind
        self._method = "stabilizer" if kind in ("xy_measurement", "bell_pair") else "statevector"
        if self._method == "statevector" and register_qubits > 12:
            raise ValueError(
                "phase_estimation register_qubits is capped at 12 (statevector simulation)"
            )
        self._register_qubits = register_qubits
        self._seed = seed
        self._circuit = self._build_circuit()
        self._simulator = self._build_simulator()

    @property
    def kind(self) -> QuantumSourceKind:
        """Return the configured source kind."""
        return self._kind

    @property
    def bits_per_shot(self) -> int:
        """Return the number of measured bits produced per Aer shot."""
        return self._register_qubits

    def _build_circuit(self) -> Any:
        from qiskit import QuantumCircuit

        w = self._register_qubits
        if self._kind == "xy_measurement":
            qc = QuantumCircuit(w, w)
            qc.h(range(w))
            qc.measure(range(w), range(w))
            return qc
        if self._kind == "bell_pair":
            qc = QuantumCircuit(2 * w, w)
            for i in range(w):
                qc.h(2 * i)
                qc.cx(2 * i, 2 * i + 1)
                qc.measure(2 * i, i)
            return qc
        # phase_estimation: Hadamard test with a controlled-phase kickback on an ancilla.
        qc = QuantumCircuit(2 * w, w)
        for i in range(w):
            control, target = 2 * i, 2 * i + 1
            qc.h(control)
            qc.x(target)
            qc.cp(np.pi / 2.0, control, target)
            qc.h(control)
            qc.measure(control, i)
        return qc

    def _build_simulator(self) -> Any:
        from qiskit_aer import AerSimulator

        return AerSimulator(method=self._method, seed_simulator=self._seed)

    def sample_bits(self, n_bits: int) -> NDArray[np.int8]:
        """Return exactly ``n_bits`` raw measurement bits."""
        if not isinstance(n_bits, int) or n_bits < 0:
            raise ValueError("n_bits must be a non-negative integer")
        if n_bits == 0:
            return np.empty(0, dtype=np.int8)
        from qiskit import transpile

        shots = (n_bits + self._register_qubits - 1) // self._register_qubits
        compiled = transpile(self._circuit, self._simulator)
        result = self._simulator.run(compiled, shots=shots, memory=True).result()
        memory = result.get_memory(compiled)
        # Each memory entry is a big-endian bitstring of length register_qubits.
        bits = np.array([[int(c) for c in record] for record in memory], dtype=np.int8).reshape(-1)
        return np.ascontiguousarray(bits[:n_bits], dtype=np.int8)


def von_neumann_debias(bits: NDArray[np.int8]) -> NDArray[np.int8]:
    """Remove first-order bias via the Von Neumann extractor.

    Consecutive non-overlapping pairs map ``01 -> 0``, ``10 -> 1``; equal pairs
    (``00``, ``11``) are discarded. The output is unbiased for any independent
    biased source, at the cost of a variable, reduced bit rate.
    """
    pairs = bits[: bits.size - (bits.size % 2)].reshape(-1, 2)
    differing = pairs[:, 0] != pairs[:, 1]
    return np.ascontiguousarray(pairs[differing, 0], dtype=np.int8)


__all__ = [
    "AerQuantumEntropySource",
    "EntropyBackend",
    "QuantumSourceKind",
    "von_neumann_debias",
]
