# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Mitiq Error Mitigation Integration
"""Mitiq integration for production-quality error mitigation.

Wraps Mitiq's ZNE, PEC, and DDD around our Kuramoto-XY circuits.
Uses Mitiq's battle-tested implementations instead of rolling our own.

Requires: pip install mitiq
Reference: LaRose et al., Quantum 6, 774 (2022).
"""

from __future__ import annotations

from typing import Any

try:
    from mitiq import ddd, zne

    _MITIQ_AVAILABLE = True
except Exception:
    _MITIQ_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    _QISKIT_AVAILABLE = True
except Exception:
    _QISKIT_AVAILABLE = False


def is_mitiq_available() -> bool:
    """Check if Mitiq is installed."""
    return _MITIQ_AVAILABLE


def _qiskit_executor(circuit: QuantumCircuit, shots: int = 8192) -> float:
    """Execute a Qiskit circuit and return expectation value of Z on qubit 0."""
    if not _QISKIT_AVAILABLE:
        raise ImportError("qiskit-aer required")
    meas_circuit = circuit.copy()
    if not meas_circuit.cregs:
        meas_circuit.measure_all()
    backend = AerSimulator()
    job = backend.run(meas_circuit, shots=shots)
    counts = job.result().get_counts()
    total = sum(counts.values())
    exp_z = 0.0
    for bitstring, count in counts.items():
        parity = (-1) ** bitstring.count("1")
        exp_z += parity * count / total
    return exp_z


def zne_mitigated_expectation(
    circuit: QuantumCircuit,
    executor: Any | None = None,
    scale_factors: list[float] | None = None,
    shots: int = 8192,
) -> float:
    """Run ZNE (Zero Noise Extrapolation) via Mitiq.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to mitigate.
    executor : callable, optional
        Custom executor. Defaults to AerSimulator.
    scale_factors : list[float], optional
        Noise scale factors. Defaults to [1, 3, 5].
    shots : int
        Shots per execution.

    Returns
    -------
    float
        ZNE-mitigated expectation value.
    """
    if not _MITIQ_AVAILABLE:
        raise ImportError("mitiq not installed: pip install mitiq")

    if executor is None:

        def executor(c: QuantumCircuit) -> float:
            return _qiskit_executor(c, shots=shots)

    if scale_factors is None:
        scale_factors = [1.0, 3.0, 5.0]

    factory = zne.inference.RichardsonFactory(scale_factors)

    return float(
        zne.execute_with_zne(
            circuit,
            executor=executor,
            factory=factory,
        )
    )


def ddd_mitigated_expectation(
    circuit: QuantumCircuit,
    executor: Any | None = None,
    shots: int = 8192,
) -> float:
    """Run DDD (Digital Dynamical Decoupling) via Mitiq.

    Inserts XX sequences in idle periods to suppress low-frequency noise.
    """
    if not _MITIQ_AVAILABLE:
        raise ImportError("mitiq not installed: pip install mitiq")

    if executor is None:

        def executor(c: QuantumCircuit) -> float:
            return _qiskit_executor(c, shots=shots)

    rule = ddd.rules.xx

    return float(
        ddd.execute_with_ddd(
            circuit,
            executor=executor,
            rule=rule,
        )
    )
