# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Noise Model
"""Heron r2 noise model for realistic local simulation.

Median calibration values from IBM ibm_fez, taken from the dated
``backend.properties`` snapshot of 2026-03-29 (retrieved read-only 2026-07-18;
0 QPU seconds). The earlier constants ("Feb 2026") were ~2x optimistic on
T1/T2, ~2x pessimistic on CZ error, ~7.5x optimistic on readout, and ~10x too
long on the two-qubit gate time; they are corrected here against source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_aer.noise import NoiseModel

# ibm_fez Heron r2 median calibration (2026-03-29 snapshot; source:
# backend.properties(datetime=2026-03-29), 156 qubits / 352 CZ pairs).
T1_US = 146.7
T2_US = 109.3
CZ_ERROR_RATE = 0.00262
READOUT_ERROR_RATE = 0.01508
SINGLE_GATE_TIME_US = 0.024
TWO_GATE_TIME_US = 0.068


def heron_r2_noise_model(
    t1_us: float = T1_US,
    t2_us: float = T2_US,
    cz_error: float = CZ_ERROR_RATE,
    readout_error: float = READOUT_ERROR_RATE,
) -> NoiseModel:
    """Build a noise model approximating IBM Heron r2 calibration data.

    Single-qubit gates get thermal relaxation only. Two-qubit gates (ecr/cz)
    get thermal relaxation + depolarizing noise at the given error rate.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    model = NoiseModel()

    sq_relax = thermal_relaxation_error(t1_us, t2_us, SINGLE_GATE_TIME_US)
    model.add_all_qubit_quantum_error(sq_relax, ["sx", "x", "rz"])

    tq_relax = thermal_relaxation_error(t1_us, t2_us, TWO_GATE_TIME_US)
    tq_depol = depolarizing_error(cz_error, 2)
    tq_combined = tq_depol.compose(tq_relax.tensor(tq_relax))
    model.add_all_qubit_quantum_error(tq_combined, ["ecr", "cz"])

    from qiskit_aer.noise import ReadoutError

    ro = ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
    model.add_all_qubit_readout_error(ro)

    return model
