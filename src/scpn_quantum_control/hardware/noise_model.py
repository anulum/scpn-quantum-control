"""Heron r2 noise model for realistic local simulation.

Median calibration values from IBM ibm_fez (February 2026).
"""

from __future__ import annotations

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# ibm_fez Heron r2 median calibration (Feb 2026)
T1_US = 300.0
T2_US = 200.0
CZ_ERROR_RATE = 0.005
READOUT_ERROR_RATE = 0.002
SINGLE_GATE_TIME_US = 0.06
TWO_GATE_TIME_US = 0.66


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
