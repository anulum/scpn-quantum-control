# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Dynamical decoupling on a noisy 4-qubit Kuramoto circuit.

Compares fidelity (via order parameter R) of a Trotter evolution
with and without XY4 DD under synthetic Heron r2 noise.
No QPU needed.
"""

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_exact_evolution
from scpn_quantum_control.hardware.experiments import (
    _R_from_xyz,
    _build_evo_base,
    _build_xyz_circuits,
)
from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model
from scpn_quantum_control.hardware.runner import HardwareRunner

n = 4
dt = 0.1
K = build_knm_paper27(L=n)
omega = OMEGA_N_16[:n]
base = _build_evo_base(n, K, omega, t=dt, trotter_reps=2)

nm = heron_r2_noise_model(cz_error=0.02)
runner = HardwareRunner(use_simulator=True, noise_model=nm)
runner.connect()

exact = classical_exact_evolution(n, dt, dt, K, omega)
R_exact = exact["R"][-1]
print(f"Exact R = {R_exact:.4f}\n")

# Without DD
qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
hw_raw = runner.run_sampler([qc_z, qc_x, qc_y], shots=5000, name="raw")
R_raw, _, *_ = _R_from_xyz(hw_raw[0].counts, hw_raw[1].counts, hw_raw[2].counts, n)
print(f"Raw     R = {R_raw:.4f}  (error = {abs(R_raw - R_exact):.4f})")

# With XY4 DD
dd_base = runner.transpile_with_dd(base)
qc_z_dd, qc_x_dd, qc_y_dd = _build_xyz_circuits(dd_base, n)
hw_dd = runner.run_sampler([qc_z_dd, qc_x_dd, qc_y_dd], shots=5000, name="dd")
R_dd, _, *_ = _R_from_xyz(hw_dd[0].counts, hw_dd[1].counts, hw_dd[2].counts, n)
print(f"DD (XY4) R = {R_dd:.4f}  (error = {abs(R_dd - R_exact):.4f})")

improvement = abs(R_raw - R_exact) - abs(R_dd - R_exact)
print(f"\nDD improvement: {improvement:+.4f}")
