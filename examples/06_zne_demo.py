"""ZNE error mitigation on a noisy simulator.

Demonstrates unitary folding + Richardson extrapolation to improve
the Kuramoto order parameter estimate under synthetic Heron r2 noise.
No QPU needed.
"""

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.experiments import (
    _R_from_xyz,
    _build_evo_base,
    _build_xyz_circuits,
)
from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model
from scpn_quantum_control.hardware.runner import HardwareRunner
from scpn_quantum_control.mitigation.zne import gate_fold_circuit, zne_extrapolate

n = 4
K = build_knm_paper27(L=n)
omega = OMEGA_N_16[:n]
base = _build_evo_base(n, K, omega, t=0.1, trotter_reps=2)

nm = heron_r2_noise_model(cz_error=0.02)
runner = HardwareRunner(use_simulator=True, noise_model=nm)
runner.connect()

scales = [1, 3, 5]
R_vals = []

for s in scales:
    folded = gate_fold_circuit(base, s)
    qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
    hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=5000, name=f"zne_s{s}")
    R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
    R_vals.append(R)
    print(f"  scale={s}: R = {R:.4f}")

result = zne_extrapolate(scales, R_vals, order=1)
print(f"\nZNE extrapolated R(0) = {result.zero_noise_estimate:.4f}")
print(f"Fit residual = {result.fit_residual:.4f}")
print(f"Raw (scale=1) R = {R_vals[0]:.4f}")
print(f"Improvement: {result.zero_noise_estimate - R_vals[0]:+.4f}")
