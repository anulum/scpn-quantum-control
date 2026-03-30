# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — QuTiP vs Qiskit Comparison (NB48)
"""
Notebook 48: QuTiP Lindblad vs Qiskit Unitary vs Our Lindblad
==============================================================

Compares three approaches to simulating the same 4-oscillator
Kuramoto-XY system:

1. QuTiP mesolve (Lindblad master equation) — community standard
2. Qiskit statevector (unitary, closed system)
3. Our LindbladKuramotoSolver (scipy Lindblad)

Shows: R(t), purity decay, and where each tool is appropriate.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# --- Parameters ---
N = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(N), range(N))))
omega = np.linspace(0.8, 1.2, N)
T_MAX = 2.0
DT = 0.05
GAMMA_AMP = 0.05
GAMMA_DEPH = 0.02

results = {}

# =====================================================================
# 1. QuTiP Lindblad (mesolve)
# =====================================================================
print("=" * 60)
print("1. QuTiP mesolve (Lindblad)")
print("=" * 60)

import qutip

# Build Hamiltonian
H_terms = []
for i in range(N):
    op_list = [qutip.qeye(2)] * N
    op_list[i] = qutip.sigmaz()
    H_terms.append(-omega[i] * qutip.tensor(op_list))

for i in range(N):
    for j in range(i + 1, N):
        if abs(K[i, j]) < 1e-12:
            continue
        for pauli_pair in [(qutip.sigmax, qutip.sigmax), (qutip.sigmay, qutip.sigmay)]:
            op_list = [qutip.qeye(2)] * N
            op_list[i] = pauli_pair[0]()
            op_list[j] = pauli_pair[1]()
            H_terms.append(-K[i, j] * qutip.tensor(op_list))

H_qutip = sum(H_terms)

# Collapse operators
c_ops = []
for i in range(N):
    if GAMMA_AMP > 0:
        op_list = [qutip.qeye(2)] * N
        op_list[i] = np.sqrt(GAMMA_AMP) * qutip.destroy(2)
        c_ops.append(qutip.tensor(op_list))
    if GAMMA_DEPH > 0:
        op_list = [qutip.qeye(2)] * N
        op_list[i] = np.sqrt(GAMMA_DEPH / 2) * qutip.sigmaz()
        c_ops.append(qutip.tensor(op_list))

# Initial state: product of Ry-rotated qubits
psi_list = []
for i in range(N):
    angle = float(omega[i]) % (2 * np.pi)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    psi_list.append(qutip.Qobj([[c], [s]]))
psi0 = qutip.tensor(psi_list)
rho0_qutip = psi0 * psi0.dag()

# Observables for order parameter
sx_ops = []
sy_ops = []
for i in range(N):
    op = [qutip.qeye(2)] * N
    op[i] = qutip.sigmax()
    sx_ops.append(qutip.tensor(op))
    op2 = [qutip.qeye(2)] * N
    op2[i] = qutip.sigmay()
    sy_ops.append(qutip.tensor(op2))

tlist = np.arange(0, T_MAX + DT, DT)

t0 = time.perf_counter()
result_qt = qutip.mesolve(
    H_qutip,
    rho0_qutip,
    tlist,
    c_ops,
    sx_ops + sy_ops,
    options={"store_states": True},
)
t_qutip = time.perf_counter() - t0

# Extract R(t)
R_qutip = np.zeros(len(tlist))
purity_qutip = np.zeros(len(tlist))
for ti in range(len(tlist)):
    z = 0.0 + 0.0j
    for i in range(N):
        z += result_qt.expect[i][ti] + 1j * result_qt.expect[N + i][ti]
    z /= N
    R_qutip[ti] = abs(z)

# Purity from states
for ti in range(len(tlist)):
    rho_t = result_qt.states[ti] if result_qt.states else None
    if rho_t is not None:
        purity_qutip[ti] = (rho_t * rho_t).tr().real

results["qutip"] = {
    "R": R_qutip.tolist(),
    "purity": purity_qutip.tolist(),
    "wall_time_ms": round(t_qutip * 1000, 1),
}
print(f"  R(0)={R_qutip[0]:.4f}, R(T)={R_qutip[-1]:.4f}, wall={t_qutip * 1000:.0f}ms")

# =====================================================================
# 2. Qiskit statevector (unitary, no dissipation)
# =====================================================================
print("\n" + "=" * 60)
print("2. Qiskit statevector (unitary)")
print("=" * 60)

from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

t0 = time.perf_counter()
solver_qiskit = QuantumKuramotoSolver(N, K, omega)
result_qiskit = solver_qiskit.run(t_max=T_MAX, dt=DT)
t_qiskit = time.perf_counter() - t0

results["qiskit_unitary"] = {
    "R": result_qiskit["R"].tolist(),
    "purity": [1.0] * len(result_qiskit["R"]),  # always 1 for unitary
    "wall_time_ms": round(t_qiskit * 1000, 1),
}
print(
    f"  R(0)={result_qiskit['R'][0]:.4f}, R(T)={result_qiskit['R'][-1]:.4f}, wall={t_qiskit * 1000:.0f}ms"
)

# =====================================================================
# 3. Our Lindblad solver
# =====================================================================
print("\n" + "=" * 60)
print("3. LindbladKuramotoSolver (scipy)")
print("=" * 60)

from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver

t0 = time.perf_counter()
solver_lb = LindbladKuramotoSolver(N, K, omega, gamma_amp=GAMMA_AMP, gamma_deph=GAMMA_DEPH)
result_lb = solver_lb.run(t_max=T_MAX, dt=DT)
t_lindblad = time.perf_counter() - t0

results["our_lindblad"] = {
    "R": result_lb["R"].tolist(),
    "purity": result_lb["purity"].tolist(),
    "wall_time_ms": round(t_lindblad * 1000, 1),
}
print(
    f"  R(0)={result_lb['R'][0]:.4f}, R(T)={result_lb['R'][-1]:.4f}, wall={t_lindblad * 1000:.0f}ms"
)
print(f"  Purity: {result_lb['purity'][0]:.4f} → {result_lb['purity'][-1]:.4f}")

# =====================================================================
# Comparison
# =====================================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"{'Method':<30} {'R(0)':>8} {'R(T)':>8} {'Purity(T)':>10} {'Time':>8}")
print("-" * 60)
print(
    f"{'QuTiP mesolve (Lindblad)':<30} {R_qutip[0]:8.4f} {R_qutip[-1]:8.4f} {purity_qutip[-1]:10.4f} {t_qutip * 1000:7.0f}ms"
)
print(
    f"{'Qiskit statevector (unitary)':<30} {result_qiskit['R'][0]:8.4f} {result_qiskit['R'][-1]:8.4f} {'1.0000':>10} {t_qiskit * 1000:7.0f}ms"
)
print(
    f"{'Our Lindblad (scipy)':<30} {result_lb['R'][0]:8.4f} {result_lb['R'][-1]:8.4f} {result_lb['purity'][-1]:10.4f} {t_lindblad * 1000:7.0f}ms"
)

print("\nKey observations:")
r_diff = abs(R_qutip[-1] - result_lb["R"][-1])
print(f"  QuTiP vs our Lindblad R(T) difference: {r_diff:.4f}")
print(f"  Unitary R(T) vs Lindblad R(T) gap: {abs(result_qiskit['R'][-1] - R_qutip[-1]):.4f}")
if purity_qutip[-1] > 0:
    print(
        f"  Purity decay: {1 - purity_qutip[-1]:.4f} (QuTiP), {1 - result_lb['purity'][-1]:.4f} (ours)"
    )

# Save
results["parameters"] = {
    "N": N,
    "K_base": 0.45,
    "alpha": 0.3,
    "T_MAX": T_MAX,
    "DT": DT,
    "gamma_amp": GAMMA_AMP,
    "gamma_deph": GAMMA_DEPH,
}
out_path = Path(__file__).resolve().parent.parent / "results" / "qutip_comparison_2026-03-30.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
