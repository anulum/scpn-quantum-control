# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""End-to-end pipeline: K_nm → quantum sim → ZNE + PEC → classical comparison.

Demonstrates the full scpn-quantum-control workflow in a single script.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control import (
    OMEGA_N_16,
    QuantumKuramotoSolver,
    build_knm_paper27,
)
from scpn_quantum_control.hardware.classical import (
    classical_exact_evolution,
)
from scpn_quantum_control.mitigation.pec import pec_sample
from scpn_quantum_control.mitigation.zne import gate_fold_circuit, zne_extrapolate
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def main() -> None:
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print("=" * 70)
    print("  scpn-quantum-control: End-to-End Pipeline Demo")
    print("=" * 70)
    print(f"\nSystem: {n} coupled Kuramoto oscillators (Paper 27 parameters)")
    print(f"K_base=0.45, alpha=0.3, omega range [{omega.min():.3f}, {omega.max():.3f}]")

    # --- Stage 1: VQE ground state ---
    print("\n--- Stage 1: VQE Ground State ---")
    t0 = time.perf_counter()
    vqe = PhaseVQE(K, omega, ansatz_reps=2)
    sol = vqe.solve(maxiter=300, seed=42)
    t_vqe = time.perf_counter() - t0
    print(f"  VQE energy:   {sol['ground_energy']:.6f}")
    print(f"  Exact energy: {sol['exact_energy']:.6f}")
    print(f"  Error:        {sol['relative_error_pct']:.4f}%")
    print(f"  Time:         {t_vqe:.1f}s")

    # --- Stage 2: Trotter dynamics ---
    print("\n--- Stage 2: Quantum Trotter Evolution ---")
    t0 = time.perf_counter()
    solver = QuantumKuramotoSolver(n, K, omega)
    result_q = solver.run(t_max=1.0, dt=0.2, trotter_per_step=3)
    t_trotter = time.perf_counter() - t0
    print(f"  R(t): {[f'{r:.3f}' for r in result_q['R']]}")
    print(f"  Time: {t_trotter:.1f}s")

    # --- Stage 3: Classical reference ---
    print("\n--- Stage 3: Classical Exact Evolution ---")
    t0 = time.perf_counter()
    result_c = classical_exact_evolution(n, t_max=1.0, dt=0.2, K=K, omega=omega)
    t_classical = time.perf_counter() - t0
    print(f"  R(t): {[f'{r:.3f}' for r in result_c['R']]}")
    print(f"  Time: {t_classical:.1f}s")

    # --- Stage 4: ZNE mitigation ---
    print("\n--- Stage 4: ZNE Error Mitigation ---")
    qc = solver.evolve(time=0.5, trotter_steps=3)
    sv_exact = Statevector.from_instruction(qc)
    exact_exp = 2 * float(sv_exact.probabilities([0])[0]) - 1

    scales = [1, 3, 5]
    exps = []
    for s in scales:
        if s == 1:
            sv_s = sv_exact
        else:
            qc_folded = gate_fold_circuit(qc, s)
            sv_s = Statevector.from_instruction(qc_folded)
        exps.append(2 * float(sv_s.probabilities([0])[0]) - 1)

    zne = zne_extrapolate(scales, exps, order=1)
    print(f"  Raw <Z_0>:       {exps[0]:.6f}")
    print(f"  ZNE <Z_0>:       {zne.zero_noise_estimate:.6f}")
    print(f"  Exact <Z_0>:     {exact_exp:.6f}")
    print(f"  ZNE residual:    {zne.fit_residual:.2e}")

    # --- Stage 5: PEC mitigation ---
    print("\n--- Stage 5: PEC Error Mitigation ---")
    pec_result = pec_sample(qc, 0.01, 3000, observable_qubit=0, rng=np.random.default_rng(42))
    print(f"  PEC <Z_0>:       {pec_result.mitigated_value:.6f}")
    print(f"  PEC overhead:    {pec_result.overhead:.1f}")
    print(f"  Exact <Z_0>:     {exact_exp:.6f}")

    # --- Summary ---
    print("\n--- Summary ---")
    q_final = result_q["R"][-1]
    c_final = result_c["R"][-1]
    err_pct = abs(q_final - c_final) / max(c_final, 1e-10) * 100
    print(f"  R(t=1.0) quantum:  {q_final:.4f}")
    print(f"  R(t=1.0) classical: {c_final:.4f}")
    print(f"  Trotter error:     {err_pct:.1f}%")
    print(f"  VQE error:         {sol['relative_error_pct']:.4f}%")
    print(f"  ZNE error:         {abs(zne.zero_noise_estimate - exact_exp):.6f}")
    print(f"  PEC error:         {abs(pec_result.mitigated_value - exact_exp):.6f}")
    print(f"\nPipeline complete. {n}-qubit system verified across 5 stages.")


if __name__ == "__main__":
    main()
