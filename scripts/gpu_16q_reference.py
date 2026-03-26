# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — 16-qubit Classical Reference via Sparse Krylov
"""Compute exact classical references for n=2..16 experiments.

Uses scipy.sparse.linalg.expm_multiply (Krylov subspace) for the time
evolution, avoiding the full 2^n × 2^n propagator matrix. Memory is
O(2^n) — 512 KB for n=16 instead of 64 GiB for the dense path.
"""

import json
import os
import time

os.environ.setdefault("SCPN_GPU_ENABLE", "1")

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
)
from scpn_quantum_control.hardware.classical import (
    classical_exact_diag,
    classical_exact_evolution,
)


def main():
    results = {}

    # 1. Exact diagonalization at n=16 (sparse eigsh, fast)
    print("=== 16-qubit exact diagonalization ===")
    K16 = build_knm_paper27(L=16)
    omega16 = OMEGA_N_16.copy()
    t0 = time.time()
    diag = classical_exact_diag(16, K16, omega16)
    t_diag = time.time() - t0
    print(f"  Ground energy: {diag['ground_energy']:.8f}")
    print(f"  Spectral gap:  {diag['spectral_gap']:.8f}")
    print(f"  Time: {t_diag:.1f}s")
    results["diag_16q"] = {
        "ground_energy": float(diag["ground_energy"]),
        "spectral_gap": float(diag["spectral_gap"]),
        "time_s": round(t_diag, 2),
    }

    # 2. Exact evolution for all qubit counts (decoherence scaling reference)
    print("\n=== Exact evolution: n=2..16, dt=0.1 ===")
    for n in [2, 4, 6, 8, 10, 12, 14, 16]:
        Kn = build_knm_paper27(L=n)
        on = OMEGA_N_16[:n]
        path = "sparse" if n >= 13 else "dense"
        t0 = time.time()
        evo = classical_exact_evolution(n, 0.1, 0.1, Kn, on)
        t_evo = time.time() - t0
        R = float(evo["R"][-1])
        print(f"  n={n:2d} ({path:6s}): R={R:.10f}, time={t_evo:.2f}s")
        results[f"evo_{n}q_dt0.1"] = {
            "R": R,
            "path": path,
            "time_s": round(t_evo, 2),
        }

    # 3. 16-qubit evolution at multiple dt values (for hardware comparison)
    print("\n=== 16-qubit evolution at multiple dt ===")
    for dt in [0.01, 0.02, 0.05, 0.1, 0.2]:
        t0 = time.time()
        evo = classical_exact_evolution(16, dt, dt, K16, omega16)
        t_evo = time.time() - t0
        R = float(evo["R"][-1])
        print(f"  dt={dt:.2f}: R={R:.10f}, time={t_evo:.2f}s")
        results[f"evo_16q_dt{dt}"] = {
            "R": R,
            "time_s": round(t_evo, 2),
        }

    # 4. Multi-step evolution at n=16 (8 steps for kuramoto-style comparison)
    print("\n=== 16-qubit 8-step evolution, dt=0.05 ===")
    t0 = time.time()
    evo_multi = classical_exact_evolution(16, 0.4, 0.05, K16, omega16)
    t_evo = time.time() - t0
    print(f"  Steps: {len(evo_multi['R']) - 1}")
    print(f"  R trajectory: {[f'{r:.6f}' for r in evo_multi['R']]}")
    print(f"  Time: {t_evo:.2f}s")
    results["evo_16q_8step_dt0.05"] = {
        "R_trajectory": [float(r) for r in evo_multi["R"]],
        "times": [float(t) for t in evo_multi["times"]],
        "time_s": round(t_evo, 2),
    }

    # 5. Exact diag for all qubit counts
    print("\n=== Exact diagonalization: n=2..16 ===")
    for n in [2, 4, 6, 8, 10, 12, 14, 16]:
        Kn = build_knm_paper27(L=n)
        on = OMEGA_N_16[:n]
        t0 = time.time()
        diag_n = classical_exact_diag(n, Kn, on)
        t_diag = time.time() - t0
        print(
            f"  n={n:2d}: E0={diag_n['ground_energy']:.8f}, gap={diag_n['spectral_gap']:.8f}, time={t_diag:.2f}s"
        )
        results[f"diag_{n}q"] = {
            "ground_energy": float(diag_n["ground_energy"]),
            "spectral_gap": float(diag_n["spectral_gap"]),
            "time_s": round(t_diag, 2),
        }

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/classical_16q_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== All results saved to {out_path} ===")
    print(f"=== Total entries: {len(results)} ===")


if __name__ == "__main__":
    main()
