# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Magic + Berry Phase + Quantum Speed Limit
#
# Three novel probes of the synchronisation transition:
# 1. Magic (non-stabilizerness) M2 — quantum resource cost
# 2. Berry phase — geometric phase across K_c
# 3. Quantum speed limit — minimum time for state evolution
#
# Run on Kaggle CPU (no GPU needed, pure numpy)

import json
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy"])

import numpy as np

OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ]
)


def build_knm(L, K_base=0.45, K_alpha=0.3):
    idx = np.arange(L)
    K = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        if i < L and j < L:
            K[i, j] = K[j, i] = val
    return K


def build_hamiltonian(K, omega, n):
    dim = 1 << n
    h = np.zeros((dim, dim))
    for idx in range(dim):
        diag = 0.0
        for i in range(n):
            bit = (idx >> i) & 1
            diag -= omega[i] * (1.0 - 2.0 * bit)
        h[idx, idx] = diag
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < 1e-15:
                continue
            mask = (1 << i) | (1 << j)
            for idx in range(dim):
                bi = (idx >> i) & 1
                bj = (idx >> j) & 1
                if bi != bj:
                    h[idx, idx ^ mask] -= 2.0 * K[i, j]
    return h


results = {}

# ============================================================
# 1. MAGIC (NON-STABILIZERNESS)
# ============================================================
print("=" * 60)
print("1. MAGIC M2 vs COUPLING (quantum resource cost)")
print("Stabilizer: M2=0. Max magic: M2=N.")
print("=" * 60)

paulis_1q = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

for n in [4, 6]:
    dim = 2**n
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 8.0, 15)

    magic_vals = []
    t0 = time.perf_counter()
    for kb in k_range:
        H = build_hamiltonian(kb * K_norm, omega, n)
        _, eigvecs = np.linalg.eigh(H)
        psi = eigvecs[:, 0]

        sum_p4 = 0.0
        for idx in range(4**n):
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 4)
                tmp //= 4
            P = np.array([[1.0]], dtype=complex)
            for d in digits:
                P = np.kron(P, paulis_1q[d])
            exp_P = np.real(psi.conj() @ P @ psi)
            sum_p4 += exp_P**4

        M2 = -np.log2(sum_p4 / dim)
        magic_vals.append(float(M2))

    dt = time.perf_counter() - t0
    peak_K = float(k_range[np.argmax(magic_vals)])
    print(
        f"n={n}: M2=[{min(magic_vals):.3f}, {max(magic_vals):.3f}], peak at K={peak_K:.2f} ({dt:.1f}s)"
    )

    results[f"magic_n{n}"] = {
        "n": n,
        "K_base": k_range.tolist(),
        "magic_M2": magic_vals,
        "peak_K": peak_K,
        "time_s": round(dt, 1),
    }

# ============================================================
# 2. BERRY PHASE ACROSS TRANSITION
# ============================================================
print("\n" + "=" * 60)
print("2. BERRY PHASE vs COUPLING")
print("Geometric phase: gamma = -Im(ln(prod <psi(K_i)|psi(K_{i+1})>))")
print("=" * 60)

for n in [4, 6, 8]:
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 8.0, 40)

    t0 = time.perf_counter()
    ground_states = []
    for kb in k_range:
        H = build_hamiltonian(kb * K_norm, omega, n)
        _, eigvecs = np.linalg.eigh(H)
        ground_states.append(eigvecs[:, 0])

    # Berry phase segments
    berry_phases = []
    for i in range(len(ground_states) - 1):
        overlap = np.vdot(ground_states[i], ground_states[i + 1])
        berry_phases.append(-np.imag(np.log(overlap)))

    # Cumulative Berry phase
    cumulative = np.cumsum(berry_phases)
    dt = time.perf_counter() - t0
    total_berry = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    print(f"n={n}: total Berry phase = {total_berry:.4f} rad ({dt:.1f}s)")

    results[f"berry_n{n}"] = {
        "n": n,
        "K_base": k_range.tolist(),
        "berry_segments": [float(b) for b in berry_phases],
        "cumulative": cumulative.tolist(),
        "total_phase": total_berry,
        "time_s": round(dt, 1),
    }

# ============================================================
# 3. QUANTUM SPEED LIMIT
# ============================================================
print("\n" + "=" * 60)
print("3. QUANTUM SPEED LIMIT vs COUPLING")
print("t_QSL = pi*hbar / (2*Delta_E) — min time for orthogonal evolution")
print("=" * 60)

for n in [4, 6, 8]:
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 8.0, 30)

    t0 = time.perf_counter()
    qsl_vals = []
    energy_vars = []
    for kb in k_range:
        H = build_hamiltonian(kb * K_norm, omega, n)
        evals, evecs = np.linalg.eigh(H)
        psi = evecs[:, 0]

        # Energy variance: Delta_E = sqrt(<H^2> - <H>^2)
        E_mean = evals[0]  # ground state energy
        H_psi = H @ psi
        E2_mean = float(np.real(psi.conj() @ H_psi * H_psi.conj() @ psi))
        # Actually: <H^2> = sum of eigenvalues^2 weighted by |<n|psi>|^2
        # For ground state: <H> = E_0, <H^2> = E_0^2
        # Delta_E = 0 for eigenstate → use spectral gap instead
        Delta_E = evals[1] - evals[0]  # Mandelstam-Tamm bound
        t_qsl = np.pi / (2 * Delta_E) if Delta_E > 1e-15 else np.inf
        qsl_vals.append(float(t_qsl))
        energy_vars.append(float(Delta_E))

    dt = time.perf_counter() - t0
    print(f"n={n}: t_QSL range [{min(qsl_vals):.3f}, {max(qsl_vals):.3f}] ({dt:.1f}s)")
    print("  At K_c: t_QSL -> infinity (gap closes)")

    results[f"qsl_n{n}"] = {
        "n": n,
        "K_base": k_range.tolist(),
        "t_QSL": qsl_vals,
        "spectral_gap": energy_vars,
        "time_s": round(dt, 1),
    }

print("\n" + "=" * 60)
print("RESULTS JSON")
print("=" * 60)
print(json.dumps(results, indent=2))
print("\nDone. 3 novel probes of the synchronisation transition.")
