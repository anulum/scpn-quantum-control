# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Automatic Backend Selection
"""Auto-select simulation backend based on system size and available resources.

Chooses between:
  - Exact diagonalisation (numpy eigh) for n <= 14
  - Z2 parity-sector ED for n <= 16
  - MPS/DMRG (quimb) for n <= 64
  - Qiskit statevector for circuit-based simulation
  - IBM hardware for real-device execution

Inspired by Maestro (Qoro, arXiv:2512.04216).
"""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


def recommend_backend(
    n: int,
    ram_gb: float = 32.0,
    has_quimb: bool = False,
    has_gpu: bool = False,
    want_open_system: bool = False,
) -> dict:
    """Recommend the best simulation backend for given system size.

    Parameters
    ----------
    n : int
        Number of oscillators (qubits).
    ram_gb : float
        Available RAM in GB.
    has_quimb : bool
        Whether quimb is installed.
    has_gpu : bool
        Whether GPU (JAX/CuPy) is available.
    want_open_system : bool
        Whether Lindblad dynamics are needed.

    Returns
    -------
    dict with keys: backend, reason, memory_mb, feasible
    """
    dim = 2**n
    sector_dim = 2 ** (n - 1)

    # Memory estimates (complex128 = 16 bytes per element)
    full_ed_mb = dim * dim * 16 / 1e6
    sector_ed_mb = sector_dim * sector_dim * 16 / 1e6
    sv_mb = dim * 16 / 1e6  # statevector only
    lindblad_mb = dim * dim * 16 / 1e6  # density matrix

    ram_mb = ram_gb * 1000

    if want_open_system:
        if lindblad_mb < ram_mb * 0.5 and n <= 12:
            return {
                "backend": "lindblad_scipy",
                "reason": f"Density matrix {lindblad_mb:.0f} MB fits in RAM",
                "memory_mb": lindblad_mb,
                "feasible": True,
            }
        if has_quimb and n <= 64:
            return {
                "backend": "tjm_mps",
                "reason": "Open-system MPS (tensor jump method) for large n",
                "memory_mb": n * 64 * 16 / 1e6,  # rough MPS estimate
                "feasible": True,
                "note": "TJM not yet implemented — use lindblad_scipy for n<=12",
            }
        return {
            "backend": "lindblad_scipy",
            "reason": f"n={n} density matrix needs {lindblad_mb:.0f} MB — may be slow",
            "memory_mb": lindblad_mb,
            "feasible": lindblad_mb < ram_mb,
        }

    # Closed-system backends
    if full_ed_mb < ram_mb * 0.3 and n <= 14:
        return {
            "backend": "exact_diag",
            "reason": f"Full ED: {full_ed_mb:.0f} MB, n={n} ≤ 14",
            "memory_mb": full_ed_mb,
            "feasible": True,
        }

    # U(1) magnetisation sectors (strongest reduction)
    from math import comb

    u1_largest_dim = comb(n, n // 2)
    u1_mb = u1_largest_dim * u1_largest_dim * 16 / 1e6
    if u1_mb < ram_mb * 0.3 and n <= 20:
        return {
            "backend": "u1_sector_ed",
            "reason": f"U(1) magnetisation ED: largest sector {u1_largest_dim} states, {u1_mb:.0f} MB",
            "memory_mb": u1_mb,
            "feasible": True,
        }

    if sector_ed_mb < ram_mb * 0.3 and n <= 16:
        return {
            "backend": "sector_ed",
            "reason": f"Z2 parity ED: {sector_ed_mb:.0f} MB per sector",
            "memory_mb": sector_ed_mb,
            "feasible": True,
        }

    if has_quimb and n <= 64:
        bond_mem = n * 64 * 64 * 16 / 1e6  # rough MPS bond=64 estimate
        return {
            "backend": "mps_dmrg",
            "reason": f"MPS/DMRG: bond=64, ~{bond_mem:.0f} MB",
            "memory_mb": bond_mem,
            "feasible": True,
        }

    if has_gpu and n <= 30:
        return {
            "backend": "gpu_statevector",
            "reason": f"GPU statevector: {sv_mb:.0f} MB on device",
            "memory_mb": sv_mb,
            "feasible": True,
        }

    if sv_mb < ram_mb * 0.5:
        return {
            "backend": "statevector",
            "reason": f"CPU statevector: {sv_mb:.0f} MB",
            "memory_mb": sv_mb,
            "feasible": True,
        }

    return {
        "backend": "hardware",
        "reason": f"n={n} too large for classical sim ({full_ed_mb:.0f} MB needed)",
        "memory_mb": 0,
        "feasible": True,
        "note": "Submit to IBM hardware",
    }


def auto_solve(
    K: np.ndarray,
    omega: np.ndarray,
    ram_gb: float = 32.0,
    want_open_system: bool = False,
    gamma_amp: float = 0.0,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.1,
) -> dict:
    """Automatically select backend and run simulation.

    Returns dict with keys: backend_used, result, recommendation
    """
    n = K.shape[0]

    try:
        from ..phase.mps_evolution import is_quimb_available

        has_quimb = is_quimb_available()
    except Exception:
        has_quimb = False

    rec = recommend_backend(
        n,
        ram_gb=ram_gb,
        has_quimb=has_quimb,
        want_open_system=want_open_system,
    )

    backend = rec["backend"]

    if backend == "lindblad_scipy" and want_open_system:
        from ..phase.lindblad import LindbladKuramotoSolver

        solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=gamma_amp, gamma_deph=gamma_deph)
        result = solver.run(t_max=t_max, dt=dt)
        return {"backend_used": backend, "result": result, "recommendation": rec}

    if backend == "exact_diag":
        H = knm_to_dense_matrix(K, omega)
        eigvals, eigvecs = np.linalg.eigh(H)
        return {
            "backend_used": backend,
            "result": {"eigvals": eigvals, "eigvecs": eigvecs, "ground_energy": float(eigvals[0])},
            "recommendation": rec,
        }

    if backend == "u1_sector_ed":
        from ..analysis.magnetisation_sectors import eigh_by_magnetisation

        result = eigh_by_magnetisation(K, omega)
        return {"backend_used": backend, "result": result, "recommendation": rec}

    if backend == "sector_ed":
        from ..analysis.symmetry_sectors import eigh_by_sector

        result = eigh_by_sector(K, omega)
        return {"backend_used": backend, "result": result, "recommendation": rec}

    if backend == "mps_dmrg":
        from ..phase.mps_evolution import dmrg_ground_state

        result = dmrg_ground_state(K, omega, bond_dim=64, max_sweeps=20)
        return {"backend_used": backend, "result": result, "recommendation": rec}

    # Fallback: Qiskit statevector
    from ..phase.xy_kuramoto import QuantumKuramotoSolver

    qk_solver = QuantumKuramotoSolver(n, K, omega)
    result = qk_solver.run(t_max=t_max, dt=dt)
    return {"backend_used": "statevector", "result": result, "recommendation": rec}
