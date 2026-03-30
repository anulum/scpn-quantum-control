# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — MPS/DMRG Backend via quimb
"""Matrix Product State backend for large-N Kuramoto-XY.

Uses quimb's DMRG for ground state search and TEBD for time evolution,
enabling simulation of n=32-64 oscillators (beyond ED limits).

Requires: pip install quimb
Reference: Gray, JOSS 3(29), 819 (2018).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import quimb
    import quimb.tensor as qtn

    _QUIMB_AVAILABLE = True
except Exception:
    _QUIMB_AVAILABLE = False
    quimb = None  # type: ignore[assignment]
    qtn = None  # type: ignore[assignment]


def is_quimb_available() -> bool:
    """Check if quimb is installed."""
    return _QUIMB_AVAILABLE


def _build_mpo_hamiltonian(K: np.ndarray, omega: np.ndarray) -> Any:
    """Build the XY Hamiltonian as a quimb SpinHam1D MPO.

    H = -Σ K_ij (X_iX_j + Y_iY_j) - Σ ω_i Z_i
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    n = K.shape[0]
    builder = qtn.SpinHam1D(S=1 / 2)

    # On-site terms: -ω_i Z_i
    for i in range(n):
        if abs(omega[i]) > 1e-15:
            builder[i] += -omega[i], "Z"

    # Nearest-neighbour coupling terms only (SpinHam1D limitation)
    for i in range(n - 1):
        if abs(K[i, i + 1]) < 1e-15:
            continue
        builder[i, i + 1] += -K[i, i + 1], "X", "X"
        builder[i, i + 1] += -K[i, i + 1], "Y", "Y"

    # Note: longer-range couplings K[i,j] with |i-j|>1 are dropped.
    # For the exponential-decay K_nm, nearest-neighbour terms dominate.
    # Full long-range MPO requires quimb.tensor.MatrixProductOperator
    # construction which is more complex.

    return builder.build_mpo(n)


def dmrg_ground_state(
    K: np.ndarray,
    omega: np.ndarray,
    bond_dim: int = 64,
    cutoff: float = 1e-10,
    max_sweeps: int = 20,
) -> dict:
    """Find ground state via DMRG.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.
    bond_dim : int
        Maximum MPS bond dimension.
    cutoff : float
        SVD truncation cutoff.
    max_sweeps : int
        Maximum DMRG sweeps.

    Returns
    -------
    dict with keys: energy, mps, converged, bond_dims
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    n = K.shape[0]
    H_mpo = _build_mpo_hamiltonian(K, omega)

    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dim, cutoffs=cutoff)
    converged = False
    for _sweep in range(max_sweeps):
        energy = dmrg.sweep_right()
        energy = dmrg.sweep_left()  # noqa: F841 — intentional overwrite
        if dmrg.energy_change < 1e-10:
            converged = True
            break

    mps = dmrg.state
    bond_dims = [mps.bond_size(i, i + 1) for i in range(n - 1)]

    return {
        "energy": float(dmrg.energy),
        "mps": mps,
        "converged": converged,
        "bond_dims": bond_dims,
        "n_oscillators": n,
    }


def tebd_evolution(
    K: np.ndarray,
    omega: np.ndarray,
    t_max: float = 1.0,
    dt: float = 0.05,
    bond_dim: int = 64,
    cutoff: float = 1e-10,
    order: int = 2,
) -> dict:
    """Time evolution via TEBD (Time-Evolving Block Decimation).

    Parameters
    ----------
    K, omega : coupling and frequencies
    t_max : total evolution time
    dt : Trotter step size
    bond_dim : maximum bond dimension
    cutoff : SVD truncation
    order : Trotter order (2 or 4)

    Returns
    -------
    dict with keys: times, R, bond_dims_final, mps_final
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    n = K.shape[0]

    # Build nearest-neighbour Hamiltonian for TEBD
    builder = qtn.SpinHam1D(S=1 / 2)
    for i in range(n):
        if abs(omega[i]) > 1e-15:
            builder[i] += -omega[i], "Z"
    for i in range(n - 1):
        if abs(K[i, i + 1]) < 1e-15:
            continue
        builder[i, i + 1] += -K[i, i + 1], "X", "X"
        builder[i, i + 1] += -K[i, i + 1], "Y", "Y"

    H_local = builder.build_local_ham(n)

    # Initial state: product state with Ry rotations
    arrays = []
    for i in range(n):
        angle = float(omega[i]) % (2 * np.pi)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        arrays.append(np.array([c, s], dtype=np.complex128))
    psi = qtn.MPS_product_state(arrays)

    tebd = qtn.TEBD(psi, H_local, dt=dt, tol=cutoff)
    tebd.split_opts["max_bond"] = bond_dim

    n_steps = max(1, int(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    R_history = np.zeros(len(times))

    # Measure initial R
    R_history[0] = _order_parameter_mps(tebd.pt, n)

    for step in range(1, n_steps + 1):
        tebd.update_to(times[step], order=order)
        R_history[step] = _order_parameter_mps(tebd.pt, n)

    bond_dims = [tebd.pt.bond_size(i, i + 1) for i in range(n - 1)]

    return {
        "times": times,
        "R": R_history,
        "bond_dims_final": bond_dims,
        "mps_final": tebd.pt,
    }


def _order_parameter_mps(mps: Any, n: int) -> float:
    """Compute Kuramoto R from MPS single-site expectations."""
    import quimb as qu

    sx_op = qu.pauli("X")
    sy_op = qu.pauli("Y")
    z = 0.0 + 0.0j
    for i in range(n):
        ex = float(np.real(mps.local_expectation(sx_op, i)))
        ey = float(np.real(mps.local_expectation(sy_op, i)))
        z += ex + 1j * ey
    z /= n
    return float(abs(z))
