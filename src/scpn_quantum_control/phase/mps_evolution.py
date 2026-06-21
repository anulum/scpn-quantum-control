# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MPS/DMRG Backend via quimb
"""Matrix Product State backend for large-N Kuramoto-XY.

Uses quimb's DMRG for ground state search and TEBD for time evolution,
enabling simulation of n=32-64 oscillators (beyond ED limits).

Requires: pip install quimb
Reference: Gray, JOSS 3(29), 819 (2018).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import quimb
    import quimb.tensor as qtn

    _QUIMB_AVAILABLE = True
except Exception:
    _QUIMB_AVAILABLE = False
    quimb = None
    qtn = None


def is_quimb_available() -> bool:
    """Check if quimb is installed."""
    return _QUIMB_AVAILABLE


def _long_range_coupling_l1(K: NDArray[np.float64], *, tol: float = 1e-15) -> float:
    """Return upper-triangle L1 weight of couplings outside adjacent bonds."""
    n = K.shape[0]
    omitted = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            if abs(K[i, j]) > tol:
                omitted += abs(float(K[i, j]))
    return omitted


def _nearest_neighbour_scope(
    K: NDArray[np.float64], allow_long_range_truncation: bool
) -> tuple[str, float]:
    """Validate or label the nearest-neighbour quimb Hamiltonian scope."""
    omitted_l1 = _long_range_coupling_l1(K)
    if omitted_l1 > 0.0 and not allow_long_range_truncation:
        raise ValueError(
            "MPS quimb SpinHam1D/TEBD path only represents nearest-neighbour couplings; "
            "pass allow_long_range_truncation=True to explicitly run the truncated "
            f"nearest-neighbour model (omitted long-range L1={omitted_l1:.6g})."
        )
    scope = "nearest_neighbour_truncated" if omitted_l1 > 0.0 else "nearest_neighbour"
    return scope, omitted_l1


def _build_mpo_hamiltonian(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    allow_long_range_truncation: bool = False,
) -> Any:
    """Build the XY Hamiltonian as a quimb SpinHam1D MPO.

    H = -Σ K_ij (X_iX_j + Y_iY_j) - Σ ω_i Z_i
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    _nearest_neighbour_scope(K, allow_long_range_truncation)
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

    return builder.build_mpo(n)


def dmrg_ground_state(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    bond_dim: int = 64,
    cutoff: float = 1e-10,
    max_sweeps: int = 20,
    allow_long_range_truncation: bool = False,
) -> dict[str, Any]:
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
    allow_long_range_truncation : bool
        Explicitly allow the quimb SpinHam1D nearest-neighbour MPO to
        omit non-adjacent K[i, j] couplings. Defaults to False so full
        K_nm inputs cannot be truncated silently.

    Returns
    -------
    dict with keys: energy, mps, converged, bond_dims
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    n = K.shape[0]
    coupling_scope, omitted_l1 = _nearest_neighbour_scope(K, allow_long_range_truncation)
    H_mpo = _build_mpo_hamiltonian(
        K,
        omega,
        allow_long_range_truncation=allow_long_range_truncation,
    )

    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dim, cutoffs=cutoff)
    converged = False
    last_energy = 0.0
    for _sweep in range(max_sweeps):
        dmrg.sweep_right()
        e_left = dmrg.sweep_left()
        current_energy = float(np.real(e_left))
        if abs(current_energy - last_energy) < 1e-10:
            converged = True
            break
        last_energy = current_energy

    mps = dmrg.state
    bond_dims_out = [mps.bond_size(i, i + 1) for i in range(n - 1)]

    return {
        "energy": last_energy,
        "mps": mps,
        "converged": converged,
        "bond_dims": bond_dims_out,
        "n_oscillators": n,
        "coupling_scope": coupling_scope,
        "omitted_coupling_l1": omitted_l1,
    }


def tebd_evolution(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float = 1.0,
    dt: float = 0.05,
    bond_dim: int = 64,
    cutoff: float = 1e-10,
    order: int = 2,
    allow_long_range_truncation: bool = False,
) -> dict[str, Any]:
    """Time evolution via TEBD (Time-Evolving Block Decimation).

    Parameters
    ----------
    K, omega : coupling and frequencies
    t_max : total evolution time
    dt : Trotter step size
    bond_dim : maximum bond dimension
    cutoff : SVD truncation
    order : Trotter order (2 or 4)
    allow_long_range_truncation : bool
        Explicitly allow the nearest-neighbour TEBD local Hamiltonian to
        omit non-adjacent K[i, j] couplings. Defaults to False so full
        K_nm inputs cannot be truncated silently.

    Notes
    -----
    Disconnected nearest-neighbour edges are represented as explicit
    zero local terms because quimb's TEBD sweeps request a local
    Hamiltonian for every adjacent bond.

    Returns
    -------
    dict with keys: times, R, bond_dims_final, mps_final
    """
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    n = K.shape[0]
    coupling_scope, omitted_l1 = _nearest_neighbour_scope(K, allow_long_range_truncation)

    # Build nearest-neighbour Hamiltonian for TEBD
    builder = qtn.SpinHam1D(S=1 / 2)
    for i in range(n):
        if abs(omega[i]) > 1e-15:
            builder[i] += -omega[i], "Z"
    for i in range(n - 1):
        if abs(K[i, i + 1]) < 1e-15:
            # TEBD sweeps ask LocalHam1D for every nearest-neighbour bond.
            # Register an explicit zero term so disconnected chains evolve.
            builder[i, i + 1] += 0.0, "I", "I"
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

    tebd = qtn.TEBD(psi, H_local, dt=dt)
    tebd.split_opts["max_bond"] = bond_dim
    tebd.split_opts["cutoff"] = cutoff

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
        "coupling_scope": coupling_scope,
        "omitted_coupling_l1": omitted_l1,
    }


def _order_parameter_mps(mps: Any, n: int) -> float:
    """Compute Kuramoto R from MPS single-site expectations."""
    import quimb as qu

    sx_dense = qu.pauli("X").toarray()
    sy_dense = qu.pauli("Y").toarray()
    z = 0.0 + 0.0j
    for i in range(n):
        ex = float(np.real(mps.compute_local_expectation_canonical({(i,): sx_dense})))
        ey = float(np.real(mps.compute_local_expectation_canonical({(i,): sy_dense})))
        z += ex + 1j * ey
    z /= n
    return float(abs(z))
