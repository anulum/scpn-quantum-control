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

from contextvars import ContextVar
from math import isfinite
from sys import float_info
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .mps_diagnostics import (
    DiscardedWeightLog,
    MPSAllocationError,
    MPSObservation,
    MPSRunDiagnostics,
    require_mps_allocation,
    validate_mps_model,
)

try:
    import quimb
    import quimb.tensor as qtn

    _QUIMB_AVAILABLE = True
except Exception:
    _QUIMB_AVAILABLE = False
    quimb = None
    qtn = None

_REPORTING_SPLIT_METHOD = "scpn_quantum_control_svd_reporting_discarded_norm"
_SUZUKI_FOURTH_ORDER_WEIGHT = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
_SUZUKI_FOURTH_ORDER_WEIGHTS = (
    _SUZUKI_FOURTH_ORDER_WEIGHT,
    _SUZUKI_FOURTH_ORDER_WEIGHT,
    1.0 - 4.0 * _SUZUKI_FOURTH_ORDER_WEIGHT,
    _SUZUKI_FOURTH_ORDER_WEIGHT,
    _SUZUKI_FOURTH_ORDER_WEIGHT,
)
_active_discarded_log: ContextVar[DiscardedWeightLog | None] = ContextVar(
    "scpn_quantum_control_mps_discarded_log", default=None
)
_reporting_split_registered = False


def is_quimb_available() -> bool:
    """Check if quimb is installed."""
    return _QUIMB_AVAILABLE


def _reporting_split_method() -> str:
    """Register the reporting SVD split with Quimb once and return its name.

    Quimb's gate application keeps the ``info`` keyword for its own use and
    does not hand it to the decomposition, so a report requested through
    ``TEBD.split_opts["info"]`` is never written. The registered method calls
    Quimb's own truncated SVD with the same arguments and adds only the report
    target of the active run. The factors it returns are those of Quimb's
    ``"svd"`` method.

    Returns
    -------
    str
        Method name accepted by ``TEBD.split_opts["method"]``.

    """
    global _reporting_split_registered
    if not _reporting_split_registered:
        from quimb.tensor import decomp

        def split(
            x: Any,
            cutoff: float = -1.0,
            cutoff_mode: Any = decomp.cutoff_mode_rsum2,
            max_bond: int = -1,
            absorb: Any = decomp.get_Usq_sqVH,
            renorm: int = 0,
        ) -> Any:
            """Run Quimb's truncated SVD and report its discarded norm.

            Parameters
            ----------
            x
                Matrix handed over by Quimb's split dispatch.
            cutoff
                Singular-value cutoff of the dispatch.
            cutoff_mode
                Cutoff mode code of the dispatch.
            max_bond
                Maximum number of retained singular values.
            absorb
                Absorption code of the dispatch.
            renorm
                Renormalization code of the dispatch.

            Returns
            -------
            Any
                The factors returned by Quimb's truncated SVD.

            """
            return decomp.svd_truncated(
                x,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                max_bond=max_bond,
                absorb=absorb,
                renorm=renorm,
                info=_active_discarded_log.get(),
            )

        decomp.register_split_driver(_REPORTING_SPLIT_METHOD)(split)
        _reporting_split_registered = True
    return _REPORTING_SPLIT_METHOD


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

    H = -Σ K_ij (Sx_i Sx_j + Sy_i Sy_j) - Σ ω_i Sz_i,
    where quimb ``SpinHam1D(S=1/2)`` uses Sα = σα/2. In Pauli notation,
    pair coefficients are K_ij/4 and on-site coefficients are ω_i/2.
    This convention differs from Pauli-normalised XY routes elsewhere.
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
    *,
    max_memory_bytes: int | None = None,
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
    max_memory_bytes : int | None
        Optional byte ceiling for declared compressed solver buffers. It can
        only lower the active allocation policy; this is not an RSS guarantee.

    Returns
    -------
    dict[str, Any]
        Ground-state energy, MPS, convergence, bond dimensions, oscillator
        count, and the explicit coupling-scope diagnostics.

    Raises
    ------
    ValueError
        The model, cutoff, sweep count or coupling-scope consent is invalid.
    MemoryError
        Declared compressed buffers exceed the admitted memory ceiling.
    ImportError
        Quimb is unavailable; no dense substitute is executed.

    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square with at least two sites")
    allocation = require_mps_allocation(
        int(K.shape[0]), bond_dim, max_memory_bytes=max_memory_bytes
    )
    n = validate_mps_model(K, omega)
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if isinstance(cutoff, bool) or not isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    if not isinstance(allow_long_range_truncation, bool):
        raise ValueError("allow_long_range_truncation must be an explicit boolean")
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

    coupling_scope, omitted_l1 = _nearest_neighbour_scope(K, allow_long_range_truncation)
    H_mpo = _build_mpo_hamiltonian(
        K,
        omega,
        allow_long_range_truncation=allow_long_range_truncation,
    )

    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dim, cutoffs=cutoff)
    converged = False
    last_energy = 0.0
    # Two sites form one two-site block: a right sweep optimizes the whole
    # system, and the pinned runtime's left sweep fails on that single block.
    single_block = n == 2
    for _sweep in range(max_sweeps):
        e_sweep = dmrg.sweep_right()
        if not single_block:
            e_sweep = dmrg.sweep_left()
        current_energy = float(np.real(e_sweep))
        energy_change = abs(current_energy - last_energy)
        last_energy = current_energy
        if energy_change < 1e-10:
            converged = True
            break

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
        "diagnostics": MPSRunDiagnostics(
            allocation=allocation,
            contraction_strategy=(
                "two-site DMRG right sweeps over the single two-site block"
                if single_block
                else "two-site DMRG right/left sweeps"
            ),
            observations=(_observe_mps(mps, n, 0.0),),
            discarded_weight=None,
            discarded_weight_scope="DMRG2 does not report local discarded weights here",
            truncation_events=0,
            trotter_error_estimate=None,
            cutoff=cutoff,
            trotter_order=None,
            time_step=None,
            backend_version=str(quimb.__version__),
        ),
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
    *,
    initial_state: NDArray[np.complex128] | None = None,
    max_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Time evolution via TEBD (Time-Evolving Block Decimation).

    Parameters
    ----------
    K
        Coupling matrix.
    omega
        Natural frequencies.
    t_max
        Total evolution time.
    dt
        Trotter step size.
    bond_dim
        Maximum MPS bond dimension.
    cutoff
        SVD truncation cutoff.
    order
        Trotter order, either two or four. Order two uses the runtime's own
        second-order step. Order four composes five of those steps with the
        Suzuki weights ``p, p, 1 - 4p, p, p`` for ``p = 1 / (4 - 4**(1/3))``.
    allow_long_range_truncation
        Explicitly allow the nearest-neighbour TEBD local Hamiltonian to
        omit non-adjacent K[i, j] couplings. Defaults to False so full
        K_nm inputs cannot be truncated silently.
    initial_state
        Optional normalized dense pure state in MPS site order. If omitted,
        retain the legacy product state prepared from ``omega``. The dense
        input is admitted by the active allocation budget and must be exactly
        representable under ``bond_dim`` and ``cutoff``; truncation refuses.
        With a zero cutoff, singular values that are exactly zero are not
        stored, so a product state starts at bond dimension one.
    max_memory_bytes
        Optional byte ceiling for declared tensor, local SVD and observation
        buffers. The active allocation policy remains an upper ceiling.

    Returns
    -------
    dict[str, Any]
        Times, order-parameter history, final bond dimensions and MPS, and the
        explicit coupling-scope diagnostics.

    Raises
    ------
    ValueError
        The model, time grid, cutoff, order, consent or preparation is invalid.
    MemoryError
        Declared compressed buffers or explicit dense preparation exceed policy.
    ImportError
        Quimb is unavailable; no dense substitute is executed.

    Notes
    -----
    Disconnected nearest-neighbour edges are represented as explicit
    zero local terms because quimb's TEBD sweeps request a local
    Hamiltonian for every adjacent bond.

    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square with at least two sites")
    if isinstance(t_max, bool) or not isfinite(t_max) or t_max < 0:
        raise ValueError("t_max must be finite and nonnegative")
    if isinstance(dt, bool) or not isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not isfinite(t_max / dt):
        raise ValueError("The requested time grid is not finite")
    if isinstance(order, bool) or not isinstance(order, int) or order not in (2, 4):
        raise ValueError("order must be two or four")
    if isinstance(cutoff, bool) or not isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    if not isinstance(allow_long_range_truncation, bool):
        raise ValueError("allow_long_range_truncation must be an explicit boolean")
    n_steps = 0 if t_max == 0 else max(1, int(t_max / dt))
    allocation = require_mps_allocation(
        int(K.shape[0]),
        bond_dim,
        observations=n_steps + 1,
        max_memory_bytes=max_memory_bytes,
    )
    n = validate_mps_model(K, omega)
    if not _QUIMB_AVAILABLE:
        raise ImportError("quimb not installed: pip install quimb")

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

    if initial_state is None:
        # Legacy initial state: product of Ry-rotated qubits.
        arrays = []
        for i in range(n):
            angle = float(omega[i]) % (2 * np.pi)
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            arrays.append(np.array([c, s], dtype=np.complex128))
        psi = qtn.MPS_product_state(arrays)
    else:
        from ..dense_budget import require_dense_allocation

        remaining_bytes = allocation.budget_bytes - allocation.workspace_bytes
        if remaining_bytes <= 0:
            raise MPSAllocationError("MPS dense preparation exceeds the remaining memory cap")
        require_dense_allocation(
            n,
            dtype=np.complex128,
            rank=1,
            object_count=2,
            max_gib=remaining_bytes / 1024**3,
            label="TEBD dense initial state",
        )
        dense = np.asarray(initial_state, dtype=np.complex128)
        if dense.ndim != 1 or dense.shape != (2**n,):
            raise ValueError("initial_state must have shape (2**n,) in MPS site order.")
        if not np.all(np.isfinite(dense)):
            raise ValueError("initial_state must contain only finite amplitudes.")
        if abs(float(np.linalg.norm(dense)) - 1.0) > 1e-10:
            raise ValueError("initial_state must be normalized.")
        if cutoff > 0:
            psi = qtn.MatrixProductState.from_dense(
                dense, dims=2, max_bond=bond_dim, cutoff=cutoff
            )
        else:
            # A zero cutoff keeps every singular value up to the bond limit,
            # including exact zeros. Those are not stored.
            psi = qtn.MatrixProductState.from_dense(
                dense,
                dims=2,
                max_bond=bond_dim,
                cutoff=float_info.min,
                cutoff_mode="abs",
            )
        represented = np.asarray(psi.to_dense()).reshape(-1)
        if float(np.linalg.norm(represented - dense)) > 1e-10:
            raise ValueError(
                "initial_state cannot be represented exactly with bond_dim and cutoff."
            )

    tebd = qtn.TEBD(psi, H_local, dt=dt)
    tebd.split_opts["max_bond"] = bond_dim
    tebd.split_opts["cutoff"] = cutoff
    tebd.split_opts["method"] = _reporting_split_method()
    discarded = DiscardedWeightLog()

    times = np.linspace(0, t_max, n_steps + 1)
    R_history = np.zeros(len(times))

    # Measure initial R
    R_history[0] = _order_parameter_mps(tebd.pt, n)
    observations = [_observe_mps(tebd.pt, n, 0.0)]

    active_log = _active_discarded_log.set(discarded)
    try:
        for step in range(1, n_steps + 1):
            if order == 2:
                tebd.update_to(times[step], order=2)
            else:
                # The pinned runtime's own fourth-order step converges at
                # second order, so its second-order steps are composed here.
                step_dt = float(times[step] - times[step - 1])
                for weight in _SUZUKI_FOURTH_ORDER_WEIGHTS:
                    tebd.step(order=2, dt=weight * step_dt)
            R_history[step] = _order_parameter_mps(tebd.pt, n)
            observations.append(_observe_mps(tebd.pt, n, float(times[step])))
    finally:
        _active_discarded_log.reset(active_log)
    if order == 2:
        trotter_error_estimate = float(tebd.err)
    else:
        grid_dt = t_max / n_steps if n_steps else 0.0
        trotter_error_estimate = float(n_steps * H_local.mean_norm() * grid_dt**5)

    bond_dims = [tebd.pt.bond_size(i, i + 1) for i in range(n - 1)]

    return {
        "times": times,
        "R": R_history,
        "bond_dims_final": bond_dims,
        "mps_final": tebd.pt,
        "coupling_scope": coupling_scope,
        "omitted_coupling_l1": omitted_l1,
        "diagnostics": MPSRunDiagnostics(
            allocation=allocation,
            contraction_strategy="canonical nearest-neighbour TEBD sweeps",
            observations=tuple(observations),
            discarded_weight=discarded.weight,
            discarded_weight_scope=(
                "Sum of squared local TEBD discarded singular-value norms; "
                "not a global state, observable or fidelity error bound"
            ),
            truncation_events=discarded.events,
            trotter_error_estimate=trotter_error_estimate,
            cutoff=cutoff,
            trotter_order=order,
            time_step=dt,
            backend_version=str(quimb.__version__),
        ),
    }


def _observe_mps(mps: Any, n: int, time: float) -> MPSObservation:
    """Measure norm, tensor storage and normalized local Z without dense export."""
    import quimb as qu

    z = qu.pauli("Z").toarray()
    local_z = tuple(
        float(
            np.real(
                mps.compute_local_expectation_canonical(
                    {(site,): z},
                    normalized=True,
                )
            )
        )
        for site in range(n)
    )
    return MPSObservation(
        time=time,
        bond_dims=tuple(int(mps.bond_size(site, site + 1)) for site in range(n - 1)),
        stored_bytes=sum(int(tensor.data.nbytes) for tensor in mps.tensors),
        norm=float(np.real(mps.norm())),
        local_z=local_z,
    )


def export_mps_state(mps: Any, *, max_memory_bytes: int | None = None) -> NDArray[np.complex128]:
    """Explicitly export an MPS only after admitting its dense output buffers.

    Parameters
    ----------
    mps
        Actual Quimb matrix-product state; its site order is retained.
    max_memory_bytes
        Optional positive byte ceiling that only lowers the active policy.

    Returns
    -------
    NDArray[np.complex128]
        Flat dense amplitudes in MPS site order, never a silent fallback.

    Raises
    ------
    ValueError
        The explicit caller ceiling is invalid.
    MemoryError
        Two declared dense state buffers exceed allocation policy.

    """
    from ..dense_budget import dense_budget_bytes, require_dense_allocation

    budget = dense_budget_bytes()
    if max_memory_bytes is not None:
        if (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes < 1
        ):
            raise ValueError("max_memory_bytes must be a positive integer")
        budget = min(budget, max_memory_bytes)
    require_dense_allocation(
        int(mps.L),
        dtype=np.complex128,
        rank=1,
        object_count=2,
        max_gib=budget / 1024**3,
        label="MPS explicit dense state export",
    )
    return np.asarray(mps.to_dense(), dtype=np.complex128).reshape(-1)


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
