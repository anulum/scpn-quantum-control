# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Documented Classical Baselines
"""Documented classical baselines for Kuramoto-XY workflows.

The functions here are deliberately small provenance surfaces:

* SciPy ODE integrates the classical Kuramoto phase equations.
* QuTiP Lindblad uses an independent density-matrix open-system solver.
* quimb TEBD reuses the project MPS backend when the tensor extra is installed.
"""

from __future__ import annotations

import importlib
import importlib.util as importlib_util
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from scpn_quantum_control.phase import mps_evolution


@dataclass
class ClassicalBaselineRun:
    """Result envelope for one classical baseline run."""

    name: str
    backend: str
    n_oscillators: int
    available: bool
    elapsed_ms: float
    times: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    order_parameter: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None

    @property
    def r_final(self) -> float | None:
        """Final Kuramoto order parameter when the run is available."""
        if self.order_parameter.size == 0:
            return None
        return float(self.order_parameter[-1])


def available_baselines() -> dict[str, bool]:
    """Return which documented classical baselines are available now."""
    return {
        "scipy_ode": True,
        "qutip_lindblad": importlib_util.find_spec("qutip") is not None,
        "mps_tebd": mps_evolution.is_quimb_available(),
    }


def scipy_ode_baseline(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    t_max: float = 1.0,
    dt: float = 0.05,
    theta0: NDArray[np.float64] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> ClassicalBaselineRun:
    """Integrate the classical Kuramoto ODE with SciPy ``solve_ivp``.

    The implemented equation is

    ``d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i)``.
    """
    K_arr, omega_arr = _validate_inputs(K, omega)
    _validate_time_grid(t_max, dt)
    if theta0 is None:
        theta_initial = np.mod(omega_arr, 2.0 * np.pi)
    else:
        theta_initial = np.asarray(theta0, dtype=float)
        if theta_initial.shape != omega_arr.shape:
            raise ValueError(
                f"theta0 must have shape {omega_arr.shape}, got {theta_initial.shape}"
            )
        if not np.all(np.isfinite(theta_initial)):
            raise ValueError("theta0 must contain only finite values")

    times = _times(t_max, dt)

    def rhs(_t: float, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        phase_delta = theta[None, :] - theta[:, None]
        coupling = np.sum(K_arr * np.sin(phase_delta), axis=1)
        return np.asarray(omega_arr + coupling, dtype=np.float64)

    start = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, float(t_max)),
        theta_initial,
        t_eval=times,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not solution.success:
        raise RuntimeError(f"SciPy Kuramoto ODE integration failed: {solution.message}")

    theta_history = solution.y.T
    order_parameter = np.array([_phase_order_parameter(theta) for theta in theta_history])
    return ClassicalBaselineRun(
        name="scipy_ode",
        backend="scipy.solve_ivp(RK45)",
        n_oscillators=K_arr.shape[0],
        available=True,
        elapsed_ms=elapsed_ms,
        times=solution.t,
        order_parameter=order_parameter,
        metadata={
            "equation": "dtheta_i=omega_i+sum_j K_ij sin(theta_j-theta_i)",
            "rtol": rtol,
            "atol": atol,
            "theta_final": theta_history[-1].tolist(),
        },
    )


def qutip_lindblad_baseline(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    gamma: float = 0.05,
    t_max: float = 0.5,
    dt: float = 0.1,
) -> ClassicalBaselineRun:
    """Run an optional QuTiP Lindblad density-matrix baseline.

    If QuTiP is not installed, the returned result is marked unavailable
    instead of fabricating a numerical value.
    """
    K_arr, omega_arr = _validate_inputs(K, omega)
    _validate_time_grid(t_max, dt)
    if gamma < 0.0 or not np.isfinite(gamma):
        raise ValueError(f"gamma must be finite and non-negative, got {gamma}")

    qutip_spec = importlib_util.find_spec("qutip")
    if qutip_spec is None:
        return _unavailable("qutip_lindblad", "qutip.mesolve", K_arr.shape[0], "qutip missing")

    qutip = importlib.import_module("qutip")
    times = _times(t_max, dt)
    start = time.perf_counter()
    hamiltonian = _qutip_xy_hamiltonian(qutip, K_arr, omega_arr)
    psi0 = qutip.tensor([qutip.basis(2, 0)] * K_arr.shape[0])
    collapse_ops = []
    if gamma > 0.0:
        collapse_ops = [
            np.sqrt(gamma) * _qutip_single(qutip, K_arr.shape[0], site, qutip.sigmam())
            for site in range(K_arr.shape[0])
        ]
    # QuTiP 5.3 makes ``e_ops`` keyword-only (mesolve takes <=4 positional args);
    # pass it by name so the call resolves on both 5.2 and 5.3.
    result = qutip.mesolve(hamiltonian, psi0, times, collapse_ops, e_ops=[])
    order_parameter = np.array(
        [_qutip_order_parameter(qutip, state, K_arr.shape[0]) for state in result.states]
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    final_state = result.states[-1]
    final_trace = _qutip_trace_or_norm(final_state)
    return ClassicalBaselineRun(
        name="qutip_lindblad",
        backend="qutip.mesolve",
        n_oscillators=K_arr.shape[0],
        available=True,
        elapsed_ms=elapsed_ms,
        times=np.asarray(result.times, dtype=float),
        order_parameter=order_parameter,
        metadata={"gamma": gamma, "final_trace": final_trace},
    )


def mps_tebd_baseline(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    t_max: float = 0.5,
    dt: float = 0.1,
    bond_dim: int = 32,
    cutoff: float = 1e-10,
) -> ClassicalBaselineRun:
    """Run an optional quimb TEBD tensor-network baseline."""
    K_arr, omega_arr = _validate_inputs(K, omega)
    _validate_time_grid(t_max, dt)
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be positive, got {bond_dim}")
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError(f"cutoff must be finite and positive, got {cutoff}")
    if not mps_evolution.is_quimb_available():
        return _unavailable("mps_tebd", "quimb.TEBD", K_arr.shape[0], "quimb missing")

    start = time.perf_counter()
    result = mps_evolution.tebd_evolution(
        K_arr,
        omega_arr,
        t_max=t_max,
        dt=dt,
        bond_dim=bond_dim,
        cutoff=cutoff,
        allow_long_range_truncation=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ClassicalBaselineRun(
        name="mps_tebd",
        backend="quimb.TEBD",
        n_oscillators=K_arr.shape[0],
        available=True,
        elapsed_ms=elapsed_ms,
        times=np.asarray(result["times"], dtype=float),
        order_parameter=np.asarray(result["R"], dtype=float),
        metadata={
            "bond_dim": bond_dim,
            "cutoff": cutoff,
            "bond_dims_final": list(result["bond_dims_final"]),
            "coupling_scope": result["coupling_scope"],
            "omitted_coupling_l1": result["omitted_coupling_l1"],
        },
    )


def run_documented_classical_baselines(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    t_max: float = 0.5,
    dt: float = 0.1,
    include_optional: bool = True,
) -> dict[str, ClassicalBaselineRun]:
    """Run the documented baseline suite for one Kuramoto problem."""
    runs = {"scipy_ode": scipy_ode_baseline(K, omega, t_max=t_max, dt=dt)}
    if include_optional:
        runs["qutip_lindblad"] = qutip_lindblad_baseline(K, omega, t_max=t_max, dt=dt)
        runs["mps_tebd"] = mps_tebd_baseline(K, omega, t_max=t_max, dt=dt)
    return runs


def _validate_inputs(
    K: NDArray[np.float64], omega: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    K_arr = np.asarray(K, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {K_arr.shape}")
    if omega_arr.shape != (K_arr.shape[0],):
        raise ValueError(f"omega must have shape {(K_arr.shape[0],)}, got {omega_arr.shape}")
    if not np.all(np.isfinite(K_arr)):
        raise ValueError("K must contain only finite values")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega must contain only finite values")
    return K_arr, omega_arr


def _validate_time_grid(t_max: float, dt: float) -> None:
    if t_max < 0.0 or not np.isfinite(t_max):
        raise ValueError(f"t_max must be finite and non-negative, got {t_max}")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be finite and positive, got {dt}")


def _times(t_max: float, dt: float) -> NDArray[np.float64]:
    n_steps = max(1, round(t_max / dt))
    return np.linspace(0.0, float(t_max), n_steps + 1, dtype=np.float64)


def _phase_order_parameter(theta: NDArray[np.float64]) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


def _unavailable(name: str, backend: str, n_oscillators: int, reason: str) -> ClassicalBaselineRun:
    return ClassicalBaselineRun(
        name=name,
        backend=backend,
        n_oscillators=n_oscillators,
        available=False,
        elapsed_ms=0.0,
        unavailable_reason=reason,
    )


def _qutip_single(qutip: Any, n: int, site: int, op: Any) -> Any:
    ops = [qutip.qeye(2)] * n
    ops[site] = op
    return qutip.tensor(list(reversed(ops)))


def _qutip_pair(qutip: Any, n: int, first: int, second: int, op1: Any, op2: Any) -> Any:
    ops = [qutip.qeye(2)] * n
    ops[first] = op1
    ops[second] = op2
    return qutip.tensor(list(reversed(ops)))


def _qutip_xy_hamiltonian(qutip: Any, K: NDArray[np.float64], omega: NDArray[np.float64]) -> Any:
    n = K.shape[0]
    hamiltonian = qutip.Qobj(np.zeros((2**n, 2**n), dtype=complex), dims=[[2] * n, [2] * n])
    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    for site in range(n):
        if abs(omega[site]) > 1e-15:
            hamiltonian = hamiltonian - omega[site] * _qutip_single(qutip, n, site, sz)
    for first in range(n):
        for second in range(first + 1, n):
            if abs(K[first, second]) < 1e-15:
                continue
            hamiltonian = hamiltonian - K[first, second] * _qutip_pair(
                qutip, n, first, second, sx, sx
            )
            hamiltonian = hamiltonian - K[first, second] * _qutip_pair(
                qutip, n, first, second, sy, sy
            )
    return hamiltonian


def _qutip_order_parameter(qutip: Any, state: Any, n: int) -> float:
    z = 0.0 + 0.0j
    for site in range(n):
        ex = qutip.expect(_qutip_single(qutip, n, site, qutip.sigmax()), state)
        ey = qutip.expect(_qutip_single(qutip, n, site, qutip.sigmay()), state)
        z += complex(ex) + 1j * complex(ey)
    return float(abs(z / n))


def _qutip_trace_or_norm(state: Any) -> float:
    if getattr(state, "isoper", False):
        return float(np.real(state.tr()))
    return float(np.real(state.norm() ** 2))
