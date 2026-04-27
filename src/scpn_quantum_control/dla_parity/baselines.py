# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity — classical baselines
"""Classical (noiseless) leakage reference for the DLA-parity protocol.

The DLA-parity Hamiltonian used in the published campaign is

    H = Σ_i ω_i Z_i + Σ_{i,i+1} K_{i,i+1} (X_i X_{i+1} + Y_i Y_{i+1})

with ω = linspace(0.8, 1.2, n), K_{ij} = 0.45 * exp(-0.3 |i - j|),
only nearest-neighbour XY coupling in the circuit, Lie-Trotter
decomposition at step t_step = 0.3, initial states |0011⟩ (even
sector) and |0001⟩ (odd sector).

Because each Trotter term commutes with the total-parity operator
(Z terms are diagonal in the computational basis; X_i X_{i+1} +
Y_i Y_{i+1} flips adjacent bits in pairs), the noiseless leakage
rate is identically zero at every depth in both sectors. This
module recomputes that reference curve from first principles so
any deviation on hardware is unambiguously hardware-origin.

Public entry points

* :func:`available_baselines` — which optional backends are
  importable in the current environment.
* :func:`compute_classical_leakage_reference` — build H, evolve
  both sectors across a depth sweep, return per-(depth, sector)
  classical leakage. Backends: ``"numpy"`` (always present),
  ``"qutip"`` (optional). ``"auto"`` picks qutip if available,
  else numpy.

No Rust tier: the reference curve for n=4 across the published
depth sweep takes a few milliseconds on numpy; a Rust path would
add maintenance burden with no measurable gain.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm

Backend = Literal["auto", "numpy", "qutip"]

DEFAULT_DEPTHS: tuple[int, ...] = (2, 4, 6, 8, 10, 14, 20, 30)
DEFAULT_T_STEP: float = 0.3
DEFAULT_INITIAL_EVEN: str = "0011"
DEFAULT_INITIAL_ODD: str = "0001"
DEFAULT_N_QUBITS: int = 4

# The published campaign used a zero-tolerance classical reference:
# |leakage| < 1e-10 on ideal evolution. Any deviation at this scale
# would indicate a Hamiltonian-builder bug, not a physics effect.
CLASSICAL_LEAKAGE_THRESHOLD: float = 1e-10


@dataclass(frozen=True, slots=True)
class ClassicalLeakagePoint:
    """Classical leakage for one (depth, sector) point."""

    depth: int
    sector: Literal["even", "odd"]
    initial: str
    leakage: float


@dataclass(frozen=True, slots=True)
class ClassicalLeakageReference:
    """Full classical reference — the noiseless per-depth leakage."""

    backend: Literal["numpy", "qutip"]
    n_qubits: int
    t_step: float
    depths: tuple[int, ...]
    points: tuple[ClassicalLeakagePoint, ...]

    @property
    def max_abs_leakage(self) -> float:
        return max((abs(p.leakage) for p in self.points), default=0.0)

    @property
    def is_zero_within_tolerance(self) -> bool:
        return self.max_abs_leakage < CLASSICAL_LEAKAGE_THRESHOLD


def _have(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def available_baselines() -> dict[str, bool]:
    """Which optional classical backends are importable right now.

    Always reports ``"numpy"`` present — the numpy backend is the
    mandatory floor. The others are opt-in.
    """
    return {
        "numpy": True,
        "qutip": _have("qutip"),
    }


def _build_k_matrix(n: int) -> np.ndarray:
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i, j] = 0.45 * math.exp(-0.3 * abs(i - j))
    return k


def _build_omega(n: int) -> np.ndarray:
    return np.linspace(0.8, 1.2, n)


def _computational_basis_state(bitstring: str) -> np.ndarray:
    idx = int(bitstring, 2)
    dim = 1 << len(bitstring)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def _pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return X, Y, Z


def _single_site(op: np.ndarray, i: int, n: int) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for k in range(n):
        m = np.kron(m, op if k == i else np.eye(2, dtype=np.complex128))
    return m


def _two_site(op1: np.ndarray, op2: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for k in range(n):
        if k == i:
            m = np.kron(m, op1)
        elif k == j:
            m = np.kron(m, op2)
        else:
            m = np.kron(m, np.eye(2, dtype=np.complex128))
    return m


def _build_hz(n: int, omega: np.ndarray) -> np.ndarray:
    X, Y, Z = _pauli_matrices()
    del X, Y
    dim = 1 << n
    hz = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        hz += float(omega[i]) * _single_site(Z, i, n)
    return hz


def _build_hxy_nn(n: int, k_matrix: np.ndarray) -> np.ndarray:
    X, Y, _ = _pauli_matrices()
    dim = 1 << n
    hxy = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        j = i + 1
        coeff = float(k_matrix[i, j])
        hxy += coeff * _two_site(X, X, i, j, n)
        hxy += coeff * _two_site(Y, Y, i, j, n)
    return hxy


def _parity_mask(n: int, initial_parity: int) -> np.ndarray:
    dim = 1 << n
    mask = np.zeros(dim, dtype=np.float64)
    for idx in range(dim):
        if bin(idx).count("1") % 2 != initial_parity:
            mask[idx] = 1.0
    return mask


def _evolve_numpy(
    psi0: np.ndarray,
    hz: np.ndarray,
    hxy: np.ndarray,
    t_step: float,
    n_steps: int,
) -> np.ndarray:
    uz = expm(-1j * hz * t_step)
    uxy = expm(-1j * hxy * t_step)
    psi = psi0
    for _ in range(n_steps):
        psi = uz @ psi
        psi = uxy @ psi
    return psi


def _leakage_numpy(
    n: int,
    initial: str,
    depths: Sequence[int],
    t_step: float,
    omega: np.ndarray,
    k_matrix: np.ndarray,
) -> dict[int, float]:
    psi0 = _computational_basis_state(initial)
    hz = _build_hz(n, omega)
    hxy = _build_hxy_nn(n, k_matrix)
    mask = _parity_mask(n, bin(int(initial, 2)).count("1") % 2)
    leakages: dict[int, float] = {}
    for d in depths:
        psi_d = _evolve_numpy(psi0, hz, hxy, t_step, d)
        probs = np.abs(psi_d) ** 2
        leakages[d] = float(np.dot(mask, probs))
    return leakages


def _leakage_qutip(
    n: int,
    initial: str,
    depths: Sequence[int],
    t_step: float,
    omega: np.ndarray,
    k_matrix: np.ndarray,
) -> dict[int, float]:
    import qutip as qt  # type: ignore[import-not-found,import-untyped]  # optional dependency

    def _qt_kron(ops: list[qt.Qobj]) -> qt.Qobj:
        out = ops[0]
        for op in ops[1:]:
            out = qt.tensor(out, op)
        return out

    I2 = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    hz = 0.0 * _qt_kron([I2] * n)
    for i in range(n):
        ops = [sz if k == i else I2 for k in range(n)]
        hz += float(omega[i]) * _qt_kron(ops)

    hxy = 0.0 * _qt_kron([I2] * n)
    for i in range(n - 1):
        j = i + 1
        coeff = float(k_matrix[i, j])
        xx_ops = [sx if k in (i, j) else I2 for k in range(n)]
        yy_ops = [sy if k in (i, j) else I2 for k in range(n)]
        hxy += coeff * _qt_kron(xx_ops)
        hxy += coeff * _qt_kron(yy_ops)

    uz = (-1j * hz * t_step).expm()
    uxy = (-1j * hxy * t_step).expm()

    psi0_vec = _computational_basis_state(initial)
    psi0 = qt.Qobj(
        psi0_vec.reshape(-1, 1),
        dims=[[2] * n, [1] * n],
    )
    initial_parity = bin(int(initial, 2)).count("1") % 2
    mask = _parity_mask(n, initial_parity)

    leakages: dict[int, float] = {}
    psi = psi0
    prev_d = 0
    for d in sorted(depths):
        steps_needed = d - prev_d
        for _ in range(steps_needed):
            psi = uz * psi
            psi = uxy * psi
        prev_d = d
        probs = np.abs(psi.full().flatten()) ** 2
        leakages[d] = float(np.dot(mask, probs))
    return leakages


def _select_backend(backend: Backend) -> Literal["numpy", "qutip"]:
    if backend == "auto":
        return "qutip" if available_baselines()["qutip"] else "numpy"
    if backend == "qutip" and not available_baselines()["qutip"]:
        raise ModuleNotFoundError(
            "qutip backend requested but qutip is not installed — "
            "install with `pip install qutip` or choose backend='numpy'",
        )
    return backend


def compute_classical_leakage_reference(
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    t_step: float = DEFAULT_T_STEP,
    initial_even: str = DEFAULT_INITIAL_EVEN,
    initial_odd: str = DEFAULT_INITIAL_ODD,
    backend: Backend = "auto",
) -> ClassicalLeakageReference:
    """Build the DLA-parity Hamiltonian and return the noiseless leakage curve.

    Parameters
    ----------
    n_qubits:
        Number of qubits (chain length).
    depths:
        Sweep of Trotter depths to evaluate.
    t_step:
        Trotter step size (seconds).
    initial_even, initial_odd:
        Bitstrings for the even / odd sector initial states.
    backend:
        ``"numpy"`` (dense matrix exponential, always available),
        ``"qutip"`` (optional), or ``"auto"``.

    Returns
    -------
    :class:`ClassicalLeakageReference`
        Per-(depth, sector) classical leakage, the backend used,
        and convenience predicates.

    Raises
    ------
    ValueError
        If ``n_qubits`` does not match the length of one of the
        initial bitstrings, or if any depth is negative.
    ModuleNotFoundError
        If the caller asked for ``backend="qutip"`` and qutip is
        not installed.
    """
    if n_qubits != len(initial_even):
        raise ValueError(
            f"initial_even={initial_even!r} has length {len(initial_even)}, expected {n_qubits}",
        )
    if n_qubits != len(initial_odd):
        raise ValueError(
            f"initial_odd={initial_odd!r} has length {len(initial_odd)}, expected {n_qubits}",
        )
    if any(d < 0 for d in depths):
        raise ValueError(f"depths must be non-negative, got {list(depths)!r}")

    resolved = _select_backend(backend)
    omega = _build_omega(n_qubits)
    k_matrix = _build_k_matrix(n_qubits)

    evolve = _leakage_qutip if resolved == "qutip" else _leakage_numpy
    leak_even = evolve(n_qubits, initial_even, depths, t_step, omega, k_matrix)
    leak_odd = evolve(n_qubits, initial_odd, depths, t_step, omega, k_matrix)

    points: list[ClassicalLeakagePoint] = []
    for d in sorted(depths):
        points.append(
            ClassicalLeakagePoint(
                depth=d,
                sector="even",
                initial=initial_even,
                leakage=leak_even[d],
            ),
        )
        points.append(
            ClassicalLeakagePoint(
                depth=d,
                sector="odd",
                initial=initial_odd,
                leakage=leak_odd[d],
            ),
        )
    return ClassicalLeakageReference(
        backend=resolved,
        n_qubits=n_qubits,
        t_step=t_step,
        depths=tuple(sorted(depths)),
        points=tuple(points),
    )
