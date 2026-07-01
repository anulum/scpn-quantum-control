# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Finite Size Scaling
"""Finite-size scaling for K_c extraction from small exact quantum systems.

Estimates the thermodynamic-limit K_c from small-N exact diagonalisation data.
For BKT-motivated finite-size studies, one common ansatz uses logarithmic
corrections:

    K_c(N) = K_c(∞) + a / (log N)²

(standard BKT FSS ansatz, Nomura-Kitazawa 2002).

This module reports finite-size gap-minimum diagnostics:
1. Computes K_c(N) from gap minimum for N = 2, 3, 4, 5 qubits
2. Fits the BKT-motivated FSS ansatz to extrapolate K_c(∞)
3. Also fits power-law K_c(N) = K_c(∞) + b/N^ν for comparison

Methods: Nomura-Kitazawa level spectroscopy (2002),
Hasenbusch-Pinn log-correction extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..bridge.knm_hamiltonian import OMEGA_N_16, knm_to_dense_matrix
from ..dense_budget import require_dense_eigensolver_workspace

FSS_CLAIM_BOUNDARY = (
    "local dense exact finite-size gap diagnostics for small Kuramoto-XY systems; "
    "fits are numerical extrapolation evidence only, no hardware execution, "
    "no isolated performance claim, and no thermodynamic-limit proof is implied"
)


@dataclass(frozen=True)
class FSSFitDiagnostics:
    """Least-squares diagnostics for one finite-size scaling ansatz.

    Parameters
    ----------
    model:
        Stable identifier for the fitted ansatz.
    extrapolated_k_c:
        Intercept of the linearised finite-size model, interpreted as
        ``K_c(infinity)`` for the selected ansatz.
    correction_coefficient:
        Coefficient multiplying the finite-size correction coordinate.
    residuals:
        Pointwise residuals ``observed K_c(N) - fitted K_c(N)`` in the order
        of the input system sizes.
    residual_norm:
        Euclidean norm of the residual vector.
    max_abs_residual:
        Largest absolute residual across the fitted system sizes.
    design_condition:
        Condition number of the two-column linear design matrix.
    rank:
        Numerical rank reported by ``numpy.linalg.lstsq``.
    n_points:
        Number of finite-size points used in the fit.
    claim_boundary:
        Claim boundary attached to the diagnostic evidence.
    """

    model: str
    extrapolated_k_c: float
    correction_coefficient: float
    residuals: tuple[float, ...]
    residual_norm: float
    max_abs_residual: float
    design_condition: float
    rank: int
    n_points: int
    claim_boundary: str = FSS_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready finite-size fit diagnostics.

        Returns
        -------
        dict[str, object]
            Primitive mapping suitable for dashboards, documentation manifests,
            and persisted audit artifacts.
        """
        return {
            "model": self.model,
            "extrapolated_k_c": self.extrapolated_k_c,
            "correction_coefficient": self.correction_coefficient,
            "residuals": list(self.residuals),
            "residual_norm": self.residual_norm,
            "max_abs_residual": self.max_abs_residual,
            "design_condition": self.design_condition,
            "rank": self.rank,
            "n_points": self.n_points,
            "claim_boundary": self.claim_boundary,
        }


@dataclass
class FSSResult:
    """Finite-size scaling result for dense local gap-minimum scans.

    The legacy extrapolated values are kept beside richer fit diagnostics so
    existing callers can continue to read ``k_c_extrapolated_bkt`` and
    ``k_c_extrapolated_power`` while promotion gates can inspect residuals and
    fit conditioning.

    Parameters
    ----------
    system_sizes:
        Qubit counts used in the local exact finite-size scan.
    k_c_values:
        Gap-minimum coupling estimates ``K_c(N)`` aligned with
        ``system_sizes``.
    gap_min_values:
        Minimum spectral gap observed at each scanned system size.
    k_c_extrapolated_bkt:
        Legacy scalar intercept from the BKT logarithmic-correction ansatz, or
        ``None`` when fewer than two finite-size points are available or the
        linear solve fails.
    k_c_extrapolated_power:
        Legacy scalar intercept from the fixed-exponent inverse-size ansatz, or
        ``None`` when fewer than two finite-size points are available or the
        linear solve fails.
    bkt_fit:
        Full least-squares diagnostics for the BKT ansatz.
    power_fit:
        Full least-squares diagnostics for the inverse-size ansatz.
    claim_boundary:
        Claim boundary describing what this exact local scan does and does not
        establish.
    """

    system_sizes: list[int]
    k_c_values: list[float]  # K_c(N) from gap minimum
    gap_min_values: list[float]  # minimum gap at each N
    k_c_extrapolated_bkt: float | None  # from BKT ansatz fit
    k_c_extrapolated_power: float | None  # from power-law fit
    bkt_fit: FSSFitDiagnostics | None = None
    power_fit: FSSFitDiagnostics | None = None
    claim_boundary: str = FSS_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready finite-size scaling evidence.

        Returns
        -------
        dict[str, Any]
            Mapping containing raw scan outputs, legacy scalar extrapolations,
            rich fit diagnostics, and the attached claim boundary.
        """
        return {
            "system_sizes": self.system_sizes,
            "k_c_values": self.k_c_values,
            "gap_min_values": self.gap_min_values,
            "k_c_extrapolated_bkt": self.k_c_extrapolated_bkt,
            "k_c_extrapolated_power": self.k_c_extrapolated_power,
            "bkt_fit": None if self.bkt_fit is None else self.bkt_fit.to_dict(),
            "power_fit": None if self.power_fit is None else self.power_fit.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


def _find_kc_from_gap(
    omega: NDArray[np.float64],
    K_topology: NDArray[np.float64],
    k_range: NDArray[np.float64],
    *,
    max_dense_gib: float | None = None,
) -> tuple[float, float]:
    """Find K_c(N) = K_base where spectral gap is minimized.

    Returns (k_c, min_gap).
    """
    n = len(omega)
    require_dense_eigensolver_workspace(
        n,
        max_gib=max_dense_gib,
        label="finite-size gap dense eigensolver workspace",
    )
    gaps = np.zeros(len(k_range))
    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        H = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
        eigvals = np.linalg.eigvalsh(H)
        gaps[idx] = float(eigvals[1] - eigvals[0])

    min_idx = int(np.argmin(gaps))
    return float(k_range[min_idx]), float(gaps[min_idx])


def _ring_topology(n: int) -> NDArray[np.float64]:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    result: NDArray[np.float64] = T
    return result


def _validate_system_sizes(system_sizes: list[int] | None) -> list[int]:
    if system_sizes is None:
        return [2, 3, 4]
    if not system_sizes:
        raise ValueError("system_sizes must contain at least one system size")
    sizes: list[int] = []
    for size in system_sizes:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("system_sizes must contain integer qubit counts")
        if size < 2:
            raise ValueError("system_sizes entries must be at least 2")
        if size > len(OMEGA_N_16):
            raise ValueError("system_sizes entries exceed the available frequency table")
        sizes.append(size)
    if len(set(sizes)) != len(sizes):
        raise ValueError("system_sizes entries must be unique")
    return sizes


def _validate_k_range(k_range: NDArray[np.float64] | None) -> NDArray[np.float64]:
    if k_range is None:
        return np.linspace(0.3, 6.0, 20, dtype=np.float64)
    values = np.asarray(k_range, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("k_range must be one-dimensional")
    if len(values) < 2:
        raise ValueError("k_range must contain at least two coupling values")
    if not np.all(np.isfinite(values)):
        raise ValueError("k_range values must be finite")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("k_range values must be strictly increasing")
    result: NDArray[np.float64] = values
    return result


def finite_size_scaling(
    system_sizes: list[int] | None = None,
    k_range: NDArray[np.float64] | None = None,
    *,
    max_dense_gib: float | None = None,
) -> FSSResult:
    """Extract K_c from multiple system sizes and extrapolate.

    Uses ring topology with Paper 27 natural frequencies. ``max_dense_gib``
    gates each exact dense gap scan before Hamiltonian/eigensolver allocation.

    Parameters
    ----------
    system_sizes:
        Optional qubit counts to scan. Defaults to ``[2, 3, 4]``.
    k_range:
        Strictly increasing one-dimensional coupling grid. Defaults to a
        deterministic local scan from 0.3 to 6.0.
    max_dense_gib:
        Optional dense workspace limit applied before each Hamiltonian and
        eigensolver allocation.

    Returns
    -------
    FSSResult
        Local exact finite-size evidence with raw gap minima, extrapolated
        scalar fields, fit diagnostics, and claim boundaries.
    """
    sizes = _validate_system_sizes(system_sizes)
    scan_values = _validate_k_range(k_range)

    k_c_list: list[float] = []
    gap_min_list: list[float] = []

    for n in sizes:
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        kc, gmin = _find_kc_from_gap(omega, T, scan_values, max_dense_gib=max_dense_gib)
        k_c_list.append(kc)
        gap_min_list.append(gmin)

    # Fit BKT ansatz: K_c(N) = K_c(∞) + a / (log N)²
    bkt_fit = _fit_bkt_diagnostics(sizes, k_c_list)
    k_c_bkt = None if bkt_fit is None else bkt_fit.extrapolated_k_c

    # Fit power-law: K_c(N) = K_c(∞) + b / N^ν
    power_fit = _fit_power_diagnostics(sizes, k_c_list)
    k_c_power = None if power_fit is None else power_fit.extrapolated_k_c

    return FSSResult(
        system_sizes=sizes,
        k_c_values=k_c_list,
        gap_min_values=gap_min_list,
        k_c_extrapolated_bkt=k_c_bkt,
        k_c_extrapolated_power=k_c_power,
        bkt_fit=bkt_fit,
        power_fit=power_fit,
    )


def _fit_bkt_ansatz(sizes: list[int], k_c_vals: list[float]) -> float | None:
    """Fit K_c(N) = K_c(∞) + a / (log N)²."""
    diagnostics = _fit_bkt_diagnostics(sizes, k_c_vals)
    return None if diagnostics is None else diagnostics.extrapolated_k_c


def _fit_bkt_diagnostics(sizes: list[int], k_c_vals: list[float]) -> FSSFitDiagnostics | None:
    """Fit BKT logarithmic-correction diagnostics."""
    if len(sizes) < 2:
        return None
    log_n_sq = np.array([1.0 / max(np.log(n) ** 2, 0.01) for n in sizes])
    return _fit_linear_diagnostics("bkt_log_correction", sizes, k_c_vals, log_n_sq)


def _fit_power_ansatz(sizes: list[int], k_c_vals: list[float]) -> float | None:
    """Fit K_c(N) = K_c(∞) + b / N."""
    diagnostics = _fit_power_diagnostics(sizes, k_c_vals)
    return None if diagnostics is None else diagnostics.extrapolated_k_c


def _fit_power_diagnostics(sizes: list[int], k_c_vals: list[float]) -> FSSFitDiagnostics | None:
    """Fit inverse-size power-law diagnostics with fixed exponent one."""
    if len(sizes) < 2:
        return None
    inv_n = np.array([1.0 / n for n in sizes])
    return _fit_linear_diagnostics("power_law_nu_1", sizes, k_c_vals, inv_n)


def _fit_linear_diagnostics(
    model: str,
    sizes: list[int],
    k_c_vals: list[float],
    correction_coordinate: NDArray[np.float64],
) -> FSSFitDiagnostics | None:
    """Fit a two-parameter linearized finite-size ansatz."""
    k_c = np.array(k_c_vals)
    A = np.column_stack([np.ones(len(sizes)), correction_coordinate])
    try:
        coefficients, _, rank, singular_values = np.linalg.lstsq(A, k_c, rcond=None)
    except np.linalg.LinAlgError:
        return None
    predicted = A @ coefficients
    residuals = k_c - predicted
    min_singular = float(np.min(singular_values)) if len(singular_values) else 0.0
    max_singular = float(np.max(singular_values)) if len(singular_values) else 0.0
    condition = float("inf") if min_singular <= 0.0 else max_singular / min_singular
    return FSSFitDiagnostics(
        model=model,
        extrapolated_k_c=float(coefficients[0]),
        correction_coefficient=float(coefficients[1]),
        residuals=tuple(float(value) for value in residuals),
        residual_norm=float(np.linalg.norm(residuals)),
        max_abs_residual=float(np.max(np.abs(residuals))),
        design_condition=condition,
        rank=int(rank),
        n_points=len(sizes),
    )
