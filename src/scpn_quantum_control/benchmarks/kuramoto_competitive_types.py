# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Shared types for the Kuramoto competitive benchmark
"""Problem and row types shared by the competitive harness and its adapters.

The competitive comparison is split across three modules so no single file
carries the orchestration *and* every external-solver integration: this module
holds the value types both sides exchange (the deterministic
:class:`KuramotoProblem` and one serialisable :class:`CompetitorRow`), the
:mod:`kuramoto_external_competitors` module holds the external-solver adapters,
and :mod:`kuramoto_competitive_benchmark` orchestrates the run. Keeping the types
here breaks the import cycle: both other modules import from this one, and it
imports from neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KuramotoProblem:
    """A deterministic Kuramoto forward problem shared by every competitor.

    Parameters
    ----------
    coupling:
        Symmetric ``(n, n)`` coupling matrix ``K_ij``.
    omega:
        Natural-frequency vector of length ``n``.
    theta0:
        Initial phase vector of length ``n``.
    t_max:
        Total integration time; positive.
    dt:
        Fixed-grid step for the fixed-step methods and the SciPy evaluation
        grid; positive and not exceeding ``t_max``.
    seed:
        Seed the problem was generated from, recorded for provenance.
    """

    coupling: NDArray[np.float64]
    omega: NDArray[np.float64]
    theta0: NDArray[np.float64]
    t_max: float
    dt: float
    seed: int

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators in the problem."""
        return int(self.omega.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of fixed-grid steps, ``round(t_max / dt)``."""
        return int(round(self.t_max / self.dt))


def build_default_problem(
    n_oscillators: int = 12,
    *,
    seed: int = 20260628,
    t_max: float = 6.0,
    dt: float = 0.01,
) -> KuramotoProblem:
    """Build the canonical deterministic Kuramoto benchmark problem.

    The coupling is a seeded symmetric non-negative matrix scaled above the
    synchronisation threshold so the dynamics are non-trivial, the frequencies
    are seeded standard-normal, and the initial phases are seeded uniform on
    ``[0, 2*pi)``. The construction is domain-agnostic and depends only on the
    seed, so the comparison is reproducible without any project constant.

    Parameters
    ----------
    n_oscillators:
        Number of oscillators, ``n >= 2``.
    seed:
        Seed for the coupling, frequencies, and initial phases.
    t_max:
        Total integration time; positive.
    dt:
        Fixed-grid step; positive and not exceeding ``t_max``.

    Returns
    -------
    KuramotoProblem
        The deterministic problem.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if n_oscillators < 2:
        raise ValueError(f"n_oscillators must be >= 2, got {n_oscillators}")
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if dt > t_max:
        raise ValueError(f"dt ({dt}) must not exceed t_max ({t_max})")

    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n_oscillators, n_oscillators))
    symmetric = 0.5 * (raw + raw.T)
    np.fill_diagonal(symmetric, 0.0)
    coupling = np.asarray(2.0 * symmetric / n_oscillators, dtype=np.float64)
    omega = np.asarray(rng.standard_normal(n_oscillators), dtype=np.float64)
    theta0 = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators), dtype=np.float64)
    return KuramotoProblem(
        coupling=coupling,
        omega=omega,
        theta0=theta0,
        t_max=float(t_max),
        dt=float(dt),
        seed=int(seed),
    )


@dataclass(frozen=True)
class CompetitorRow:
    """One solver's contribution to the competitive comparison.

    Parameters
    ----------
    method:
        Stable identifier (e.g. ``ours_rk4_rust``, ``scipy_solve_ivp``).
    backend:
        Human-readable backend description.
    family:
        ``ours`` for our toolkit, ``external`` for a third-party competitor.
    language:
        Implementation language of the backend (``python``, ``rust`` or
        ``julia``).
    available:
        Whether the solver produced a result.
    version:
        Recorded package version, or ``None`` when unavailable/not captured.
    r_final:
        Final Kuramoto order parameter, or ``None`` when unavailable.
    r_error_vs_reference:
        Absolute final-order-parameter error against the reference method, or
        ``None`` for the reference row itself or an unavailable row.
    elapsed_ms:
        Advisory wall-clock time in milliseconds, or ``None`` when unavailable.
    install_command:
        Command to install an absent external competitor (or build the Rust
        tier), or ``None`` when the row is available.
    unavailable_reason:
        Populated only when ``available`` is ``False``.
    """

    method: str
    backend: str
    family: str
    language: str
    available: bool
    version: str | None
    r_final: float | None
    r_error_vs_reference: float | None
    elapsed_ms: float | None
    install_command: str | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for the row."""
        return {
            "method": self.method,
            "backend": self.backend,
            "family": self.family,
            "language": self.language,
            "available": self.available,
            "version": self.version,
            "r_final": self.r_final,
            "r_error_vs_reference": self.r_error_vs_reference,
            "elapsed_ms": self.elapsed_ms,
            "install_command": self.install_command,
            "unavailable_reason": self.unavailable_reason,
        }
