# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Reproducible classical-vs-quantum comparison artifact
"""Deterministic head-to-head comparison of classical and quantum Kuramoto routes.

This module composes the existing documented baselines into a single
serialisable record so an onboarding example can emit a reproducible artifact
instead of printing transient numbers. It does not reimplement any solver:

* the classical ODE row reuses :func:`scipy_ode_baseline`;
* the exact reference row reuses :func:`classical_exact_evolution`;
* the quantum row reuses :class:`QuantumKuramotoSolver`.

Reproducibility contract
------------------------
The dynamics comparison is RNG-free: identical inputs yield byte-identical
order-parameter values across runs and machines. The ``seed`` is recorded in
the artifact and only governs the optional randomised initial-phase mode; the
default initial condition is derived deterministically from ``omega``. Wall-clock
timing is advisory and machine-dependent, so it is reported but explicitly
excluded from the reproducible-quantity set.

Claim boundary
--------------
For the system sizes this comparison can run (``n <= 16``) the classical exact
route is both faster and exact, so the artifact records *no quantum advantage*.
The quantum row exists to document agreement and Trotter discretisation error,
not to claim speed-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_exact_evolution
from scpn_quantum_control.phase import QuantumKuramotoSolver

from .classical_baselines import scipy_ode_baseline

#: Maximum oscillator count for which the bundled ``OMEGA_N_16`` table applies.
MAX_TABLE_OSCILLATORS = 16

#: Documented failure modes shared by every comparison artifact.
FAILURE_MODES: tuple[str, ...] = (
    "quantum_trotter: first-order Trotter introduces O(dt^2) discretisation "
    "error that grows with dt and shrinks as trotter_per_step rises.",
    "classical_exact: the dense propagator path costs O(2^2n) memory below "
    "n=13 and switches to a sparse Krylov path above it; both become "
    "infeasible well before n=30.",
    "classical_ode: the SciPy route integrates the classical phase model, "
    "not the quantum XY Hamiltonian, so order-parameter agreement is an "
    "approximate cross-check rather than a derivation.",
    "scope: at n<=16 there is no quantum advantage on this path; the classical "
    "exact route is faster and exact, so timing must not be read as a speed-up "
    "claim.",
)

#: Bounded-claim statement embedded in every artifact.
CLAIM_BOUNDARY = (
    "Reproducible quantities are the order-parameter values and their error "
    "against the exact reference; timing is advisory. The comparison documents "
    "agreement and discretisation error, not quantum speed-up."
)

#: Determinism statement embedded in every artifact.
DETERMINISM = (
    "RNG-free given identical inputs; the recorded seed governs only the "
    "optional randomised initial-phase mode. Timing is excluded from the "
    "reproducible-quantity set."
)


@dataclass(frozen=True)
class ComparisonMethodRow:
    """One method's contribution to the head-to-head comparison.

    Parameters
    ----------
    method:
        Stable identifier (``classical_ode``, ``classical_exact`` or
        ``quantum_trotter``).
    backend:
        Human-readable backend description.
    available:
        Whether the method produced a result.
    r_final:
        Final Kuramoto order parameter, or ``None`` when unavailable.
    r_error_vs_exact:
        Absolute final-order-parameter error against the exact reference, or
        ``None`` for the reference row itself or an unavailable row.
    elapsed_ms:
        Advisory wall-clock time in milliseconds.
    unavailable_reason:
        Populated only when ``available`` is ``False``.
    """

    method: str
    backend: str
    available: bool
    r_final: float | None
    r_error_vs_exact: float | None
    elapsed_ms: float
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for the row."""
        return {
            "method": self.method,
            "backend": self.backend,
            "available": self.available,
            "r_final": self.r_final,
            "r_error_vs_exact": self.r_error_vs_exact,
            "elapsed_ms": self.elapsed_ms,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class ReproducibleKuramotoComparison:
    """Serialisable record of one classical-vs-quantum Kuramoto comparison."""

    n_oscillators: int
    t_max: float
    dt: float
    trotter_per_step: int
    seed: int
    initial_condition: str
    reference_method: str
    rows: tuple[ComparisonMethodRow, ...]
    failure_modes: tuple[str, ...] = FAILURE_MODES
    claim_boundary: str = CLAIM_BOUNDARY
    determinism: str = DETERMINISM
    metadata: dict[str, Any] = field(default_factory=dict)

    def row(self, method: str) -> ComparisonMethodRow:
        """Return the row for ``method``.

        Raises
        ------
        KeyError
            If no row carries the requested method identifier.
        """
        for candidate in self.rows:
            if candidate.method == method:
                return candidate
        raise KeyError(method)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for the full comparison."""
        return {
            "n_oscillators": self.n_oscillators,
            "t_max": self.t_max,
            "dt": self.dt,
            "trotter_per_step": self.trotter_per_step,
            "seed": self.seed,
            "initial_condition": self.initial_condition,
            "reference_method": self.reference_method,
            "rows": [row.to_dict() for row in self.rows],
            "failure_modes": list(self.failure_modes),
            "claim_boundary": self.claim_boundary,
            "determinism": self.determinism,
            "metadata": dict(self.metadata),
        }


def _resolve_problem(
    n_oscillators: int,
    K: NDArray[np.float64] | None,
    omega: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve and validate the coupling matrix and frequency vector."""
    if K is None:
        coupling = build_knm_paper27(L=n_oscillators)
    else:
        coupling = np.asarray(K, dtype=np.float64)
        if coupling.shape != (n_oscillators, n_oscillators):
            raise ValueError(
                f"K must have shape {(n_oscillators, n_oscillators)}, got {coupling.shape}"
            )
    if omega is None:
        frequencies = OMEGA_N_16[:n_oscillators].copy()
    else:
        frequencies = np.asarray(omega, dtype=np.float64)
        if frequencies.shape != (n_oscillators,):
            raise ValueError(f"omega must have shape {(n_oscillators,)}, got {frequencies.shape}")
    return coupling, frequencies


def _initial_phases(
    omega: NDArray[np.float64],
    seed: int,
    randomise_initial_phases: bool,
) -> tuple[NDArray[np.float64] | None, str]:
    """Return the initial phase vector and its provenance label."""
    if not randomise_initial_phases:
        return None, "omega_phase"
    rng = np.random.default_rng(seed)
    phases = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=omega.shape[0]), dtype=np.float64)
    return phases, "seeded_uniform"


def run_reproducible_kuramoto_comparison(
    n_oscillators: int = 8,
    *,
    t_max: float = 1.0,
    dt: float = 0.1,
    trotter_per_step: int = 5,
    seed: int = 42,
    randomise_initial_phases: bool = False,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
) -> ReproducibleKuramotoComparison:
    """Run the deterministic classical-vs-quantum Kuramoto comparison.

    Parameters
    ----------
    n_oscillators:
        Number of oscillators / qubits, ``2 <= n <= 16``.
    t_max:
        Total evolution time; must be positive.
    dt:
        Time step; must be positive and not exceed ``t_max``.
    trotter_per_step:
        Trotter sub-steps per ``dt`` for the quantum route; must be ``>= 1``.
    seed:
        Recorded seed; governs the optional randomised initial-phase mode only.
        Must be non-negative.
    randomise_initial_phases:
        When ``True`` the initial phases are drawn from a seeded uniform
        distribution; otherwise they are derived deterministically from
        ``omega`` exactly as the documented baselines do.
    K:
        Optional coupling matrix; defaults to the Paper 27 ``K_nm`` matrix.
    omega:
        Optional frequency vector; defaults to the bundled ``OMEGA_N_16`` slice.

    Returns
    -------
    ReproducibleKuramotoComparison
        The exact route is the reference; the classical ODE and quantum Trotter
        rows carry their error against it.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if not 2 <= n_oscillators <= MAX_TABLE_OSCILLATORS:
        raise ValueError(
            f"n_oscillators must satisfy 2 <= n <= {MAX_TABLE_OSCILLATORS}, got {n_oscillators}"
        )
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if dt > t_max:
        raise ValueError(f"dt ({dt}) must not exceed t_max ({t_max})")
    if trotter_per_step < 1:
        raise ValueError(f"trotter_per_step must be >= 1, got {trotter_per_step}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    coupling, frequencies = _resolve_problem(n_oscillators, K, omega)
    theta0, initial_condition = _initial_phases(frequencies, seed, randomise_initial_phases)

    start = time.perf_counter()
    exact = classical_exact_evolution(n_oscillators, t_max, dt, coupling, frequencies)
    exact_ms = (time.perf_counter() - start) * 1000.0
    r_exact = float(exact["R"][-1])

    ode = scipy_ode_baseline(coupling, frequencies, t_max=t_max, dt=dt, theta0=theta0)
    r_ode = ode.r_final

    solver = QuantumKuramotoSolver(n_oscillators, coupling, frequencies)
    start = time.perf_counter()
    quantum = solver.run(t_max=t_max, dt=dt, trotter_per_step=trotter_per_step)
    quantum_ms = (time.perf_counter() - start) * 1000.0
    r_quantum = float(quantum["R"][-1])

    rows = (
        ComparisonMethodRow(
            method="classical_exact",
            backend="scipy matrix exponential / sparse Krylov",
            available=True,
            r_final=r_exact,
            r_error_vs_exact=None,
            elapsed_ms=exact_ms,
        ),
        ComparisonMethodRow(
            method="classical_ode",
            backend=ode.backend,
            available=ode.available,
            r_final=r_ode,
            r_error_vs_exact=None if r_ode is None else abs(r_ode - r_exact),
            elapsed_ms=ode.elapsed_ms,
            unavailable_reason=ode.unavailable_reason,
        ),
        ComparisonMethodRow(
            method="quantum_trotter",
            backend="statevector Trotter (QuantumKuramotoSolver)",
            available=True,
            r_final=r_quantum,
            r_error_vs_exact=abs(r_quantum - r_exact),
            elapsed_ms=quantum_ms,
        ),
    )

    return ReproducibleKuramotoComparison(
        n_oscillators=n_oscillators,
        t_max=t_max,
        dt=dt,
        trotter_per_step=trotter_per_step,
        seed=seed,
        initial_condition=initial_condition,
        reference_method="classical_exact",
        rows=rows,
        metadata={
            "coupling_source": "paper27" if K is None else "caller",
            "omega_source": "omega_n_16" if omega is None else "caller",
        },
    )
