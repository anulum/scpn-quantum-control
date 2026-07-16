# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto-XY-aware discrete layout cost model
"""Kuramoto-XY-aware discrete cost model for qubit-layout selection.

A candidate initial layout of the XY-Trotter circuit onto a hardware coupling
map is scored by one number,

    ``C(layout, K, omega, coupling_map)``
        ``= w_depth · post-routing depth``
        ``+ w_error · Trotter error bound``
        ``+ w_infidelity · (1 − DynQ mean gate fidelity)``,

so a discrete optimiser (KT-3) can compare layouts on a single objective. The
cost is **continuous** in the couplings ``K``, frequencies ``omega``, and gate
fidelities, and **discrete** in the layout: only the post-routing depth depends
on the integer layout, through the SWAP overhead the coupling map forces on the
all-to-all XY interaction.

The three terms reuse existing surfaces rather than reimplementing them:

* post-routing depth reuses :func:`~scpn_quantum_control.phase.xy_compiler.compile_xy_trotter`
  and Qiskit routing (injectable via ``depth_provider`` so a caller — or a test —
  can supply a cheaper or deterministic depth model);
* the Trotter error reuses
  :func:`~scpn_quantum_control.phase.trotter_error.trotter_error_bound`;
* the DynQ mean gate fidelity is the
  :class:`~scpn_quantum_control.hardware.qubit_mapper.QubitMappingResult`
  region fidelity (see :func:`dynq_mean_gate_fidelity`).

The cost function is pure and side-effect-free given a ``depth_provider``, so
KT-3's optimiser can call it in a tight loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..phase.trotter_error import trotter_error_bound
from ..phase.xy_compiler import compile_xy_trotter
from .qubit_mapper import QubitMappingResult

FloatArray = NDArray[np.float64]

#: Default single-qubit and two-qubit basis for routed-depth transpilation.
_DEFAULT_BASIS_GATES: tuple[str, ...] = ("cx", "rz", "rx", "ry")


class DepthProvider(Protocol):
    """Callable returning the post-routing depth of a layout.

    Implementations map a candidate ``layout`` of the ``n``-qubit XY-Trotter
    circuit onto ``coupling_map`` and return the routed circuit depth. The
    default is :func:`routed_layout_depth`; tests and optimisers may inject a
    cheaper deterministic model.
    """

    def __call__(
        self,
        layout: tuple[int, ...],
        K: FloatArray,
        omega: FloatArray,
        coupling_map: Any,
        *,
        t: float,
        reps: int,
    ) -> int:
        """Return the routed depth for ``layout``."""
        ...


@dataclass(frozen=True)
class CostWeights:
    """Non-negative weights combining the three cost terms.

    Parameters
    ----------
    depth
        Weight on the post-routing circuit depth (units: per depth layer).
    trotter_error
        Weight on the Trotter error bound.
    infidelity
        Weight on ``1 − mean gate fidelity``.
    """

    depth: float = 1.0
    trotter_error: float = 1.0
    infidelity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the weights.

        Raises
        ------
        ValueError
            If any weight is non-finite or negative, or if all three are zero.
        """
        for name, value in (
            ("depth", self.depth),
            ("trotter_error", self.trotter_error),
            ("infidelity", self.infidelity),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if self.depth == 0.0 and self.trotter_error == 0.0 and self.infidelity == 0.0:
            raise ValueError("at least one weight must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of the weights."""
        return {
            "depth": self.depth,
            "trotter_error": self.trotter_error,
            "infidelity": self.infidelity,
        }


@dataclass(frozen=True)
class LayoutCost:
    """Scored cost of one candidate layout with its component breakdown."""

    total: float
    routed_depth: int
    trotter_error: float
    mean_gate_fidelity: float
    depth_term: float
    trotter_error_term: float
    infidelity_term: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable mapping of the cost and its terms."""
        return {
            "total": self.total,
            "routed_depth": self.routed_depth,
            "trotter_error": self.trotter_error,
            "mean_gate_fidelity": self.mean_gate_fidelity,
            "depth_term": self.depth_term,
            "trotter_error_term": self.trotter_error_term,
            "infidelity_term": self.infidelity_term,
        }


def dynq_mean_gate_fidelity(result: QubitMappingResult) -> float:
    """Return the DynQ selected-region mean gate fidelity.

    Parameters
    ----------
    result
        A DynQ mapping result from
        :func:`~scpn_quantum_control.hardware.qubit_mapper.dynq_initial_layout`.

    Returns
    -------
    float
        The mean gate fidelity of the selected execution region.
    """
    return float(result.selected_region.mean_gate_fidelity)


def routed_layout_depth(
    layout: tuple[int, ...],
    K: FloatArray,
    omega: FloatArray,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
    basis_gates: tuple[str, ...] = _DEFAULT_BASIS_GATES,
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
) -> int:
    """Return the post-routing depth of the XY-Trotter circuit under ``layout``.

    Builds the XY-optimised Trotter circuit with
    :func:`~scpn_quantum_control.phase.xy_compiler.compile_xy_trotter`, then
    routes it onto ``coupling_map`` with ``layout`` as the initial layout and
    returns the transpiled depth. The SWAP overhead the coupling map forces on
    the all-to-all XY interaction is exactly what makes the cost discrete in the
    layout.

    Parameters
    ----------
    layout
        Initial layout: logical qubit ``i`` is placed on physical qubit
        ``layout[i]``.
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    coupling_map
        A Qiskit ``CouplingMap`` (or edge list) describing hardware connectivity.
    t, reps
        Evolution time and Trotter repetitions for the compiled circuit.
    basis_gates
        Target basis for transpilation.
    optimization_level
        Qiskit optimisation level for routing.
    seed_transpiler
        Transpiler seed; Qiskit routing is stochastic when unseeded, so pass a
        seed whenever the depth feeds a reproducible cost landscape (the KT-3
        optimiser and the layout-method comparison do).

    Returns
    -------
    int
        The routed circuit depth.
    """
    from qiskit import transpile

    circuit = compile_xy_trotter(K, omega, t, reps)
    routed = transpile(
        circuit,
        coupling_map=coupling_map,
        initial_layout=list(layout),
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    return int(routed.depth())


def _validate_inputs(
    layout: tuple[int, ...],
    K: FloatArray,
    omega: FloatArray,
    mean_gate_fidelity: float,
    t: float,
    reps: int,
) -> None:
    """Validate the cost-function inputs (fail-closed).

    Raises
    ------
    ValueError
        If the layout is not a distinct integer placement of the right length,
        the problem arrays are malformed, the fidelity is outside ``[0, 1]``, or
        the time/reps are non-positive.
    """
    n = K.shape[0]
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    if omega.shape != (n,):
        raise ValueError(f"omega must have shape {(n,)}, got {omega.shape}")
    if len(layout) != n:
        raise ValueError(f"layout must have length {n}, got {len(layout)}")
    if len(set(layout)) != n:
        raise ValueError("layout must place each logical qubit on a distinct physical qubit")
    if any(index < 0 for index in layout):
        raise ValueError("layout physical indices must be non-negative")
    if not isfinite(mean_gate_fidelity) or not 0.0 <= mean_gate_fidelity <= 1.0:
        raise ValueError("mean_gate_fidelity must be in [0, 1]")
    if not isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and positive")
    if reps < 1:
        raise ValueError("reps must be a positive integer")


def kuramoto_layout_cost(
    layout: tuple[int, ...],
    K: FloatArray,
    omega: FloatArray,
    coupling_map: Any,
    *,
    mean_gate_fidelity: float,
    weights: CostWeights | None = None,
    t: float = 0.1,
    reps: int = 5,
    order: int = 1,
    depth_provider: DepthProvider = routed_layout_depth,
) -> LayoutCost:
    """Score one candidate layout on the Kuramoto-XY-aware cost.

    Parameters
    ----------
    layout
        Initial layout: logical qubit ``i`` on physical qubit ``layout[i]``.
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    coupling_map
        Hardware connectivity passed through to ``depth_provider``.
    mean_gate_fidelity
        DynQ selected-region mean gate fidelity in ``[0, 1]`` (see
        :func:`dynq_mean_gate_fidelity`).
    weights
        Term weights; defaults to unit :class:`CostWeights`.
    t, reps, order
        Evolution time, Trotter repetitions, and product-formula order for the
        Trotter-error bound and the compiled circuit.
    depth_provider
        Callable returning the post-routing depth; defaults to
        :func:`routed_layout_depth`.

    Returns
    -------
    LayoutCost
        The total cost and its three weighted component terms.
    """
    weights = weights or CostWeights()
    _validate_inputs(layout, K, omega, mean_gate_fidelity, t, reps)

    routed_depth = depth_provider(layout, K, omega, coupling_map, t=t, reps=reps)
    error_bound = trotter_error_bound(K, omega, t, reps, order=order)
    infidelity = 1.0 - mean_gate_fidelity

    depth_term = weights.depth * routed_depth
    error_term = weights.trotter_error * error_bound
    infidelity_term = weights.infidelity * infidelity
    total = depth_term + error_term + infidelity_term

    return LayoutCost(
        total=total,
        routed_depth=routed_depth,
        trotter_error=error_bound,
        mean_gate_fidelity=mean_gate_fidelity,
        depth_term=depth_term,
        trotter_error_term=error_term,
        infidelity_term=infidelity_term,
    )
