# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific design declarations
"""Immutable supplied design inputs, independent of Studio and solver engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _snapshot(
    values: object, *, complex_values: bool = False
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Take a finite numeric snapshot backed by immutable owned bytes."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("scientific input must be a rectangular numeric array") from exc
    if array.dtype.kind not in ("i", "u", "f", "c"):
        raise ValueError("scientific input requires numeric scalars without coercion")
    if array.dtype.kind == "c" and not complex_values:
        raise ValueError("real scientific input cannot contain complex scalars")
    if complex_values:
        result = np.asarray(array, dtype=np.complex128)
        if not np.isfinite(result).all():
            raise ValueError("scientific input must be finite")
        return np.frombuffer(result.tobytes(), dtype=np.complex128).reshape(result.shape)
    real = np.asarray(array, dtype=np.float64)
    if not np.isfinite(real).all():
        raise ValueError("scientific input must be finite")
    return np.frombuffer(real.tobytes(), dtype=np.float64).reshape(real.shape)


@dataclass(frozen=True, slots=True)
class ScientificUnits:
    """Explicit units for time, rates, state and observable values.

    Parameters
    ----------
    time
        ``s`` for dimensional time or ``1`` for declared dimensionless time.
    frequency
        ``rad/s`` with dimensional time or ``1`` with dimensionless time.
    coupling
        ``rad/s`` with dimensional time or ``1`` with dimensionless time.
    state
        ``rad`` for phase angles or ``1`` for quantum amplitudes.
    observable
        ``1`` for either supported dimensionless observable.

    """

    time: str
    frequency: str
    coupling: str
    state: str
    observable: str


@dataclass(frozen=True, slots=True)
class DesignObjective:
    """Declared design objective with an optional finite target.

    Parameters
    ----------
    kind
        Simulation, phase synchronisation, observable maximisation or gate cost.
    target
        Dimensionless observable target, gate count or no target for simulation.
    unit
        ``1`` for observables/simulation; ``gate`` for gate-count objectives.

    """

    kind: Literal["simulate", "synchronise", "maximise_observable", "minimise_gate_cost"]
    target: float | None
    unit: str


@dataclass(frozen=True, slots=True)
class ScientificDesign:
    """Immutable model, graph, state/history, observable and objective inputs.

    Parameters
    ----------
    model
        ``phase_kuramoto`` or ``quantum_xy``; no equivalence is implied.
    normalisation
        Pairwise sum or, for phase models, population-mean coupling.
    coordinate_space
        Logical indices or caller-declared physical indices; no placement proof.
    units
        Exact unit declarations; conversion is never inferred.
    topology
        Unique sorted undirected index pairs, including declared zero-weight edges.
    initial_state
        Phase vector of shape ``(N,)`` or normalized amplitudes ``(2**N,)``.
    observable
        Phase order parameter or weighted per-qubit spin-Z expectation.
    observable_weights
        Finite per-oscillator weights of shape ``(N,)``.
    objective
        Explicit objective and target units.
    history_times
        Optional phase-history times ``(H,)``; last time is zero.
    history_states
        Paired phase-history states ``(H,N)``; final row equals initial_state.
    trainable
        Names of trainable numeric parameters; absence declares none trainable.

    Notes
    -----
    Arrays are copied to immutable byte buffers. Cross-field shapes, units and
    model constraints are checked by the public problem binding, before export.

    """

    model: Literal["phase_kuramoto", "quantum_xy"]
    normalisation: Literal["pairwise_sum", "population_mean"]
    coordinate_space: Literal["logical", "physical"]
    units: ScientificUnits
    topology: tuple[tuple[int, int], ...]
    initial_state: NDArray[np.float64] | NDArray[np.complex128]
    observable: Literal["phase_order_parameter", "spin_z"]
    observable_weights: NDArray[np.float64]
    objective: DesignObjective
    history_times: NDArray[np.float64] | None = None
    history_states: NDArray[np.float64] | None = None
    trainable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot supplied numeric values without retaining mutable buffers.

        Raises
        ------
        ValueError
            Numeric input is nonfinite, nonnumeric or has unsupported coercion.

        """
        object.__setattr__(
            self,
            "initial_state",
            _snapshot(self.initial_state, complex_values=self.model == "quantum_xy"),
        )
        object.__setattr__(self, "observable_weights", _snapshot(self.observable_weights))
        try:
            topology = tuple(tuple(edge) for edge in self.topology)
            trainable = tuple(self.trainable)
        except TypeError as exc:
            raise ValueError("topology and trainable require sequences") from exc
        if not all(isinstance(key, str) for key in trainable):
            raise ValueError("trainable requires string parameter keys")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "trainable", trainable)
        for key in ("history_times", "history_states"):
            values = getattr(self, key)
            if values is not None:
                object.__setattr__(self, key, _snapshot(values))
