# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PennyLane Bridge Test Helpers
"""Typed PennyLane fakes shared by bridge and provider integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

FloatArray = NDArray[np.float64]


class _FakeObservable:
    def __init__(
        self,
        name: str,
        wires: int | tuple[int, ...] | list[int],
        *,
        coefficient: float = 1.0,
        terms: tuple[_FakeObservable, ...] = (),
    ) -> None:
        self.name = name
        self.wires = tuple(wires) if isinstance(wires, list | tuple) else (int(wires),)
        self.coefficient = float(coefficient)
        self.terms = terms

    def __matmul__(self, other: _FakeObservable) -> _FakeObservable:
        return _FakeObservable("Prod", self.wires + other.wires, terms=(self, other))

    def __rmul__(self, coefficient: float) -> _FakeObservable:
        return _FakeObservable(
            self.name,
            self.wires,
            coefficient=float(coefficient) * self.coefficient,
            terms=self.terms,
        )


class _FakePennyLane:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.devices: list[dict[str, object]] = []

    def device(self, name: str, *, wires: int, shots: int | None = None) -> dict[str, object]:
        payload = {"name": name, "wires": wires, "shots": shots}
        self.devices.append(payload)
        return payload

    def qnode(
        self,
        device: object,
        **metadata: object,
    ) -> Callable[[Callable[[FloatArray], object]], Callable[[FloatArray], float]]:
        def decorate(function: Callable[[FloatArray], object]) -> Callable[[FloatArray], float]:
            def wrapper(params: FloatArray) -> float:
                function(params)
                circuit = cast(Any, wrapper)._scpn_phase_qnode_circuit
                return execute_phase_qnode_circuit(circuit, params).value

            cast(Any, wrapper).device = device
            cast(Any, wrapper).metadata = metadata
            return wrapper

        return decorate

    def grad(self, qnode: Callable[[FloatArray], object]) -> Callable[[FloatArray], FloatArray]:
        def gradient(params: FloatArray) -> FloatArray:
            circuit = cast(Any, qnode)._scpn_phase_qnode_circuit
            return parameter_shift_phase_qnode_gradient(circuit, params).gradient

        return gradient

    def expval(self, observable: _FakeObservable) -> _FakeObservable:
        self.calls.append(("expval", (observable,), {}))
        return observable

    def Hamiltonian(
        self,
        coefficients: list[float],
        observables: list[_FakeObservable],
    ) -> _FakeObservable:
        self.calls.append(("Hamiltonian", (tuple(coefficients), tuple(observables)), {}))
        return _FakeObservable("Hamiltonian", (), terms=tuple(observables))

    def Hermitian(self, matrix: FloatArray, *, wires: range) -> _FakeObservable:
        self.calls.append(("Hermitian", (np.asarray(matrix),), {"wires": tuple(wires)}))
        return _FakeObservable("Hermitian", tuple(wires))

    def __getattr__(self, name: str) -> Callable[..., _FakeObservable]:
        if name in {
            "Hadamard",
            "PauliX",
            "PauliY",
            "PauliZ",
            "S",
            "T",
            "SX",
            "CNOT",
            "CZ",
            "CY",
            "SWAP",
            "RX",
            "RY",
            "RZ",
            "PhaseShift",
            "CRX",
            "CRY",
            "CRZ",
            "IsingXX",
            "IsingYY",
            "IsingZZ",
        }:
            return lambda *args, **kwargs: self._operation(name, *args, **kwargs)
        raise AttributeError(name)

    def _operation(self, name: str, *args: object, **kwargs: object) -> _FakeObservable:
        self.calls.append((name, args, kwargs))
        wires = kwargs.get("wires", ())
        wire_value = cast(int | tuple[int, ...] | list[int], wires)
        if name in {"PauliX", "PauliY", "PauliZ"}:
            return _FakeObservable(name, wire_value)
        return _FakeObservable(name, wire_value)


def _objective(values: FloatArray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def _closed_form_gradient(values: FloatArray) -> FloatArray:
    return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)
