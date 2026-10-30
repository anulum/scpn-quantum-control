# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Registered Phase-QNode gate dispatch
"""Select the native operator for a registered Phase-QNode gate name.

The dispatcher owns which construction each registered gate uses. Every
numerical operation is a supplied factory, so callers keep their own native
runtime, constant creation order and live dependency lookups, and every
native tensor graph and parameter gradient is preserved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

NativeOperator = Callable[[], Any]
"""Zero-argument factory returning one native operator."""


@dataclass(frozen=True)
class GateOperatorFactories:
    """Native operator constructions used by :func:`dispatch_gate_operator`.

    Parameters
    ----------
    identity, x, y, z, h, s, t, sx
        One-qubit operator factories.
    rotation
        Build ``exp(-i theta P / 2)`` from ``theta``, an identity factory and
        a Pauli-product factory.
    phase
        Build the phase gate for ``theta``.
    two_qubit
        Build the fixed two-qubit operator named ``"cnot"``, ``"cy"`` or ``"swap"``.
    controlled_z
        Build the controlled-Z operator with the given number of controls.
    controlled
        Resolve the callable that adds one control qubit to a target operator.
        It is called before the target is built, so the callable is captured
        before any target construction runs.
    ccnot, cswap
        Three-qubit Toffoli and Fredkin operator factories.
    two_qubit_identity
        Four-dimensional identity factory.
    kron
        Resolve the native Kronecker product; it is called before either
        factor is built.
    gate
        Recursive gate lookup used for controlled rotations.
    unsupported
        Called for a gate name this dispatcher does not register; it must
        raise the caller's refusal.

    """

    identity: NativeOperator
    x: NativeOperator
    y: NativeOperator
    z: NativeOperator
    h: NativeOperator
    s: NativeOperator
    t: NativeOperator
    sx: NativeOperator
    rotation: Callable[[Any, NativeOperator, NativeOperator], Any]
    phase: Callable[[Any], Any]
    two_qubit: Callable[[str], Any]
    controlled_z: Callable[[int], Any]
    controlled: Callable[[], Callable[[Any], Any]]
    ccnot: NativeOperator
    cswap: NativeOperator
    two_qubit_identity: NativeOperator
    kron: Callable[[], Callable[[Any, Any], Any]]
    gate: Callable[[str, Any], Any]
    unsupported: Callable[[str], Any]


def dispatch_gate_operator(gate: str, theta: Any, factories: GateOperatorFactories) -> Any:
    """Return the native operator for one registered Phase-QNode gate.

    Parameters
    ----------
    gate
        Registered gate name, such as ``"rx"``, ``"cnot"`` or ``"rzz"``.
    theta
        Native rotation angle; unused by fixed gates.
    factories
        Caller-supplied native constructions.

    Returns
    -------
    Any
        The native operator built by the matching factory.

    """
    fixed: dict[str, NativeOperator] = {
        "h": factories.h,
        "x": factories.x,
        "y": factories.y,
        "z": factories.z,
        "s": factories.s,
        "t": factories.t,
        "sx": factories.sx,
    }
    if gate in fixed:
        return fixed[gate]()
    rotations: dict[str, NativeOperator] = {
        "rx": factories.x,
        "ry": factories.y,
        "rz": factories.z,
    }
    if gate in rotations:
        return factories.rotation(theta, factories.identity, rotations[gate])
    if gate == "phase":
        return factories.phase(theta)
    if gate in ("cnot", "cy", "swap"):
        return factories.two_qubit(gate)
    if gate == "cz":
        return factories.controlled_z(1)
    controlled: dict[str, NativeOperator] = {
        "ch": factories.h,
        "cs": factories.s,
        "ct": factories.t,
    }
    if gate in controlled:
        return factories.controlled()(controlled[gate]())
    if gate == "ccnot":
        return factories.ccnot()
    if gate == "ccz":
        return factories.controlled_z(2)
    if gate == "cswap":
        return factories.cswap()
    if gate in ("crx", "cry", "crz"):
        return factories.controlled()(factories.gate(gate[1:], theta))
    pairs: dict[str, NativeOperator] = {"rxx": factories.x, "ryy": factories.y, "rzz": factories.z}
    if gate in pairs:
        single = pairs[gate]
        return factories.rotation(
            theta, factories.two_qubit_identity, lambda: factories.kron()(single(), single())
        )
    return factories.unsupported(gate)


def refuse_unregistered_gate(message: str) -> Callable[[str], NoReturn]:
    """Return an ``unsupported`` factory raising ``ValueError`` for a gate name.

    Parameters
    ----------
    message
        Refusal prefix; the gate name follows it after ``": "``.

    Returns
    -------
    Callable[[str], NoReturn]
        Factory that raises ``ValueError(f"{message}: {name}")``.

    """

    def refuse(name: str) -> NoReturn:
        raise ValueError(f"{message}: {name}")

    return refuse


__all__ = [
    "GateOperatorFactories",
    "NativeOperator",
    "dispatch_gate_operator",
    "refuse_unregistered_gate",
]
