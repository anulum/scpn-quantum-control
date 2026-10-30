# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Phase-QNode gate operators
"""Build registered Phase-QNode gate operators on a supplied native runtime.

Each factory is a thin construction over the shared gate numerics and
operator leaves; :func:`~scpn_quantum_control.phase_qnode_gate_dispatch.dispatch_gate_operator`
selects which one a gate uses. Operators are built on demand, one gate at a
time, and keep the native autograd graph of the rotation angle.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from .phase_qnode_gate_dispatch import (
    GateOperatorFactories,
    dispatch_gate_operator,
    refuse_unregistered_gate,
)
from .phase_qnode_gate_numerics import (
    controlled_z_operator,
    hadamard_operator,
    jax_phase_operator,
    pauli_rotation,
    quarter_turn_phase,
    sqrt_x_operator,
    torch_phase_operator,
)
from .phase_qnode_native_operators import (
    jax_basis_transposition,
    jax_controlled_operator,
    pauli_operator_entries,
    phase_operator_entries,
    torch_basis_transposition,
    torch_controlled_operator_rows,
    two_qubit_operator_entries,
)


def torch_gate_operator(torch_module: Any, gate: str, theta: Any) -> Any:
    """Return the complex128 Torch operator for one registered gate.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module; this function never imports it.
    gate
        Registered gate name, such as ``"rx"``, ``"cnot"`` or ``"rzz"``.
    theta
        Torch rotation angle; unused by fixed gates.

    Returns
    -------
    Any
        Native complex128 operator of dimension two, four or eight.

    Raises
    ------
    ValueError
        If ``gate`` is not a registered Phase-QNode gate.

    """
    complex_dtype = torch_module.complex128

    def complex_tensor(values: object) -> Any:
        return torch_module.as_tensor(values, dtype=complex_dtype)

    def real_tensor(values: object) -> Any:
        return torch_module.as_tensor(values, dtype=torch_module.float64)

    def rotation(angle: Any, identity: Any, operator: Any) -> Any:
        return pauli_rotation(
            torch_module, angle, complex_tensor(1.0j), identity=identity, operator=operator
        )

    def phase(angle: Any) -> Any:
        return torch_phase_operator(
            torch_module,
            angle,
            complex_tensor(1.0 + 0.0j),
            complex_tensor(0.0 + 0.0j),
            complex_tensor(1.0j),
        )

    def controlled(target: Any) -> Any:
        return torch_module.stack(
            tuple(
                torch_controlled_operator_rows(torch_module, target, complex_tensor=complex_tensor)
            )
        )

    return dispatch_gate_operator(
        gate,
        theta,
        GateOperatorFactories(
            identity=lambda: torch_module.eye(2, dtype=complex_dtype),
            x=lambda: complex_tensor(pauli_operator_entries("x")),
            y=lambda: complex_tensor(pauli_operator_entries("y")),
            z=lambda: complex_tensor(pauli_operator_entries("z")),
            h=lambda: hadamard_operator(
                torch_module, real_tensor=real_tensor, complex_tensor=complex_tensor
            ),
            s=lambda: complex_tensor(phase_operator_entries(1.0j)),
            t=lambda: complex_tensor(phase_operator_entries(quarter_turn_phase(lambda: np))),
            sx=lambda: sqrt_x_operator(complex_tensor=complex_tensor),
            rotation=rotation,
            phase=phase,
            two_qubit=lambda name: complex_tensor(two_qubit_operator_entries(name)),
            controlled_z=lambda controls: controlled_z_operator(
                torch_module, controls, complex_tensor=complex_tensor
            ),
            controlled=lambda: controlled,
            ccnot=lambda: torch_basis_transposition(torch_module, 6, 7),
            cswap=lambda: torch_basis_transposition(torch_module, 5, 6),
            two_qubit_identity=lambda: torch_module.eye(4, dtype=complex_dtype),
            kron=lambda: torch_module.kron,
            gate=lambda name, angle: torch_gate_operator(torch_module, name, angle),
            unsupported=refuse_unregistered_gate("unsupported PyTorch Phase-QNode gate"),
        ),
    )


def jax_gate_operator(jnp: Any, gate: str, theta: Any) -> Any:
    """Return the complex128 JAX operator for one registered gate.

    Parameters
    ----------
    jnp
        Supplied ``jax.numpy`` namespace; this function never imports it.
    gate
        Registered gate name, such as ``"rx"``, ``"cnot"`` or ``"rzz"``.
    theta
        JAX rotation angle; unused by fixed gates.

    Returns
    -------
    Any
        Native complex128 operator of dimension two, four or eight.

    Raises
    ------
    ValueError
        If ``gate`` is not a registered Phase-QNode gate.

    """
    complex_dtype = jnp.complex128

    def complex_tensor(values: object) -> Any:
        return jnp.asarray(values, dtype=complex_dtype)

    def rotation(angle: Any, identity: Any, operator: Any) -> Any:
        return pauli_rotation(
            jnp, angle, complex_tensor(1.0j), identity=identity, operator=operator
        )

    def phase(angle: Any) -> Any:
        return jax_phase_operator(
            jnp,
            angle,
            complex_tensor(1.0 + 0.0j),
            complex_tensor(0.0 + 0.0j),
            complex_tensor(1.0j),
            dtype=complex_dtype,
        )

    return dispatch_gate_operator(
        gate,
        theta,
        GateOperatorFactories(
            identity=lambda: jnp.eye(2, dtype=complex_dtype),
            x=lambda: complex_tensor(pauli_operator_entries("x")),
            y=lambda: complex_tensor(pauli_operator_entries("y")),
            z=lambda: complex_tensor(pauli_operator_entries("z")),
            h=lambda: hadamard_operator(
                jnp, real_tensor=lambda values: jnp.asarray(values), complex_tensor=complex_tensor
            ),
            s=lambda: complex_tensor(phase_operator_entries(1.0j)),
            t=lambda: complex_tensor(phase_operator_entries(quarter_turn_phase(lambda: np))),
            sx=lambda: sqrt_x_operator(complex_tensor=complex_tensor),
            rotation=rotation,
            phase=phase,
            two_qubit=lambda name: complex_tensor(two_qubit_operator_entries(name)),
            controlled_z=lambda controls: controlled_z_operator(
                jnp, controls, complex_tensor=complex_tensor
            ),
            controlled=lambda: partial(jax_controlled_operator, jnp),
            ccnot=lambda: jax_basis_transposition(jnp, 6, 7),
            cswap=lambda: jax_basis_transposition(jnp, 5, 6),
            two_qubit_identity=lambda: jnp.eye(4, dtype=complex_dtype),
            kron=lambda: jnp.kron,
            gate=lambda name, angle: jax_gate_operator(jnp, name, angle),
            unsupported=refuse_unregistered_gate("unsupported JAX Phase-QNode gate"),
        ),
    )


__all__ = ["jax_gate_operator", "torch_gate_operator"]
