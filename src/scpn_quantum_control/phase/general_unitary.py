# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — general single-qubit unitary support for Phase-QNode
"""U3 and arbitrary single-qubit unitary support via a registered ZYZ decomposition.

The registered Phase-QNode gate set differentiates one trainable angle per
operation. A U3 gate (and any 2x2 unitary) is therefore expressed through the
exact Euler ``RZ(phi) RY(theta) RZ(lambda)`` decomposition into three registered
single-Pauli rotations, each of which carries the canonical two-term
parameter-shift rule. This keeps general-unitary coverage analytic and
fail-closed without enlarging the differentiable-gate primitive set.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .qnode_circuit import PhaseQNodeOperation

# Angle below which a Z rotation is treated as identity for the gamma = pi gauge.
_GIMBAL_TOL = 1e-9
ComplexArray: TypeAlias = NDArray[np.complex128]


def su2_zyz_angles(unitary: ComplexArray) -> tuple[float, float, float]:
    """Return ``(phi, theta, lam)`` with ``U ∝ RZ(phi) RY(theta) RZ(lam)``.

    The global phase is discarded (irrelevant to expectation values). The
    framework conventions are ``RZ(a) = diag(e^{-ia/2}, e^{ia/2})`` and
    ``RY(b) = [[cos b/2, -sin b/2], [sin b/2, cos b/2]]``.

    Args:
        unitary: a ``(2, 2)`` complex unitary matrix.

    Returns:
        The ZYZ Euler angles ``(phi, theta, lam)``.
    """
    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError("unitary must be a 2x2 matrix")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-8):
        raise ValueError("unitary must be unitary (U^dagger U = I)")

    # Strip the global phase so the matrix lies in SU(2).
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    special = matrix / np.sqrt(determinant)

    theta = 2.0 * float(np.arctan2(abs(special[1, 0]), abs(special[0, 0])))
    if abs(special[0, 0]) < _GIMBAL_TOL:
        # theta = pi: only phi - lam is fixed; pin lam = 0 (gauge choice).
        phi = 2.0 * float(np.angle(special[1, 0]))
        lam = 0.0
    elif abs(special[1, 0]) < _GIMBAL_TOL:
        # theta = 0: only phi + lam is fixed; pin lam = 0 (gauge choice).
        phi = -2.0 * float(np.angle(special[0, 0]))
        lam = 0.0
    else:
        angle_00 = float(np.angle(special[0, 0]))
        angle_10 = float(np.angle(special[1, 0]))
        phi = angle_10 - angle_00
        lam = -angle_10 - angle_00
    return phi, theta, lam


def build_u3_operations(
    qubit: int,
    parameter_indices: tuple[int, int, int],
) -> tuple[PhaseQNodeOperation, ...]:
    """Return a registered ``RZ·RY·RZ`` decomposition of a U3 gate on ``qubit``.

    ``parameter_indices`` are ``(theta_index, phi_index, lam_index)`` into the
    Phase-QNode parameter vector; the resulting operations implement
    ``U3(theta, phi, lam) ∝ RZ(phi) RY(theta) RZ(lam)`` up to a global phase, and
    each registered rotation differentiates with the exact two-term rule.

    Args:
        qubit: target qubit index (non-negative).
        parameter_indices: ``(theta_index, phi_index, lam_index)``, distinct and
            non-negative.

    Returns:
        The three registered operations in circuit order ``(RZ(lam), RY(theta),
        RZ(phi))``.
    """
    if not isinstance(qubit, int) or isinstance(qubit, bool) or qubit < 0:
        raise ValueError("qubit must be a non-negative integer")
    if len(parameter_indices) != 3:
        raise ValueError("parameter_indices must be (theta_index, phi_index, lam_index)")
    theta_index, phi_index, lam_index = parameter_indices
    for index in (theta_index, phi_index, lam_index):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("parameter indices must be non-negative integers")
    if len({theta_index, phi_index, lam_index}) != 3:
        raise ValueError("U3 parameter indices must be distinct")
    return (
        PhaseQNodeOperation("rz", (qubit,), parameter_index=lam_index),
        PhaseQNodeOperation("ry", (qubit,), parameter_index=theta_index),
        PhaseQNodeOperation("rz", (qubit,), parameter_index=phi_index),
    )
