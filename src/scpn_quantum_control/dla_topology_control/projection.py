# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fixed-active-set topology derivatives
"""Fail-closed JVP/VJP contracts around the existing topology ledger."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.topology_control.constraints import (
    HardwareEmbeddingConstraint,
    TopologyConstraintLedger,
    canonical_edge,
)

from .schema import (
    DLA_TOPOLOGY_CLAIM_BOUNDARY,
    ConstraintSupportRow,
    DifferentiabilityKind,
    DifferentiabilityReport,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _read_only_float(value: NDArray[np.float64]) -> FloatArray:
    out = np.array(value, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _square_finite(value: NDArray[np.float64], name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(matrix, dtype=np.float64, copy=True)


def _symmetric_zero_diagonal(value: FloatArray) -> FloatArray:
    result = np.asarray((value + value.T) / 2.0, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    return result


def _off_diagonal(value: FloatArray) -> FloatArray:
    return np.asarray(value[~np.eye(value.shape[0], dtype=bool)], dtype=np.float64)


def _near(values: FloatArray, boundary: float, margin: float) -> bool:
    return bool(np.any(np.abs(values - boundary) <= margin))


@dataclass(frozen=True, slots=True)
class TopologyProjectionDifferential:
    """One exact topology-ledger projection and its local JVP.

    Arrays are copied and read-only. ``projected_tangent`` is valid only for
    the fixed active set recorded by ``support``.

    Parameters
    ----------
    matrix:
        Finite square primal matrix copied into the record.
    tangent:
        Finite square tangent with the same shape.
    projected:
        Output of the production ``TopologyConstraintLedger.project`` call.
    projected_tangent:
        Exact local JVP for the recorded fixed active set.
    support:
        Fully supported derivative report for the exact primal point.
    content_digest:
        SHA-256 binding arrays and support decisions.
    claim_boundary:
        Limit on discrete, physical, and operational interpretation.

    """

    matrix: FloatArray
    tangent: FloatArray
    projected: FloatArray
    projected_tangent: FloatArray
    support: DifferentiabilityReport
    content_digest: str
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate array custody, support, digest, and claim boundary."""
        arrays = {
            "matrix": self.matrix,
            "tangent": self.tangent,
            "projected": self.projected,
            "projected_tangent": self.projected_tangent,
        }
        shape: tuple[int, ...] | None = None
        for name, value in arrays.items():
            checked = _square_finite(value, name)
            if shape is None:
                shape = checked.shape
            elif checked.shape != shape:
                raise ValueError("all topology differential arrays must have equal shape")
            object.__setattr__(self, name, _read_only_float(checked))
        self.support.require_supported()
        if len(self.content_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_digest
        ):
            raise ValueError("content_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())


def topology_projection_support(
    ledger: TopologyConstraintLedger,
    matrix: NDArray[np.float64],
    *,
    margin: float = 1.0e-8,
) -> DifferentiabilityReport:
    """Classify the exact active branch of ``TopologyConstraintLedger.project``.

    The report is deliberately conservative. Symmetry, masks, and frozen edges
    are affine. Sign and clipping branches are supported only away from kinks.
    Inactive total-weight intervals are supported; active rescaling and
    connectivity thresholds fail closed.

    Parameters
    ----------
    ledger:
        Existing production topology constraint ledger.
    matrix:
        Finite square primal point.
    margin:
        Positive distance used to reject branch boundaries.

    Returns
    -------
    DifferentiabilityReport
        Ordered decisions for symmetry, sign, bounds, masks, frozen edges,
        total-weight policy, and algebraic-connectivity policy.

    Raises
    ------
    ValueError
        If the ledger, matrix, margin, or configured reference is invalid.

    """
    if not isinstance(ledger, TopologyConstraintLedger):
        raise ValueError("ledger must be a TopologyConstraintLedger")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    primal = _square_finite(matrix, "matrix")
    symmetric = _symmetric_zero_diagonal(primal)
    rows: list[ConstraintSupportRow] = [
        ConstraintSupportRow(
            "symmetry_and_zero_diagonal",
            "supported",
            DifferentiabilityKind.LINEAR,
            "JVP and VJP use the self-adjoint symmetrisation/diagonal-removal map",
            "does not differentiate graph connectivity or persistent homology",
        )
    ]

    after_sign = symmetric.copy()
    if ledger.sign_policy == "signed":
        rows.append(
            ConstraintSupportRow(
                "sign_policy",
                "supported",
                DifferentiabilityKind.LINEAR,
                "signed policy is the identity on the symmetrised matrix",
                "support is local to this ledger policy",
            )
        )
    elif ledger.sign_policy == "nonnegative":
        values = _off_diagonal(symmetric)
        near_kink = bool(np.any(np.abs(values) <= margin))
        rows.append(
            ConstraintSupportRow(
                "sign_policy",
                "unsupported" if near_kink else "supported",
                DifferentiabilityKind.NON_SMOOTH
                if near_kink
                else DifferentiabilityKind.PIECEWISE_SMOOTH,
                "nonnegative maximum is at a zero-valued kink"
                if near_kink
                else "nonnegative maximum has a fixed positive/negative active branch",
                "no derivative is invented at a sign kink",
            )
        )
        after_sign = np.maximum(after_sign, 0.0)
    else:
        reference = ledger.fixed_sign_reference
        if reference is None:
            raise ValueError("fixed_sign policy requires fixed_sign_reference")
        fixed = _square_finite(reference, "fixed_sign_reference")
        if fixed.shape != symmetric.shape:
            raise ValueError("fixed_sign_reference must match matrix shape")
        relevant = (np.abs(fixed) > 0.0) & ~np.eye(fixed.shape[0], dtype=bool)
        near_kink = bool(np.any(np.abs(symmetric[relevant]) <= margin))
        rows.append(
            ConstraintSupportRow(
                "sign_policy",
                "unsupported" if near_kink else "supported",
                DifferentiabilityKind.NON_SMOOTH
                if near_kink
                else DifferentiabilityKind.PIECEWISE_SMOOTH,
                "fixed-sign absolute value is at a zero-valued kink"
                if near_kink
                else "fixed-sign absolute value has a fixed non-zero active branch",
                "no derivative is invented at an absolute-value kink",
            )
        )
        after_sign = np.sign(fixed) * np.abs(after_sign)

    off_sign = _off_diagonal(after_sign)
    lower = float(ledger.bounds.lower)
    upper = float(ledger.bounds.upper)
    bounds_kink = _near(off_sign, lower, margin) or _near(off_sign, upper, margin)
    rows.append(
        ConstraintSupportRow(
            "uniform_bounds",
            "unsupported" if bounds_kink else "supported",
            DifferentiabilityKind.NON_SMOOTH
            if bounds_kink
            else DifferentiabilityKind.PIECEWISE_SMOOTH,
            "clip is at a lower/upper-bound kink"
            if bounds_kink
            else "clip has a fixed interior/lower/upper active branch",
            "the derivative is invalid if the active bound changes",
        )
    )
    after_bounds = np.clip(after_sign, lower, upper)
    np.fill_diagonal(after_bounds, 0.0)

    rows.append(
        ConstraintSupportRow(
            "hardware_edge_mask",
            "supported",
            DifferentiabilityKind.LINEAR,
            "the fixed hardware mask is an elementwise linear projector",
            "changing the discrete edge set is not differentiated",
        )
    )
    rows.append(
        ConstraintSupportRow(
            "frozen_edges",
            "supported",
            DifferentiabilityKind.AFFINE,
            "frozen entries have zero tangent and fixed primal values",
            "changing edge identities or frozen values is not differentiated",
        )
    )

    projected_before_budget = after_bounds.copy()
    if ledger.hardware_edges is not None:
        mask = HardwareEmbeddingConstraint.from_edges(ledger.hardware_edges).mask(
            projected_before_budget.shape[0]
        )
        projected_before_budget *= mask
    for i, j in ledger.frozen_edges:
        edge = canonical_edge(i, j)
        value = float(ledger.frozen_edges[(i, j)])
        projected_before_budget[edge[0], edge[1]] = value
        projected_before_budget[edge[1], edge[0]] = value

    if ledger.total_weight is None:
        budget_supported = True
        budget_evidence = "no total-weight rescaling branch is configured"
    else:
        low, high = ledger.total_weight
        total = float(np.sum(projected_before_budget))
        budget_supported = low + margin < total < high - margin
        budget_evidence = (
            "total weight lies strictly inside the configured interval"
            if budget_supported
            else "total-weight projection is active or at an interval boundary"
        )
    rows.append(
        ConstraintSupportRow(
            "total_weight",
            "supported" if budget_supported else "unsupported",
            DifferentiabilityKind.LINEAR
            if budget_supported
            else DifferentiabilityKind.PIECEWISE_SMOOTH,
            budget_evidence,
            "active coupled rescaling requires a separately derived sensitivity rule",
        )
    )

    connectivity_supported = ledger.algebraic_connectivity_min == 0.0
    rows.append(
        ConstraintSupportRow(
            "algebraic_connectivity_threshold",
            "supported" if connectivity_supported else "unsupported",
            DifferentiabilityKind.NOT_APPLICABLE
            if connectivity_supported
            else DifferentiabilityKind.NON_SMOOTH,
            "no connectivity threshold participates in this projection"
            if connectivity_supported
            else "positive connectivity threshold is a violation check, not a projected smooth map",
            "eigenvalue degeneracies and graph-connectivity changes are not differentiated",
        )
    )
    ledger.project(primal)
    return DifferentiabilityReport(tuple(rows))


def _local_projection_derivative(
    ledger: TopologyConstraintLedger,
    matrix: FloatArray,
    tangent: FloatArray,
) -> FloatArray:
    symmetric = _symmetric_zero_diagonal(matrix)
    derivative = _symmetric_zero_diagonal(tangent)

    if ledger.sign_policy == "nonnegative":
        derivative *= symmetric > 0.0
        after_sign = np.maximum(symmetric, 0.0)
    elif ledger.sign_policy == "fixed_sign":
        reference = _square_finite(
            cast(NDArray[np.float64], ledger.fixed_sign_reference),
            "fixed_sign_reference",
        )
        derivative *= np.sign(reference) * np.sign(symmetric)
        after_sign = np.sign(reference) * np.abs(symmetric)
    else:
        after_sign = symmetric

    interior = (after_sign > ledger.bounds.lower) & (after_sign < ledger.bounds.upper)
    derivative *= interior
    np.fill_diagonal(derivative, 0.0)

    if ledger.hardware_edges is not None:
        mask = HardwareEmbeddingConstraint.from_edges(ledger.hardware_edges).mask(
            derivative.shape[0]
        )
        derivative *= mask
    for i, j in ledger.frozen_edges:
        edge = canonical_edge(i, j)
        derivative[edge[0], edge[1]] = 0.0
        derivative[edge[1], edge[0]] = 0.0
    return _symmetric_zero_diagonal(derivative)


def topology_projection_jvp(
    ledger: TopologyConstraintLedger,
    matrix: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    margin: float = 1.0e-8,
) -> TopologyProjectionDifferential:
    """Return the production-ledger projection and exact local JVP.

    Raises
    ------
    UnsupportedDifferentiableConstraintError
        If any exact active branch lacks a supported derivative rule.
    ValueError
        If matrices or ``margin`` violate their contracts.

    Returns
    -------
    TopologyProjectionDifferential
        Production forward projection, exact supported local JVP, immutable
        input custody, support report, and content digest.

    """
    primal = _square_finite(matrix, "matrix")
    direction = _square_finite(tangent, "tangent")
    if direction.shape != primal.shape:
        raise ValueError("tangent must match matrix shape")
    support = topology_projection_support(ledger, primal, margin=margin)
    support.require_supported()
    projected = ledger.project(primal)
    projected_tangent = _local_projection_derivative(ledger, primal, direction)
    digest = hashlib.sha256(
        primal.tobytes()
        + direction.tobytes()
        + projected.tobytes()
        + projected_tangent.tobytes()
        + support.content_digest.encode()
    ).hexdigest()
    return TopologyProjectionDifferential(
        matrix=primal,
        tangent=direction,
        projected=projected,
        projected_tangent=projected_tangent,
        support=support,
        content_digest=digest,
    )


def topology_projection_vjp(
    ledger: TopologyConstraintLedger,
    matrix: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    margin: float = 1.0e-8,
) -> FloatArray:
    """Apply the exact local adjoint Jacobian on a supported active branch.

    Parameters
    ----------
    ledger:
        Existing production topology constraint ledger.
    matrix:
        Finite square primal point at which the active set is classified.
    cotangent:
        Finite square output cotangent with the same shape as ``matrix``.
    margin:
        Positive refusal distance around sign and clipping boundaries.

    Returns
    -------
    numpy.ndarray
        Read-only input cotangent produced by the exact local adjoint rule.

    Raises
    ------
    UnsupportedDifferentiableConstraintError
        If any active operation has no supported derivative.
    ValueError
        If matrices, ledger configuration, or ``margin`` is invalid.

    """
    primal = _square_finite(matrix, "matrix")
    dual = _square_finite(cotangent, "cotangent")
    if dual.shape != primal.shape:
        raise ValueError("cotangent must match matrix shape")
    report = topology_projection_support(ledger, primal, margin=margin)
    report.require_supported()
    return _read_only_float(_local_projection_derivative(ledger, primal, dual))


__all__ = [
    "FloatArray",
    "TopologyProjectionDifferential",
    "topology_projection_jvp",
    "topology_projection_support",
    "topology_projection_vjp",
]
