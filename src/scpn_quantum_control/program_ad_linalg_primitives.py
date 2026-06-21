# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD linalg primitive rules
"""Static linear-algebra derivative rules and registry contracts for Program AD."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .differentiable_result_contracts import _normalise_claim_boundary
from .program_ad_array_indexing import _normalise_axis
from .program_ad_registry import (
    _PROGRAM_AD_LINALG_IDENTITIES,
    _PROGRAM_AD_LINALG_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .program_ad_shape_transforms import (
    _program_ad_float64_vector_result,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
)


def _is_program_ad_trace_value(value: object) -> bool:
    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _validate_program_ad_linalg_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    if contract.static_argument_rule is None:
        raise ValueError(
            f"program AD primitive {contract.identity.key} requires a static argument rule"
        )
    static_signature = contract.static_argument_rule(args)
    if not isinstance(static_signature, tuple):
        raise ValueError(
            f"program AD primitive {contract.identity.key} static rule must return a tuple"
        )
    if contract.shape_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} requires a shape rule")
    shape = contract.shape_rule(args)
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"program AD primitive {contract.identity.key} shape rule must return "
            "non-negative integer dimensions"
        )
    if contract.dtype_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} requires a dtype rule")
    dtype = contract.dtype_rule(args)
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(
            f"program AD primitive {contract.identity.key} dtype rule must return a dtype name"
        )


def _program_ad_linalg_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD linalg primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_linalg_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD linalg primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


@dataclass(frozen=True)
class ProgramADLinalgConditioningDiagnostic:
    """Conditioning report for numerically sensitive program-AD linalg primitives."""

    primitive: str
    shape: tuple[int, ...]
    status: str
    differentiability_ready: bool
    condition_number: float
    rank: int
    smallest_scale: float
    largest_scale: float
    minimum_gap: float | None
    threshold: float
    required_boundary: str
    message: str
    claim_boundary: str = (
        "program-AD linalg conditioning diagnostic only; no provider, hardware, "
        "native-framework, or production benchmark evidence is implied"
    )

    def __post_init__(self) -> None:
        if not self.primitive:
            raise ValueError("conditioning diagnostic primitive must be non-empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("conditioning diagnostic shape dimensions must be non-negative")
        if self.status not in {
            "well_conditioned",
            "ill_conditioned",
            "rank_deficient",
            "zero_norm_boundary",
        }:
            raise ValueError("conditioning diagnostic status is unsupported")
        for field_name, value in (
            ("condition_number", self.condition_number),
            ("smallest_scale", self.smallest_scale),
            ("largest_scale", self.largest_scale),
            ("threshold", self.threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"conditioning diagnostic {field_name} must be finite non-negative"
                )
        if self.minimum_gap is not None and (
            not math.isfinite(self.minimum_gap) or self.minimum_gap < 0.0
        ):
            raise ValueError("conditioning diagnostic minimum_gap must be finite non-negative")
        if self.rank < 0:
            raise ValueError("conditioning diagnostic rank must be non-negative")
        if not self.required_boundary:
            raise ValueError("conditioning diagnostic required_boundary must be non-empty")
        if not self.message:
            raise ValueError("conditioning diagnostic message must be non-empty")
        object.__setattr__(
            self,
            "claim_boundary",
            _normalise_claim_boundary("conditioning diagnostic", self.claim_boundary),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready representation."""

        return {
            "primitive": self.primitive,
            "shape": list(self.shape),
            "status": self.status,
            "differentiability_ready": self.differentiability_ready,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "smallest_scale": self.smallest_scale,
            "largest_scale": self.largest_scale,
            "minimum_gap": self.minimum_gap,
            "threshold": self.threshold,
            "required_boundary": self.required_boundary,
            "message": self.message,
            "claim_boundary": self.claim_boundary,
        }


_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES: Mapping[str, str] = {
    "norm": "non-zero norm for norm-gradient division",
    "det": "non-singular matrix away from determinant rank drop",
    "inv": "non-singular matrix inverse",
    "solve": "non-singular linear system with compatible right-hand side",
    "matrix_power": "non-singular matrix for negative powers",
    "eig": "real simple diagonalizable eigensystem",
    "eigh": "symmetric matrix with distinct eigenvalues",
    "eigvals": "real simple diagonalizable spectrum",
    "eigvalsh": "symmetric matrix with distinct eigenvalues",
    "svd": "distinct positive singular values",
    "pinv": "constant rank away from rank threshold crossing",
}


def _program_ad_linalg_conditioning_matrix(
    primitive: str,
    values: ArrayLike,
) -> NDArray[np.float64]:
    matrix = _as_real_numeric_array(f"program AD linalg {primitive} conditioning values", values)
    if matrix.ndim != 2:
        raise ValueError(f"program AD linalg {primitive} conditioning requires a rank-2 matrix")
    if 0 in matrix.shape:
        raise ValueError(f"program AD linalg {primitive} conditioning requires non-empty axes")
    return matrix


def _program_ad_linalg_condition_number(
    singular_values: NDArray[np.float64],
) -> tuple[float, float, float]:
    if singular_values.size == 0:
        return 0.0, 0.0, 0.0
    largest = float(np.max(singular_values))
    smallest = float(np.min(singular_values))
    if smallest == 0.0:
        return math.inf, smallest, largest
    return float(largest / smallest), smallest, largest


def _program_ad_linalg_minimum_gap(values: NDArray[np.float64]) -> float | None:
    if values.size < 2:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return float(np.min(np.diff(ordered)))


def _program_ad_linalg_diagnostic_from_singular_values(
    primitive: str,
    shape: tuple[int, ...],
    singular_values: NDArray[np.float64],
    *,
    condition_threshold: float,
    rank_tolerance: float,
    minimum_gap: float | None,
) -> ProgramADLinalgConditioningDiagnostic:
    condition_number, smallest, largest = _program_ad_linalg_condition_number(singular_values)
    rank = int(np.sum(singular_values > rank_tolerance))
    full_rank = rank == int(singular_values.size)
    if not full_rank:
        return ProgramADLinalgConditioningDiagnostic(
            primitive=primitive,
            shape=shape,
            status="rank_deficient",
            differentiability_ready=False,
            condition_number=0.0 if math.isinf(condition_number) else condition_number,
            rank=rank,
            smallest_scale=smallest,
            largest_scale=largest,
            minimum_gap=minimum_gap,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[primitive],
            message=(
                f"program AD linalg {primitive} is at a rank threshold boundary; "
                "the derivative contract remains fail-closed"
            ),
        )
    status = "ill_conditioned" if condition_number > condition_threshold else "well_conditioned"
    message = (
        f"program AD linalg {primitive} is ill-conditioned but remains differentiable "
        "inside the declared rank/spectrum boundary"
        if status == "ill_conditioned"
        else f"program AD linalg {primitive} conditioning is inside the declared boundary"
    )
    return ProgramADLinalgConditioningDiagnostic(
        primitive=primitive,
        shape=shape,
        status=status,
        differentiability_ready=True,
        condition_number=condition_number,
        rank=rank,
        smallest_scale=smallest,
        largest_scale=largest,
        minimum_gap=minimum_gap,
        threshold=condition_threshold,
        required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[primitive],
        message=message,
    )


def diagnose_program_ad_linalg_conditioning(
    primitive: str,
    values: ArrayLike,
    *,
    condition_threshold: float = 1.0e12,
    rank_tolerance: float = 1.0e-12,
) -> ProgramADLinalgConditioningDiagnostic:
    """Diagnose conditioning for supported norm and program-AD linalg primitives."""

    name = str(primitive).strip().lower()
    if name not in _PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES:
        raise ValueError(f"unsupported program AD linalg conditioning primitive {primitive!r}")
    if not math.isfinite(condition_threshold) or condition_threshold <= 0.0:
        raise ValueError("program AD linalg conditioning threshold must be positive and finite")
    if not math.isfinite(rank_tolerance) or rank_tolerance < 0.0:
        raise ValueError("program AD linalg rank tolerance must be finite non-negative")

    if name == "norm":
        array = _as_real_numeric_array("program AD linalg norm conditioning values", values)
        norm_value = float(np.linalg.norm(array.reshape(-1), ord=2))
        if norm_value <= rank_tolerance:
            return ProgramADLinalgConditioningDiagnostic(
                primitive=name,
                shape=tuple(int(dimension) for dimension in array.shape),
                status="zero_norm_boundary",
                differentiability_ready=False,
                condition_number=0.0,
                rank=0,
                smallest_scale=0.0,
                largest_scale=norm_value,
                minimum_gap=None,
                threshold=condition_threshold,
                required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
                message="program AD linalg norm is at the zero norm nondifferentiable boundary",
            )
        return ProgramADLinalgConditioningDiagnostic(
            primitive=name,
            shape=tuple(int(dimension) for dimension in array.shape),
            status="well_conditioned",
            differentiability_ready=True,
            condition_number=1.0,
            rank=1,
            smallest_scale=norm_value,
            largest_scale=norm_value,
            minimum_gap=None,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
            message="program AD linalg norm is away from the zero norm boundary",
        )

    matrix = _program_ad_linalg_conditioning_matrix(name, values)
    singular_values = np.linalg.svd(matrix, compute_uv=False).astype(np.float64)
    minimum_gap: float | None = None
    if name in {"eig", "eigvals"}:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        if np.max(np.abs(eigenvalues.imag)) > 1.0e-10:
            return ProgramADLinalgConditioningDiagnostic(
                primitive=name,
                shape=tuple(int(dimension) for dimension in matrix.shape),
                status="rank_deficient",
                differentiability_ready=False,
                condition_number=0.0,
                rank=int(np.linalg.matrix_rank(matrix, tol=rank_tolerance)),
                smallest_scale=0.0,
                largest_scale=float(np.max(np.abs(eigenvalues))),
                minimum_gap=None,
                threshold=condition_threshold,
                required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
                message=f"program AD linalg {name} conditioning requires real eigenvalues",
            )
        minimum_gap = _program_ad_linalg_minimum_gap(eigenvalues.real.astype(np.float64))
        singular_values = np.linalg.svd(eigenvectors.real, compute_uv=False).astype(np.float64)
    elif name in {"eigh", "eigvalsh"}:
        _program_ad_linalg_require_symmetric(name, matrix)
        eigenvalues = np.asarray(np.linalg.eigvalsh(matrix), dtype=np.float64)
        minimum_gap = _program_ad_linalg_minimum_gap(eigenvalues)
        singular_values = np.abs(eigenvalues).astype(np.float64)
    elif name == "svd":
        minimum_gap = _program_ad_linalg_minimum_gap(singular_values)

    if minimum_gap is not None and minimum_gap <= rank_tolerance:
        return ProgramADLinalgConditioningDiagnostic(
            primitive=name,
            shape=tuple(int(dimension) for dimension in matrix.shape),
            status="rank_deficient",
            differentiability_ready=False,
            condition_number=0.0,
            rank=int(np.sum(singular_values > rank_tolerance)),
            smallest_scale=float(np.min(singular_values)) if singular_values.size else 0.0,
            largest_scale=float(np.max(singular_values)) if singular_values.size else 0.0,
            minimum_gap=minimum_gap,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
            message=(
                f"program AD linalg {name} is at a repeated spectrum boundary; "
                "the derivative contract remains fail-closed"
            ),
        )

    return _program_ad_linalg_diagnostic_from_singular_values(
        name,
        tuple(int(dimension) for dimension in matrix.shape),
        singular_values,
        condition_threshold=condition_threshold,
        rank_tolerance=rank_tolerance,
        minimum_gap=minimum_gap,
    )


def _program_ad_linalg_square_matrix(
    primitive_name: str,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD linalg {primitive_name} values", values).reshape(
        -1
    )
    size = int(vector.size)
    rows = int(math.isqrt(size))
    if rows * rows != size:
        raise ValueError(
            f"program AD linalg {primitive_name} direct rule requires a flattened square matrix"
        )
    return vector.reshape(rows, rows)


def _program_ad_linalg_det_cofactor_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("program AD linalg det direct rule requires a square matrix")
    if rows == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if rows == 1:
        return np.ones((1, 1), dtype=np.float64)
    cofactors = np.zeros_like(matrix, dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
            cofactors[row, col] = ((-1.0) ** (row + col)) * float(np.linalg.det(minor))
    return cofactors


def _program_ad_linalg_det_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    return np.array([float(np.linalg.det(matrix))], dtype=np.float64)


def _program_ad_linalg_det_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    tangent_matrix = _program_ad_linalg_square_matrix("det", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg det tangent shape must match matrix shape")
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return np.array([float(np.sum(cofactors * tangent_matrix))], dtype=np.float64)


def _program_ad_linalg_scalar_cotangent(
    primitive_name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD linalg {primitive_name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD linalg {primitive_name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_linalg_det_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    scalar_cotangent = _program_ad_linalg_scalar_cotangent("det", cotangent)
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return _program_ad_float64_vector_result(scalar_cotangent * cofactors)


def _program_ad_linalg_inv_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    return np.linalg.inv(matrix).reshape(-1).astype(np.float64)


def _program_ad_linalg_inv_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    tangent_matrix = _program_ad_linalg_square_matrix("inv", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg inv tangent shape must match matrix shape")
    inverse = np.linalg.inv(matrix)
    return (-(inverse @ tangent_matrix @ inverse)).reshape(-1).astype(np.float64)


def _program_ad_linalg_inv_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    cotangent_matrix = _program_ad_linalg_square_matrix("inv cotangent", cotangent)
    if cotangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg inv VJP cotangent shape must match output shape")
    inverse = np.linalg.inv(matrix)
    return _program_ad_float64_vector_result(-(inverse.T @ cotangent_matrix @ inverse.T))


def _program_ad_linalg_solve_split(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD linalg {primitive_name} values", values).reshape(
        -1
    )
    total = int(vector.size)
    rows = int((math.isqrt(1 + 4 * total) - 1) // 2)
    if rows * rows + rows != total:
        raise ValueError(
            "program AD linalg solve direct rule requires flattened square matrix "
            "followed by vector right-hand side"
        )
    matrix = vector[: rows * rows].reshape(rows, rows)
    rhs = vector[rows * rows :]
    return matrix, rhs


def _program_ad_linalg_solve_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    return np.linalg.solve(matrix, rhs).astype(np.float64)


def _program_ad_linalg_solve_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    tangent_matrix, tangent_rhs = _program_ad_linalg_solve_split("solve", tangent)
    if tangent_matrix.shape != matrix.shape or tangent_rhs.shape != rhs.shape:
        raise ValueError("program AD linalg solve tangent shape must match primal shape")
    solution = np.linalg.solve(matrix, rhs)
    return np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution).astype(np.float64)


def _program_ad_linalg_solve_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg solve cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != rhs.shape:
        raise ValueError("program AD linalg solve VJP cotangent shape must match solution shape")
    solution = np.linalg.solve(matrix, rhs)
    rhs_adjoint = np.linalg.solve(matrix.T, cotangent_vector)
    matrix_adjoint = -np.outer(rhs_adjoint, solution)
    return _program_ad_float64_vector_result(
        np.concatenate((matrix_adjoint.reshape(-1), rhs_adjoint))
    )


def _program_ad_linalg_normalise_solve_shapes(
    matrix_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, ...]]:
    matrix = tuple(int(dimension) for dimension in matrix_shape)
    rhs = tuple(int(dimension) for dimension in rhs_shape)
    if len(matrix) != 2 or matrix[0] != matrix[1]:
        raise ValueError("program AD linalg solve direct rule requires a square matrix")
    if any(dimension < 0 for dimension in (*matrix, *rhs)):
        raise ValueError("program AD linalg solve direct rule requires non-negative dimensions")
    if len(rhs) not in {1, 2}:
        raise ValueError("program AD linalg solve direct rule requires rank-1 or rank-2 rhs")
    if rhs[0] != matrix[0]:
        raise ValueError(
            "program AD linalg solve direct rule right-hand side rows must match matrix"
        )
    return matrix, rhs


def _program_ad_linalg_solve_static_split(
    name: str,
    values: NDArray[np.float64],
    *,
    matrix_shape: tuple[int, int],
    rhs_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD linalg solve {name}", values).reshape(-1)
    matrix_size = _program_ad_shape_static_size(matrix_shape)
    rhs_size = _program_ad_shape_static_size(rhs_shape)
    if vector.size != matrix_size + rhs_size:
        raise ValueError(
            "program AD linalg solve direct rule requires flattened matrix followed by rhs"
        )
    return (
        vector[:matrix_size].reshape(matrix_shape),
        vector[matrix_size:].reshape(rhs_shape),
    )


def program_ad_linalg_solve_derivative_rule(
    matrix_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed solve primitive signature."""

    matrix_static_shape, rhs_static_shape = _program_ad_linalg_normalise_solve_shapes(
        matrix_shape, rhs_shape
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        return _program_ad_float64_vector_result(np.linalg.solve(matrix, rhs))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        tangent_matrix, tangent_rhs = _program_ad_linalg_solve_static_split(
            "tangent", tangent, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        solution = np.linalg.solve(matrix, rhs)
        return _program_ad_float64_vector_result(
            np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg solve cotangent", cotangent
        ).reshape(-1)
        rhs_size = _program_ad_shape_static_size(rhs_static_shape)
        if cotangent_vector.size != rhs_size:
            raise ValueError(
                "program AD linalg solve VJP cotangent shape must match solution shape"
            )
        cotangent_rhs = cotangent_vector.reshape(rhs_static_shape)
        solution = np.linalg.solve(matrix, rhs)
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent_rhs)
        if rhs_adjoint.ndim == 1:
            matrix_adjoint = -np.outer(rhs_adjoint, solution)
        else:
            matrix_adjoint = -(rhs_adjoint @ solution.T)
        return _program_ad_float64_vector_result(
            np.concatenate((matrix_adjoint.reshape(-1), rhs_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_solve_"
            f"{_program_ad_shape_signature(matrix_static_shape)}_rhs_"
            f"{_program_ad_shape_signature(rhs_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_matrix_power_derivative_rule(
    power: int | np.integer,
) -> CustomDerivativeRule:
    """Build a direct value/JVP rule for a fixed matrix-power primitive."""

    if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
        raise ValueError("program AD linalg matrix_power derivative rule requires integer power")
    exponent = int(power)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        return np.linalg.matrix_power(matrix, exponent).reshape(-1).astype(np.float64)

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        tangent_matrix = _program_ad_linalg_square_matrix("matrix_power", tangent)
        if tangent_matrix.shape != matrix.shape:
            raise ValueError(
                "program AD linalg matrix_power tangent shape must match matrix shape"
            )
        if exponent == 0:
            return np.zeros_like(matrix, dtype=np.float64).reshape(-1)
        if exponent > 0:
            total = np.zeros_like(matrix, dtype=np.float64)
            powers = [np.linalg.matrix_power(matrix, index) for index in range(exponent)]
            for index in range(exponent):
                total = total + powers[index] @ tangent_matrix @ powers[exponent - 1 - index]
            return total.reshape(-1).astype(np.float64)
        inverse = np.linalg.inv(matrix)
        inverse_tangent = -(inverse @ tangent_matrix @ inverse)
        positive_exponent = -exponent
        total = np.zeros_like(matrix, dtype=np.float64)
        powers = [np.linalg.matrix_power(inverse, index) for index in range(positive_exponent)]
        for index in range(positive_exponent):
            total = total + powers[index] @ inverse_tangent @ powers[positive_exponent - 1 - index]
        return total.reshape(-1).astype(np.float64)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        cotangent_matrix = _program_ad_linalg_square_matrix("matrix_power cotangent", cotangent)
        if cotangent_matrix.shape != matrix.shape:
            raise ValueError(
                "program AD linalg matrix_power VJP cotangent shape must match output shape"
            )
        if exponent == 0:
            return np.zeros_like(matrix, dtype=np.float64).reshape(-1)
        if exponent > 0:
            total = np.zeros_like(matrix, dtype=np.float64)
            powers = [np.linalg.matrix_power(matrix, index) for index in range(exponent)]
            for index in range(exponent):
                total = total + powers[index].T @ cotangent_matrix @ powers[exponent - 1 - index].T
            return total.reshape(-1).astype(np.float64)
        inverse = np.linalg.inv(matrix)
        positive_exponent = -exponent
        inverse_adjoint = np.zeros_like(matrix, dtype=np.float64)
        powers = [np.linalg.matrix_power(inverse, index) for index in range(positive_exponent)]
        for index in range(positive_exponent):
            inverse_adjoint = (
                inverse_adjoint
                + powers[index].T @ cotangent_matrix @ powers[positive_exponent - 1 - index].T
            )
        return _program_ad_float64_vector_result(-(inverse.T @ inverse_adjoint @ inverse.T))

    return CustomDerivativeRule(
        name=f"program_ad_linalg_matrix_power_{exponent}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _normalise_program_ad_linalg_multi_dot_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in operand_shapes)
    if len(shapes) < 2:
        raise ValueError(
            "program AD linalg multi_dot derivative rule requires at least two shapes"
        )
    for index, shape in enumerate(shapes):
        if len(shape) not in {1, 2}:
            raise ValueError("program AD linalg multi_dot derivative rule supports rank-1/rank-2")
        if any(dim <= 0 for dim in shape):
            raise ValueError(
                "program AD linalg multi_dot derivative rule dimensions must be positive"
            )
        if 0 < index < len(shapes) - 1 and len(shape) != 2:
            raise ValueError(
                "program AD linalg multi_dot derivative rule middle operands must be rank-2"
            )
    _program_ad_linalg_multi_dot_shape((tuple(np.zeros(shape) for shape in shapes),))
    return shapes


def _split_program_ad_linalg_multi_dot_operands(
    name: str,
    values: NDArray[np.float64],
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _as_real_numeric_array(f"program AD linalg multi_dot {name}", values).reshape(-1)
    expected_size = sum(int(np.prod(shape)) for shape in operand_shapes)
    if vector.size != expected_size:
        raise ValueError("program AD linalg multi_dot direct rule values size must match shapes")
    operands: list[NDArray[np.float64]] = []
    cursor = 0
    for shape in operand_shapes:
        size = int(np.prod(shape))
        operands.append(vector[cursor : cursor + size].reshape(shape))
        cursor += size
    return tuple(operands)


def _as_flat_multi_dot_result(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def program_ad_linalg_multi_dot_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build a direct value/JVP rule for a fixed multi-dot operand signature."""

    shapes = _normalise_program_ad_linalg_multi_dot_shapes(operand_shapes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        return _as_flat_multi_dot_result(np.linalg.multi_dot(operands))

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        tangent_operands = _split_program_ad_linalg_multi_dot_operands("tangent", tangent, shapes)
        total: NDArray[np.float64] | None = None
        for index, tangent_operand in enumerate(tangent_operands):
            varied = operands[:index] + (tangent_operand,) + operands[index + 1 :]
            contribution = _as_flat_multi_dot_result(np.linalg.multi_dot(varied))
            total = contribution if total is None else total + contribution
        if total is None:
            raise ValueError("program AD linalg multi_dot direct rule requires operands")
        return total.astype(np.float64)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        output = _as_flat_multi_dot_result(np.linalg.multi_dot(operands))
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg multi_dot cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.shape != output.shape:
            raise ValueError(
                "program AD linalg multi_dot VJP cotangent shape must match output shape"
            )
        adjoints: list[NDArray[np.float64]] = []
        for operand_index, operand in enumerate(operands):
            operand_adjoint = np.zeros_like(operand, dtype=np.float64)
            for element_index in np.ndindex(operand.shape):
                basis = np.zeros_like(operand, dtype=np.float64)
                basis[element_index] = 1.0
                varied = operands[:operand_index] + (basis,) + operands[operand_index + 1 :]
                contribution = _as_flat_multi_dot_result(np.linalg.multi_dot(varied))
                operand_adjoint[element_index] = float(np.dot(cotangent_vector, contribution))
            adjoints.append(operand_adjoint.reshape(-1))
        return _program_ad_float64_vector_result(np.concatenate(adjoints))

    signature = "x".join("_".join(str(dim) for dim in shape) for shape in shapes)
    return CustomDerivativeRule(
        name=f"program_ad_linalg_multi_dot_{signature}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_offset(name: str, offset: int | np.integer) -> int:
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError(f"program AD linalg {name} derivative rule requires integer offset")
    return int(offset)


def _program_ad_linalg_rank2_shape(
    name: str,
    source_shape: Sequence[int],
) -> tuple[int, int]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if len(shape) != 2:
        raise ValueError(f"program AD linalg {name} derivative rule requires a rank-2 matrix")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"program AD linalg {name} derivative rule dimensions must be positive")
    return shape


def _program_ad_linalg_trace_positions(
    matrix_shape: tuple[int, int],
    offset: int,
) -> tuple[tuple[int, int], ...]:
    rows, cols = matrix_shape
    positions = tuple((row, row + offset) for row in range(rows) if 0 <= row + offset < cols)
    if not positions:
        raise ValueError("program AD linalg trace offset selects an empty diagonal")
    return positions


def _program_ad_linalg_trace_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    return _program_ad_float64_vector_result([float(np.trace(matrix))])


def _program_ad_linalg_trace_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    tangent_matrix = _program_ad_linalg_square_matrix("trace", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg trace tangent shape must match matrix shape")
    return _program_ad_float64_vector_result([float(np.trace(tangent_matrix))])


def _program_ad_linalg_trace_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg trace cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != 1:
        raise ValueError("program AD linalg trace VJP cotangent must be scalar")
    return _program_ad_float64_vector_result(cotangent_vector[0] * np.eye(matrix.shape[0]))


def program_ad_linalg_trace_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    offset: int | np.integer = 0,
    axis1: int | np.integer = 0,
    axis2: int | np.integer = 1,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed trace primitive signature."""

    trace_axis1 = _program_ad_linalg_offset("trace", axis1)
    trace_axis2 = _program_ad_linalg_offset("trace", axis2)
    if (trace_axis1, trace_axis2) != (0, 1):
        raise ValueError("program AD linalg trace derivative rule supports axis1=0 and axis2=1")
    trace_offset = _program_ad_linalg_offset("trace", offset)
    static_shape = _program_ad_linalg_rank2_shape("trace", matrix_shape)
    positions = _program_ad_linalg_trace_positions(static_shape, trace_offset)
    size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg trace {name}", values).reshape(-1)
        if vector.size != size:
            raise ValueError("program AD linalg trace direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        return _program_ad_float64_vector_result(
            [sum(float(matrix[row, col]) for row, col in positions)]
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        del values
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            [sum(float(tangent_matrix[row, col]) for row, col in positions)]
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg trace cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != 1:
            raise ValueError("program AD linalg trace VJP cotangent must be scalar")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for row, col in positions:
            adjoint[row, col] += cotangent_vector[0]
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_trace_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{trace_offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_diag_positions(
    source_shape: tuple[int, ...],
    offset: int,
) -> tuple[tuple[int, int], ...]:
    if len(source_shape) == 1:
        size = source_shape[0] + abs(offset)
        positions = tuple(
            (index, index + offset) if offset >= 0 else (index - offset, index)
            for index in range(source_shape[0])
        )
        if any(row < 0 or row >= size or col < 0 or col >= size for row, col in positions):
            raise ValueError("program AD linalg diag offset is inconsistent with vector shape")
        return positions
    if len(source_shape) == 2:
        rows, cols = source_shape
        positions = tuple((row, row + offset) for row in range(rows) if 0 <= row + offset < cols)
        if not positions:
            raise ValueError("program AD linalg diag offset selects an empty diagonal")
        return positions
    raise ValueError("program AD linalg diag derivative rule requires rank-1 or rank-2 input")


def _program_ad_linalg_diag_shape_from_source(
    source_shape: tuple[int, ...],
    offset: int,
) -> tuple[int, ...]:
    positions = _program_ad_linalg_diag_positions(source_shape, offset)
    if len(source_shape) == 1:
        size = source_shape[0] + abs(offset)
        return (size, size)
    return (len(positions),)


def program_ad_linalg_diag_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: int | np.integer = 0,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed diagonal primitive signature."""

    static_shape = tuple(int(dimension) for dimension in source_shape)
    if len(static_shape) not in {1, 2}:
        raise ValueError("program AD linalg diag derivative rule requires rank-1 or rank-2 input")
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg diag derivative rule dimensions must be positive")
    offset = _program_ad_linalg_offset("diag", k)
    positions = _program_ad_linalg_diag_positions(static_shape, offset)
    source_size = _program_ad_shape_static_size(static_shape)
    output_shape = _program_ad_linalg_diag_shape_from_source(static_shape, offset)
    output_size = _program_ad_shape_static_size(output_shape)

    def split_source(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diag {name}", values).reshape(-1)
        if vector.size != source_size:
            raise ValueError("program AD linalg diag direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diag {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg diag VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = split_source("values", values)
        return _program_ad_float64_vector_result(np.diag(source, k=offset))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        del values
        source_tangent = split_source("tangent", tangent)
        return _program_ad_float64_vector_result(np.diag(source_tangent, k=offset))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split_source("values", values)
        cotangent_array = split_output("cotangent", cotangent)
        if len(static_shape) == 1:
            return _program_ad_float64_vector_result(np.diag(cotangent_array, k=offset))
        adjoint = np.zeros(static_shape, dtype=np.float64)
        cotangent_vector = cotangent_array.reshape(-1)
        for index, (row, col) in enumerate(positions):
            adjoint[row, col] += cotangent_vector[index]
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_diag_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_diagflat_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: int | np.integer = 0,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed diagflat primitive signature."""

    static_shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg diagflat derivative rule dimensions must be positive")
    source_size = _program_ad_shape_static_size(static_shape)
    if source_size <= 0:
        raise ValueError("program AD linalg diagflat derivative rule requires non-empty input")
    offset = _program_ad_linalg_offset("diagflat", k)
    output_shape = (source_size + abs(offset), source_size + abs(offset))
    output_size = _program_ad_shape_static_size(output_shape)

    def split_source(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diagflat {name}", values).reshape(-1)
        if vector.size != source_size:
            raise ValueError("program AD linalg diagflat direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diagflat {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg diagflat VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = split_source("values", values)
        return _program_ad_float64_vector_result(np.diagflat(source, k=offset))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        split_source("values", values)
        tangent_source = split_source("tangent", tangent)
        return _program_ad_float64_vector_result(np.diagflat(tangent_source, k=offset))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split_source("values", values)
        cotangent_matrix = split_output("cotangent", cotangent)
        adjoint_flat = np.diag(cotangent_matrix, k=offset)
        if adjoint_flat.size != source_size:
            raise ValueError("program AD linalg diagflat VJP diagonal size must match source")
        return _program_ad_float64_vector_result(adjoint_flat.reshape(static_shape))

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_diagflat_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_require_symmetric(
    primitive_name: str,
    matrix: NDArray[np.float64],
) -> None:
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"program AD linalg {primitive_name} requires a symmetric matrix")


def _program_ad_linalg_require_distinct_eigenvalues(
    eigenvalues: NDArray[np.float64],
    primitive_name: str,
) -> None:
    if eigenvalues.size <= 1:
        return
    gaps = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    strict_gaps = gaps[np.triu_indices(eigenvalues.size, k=1)]
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(strict_gaps)) <= 1.0e-10 * scale:
        raise ValueError(f"program AD linalg {primitive_name} requires distinct eigenvalues")


def _program_ad_linalg_real_simple_eig_decomposition_from_matrix(
    primitive_name: str,
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if matrix.ndim != 2:
        raise ValueError(f"program AD linalg {primitive_name} requires a rank-2 matrix")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"program AD linalg {primitive_name} requires a square matrix")
    eigenvalues_complex, right_complex = np.linalg.eig(matrix)
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues_complex))))
    tolerance = 1.0e-10 * eigenvalue_scale
    if float(np.max(np.abs(np.imag(eigenvalues_complex)))) > tolerance:
        raise ValueError(f"program AD linalg {primitive_name} requires real eigenvalues")
    eigenvalues = np.asarray(np.real(eigenvalues_complex), dtype=np.float64)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, primitive_name)
    eigenvector_scale = max(1.0, float(np.max(np.abs(right_complex))))
    if float(np.max(np.abs(np.imag(right_complex)))) > 1.0e-10 * eigenvector_scale:
        raise ValueError(f"program AD linalg {primitive_name} requires real eigenvectors")
    right_eigenvectors = np.asarray(np.real(right_complex), dtype=np.float64)
    try:
        left_eigenvector_rows = np.linalg.inv(right_eigenvectors)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"program AD linalg {primitive_name} requires a diagonalizable matrix"
        ) from exc
    condition = float(np.linalg.cond(right_eigenvectors))
    if not math.isfinite(condition) or condition > 1.0e10:
        raise ValueError(
            f"program AD linalg {primitive_name} requires a well-conditioned eigenbasis"
        )
    return eigenvalues, right_eigenvectors, left_eigenvector_rows.astype(np.float64)


def _program_ad_linalg_real_simple_eig_decomposition(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = _program_ad_linalg_square_matrix(primitive_name, values)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix(primitive_name, matrix)
    )
    return matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows


def _program_ad_linalg_eig_eigenvector_jvp_matrix(
    eigenvalues: NDArray[np.float64],
    right_eigenvectors: NDArray[np.float64],
    left_eigenvector_rows: NDArray[np.float64],
    tangent_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the real-simple eigenvector JVP matrix for one tangent direction."""

    size = eigenvalues.size
    tangent = np.zeros_like(right_eigenvectors, dtype=np.float64)
    for column in range(size):
        source: NDArray[np.float64] = np.asarray(right_eigenvectors[:, column], dtype=np.float64)
        raw_column: NDArray[np.float64] = np.zeros(size, dtype=np.float64)
        for other in range(size):
            if other == column:
                continue
            scale = float(left_eigenvector_rows[other, :] @ tangent_matrix @ source) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            raw_column = raw_column + scale * np.asarray(
                right_eigenvectors[:, other], dtype=np.float64
            )
        tangent[:, column] = raw_column - source * float(source.T @ raw_column)
    return tangent


def _program_ad_linalg_require_distinct_positive_singular_values(
    singular_values: NDArray[np.float64],
    primitive_name: str,
) -> None:
    if singular_values.size == 0:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )
    scale = max(1.0, float(np.max(np.abs(singular_values))))
    tolerance = 1.0e-10 * scale
    if float(np.min(singular_values)) <= tolerance:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )
    if singular_values.size <= 1:
        return
    gaps = np.abs(singular_values[:, None] - singular_values[None, :])
    strict_gaps = gaps[np.triu_indices(singular_values.size, k=1)]
    if float(np.min(strict_gaps)) <= tolerance:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )


def _program_ad_linalg_normalise_rcond(value: object) -> float:
    if value is None:
        return 1.0e-15
    if _is_program_ad_trace_value(value):
        raise ValueError("program AD linalg pinv rcond must be static")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("program AD linalg pinv rcond must be a static real scalar")
    cutoff = float(value)
    if cutoff < 0.0 or not math.isfinite(cutoff):
        raise ValueError("program AD linalg pinv rcond must be finite and non-negative")
    return cutoff


def _program_ad_linalg_require_constant_full_rank(
    matrix: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    *,
    rcond: float,
) -> None:
    rank = min(matrix.shape)
    if singular_values.size != rank:
        raise ValueError("program AD linalg pinv requires rank-2 singular values")
    scale = max(1.0, float(np.max(np.abs(singular_values))))
    threshold = rcond * scale
    if float(np.min(singular_values)) <= threshold:
        raise ValueError(
            "program AD linalg pinv requires a constant full-rank matrix above cutoff"
        )


def _program_ad_linalg_pinv_value_matrix(
    matrix: NDArray[np.float64],
    *,
    rcond: float = 1.0e-15,
) -> NDArray[np.float64]:
    if matrix.ndim != 2:
        raise ValueError("program AD linalg pinv requires a rank-2 matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("program AD linalg pinv requires non-empty matrix dimensions")
    _left, singular_values, _right_h = np.linalg.svd(matrix, full_matrices=False)
    _program_ad_linalg_require_constant_full_rank(matrix, singular_values, rcond=rcond)
    return np.linalg.pinv(matrix, rcond=rcond, hermitian=False).astype(np.float64)


def _program_ad_linalg_pinv_jvp_matrix(
    matrix: NDArray[np.float64],
    pinv: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    if tangent.shape != matrix.shape:
        raise ValueError("program AD linalg pinv tangent shape must match matrix shape")
    left_projector = np.eye(matrix.shape[1], dtype=np.float64) - pinv @ matrix
    right_projector = np.eye(matrix.shape[0], dtype=np.float64) - matrix @ pinv
    return cast(
        NDArray[np.float64],
        (
            -pinv @ tangent @ pinv
            + pinv @ pinv.T @ tangent.T @ right_projector
            + left_projector @ tangent.T @ pinv.T @ pinv
        ).astype(np.float64),
    )


def _program_ad_linalg_pinv_vjp_matrix(
    matrix: NDArray[np.float64],
    pinv: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    if cotangent.shape != pinv.shape:
        raise ValueError("program AD linalg pinv VJP cotangent shape must match output shape")
    left_projector = np.eye(matrix.shape[1], dtype=np.float64) - pinv @ matrix
    right_projector = np.eye(matrix.shape[0], dtype=np.float64) - matrix @ pinv
    return cast(
        NDArray[np.float64],
        (
            -pinv.T @ cotangent @ pinv.T
            + right_projector.T @ cotangent.T @ pinv @ pinv.T
            + pinv.T @ pinv @ cotangent.T @ left_projector.T
        ).astype(np.float64),
    )


def _program_ad_linalg_pinv_square_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    return _program_ad_float64_vector_result(_program_ad_linalg_pinv_value_matrix(matrix))


def _program_ad_linalg_pinv_square_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    tangent_matrix = _program_ad_linalg_square_matrix("pinv tangent", tangent)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix)
    return _program_ad_float64_vector_result(
        _program_ad_linalg_pinv_jvp_matrix(matrix, pinv, tangent_matrix)
    )


def _program_ad_linalg_pinv_square_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix)
    cotangent_matrix = _as_real_numeric_array("program AD linalg pinv cotangent", cotangent)
    if cotangent_matrix.size != pinv.size:
        raise ValueError("program AD linalg pinv VJP cotangent size must match output")
    return _program_ad_float64_vector_result(
        _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent_matrix.reshape(pinv.shape))
    )


def _program_ad_linalg_uplo(
    value: object,
    primitive_name: str,
) -> Literal["L", "U"]:
    uplo_value = str(value).upper()
    if uplo_value not in {"L", "U"}:
        raise ValueError(f"program AD linalg {primitive_name} requires UPLO='L' or UPLO='U'")
    return cast(Literal["L", "U"], uplo_value)


def _program_ad_linalg_eigvalsh_decomposition(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    uplo: str = "L",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = _program_ad_linalg_square_matrix(primitive_name, values)
    _program_ad_linalg_require_symmetric(primitive_name, matrix)
    uplo_value = _program_ad_linalg_uplo(uplo, primitive_name)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, primitive_name)
    return matrix, eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)


def _program_ad_linalg_eigvalsh_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, _eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    return _program_ad_float64_vector_result(eigenvalues)


def _program_ad_linalg_eigvals_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, _right_eigenvectors, _left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    return _program_ad_float64_vector_result(eigenvalues)


def _program_ad_linalg_eigvals_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eigvals tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigvals tangent shape must match matrix shape")
    return _program_ad_float64_vector_result(
        [
            float(left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index])
            for index in range(matrix.shape[0])
        ]
    )


def _program_ad_linalg_eigvals_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigvals cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != matrix.shape[0]:
        raise ValueError("program AD linalg eigvals VJP cotangent size must match spectrum")
    adjoint = np.zeros_like(matrix, dtype=np.float64)
    for index, weight in enumerate(cotangent_vector):
        adjoint = adjoint + float(weight) * np.outer(
            left_eigenvector_rows[index, :], right_eigenvectors[:, index]
        )
    return _program_ad_float64_vector_result(adjoint)


def _program_ad_linalg_eig_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, right_eigenvectors, _left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eig", values)
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalues, right_eigenvectors.reshape(-1)))
    )


def _program_ad_linalg_eig_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eig", values)
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eig tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eig tangent shape must match matrix shape")
    eigenvalue_tangent = np.array(
        [
            float(left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index])
            for index in range(matrix.shape[0])
        ],
        dtype=np.float64,
    )
    eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
        eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
    )


def _program_ad_linalg_eig_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("eig", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eig cotangent", cotangent
    ).reshape(-1)
    output_size = matrix.shape[0] + matrix.size
    if cotangent_vector.size != output_size:
        raise ValueError("program AD linalg eig VJP cotangent size must match output")
    adjoint = np.zeros_like(matrix, dtype=np.float64)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            basis = np.zeros_like(matrix, dtype=np.float64)
            basis[row, col] = 1.0
            jvp = _program_ad_linalg_eig_jvp(values, basis.reshape(-1))
            adjoint[row, col] = float(jvp @ cotangent_vector)
    return _program_ad_float64_vector_result(adjoint)


def program_ad_linalg_eig_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed real-simple eig primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eig", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eig derivative rule requires a square matrix")
    matrix_size = _program_ad_shape_static_size(static_shape)
    output_size = static_shape[0] + matrix_size

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eig {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eig direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        eigenvalues, right_eigenvectors, _left_eigenvector_rows = (
            _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalues, right_eigenvectors.reshape(-1)))
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        tangent_matrix = split("tangent", tangent)
        eigenvalues, right_eigenvectors, left_eigenvector_rows = (
            _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
        )
        eigenvalue_tangent = np.array(
            [
                float(
                    left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index]
                )
                for index in range(static_shape[0])
            ],
            dtype=np.float64,
        )
        eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
            eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eig cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg eig VJP cotangent size must match output")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for row in range(static_shape[0]):
            for col in range(static_shape[1]):
                basis = np.zeros(static_shape, dtype=np.float64)
                basis[row, col] = 1.0
                adjoint[row, col] = float(jvp_rule(values, basis.reshape(-1)) @ cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(f"program_ad_linalg_eig_{_program_ad_shape_signature(static_shape)}_direct_rule"),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_eigvals_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed real-simple eigvals primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigvals", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigvals derivative rule requires a square matrix")
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigvals {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigvals direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        return _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eigvals", matrix)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        eigenvalues, _right_eigenvectors, _left_eigenvector_rows = decompose("values", values)
        return _program_ad_float64_vector_result(eigenvalues)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _eigenvalues, right_eigenvectors, left_eigenvector_rows = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            [
                float(
                    left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index]
                )
                for index in range(static_shape[0])
            ]
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _eigenvalues, right_eigenvectors, left_eigenvector_rows = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigvals cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != static_shape[0]:
            raise ValueError("program AD linalg eigvals VJP cotangent size must match spectrum")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for index, weight in enumerate(cotangent_vector):
            adjoint = adjoint + float(weight) * np.outer(
                left_eigenvector_rows[index, :], right_eigenvectors[:, index]
            )
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_eigvals_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_eigvalsh_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eigvalsh tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigvalsh tangent shape must match matrix shape")
    _program_ad_linalg_require_symmetric("eigvalsh tangent", tangent_matrix)
    return _program_ad_float64_vector_result(
        np.array(
            [
                float(eigenvector.T @ tangent_matrix @ eigenvector)
                for eigenvector in eigenvectors.T
            ],
            dtype=np.float64,
        )
    )


def _program_ad_linalg_eigvalsh_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigvalsh cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != eigenvalues.size:
        raise ValueError("program AD linalg eigvalsh VJP cotangent size must match eigenvalues")
    adjoint = eigenvectors @ np.diag(cotangent_vector) @ eigenvectors.T
    return _program_ad_float64_vector_result(adjoint)


def _program_ad_linalg_eigh_eigenvector_jvp_matrix(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    tangent_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = eigenvalues.size
    tangent = np.zeros_like(eigenvectors, dtype=np.float64)
    for column in range(size):
        for other in range(size):
            if other == column:
                continue
            scale = float(
                eigenvectors[:, other].T @ tangent_matrix @ eigenvectors[:, column]
            ) / float(eigenvalues[column] - eigenvalues[other])
            tangent[:, column] = tangent[:, column] + scale * eigenvectors[:, other]
    return tangent


def _program_ad_linalg_eigh_vjp_matrix(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    eigenvalue_cotangent: NDArray[np.float64],
    eigenvector_cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    adjoint = eigenvectors @ np.diag(eigenvalue_cotangent) @ eigenvectors.T
    size = eigenvalues.size
    for column in range(size):
        cotangent_column = eigenvector_cotangent[:, column]
        for other in range(size):
            if other == column:
                continue
            scale = float(eigenvectors[:, other].T @ cotangent_column) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            adjoint = adjoint + scale * np.outer(eigenvectors[:, other], eigenvectors[:, column])
    symmetric_adjoint = 0.5 * (adjoint + adjoint.T)
    return cast(NDArray[np.float64], symmetric_adjoint.astype(np.float64))


def _program_ad_linalg_eigh_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalues, eigenvectors.reshape(-1)))
    )


def _program_ad_linalg_eigh_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    tangent_matrix = _program_ad_linalg_square_matrix("eigh tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigh tangent shape must match matrix shape")
    _program_ad_linalg_require_symmetric("eigh tangent", tangent_matrix)
    eigenvalue_tangent = np.array(
        [float(eigenvector.T @ tangent_matrix @ eigenvector) for eigenvector in eigenvectors.T],
        dtype=np.float64,
    )
    eigenvector_tangent = _program_ad_linalg_eigh_eigenvector_jvp_matrix(
        eigenvalues, eigenvectors, tangent_matrix
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
    )


def _program_ad_linalg_eigh_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigh cotangent", cotangent
    ).reshape(-1)
    size = matrix.shape[0]
    output_size = size + size * size
    if cotangent_vector.size != output_size:
        raise ValueError("program AD linalg eigh VJP cotangent size must match output")
    eigenvalue_cotangent = cotangent_vector[:size]
    eigenvector_cotangent = cotangent_vector[size:].reshape(size, size)
    return _program_ad_float64_vector_result(
        _program_ad_linalg_eigh_vjp_matrix(
            eigenvalues, eigenvectors, eigenvalue_cotangent, eigenvector_cotangent
        )
    )


def program_ad_linalg_eigh_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    uplo: str = "L",
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed symmetric eigh primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigh", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigh derivative rule requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "eigh derivative rule")
    matrix_size = _program_ad_shape_static_size(static_shape)
    output_size = static_shape[0] + matrix_size

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigh {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigh direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        _program_ad_linalg_require_symmetric("eigh", matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigh")
        return matrix, eigenvalues, eigenvectors

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalues, eigenvectors.reshape(-1)))
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        _program_ad_linalg_require_symmetric("eigh tangent", tangent_matrix)
        eigenvalue_tangent = np.array(
            [
                float(eigenvector.T @ tangent_matrix @ eigenvector)
                for eigenvector in eigenvectors.T
            ],
            dtype=np.float64,
        )
        eigenvector_tangent = _program_ad_linalg_eigh_eigenvector_jvp_matrix(
            eigenvalues, eigenvectors, tangent_matrix
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigh cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg eigh VJP cotangent size must match output")
        eigenvalue_cotangent = cotangent_vector[: static_shape[0]]
        eigenvector_cotangent = cotangent_vector[static_shape[0] :].reshape(static_shape)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_eigh_vjp_matrix(
                eigenvalues, eigenvectors, eigenvalue_cotangent, eigenvector_cotangent
            )
        )

    return CustomDerivativeRule(
        name=(f"program_ad_linalg_eigh_{_program_ad_shape_signature(static_shape)}_direct_rule"),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_eigvalsh_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    uplo: str = "L",
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed symmetric eigvalsh primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigvalsh", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigvalsh derivative rule requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "eigvalsh derivative rule")
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigvalsh {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigvalsh direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        eigenvalues, _eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        return _program_ad_float64_vector_result(eigenvalues)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        tangent_matrix = split("tangent", tangent)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        _program_ad_linalg_require_symmetric("eigvalsh tangent", tangent_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        return _program_ad_float64_vector_result(
            np.array(
                [
                    float(eigenvector.T @ tangent_matrix @ eigenvector)
                    for eigenvector in eigenvectors.T
                ],
                dtype=np.float64,
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = split("values", values)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigvalsh cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != static_shape[0]:
            raise ValueError(
                "program AD linalg eigvalsh VJP cotangent size must match eigenvalues"
            )
        return _program_ad_float64_vector_result(
            eigenvectors @ np.diag(cotangent_vector) @ eigenvectors.T
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_eigvalsh_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_svdvals_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed-shape SVD singular values."""

    static_shape = _program_ad_linalg_rank2_shape("svd", matrix_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg svd derivative rule requires positive dimensions")
    output_size = min(static_shape)
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg svd {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg svd direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str,
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
        _program_ad_linalg_require_distinct_positive_singular_values(singular_values, "svd")
        return matrix, left, singular_values, right_h

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, _left, singular_values, _right_h = decompose("values", values)
        return _program_ad_float64_vector_result(singular_values)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, left, _singular_values, right_h = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            np.array(
                [
                    float(left[:, index].T @ tangent_matrix @ right_h[index, :])
                    for index in range(output_size)
                ],
                dtype=np.float64,
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _matrix, left, _singular_values, right_h = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg svd cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg svd VJP cotangent size must match singular values")
        return _program_ad_float64_vector_result(left @ np.diag(cotangent_vector) @ right_h)

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_svdvals_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_pinv_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    rcond: float | None = None,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed-shape full-rank pseudoinverse."""

    static_shape = _program_ad_linalg_rank2_shape("pinv", matrix_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg pinv derivative rule requires positive dimensions")
    cutoff = _program_ad_linalg_normalise_rcond(rcond)
    output_shape = (static_shape[1], static_shape[0])
    input_size = _program_ad_shape_static_size(static_shape)
    output_size = _program_ad_shape_static_size(output_shape)

    def split_input(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg pinv {name}", values).reshape(-1)
        if vector.size != input_size:
            raise ValueError("program AD linalg pinv direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg pinv {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg pinv VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        tangent_matrix = split_input("tangent", tangent)
        pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_jvp_matrix(matrix, pinv, tangent_matrix)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        cotangent_matrix = split_output("cotangent", cotangent)
        pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent_matrix)
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_pinv_"
            f"{_program_ad_shape_signature(static_shape)}_rcond_{cutoff:.3e}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "det":
        return CustomDerivativeRule(
            name="program_ad_linalg_det_direct_rule",
            value_fn=_program_ad_linalg_det_value,
            jvp_rule=_program_ad_linalg_det_jvp,
            vjp_rule=_program_ad_linalg_det_vjp,
        )
    if name == "inv":
        return CustomDerivativeRule(
            name="program_ad_linalg_inv_direct_rule",
            value_fn=_program_ad_linalg_inv_value,
            jvp_rule=_program_ad_linalg_inv_jvp,
            vjp_rule=_program_ad_linalg_inv_vjp,
        )
    if name == "solve":
        return CustomDerivativeRule(
            name="program_ad_linalg_solve_direct_rule",
            value_fn=_program_ad_linalg_solve_value,
            jvp_rule=_program_ad_linalg_solve_jvp,
            vjp_rule=_program_ad_linalg_solve_vjp,
        )
    if name == "trace":
        return CustomDerivativeRule(
            name="program_ad_linalg_trace_direct_rule",
            value_fn=_program_ad_linalg_trace_value,
            jvp_rule=_program_ad_linalg_trace_jvp,
            vjp_rule=_program_ad_linalg_trace_vjp,
        )
    if name == "eig":
        return CustomDerivativeRule(
            name="program_ad_linalg_eig_direct_rule",
            value_fn=_program_ad_linalg_eig_value,
            jvp_rule=_program_ad_linalg_eig_jvp,
            vjp_rule=_program_ad_linalg_eig_vjp,
        )
    if name == "eigh":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigh_direct_rule",
            value_fn=_program_ad_linalg_eigh_value,
            jvp_rule=_program_ad_linalg_eigh_jvp,
            vjp_rule=_program_ad_linalg_eigh_vjp,
        )
    if name == "eigvalsh":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigvalsh_direct_rule",
            value_fn=_program_ad_linalg_eigvalsh_value,
            jvp_rule=_program_ad_linalg_eigvalsh_jvp,
            vjp_rule=_program_ad_linalg_eigvalsh_vjp,
        )
    if name == "eigvals":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigvals_direct_rule",
            value_fn=_program_ad_linalg_eigvals_value,
            jvp_rule=_program_ad_linalg_eigvals_jvp,
            vjp_rule=_program_ad_linalg_eigvals_vjp,
        )
    if name == "pinv":
        return CustomDerivativeRule(
            name="program_ad_linalg_pinv_square_direct_rule",
            value_fn=_program_ad_linalg_pinv_square_value,
            jvp_rule=_program_ad_linalg_pinv_square_jvp,
            vjp_rule=_program_ad_linalg_pinv_square_vjp,
        )
    return CustomDerivativeRule(
        name=f"program_ad_linalg_{name}_trace_contract",
        value_fn=_program_ad_linalg_direct_value,
        jvp_rule=_program_ad_linalg_direct_jvp,
    )


def _program_ad_linalg_shape_of(value: object) -> tuple[int, ...]:
    if _is_program_ad_trace_value(value):
        raw_shape = cast(Any, value).shape
        return tuple(int(dimension) for dimension in raw_shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_linalg_require_matrix_shape(name: str, value: object) -> tuple[int, int]:
    shape = _program_ad_linalg_shape_of(value)
    if len(shape) != 2:
        raise ValueError(f"program AD linalg {name} shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows != cols:
        raise ValueError(f"program AD linalg {name} shape rule requires a square matrix")
    return rows, cols


def _program_ad_linalg_det_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg det shape rule requires one matrix")
    _program_ad_linalg_require_matrix_shape("det", args[0])
    return ()


def _program_ad_linalg_inv_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg inv shape rule requires one matrix")
    return _program_ad_linalg_require_matrix_shape("inv", args[0])


def _program_ad_linalg_solve_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg solve shape rule requires matrix and right-hand side")
    rows, _cols = _program_ad_linalg_require_matrix_shape("solve", args[0])
    rhs_shape = _program_ad_linalg_shape_of(args[1])
    if len(rhs_shape) == 1:
        if rhs_shape[0] != rows:
            raise ValueError("program AD linalg solve shape rule vector length must match matrix")
        return rhs_shape
    if len(rhs_shape) == 2:
        if rhs_shape[0] != rows:
            raise ValueError("program AD linalg solve shape rule rhs rows must match matrix")
        return rhs_shape
    raise ValueError("program AD linalg solve shape rule requires rank-1 or rank-2 rhs")


def _program_ad_linalg_trace_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 4}:
        raise ValueError(
            "program AD linalg trace shape rule requires matrix and optional static axes"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg trace shape rule requires a rank-2 matrix")
    offset = 0
    axis1 = 0
    axis2 = 1
    if len(args) == 4:
        offset = _program_ad_linalg_offset("trace", cast(int | np.integer, args[1]))
        axis1 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[2]))
        axis2 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[3]))
    if (axis1, axis2) != (0, 1):
        raise ValueError("program AD linalg trace shape rule supports axis1=0 and axis2=1")
    _program_ad_linalg_trace_positions(shape, offset)
    return ()


def _program_ad_linalg_diag_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD linalg diag shape rule requires source and optional offset")
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diag", cast(int | np.integer, args[1]))
    if len(shape) not in {1, 2}:
        raise ValueError("program AD linalg diag shape rule requires rank-1 or rank-2 input")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("program AD linalg diag shape rule dimensions must be positive")
    return _program_ad_linalg_diag_shape_from_source(shape, offset)


def _program_ad_linalg_diagflat_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD linalg diagflat shape rule requires source and optional offset"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diagflat", cast(int | np.integer, args[1]))
    source_size = int(np.prod(shape))
    if source_size <= 0:
        raise ValueError("program AD linalg diagflat shape rule requires non-empty input")
    output_size = source_size + abs(offset)
    return (output_size, output_size)


def _program_ad_linalg_matrix_power_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg matrix_power shape rule requires matrix and power")
    if isinstance(args[1], bool) or not isinstance(args[1], (int, np.integer)):
        raise ValueError(
            "program AD linalg matrix_power shape rule requires a static integer power"
        )
    return _program_ad_linalg_require_matrix_shape("matrix_power", args[0])


def _program_ad_linalg_multi_dot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg multi_dot shape rule requires one operand sequence")
    operands = args[0]
    if (
        _is_program_ad_trace_value(operands)
        or isinstance(operands, np.ndarray)
        or not isinstance(operands, Sequence)
    ):
        raise ValueError(
            "program AD linalg multi_dot shape rule requires a static operand sequence"
        )
    shapes = tuple(_program_ad_linalg_shape_of(operand) for operand in operands)
    if len(shapes) < 2:
        raise ValueError("program AD linalg multi_dot shape rule requires at least two operands")
    for index, shape in enumerate(shapes):
        if len(shape) not in {1, 2}:
            raise ValueError(
                "program AD linalg multi_dot shape rule supports rank-1 and rank-2 operands"
            )
        if 0 < index < len(shapes) - 1 and len(shape) != 2:
            raise ValueError(
                "program AD linalg multi_dot shape rule middle operands must be rank-2"
            )

    result_shape = shapes[0]
    for next_shape in shapes[1:]:
        if len(result_shape) == 1 and len(next_shape) == 1:
            if result_shape[0] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = ()
        elif len(result_shape) == 1 and len(next_shape) == 2:
            if result_shape[0] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (next_shape[1],)
        elif len(result_shape) == 2 and len(next_shape) == 1:
            if result_shape[1] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (result_shape[0],)
        elif len(result_shape) == 2 and len(next_shape) == 2:
            if result_shape[1] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (result_shape[0], next_shape[1])
        else:
            raise ValueError(
                "program AD linalg multi_dot shape rule encountered scalar intermediate"
            )
    return result_shape


def _program_ad_linalg_eigh_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigh shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigh", args[0])
    return (rows, rows, rows)


def _program_ad_linalg_eig_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eig shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eig", args[0])
    return (rows, rows, rows)


def _program_ad_linalg_eigvalsh_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigvalsh shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigvalsh", args[0])
    return (rows,)


def _program_ad_linalg_eigvals_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigvals shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigvals", args[0])
    return (rows,)


def _program_ad_linalg_svd_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg svd shape rule requires one matrix")
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg svd shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD linalg svd shape rule requires non-empty dimensions")
    return (min(rows, cols),)


def _program_ad_linalg_pinv_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg pinv shape rule requires one matrix")
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg pinv shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD linalg pinv shape rule requires non-empty dimensions")
    return (cols, rows)


def _program_ad_linalg_dtype_rule(args: tuple[object, ...]) -> str:
    arrays: list[NDArray[np.float64]] = []
    for arg in args:
        if _is_program_ad_trace_value(arg):
            continue
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, np.ndarray)):
            for item in arg:
                if _is_program_ad_trace_value(item):
                    continue
                arrays.append(_as_real_numeric_array("program AD linalg dtype operand", item))
        elif isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
            continue
        else:
            arrays.append(_as_real_numeric_array("program AD linalg dtype operand", arg))
    if not arrays:
        return "float64"
    return str(np.result_type(*(array.dtype for array in arrays)))


def _program_ad_linalg_no_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    del args
    return ()


def _program_ad_linalg_matrix_power_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg matrix_power static rule requires matrix and power")
    power = args[1]
    if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
        raise ValueError("program AD linalg matrix_power static rule requires an integer power")
    return (int(power),)


def _program_ad_linalg_trace_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 4}:
        raise ValueError(
            "program AD linalg trace static rule requires matrix and optional static axes"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg trace static rule requires a rank-2 matrix")
    offset = 0
    axis1 = 0
    axis2 = 1
    if len(args) == 4:
        offset = _program_ad_linalg_offset("trace", cast(int | np.integer, args[1]))
        axis1 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[2]))
        axis2 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[3]))
    if (axis1, axis2) != (0, 1):
        raise ValueError("program AD linalg trace static rule supports axis1=0 and axis2=1")
    _program_ad_linalg_trace_positions(shape, offset)
    return (shape, offset, axis1, axis2)


def _program_ad_linalg_diag_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD linalg diag static rule requires source and optional offset")
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diag", cast(int | np.integer, args[1]))
    if len(shape) not in {1, 2}:
        raise ValueError("program AD linalg diag static rule requires rank-1 or rank-2 input")
    _program_ad_linalg_diag_shape_from_source(shape, offset)
    return (shape, offset)


def _program_ad_linalg_diagflat_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD linalg diagflat static rule requires source and optional offset"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diagflat", cast(int | np.integer, args[1]))
    if int(np.prod(shape)) <= 0:
        raise ValueError("program AD linalg diagflat static rule requires non-empty input")
    return (shape, offset)


def _program_ad_linalg_multi_dot_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg multi_dot static rule requires one operand sequence")
    operands = args[0]
    if (
        _is_program_ad_trace_value(operands)
        or isinstance(operands, np.ndarray)
        or not isinstance(operands, Sequence)
    ):
        raise ValueError(
            "program AD linalg multi_dot static rule requires a static operand sequence"
        )
    shapes = tuple(_program_ad_linalg_shape_of(operand) for operand in operands)
    if len(shapes) < 2:
        raise ValueError("program AD linalg multi_dot static rule requires at least two operands")
    return shapes


_PROGRAM_AD_LINALG_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "det": _program_ad_linalg_det_shape,
    "inv": _program_ad_linalg_inv_shape,
    "solve": _program_ad_linalg_solve_shape,
    "trace": _program_ad_linalg_trace_shape,
    "diag": _program_ad_linalg_diag_shape,
    "diagflat": _program_ad_linalg_diagflat_shape,
    "matrix_power": _program_ad_linalg_matrix_power_shape,
    "multi_dot": _program_ad_linalg_multi_dot_shape,
    "eig": _program_ad_linalg_eig_shape,
    "eigh": _program_ad_linalg_eigh_shape,
    "eigvals": _program_ad_linalg_eigvals_shape,
    "eigvalsh": _program_ad_linalg_eigvalsh_shape,
    "svd": _program_ad_linalg_svd_shape,
    "pinv": _program_ad_linalg_pinv_shape,
}

_PROGRAM_AD_LINALG_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "det": _program_ad_linalg_no_static_arguments,
    "inv": _program_ad_linalg_no_static_arguments,
    "solve": _program_ad_linalg_no_static_arguments,
    "trace": _program_ad_linalg_trace_static_arguments,
    "diag": _program_ad_linalg_diag_static_arguments,
    "diagflat": _program_ad_linalg_diagflat_static_arguments,
    "matrix_power": _program_ad_linalg_matrix_power_static_arguments,
    "multi_dot": _program_ad_linalg_multi_dot_static_arguments,
    "eig": _program_ad_linalg_no_static_arguments,
    "eigh": _program_ad_linalg_no_static_arguments,
    "eigvals": _program_ad_linalg_no_static_arguments,
    "eigvalsh": _program_ad_linalg_no_static_arguments,
    "svd": _program_ad_linalg_no_static_arguments,
    "pinv": _program_ad_linalg_no_static_arguments,
}


def _program_ad_linalg_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD linalg batching axes must match argument count")
    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            mapped.append(None)
            continue
        array = _as_real_numeric_array(f"program AD linalg batched argument {index}", arg)
        axis_index = _normalise_axis(f"axes[{index}]", axis, array.ndim)
        size = int(array.shape[axis_index])
        if size <= 0:
            raise ValueError("program AD linalg batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD linalg batching axes must share one batch size")
        mapped.append((array, axis_index))
    if batch_size is None:
        raise ValueError("program AD linalg batching requires at least one mapped axis")

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_args: list[object] = []
        for original, mapped_arg in zip(args, mapped, strict=True):
            if mapped_arg is None:
                sliced_args.append(original)
                continue
            array, axis_index = mapped_arg
            sliced_args.append(np.take(array, batch_index, axis=axis_index))
        outputs.append(
            _as_real_numeric_array("program AD linalg batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    axis_index = _normalise_axis("out_axes", out_axes, stacked.ndim)
    return np.moveaxis(stacked, 0, axis_index)


def _program_ad_linalg_lowering_metadata(name: str) -> Mapping[str, str]:
    nondifferentiable_boundaries = {
        "det": "singular_matrix_rank_drop",
        "inv": "singular_matrix_inverse",
        "solve": "singular_or_incompatible_linear_system",
        "trace": "static_diagonal_offset_axis_pair",
        "diag": "static_diagonal_offset_rank",
        "diagflat": "static_flattened_diagonal_offset_rank",
        "matrix_power": "negative_power_singular_matrix",
        "multi_dot": "static_shape_alignment",
        "eig": "real_simple_diagonalizable_eigensystem",
        "eigh": "symmetric_matrix_distinct_eigenvalues",
        "eigvals": "real_simple_diagonalizable_spectrum",
        "eigvalsh": "symmetric_matrix_distinct_eigenvalues",
        "svd": "distinct_positive_singular_values",
        "pinv": "rank_threshold_crossing",
    }
    metadata = {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff linalg dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.linalg.{name}",
        "llvm": "blocked_until_executable_linalg_lowering",
        "rust": "blocked_until_polyglot_linalg_ad",
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
        "static_argument_rule": "none",
        "static_derivative_factory": "not_required",
        "static_signature": "none",
        "conditioning_diagnostic": "diagnose_program_ad_linalg_conditioning",
    }
    if name == "solve":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_solve_derivative_rule",
                "static_signature": "matrix_shape:rank2_square;rhs_shape:rank1_or_rank2",
            }
        )
    elif name == "trace":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_trace_derivative_rule",
                "static_signature": "matrix_shape:rank2;offset_axis_pair",
            }
        )
    elif name == "diag":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_diag_derivative_rule",
                "static_signature": "source_shape:rank1_or_rank2;k",
            }
        )
    elif name == "diagflat":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_diagflat_derivative_rule",
                "static_signature": "source_shape:ranked_tensor_shape;k",
            }
        )
    elif name == "matrix_power":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_matrix_power_derivative_rule",
                "static_signature": "power:i64",
            }
        )
    elif name == "multi_dot":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_multi_dot_derivative_rule",
                "static_signature": "operand_shapes:ranked_tensor_shape_sequence",
            }
        )
    elif name == "eig":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eig_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_real_simple_eigensystem",
            }
        )
    elif name == "eigh":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigh_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_symmetric_distinct_spectrum",
            }
        )
    elif name == "eigvalsh":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigvalsh_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_symmetric_distinct_spectrum",
            }
        )
    elif name == "eigvals":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigvals_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_real_simple_spectrum",
            }
        )
    elif name == "svd":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_svdvals_derivative_rule",
                "static_signature": "matrix_shape:rank2;compute_uv:false;hermitian:false",
            }
        )
    elif name == "pinv":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_pinv_derivative_rule",
                "static_signature": "matrix_shape:rank2_full_rank;rcond:static_f64",
            }
        )
    return metadata


def _register_program_ad_linalg_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_LINALG_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        rule = _program_ad_linalg_derivative_rule(name)
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=rule,
                batching_rule=_program_ad_linalg_batching_rule,
                lowering_metadata=_program_ad_linalg_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_LINALG_SHAPE_RULES[name],
                dtype_rule=_program_ad_linalg_dtype_rule,
                static_argument_rule=_PROGRAM_AD_LINALG_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_LINALG_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_linalg_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_LINALG_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"unsupported program AD linalg primitive {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_LINALG_POLICY:
        raise ValueError(
            f"program AD linalg primitive {name} must declare policy {_PROGRAM_AD_LINALG_POLICY}"
        )
    if contract.effect != "pure":
        raise ValueError(f"program AD linalg primitive {name} must be pure")
    metadata = contract.lowering_metadata
    missing = []
    if contract.batching_rule is None:
        missing.append("batching_rule")
    if not metadata:
        missing.append("lowering_metadata")
    if not metadata.get("mlir_op"):
        missing.append("mlir_op")
    if not metadata.get("nondifferentiable_boundary"):
        missing.append("nondifferentiable_boundary")
    if metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
        missing.append("nondifferentiable_boundary_policy")
    if contract.shape_rule is None:
        missing.append("shape_rule")
    if contract.dtype_rule is None:
        missing.append("dtype_rule")
    if contract.static_argument_rule is None:
        missing.append("static_argument_rule")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"incomplete program AD linalg primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_linalg_contract_dispatch(contract, args)
    return contract


__all__ = [
    "ProgramADLinalgConditioningDiagnostic",
    "diagnose_program_ad_linalg_conditioning",
    "program_ad_linalg_diag_derivative_rule",
    "program_ad_linalg_diagflat_derivative_rule",
    "program_ad_linalg_eig_derivative_rule",
    "program_ad_linalg_eigh_derivative_rule",
    "program_ad_linalg_eigvals_derivative_rule",
    "program_ad_linalg_eigvalsh_derivative_rule",
    "program_ad_linalg_matrix_power_derivative_rule",
    "program_ad_linalg_multi_dot_derivative_rule",
    "program_ad_linalg_pinv_derivative_rule",
    "program_ad_linalg_solve_derivative_rule",
    "program_ad_linalg_svdvals_derivative_rule",
    "program_ad_linalg_trace_derivative_rule",
    "_program_ad_linalg_det_cofactor_matrix",
    "_program_ad_linalg_eig_eigenvector_jvp_matrix",
    "_program_ad_linalg_eigh_eigenvector_jvp_matrix",
    "_program_ad_linalg_eigh_vjp_matrix",
    "_program_ad_linalg_normalise_rcond",
    "_program_ad_linalg_pinv_jvp_matrix",
    "_program_ad_linalg_pinv_value_matrix",
    "_program_ad_linalg_pinv_vjp_matrix",
    "_program_ad_linalg_real_simple_eig_decomposition_from_matrix",
    "_program_ad_linalg_require_distinct_eigenvalues",
    "_program_ad_linalg_require_distinct_positive_singular_values",
    "_program_ad_linalg_require_symmetric",
    "_program_ad_linalg_uplo",
    "_register_program_ad_linalg_primitive_contracts",
    "_require_program_ad_linalg_contract",
]
