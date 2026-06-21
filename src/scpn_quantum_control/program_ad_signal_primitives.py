# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD signal primitive rules
"""Static convolution and correlation derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import (
    _program_ad_array_normalise_static_shape,
    _program_ad_array_static_size,
)
from .program_ad_registry import CustomDerivativeRule


def _normalise_convolve_mode(mode: object) -> Literal["full", "same", "valid"]:
    if not isinstance(mode, str) or mode not in {"full", "same", "valid"}:
        raise ValueError("program AD np.convolve mode must be 'full', 'same', or 'valid'")
    return cast(Literal["full", "same", "valid"], mode)


def _normalise_correlate_mode(mode: object) -> Literal["full", "same", "valid"]:
    if not isinstance(mode, str) or mode not in {"full", "same", "valid"}:
        raise ValueError("program AD np.correlate mode must be 'full', 'same', or 'valid'")
    return cast(Literal["full", "same", "valid"], mode)


def _convolve_output_window(
    left_size: int, right_size: int, mode: Literal["full", "same", "valid"]
) -> tuple[int, int]:
    if mode == "full":
        return 0, left_size + right_size - 1
    if mode == "same":
        output_size = max(left_size, right_size)
        start = (min(left_size, right_size) - 1) // 2
        return start, start + output_size
    output_size = max(left_size, right_size) - min(left_size, right_size) + 1
    start = min(left_size, right_size) - 1
    return start, start + output_size


def _program_ad_signal_convolve_static_shape(role: str, shape: Sequence[int]) -> tuple[int, ...]:
    normalised = _program_ad_array_normalise_static_shape(f"signal convolve {role}", shape)
    if len(normalised) != 1:
        raise ValueError("program AD signal convolve direct rule requires rank-1 operands")
    if normalised[0] <= 0:
        raise ValueError("program AD signal convolve direct rule requires non-empty operands")
    return normalised


def _program_ad_signal_convolve_source_size(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> int:
    return _program_ad_array_static_size(left_shape) + _program_ad_array_static_size(right_shape)


def _program_ad_signal_convolve_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD signal convolve {role}", values).reshape(-1)
    left_size = _program_ad_array_static_size(left_shape)
    expected_size = _program_ad_signal_convolve_source_size(left_shape, right_shape)
    if vector.size != expected_size:
        raise ValueError(
            f"program AD signal convolve direct rule requires {expected_size} {role} values"
        )
    return vector[:left_size], vector[left_size:]


def _program_ad_signal_convolve_output_size(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> int:
    left_size = _program_ad_array_static_size(left_shape)
    right_size = _program_ad_array_static_size(right_shape)
    start, stop = _convolve_output_window(left_size, right_size, mode)
    return stop - start


def _program_ad_signal_convolve_direct_value(
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    return np.convolve(left, right, mode=mode).astype(np.float64, copy=False)


def _program_ad_signal_convolve_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    tangent_left, tangent_right = _program_ad_signal_convolve_split_source(
        "tangent", tangent, left_shape=left_shape, right_shape=right_shape
    )
    return (
        np.convolve(tangent_left, right, mode=mode) + np.convolve(left, tangent_right, mode=mode)
    ).astype(np.float64, copy=False)


def _program_ad_signal_convolve_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD signal convolve cotangent", cotangent
    ).reshape(-1)
    output_size = _program_ad_signal_convolve_output_size(left_shape, right_shape, mode)
    if cotangent_vector.size != output_size:
        raise ValueError("program AD signal convolve VJP requires cotangent matching output size")

    left_adjoint = np.zeros(left.size, dtype=np.float64)
    right_adjoint = np.zeros(right.size, dtype=np.float64)
    for index in range(left.size):
        basis = np.zeros(left.size, dtype=np.float64)
        basis[index] = 1.0
        left_adjoint[index] = float(
            np.sum(np.convolve(basis, right, mode=mode) * cotangent_vector)
        )
    for index in range(right.size):
        basis = np.zeros(right.size, dtype=np.float64)
        basis[index] = 1.0
        right_adjoint[index] = float(
            np.sum(np.convolve(left, basis, mode=mode) * cotangent_vector)
        )
    return np.concatenate([left_adjoint, right_adjoint])


def program_ad_signal_convolve_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    mode: object = "full",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.convolve`` operands."""

    left = _program_ad_signal_convolve_static_shape("left", left_shape)
    right = _program_ad_signal_convolve_static_shape("right", right_shape)
    mode_value = _normalise_convolve_mode(mode)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_value(
            values, left_shape=left, right_shape=right, mode=mode_value
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_jvp(
            values, tangent, left_shape=left, right_shape=right, mode=mode_value
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_vjp(
            values, cotangent, left_shape=left, right_shape=right, mode=mode_value
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_signal_convolve_"
            f"left{left[0]}_right{right[0]}_mode_{mode_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_signal_correlate_static_shape(role: str, shape: Sequence[int]) -> tuple[int, ...]:
    normalised = _program_ad_array_normalise_static_shape(f"signal correlate {role}", shape)
    if len(normalised) != 1:
        raise ValueError("program AD signal correlate direct rule requires rank-1 operands")
    if normalised[0] <= 0:
        raise ValueError("program AD signal correlate direct rule requires non-empty operands")
    return normalised


def _program_ad_signal_correlate_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD signal correlate {role}", values).reshape(-1)
    left_size = _program_ad_array_static_size(left_shape)
    expected_size = _program_ad_signal_convolve_source_size(left_shape, right_shape)
    if vector.size != expected_size:
        raise ValueError(
            f"program AD signal correlate direct rule requires {expected_size} {role} values"
        )
    return vector[:left_size], vector[left_size:]


def _program_ad_signal_correlate_output_size(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> int:
    left = np.zeros(left_shape[0], dtype=np.float64)
    right = np.zeros(right_shape[0], dtype=np.float64)
    return int(np.correlate(left, right, mode=mode).size)


def _program_ad_signal_correlate_direct_value(
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    return np.correlate(left, right, mode=mode).astype(np.float64, copy=False)


def _program_ad_signal_correlate_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    tangent_left, tangent_right = _program_ad_signal_correlate_split_source(
        "tangent", tangent, left_shape=left_shape, right_shape=right_shape
    )
    return (
        np.correlate(tangent_left, right, mode=mode) + np.correlate(left, tangent_right, mode=mode)
    ).astype(np.float64, copy=False)


def _program_ad_signal_correlate_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD signal correlate cotangent", cotangent
    ).reshape(-1)
    output_size = _program_ad_signal_correlate_output_size(left_shape, right_shape, mode)
    if cotangent_vector.size != output_size:
        raise ValueError("program AD signal correlate VJP requires cotangent matching output size")

    left_adjoint = np.zeros(left.size, dtype=np.float64)
    right_adjoint = np.zeros(right.size, dtype=np.float64)
    for index in range(left.size):
        basis = np.zeros(left.size, dtype=np.float64)
        basis[index] = 1.0
        left_adjoint[index] = float(
            np.sum(np.correlate(basis, right, mode=mode) * cotangent_vector)
        )
    for index in range(right.size):
        basis = np.zeros(right.size, dtype=np.float64)
        basis[index] = 1.0
        right_adjoint[index] = float(
            np.sum(np.correlate(left, basis, mode=mode) * cotangent_vector)
        )
    return np.concatenate([left_adjoint, right_adjoint])


def program_ad_signal_correlate_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    mode: object = "valid",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.correlate`` operands."""

    left = _program_ad_signal_correlate_static_shape("left", left_shape)
    right = _program_ad_signal_correlate_static_shape("right", right_shape)
    mode_value = _normalise_correlate_mode(mode)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_value(
            values, left_shape=left, right_shape=right, mode=mode_value
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_jvp(
            values, tangent, left_shape=left, right_shape=right, mode=mode_value
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_vjp(
            values, cotangent, left_shape=left, right_shape=right, mode=mode_value
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_signal_correlate_"
            f"left{left[0]}_right{right[0]}_mode_{mode_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
