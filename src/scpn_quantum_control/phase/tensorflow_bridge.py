# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase TensorFlow Bridge
"""Optional TensorFlow interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    value_and_parameter_shift_grad,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class PhaseTensorFlowParameterShiftResult:
    """Result from the optional TensorFlow phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    tensorflow_value: Any
    tensorflow_gradient: Any
    method: str
    evaluations: int
    host_boundary: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "host_boundary": self.host_boundary,
            "shift_terms": self.shift_terms,
            "tensorflow_value_type": type(self.tensorflow_value).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


def _load_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is unavailable; install scpn-quantum-control[tensorflow]"
        ) from exc
    return tf


def is_phase_tensorflow_available() -> bool:
    """Return whether the optional phase TensorFlow bridge can import TensorFlow."""
    try:
        _load_tensorflow()
    except ImportError:
        return False
    return True


def _as_parameter_vector(name: str, values: object, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _tensorflow_values_to_numpy(values: object) -> FloatArray:
    candidate = values
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return _as_parameter_vector("values", candidate)


def _tensorflow_tensor(tensorflow_module: Any, values: object) -> Any:
    convert = getattr(tensorflow_module, "convert_to_tensor", None)
    if not callable(convert):
        raise RuntimeError("TensorFlow module does not expose convert_to_tensor")
    dtype = getattr(tensorflow_module, "float64", None)
    if dtype is None:
        return convert(values)
    return convert(values, dtype=dtype)


def tensorflow_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike | object,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseTensorFlowParameterShiftResult:
    """Return phase parameter-shift value and gradient as NumPy and TensorFlow tensors.

    This is a host-boundary bridge. The quantum expectation is evaluated through
    SCPN's deterministic parameter-shift rule and the result is converted to
    TensorFlow tensors for framework pipelines. It does not claim native
    TensorFlow autodiff through a quantum simulator.
    """
    tensorflow_module = _load_tensorflow()
    parameter_values = _tensorflow_values_to_numpy(values)
    result: GradientResult = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    gradient = _as_parameter_vector(
        "TensorFlow parameter-shift gradient",
        result.gradient,
        width=parameter_values.size,
    )
    return PhaseTensorFlowParameterShiftResult(
        value=float(result.value),
        gradient=gradient,
        tensorflow_value=_tensorflow_tensor(
            tensorflow_module,
            np.asarray(result.value, dtype=np.float64),
        ),
        tensorflow_gradient=_tensorflow_tensor(tensorflow_module, gradient),
        method=result.method,
        evaluations=result.evaluations,
        host_boundary=True,
        shift_terms=shift_terms,
    )


__all__ = [
    "PhaseTensorFlowParameterShiftResult",
    "is_phase_tensorflow_available",
    "tensorflow_parameter_shift_value_and_grad",
]
