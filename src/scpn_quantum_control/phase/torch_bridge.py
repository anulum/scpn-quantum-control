# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PyTorch Bridge
"""Optional PyTorch interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    value_and_parameter_shift_grad,
)

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class PhaseTorchParameterShiftResult:
    """Result from the optional PyTorch phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    torch_value: Any
    torch_gradient: Any
    method: str
    evaluations: int
    host_boundary: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "host_boundary": self.host_boundary,
            "torch_value_type": type(self.torch_value).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is unavailable; install scpn-quantum-control[torch]") from exc
    return torch


def is_phase_torch_available() -> bool:
    """Return whether the optional phase PyTorch bridge can import PyTorch."""
    try:
        _load_torch()
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


def _torch_values_to_numpy(values: object) -> FloatArray:
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return _as_parameter_vector("values", candidate)


def _torch_tensor(torch_module: Any, values: object) -> Any:
    dtype = getattr(torch_module, "float64", None)
    as_tensor = getattr(torch_module, "as_tensor", None)
    if callable(as_tensor):
        if dtype is None:
            return as_tensor(values)
        return as_tensor(values, dtype=dtype)
    tensor = getattr(torch_module, "tensor", None)
    if callable(tensor):
        if dtype is None:
            return tensor(values)
        return tensor(values, dtype=dtype)
    raise RuntimeError("PyTorch module does not expose as_tensor or tensor")


def torch_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike | object,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseTorchParameterShiftResult:
    """Return phase parameter-shift value and gradient as NumPy and PyTorch tensors.

    This is a host-boundary bridge. The quantum expectation is evaluated through
    SCPN's deterministic parameter-shift rule and the result is converted to
    PyTorch tensors for framework pipelines. It does not claim native PyTorch
    autograd through a quantum simulator.
    """
    torch_module = _load_torch()
    parameter_values = _torch_values_to_numpy(values)
    result: GradientResult = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    gradient = _as_parameter_vector(
        "PyTorch parameter-shift gradient",
        result.gradient,
        width=parameter_values.size,
    )
    return PhaseTorchParameterShiftResult(
        value=float(result.value),
        gradient=gradient,
        torch_value=_torch_tensor(torch_module, np.asarray(result.value, dtype=np.float64)),
        torch_gradient=_torch_tensor(torch_module, gradient),
        method="parameter_shift",
        evaluations=result.evaluations,
        host_boundary=True,
    )


__all__ = [
    "PhaseTorchParameterShiftResult",
    "is_phase_torch_available",
    "torch_parameter_shift_value_and_grad",
]
