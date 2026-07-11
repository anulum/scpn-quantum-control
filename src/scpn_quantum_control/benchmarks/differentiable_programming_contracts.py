# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Benchmark Contracts
"""Dependency-light result contracts shared by differentiable benchmark suites."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DifferentiableProgrammingBenchmarkResult:
    """Conformance result for one differentiable-programming benchmark case."""

    case_id: str
    category: str
    value: float
    gradient: NDArray[np.float64]
    analytic_gradient: NDArray[np.float64]
    max_abs_gradient_error: float
    adjoint_supported: bool
    max_abs_adjoint_error: float | None
    claim_boundary: str
    blocked_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("benchmark case_id must be non-empty")
        if not self.category:
            raise ValueError("benchmark category must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("benchmark value must be finite")
        gradient = _as_gradient("gradient", self.gradient)
        analytic = _as_gradient("analytic_gradient", self.analytic_gradient)
        if gradient.shape != analytic.shape:
            raise ValueError("benchmark gradient and analytic_gradient shapes must match")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("benchmark max_abs_gradient_error must be finite and non-negative")
        if not isinstance(self.adjoint_supported, bool):
            raise ValueError("benchmark adjoint_supported must be a boolean")
        if self.max_abs_adjoint_error is not None and (
            self.max_abs_adjoint_error < 0.0 or not math.isfinite(self.max_abs_adjoint_error)
        ):
            raise ValueError("benchmark max_abs_adjoint_error must be finite or None")
        if not self.claim_boundary:
            raise ValueError("benchmark claim_boundary must be non-empty")
        if any(not isinstance(reason, str) or not reason for reason in self.blocked_reasons):
            raise ValueError("benchmark blocked_reasons must contain non-empty strings")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "analytic_gradient", analytic)

    @property
    def passed(self) -> bool:
        """Return whether implemented gradients match the analytic reference."""
        return (
            not self.blocked_reasons
            and self.max_abs_gradient_error <= 1.0e-12
            and (self.max_abs_adjoint_error is None or self.max_abs_adjoint_error <= 1.0e-12)
        )


@dataclass(frozen=True)
class DifferentiableProgrammingExternalReferenceResult:
    """Program-AD comparison against an independently executed autodiff backend."""

    case_id: str
    backend: str
    program_value: float
    reference_value: float
    program_gradient: NDArray[np.float64]
    reference_gradient: NDArray[np.float64]
    max_abs_value_error: float
    max_abs_gradient_error: float
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("external reference case_id must be non-empty")
        if not self.backend:
            raise ValueError("external reference backend must be non-empty")
        if not math.isfinite(self.program_value) or not math.isfinite(self.reference_value):
            raise ValueError("external reference values must be finite")
        program_gradient = _as_gradient("program_gradient", self.program_gradient)
        reference_gradient = _as_gradient("reference_gradient", self.reference_gradient)
        if program_gradient.shape != reference_gradient.shape:
            raise ValueError("external reference gradient shapes must match")
        if self.max_abs_value_error < 0.0 or not math.isfinite(self.max_abs_value_error):
            raise ValueError("external reference value error must be finite and non-negative")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("external reference gradient error must be finite and non-negative")
        if not self.claim_boundary:
            raise ValueError("external reference claim_boundary must be non-empty")
        object.__setattr__(self, "program_gradient", program_gradient)
        object.__setattr__(self, "reference_gradient", reference_gradient)

    @property
    def passed(self) -> bool:
        """Return whether program AD matches the external reference backend."""
        return self.max_abs_value_error <= 1.0e-10 and self.max_abs_gradient_error <= 1.0e-10


@dataclass(frozen=True)
class QuantumGradientBenchmarkResult:
    """Conformance result for deterministic quantum-gradient benchmark rows."""

    case_id: str
    category: str
    value: float
    parameter_shift_gradient: NDArray[np.float64]
    finite_difference_gradient: NDArray[np.float64]
    analytic_gradient: NDArray[np.float64]
    max_abs_reference_error: float
    max_abs_finite_difference_error: float
    verification_passed: bool
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("quantum gradient case_id must be non-empty")
        if not self.category:
            raise ValueError("quantum gradient category must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("quantum gradient value must be finite")
        parameter_shift_gradient = _as_gradient(
            "parameter_shift_gradient",
            self.parameter_shift_gradient,
        )
        finite_difference_gradient = _as_gradient(
            "finite_difference_gradient",
            self.finite_difference_gradient,
        )
        analytic_gradient = _as_gradient("analytic_gradient", self.analytic_gradient)
        if (
            parameter_shift_gradient.shape != finite_difference_gradient.shape
            or parameter_shift_gradient.shape != analytic_gradient.shape
        ):
            raise ValueError("quantum gradient benchmark gradient shapes must match")
        if self.max_abs_reference_error < 0.0 or not math.isfinite(self.max_abs_reference_error):
            raise ValueError("quantum gradient reference error must be finite and non-negative")
        if self.max_abs_finite_difference_error < 0.0 or not math.isfinite(
            self.max_abs_finite_difference_error
        ):
            raise ValueError(
                "quantum gradient finite-difference error must be finite and non-negative"
            )
        if not isinstance(self.verification_passed, bool):
            raise ValueError("quantum gradient verification_passed must be a boolean")
        if self.evaluations <= 0:
            raise ValueError("quantum gradient evaluations must be positive")
        if not self.claim_boundary:
            raise ValueError("quantum gradient claim_boundary must be non-empty")
        object.__setattr__(
            self,
            "parameter_shift_gradient",
            parameter_shift_gradient,
        )
        object.__setattr__(
            self,
            "finite_difference_gradient",
            finite_difference_gradient,
        )
        object.__setattr__(self, "analytic_gradient", analytic_gradient)

    @property
    def passed(self) -> bool:
        """Return whether parameter-shift gradients passed all reference checks."""
        return (
            self.verification_passed
            and self.max_abs_reference_error <= 1.0e-12
            and self.max_abs_finite_difference_error <= 1.0e-5
        )


def _as_gradient(name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _max_abs_error(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(
        np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


__all__ = [
    "DifferentiableProgrammingBenchmarkResult",
    "DifferentiableProgrammingExternalReferenceResult",
    "QuantumGradientBenchmarkResult",
]
