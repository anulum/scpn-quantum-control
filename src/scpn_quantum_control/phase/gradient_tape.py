# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Gradient Tape
"""Context-managed quantum-gradient tape for phase objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    StochasticGradientResult,
    value_and_parameter_shift_grad,
)
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend
from .param_shift import parameter_shift_gradient_with_uncertainty

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
TapeKind = Literal["deterministic", "stochastic"]


@dataclass(frozen=True)
class TapeGradientRecord:
    """One recorded quantum-gradient evaluation."""

    name: str
    kind: TapeKind
    plan: QuantumGradientPlan
    result: GradientResult | StochasticGradientResult

    @property
    def gradient(self) -> FloatArray:
        """Return the recorded gradient vector."""
        return self.result.gradient

    @property
    def value(self) -> float:
        """Return the recorded objective value."""
        return self.result.value

    @property
    def evaluations(self) -> int:
        """Return planned quantum objective evaluations, excluding tape bookkeeping."""
        return self.plan.evaluations

    @property
    def method(self) -> str:
        """Return the replay method recorded by the gradient result."""
        return self.result.method

    @property
    def shift_terms(self) -> int:
        """Return the number of parameter-shift terms used per parameter."""
        return self.plan.shift_terms

    @property
    def standard_error(self) -> FloatArray | None:
        """Return finite-shot standard errors when the record is stochastic."""
        if isinstance(self.result, StochasticGradientResult):
            return self.result.standard_error
        return None

    @property
    def confidence_radius(self) -> FloatArray | None:
        """Return finite-shot confidence radii when the record is stochastic."""
        if isinstance(self.result, StochasticGradientResult):
            return self.result.confidence_radius
        return None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready replay provenance for audit logs and notebooks."""
        standard_error = self.standard_error
        confidence_radius = self.confidence_radius
        return {
            "name": self.name,
            "kind": self.kind,
            "backend": self.plan.backend,
            "plan_method": self.plan.method,
            "method": self.method,
            "n_params": self.plan.n_params,
            "shift_terms": self.plan.shift_terms,
            "evaluations": self.plan.evaluations,
            "shots": self.plan.shots,
            "seed": self.plan.seed,
            "finite_shot": self.plan.finite_shot,
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "standard_error": (None if standard_error is None else standard_error.tolist()),
            "confidence_radius": (
                None if confidence_radius is None else confidence_radius.tolist()
            ),
        }


def _as_parameter_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _shift_term_count(rule: ParameterShiftRule | None) -> int:
    if rule is None:
        return 1
    return len(rule.terms)


def _as_finite_shot_array(name: str, values: ArrayLike) -> FloatArray:
    records = np.asarray(values, dtype=float)
    if records.ndim not in (1, 2):
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if records.size == 0:
        raise ValueError(f"{name} must not be empty")
    if records.ndim == 2 and (records.shape[0] == 0 or records.shape[1] == 0):
        raise ValueError(f"{name} must have non-empty shift and parameter axes")
    if not np.all(np.isfinite(records)):
        raise ValueError(f"{name} must contain only finite values")
    return records.astype(np.float64, copy=True)


def _require_matching_finite_shot_shape(
    name: str,
    values: ArrayLike,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    records = _as_finite_shot_array(name, values)
    if records.shape != expected_shape:
        raise ValueError(f"{name} must match plus_values shape {expected_shape}")
    return records


class QuantumGradientTape:
    """Context manager that records supported phase-gradient evaluations."""

    def __init__(
        self,
        *,
        backend: str = "statevector",
        shots: int | None = None,
        seed: int | None = None,
        confidence_level: float = 0.95,
        allow_hardware: bool = False,
    ) -> None:
        """Create a tape with backend policy shared by all records."""
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.confidence_level = confidence_level
        self.allow_hardware = allow_hardware
        self._records: list[TapeGradientRecord] = []
        self._active = False

    def __enter__(self) -> QuantumGradientTape:
        """Activate the tape context."""
        if self._active:
            raise RuntimeError("gradient tape is already active")
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Deactivate the tape context."""
        self._active = False

    @property
    def records(self) -> tuple[TapeGradientRecord, ...]:
        """Return immutable view of recorded gradient evaluations."""
        return tuple(self._records)

    def clear(self) -> None:
        """Clear recorded evaluations while preserving backend policy."""
        self._records.clear()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "gradient tape records can only be created inside an active context"
            )

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("record name must be a non-empty string")
        return name.strip()

    @staticmethod
    def _ensure_supported(plan: QuantumGradientPlan) -> None:
        if plan.fail_closed:
            joined = "; ".join(plan.reasons)
            raise ValueError(f"backend gradient plan is unsupported: {joined}")

    def record_parameter_shift(
        self,
        name: str,
        objective: ScalarObjective,
        params: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> TapeGradientRecord:
        """Record deterministic parameter-shift value and gradient."""
        self._require_active()
        record_name = self._validate_name(name)
        values = _as_parameter_vector("params", params)
        shift_terms = _shift_term_count(rule)
        plan = plan_quantum_gradient_backend(
            self.backend,
            n_params=values.size,
            shift_terms=shift_terms,
            method="parameter_shift",
            seed=self.seed,
            allow_hardware=self.allow_hardware,
        )
        self._ensure_supported(plan)
        result = value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        record = TapeGradientRecord(
            name=record_name,
            kind="deterministic",
            plan=plan,
            result=result,
        )
        self._records.append(record)
        return record

    def record_finite_shot_parameter_shift(
        self,
        name: str,
        *,
        plus_values: ArrayLike,
        minus_values: ArrayLike,
        plus_variances: ArrayLike,
        minus_variances: ArrayLike,
        value: float = 0.0,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        confidence_z: float = 1.959963984540054,
    ) -> TapeGradientRecord:
        """Record finite-shot parameter-shift gradient with uncertainty."""
        self._require_active()
        record_name = self._validate_name(name)
        plus = _as_finite_shot_array("plus_values", plus_values)
        shift_terms = _shift_term_count(rule)
        if plus.ndim == 1:
            if shift_terms != 1:
                raise ValueError(
                    "multi-term finite-shot tape records require two-dimensional "
                    "plus_values with shape (shift_terms, n_params)"
                )
            n_params = plus.size
        else:
            if plus.shape[0] != shift_terms:
                raise ValueError(
                    "plus_values first axis must match the parameter-shift "
                    f"rule term count {shift_terms}"
                )
            n_params = plus.shape[1]
        minus = _require_matching_finite_shot_shape(
            "minus_values",
            minus_values,
            plus.shape,
        )
        plus_var = _require_matching_finite_shot_shape(
            "plus_variances",
            plus_variances,
            plus.shape,
        )
        minus_var = _require_matching_finite_shot_shape(
            "minus_variances",
            minus_variances,
            plus.shape,
        )
        plan = plan_quantum_gradient_backend(
            self.backend,
            n_params=n_params,
            shift_terms=shift_terms,
            method="stochastic_parameter_shift",
            shots=self.shots,
            seed=self.seed,
            finite_shot=True,
            confidence_level=self.confidence_level,
            allow_hardware=self.allow_hardware,
        )
        self._ensure_supported(plan)
        if plan.shots is None:
            raise ValueError("finite-shot tape records require an explicit shot plan")
        result = parameter_shift_gradient_with_uncertainty(
            plus,
            minus,
            plus_var,
            minus_var,
            shots=plan.shots,
            backend=plan.backend,
            value=value,
            parameters=parameters,
            rule=rule,
            confidence_level=self.confidence_level,
            confidence_z=confidence_z,
        )
        record = TapeGradientRecord(
            name=record_name,
            kind="stochastic",
            plan=plan,
            result=result,
        )
        self._records.append(record)
        return record


def gradient_tape(
    *,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int | None = None,
    confidence_level: float = 0.95,
    allow_hardware: bool = False,
) -> QuantumGradientTape:
    """Return a context-managed quantum-gradient tape."""
    return QuantumGradientTape(
        backend=backend,
        shots=shots,
        seed=seed,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )


__all__ = [
    "QuantumGradientTape",
    "TapeGradientRecord",
    "TapeKind",
    "gradient_tape",
]
