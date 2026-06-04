# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Gradient Backend Planner
"""Backend-aware quantum-gradient planning for phase objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GradientMethod = Literal[
    "auto",
    "parameter_shift",
    "stochastic_parameter_shift",
    "finite_difference",
    "spsa",
    "unsupported",
]


@dataclass(frozen=True)
class QuantumGradientBackendCapability:
    """Declared gradient capabilities for one execution backend family."""

    backend: str
    family: str
    supports_parameter_shift: bool
    supports_finite_shot: bool
    supports_adjoint: bool
    supports_spsa: bool
    hardware: bool
    default_shots: int | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class QuantumGradientPlan:
    """Fail-closed gradient execution plan for a supported or unsupported backend."""

    backend: str
    family: str
    method: str
    supported: bool
    n_params: int
    shift_terms: int
    evaluations: int
    shots: int | None
    seed: int | None
    finite_shot: bool
    confidence_level: float | None
    requires_hardware_approval: bool
    reasons: tuple[str, ...]
    alternatives: tuple[str, ...]

    @property
    def fail_closed(self) -> bool:
        """Return true when this plan intentionally refuses execution."""
        return not self.supported


_STATEVECTOR_ALIASES = {
    "statevector",
    "statevector_simulator",
    "simulator",
    "exact",
    "exact_simulator",
    "local_statevector",
    "qiskit_statevector",
}

_SHOT_SIMULATOR_ALIASES = {
    "shots",
    "shot_simulator",
    "finite_shot",
    "finite_shot_simulator",
    "qasm",
    "qasm_simulator",
    "aer",
    "aer_simulator",
    "local_qasm",
}

_HARDWARE_ALIASES = {
    "hardware",
    "qpu",
    "ibm",
    "ibm_quantum",
    "braket",
    "rigetti",
    "ionq",
    "quantinuum",
    "quera",
    "pasqal",
    "quandela",
    "pennylane_device",
}


def _normalise_backend(backend: str) -> str:
    key = backend.strip().lower().replace("-", "_").replace(".", "_")
    if key in _STATEVECTOR_ALIASES:
        return "statevector_simulator"
    if key in _SHOT_SIMULATOR_ALIASES:
        return "finite_shot_simulator"
    if key in _HARDWARE_ALIASES:
        return "hardware_qpu"
    if not key:
        raise ValueError("backend must be a non-empty string")
    return key


def _normalise_method(method: str) -> GradientMethod:
    key = method.strip().lower().replace("-", "_")
    if key in {
        "auto",
        "parameter_shift",
        "stochastic_parameter_shift",
        "finite_difference",
        "spsa",
        "unsupported",
    }:
        return key  # type: ignore[return-value]
    raise ValueError(
        "method must be one of: auto, parameter_shift, stochastic_parameter_shift, "
        "finite_difference, spsa, unsupported"
    )


def _positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _positive_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    return confidence


def quantum_gradient_backend_capability(backend: str) -> QuantumGradientBackendCapability:
    """Return declared gradient capabilities for a known backend family."""
    normalised = _normalise_backend(backend)
    if normalised == "statevector_simulator":
        return QuantumGradientBackendCapability(
            backend=normalised,
            family="statevector",
            supports_parameter_shift=True,
            supports_finite_shot=False,
            supports_adjoint=False,
            supports_spsa=False,
            hardware=False,
            default_shots=None,
            notes=("deterministic local expectation route",),
        )
    if normalised == "finite_shot_simulator":
        return QuantumGradientBackendCapability(
            backend=normalised,
            family="finite_shot_simulator",
            supports_parameter_shift=True,
            supports_finite_shot=True,
            supports_adjoint=False,
            supports_spsa=True,
            hardware=False,
            default_shots=4096,
            notes=("local finite-shot route with explicit uncertainty metadata",),
        )
    if normalised == "hardware_qpu":
        return QuantumGradientBackendCapability(
            backend=normalised,
            family="hardware",
            supports_parameter_shift=True,
            supports_finite_shot=True,
            supports_adjoint=False,
            supports_spsa=True,
            hardware=True,
            default_shots=None,
            notes=("hardware gradients are policy-gated and disabled by default",),
        )
    return QuantumGradientBackendCapability(
        backend=normalised,
        family="unknown",
        supports_parameter_shift=False,
        supports_finite_shot=False,
        supports_adjoint=False,
        supports_spsa=False,
        hardware=False,
        default_shots=None,
        notes=("unknown backend family",),
    )


def plan_quantum_gradient_backend(
    backend: str,
    *,
    n_params: int,
    shift_terms: int = 1,
    method: str = "auto",
    shots: int | None = None,
    seed: int | None = None,
    finite_shot: bool = False,
    confidence_level: float | None = None,
    allow_hardware: bool = False,
) -> QuantumGradientPlan:
    """Plan a quantum-gradient method with fail-closed backend boundaries."""
    if isinstance(n_params, bool) or not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("n_params must be a positive integer")
    if isinstance(shift_terms, bool) or not isinstance(shift_terms, int) or shift_terms <= 0:
        raise ValueError("shift_terms must be a positive integer")
    shot_count = _positive_int("shots", shots)
    seed_value = _non_negative_seed(seed)
    confidence = _positive_confidence(confidence_level)
    mode = _normalise_method(method)
    capability = quantum_gradient_backend_capability(backend)

    reasons: list[str] = []
    alternatives: list[str] = []
    selected = mode
    selected_shots = shot_count

    if capability.hardware and not allow_hardware:
        return QuantumGradientPlan(
            backend=capability.backend,
            family=capability.family,
            method="unsupported",
            supported=False,
            n_params=n_params,
            shift_terms=shift_terms,
            evaluations=0,
            shots=shot_count,
            seed=seed_value,
            finite_shot=True,
            confidence_level=confidence,
            requires_hardware_approval=True,
            reasons=("hardware gradient execution requires explicit hardware policy approval",),
            alternatives=("statevector_simulator", "finite_shot_simulator"),
        )

    if capability.family == "unknown":
        return QuantumGradientPlan(
            backend=capability.backend,
            family=capability.family,
            method="unsupported",
            supported=False,
            n_params=n_params,
            shift_terms=shift_terms,
            evaluations=0,
            shots=shot_count,
            seed=seed_value,
            finite_shot=finite_shot,
            confidence_level=confidence,
            requires_hardware_approval=False,
            reasons=("unknown backend has no registered gradient capability",),
            alternatives=("statevector_simulator", "finite_shot_simulator"),
        )

    if mode == "auto":
        if finite_shot or capability.family == "finite_shot_simulator" or capability.hardware:
            selected = "stochastic_parameter_shift"
        else:
            selected = "parameter_shift"

    if selected == "parameter_shift":
        if finite_shot or capability.family == "finite_shot_simulator":
            reasons.append("finite-shot execution needs stochastic_parameter_shift")
            alternatives.append("stochastic_parameter_shift")
        elif not capability.supports_parameter_shift:
            reasons.append("backend does not support parameter-shift")
        else:
            return QuantumGradientPlan(
                backend=capability.backend,
                family=capability.family,
                method="parameter_shift",
                supported=True,
                n_params=n_params,
                shift_terms=shift_terms,
                evaluations=2 * shift_terms * n_params,
                shots=None,
                seed=seed_value,
                finite_shot=False,
                confidence_level=None,
                requires_hardware_approval=False,
                reasons=capability.notes,
                alternatives=tuple(alternatives),
            )

    if selected == "stochastic_parameter_shift":
        if not capability.supports_finite_shot:
            reasons.append("backend has no finite-shot estimator support")
            alternatives.append("parameter_shift")
        else:
            selected_shots = selected_shots or capability.default_shots
            if selected_shots is None:
                reasons.append("shots are required for finite-shot gradient planning")
            else:
                return QuantumGradientPlan(
                    backend=capability.backend,
                    family=capability.family,
                    method="stochastic_parameter_shift",
                    supported=True,
                    n_params=n_params,
                    shift_terms=shift_terms,
                    evaluations=2 * shift_terms * n_params,
                    shots=selected_shots,
                    seed=seed_value,
                    finite_shot=True,
                    confidence_level=confidence or 0.95,
                    requires_hardware_approval=capability.hardware,
                    reasons=capability.notes,
                    alternatives=tuple(alternatives),
                )

    if selected == "spsa":
        if capability.supports_spsa:
            selected_shots = selected_shots or capability.default_shots
            return QuantumGradientPlan(
                backend=capability.backend,
                family=capability.family,
                method="spsa",
                supported=True,
                n_params=n_params,
                shift_terms=shift_terms,
                evaluations=2,
                shots=selected_shots,
                seed=seed_value,
                finite_shot=True,
                confidence_level=confidence or 0.95,
                requires_hardware_approval=capability.hardware,
                reasons=capability.notes + ("approximate diagnostic fallback",),
                alternatives=("stochastic_parameter_shift",),
            )
        reasons.append("backend does not support SPSA fallback")
        alternatives.append("stochastic_parameter_shift")

    if selected == "finite_difference":
        reasons.append("finite_difference is diagnostic-only and not a promoted quantum gradient")
        alternatives.append("parameter_shift")

    if selected == "unsupported":
        reasons.append("method explicitly disabled")
        alternatives.append("parameter_shift")

    return QuantumGradientPlan(
        backend=capability.backend,
        family=capability.family,
        method="unsupported",
        supported=False,
        n_params=n_params,
        shift_terms=shift_terms,
        evaluations=0,
        shots=selected_shots,
        seed=seed_value,
        finite_shot=finite_shot,
        confidence_level=confidence,
        requires_hardware_approval=capability.hardware,
        reasons=tuple(reasons) or ("no supported gradient method selected",),
        alternatives=tuple(dict.fromkeys(alternatives)),
    )


__all__ = [
    "GradientMethod",
    "QuantumGradientBackendCapability",
    "QuantumGradientPlan",
    "plan_quantum_gradient_backend",
    "quantum_gradient_backend_capability",
]
