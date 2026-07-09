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


@dataclass(frozen=True)
class QuantumGradientRejectedMethod:
    """Rejected method candidate from a deterministic backend planner explanation.

    Parameters
    ----------
    method
        Candidate method that was considered and not selected.
    reasons
        Deterministic reasons the candidate was not selected.
    supported_if_requested
        Whether the candidate would be executable if requested directly with
        the same backend capability and shot controls.
    """

    method: str
    reasons: tuple[str, ...]
    supported_if_requested: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready rejected-method metadata."""
        return {
            "method": self.method,
            "reasons": list(self.reasons),
            "supported_if_requested": self.supported_if_requested,
        }


@dataclass(frozen=True)
class QuantumGradientShotPolicy:
    """Shot and uncertainty policy attached to a planner explanation.

    Parameters
    ----------
    finite_shot
        Whether the selected plan consumes finite-shot samples.
    requested_shots
        Caller-supplied shot count before planner defaults are applied.
    planned_shots
        Shot count in the selected plan after defaults and validation.
    defaulted
        Whether the planner supplied a backend default shot count.
    confidence_level
        Confidence level used by finite-shot uncertainty metadata.
    seed
        Optional deterministic seed for stochastic planners.
    reasons
        Human-readable shot-policy explanation.
    """

    finite_shot: bool
    requested_shots: int | None
    planned_shots: int | None
    defaulted: bool
    confidence_level: float | None
    seed: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready shot-policy metadata."""
        return {
            "finite_shot": self.finite_shot,
            "requested_shots": self.requested_shots,
            "planned_shots": self.planned_shots,
            "defaulted": self.defaulted,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class QuantumGradientMethodExplanation:
    """Deterministic explanation of a backend gradient-method decision.

    Parameters
    ----------
    capability
        Normalised backend capability used by the planner.
    selected_plan
        Existing execution plan selected by
        :func:`plan_quantum_gradient_backend`.
    rejected_methods
        Ordered method candidates that were not selected.
    shot_policy
        Shot, confidence, and seed policy for the selected plan.
    fallback_path
        Ordered safe fallback routes for unsupported or degraded execution.
    requested_method
        Normalised method requested by the caller.
    claim_boundary
        Claim boundary for this explanation object.
    """

    capability: QuantumGradientBackendCapability
    selected_plan: QuantumGradientPlan
    rejected_methods: tuple[QuantumGradientRejectedMethod, ...]
    shot_policy: QuantumGradientShotPolicy
    fallback_path: tuple[str, ...]
    requested_method: str
    claim_boundary: str = (
        "backend planner explanation only; provider execution, hardware gradients, "
        "benchmark promotion, and framework-specific transforms require separate evidence"
    )

    @property
    def selected_method(self) -> str:
        """Return the method selected by the wrapped backend plan."""
        return self.selected_plan.method

    @property
    def supported(self) -> bool:
        """Return whether the selected plan is executable."""
        return self.selected_plan.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready explanation metadata."""
        return {
            "backend": self.capability.backend,
            "family": self.capability.family,
            "requested_method": self.requested_method,
            "selected_method": self.selected_method,
            "supported": self.supported,
            "selected_plan": _plan_to_dict(self.selected_plan),
            "rejected_methods": [method.to_dict() for method in self.rejected_methods],
            "shot_policy": self.shot_policy.to_dict(),
            "fallback_path": list(self.fallback_path),
            "claim_boundary": self.claim_boundary,
        }


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
    "ibm_brisbane",
    "ibm_fez",
    "ibm_kingston",
    "braket",
    "rigetti",
    "ionq",
    "quantinuum",
    "quera",
    "pasqal",
    "quandela",
    "pennylane_device",
}

_EXPLANATION_METHOD_ORDER: tuple[GradientMethod, ...] = (
    "parameter_shift",
    "stochastic_parameter_shift",
    "spsa",
    "finite_difference",
)


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


def explain_quantum_gradient_method(
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
) -> QuantumGradientMethodExplanation:
    """Explain a backend gradient-method decision without executing gradients.

    Parameters
    ----------
    backend
        Backend family or alias to plan against.
    n_params
        Number of trainable scalar parameters.
    shift_terms
        Number of parameter-shift terms per trainable parameter.
    method
        Requested method, or ``"auto"`` for planner selection.
    shots
        Optional finite-shot budget.
    seed
        Optional deterministic seed for stochastic routes.
    finite_shot
        Whether the caller requires finite-shot planning.
    confidence_level
        Optional finite-shot confidence level.
    allow_hardware
        Whether policy-gated hardware routes may plan execution.

    Returns
    -------
    QuantumGradientMethodExplanation
        Deterministic selected method, rejected alternatives, shot policy, and
        fallback path for the requested backend capability combination.
    """
    selected_plan = plan_quantum_gradient_backend(
        backend,
        n_params=n_params,
        shift_terms=shift_terms,
        method=method,
        shots=shots,
        seed=seed,
        finite_shot=finite_shot,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )
    capability = quantum_gradient_backend_capability(backend)
    requested_method = _normalise_method(method)
    candidates = tuple(
        candidate for candidate in _EXPLANATION_METHOD_ORDER if candidate != selected_plan.method
    )
    rejected = tuple(
        _explain_rejected_method(
            candidate,
            selected_plan=selected_plan,
            requested_method=requested_method,
            backend=backend,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            seed=seed,
            finite_shot=finite_shot,
            confidence_level=confidence_level,
            allow_hardware=allow_hardware,
        )
        for candidate in candidates
    )
    return QuantumGradientMethodExplanation(
        capability=capability,
        selected_plan=selected_plan,
        rejected_methods=rejected,
        shot_policy=_shot_policy(selected_plan, requested_shots=shots),
        fallback_path=_fallback_path(selected_plan, rejected),
        requested_method=requested_method,
    )


def _explain_rejected_method(
    method: GradientMethod,
    *,
    selected_plan: QuantumGradientPlan,
    requested_method: GradientMethod,
    backend: str,
    n_params: int,
    shift_terms: int,
    shots: int | None,
    seed: int | None,
    finite_shot: bool,
    confidence_level: float | None,
    allow_hardware: bool,
) -> QuantumGradientRejectedMethod:
    """Return the deterministic rejection row for one candidate method."""
    candidate_plan = plan_quantum_gradient_backend(
        backend,
        n_params=n_params,
        shift_terms=shift_terms,
        method=method,
        shots=shots,
        seed=seed,
        finite_shot=finite_shot,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )
    reasons = _candidate_rejection_reasons(
        method,
        candidate_plan=candidate_plan,
        selected_plan=selected_plan,
        requested_method=requested_method,
    )
    return QuantumGradientRejectedMethod(
        method=method,
        reasons=reasons,
        supported_if_requested=candidate_plan.supported,
    )


def _candidate_rejection_reasons(
    method: GradientMethod,
    *,
    candidate_plan: QuantumGradientPlan,
    selected_plan: QuantumGradientPlan,
    requested_method: GradientMethod,
) -> tuple[str, ...]:
    """Build stable reasons for a non-selected candidate method."""
    if not candidate_plan.supported:
        return candidate_plan.reasons
    if requested_method != "auto" and method != requested_method:
        return (f"caller explicitly requested {requested_method}",)
    if selected_plan.method == "parameter_shift":
        return ("deterministic local parameter-shift route has lower estimator noise",)
    if selected_plan.method == "stochastic_parameter_shift":
        return ("finite-shot planning requires explicit shot uncertainty metadata",)
    if selected_plan.method == "spsa":
        return ("caller selected SPSA diagnostic fallback",)
    return (f"{selected_plan.method} selected by backend planner",)


def _shot_policy(
    selected_plan: QuantumGradientPlan,
    *,
    requested_shots: int | None,
) -> QuantumGradientShotPolicy:
    """Return shot metadata for a selected backend plan."""
    if selected_plan.fail_closed:
        return QuantumGradientShotPolicy(
            finite_shot=selected_plan.finite_shot,
            requested_shots=requested_shots,
            planned_shots=selected_plan.shots,
            defaulted=False,
            confidence_level=selected_plan.confidence_level,
            seed=selected_plan.seed,
            reasons=("unsupported plan does not allocate executable shots",),
        )
    if selected_plan.finite_shot:
        defaulted = requested_shots is None and selected_plan.shots is not None
        reason = (
            "finite-shot route uses backend default shots"
            if defaulted
            else "finite-shot route uses caller supplied shots"
        )
        return QuantumGradientShotPolicy(
            finite_shot=True,
            requested_shots=requested_shots,
            planned_shots=selected_plan.shots,
            defaulted=defaulted,
            confidence_level=selected_plan.confidence_level,
            seed=selected_plan.seed,
            reasons=(reason, "confidence metadata is required for uncertainty reporting"),
        )
    return QuantumGradientShotPolicy(
        finite_shot=False,
        requested_shots=requested_shots,
        planned_shots=None,
        defaulted=False,
        confidence_level=None,
        seed=selected_plan.seed,
        reasons=("deterministic route does not consume finite-shot samples",),
    )


def _fallback_path(
    selected_plan: QuantumGradientPlan,
    rejected_methods: tuple[QuantumGradientRejectedMethod, ...],
) -> tuple[str, ...]:
    """Return stable fallback routes for the selected planner explanation."""
    if selected_plan.fail_closed:
        return tuple(dict.fromkeys(selected_plan.alternatives))
    supported_methods = tuple(
        method.method for method in rejected_methods if method.supported_if_requested
    )
    if selected_plan.method == "parameter_shift":
        return tuple(dict.fromkeys((*supported_methods, "finite_difference_diagnostic")))
    if selected_plan.method == "stochastic_parameter_shift":
        return tuple(dict.fromkeys((*supported_methods, "increase_shots_or_use_statevector")))
    if selected_plan.method == "spsa":
        return tuple(dict.fromkeys((*supported_methods, "stochastic_parameter_shift")))
    return supported_methods


def _plan_to_dict(plan: QuantumGradientPlan) -> dict[str, object]:
    """Return JSON-ready backend-plan metadata."""
    return {
        "backend": plan.backend,
        "family": plan.family,
        "method": plan.method,
        "supported": plan.supported,
        "n_params": plan.n_params,
        "shift_terms": plan.shift_terms,
        "evaluations": plan.evaluations,
        "shots": plan.shots,
        "seed": plan.seed,
        "finite_shot": plan.finite_shot,
        "confidence_level": plan.confidence_level,
        "requires_hardware_approval": plan.requires_hardware_approval,
        "reasons": list(plan.reasons),
        "alternatives": list(plan.alternatives),
    }


__all__ = [
    "GradientMethod",
    "QuantumGradientBackendCapability",
    "QuantumGradientMethodExplanation",
    "QuantumGradientPlan",
    "QuantumGradientRejectedMethod",
    "QuantumGradientShotPolicy",
    "explain_quantum_gradient_method",
    "plan_quantum_gradient_backend",
    "quantum_gradient_backend_capability",
]
