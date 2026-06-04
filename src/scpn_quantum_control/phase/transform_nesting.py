# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gradient Transform Nesting
"""Fail-closed transform-nesting planner for quantum gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .gradient_support_matrix import GradientSupportPlan, plan_gradient_support

TransformName = Literal[
    "grad",
    "value_and_grad",
    "hessian",
    "gradient_tape",
    "jvp",
    "vjp",
    "jacfwd",
    "jacrev",
    "vmap",
]


@dataclass(frozen=True)
class GradientTransformNestingPlan:
    """Support decision for a nested quantum-gradient transform request."""

    transforms: tuple[str, ...]
    gate: str
    observable: str
    backend: str
    adapter: str
    supported: bool
    strategy: str
    differentiation_order: int
    support_plan: GradientSupportPlan
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    alternatives: tuple[str, ...]
    requires_deterministic_backend: bool
    requires_host_boundary: bool
    claim_boundary: str

    @property
    def fail_closed(self) -> bool:
        """Return true when the nested transform must not execute."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready transform-nesting metadata."""
        return {
            "transforms": list(self.transforms),
            "gate": self.gate,
            "observable": self.observable,
            "backend": self.backend,
            "adapter": self.adapter,
            "supported": self.supported,
            "strategy": self.strategy,
            "differentiation_order": self.differentiation_order,
            "support_plan": self.support_plan.to_dict(),
            "blocked_reasons": list(self.blocked_reasons),
            "warnings": list(self.warnings),
            "alternatives": list(self.alternatives),
            "requires_deterministic_backend": self.requires_deterministic_backend,
            "requires_host_boundary": self.requires_host_boundary,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GradientTransformNestingAuditResult:
    """Built-in audit of supported and blocked transform-nesting routes."""

    plans: tuple[GradientTransformNestingPlan, ...]
    passed: bool
    claim_boundary: str

    @property
    def supported_plans(self) -> tuple[GradientTransformNestingPlan, ...]:
        """Return audit plans that are supported."""
        return tuple(plan for plan in self.plans if plan.supported)

    @property
    def blocked_plans(self) -> tuple[GradientTransformNestingPlan, ...]:
        """Return audit plans that fail closed."""
        return tuple(plan for plan in self.plans if plan.fail_closed)

    @property
    def failing_plans(self) -> tuple[GradientTransformNestingPlan, ...]:
        """Return plans that violate the built-in support expectations."""
        expected_supported = {
            ("grad", "native", "statevector"),
            ("value_and_grad", "native", "statevector"),
            ("hessian", "native", "statevector"),
            ("grad.grad", "native", "statevector"),
            ("value_and_grad", "jax", "statevector"),
            ("gradient_tape", "native", "statevector"),
        }
        failures: list[GradientTransformNestingPlan] = []
        for plan in self.plans:
            key = (".".join(plan.transforms), plan.adapter, plan.backend)
            if key in expected_supported and not plan.supported:
                failures.append(plan)
            if key not in expected_supported and plan.supported:
                failures.append(plan)
        return tuple(failures)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit metadata."""
        return {
            "plans": [plan.to_dict() for plan in self.plans],
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


_ALIASES = {
    "value_grad": "value_and_grad",
    "value-and-grad": "value_and_grad",
    "tape": "gradient_tape",
    "grad_tape": "gradient_tape",
    "jacobian_forward": "jacfwd",
    "jacobian_reverse": "jacrev",
    "batch": "vmap",
}
_SUPPORTED_TRANSFORMS = {
    "grad",
    "value_and_grad",
    "hessian",
    "gradient_tape",
    "jvp",
    "vjp",
    "jacfwd",
    "jacrev",
    "vmap",
}


def plan_gradient_transform_nesting(
    transforms: str | tuple[str, ...],
    *,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
) -> GradientTransformNestingPlan:
    """Plan a nested quantum-gradient transform stack with fail-closed rules."""
    normalised = _normalise_transforms(transforms)
    canonical_transform = _canonical_support_transform(normalised)
    support_plan = plan_gradient_support(
        gate=gate,
        observable=observable,
        backend=backend,
        transform=canonical_transform,
        adapter=adapter,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        allow_hardware=allow_hardware,
    )
    blocked_reasons = list(support_plan.blocked_reasons)
    warnings = list(support_plan.warnings)
    alternatives = list(support_plan.alternatives)
    strategy = _strategy(normalised, support_plan)
    differentiation_order = _differentiation_order(normalised)

    if len(normalised) > 2:
        blocked_reasons.append("transform nesting depth greater than two is not supported")
        alternatives.extend(("grad", "hessian"))
    if not _all_known(normalised):
        blocked_reasons.append("unknown transform in nesting stack")
        alternatives.extend(_known_transform_alternatives())
    if "vmap" in normalised:
        blocked_reasons.append("vmap over quantum-gradient executions is not implemented")
        alternatives.extend(("manual loop", "batched classical post-processing"))
    if "jvp" in normalised or "vjp" in normalised:
        blocked_reasons.append("quantum-gradient jvp/vjp transform execution is not implemented")
        alternatives.extend(("grad", "value_and_grad"))
    if "jacfwd" in normalised or "jacrev" in normalised:
        blocked_reasons.append("jacfwd/jacrev quantum transform algebra is not implemented")
        alternatives.extend(("grad", "hessian"))
    if "gradient_tape" in normalised and len(normalised) > 1:
        blocked_reasons.append(
            "gradient tape records supported evaluations but is not itself nestable"
        )
        alternatives.extend(("gradient_tape", "grad"))
    if normalised in {("grad", "grad"), ("hessian",)} and support_plan.backend_plan.finite_shot:
        blocked_reasons.append(
            "second-order transform nesting requires deterministic local expectations"
        )
        alternatives.extend(("statevector", "grad"))
    if (
        normalised in {("grad", "grad"), ("hessian",)}
        and support_plan.backend_plan.requires_hardware_approval
    ):
        blocked_reasons.append("second-order hardware-gradient execution is policy-blocked")
        alternatives.extend(("statevector", "grad"))
    if normalised == ("grad", "grad") and adapter != "native":
        blocked_reasons.append("nested grad-of-grad is supported only on the native local route")
        alternatives.extend(("native", "hessian"))
    if normalised == ("hessian",) and adapter != "native":
        blocked_reasons.append("hessian transform is supported only on the native local route")
        alternatives.extend(("native", "grad"))
    if adapter != "native" and len(normalised) > 1:
        blocked_reasons.append(
            "ML/provider adapters support only declared single-transform bridge surfaces"
        )
        alternatives.extend(("native", normalised[-1]))

    supported = not blocked_reasons
    return GradientTransformNestingPlan(
        transforms=normalised,
        gate=support_plan.gate,
        observable=support_plan.observable,
        backend=support_plan.backend,
        adapter=support_plan.adapter,
        supported=supported,
        strategy="unsupported" if not supported else strategy,
        differentiation_order=differentiation_order,
        support_plan=support_plan,
        blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
        warnings=tuple(dict.fromkeys(warnings + list(_nesting_warnings(normalised, adapter)))),
        alternatives=tuple(dict.fromkeys(alternatives)),
        requires_deterministic_backend=normalised in {("grad", "grad"), ("hessian",)},
        requires_host_boundary=adapter != "native",
        claim_boundary=_claim_boundary(normalised, adapter, supported),
    )


def assert_gradient_transform_nesting_supported(
    plan: GradientTransformNestingPlan,
) -> GradientTransformNestingPlan:
    """Return a supported nesting plan or raise with fail-closed reasons."""
    if plan.fail_closed:
        raise ValueError("; ".join(plan.blocked_reasons))
    return plan


def run_gradient_transform_nesting_audit() -> GradientTransformNestingAuditResult:
    """Run representative transform-nesting support and fail-closed checks."""
    plans = (
        plan_gradient_transform_nesting("grad", n_params=2),
        plan_gradient_transform_nesting("value_and_grad", n_params=2),
        plan_gradient_transform_nesting("hessian", n_params=2),
        plan_gradient_transform_nesting(("grad", "grad"), n_params=2),
        plan_gradient_transform_nesting("value_grad", adapter="jax", n_params=2),
        plan_gradient_transform_nesting("gradient_tape", n_params=2),
        plan_gradient_transform_nesting(("vmap", "grad"), adapter="jax", n_params=2),
        plan_gradient_transform_nesting(
            "hessian", backend="qasm_simulator", n_params=2, shots=400
        ),
        plan_gradient_transform_nesting(("grad", "grad"), adapter="pytorch", n_params=2),
        plan_gradient_transform_nesting(("gradient_tape", "grad"), n_params=2),
        plan_gradient_transform_nesting("jvp", n_params=2),
        plan_gradient_transform_nesting("jacrev", n_params=2),
        plan_gradient_transform_nesting("grad", backend="hardware", n_params=2, shots=1024),
    )
    result = GradientTransformNestingAuditResult(
        plans=plans,
        passed=False,
        claim_boundary=(
            "transform-nesting audit only; supported routes are bounded local or "
            "single-adapter gradient surfaces, blocked routes are fail-closed planning "
            "evidence, and no universal transform algebra or hardware-gradient claim is implied"
        ),
    )
    return GradientTransformNestingAuditResult(
        plans=plans,
        passed=not result.failing_plans,
        claim_boundary=result.claim_boundary,
    )


def _normalise_transforms(transforms: str | tuple[str, ...]) -> tuple[str, ...]:
    raw = (transforms,) if isinstance(transforms, str) else transforms
    if not raw:
        raise ValueError("at least one transform is required")
    normalised: list[str] = []
    for transform in raw:
        key = transform.strip().lower().replace("-", "_").replace(".", "_")
        if not key:
            raise ValueError("transform names must be non-empty")
        normalised.append(_ALIASES.get(key, key))
    return tuple(normalised)


def _canonical_support_transform(transforms: tuple[str, ...]) -> str:
    if transforms == ("grad", "grad"):
        return "hessian"
    if transforms == ("gradient_tape",):
        return "gradient_tape"
    if transforms and transforms[-1] in {"grad", "value_and_grad", "hessian", "gradient_tape"}:
        return transforms[-1]
    return transforms[-1]


def _all_known(transforms: tuple[str, ...]) -> bool:
    return all(transform in _SUPPORTED_TRANSFORMS for transform in transforms)


def _differentiation_order(transforms: tuple[str, ...]) -> int:
    if transforms in {("hessian",), ("grad", "grad")}:
        return 2
    if any(transform in {"jacfwd", "jacrev"} for transform in transforms):
        return 1
    if any(transform in {"grad", "value_and_grad", "jvp", "vjp"} for transform in transforms):
        return 1
    return 0


def _strategy(transforms: tuple[str, ...], support_plan: GradientSupportPlan) -> str:
    if transforms == ("grad",):
        return support_plan.recommended_method
    if transforms == ("value_and_grad",):
        return support_plan.recommended_method
    if transforms == ("hessian",):
        return "native_parameter_shift_hessian"
    if transforms == ("grad", "grad"):
        return "native_hessian_via_nested_grad"
    if transforms == ("gradient_tape",):
        return "record_supported_parameter_shift_tape"
    return "unsupported"


def _nesting_warnings(transforms: tuple[str, ...], adapter: str) -> tuple[str, ...]:
    warnings: list[str] = []
    if transforms in {("grad", "grad"), ("hessian",)}:
        warnings.append("second-order routes are deterministic local diagnostics")
    if adapter != "native":
        warnings.append("adapter route crosses an explicit host or agreement boundary")
    if transforms == ("gradient_tape",):
        warnings.append(
            "gradient tape is a record/replay evidence surface, not arbitrary program AD"
        )
    return tuple(warnings)


def _known_transform_alternatives() -> tuple[str, ...]:
    return ("grad", "value_and_grad", "hessian", "gradient_tape")


def _claim_boundary(transforms: tuple[str, ...], adapter: str, supported: bool) -> str:
    if not supported:
        return "unsupported transform nesting; no derivative execution or production claim is permitted"
    if transforms in {("grad", "grad"), ("hessian",)}:
        return "deterministic local second-order diagnostic; no hardware Hessian or universal curvature claim is implied"
    if transforms == ("gradient_tape",):
        return "supported phase-gradient tape records only; arbitrary program tape nesting remains outside this surface"
    if adapter != "native":
        return f"{adapter} single-transform bridge support; arbitrary nested framework autodiff is not claimed"
    return "bounded first-order local quantum-gradient transform support"
