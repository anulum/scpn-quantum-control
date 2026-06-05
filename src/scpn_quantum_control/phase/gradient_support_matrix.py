# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gradient Support Matrix
"""Executable support matrix for quantum-gradient combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend

SupportCategory = Literal["gate", "observable", "backend", "transform", "adapter"]


@dataclass(frozen=True)
class GradientSupportCapability:
    """Declared support contract for one gradient surface component."""

    category: SupportCategory
    name: str
    supported: bool
    gradient_methods: tuple[str, ...]
    conditions: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    alternatives: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready capability metadata."""
        return {
            "category": self.category,
            "name": self.name,
            "supported": self.supported,
            "gradient_methods": list(self.gradient_methods),
            "conditions": list(self.conditions),
            "blocked_reasons": list(self.blocked_reasons),
            "alternatives": list(self.alternatives),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GradientSupportPlan:
    """Fail-closed support decision for a complete gradient request."""

    gate: str
    observable: str
    backend: str
    transform: str
    adapter: str
    supported: bool
    recommended_method: str
    evaluation_mode: str
    capabilities: tuple[GradientSupportCapability, ...]
    backend_plan: QuantumGradientPlan
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    alternatives: tuple[str, ...]
    requires_finite_shot_variance: bool
    requires_hardware_policy: bool
    claim_boundary: str

    @property
    def fail_closed(self) -> bool:
        """Return true when this request is intentionally unsupported."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready support-plan metadata."""
        return {
            "gate": self.gate,
            "observable": self.observable,
            "backend": self.backend,
            "transform": self.transform,
            "adapter": self.adapter,
            "supported": self.supported,
            "recommended_method": self.recommended_method,
            "evaluation_mode": self.evaluation_mode,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "backend_plan": _backend_plan_to_dict(self.backend_plan),
            "blocked_reasons": list(self.blocked_reasons),
            "warnings": list(self.warnings),
            "alternatives": list(self.alternatives),
            "requires_finite_shot_variance": self.requires_finite_shot_variance,
            "requires_hardware_policy": self.requires_hardware_policy,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GradientSupportMatrixAuditResult:
    """Built-in audit for representative supported and blocked combinations."""

    plans: tuple[GradientSupportPlan, ...]
    passed: bool
    claim_boundary: str

    @property
    def supported_plans(self) -> tuple[GradientSupportPlan, ...]:
        """Return supported audit plans."""
        return tuple(plan for plan in self.plans if plan.supported)

    @property
    def blocked_plans(self) -> tuple[GradientSupportPlan, ...]:
        """Return fail-closed audit plans."""
        return tuple(plan for plan in self.plans if plan.fail_closed)

    @property
    def failing_plans(self) -> tuple[GradientSupportPlan, ...]:
        """Return audit plans that violate expected support invariants."""
        expected_supported = {
            ("ry", "pauli_expectation", "statevector", "grad", "native"),
            ("rz", "kuramoto_xy_energy", "qasm_simulator", "grad", "native"),
            ("ry", "pauli_expectation", "statevector", "value_and_grad", "jax"),
            ("rx", "sparse_pauli_sum", "statevector", "grad", "qiskit"),
        }
        failures: list[GradientSupportPlan] = []
        for plan in self.plans:
            key = (plan.gate, plan.observable, plan.backend, plan.transform, plan.adapter)
            if key in expected_supported and not plan.supported:
                failures.append(plan)
            if key not in expected_supported and plan.supported:
                failures.append(plan)
        return tuple(failures)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready support-matrix audit metadata."""
        return {
            "plans": [plan.to_dict() for plan in self.plans],
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


_GATE_CAPABILITIES: dict[str, GradientSupportCapability] = {
    "rx": GradientSupportCapability(
        category="gate",
        name="rx",
        supported=True,
        gradient_methods=("parameter_shift", "multi_frequency_parameter_shift"),
        conditions=("single-parameter Pauli rotation with declared generator spectrum",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="gate-level parameter-shift support only; circuit topology must preserve parameter indexing",
    ),
    "ry": GradientSupportCapability(
        category="gate",
        name="ry",
        supported=True,
        gradient_methods=("parameter_shift", "multi_frequency_parameter_shift"),
        conditions=("single-parameter Pauli rotation with declared generator spectrum",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="gate-level parameter-shift support only; circuit topology must preserve parameter indexing",
    ),
    "rz": GradientSupportCapability(
        category="gate",
        name="rz",
        supported=True,
        gradient_methods=("parameter_shift", "multi_frequency_parameter_shift"),
        conditions=("single-parameter Pauli rotation with declared generator spectrum",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="gate-level parameter-shift support only; circuit topology must preserve parameter indexing",
    ),
    "phase_rotation": GradientSupportCapability(
        category="gate",
        name="phase_rotation",
        supported=True,
        gradient_methods=("parameter_shift",),
        conditions=("smooth periodic phase objective with stable parameter map",),
        blocked_reasons=(),
        alternatives=("rx", "ry", "rz"),
        claim_boundary="phase-objective support only; arbitrary hardware pulse calibration is outside this matrix",
    ),
    "controlled_phase": GradientSupportCapability(
        category="gate",
        name="controlled_phase",
        supported=True,
        gradient_methods=("parameter_shift",),
        conditions=("trainable controlled phase with declared generator spectrum",),
        blocked_reasons=(),
        alternatives=("rz", "phase_rotation"),
        claim_boundary="controlled phase support requires explicit parameter binding before execution",
    ),
    "cz": GradientSupportCapability(
        category="gate",
        name="cz",
        supported=True,
        gradient_methods=("non_trainable_topology",),
        conditions=(
            "allowed as a fixed entangling gate; no trainable derivative is emitted for the gate itself",
        ),
        blocked_reasons=(),
        alternatives=("controlled_phase",),
        claim_boundary="fixed topology gate support only; trainable CZ-like rotations must use controlled_phase",
    ),
}

_OBSERVABLE_CAPABILITIES: dict[str, GradientSupportCapability] = {
    "pauli_expectation": GradientSupportCapability(
        category="observable",
        name="pauli_expectation",
        supported=True,
        gradient_methods=("parameter_shift", "stochastic_parameter_shift"),
        conditions=("bounded expectation value with declared shots for stochastic routes",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="expectation-gradient support only; tomography and detector calibration remain separate evidence",
    ),
    "sparse_pauli_sum": GradientSupportCapability(
        category="observable",
        name="sparse_pauli_sum",
        supported=True,
        gradient_methods=("parameter_shift", "stochastic_parameter_shift"),
        conditions=(
            "term weights finite and each term has compatible shifted expectation records",
        ),
        blocked_reasons=(),
        alternatives=("pauli_expectation",),
        claim_boundary="linear Pauli-sum support only; non-Pauli observables require adapters or decomposition evidence",
    ),
    "kuramoto_xy_energy": GradientSupportCapability(
        category="observable",
        name="kuramoto_xy_energy",
        supported=True,
        gradient_methods=("parameter_shift", "term_gradient"),
        conditions=("fixed coupling topology and stable phase-parameter ordering",),
        blocked_reasons=(),
        alternatives=("sparse_pauli_sum", "pauli_expectation"),
        claim_boundary="Kuramoto-XY energy-gradient support only; measured-system validation is a separate claim gate",
    ),
}

_TRANSFORM_CAPABILITIES: dict[str, GradientSupportCapability] = {
    "grad": GradientSupportCapability(
        category="transform",
        name="grad",
        supported=True,
        gradient_methods=("parameter_shift", "stochastic_parameter_shift"),
        conditions=("scalar objective with stable trainable-parameter vector",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="first-order scalar-gradient support only",
    ),
    "value_and_grad": GradientSupportCapability(
        category="transform",
        name="value_and_grad",
        supported=True,
        gradient_methods=("parameter_shift", "stochastic_parameter_shift"),
        conditions=("scalar value and gradient evaluated on the same parameter vector",),
        blocked_reasons=(),
        alternatives=("grad",),
        claim_boundary="value-plus-gradient support only; transform nesting is bounded by adapter policy",
    ),
    "hessian": GradientSupportCapability(
        category="transform",
        name="hessian",
        supported=True,
        gradient_methods=("parameter_shift_hessian",),
        conditions=("deterministic local backend and standard shift-compatible smooth objective",),
        blocked_reasons=(),
        alternatives=("grad", "finite_difference_diagnostic"),
        claim_boundary="local curvature diagnostic only; not a universal quantum Fisher or hardware Hessian claim",
    ),
    "gradient_tape": GradientSupportCapability(
        category="transform",
        name="gradient_tape",
        supported=True,
        gradient_methods=("recorded_parameter_shift",),
        conditions=("supported phase parameter-shift records only",),
        blocked_reasons=(),
        alternatives=("grad", "value_and_grad"),
        claim_boundary="phase-gradient tape support only; arbitrary Python tape semantics remain outside this matrix",
    ),
    "jvp": GradientSupportCapability(
        category="transform",
        name="jvp",
        supported=True,
        gradient_methods=("parameter_shift_directional_derivative",),
        conditions=(
            "scalar objective, deterministic local backend, and tangent matching parameter shape",
        ),
        blocked_reasons=(),
        alternatives=("grad", "value_and_grad"),
        claim_boundary="scalar local JVP via parameter-shift gradient only; not arbitrary program JVP",
    ),
    "vjp": GradientSupportCapability(
        category="transform",
        name="vjp",
        supported=True,
        gradient_methods=("parameter_shift_pullback",),
        conditions=("scalar objective, deterministic local backend, and scalar cotangent",),
        blocked_reasons=(),
        alternatives=("grad", "value_and_grad"),
        claim_boundary="scalar local VJP via parameter-shift gradient only; not arbitrary program VJP",
    ),
    "jacfwd": GradientSupportCapability(
        category="transform",
        name="jacfwd",
        supported=True,
        gradient_methods=("parameter_shift_scalar_jacobian",),
        conditions=("scalar objective with stable parameter vector; returns one-row Jacobian",),
        blocked_reasons=(),
        alternatives=("grad", "hessian"),
        claim_boundary="scalar local Jacobian via parameter-shift gradient only; not full vector-output jacfwd",
    ),
    "jacrev": GradientSupportCapability(
        category="transform",
        name="jacrev",
        supported=True,
        gradient_methods=("parameter_shift_scalar_jacobian",),
        conditions=("scalar objective with stable parameter vector; returns one-row Jacobian",),
        blocked_reasons=(),
        alternatives=("grad", "hessian"),
        claim_boundary="scalar local Jacobian via parameter-shift gradient only; not full vector-output jacrev",
    ),
}

_ADAPTER_CAPABILITIES: dict[str, GradientSupportCapability] = {
    "native": GradientSupportCapability(
        category="adapter",
        name="native",
        supported=True,
        gradient_methods=(
            "parameter_shift",
            "stochastic_parameter_shift",
            "parameter_shift_hessian",
            "parameter_shift_directional_derivative",
            "parameter_shift_pullback",
            "parameter_shift_scalar_jacobian",
            "manual_vmap_parameter_shift_grad",
        ),
        conditions=("NumPy-compatible callable or phase namespace objective",),
        blocked_reasons=(),
        alternatives=(),
        claim_boundary="native Python/NumPy route; no ML-framework tracing claim is implied",
    ),
    "jax": GradientSupportCapability(
        category="adapter",
        name="jax",
        supported=True,
        gradient_methods=("host_callback_parameter_shift",),
        conditions=("value_and_grad or grad through explicit host callback boundary",),
        blocked_reasons=(),
        alternatives=("native",),
        claim_boundary="JAX host-callback bridge only; native JAX tracing through arbitrary quantum simulators is not claimed",
    ),
    "pytorch": GradientSupportCapability(
        category="adapter",
        name="pytorch",
        supported=True,
        gradient_methods=("tensor_bridge_parameter_shift",),
        conditions=("tensor conversion around supported host parameter-shift result",),
        blocked_reasons=(),
        alternatives=("native",),
        claim_boundary="PyTorch tensor bridge only; autograd through arbitrary provider execution is not claimed",
    ),
    "tensorflow": GradientSupportCapability(
        category="adapter",
        name="tensorflow",
        supported=True,
        gradient_methods=("tensor_bridge_parameter_shift",),
        conditions=("tensor conversion around supported host parameter-shift result",),
        blocked_reasons=(),
        alternatives=("native",),
        claim_boundary="TensorFlow tensor bridge only; graph-mode provider autodiff is not claimed",
    ),
    "pennylane": GradientSupportCapability(
        category="adapter",
        name="pennylane",
        supported=True,
        gradient_methods=("agreement_check",),
        conditions=("caller supplies PennyLane value and gradient for agreement checks",),
        blocked_reasons=(),
        alternatives=("native", "qiskit"),
        claim_boundary="PennyLane agreement/round-trip evidence only; full QNode migration is not claimed",
    ),
    "qiskit": GradientSupportCapability(
        category="adapter",
        name="qiskit",
        supported=True,
        gradient_methods=("shifted_circuit_generation", "statevector_parameter_shift"),
        conditions=(
            "fully bound shifted circuits and local execution or explicit provider callback",
        ),
        blocked_reasons=(),
        alternatives=("native",),
        claim_boundary="Qiskit shifted-circuit/local execution support; live hardware submission remains policy-gated",
    ),
}

_UNSUPPORTED_CAPABILITIES: dict[SupportCategory, GradientSupportCapability] = {
    "gate": GradientSupportCapability(
        category="gate",
        name="unsupported_gate",
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=("gate has no registered parameter-shift generator spectrum",),
        alternatives=("rx", "ry", "rz", "phase_rotation", "controlled_phase"),
        claim_boundary="unsupported gate; no derivative execution is permitted",
    ),
    "observable": GradientSupportCapability(
        category="observable",
        name="unsupported_observable",
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=("observable has no registered expectation-gradient contract",),
        alternatives=("pauli_expectation", "sparse_pauli_sum", "kuramoto_xy_energy"),
        claim_boundary="unsupported observable; no derivative execution is permitted",
    ),
    "backend": GradientSupportCapability(
        category="backend",
        name="unsupported_backend",
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=("backend has no registered gradient execution policy",),
        alternatives=("statevector_simulator", "finite_shot_simulator"),
        claim_boundary="unsupported backend; no derivative execution is permitted",
    ),
    "transform": GradientSupportCapability(
        category="transform",
        name="unsupported_transform",
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=("transform is outside the bounded quantum-gradient algebra",),
        alternatives=("grad", "value_and_grad", "hessian", "gradient_tape"),
        claim_boundary="unsupported transform; no derivative execution is permitted",
    ),
    "adapter": GradientSupportCapability(
        category="adapter",
        name="unsupported_adapter",
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=("adapter has no registered quantum-gradient bridge",),
        alternatives=("native", "jax", "pytorch", "tensorflow", "pennylane", "qiskit"),
        claim_boundary="unsupported adapter; no derivative execution is permitted",
    ),
}

_ALIAS_MAP: dict[SupportCategory, dict[str, str]] = {
    "gate": {
        "x_rotation": "rx",
        "y_rotation": "ry",
        "z_rotation": "rz",
        "phase": "phase_rotation",
        "cp": "controlled_phase",
        "crz": "controlled_phase",
    },
    "observable": {
        "z": "pauli_expectation",
        "x": "pauli_expectation",
        "y": "pauli_expectation",
        "pauli": "pauli_expectation",
        "hamiltonian": "sparse_pauli_sum",
        "xy_energy": "kuramoto_xy_energy",
    },
    "backend": {
        "simulator": "statevector",
        "statevector_simulator": "statevector",
        "exact": "statevector",
        "finite_shot_simulator": "qasm_simulator",
        "qasm": "qasm_simulator",
        "aer": "qasm_simulator",
        "hardware_qpu": "hardware",
        "qpu": "hardware",
        "ibm_quantum": "hardware",
    },
    "transform": {
        "value_grad": "value_and_grad",
        "tape": "gradient_tape",
    },
    "adapter": {
        "torch": "pytorch",
        "tf": "tensorflow",
        "qml": "pennylane",
    },
}


def list_gradient_support_capabilities(
    category: SupportCategory | None = None,
) -> tuple[GradientSupportCapability, ...]:
    """List registered support capabilities, optionally filtered by category."""
    registries = {
        "gate": _GATE_CAPABILITIES,
        "observable": _OBSERVABLE_CAPABILITIES,
        "transform": _TRANSFORM_CAPABILITIES,
        "adapter": _ADAPTER_CAPABILITIES,
    }
    if category is not None:
        _validate_category(category)
        if category == "backend":
            return (
                _backend_capability_from_plan("statevector", n_params=1),
                _backend_capability_from_plan("qasm_simulator", n_params=1, shots=4096),
                _backend_capability_from_plan("hardware", n_params=1, shots=4096),
            )
        return tuple(registries[category].values())
    capabilities: list[GradientSupportCapability] = []
    for selected in ("gate", "observable", "backend", "transform", "adapter"):
        capabilities.extend(list_gradient_support_capabilities(selected))
    return tuple(capabilities)


def gradient_support_capability(
    category: SupportCategory,
    name: str,
) -> GradientSupportCapability:
    """Return the support capability for one matrix component."""
    _validate_category(category)
    key = _normalise_name(category, name)
    if category == "gate":
        return _GATE_CAPABILITIES.get(key, _unsupported(category, key))
    if category == "observable":
        return _OBSERVABLE_CAPABILITIES.get(key, _unsupported(category, key))
    if category == "transform":
        return _TRANSFORM_CAPABILITIES.get(key, _unsupported(category, key))
    if category == "adapter":
        return _ADAPTER_CAPABILITIES.get(key, _unsupported(category, key))
    return _backend_capability_from_plan(key, n_params=1)


def plan_gradient_support(
    *,
    gate: str,
    observable: str,
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
) -> GradientSupportPlan:
    """Plan whether a full quantum-gradient request is supported."""
    if isinstance(n_params, bool) or not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("n_params must be a positive integer")
    if isinstance(shift_terms, bool) or not isinstance(shift_terms, int) or shift_terms <= 0:
        raise ValueError("shift_terms must be a positive integer")

    gate_key = _normalise_name("gate", gate)
    observable_key = _normalise_name("observable", observable)
    backend_key = _normalise_name("backend", backend)
    transform_key = _normalise_name("transform", transform)
    adapter_key = _normalise_name("adapter", adapter)

    backend_plan = plan_quantum_gradient_backend(
        backend_key,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        finite_shot=shots is not None,
        allow_hardware=allow_hardware,
    )
    capabilities = (
        gradient_support_capability("gate", gate_key),
        gradient_support_capability("observable", observable_key),
        _backend_capability_from_plan(
            backend_key, n_params=n_params, shift_terms=shift_terms, shots=shots
        ),
        gradient_support_capability("transform", transform_key),
        gradient_support_capability("adapter", adapter_key),
    )
    blocked_reasons = _blocked_reasons(capabilities, backend_plan)
    warnings = list(_compatibility_warnings(transform_key, adapter_key, backend_plan))
    alternatives = _alternatives(capabilities)

    if transform_key == "hessian" and backend_plan.finite_shot:
        blocked_reasons += ("hessian support is limited to deterministic local backends",)
        alternatives += ("grad", "statevector")
    if transform_key in {"jvp", "vjp", "jacfwd", "jacrev"} and backend_plan.finite_shot:
        blocked_reasons += (
            "directional and scalar-Jacobian transforms require deterministic local expectations",
        )
        alternatives += ("grad", "statevector")
    if adapter_key in {"jax", "pytorch", "tensorflow"} and transform_key not in {
        "grad",
        "value_and_grad",
    }:
        blocked_reasons += (
            f"{adapter_key} bridge supports first-order value/gradient calls only",
        )
        alternatives += ("native", "grad", "value_and_grad")
    if adapter_key == "pennylane" and transform_key not in {"grad", "value_and_grad"}:
        blocked_reasons += ("pennylane bridge is an agreement surface for first-order gradients",)
        alternatives += ("native", "grad", "value_and_grad")
    if adapter_key == "qiskit" and transform_key not in {"grad", "value_and_grad"}:
        blocked_reasons += ("qiskit bridge supports shifted-circuit first-order gradients",)
        alternatives += ("native", "grad", "value_and_grad")

    supported = not blocked_reasons
    recommended_method = _recommended_method(transform_key, adapter_key, backend_plan, supported)
    return GradientSupportPlan(
        gate=gate_key,
        observable=observable_key,
        backend=backend_key,
        transform=transform_key,
        adapter=adapter_key,
        supported=supported,
        recommended_method=recommended_method,
        evaluation_mode=_evaluation_mode(transform_key, adapter_key, backend_plan, supported),
        capabilities=capabilities,
        backend_plan=backend_plan,
        blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        alternatives=tuple(dict.fromkeys(alternatives)),
        requires_finite_shot_variance=backend_plan.finite_shot and supported,
        requires_hardware_policy=backend_plan.requires_hardware_approval,
        claim_boundary=_claim_boundary(supported, transform_key, adapter_key, backend_plan),
    )


def assert_gradient_support(plan: GradientSupportPlan) -> GradientSupportPlan:
    """Return a supported plan or raise with its fail-closed reasons."""
    if plan.fail_closed:
        raise ValueError("; ".join(plan.blocked_reasons))
    return plan


def run_gradient_support_matrix_audit() -> GradientSupportMatrixAuditResult:
    """Run representative support-matrix invariants."""
    plans = (
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            backend="statevector",
            transform="grad",
            adapter="native",
            n_params=2,
        ),
        plan_gradient_support(
            gate="rz",
            observable="kuramoto_xy_energy",
            backend="qasm_simulator",
            transform="grad",
            adapter="native",
            n_params=2,
            shots=400,
        ),
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            backend="statevector",
            transform="value_and_grad",
            adapter="jax",
            n_params=2,
        ),
        plan_gradient_support(
            gate="rx",
            observable="sparse_pauli_sum",
            backend="statevector",
            transform="grad",
            adapter="qiskit",
            n_params=1,
        ),
        plan_gradient_support(
            gate="arbitrary_unitary",
            observable="pauli_expectation",
            backend="statevector",
            transform="grad",
            adapter="native",
            n_params=2,
        ),
        plan_gradient_support(
            gate="ry",
            observable="arbitrary_povm",
            backend="statevector",
            transform="grad",
            adapter="native",
            n_params=2,
        ),
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            backend="hardware",
            transform="grad",
            adapter="native",
            n_params=2,
            shots=1024,
        ),
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            backend="statevector",
            transform="vmap",
            adapter="jax",
            n_params=2,
        ),
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            backend="qasm_simulator",
            transform="hessian",
            adapter="native",
            n_params=2,
            shots=400,
        ),
    )
    result = GradientSupportMatrixAuditResult(
        plans=plans,
        passed=False,
        claim_boundary=(
            "gradient support matrix audit only; supported entries identify bounded "
            "local or host-bridge gradient surfaces, blocked entries are fail-closed "
            "planning evidence, and no live hardware-gradient or universal transform claim is implied"
        ),
    )
    return GradientSupportMatrixAuditResult(
        plans=plans,
        passed=not result.failing_plans,
        claim_boundary=result.claim_boundary,
    )


def _validate_category(category: SupportCategory) -> None:
    if category not in {"gate", "observable", "backend", "transform", "adapter"}:
        raise ValueError("unknown support category")


def _normalise_name(category: SupportCategory, name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(".", "_")
    if not key:
        raise ValueError(f"{category} name must be non-empty")
    return _ALIAS_MAP.get(category, {}).get(key, key)


def _unsupported(category: SupportCategory, name: str) -> GradientSupportCapability:
    template = _UNSUPPORTED_CAPABILITIES[category]
    return GradientSupportCapability(
        category=category,
        name=name,
        supported=False,
        gradient_methods=template.gradient_methods,
        conditions=template.conditions,
        blocked_reasons=template.blocked_reasons,
        alternatives=template.alternatives,
        claim_boundary=template.claim_boundary,
    )


def _backend_capability_from_plan(
    backend: str,
    *,
    n_params: int,
    shift_terms: int = 1,
    shots: int | None = None,
) -> GradientSupportCapability:
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        finite_shot=shots is not None,
    )
    if plan.supported:
        return GradientSupportCapability(
            category="backend",
            name=plan.backend,
            supported=True,
            gradient_methods=(plan.method,),
            conditions=_backend_conditions(plan),
            blocked_reasons=(),
            alternatives=plan.alternatives,
            claim_boundary="backend support follows the fail-closed quantum-gradient backend planner",
        )
    return GradientSupportCapability(
        category="backend",
        name=plan.backend,
        supported=False,
        gradient_methods=(),
        conditions=(),
        blocked_reasons=plan.reasons,
        alternatives=plan.alternatives,
        claim_boundary="backend is unsupported for this gradient request",
    )


def _backend_conditions(plan: QuantumGradientPlan) -> tuple[str, ...]:
    if plan.finite_shot:
        return ("finite-shot route requires per-sample variance and shot metadata",)
    return ("deterministic local expectation route",)


def _blocked_reasons(
    capabilities: tuple[GradientSupportCapability, ...],
    backend_plan: QuantumGradientPlan,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for capability in capabilities:
        if not capability.supported:
            reasons.extend(capability.blocked_reasons)
    if backend_plan.fail_closed:
        reasons.extend(backend_plan.reasons)
    return tuple(reasons)


def _compatibility_warnings(
    transform: str,
    adapter: str,
    backend_plan: QuantumGradientPlan,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if backend_plan.finite_shot:
        warnings.append(
            "finite-shot gradients require variance propagation and confidence metadata"
        )
    if adapter in {"jax", "pytorch", "tensorflow"}:
        warnings.append(
            f"{adapter} route crosses an explicit host callback or tensor bridge boundary"
        )
    if adapter == "pennylane":
        warnings.append("pennylane route is agreement evidence, not full QNode ownership transfer")
    if adapter == "qiskit":
        warnings.append(
            "qiskit route is shifted-circuit/local execution unless provider policy is supplied"
        )
    if transform == "hessian":
        warnings.append("hessian route is a local curvature diagnostic")
    return tuple(warnings)


def _alternatives(capabilities: tuple[GradientSupportCapability, ...]) -> tuple[str, ...]:
    alternatives: list[str] = []
    for capability in capabilities:
        alternatives.extend(capability.alternatives)
    return tuple(alternatives)


def _recommended_method(
    transform: str,
    adapter: str,
    backend_plan: QuantumGradientPlan,
    supported: bool,
) -> str:
    if not supported:
        return "unsupported"
    if adapter == "jax":
        return "jax_host_callback_parameter_shift"
    if adapter in {"pytorch", "tensorflow"}:
        return f"{adapter}_tensor_bridge_parameter_shift"
    if adapter == "pennylane":
        return "pennylane_gradient_agreement"
    if adapter == "qiskit":
        return "qiskit_shifted_circuit_parameter_shift"
    if transform == "hessian":
        return "parameter_shift_hessian"
    if transform == "gradient_tape":
        return "recorded_parameter_shift_tape"
    return backend_plan.method


def _evaluation_mode(
    transform: str,
    adapter: str,
    backend_plan: QuantumGradientPlan,
    supported: bool,
) -> str:
    if not supported:
        return "fail_closed"
    if backend_plan.finite_shot:
        return "finite_shot_callback"
    if adapter != "native":
        return "host_bridge"
    if transform == "hessian":
        return "local_curvature_diagnostic"
    return "deterministic_local"


def _claim_boundary(
    supported: bool,
    transform: str,
    adapter: str,
    backend_plan: QuantumGradientPlan,
) -> str:
    if not supported:
        return "unsupported combination; no derivative execution or production claim is permitted"
    if backend_plan.finite_shot:
        return "finite-shot gradient support with explicit variance metadata; no hardware claim is implied"
    if adapter != "native":
        return f"{adapter} bounded bridge support; native framework autodiff through arbitrary providers is not claimed"
    if transform == "hessian":
        return "deterministic local Hessian diagnostic; no hardware Hessian or universal curvature claim is implied"
    return "deterministic local quantum-gradient support for registered gates and observables"


def _backend_plan_to_dict(plan: QuantumGradientPlan) -> dict[str, object]:
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
