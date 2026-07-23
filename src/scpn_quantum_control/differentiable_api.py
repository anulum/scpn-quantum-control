# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API module
# scpn-quantum-control -- unified differentiable API facade
"""Unified differentiable-programming facade over supported local routes.

All returned evidence retains the claim boundary of the delegated route.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .analysis.finite_size_scaling import FSS_CLAIM_BOUNDARY, finite_size_scaling
from .benchmarks.differentiable_isolated_benchmark_plan import (
    run_differentiable_isolated_benchmark_plan,
)
from .compiler.mlir import (
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
)
from .differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    PrimitiveIdentity,
    ScalarObjective,
    VectorObjective,
    compile_whole_program_frontend,
    value_and_grad,
    value_and_hessian,
    value_and_jacobian,
)
from .differentiable_api_contracts import (
    DifferentiabilityDiagnosticReport,
    DifferentiableDashboardCapabilityRow,
    DifferentiableDashboardCapabilityState,
    DifferentiableDashboardStatus,
    UnifiedDifferentiableAPIResult,
    UnifiedDifferentiableOperation,
)
from .differentiable_architecture_map import run_differentiable_architecture_map
from .differentiable_baseline_scorecard import run_differentiable_baseline_scorecard
from .differentiable_benchmark_report import build_differentiable_benchmark_report
from .differentiable_competitive_baselines import run_competitive_baseline_refresh
from .differentiable_dashboard import differentiable_dashboard_status
from .differentiable_dependency_environment_map import (
    run_differentiable_dependency_environment_map,
)
from .differentiable_rust_python_inventory import run_differentiable_rust_python_inventory
from .differentiable_transform_algebra import run_transform_algebra_audit
from .phase.gradient_backend import (
    plan_quantum_gradient_backend,
    quantum_gradient_backend_capability,
)
from .phase.gradient_support_matrix import plan_gradient_support
from .phase.qnn_framework_bridge_matrix import run_bounded_qnn_framework_bridge_matrix


def differentiable_value(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective without exposing its computed gradient.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered scalar differentiation method.
    step:
        Optional finite-difference step forwarded to the selected method.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``value`` envelope with evaluation provenance.

    """
    result = value_and_grad(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="value",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_gradient(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and its gradient.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered scalar differentiation method.
    step:
        Optional finite-difference step forwarded to the selected method.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``gradient`` envelope with a copied ``float64`` gradient.

    """
    result = value_and_grad(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="gradient",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=result.gradient.astype(np.float64, copy=True),
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a vector objective and its Jacobian.

    Parameters
    ----------
    objective:
        Vector objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered Jacobian method.
    step:
        Positive finite-difference step used by diagnostic routes.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``jacobian`` envelope with objective and evaluation data.

    """
    result = value_and_jacobian(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="jacobian",
        supported=True,
        method=result.method,
        value=None,
        gradient=None,
        jacobian=result.jacobian.astype(np.float64, copy=True),
        hessian=None,
        payload={
            "objective_value": result.value.tolist(),
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and its Hessian.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered Hessian method.
    step:
        Positive finite-difference step used by diagnostic routes.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``hessian`` envelope with evaluation provenance.

    """
    result = value_and_hessian(objective, values, method=method, step=step)
    return UnifiedDifferentiableAPIResult(
        operation="hessian",
        supported=True,
        method=result.method,
        value=float(result.value),
        gradient=None,
        jacobian=None,
        hessian=result.hessian.astype(np.float64, copy=True),
        payload={
            "evaluations": result.evaluations,
            "parameter_names": list(result.parameter_names),
            "trainable": list(result.trainable),
        },
    )


def differentiable_support_report(
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
) -> UnifiedDifferentiableAPIResult:
    """Return a fail-closed support report for a quantum-gradient route.

    Parameters
    ----------
    gate, observable:
        Registered circuit gate and observable identifiers.
    backend, transform, adapter:
        Requested execution, differentiation, and integration route.
    n_params, shift_terms:
        Parameter count and parameter-shift terms per parameter.
    shots:
        Optional finite-shot budget.
    allow_hardware:
        Whether planning may consider an explicitly approved hardware route.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Support-plan envelope, including blocked reasons when unsupported.

    Raises
    ------
    ValueError
        If a planning dimension or shot budget is invalid.

    """
    plan = plan_gradient_support(
        gate=gate,
        observable=observable,
        backend=backend,
        transform=transform,
        adapter=adapter,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        allow_hardware=allow_hardware,
    )
    return UnifiedDifferentiableAPIResult(
        operation="support_report",
        supported=plan.supported,
        method=plan.recommended_method,
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=plan.to_dict(),
        claim_boundary=plan.claim_boundary,
    )


def explain_differentiability(
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
) -> DifferentiabilityDiagnosticReport:
    """Explain whether a differentiable route can run and why it may fail closed.

    Parameters
    ----------
    gate, observable:
        Registered circuit gate and observable identifiers.
    backend, transform, adapter:
        Requested execution, differentiation, and integration route.
    n_params, shift_terms:
        Parameter count and parameter-shift terms per parameter.
    shots:
        Optional finite-shot budget.
    allow_hardware:
        Whether planning may consider an explicitly approved hardware route.

    Returns
    -------
    DifferentiabilityDiagnosticReport
        Dependency, device, backend, alternative, and blocked-reason matrices.

    Raises
    ------
    ValueError
        If a planning dimension or shot budget is invalid.

    """
    plan = plan_gradient_support(
        gate=gate,
        observable=observable,
        backend=backend,
        transform=transform,
        adapter=adapter,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        allow_hardware=allow_hardware,
    )
    support_payload = plan.to_dict()
    backend_names = _diagnostic_backend_names(backend)
    backend_matrix = tuple(
        _backend_plan_row(
            selected_backend,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
        for selected_backend in backend_names
    )
    alternatives = _unique_strings(
        plan.alternatives
        + plan.backend_plan.alternatives
        + tuple(
            str(alternative)
            for row in backend_matrix
            for alternative in cast(Sequence[object], row["alternatives"])
        )
    )
    return DifferentiabilityDiagnosticReport(
        request={
            "gate": gate,
            "observable": observable,
            "backend": backend,
            "transform": transform,
            "adapter": adapter,
            "n_params": n_params,
            "shift_terms": shift_terms,
            "shots": shots,
            "allow_hardware": allow_hardware,
        },
        supported=plan.supported,
        blocked_reasons=plan.blocked_reasons,
        suggested_alternatives=alternatives,
        dependency_matrix=_dependency_matrix_rows(),
        device_matrix=tuple(_device_capability_row(name) for name in backend_names),
        backend_matrix=backend_matrix,
        support_payload=support_payload,
        claim_boundary=(
            "differentiability diagnostic report only; support, dependency, "
            "device, and backend rows are planning evidence and do not execute "
            "objectives, provider callbacks, hardware jobs, or performance benchmarks"
        ),
    )


def differentiable_compile_report(
    *,
    primitive_identities: Sequence[str | PrimitiveIdentity] | None = None,
    registry: CustomDerivativeRegistry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    transform: str = "jvp_vjp_adjoint",
) -> UnifiedDifferentiableAPIResult:
    """Return compiler-AD planning evidence for registered primitives.

    Parameters
    ----------
    primitive_identities:
        Optional non-empty subset of registered primitive identities.
    registry:
        Custom-derivative registry used to build the transform plan.
    transform:
        Compiler-AD transform requested for every selected primitive.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Compile-plan envelope containing deterministic MLIR interchange text.

    Raises
    ------
    ValueError
        If the subset is empty, unknown, or otherwise invalid.

    """
    plan = build_compiler_ad_transform_plan(registry, transform=transform)
    selected = _selected_primitive_keys(primitive_identities)
    statuses = (
        plan.statuses
        if selected is None
        else tuple(status for status in plan.statuses if status.identity.key in selected)
    )
    if selected is not None and len(statuses) != len(selected):
        present = {status.identity.key for status in statuses}
        missing = ", ".join(sorted(selected - present))
        raise ValueError(f"unknown primitive identities for compile report: {missing}")
    filtered_plan = type(plan)(
        statuses=statuses,
        dialect=plan.dialect,
        transform=plan.transform,
        executable_backend=plan.executable_backend,
        claim_boundary=plan.claim_boundary,
    )
    mlir_module = compile_compiler_ad_transform_plan_to_mlir(filtered_plan)
    return UnifiedDifferentiableAPIResult(
        operation="compile_report",
        supported=bool(statuses),
        method=f"compiler_ad_{filtered_plan.transform}",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "primitive_count": len(statuses),
            "primitive_identities": [status.identity.key for status in statuses],
            "executable_backend": filtered_plan.executable_backend,
            "mlir": mlir_module.text,
        },
        claim_boundary=filtered_plan.claim_boundary,
    )


def differentiable_benchmark_report() -> UnifiedDifferentiableAPIResult:
    """Return local non-performance benchmark/conformance evidence."""
    report = build_differentiable_benchmark_report()
    return UnifiedDifferentiableAPIResult(
        operation="benchmark_report",
        supported=report.supported,
        method=report.method,
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=report.payload,
        claim_boundary=report.claim_boundary,
    )


def differentiable_baseline_scorecard_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded baseline-category scorecard evidence."""
    scorecard = run_differentiable_baseline_scorecard()
    return UnifiedDifferentiableAPIResult(
        operation="baseline_scorecard",
        supported=scorecard.promotion_ready,
        method="differentiable_baseline_scorecard",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=scorecard.to_dict(),
        claim_boundary=scorecard.claim_boundary,
    )


def differentiable_competitive_baseline_refresh_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded competitive-baseline freshness evidence."""
    refresh = run_competitive_baseline_refresh()
    return UnifiedDifferentiableAPIResult(
        operation="competitive_baseline_refresh",
        supported=False,
        method="differentiable_competitive_baselines",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=refresh.to_dict(),
        claim_boundary=refresh.claim_boundary,
    )


def differentiable_rust_python_inventory_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded Rust/Python rustification inventory evidence."""
    inventory = run_differentiable_rust_python_inventory()
    return UnifiedDifferentiableAPIResult(
        operation="rust_python_inventory",
        supported=inventory.rustification_ready,
        method="differentiable_rust_python_inventory",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=inventory.to_dict(),
        claim_boundary=inventory.claim_boundary,
    )


def differentiable_architecture_map_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded architecture and Rustification routing evidence."""
    architecture_map = run_differentiable_architecture_map()
    return UnifiedDifferentiableAPIResult(
        operation="architecture_rustification_map",
        supported=architecture_map.rustification_ready,
        method="differentiable_architecture_map",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=architecture_map.to_dict(),
        claim_boundary=architecture_map.claim_boundary,
    )


def differentiable_dependency_environment_map_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded dependency and environment evidence."""
    environment_map = run_differentiable_dependency_environment_map()
    return UnifiedDifferentiableAPIResult(
        operation="dependency_environment_map",
        supported=environment_map.environment_ready,
        method="differentiable_dependency_environment_map",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=environment_map.to_dict(),
        claim_boundary=environment_map.claim_boundary,
    )


def differentiable_isolated_benchmark_plan_report() -> UnifiedDifferentiableAPIResult:
    """Return claim-bounded isolated benchmark batch planning evidence."""
    plan = run_differentiable_isolated_benchmark_plan()
    return UnifiedDifferentiableAPIResult(
        operation="isolated_benchmark_plan",
        supported=plan.promotion_ready,
        method="differentiable_isolated_benchmark_plan",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=plan.to_dict(),
        claim_boundary=plan.claim_boundary,
    )


def differentiable_transform_algebra_report() -> UnifiedDifferentiableAPIResult:
    """Return executable local transform-algebra metamorphic evidence."""
    audit = run_transform_algebra_audit()
    return UnifiedDifferentiableAPIResult(
        operation="transform_algebra_report",
        supported=audit.passed,
        method="differentiable_transform_algebra",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=audit.to_dict(),
        claim_boundary=audit.claim_boundary,
    )


def differentiable_qfi_fss_report(
    *,
    system_sizes: Sequence[int] | None = None,
    k_range: ArrayLike | None = None,
    max_dense_gib: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Return bounded QFI/FSS finite-size evidence for differentiable dashboards.

    Parameters
    ----------
    system_sizes:
        Optional finite-size qubit counts forwarded to the local dense FSS scan.
    k_range:
        Optional coupling grid. Values are converted to ``float64`` and
        validated by :func:`finite_size_scaling`.
    max_dense_gib:
        Optional dense workspace budget forwarded to each exact gap scan.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``qfi_fss_report`` result whose payload contains serialized
        finite-size-scaling evidence and whose claim boundary prevents hardware,
        performance, or thermodynamic-limit promotion.

    """
    scan_range = None if k_range is None else np.asarray(k_range, dtype=np.float64)
    result = finite_size_scaling(
        system_sizes=None if system_sizes is None else list(system_sizes),
        k_range=scan_range,
        max_dense_gib=max_dense_gib,
    )
    return UnifiedDifferentiableAPIResult(
        operation="qfi_fss_report",
        supported=True,
        method="qfi_finite_size_scaling",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=result.to_dict(),
        claim_boundary=FSS_CLAIM_BOUNDARY,
    )


def differentiable_frontend_report(
    objective: Callable[..., object],
) -> UnifiedDifferentiableAPIResult:
    """Return static whole-program bytecode/source frontend evidence.

    The objective is inspected but not executed. The returned payload is
    suitable for dashboards and audits that need deterministic source/bytecode
    metadata without promoting executable compiler lowering, provider, hardware,
    or performance claims.

    Parameters
    ----------
    objective:
        Python callable to inspect without executing.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Static frontend envelope with source and bytecode provenance.

    """
    report = compile_whole_program_frontend(objective)
    return UnifiedDifferentiableAPIResult(
        operation="frontend_report",
        supported=report.frontend_ready,
        method="static_bytecode_source_frontend_preflight",
        value=None,
        gradient=None,
        jacobian=None,
        hessian=None,
        payload=report.to_dict(),
        claim_boundary=report.claim_boundary,
    )


def differentiable_api(
    operation: UnifiedDifferentiableOperation,
    *,
    objective: Callable[[Any], Any] | None = None,
    values: ArrayLike | None = None,
    method: str | None = None,
    step: float | None = None,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    n_params: int = 1,
    shift_terms: int = 1,
    shots: int | None = None,
    allow_hardware: bool = False,
    primitive_identities: Sequence[str | PrimitiveIdentity] | None = None,
    registry: CustomDerivativeRegistry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    system_sizes: Sequence[int] | None = None,
    k_range: ArrayLike | None = None,
    max_dense_gib: float | None = None,
) -> UnifiedDifferentiableAPIResult:
    """Dispatch one supported unified differentiable operation.

    Parameters
    ----------
    operation:
        Public operation identifier selecting the delegated route.
    objective, values:
        Numerical inputs required by value, derivative, and frontend routes.
    method, step:
        Optional differentiation method and diagnostic step override.
    gate, observable, backend, transform, adapter:
        Quantum-gradient support and diagnostic route selectors.
    n_params, shift_terms, shots, allow_hardware:
        Quantum-gradient planning dimensions and hardware-policy controls.
    primitive_identities, registry:
        Compiler-AD primitive subset and derivative registry.
    system_sizes, k_range, max_dense_gib:
        Bounded QFI finite-size-scaling inputs and dense workspace budget.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Stable evidence envelope returned by the delegated public route.

    Raises
    ------
    ValueError
        If the operation is unknown or a required route input is absent.

    """
    if operation == "value":
        return differentiable_value(
            _require_objective(objective),
            _require_values(values),
            method="parameter_shift" if method is None else method,
            step=step,
        )
    if operation == "gradient":
        return differentiable_gradient(
            _require_objective(objective),
            _require_values(values),
            method="parameter_shift" if method is None else method,
            step=step,
        )
    if operation == "jacobian":
        return differentiable_jacobian(
            _require_objective(objective),
            _require_values(values),
            method="finite_difference" if method is None else method,
            step=1.0e-6 if step is None else step,
        )
    if operation == "hessian":
        return differentiable_hessian(
            cast(ScalarObjective, _require_objective(objective)),
            _require_values(values),
            method="finite_difference" if method is None else method,
            step=1.0e-4 if step is None else step,
        )
    if operation == "support_report":
        return differentiable_support_report(
            gate=gate,
            observable=observable,
            backend=backend,
            transform=transform,
            adapter=adapter,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
    if operation == "diagnostic_report":
        report = explain_differentiability(
            gate=gate,
            observable=observable,
            backend=backend,
            transform=transform,
            adapter=adapter,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            allow_hardware=allow_hardware,
        )
        return UnifiedDifferentiableAPIResult(
            operation="diagnostic_report",
            supported=report.supported,
            method="differentiability_diagnostic",
            value=None,
            gradient=None,
            jacobian=None,
            hessian=None,
            payload=report.to_dict(),
            claim_boundary=report.claim_boundary,
        )
    if operation == "compile_report":
        return differentiable_compile_report(
            primitive_identities=primitive_identities,
            registry=registry,
            transform="jvp_vjp_adjoint" if method is None else method,
        )
    if operation == "benchmark_report":
        return differentiable_benchmark_report()
    if operation == "baseline_scorecard":
        return differentiable_baseline_scorecard_report()
    if operation == "competitive_baseline_refresh":
        return differentiable_competitive_baseline_refresh_report()
    if operation == "rust_python_inventory":
        return differentiable_rust_python_inventory_report()
    if operation == "architecture_rustification_map":
        return differentiable_architecture_map_report()
    if operation == "dependency_environment_map":
        return differentiable_dependency_environment_map_report()
    if operation == "isolated_benchmark_plan":
        return differentiable_isolated_benchmark_plan_report()
    if operation == "transform_algebra_report":
        return differentiable_transform_algebra_report()
    if operation == "qfi_fss_report":
        return differentiable_qfi_fss_report(
            system_sizes=system_sizes,
            k_range=k_range,
            max_dense_gib=max_dense_gib,
        )
    if operation == "frontend_report":
        return differentiable_frontend_report(_require_objective(objective))
    if operation == "dashboard_status":
        status = differentiable_dashboard_status()
        return UnifiedDifferentiableAPIResult(
            operation="dashboard_status",
            supported=status.status_api_ready,
            method="claim_bounded_dashboard_status",
            value=None,
            gradient=None,
            jacobian=None,
            hessian=None,
            payload=status.to_dict(),
            claim_boundary=status.claim_boundary,
        )
    raise ValueError(f"unsupported unified differentiable operation: {operation!r}")


def _require_objective(objective: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    if objective is None:
        raise ValueError("objective is required for this differentiable operation")
    return objective


def _require_values(values: ArrayLike | None) -> ArrayLike:
    if values is None:
        raise ValueError("values are required for this differentiable operation")
    return values


def _selected_primitive_keys(
    primitive_identities: Sequence[str | PrimitiveIdentity] | None,
) -> set[str] | None:
    if primitive_identities is None:
        return None
    keys = {PrimitiveIdentity.parse(identity).key for identity in primitive_identities}
    if not keys:
        raise ValueError("primitive_identities must be non-empty when provided")
    return keys


def _dependency_matrix_rows() -> tuple[Mapping[str, object], ...]:
    bridge_matrix = run_bounded_qnn_framework_bridge_matrix()
    return tuple(
        {
            "framework": capability.framework,
            "optional_dependency": capability.optional_dependency,
            "runtime_dependency_required": capability.runtime_dependency_required,
            "implemented": capability.implemented,
            "supported": capability.supported,
            "public_api": capability.public_api,
            "gradient_route": capability.gradient_route,
            "native_framework_autodiff": capability.native_framework_autodiff,
            "tensor_output": capability.tensor_output,
            "host_boundary": capability.host_boundary,
            "fail_closed_reason": capability.fail_closed_reason,
        }
        for capability in bridge_matrix.capabilities
    )


def _diagnostic_backend_names(requested_backend: str) -> tuple[str, ...]:
    return _unique_strings(
        (
            requested_backend,
            "statevector_simulator",
            "finite_shot_simulator",
            "hardware_qpu",
        )
    )


def _device_capability_row(backend: str) -> Mapping[str, object]:
    capability = quantum_gradient_backend_capability(backend)
    return {
        "backend": capability.backend,
        "family": capability.family,
        "hardware": capability.hardware,
        "supports_parameter_shift": capability.supports_parameter_shift,
        "supports_finite_shot": capability.supports_finite_shot,
        "supports_adjoint": capability.supports_adjoint,
        "supports_spsa": capability.supports_spsa,
        "default_shots": capability.default_shots,
        "notes": list(capability.notes),
    }


def _backend_plan_row(
    backend: str,
    *,
    n_params: int,
    shift_terms: int,
    shots: int | None,
    allow_hardware: bool,
) -> Mapping[str, object]:
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=n_params,
        shift_terms=shift_terms,
        shots=shots,
        finite_shot=shots is not None,
        allow_hardware=allow_hardware,
    )
    return {
        "backend": plan.backend,
        "family": plan.family,
        "method": plan.method,
        "supported": plan.supported,
        "fail_closed": plan.fail_closed,
        "evaluations": plan.evaluations,
        "shots": plan.shots,
        "finite_shot": plan.finite_shot,
        "confidence_level": plan.confidence_level,
        "requires_hardware_approval": plan.requires_hardware_approval,
        "reasons": list(plan.reasons),
        "alternatives": list(plan.alternatives),
    }


def _unique_strings(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "DifferentiableDashboardCapabilityRow",
    "DifferentiableDashboardCapabilityState",
    "DifferentiableDashboardStatus",
    "DifferentiabilityDiagnosticReport",
    "UnifiedDifferentiableAPIResult",
    "UnifiedDifferentiableOperation",
    "differentiable_api",
    "differentiable_architecture_map_report",
    "differentiable_benchmark_report",
    "differentiable_compile_report",
    "differentiable_competitive_baseline_refresh_report",
    "differentiable_dashboard_status",
    "differentiable_dependency_environment_map_report",
    "differentiable_frontend_report",
    "differentiable_gradient",
    "differentiable_hessian",
    "differentiable_isolated_benchmark_plan_report",
    "differentiable_jacobian",
    "differentiable_qfi_fss_report",
    "differentiable_rust_python_inventory_report",
    "differentiable_baseline_scorecard_report",
    "differentiable_support_report",
    "differentiable_transform_algebra_report",
    "differentiable_value",
    "explain_differentiability",
]
