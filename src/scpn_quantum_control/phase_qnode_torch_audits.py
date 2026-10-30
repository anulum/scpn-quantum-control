# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch Phase-QNode capability audits
"""Execute PyTorch Phase-QNode capabilities and compare them with parameter shift.

Each audit runs one real PyTorch capability on an admitted registered circuit:
autograd, ``torch.func`` transforms, ``torch.compile`` and the compiler
boundary classification. The caller supplies the PyTorch module and admitted
parameters; this module never loads a framework. Every native operation is
reached through a resolver called at the point where the audit needs it, so a
caller can keep its own named operations live.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .differentiable_axis_compilation import register_torch_axis_compilation
from .differentiable_tensor_construction import (
    TensorConstructionPrimitives,
    construct_torch_tensor,
    prepare_torch_trainable_tensor,
)
from .differentiable_tensor_materialization import (
    TensorMaterializationPrimitives,
    materialize_torch_host_values,
)
from .differentiable_torch_compile_specialisation import torch_static_integer_specialisation
from .phase.qnode_circuit import (
    DenseHermitianObservable,
    PhaseQNodeCircuit,
    PhaseQNodeGradientResult,
    PhaseQNodeSupportError,
    PhaseQNodeSupportReport,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
)
from .phase.torch_bridge_contracts import (
    PhaseTorchCompileBoundaryAuditResult,
    PhaseTorchCompileBoundaryRoute,
    PhaseTorchPhaseQNodeCompileResult,
    PhaseTorchPhaseQNodeStatevectorResult,
    PhaseTorchPhaseQNodeTransformResult,
)
from .phase_qnode_audit_admission import (
    admitted_parameter_matrix,
    admitted_parameter_vector,
    admitted_torch_tolerance,
)
from .phase_qnode_native_statevector import torch_statevector_value_and_state

FloatArray: TypeAlias = NDArray[np.float64]
Resolver: TypeAlias = Callable[[], Callable[..., Any]]
"""Zero-argument callable returning the operation to call at that moment."""

DYNAMIC_COMPILE_QUALIFIED_TORCH: frozenset[tuple[int, int]] = frozenset({(2, 13), (2, 15)})
"""PyTorch minor releases on which the dynamic compile audit was qualified."""


@dataclass(frozen=True)
class TorchAuditOperations:
    """Resolvers for the native operations used by the PyTorch audits.

    Each field returns the operation when called; the audits call it exactly
    where the operation is needed.

    Parameters
    ----------
    support_report, parameter_shift
        Circuit support report and parameter-shift reference.
    support_error
        Exception class raised with an unsupported circuit's report.
    value_and_state
        ``(torch, circuit, parameters) -> (value, state)``.
    scalar_value
        ``(torch, circuit, parameters, dense_matrix=None) -> value``.
    compile_dense_matrix
        ``(torch, circuit) -> tensor | None`` converting a dense observable.
    static_integers
        ``(torch, dynamic) -> context`` for dynamic compilation.
    tensor, trainable_tensor
        ``(torch, values) -> tensor``, without and with gradient tracking.
    scalar_to_float, values_to_numpy, batch_to_numpy
        Host conversions of native scalars, vectors and batches.
    parameter_vector, parameter_matrix
        Admission of host vectors and matrices with a name and width.

    """

    support_report: Resolver
    parameter_shift: Resolver
    support_error: Resolver
    value_and_state: Resolver
    scalar_value: Resolver
    compile_dense_matrix: Resolver
    static_integers: Resolver
    tensor: Resolver
    trainable_tensor: Resolver
    scalar_to_float: Resolver
    values_to_numpy: Resolver
    batch_to_numpy: Resolver
    parameter_vector: Resolver
    parameter_matrix: Resolver


def torch_host_values(values: object) -> object:
    """Return host values of a PyTorch tensor, or ``values`` unchanged.

    Parameters
    ----------
    values
        Native tensor or host values.

    Returns
    -------
    object
        Detached host values ready for NumPy conversion.

    """
    return materialize_torch_host_values(
        values,
        primitives=TensorMaterializationPrimitives(
            attribute=lambda: getattr, predicate=lambda: callable
        ),
    )


def torch_scalar_to_float(
    values: object, *, host_values: Callable[[object], object] = torch_host_values
) -> float:
    """Return a finite host float from a scalar-like PyTorch value.

    Parameters
    ----------
    values
        Native scalar tensor or scalar-like host value.
    host_values
        Host conversion applied before the scalar checks.

    Returns
    -------
    float
        The finite scalar.

    Raises
    ------
    ValueError
        If the value is not scalar-like or not finite.

    """
    scalar = np.asarray(host_values(values), dtype=float)
    if scalar.shape not in ((), (1,)):
        raise ValueError(f"PyTorch scalar value must be scalar-like, got {scalar.shape}")
    value = float(scalar.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("PyTorch scalar value must be finite")
    return value


def torch_func_transforms(torch_module: Any) -> tuple[Any, Any, Any]:
    """Return ``torch.func`` ``grad``, ``vmap`` and ``jacrev``.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.

    Returns
    -------
    tuple[Any, Any, Any]
        The three transform callables.

    Raises
    ------
    RuntimeError
        If the module does not expose all three transforms.

    """
    torch_func = getattr(torch_module, "func", None)
    grad = getattr(torch_func, "grad", None)
    vmap = getattr(torch_func, "vmap", None)
    jacrev = getattr(torch_func, "jacrev", None)
    if torch_func is None or not callable(grad) or not callable(vmap) or not callable(jacrev):
        raise RuntimeError("PyTorch module does not expose torch.func.grad/vmap/jacrev")
    return grad, vmap, jacrev


def torch_compile_function(
    torch_module: Any, *, register: Resolver = lambda: register_torch_axis_compilation
) -> Any:
    """Return ``torch.compile`` after registering the axis-order primitive.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    register
        Resolver of the axis-order registration, called after ``torch.compile``
        is found.

    Returns
    -------
    Any
        The ``torch.compile`` callable.

    Raises
    ------
    RuntimeError
        If the module does not expose ``torch.compile``.

    """
    compile_fn = getattr(torch_module, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError("PyTorch module does not expose torch.compile")
    register()(torch_module)
    return compile_fn


def torch_compile_dense_matrix(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    *,
    complex_tensor: Callable[[Any, object], Any] | None = None,
) -> Any:
    """Convert a dense observable before compilation, or return ``None``.

    Dynamo turns a NumPy constant inside a compiled ``torch.func.grad`` into a
    ``from_numpy`` guard whose dispatch keys differ under the functorch
    transform, so PyTorch 2.13 rejects the frame it has just compiled. A tensor
    converted outside the compiled region carries the same values without that
    guard.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Admitted registered circuit.
    complex_tensor
        ``(torch, values) -> tensor`` conversion; complex128 ``as_tensor`` by default.

    Returns
    -------
    Any
        Complex128 tensor of a dense observable, otherwise ``None``.

    """
    observable = circuit.observable
    if isinstance(observable, DenseHermitianObservable):
        if complex_tensor is None:
            return torch_module.as_tensor(observable.matrix, dtype=torch_module.complex128)
        return complex_tensor(torch_module, observable.matrix)
    return None


def compile_boundary_exception_reason(exc: Exception) -> str:
    """Return a one-line, bounded description of a compiler failure.

    Parameters
    ----------
    exc
        Exception raised by a compile route.

    Returns
    -------
    str
        Exception type and the first line of its message, at most 240 characters.

    """
    detail = str(exc).splitlines()[0] if str(exc).splitlines() else repr(exc)
    return f"{type(exc).__name__}: {detail[:240]}"


def _ensure_supported(
    operations: TorchAuditOperations, circuit: PhaseQNodeCircuit, values: FloatArray
) -> None:
    report: PhaseQNodeSupportReport = operations.support_report()(circuit, values)
    if not report.supported:
        raise operations.support_error()(report)


def torch_statevector_gradient_audit(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    operations: TorchAuditOperations,
) -> PhaseTorchPhaseQNodeStatevectorResult:
    """Differentiate a circuit with PyTorch autograd and compare with parameter shift.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance for value and gradient.
    operations
        Native operation resolvers.

    Returns
    -------
    PhaseTorchPhaseQNodeStatevectorResult
        Autograd value, gradient and state with their reference errors.

    Raises
    ------
    RuntimeError
        If autograd produces no gradient.

    """
    _ensure_supported(operations, circuit, parameter_values)
    parameter_shift: PhaseQNodeGradientResult = operations.parameter_shift()(
        circuit, parameter_values
    )
    trainable_params = operations.trainable_tensor()(torch_module, parameter_values)
    value, state_obj = operations.value_and_state()(torch_module, circuit, trainable_params)
    value.backward()
    gradient_obj = trainable_params.grad
    if gradient_obj is None:
        raise RuntimeError("PyTorch Phase-QNode autograd did not produce a gradient")
    value_scalar = operations.scalar_to_float()(value)
    gradient = operations.parameter_vector()(
        "PyTorch Phase-QNode gradient",
        operations.values_to_numpy()(gradient_obj),
        width=parameter_values.size,
    )
    state = np.asarray(state_obj.detach().cpu().numpy(), dtype=np.complex128)
    max_abs_error = float(np.max(np.abs(gradient - parameter_shift.gradient), initial=0.0))
    l2_error = float(np.linalg.norm(gradient - parameter_shift.gradient))
    passed = bool(
        abs(value_scalar - parameter_shift.value) <= tolerance and max_abs_error <= tolerance
    )
    return PhaseTorchPhaseQNodeStatevectorResult(
        value=value_scalar,
        gradient=gradient,
        state=state,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        torch_value=value,
        torch_gradient=gradient_obj,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=passed,
        native_framework_autodiff=True,
        host_boundary=False,
    )


def torch_func_transform_audit(
    torch_module: Any,
    transforms: tuple[Any, Any, Any],
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    parameter_batch: FloatArray,
    tolerance: float,
    operations: TorchAuditOperations,
) -> PhaseTorchPhaseQNodeTransformResult:
    """Run ``torch.func`` grad, jacrev and vmap and compare with parameter shift.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    transforms
        ``torch.func`` ``grad``, ``vmap`` and ``jacrev``.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    parameter_batch
        Admitted float64 batch with one parameter row per entry.
    tolerance
        Admitted non-negative tolerance.
    operations
        Native operation resolvers.

    Returns
    -------
    PhaseTorchPhaseQNodeTransformResult
        Transform outputs with their joint reference error.

    """
    torch_func_grad, torch_func_vmap, torch_func_jacrev = transforms
    _ensure_supported(operations, circuit, parameter_values)

    value_function = partial(operations.scalar_value(), torch_module, circuit)
    grad_fn = torch_func_grad(value_function)
    vmap_grad_fn = torch_func_vmap(grad_fn)
    jacrev_fn = torch_func_jacrev(value_function)
    torch_params = operations.tensor()(torch_module, parameter_values)
    torch_params_batch = operations.tensor()(torch_module, parameter_batch)
    torch_value, _torch_state = operations.value_and_state()(torch_module, circuit, torch_params)
    torch_gradient = grad_fn(torch_params)
    torch_jacrev_gradient = jacrev_fn(torch_params)
    torch_vmap_gradients = vmap_grad_fn(torch_params_batch)
    value = operations.scalar_to_float()(torch_value)
    gradient = operations.parameter_vector()(
        "PyTorch Phase-QNode torch.func gradient",
        operations.values_to_numpy()(torch_gradient),
        width=parameter_values.size,
    )
    jacrev_gradient = operations.parameter_vector()(
        "PyTorch Phase-QNode torch.func jacrev gradient",
        operations.values_to_numpy()(torch_jacrev_gradient),
        width=parameter_values.size,
    )
    vmap_gradients = operations.parameter_matrix()(
        "PyTorch Phase-QNode torch.func vmap gradients",
        operations.batch_to_numpy()(torch_vmap_gradients),
        width=parameter_values.size,
    )
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)
    parameter_shift_batch_gradients = np.vstack(
        [operations.parameter_shift()(circuit, row).gradient for row in parameter_batch]
    ).astype(np.float64, copy=False)
    deltas = (
        gradient - parameter_shift.gradient,
        jacrev_gradient - parameter_shift.gradient,
        (vmap_gradients - parameter_shift_batch_gradients).reshape(-1),
        np.asarray([value - parameter_shift.value], dtype=np.float64),
    )
    flat_delta = np.concatenate([delta.reshape(-1) for delta in deltas])
    max_abs_error = float(np.max(np.abs(flat_delta))) if flat_delta.size else 0.0
    l2_error = float(np.linalg.norm(flat_delta))
    return PhaseTorchPhaseQNodeTransformResult(
        value=value,
        gradient=gradient,
        jacrev_gradient=jacrev_gradient,
        vmap_gradients=vmap_gradients,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        parameter_shift_batch_gradients=parameter_shift_batch_gradients,
        torch_value=torch_value,
        torch_gradient=torch_gradient,
        torch_jacrev_gradient=torch_jacrev_gradient,
        torch_vmap_gradients=torch_vmap_gradients,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=bool(max_abs_error <= tolerance),
        func_grad_supported=True,
        func_vmap_supported=True,
        func_jacrev_supported=True,
    )


def torch_compile_audit(
    torch_module: Any,
    compile_fn: Any,
    torch_func_grad: Any,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    operations: TorchAuditOperations,
    *,
    fullgraph: bool,
    dynamic: bool,
) -> PhaseTorchPhaseQNodeCompileResult:
    """Compile the value and gradient functions and compare with parameter shift.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    compile_fn
        ``torch.compile``.
    torch_func_grad
        ``torch.func.grad``.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    operations
        Native operation resolvers.
    fullgraph, dynamic
        ``torch.compile`` flags. ``dynamic=True`` is refused before compilation
        by ``operations.static_integers`` on an unqualified runtime.

    Returns
    -------
    PhaseTorchPhaseQNodeCompileResult
        Compiled value and gradient with their reference error.

    """
    _ensure_supported(operations, circuit, parameter_values)
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)

    value_function = partial(
        operations.scalar_value(),
        torch_module,
        circuit,
        dense_matrix=operations.compile_dense_matrix()(torch_module, circuit),
    )
    static_integers: AbstractContextManager[object] = operations.static_integers()(
        torch_module, bool(dynamic)
    )
    grad_fn = torch_func_grad(value_function)
    compiled_value_fn = compile_fn(
        value_function, fullgraph=bool(fullgraph), dynamic=bool(dynamic)
    )
    compiled_grad_fn = compile_fn(grad_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    torch_params = operations.tensor()(torch_module, parameter_values)
    with static_integers:
        torch_value = compiled_value_fn(torch_params)
        torch_gradient = compiled_grad_fn(torch_params)
    value = operations.scalar_to_float()(torch_value)
    gradient = operations.parameter_vector()(
        "PyTorch Phase-QNode torch.compile gradient",
        operations.values_to_numpy()(torch_gradient),
        width=parameter_values.size,
    )
    delta = gradient - parameter_shift.gradient
    max_abs_error = float(
        max(
            abs(value - parameter_shift.value),
            float(np.max(np.abs(delta))) if delta.size else 0.0,
        )
    )
    l2_error = float(np.linalg.norm(delta))
    return PhaseTorchPhaseQNodeCompileResult(
        value=value,
        gradient=gradient,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        torch_value=torch_value,
        torch_gradient=torch_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=bool(max_abs_error <= tolerance),
        torch_compile_supported=True,
        compiled_value_supported=True,
        compiled_gradient_supported=True,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )


def torch_compile_boundary_route(
    *,
    name: str,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    fullgraph: bool,
    dynamic: bool,
    requires: tuple[str, ...],
    passed_reason: str,
    blocked_reason_after_pass: str | None,
    compile_audit: Resolver,
    exception_reason: Resolver,
) -> tuple[PhaseTorchCompileBoundaryRoute, PhaseTorchPhaseQNodeCompileResult | None]:
    """Execute one compile route and classify it as passed or blocked.

    Parameters
    ----------
    name
        Route name.
    circuit
        Admitted registered circuit.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    fullgraph, dynamic
        ``torch.compile`` flags of the route.
    requires
        Artefacts the route needs before promotion.
    passed_reason
        Reason recorded when the route passes and is promotable.
    blocked_reason_after_pass
        Reason recorded when the route executes correctly but stays blocked.
    compile_audit
        Resolver of ``(circuit, values, *, tolerance, fullgraph, dynamic) -> result``.
    exception_reason
        Resolver of the exception formatter, called only after a failure.

    Returns
    -------
    tuple[PhaseTorchCompileBoundaryRoute, PhaseTorchPhaseQNodeCompileResult | None]
        The classified route and the compile result when execution completed.

    """
    try:
        result: PhaseTorchPhaseQNodeCompileResult = compile_audit()(
            circuit,
            parameter_values,
            tolerance=tolerance,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except Exception as exc:
        reason = f"{name} execution is blocked: {exception_reason()(exc)}"
        return (
            PhaseTorchCompileBoundaryRoute(
                name=name,
                status="blocked",
                reason=reason,
                execution_passed=False,
                fullgraph=fullgraph,
                dynamic=dynamic,
                requires=requires,
                exception_type=type(exc).__name__,
            ),
            None,
        )
    status = "passed" if result.passed and blocked_reason_after_pass is None else "blocked"
    if result.passed and blocked_reason_after_pass is not None:
        reason = blocked_reason_after_pass
    elif result.passed:
        reason = passed_reason
    else:
        reason = (
            f"{name} execution disagrees with the SCPN parameter-shift reference: "
            f"max_abs_error={result.max_abs_error:g}"
        )
    return (
        PhaseTorchCompileBoundaryRoute(
            name=name,
            status=status,
            reason=reason,
            execution_passed=result.passed,
            fullgraph=fullgraph,
            dynamic=dynamic,
            requires=requires,
            value=result.value,
            max_abs_reference_error=result.max_abs_error,
        ),
        result,
    )


def torch_aot_autograd_boundary_route(
    torch_module: Any, *, fullgraph_route: PhaseTorchCompileBoundaryRoute
) -> PhaseTorchCompileBoundaryRoute:
    """Classify the AOTAutograd/export route, which stays blocked without artefacts.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module, inspected for ``torch.export`` and AOTAutograd.
    fullgraph_route
        Classified fullgraph compile route.

    Returns
    -------
    PhaseTorchCompileBoundaryRoute
        Blocked route recording which export entry points exist.

    """
    export_module = getattr(torch_module, "export", None)
    export_fn = getattr(export_module, "export", None)
    functorch_module = getattr(torch_module, "_functorch", None)
    aot_autograd = getattr(functorch_module, "aot_autograd", None)
    reason = (
        "AOTAutograd/export promotion is blocked until a persistent exported "
        "program and AOT compile artifact exist; "
        f"torch.export.export available={callable(export_fn)}, "
        f"torch._functorch.aot_autograd available={callable(aot_autograd)}, "
        f"fullgraph route status={fullgraph_route.status}"
    )
    return PhaseTorchCompileBoundaryRoute(
        name="aot_autograd_export_boundary",
        status="blocked",
        reason=reason,
        execution_passed=False,
        fullgraph=True,
        dynamic=False,
        requires=(
            "torch_export_exported_program_artifact",
            "aot_autograd_partition_artifact",
            "aot_compile_artifact",
            "graph_break_free_fullgraph_artifact",
        ),
    )


def torch_compile_boundary_audit(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    operations: TorchAuditOperations,
    *,
    execute_route: Resolver,
    aot_route: Resolver,
) -> PhaseTorchCompileBoundaryAuditResult:
    """Execute the non-fullgraph, dynamic and fullgraph routes and classify them.

    Non-fullgraph compilation must agree with parameter shift. Dynamic-shape,
    fullgraph and AOTAutograd/export routes are executed or inspected and stay
    blocked for promotion until their artefacts exist.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    operations
        Native operation resolvers.
    execute_route
        Resolver of the route executor with the keywords of
        :func:`torch_compile_boundary_route` other than its resolvers.
    aot_route
        Resolver of the AOTAutograd/export classifier.

    Returns
    -------
    PhaseTorchCompileBoundaryAuditResult
        Classified routes, the non-fullgraph result and the overall verdict.

    """
    _ensure_supported(operations, circuit, parameter_values)
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)
    dynamic_compile_blocker = (
        "dynamic compile execution succeeds on the fixed-shape audit input, but "
        "dynamic-shape promotion remains blocked until variable-shape compile artifacts "
        "and guard reports exist"
    )
    fullgraph_compile_blocker = (
        "fullgraph compile returns correct fixed-shape values locally, but promotion "
        "remains blocked until graph-break-free compiled-frame evidence and "
        "AOT-compatible artifacts are recorded"
    )
    non_fullgraph_route, non_fullgraph_result = execute_route()(
        name="non_fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance,
        fullgraph=False,
        dynamic=False,
        requires=(),
        passed_reason=(
            "non-fullgraph torch.compile value and gradient execution matches the "
            "SCPN parameter-shift reference on the registered CPU statevector route"
        ),
        blocked_reason_after_pass=None,
    )
    dynamic_route, _dynamic_result = execute_route()(
        name="dynamic_non_fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance,
        fullgraph=False,
        dynamic=True,
        requires=(
            "variable_shape_compile_artifact",
            "dynamic_shape_guard_report",
        ),
        passed_reason="",
        blocked_reason_after_pass=dynamic_compile_blocker,
    )
    fullgraph_route, _fullgraph_result = execute_route()(
        name="fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance,
        fullgraph=True,
        dynamic=False,
        requires=(
            "graph_break_free_fullgraph_artifact",
            "compiled_frame_evidence",
            "static_shape_symbolic_integer_guards",
        ),
        passed_reason="",
        blocked_reason_after_pass=fullgraph_compile_blocker,
    )
    aot = aot_route()(torch_module, fullgraph_route=fullgraph_route)
    routes = (non_fullgraph_route, dynamic_route, fullgraph_route, aot)
    if non_fullgraph_result is None:
        non_fullgraph_value = parameter_shift.value
        non_fullgraph_gradient = parameter_shift.gradient.copy()
        max_abs_reference_error = tolerance + 1.0
    else:
        non_fullgraph_value = non_fullgraph_result.value
        non_fullgraph_gradient = non_fullgraph_result.gradient.copy()
        max_abs_reference_error = non_fullgraph_result.max_abs_error
    passed = bool(
        non_fullgraph_route.status == "passed"
        and dynamic_route.status == "blocked"
        and fullgraph_route.status == "blocked"
        and aot.status == "blocked"
    )
    return PhaseTorchCompileBoundaryAuditResult(
        routes=routes,
        non_fullgraph_value=non_fullgraph_value,
        non_fullgraph_gradient=non_fullgraph_gradient,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        max_abs_reference_error=max_abs_reference_error,
        tolerance=tolerance,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        passed=passed,
        persistent_export_claim=False,
        provider_claim=False,
        performance_claim=False,
    )


def _construction_primitives() -> TensorConstructionPrimitives:
    return TensorConstructionPrimitives(
        attribute=lambda: getattr, predicate=lambda: callable, invalid=lambda: RuntimeError
    )


def _native_tensor(torch_module: Any, values: object) -> Any:
    return construct_torch_tensor(torch_module, values, primitives=_construction_primitives())


def _native_trainable_tensor(torch_module: Any, values: FloatArray) -> Any:
    return prepare_torch_trainable_tensor(
        _native_tensor(torch_module, values), primitives=_construction_primitives()
    )


def _native_scalar_value(
    torch_module: Any, circuit: PhaseQNodeCircuit, parameters: Any, dense_matrix: Any = None
) -> Any:
    value, _state = torch_statevector_value_and_state(
        torch_module, circuit, parameters, dense_matrix=dense_matrix
    )
    return value


NATIVE_TORCH_AUDIT_OPERATIONS = TorchAuditOperations(
    support_report=lambda: phase_qnode_support_report,
    parameter_shift=lambda: parameter_shift_phase_qnode_gradient,
    support_error=lambda: PhaseQNodeSupportError,
    value_and_state=lambda: torch_statevector_value_and_state,
    scalar_value=lambda: _native_scalar_value,
    compile_dense_matrix=lambda: torch_compile_dense_matrix,
    static_integers=lambda: partial(
        torch_static_integer_specialisation,
        purpose="dynamic Phase-QNode compile audit",
        qualified_releases=DYNAMIC_COMPILE_QUALIFIED_TORCH,
    ),
    tensor=lambda: _native_tensor,
    trainable_tensor=lambda: _native_trainable_tensor,
    scalar_to_float=lambda: torch_scalar_to_float,
    values_to_numpy=lambda: (
        lambda values: admitted_parameter_vector("values", torch_host_values(values))
    ),
    batch_to_numpy=lambda: (
        lambda values: admitted_parameter_matrix("values", torch_host_values(values))
    ),
    parameter_vector=lambda: admitted_parameter_vector,
    parameter_matrix=lambda: admitted_parameter_matrix,
)
"""Operations of the lower PyTorch statevector engine and admission leaves."""


def _native_compile_route(torch_module: Any) -> Callable[..., Any]:
    def compile_audit(
        circuit: PhaseQNodeCircuit,
        parameter_values: FloatArray,
        *,
        tolerance: float,
        fullgraph: bool,
        dynamic: bool,
    ) -> PhaseTorchPhaseQNodeCompileResult:
        return native_torch_compile_audit(
            torch_module,
            circuit,
            parameter_values,
            tolerance=tolerance,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    return compile_audit


def native_torch_statevector_gradient_audit(
    torch_module: Any, circuit: PhaseQNodeCircuit, params: object, *, tolerance: float = 1e-6
) -> PhaseTorchPhaseQNodeStatevectorResult:
    """Run the autograd audit on the lower statevector engine.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    tolerance
        Non-negative tolerance, admitted as on the PyTorch route.

    Returns
    -------
    PhaseTorchPhaseQNodeStatevectorResult
        The audit result.

    """
    tolerance_value = admitted_torch_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    return torch_statevector_gradient_audit(
        torch_module, circuit, parameter_values, tolerance_value, NATIVE_TORCH_AUDIT_OPERATIONS
    )


def native_torch_func_transform_audit(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
    *,
    params_batch: object,
    tolerance: float = 1e-6,
) -> PhaseTorchPhaseQNodeTransformResult:
    """Run the ``torch.func`` audit on the lower statevector engine.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module; ``torch.func`` is required.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    params_batch
        Parameter rows for ``vmap``, admitted as a finite matrix.
    tolerance
        Non-negative tolerance, admitted as on the PyTorch route.

    Returns
    -------
    PhaseTorchPhaseQNodeTransformResult
        The audit result.

    """
    transforms = torch_func_transforms(torch_module)
    tolerance_value = admitted_torch_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    parameter_batch = admitted_parameter_matrix(
        "params_batch",
        admitted_parameter_matrix("params_batch", torch_host_values(params_batch)),
        width=parameter_values.size,
    )
    return torch_func_transform_audit(
        torch_module,
        transforms,
        circuit,
        parameter_values,
        parameter_batch,
        tolerance_value,
        NATIVE_TORCH_AUDIT_OPERATIONS,
    )


def native_torch_compile_audit(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
    *,
    tolerance: float = 1e-6,
    fullgraph: bool = False,
    dynamic: bool = False,
) -> PhaseTorchPhaseQNodeCompileResult:
    """Run the ``torch.compile`` audit on the lower statevector engine.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module; ``torch.compile`` and ``torch.func`` are required.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    tolerance
        Non-negative tolerance, admitted as on the PyTorch route.
    fullgraph, dynamic
        ``torch.compile`` flags; ``dynamic=True`` is admitted only on
        :data:`DYNAMIC_COMPILE_QUALIFIED_TORCH` releases.

    Returns
    -------
    PhaseTorchPhaseQNodeCompileResult
        The audit result.

    """
    compile_fn = torch_compile_function(torch_module)
    torch_func_grad, _vmap, _jacrev = torch_func_transforms(torch_module)
    tolerance_value = admitted_torch_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    return torch_compile_audit(
        torch_module,
        compile_fn,
        torch_func_grad,
        circuit,
        parameter_values,
        tolerance_value,
        NATIVE_TORCH_AUDIT_OPERATIONS,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )


def native_torch_compile_boundary_audit(
    torch_module: Any, circuit: PhaseQNodeCircuit, params: object, *, tolerance: float = 1e-6
) -> PhaseTorchCompileBoundaryAuditResult:
    """Run the compiler boundary audit on the lower statevector engine.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    tolerance
        Non-negative tolerance, admitted as on the PyTorch route.

    Returns
    -------
    PhaseTorchCompileBoundaryAuditResult
        The audit result; every compile route really executes.

    """
    tolerance_value = admitted_torch_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    return torch_compile_boundary_audit(
        torch_module,
        circuit,
        parameter_values,
        tolerance_value,
        NATIVE_TORCH_AUDIT_OPERATIONS,
        execute_route=lambda: partial(
            torch_compile_boundary_route,
            compile_audit=lambda: _native_compile_route(torch_module),
            exception_reason=lambda: compile_boundary_exception_reason,
        ),
        aot_route=lambda: torch_aot_autograd_boundary_route,
    )


__all__ = [
    "DYNAMIC_COMPILE_QUALIFIED_TORCH",
    "NATIVE_TORCH_AUDIT_OPERATIONS",
    "TorchAuditOperations",
    "compile_boundary_exception_reason",
    "native_torch_compile_audit",
    "native_torch_compile_boundary_audit",
    "native_torch_func_transform_audit",
    "native_torch_statevector_gradient_audit",
    "torch_aot_autograd_boundary_route",
    "torch_compile_audit",
    "torch_compile_boundary_audit",
    "torch_compile_boundary_route",
    "torch_compile_dense_matrix",
    "torch_compile_function",
    "torch_func_transform_audit",
    "torch_func_transforms",
    "torch_host_values",
    "torch_scalar_to_float",
    "torch_statevector_gradient_audit",
]
