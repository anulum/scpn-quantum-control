# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PyTorch Bridge
"""Optional PyTorch execution facade for phase gradients.

Immutable result records live in :mod:`.torch_bridge_contracts`, and bounded
gradient implementations plus their direct runtime primitives live in
:mod:`.torch_gradients`. Registered-QNode statevector, transform, and compiler
diagnostic routes live in :mod:`.torch_qnode_transforms`. This module retains
compatibility, module, training, cloud, and maturity routes while those
responsibilities undergo bounded decomposition.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
)
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
)
from .qnode_circuit import (
    PhaseQNodeCircuit,
)
from .torch_bridge_contracts import (
    PhaseTorchAutogradQNNGradientResult,
    PhaseTorchCloudValidationRunSpec,
    PhaseTorchCompileBoundaryAuditResult,
    PhaseTorchCompileBoundaryRoute,
    PhaseTorchCompileCompatibilityResult,
    PhaseTorchEcosystemMaturityAuditResult,
    PhaseTorchEcosystemMaturityRoute,
    PhaseTorchFuncCompatibilityResult,
    PhaseTorchLiveOverlayEvidence,
    PhaseTorchMaturityAuditResult,
    PhaseTorchModuleWrapperAuditResult,
    PhaseTorchParameterShiftResult,
    PhaseTorchPhaseQNodeCompileResult,
    PhaseTorchPhaseQNodeLoweringMatrixResult,
    PhaseTorchPhaseQNodeLoweringRoute,
    PhaseTorchPhaseQNodeStatevectorResult,
    PhaseTorchPhaseQNodeTransformResult,
    PhaseTorchQNNGradientResult,
    PhaseTorchTrainingLoopAuditResult,
)
from .torch_bridge_contracts import (
    _json_ready as _json_ready,
)
from .torch_bridge_contracts import (
    _result_to_dict as _result_to_dict,
)
from .torch_gradients import (
    _as_feature_matrix as _as_feature_matrix,
)
from .torch_gradients import (
    _as_label_vector as _as_label_vector,
)
from .torch_gradients import (
    _as_non_negative_tolerance as _as_non_negative_tolerance,
)
from .torch_gradients import (
    _as_parameter_vector as _as_parameter_vector,
)
from .torch_gradients import (
    _bounded_qnn_loss_gradient_reference as _bounded_qnn_loss_gradient_reference,
)
from .torch_gradients import (
    _load_torch as _load_torch,
)
from .torch_gradients import (
    _torch_autograd_function as _torch_autograd_function,
)
from .torch_gradients import (
    _torch_autograd_grad as _torch_autograd_grad,
)
from .torch_gradients import (
    _torch_tensor as _torch_tensor,
)
from .torch_gradients import (
    _torch_trainable_tensor as _torch_trainable_tensor,
)
from .torch_gradients import (
    _torch_values_to_numpy as _torch_values_to_numpy,
)
from .torch_gradients import (
    torch_autograd_qnn_value_and_grad as _torch_autograd_qnn_value_and_grad,
)
from .torch_gradients import (
    torch_bounded_qnn_value_and_grad as _torch_bounded_qnn_value_and_grad,
)
from .torch_gradients import (
    torch_parameter_shift_value_and_grad as _torch_parameter_shift_value_and_grad,
)
from .torch_qnode_transforms import (
    _as_parameter_matrix as _as_parameter_matrix,
)
from .torch_qnode_transforms import (
    _compile_boundary_exception_reason as _compile_boundary_exception_reason,
)
from .torch_qnode_transforms import (
    _torch_aot_autograd_boundary_route as _torch_aot_autograd_boundary_route,
)
from .torch_qnode_transforms import (
    _torch_apply_gate_matrix as _torch_apply_gate_matrix,
)
from .torch_qnode_transforms import (
    _torch_apply_term_operator as _torch_apply_term_operator,
)
from .torch_qnode_transforms import (
    _torch_batch_to_numpy as _torch_batch_to_numpy,
)
from .torch_qnode_transforms import (
    _torch_ccnot_matrix as _torch_ccnot_matrix,
)
from .torch_qnode_transforms import (
    _torch_compile as _torch_compile,
)
from .torch_qnode_transforms import (
    _torch_compile_boundary_execution_route as _torch_compile_boundary_execution_route,
)
from .torch_qnode_transforms import (
    _torch_complex_tensor as _torch_complex_tensor,
)
from .torch_qnode_transforms import (
    _torch_controlled as _torch_controlled,
)
from .torch_qnode_transforms import (
    _torch_cswap_matrix as _torch_cswap_matrix,
)
from .torch_qnode_transforms import (
    _torch_expectation_value as _torch_expectation_value,
)
from .torch_qnode_transforms import (
    _torch_func_transforms as _torch_func_transforms,
)
from .torch_qnode_transforms import (
    _torch_gate_matrix as _torch_gate_matrix,
)
from .torch_qnode_transforms import (
    _torch_matrix_to_numpy as _torch_matrix_to_numpy,
)
from .torch_qnode_transforms import (
    _torch_operation_theta as _torch_operation_theta,
)
from .torch_qnode_transforms import (
    _torch_pauli_matrix as _torch_pauli_matrix,
)
from .torch_qnode_transforms import (
    _torch_phase_qnode_value_and_state as _torch_phase_qnode_value_and_state,
)
from .torch_qnode_transforms import (
    _torch_real_tensor as _torch_real_tensor,
)
from .torch_qnode_transforms import (
    _torch_scalar_to_float as _torch_scalar_to_float,
)
from .torch_qnode_transforms import (
    _torch_symmetrized_product_expectation as _torch_symmetrized_product_expectation,
)
from .torch_qnode_transforms import (
    _torch_term_expectation as _torch_term_expectation,
)
from .torch_qnode_transforms import (
    _torch_term_product_expectation as _torch_term_product_expectation,
)
from .torch_qnode_transforms import (
    torch_phase_qnode_compile_audit as _torch_phase_qnode_compile_audit,
)
from .torch_qnode_transforms import (
    torch_phase_qnode_compile_boundary_audit as _torch_phase_qnode_compile_boundary_audit,
)
from .torch_qnode_transforms import (
    torch_phase_qnode_transform_audit as _torch_phase_qnode_transform_audit,
)
from .torch_qnode_transforms import (
    torch_phase_qnode_value_and_grad as _torch_phase_qnode_value_and_grad,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


def is_phase_torch_available() -> bool:
    """Return whether the optional phase PyTorch bridge can import PyTorch."""
    try:
        _load_torch()
    except ImportError:
        return False
    return True


def run_torch_phase_qnode_lowering_matrix() -> PhaseTorchPhaseQNodeLoweringMatrixResult:
    """Return the PyTorch parity matrix for registered Phase-QNode lowering.

    The current PyTorch surface is production-grade for the bounded QNN routes
    listed here and for deterministic registered statevector Phase-QNode
    execution. Finite-shot, provider, hardware, dynamic-circuit, and isolated
    benchmark promotion routes stay blocked until their artefacts exist.
    """
    routes = (
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_analytic_tensor",
            status="passed",
            reason="bounded phase-QNN analytic tensor value-and-gradient is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_custom_autograd",
            status="passed",
            reason="bounded phase-QNN custom torch.autograd.Function route is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_torch_func",
            status="passed",
            reason="bounded torch.func grad/vmap/jacrev compatibility is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_torch_compile",
            status="passed",
            reason="bounded torch.compile gradient compatibility is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_module_layer_wrapper",
            status="passed",
            reason="bounded nn.Module and layer wrappers are implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_statevector_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits lower into "
                "native PyTorch autograd execution without host callbacks"
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_torch_func_transform_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute "
                "through torch.func grad, jacrev, and vmap transform routes on CPU"
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_torch_compile_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute through "
                "non-fullgraph torch.compile value and gradient routes on CPU"
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_torch_compile_boundary_diagnostic",
            status="passed",
            reason=(
                "registered Phase-QNode torch.compile boundary audit classifies "
                "non-fullgraph, dynamic-shape, fullgraph, and AOTAutograd/export "
                "routes without promoting blocked compiler artifacts"
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_torch_compile_fullgraph_lowering",
            status="blocked",
            reason=(
                "fullgraph registered Phase-QNode torch.compile lowering is still blocked "
                "by PyTorch Dynamo data-dependent symbolic-shape extraction"
            ),
            requires=(
                "registered_phase_qnode_fullgraph_compile_artifact",
                "static_shape_symbolic_integer_guards",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_cuda_device_lowering",
            status="blocked",
            reason="CUDA/device promotion requires compatible visible hardware and smoke artefacts",
            requires=(
                "compatible_cuda_device",
                "successful_cuda_tensor_smoke",
                "device_specific_phase_qnode_gradient_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_finite_shot_lowering",
            status="blocked",
            reason="finite-shot Torch lowering needs uncertainty and sampler provenance",
            requires=(
                "shot_policy",
                "rng_seed_provenance",
                "uncertainty_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_provider_lowering",
            status="blocked",
            reason="provider callbacks are not Torch compiler/autograd-safe yet",
            requires=(
                "provider_allowlist",
                "callback_transform_safety_audit",
                "provider_execution_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_hardware_lowering",
            status="blocked",
            reason="live hardware lowering requires explicit ticketed execution evidence",
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_dynamic_circuit_lowering",
            status="blocked",
            reason="mid-circuit measurement and feedback are outside the Torch lowering boundary",
            requires=(
                "dynamic_circuit_semantics",
                "classical_feedback_contract",
                "gradient_policy",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion needs isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseTorchPhaseQNodeLoweringMatrixResult(routes=routes)


def _as_positive_learning_rate(value: float) -> float:
    learning_rate = float(value)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite float")
    return learning_rate


def _as_positive_step_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("steps must be a positive integer")
    if value <= 0:
        raise ValueError("steps must be a positive integer")
    return value


def _torch_nn_module_and_parameter(torch_module: Any) -> tuple[Any, Any]:
    torch_nn = getattr(torch_module, "nn", None)
    module_base = getattr(torch_nn, "Module", None)
    parameter_cls = getattr(torch_nn, "Parameter", None)
    if torch_nn is None or module_base is None or not callable(parameter_cls):
        raise RuntimeError("PyTorch module does not expose torch.nn.Module and torch.nn.Parameter")
    return module_base, parameter_cls


def _torch_cuda_metadata(torch_module: Any) -> tuple[bool, int, tuple[str, ...], bool, str]:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    device_count_fn = getattr(cuda, "device_count", None)
    if cuda is None or not callable(is_available) or not callable(device_count_fn):
        return False, 0, (), False, "PyTorch CUDA runtime metadata is unavailable"
    cuda_available = bool(is_available())
    device_count = int(device_count_fn()) if cuda_available else 0
    device_names: list[str] = []
    get_device_name = getattr(cuda, "get_device_name", None)
    for index in range(device_count):
        if callable(get_device_name):
            try:
                device_names.append(str(get_device_name(index)))
            except Exception as exc:  # pragma: no cover - defensive optional runtime probe
                device_names.append(f"unavailable:{type(exc).__name__}")
        else:
            device_names.append(f"cuda:{index}")
    if not cuda_available or device_count == 0:
        return cuda_available, device_count, tuple(device_names), False, "no visible CUDA devices"
    try:
        device = torch_module.device("cuda", 0)
        sample = torch_module.ones((1,), dtype=torch_module.float64, device=device)
        _ = (sample + sample).detach().cpu().numpy()
    except Exception as exc:
        return (
            cuda_available,
            device_count,
            tuple(device_names),
            False,
            f"CUDA smoke execution failed: {type(exc).__name__}: {exc}",
        )
    return cuda_available, device_count, tuple(device_names), True, "CUDA smoke execution passed"


def _torch_parameter_count(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(1 for _parameter in parameters())


def _torch_bounded_qnn_loss_tensor(
    torch_module: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    shifted = feature_tensor + parameter_tensor.unsqueeze(0)
    probabilities = 0.5 * (1.0 - torch_module.cos(shifted))
    predictions = torch_module.mean(probabilities, dim=1)
    residual = predictions - label_tensor
    return torch_module.mean(residual * residual)


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
    return _torch_parameter_shift_value_and_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
        _torch_loader=_load_torch,
    )


def torch_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into PyTorch.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It excludes finite-shot sampling, provider callbacks,
    hardware execution, density/noise channels, dynamic circuits, and
    promotion-grade performance evidence.
    """
    return _torch_phase_qnode_value_and_grad(
        circuit,
        params,
        tolerance=tolerance,
        _torch_loader=_load_torch,
    )


def torch_phase_qnode_transform_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchPhaseQNodeTransformResult:
    """Audit registered Phase-QNode execution through ``torch.func`` transforms.

    The accepted surface is deterministic local statevector execution for the
    registered ``PhaseQNodeCircuit`` gate and observable family. The audit
    verifies ``torch.func.grad``, ``torch.func.jacrev``, and ``torch.func.vmap``
    against SCPN parameter-shift references. It does not promote finite-shot,
    provider, hardware, CUDA, ``torch.compile``, or performance claims.
    """
    return _torch_phase_qnode_transform_audit(
        circuit,
        params,
        params_batch=params_batch,
        tolerance=tolerance,
        _torch_loader=_load_torch,
    )


def torch_phase_qnode_compile_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
    fullgraph: bool = False,
    dynamic: bool = False,
) -> PhaseTorchPhaseQNodeCompileResult:
    """Audit registered Phase-QNode execution through ``torch.compile``.

    The audit compiles the deterministic local statevector value function and
    the corresponding ``torch.func.grad`` gradient function for the registered
    ``PhaseQNodeCircuit`` gate and observable family. It excludes finite-shot,
    provider, hardware, CUDA promotion, and performance claims.
    """
    return _torch_phase_qnode_compile_audit(
        circuit,
        params,
        tolerance=tolerance,
        fullgraph=fullgraph,
        dynamic=dynamic,
        _torch_loader=_load_torch,
    )


def torch_phase_qnode_compile_boundary_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchCompileBoundaryAuditResult:
    """Classify PyTorch compiler boundaries for registered Phase-QNode lowering.

    The audit executes the deterministic local Phase-QNode ``torch.compile``
    route in non-fullgraph, dynamic, and fullgraph modes against SCPN
    parameter-shift references. It deliberately reports dynamic-shape,
    fullgraph, AOTAutograd/export, provider, hardware, CUDA, isolated benchmark,
    and performance surfaces as blocked until their promotion artefacts exist.
    """
    return _torch_phase_qnode_compile_boundary_audit(
        circuit,
        params,
        tolerance=tolerance,
        _torch_loader=_load_torch,
    )


def torch_bounded_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchQNNGradientResult:
    """Return bounded phase-QNN loss and gradient as NumPy plus PyTorch tensors.

    This route is deliberately narrower than arbitrary PyTorch autograd through
    a quantum simulator. It expresses the bounded phase-QNN gradient in the same
    tensor-ready analytic form used by the parameter-shift classifier and
    compares it against the canonical SCPN parameter-shift gradient before
    returning tensors to PyTorch workflows.
    """
    return _torch_bounded_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        _torch_loader=_load_torch,
    )


def torch_autograd_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchAutogradQNNGradientResult:
    """Return bounded phase-QNN loss and gradient through ``torch.autograd.Function``.

    This is a native PyTorch autograd route for the bounded phase-QNN surface
    only. The custom backward is the audited bounded analytic gradient, checked
    against the canonical SCPN parameter-shift gradient before the result is
    returned.
    """
    return _torch_autograd_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        _torch_loader=_load_torch,
    )


def run_torch_func_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchFuncCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.func`` transforms."""
    torch_module = _load_torch()
    torch_func_grad, torch_func_vmap, torch_func_jacrev = _torch_func_transforms(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    parameter_batch = _torch_batch_to_numpy(params_batch)
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        parameter_batch,
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)

    def loss_fn(parameter_tensor: Any) -> Any:
        return _torch_bounded_qnn_loss_tensor(
            torch_module,
            feature_tensor,
            label_tensor,
            parameter_tensor,
        )

    grad_fn = torch_func_grad(loss_fn)
    vmap_grad_fn = torch_func_vmap(grad_fn)
    jacrev_fn = torch_func_jacrev(loss_fn)
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_params_batch = _torch_tensor(torch_module, parameter_batch)
    torch_grad_gradient = grad_fn(torch_params)
    torch_vmap_gradients = vmap_grad_fn(torch_params_batch)
    torch_jacrev_gradient = jacrev_fn(torch_params)
    grad_gradient = _torch_values_to_numpy(torch_grad_gradient)
    vmap_gradients = _torch_batch_to_numpy(torch_vmap_gradients)
    jacrev_gradient = _torch_values_to_numpy(torch_jacrev_gradient)
    parameter_shift_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    parameter_shift_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_gradient,
        width=parameter_values.size,
    )
    parameter_shift_batch_gradients = np.vstack(
        [
            parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, row)
            for row in parameter_batch
        ],
    )
    deltas = (
        grad_gradient - parameter_shift_gradient,
        jacrev_gradient - parameter_shift_gradient,
        (vmap_gradients - parameter_shift_batch_gradients).reshape(-1),
    )
    flat_delta = np.concatenate([delta.reshape(-1) for delta in deltas])
    max_abs_error = float(np.max(np.abs(flat_delta))) if flat_delta.size else 0.0
    l2_error = float(np.linalg.norm(flat_delta))
    return PhaseTorchFuncCompatibilityResult(
        grad_gradient=grad_gradient,
        vmap_gradients=vmap_gradients,
        jacrev_gradient=jacrev_gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        parameter_shift_batch_gradients=parameter_shift_batch_gradients,
        torch_grad_gradient=torch_grad_gradient,
        torch_vmap_gradients=torch_vmap_gradients,
        torch_jacrev_gradient=torch_jacrev_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        func_grad_supported=True,
        func_vmap_supported=True,
        func_jacrev_supported=True,
    )


def run_torch_compile_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
) -> PhaseTorchCompileCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.compile``."""
    torch_module = _load_torch()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)

    def loss_fn(parameter_tensor: Any) -> Any:
        return _torch_bounded_qnn_loss_tensor(
            torch_module,
            feature_tensor,
            label_tensor,
            parameter_tensor,
        )

    grad_fn = torch_func_grad(loss_fn)
    compiled_grad_fn = compile_fn(grad_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_gradient = compiled_grad_fn(torch_params)
    gradient = _torch_values_to_numpy(torch_gradient)
    parameter_shift_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    parameter_shift_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_gradient,
        width=parameter_values.size,
    )
    delta = gradient - parameter_shift_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTorchCompileCompatibilityResult(
        gradient=gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        torch_gradient=torch_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        torch_compile_supported=True,
        compiled_loss_supported=True,
        compiled_gradient_supported=True,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )


def torch_bounded_qnn_module(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
) -> Any:
    """Return a PyTorch ``nn.Module`` wrapper for the bounded phase-QNN loss."""
    torch_module = _load_torch()
    module_base, parameter_cls = _torch_nn_module_and_parameter(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(initial_params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "initial_params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)
    parameter_tensor = _torch_tensor(torch_module, parameter_values)

    class _BoundedPhaseQNNModule(module_base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            register_buffer = getattr(self, "register_buffer", None)
            if callable(register_buffer):
                register_buffer("features", feature_tensor)
                register_buffer("labels", label_tensor)
            else:
                self.features = feature_tensor
                self.labels = label_tensor
            self.params = parameter_cls(parameter_tensor, requires_grad=bool(trainable))
            self.feature_width = int(feature_matrix.shape[1])
            self.host_boundary = False
            self.native_framework_autodiff = True
            self.claim_boundary = "bounded_torch_module_layer_wrapper"

        def forward(self, params: Any | None = None) -> Any:
            parameter_source = self.params if params is None else params
            return _torch_bounded_qnn_loss_tensor(
                torch_module,
                self.features,
                self.labels,
                parameter_source,
            )

        def parameter_shift_gradient(self, params: Any | None = None) -> FloatArray:
            parameter_source = self.params if params is None else params
            raw_params = _torch_values_to_numpy(parameter_source)
            raw_params = _as_parameter_vector(
                "PyTorch bounded phase-QNN module parameters",
                raw_params,
                width=feature_matrix.shape[1],
            )
            reference_gradient = parameter_shift_qnn_classifier_gradient(
                feature_matrix,
                label_vector,
                raw_params,
            )
            return _as_parameter_vector(
                "SCPN bounded phase-QNN parameter-shift gradient",
                reference_gradient,
                width=feature_matrix.shape[1],
            )

    return _BoundedPhaseQNNModule()


def torch_bounded_qnn_layer(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
) -> Any:
    """Return the bounded phase-QNN wrapper using layer-oriented naming."""
    return torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
    )


def run_torch_module_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchModuleWrapperAuditResult:
    """Audit bounded phase-QNN PyTorch module/layer wrapper gradients."""
    torch_module = _load_torch()
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=True,
    )
    torch_params = module.params
    torch_loss = module()

    def loss_fn(parameter_tensor: Any) -> Any:
        return module(parameter_tensor)

    grad_fn = torch_func_grad(loss_fn)
    torch_gradient = grad_fn(torch_params)
    gradient = _torch_values_to_numpy(torch_gradient)
    parameter_shift_gradient = module.parameter_shift_gradient(torch_params)
    delta = gradient - parameter_shift_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTorchModuleWrapperAuditResult(
        loss=_torch_scalar_to_float(torch_loss),
        gradient=gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        torch_module=module,
        torch_loss=torch_loss,
        torch_gradient=torch_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        module_wrapper_supported=True,
        layer_wrapper_supported=True,
        trainable_parameters=_torch_parameter_count(module),
    )


def run_torch_training_loop_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    learning_rate: float = 0.1,
    steps: int = 4,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
) -> PhaseTorchTrainingLoopAuditResult:
    """Audit a bounded PyTorch module training loop against SCPN references.

    The loop uses the bounded phase-QNN ``nn.Module`` wrapper, compiles its loss
    callable with ``torch.compile``, obtains gradients through ``torch.func``,
    and applies deterministic gradient-descent updates. It is local functional
    correctness evidence only; CUDA, provider, finite-shot, hardware, isolated
    benchmark, and performance promotion claims remain outside this route.
    """
    torch_module = _load_torch()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    del _torch_func_vmap, _torch_func_jacrev
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(initial_params)
    parameter_values = _as_parameter_vector(
        "initial_params",
        parameter_values,
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    learning_rate_value = _as_positive_learning_rate(learning_rate)
    step_count = _as_positive_step_count(steps)
    module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        trainable=True,
    )

    def loss_fn(parameter_tensor: Any) -> Any:
        return module(parameter_tensor)

    compiled_loss_fn = compile_fn(loss_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    grad_fn = torch_func_grad(loss_fn)
    compiled_grad_fn = compile_fn(grad_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    current_params = parameter_values.copy()
    loss_values: list[float] = []
    gradient_values: list[FloatArray] = []
    gradient_deltas: list[FloatArray] = []
    for _index in range(step_count):
        torch_params = _torch_tensor(torch_module, current_params)
        torch_loss = compiled_loss_fn(torch_params)
        torch_gradient = compiled_grad_fn(torch_params)
        loss_values.append(_torch_scalar_to_float(torch_loss))
        gradient = _as_parameter_vector(
            "PyTorch training-loop gradient",
            _torch_values_to_numpy(torch_gradient),
            width=parameter_values.size,
        )
        reference_gradient = parameter_shift_qnn_classifier_gradient(
            feature_matrix,
            label_vector,
            current_params,
        )
        reference_gradient = _as_parameter_vector(
            "SCPN bounded phase-QNN parameter-shift gradient",
            reference_gradient,
            width=parameter_values.size,
        )
        gradient_values.append(gradient)
        gradient_deltas.append(gradient - reference_gradient)
        current_params = current_params - learning_rate_value * gradient

    final_torch_params = _torch_tensor(torch_module, current_params)
    final_torch_loss = compiled_loss_fn(final_torch_params)
    final_torch_gradient = compiled_grad_fn(final_torch_params)
    final_gradient = _as_parameter_vector(
        "PyTorch training-loop final gradient",
        _torch_values_to_numpy(final_torch_gradient),
        width=parameter_values.size,
    )
    parameter_shift_final_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        current_params,
    )
    parameter_shift_final_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN final parameter-shift gradient",
        parameter_shift_final_gradient,
        width=parameter_values.size,
    )
    gradient_deltas.append(final_gradient - parameter_shift_final_gradient)
    loss_values.append(_torch_scalar_to_float(final_torch_loss))
    flat_delta = np.concatenate([delta.reshape(-1) for delta in gradient_deltas])
    max_abs_gradient_error = float(np.max(np.abs(flat_delta))) if flat_delta.size else 0.0
    l2_gradient_error = float(np.linalg.norm(flat_delta))
    loss_history = _as_parameter_vector("PyTorch training-loop loss history", loss_values)
    gradient_history = _as_parameter_matrix(
        "PyTorch training-loop gradient history",
        np.vstack(gradient_values),
        width=parameter_values.size,
    )
    passed = bool(
        max_abs_gradient_error <= tolerance_value
        and np.all(np.isfinite(loss_history))
        and float(loss_history[-1]) <= float(loss_history[0]) + tolerance_value
    )
    return PhaseTorchTrainingLoopAuditResult(
        initial_params=parameter_values,
        final_params=current_params,
        loss_history=loss_history,
        gradient_history=gradient_history,
        final_gradient=final_gradient,
        parameter_shift_final_gradient=parameter_shift_final_gradient,
        max_abs_gradient_error=max_abs_gradient_error,
        l2_gradient_error=l2_gradient_error,
        tolerance=tolerance_value,
        passed=passed,
        steps=step_count,
        learning_rate=learning_rate_value,
        torch_module=module,
        torch_final_loss=final_torch_loss,
        torch_final_gradient=final_torch_gradient,
        module_wrapper_supported=True,
        func_grad_supported=True,
        torch_compile_supported=True,
        compiled_loss_supported=True,
        parameter_update_supported=bool(not np.allclose(current_params, parameter_values)),
    )


def run_torch_ecosystem_maturity_audit() -> PhaseTorchEcosystemMaturityAuditResult:
    """Audit broad PyTorch module, transform, compiler, and device maturity.

    The audit is capability evidence, not a provider or performance promotion.
    It records which framework surfaces are present in the installed PyTorch
    runtime and keeps GPU/device and full compiler maturity blocked unless a
    local CUDA smoke execution succeeds and the broader transform/compiler
    routes are available.
    """
    torch_module = _load_torch()
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    routes: list[PhaseTorchEcosystemMaturityRoute] = []
    try:
        _torch_nn_module_and_parameter(torch_module)
    except RuntimeError as exc:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="nn_module_parameter_surface",
                status="blocked",
                reason=str(exc),
                requires=("torch.nn.Module", "torch.nn.Parameter"),
            )
        )
    else:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="nn_module_parameter_surface",
                status="passed",
                reason="torch.nn.Module and torch.nn.Parameter are available",
            )
        )

    try:
        _torch_func_transforms(torch_module)
    except RuntimeError as exc:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_func_grad_vmap_jacrev",
                status="blocked",
                reason=str(exc),
                requires=("torch.func.grad", "torch.func.vmap", "torch.func.jacrev"),
            )
        )
    else:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_func_grad_vmap_jacrev",
                status="passed",
                reason="torch.func.grad, torch.func.vmap, and torch.func.jacrev are available",
            )
        )

    torch_func = getattr(torch_module, "func", None)
    jacfwd = getattr(torch_func, "jacfwd", None)
    hessian = getattr(torch_func, "hessian", None)
    if callable(jacfwd) and callable(hessian):
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_func_jacfwd_hessian",
                status="passed",
                reason="torch.func.jacfwd and torch.func.hessian are available",
            )
        )
    else:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_func_jacfwd_hessian",
                status="blocked",
                reason="torch.func.jacfwd and torch.func.hessian are not both available",
                requires=("torch.func.jacfwd", "torch.func.hessian"),
            )
        )

    try:
        _torch_compile(torch_module)
    except RuntimeError as exc:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_compile_callable",
                status="blocked",
                reason=str(exc),
                requires=("torch.compile",),
            )
        )
    else:
        routes.append(
            PhaseTorchEcosystemMaturityRoute(
                name="torch_compile_callable",
                status="passed",
                reason="torch.compile callable is available",
            )
        )

    cuda_available, cuda_device_count, cuda_device_names, cuda_smoke_passed, cuda_reason = (
        _torch_cuda_metadata(torch_module)
    )
    routes.append(
        PhaseTorchEcosystemMaturityRoute(
            name="cuda_accelerator_device",
            status="passed" if cuda_smoke_passed else "blocked",
            reason=cuda_reason,
            requires=()
            if cuda_smoke_passed
            else ("compatible_cuda_device", "successful_cuda_tensor_smoke"),
        )
    )
    routes.append(
        PhaseTorchEcosystemMaturityRoute(
            name="registered_phase_qnode_torch_compile_lowering",
            status="passed",
            reason=("registered Phase-QNode non-fullgraph torch.compile lowering is implemented"),
        )
    )
    routes.append(
        PhaseTorchEcosystemMaturityRoute(
            name="registered_phase_qnode_torch_compile_fullgraph_lowering",
            status="blocked",
            reason=(
                "registered Phase-QNode fullgraph torch.compile lowering is still blocked "
                "by PyTorch Dynamo data-dependent symbolic-shape extraction"
            ),
            requires=(
                "registered_phase_qnode_fullgraph_compile_artifact",
                "static_shape_symbolic_integer_guards",
            ),
        )
    )
    return PhaseTorchEcosystemMaturityAuditResult(
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        routes=tuple(routes),
    )


def plan_torch_cloud_validation_batch(
    *,
    runner: str = "jarvislabs",
    accelerator_backend: str = "cuda",
) -> PhaseTorchCloudValidationRunSpec:
    """Plan the PyTorch cloud validation batch for blocked local routes.

    Parameters
    ----------
    runner:
        Human-readable runner label used in the downstream validation queue.
    accelerator_backend:
        Accelerator runtime requested for the cloud rerun, usually ``"cuda"``
        for the PyTorch wheels used by the differentiable-programming lane.

    Returns
    -------
    PhaseTorchCloudValidationRunSpec
        JSON-ready scheduling metadata with local skip status, required cloud
        artefacts, environment constraints, and reproduction commands.
    """
    clean_runner = runner.strip()
    if not clean_runner:
        raise ValueError("runner must be a non-empty string")
    clean_backend = accelerator_backend.strip().lower()
    if clean_backend not in {"cuda", "rocm"}:
        raise ValueError("accelerator_backend must be 'cuda' or 'rocm'")

    ecosystem = run_torch_ecosystem_maturity_audit()
    lowering_matrix = run_torch_phase_qnode_lowering_matrix()
    cloud_route_names = {
        "cuda_accelerator_device",
        "registered_phase_qnode_torch_compile_fullgraph_lowering",
        "registered_phase_qnode_cuda_device_lowering",
        "isolated_benchmark_artifact",
    }
    blocked_ecosystem_routes = tuple(
        route.name
        for route in ecosystem.routes
        if route.status != "passed" and route.name in cloud_route_names
    )
    blocked_lowering_routes = tuple(
        route.name
        for route in lowering_matrix.routes
        if route.status != "passed" and route.name in cloud_route_names
    )
    blocked_local_routes = blocked_ecosystem_routes + blocked_lowering_routes
    cuda_route = next(
        route for route in ecosystem.routes if route.name == "cuda_accelerator_device"
    )
    local_execution_status = (
        "local_accelerator_ready"
        if cuda_route.status == "passed"
        else "skipped_incompatible_local_hardware"
    )
    commands = (
        ".venv/bin/python -m pytest "
        "tests/test_phase_framework_bridges.py::"
        "test_torch_phase_qnode_compile_audit_lowers_registered_statevector "
        "tests/test_phase_framework_bridges.py::"
        "test_torch_phase_qnode_value_and_grad_lowers_registered_statevector "
        "tests/test_phase_framework_bridges.py::"
        "test_torch_phase_qnode_transform_audit_checks_grad_jacrev_and_vmap -q",
        ".venv/bin/python -m pytest "
        "tests/test_differentiable_programming_benchmarks.py::"
        "test_quantum_gradient_benchmark_suite_matches_analytic_references -q",
        ".venv/bin/python - <<'PY'\n"
        "from scpn_quantum_control.phase import plan_torch_cloud_validation_batch\n"
        "print(plan_torch_cloud_validation_batch().to_dict())\n"
        "PY",
    )
    return PhaseTorchCloudValidationRunSpec(
        runner=clean_runner,
        local_execution_status=local_execution_status,
        local_skip_reason=cuda_route.reason if cuda_route.status != "passed" else "",
        torch_version=ecosystem.torch_version,
        cuda_available=ecosystem.cuda_available,
        cuda_device_count=ecosystem.cuda_device_count,
        cuda_device_names=ecosystem.cuda_device_names,
        blocked_local_routes=blocked_local_routes,
        required_artifacts=(
            "registered_phase_qnode_fullgraph_compile_artifact",
            "static_shape_symbolic_integer_guard_artifact",
            "cuda_device_phase_qnode_gradient_artifact",
            "successful_cuda_tensor_smoke_artifact",
            "isolated_benchmark_artifact",
            "host_load_and_affinity_metadata",
        ),
        required_environment={
            "accelerator_backend": clean_backend,
            "minimum_cuda_compute_capability": "7.5" if clean_backend == "cuda" else None,
            "visible_device_metadata_required": True,
            "host_load_metadata_required": True,
            "isolated_affinity_required_for_promotion": True,
            "network_required": False,
            "hardware_submission_allowed": False,
        },
        commands=commands,
        ready_for_cloud_dispatch=bool(blocked_local_routes),
    )


def run_torch_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
    live_overlay_artifact_path: str | Path | None = None,
) -> PhaseTorchMaturityAuditResult:
    """Aggregate bounded PyTorch evidence and provider-level parity blockers.

    The audit records the bounded phase-QNN routes that are implemented today:
    tensor-ready analytic gradients, custom autograd, ``torch.func`` transforms,
    ``torch.compile``, and module/layer wrappers. It deliberately keeps broader
    PyTorch-provider maturity blocked until finite-shot/provider/hardware
    Phase-QNode lowering, full compiler/autograd integration, live overlay
    evidence, and isolated benchmark artefacts are present.
    """
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        _torch_values_to_numpy(params),
        width=feature_matrix.shape[1],
    )
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        _torch_matrix_to_numpy("params_batch", params_batch),
        width=feature_matrix.shape[1],
    )

    analytic_tensor = torch_bounded_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    custom_autograd = torch_autograd_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    torch_func = run_torch_func_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    torch_compile = run_torch_compile_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    module_layer_wrapper = run_torch_module_wrapper_audit(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        tolerance=tolerance_value,
    )
    training_loop = run_torch_training_loop_audit(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        tolerance=tolerance_value,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    ecosystem_maturity = run_torch_ecosystem_maturity_audit()
    cloud_validation_batch = plan_torch_cloud_validation_batch()
    live_overlay = (
        _load_torch_live_overlay_evidence(live_overlay_artifact_path)
        if live_overlay_artifact_path is not None
        else None
    )

    evidence: dict[str, object] = {
        "analytic_tensor": analytic_tensor,
        "custom_autograd": custom_autograd,
        "torch_func": torch_func,
        "torch_compile": torch_compile,
        "module_layer_wrapper": module_layer_wrapper,
        "training_loop": training_loop,
        "ecosystem_maturity": ecosystem_maturity,
        "phase_qnode_lowering_matrix": run_torch_phase_qnode_lowering_matrix(),
        "cloud_validation_batch": cloud_validation_batch,
    }
    if live_overlay is not None:
        evidence["live_overlay"] = live_overlay
    bounded_model_ready = all(
        bool(getattr(result, "passed", False))
        for name, result in evidence.items()
        if name
        not in {
            "phase_qnode_lowering_matrix",
            "ecosystem_maturity",
            "cloud_validation_batch",
        }
    )
    lowering_matrix = evidence["phase_qnode_lowering_matrix"]
    if not isinstance(lowering_matrix, PhaseTorchPhaseQNodeLoweringMatrixResult):
        raise RuntimeError("PyTorch maturity audit requires a phase-QNode lowering matrix.")
    required_capabilities = {
        "analytic_tensor": "passed" if analytic_tensor.passed else "failed",
        "custom_autograd": "passed" if custom_autograd.passed else "failed",
        "torch_func": "passed" if torch_func.passed else "failed",
        "torch_compile": "passed" if torch_compile.passed else "failed",
        "module_layer_wrapper": "passed" if module_layer_wrapper.passed else "failed",
        "training_loop": "passed" if training_loop.passed else "failed",
        "torch_ecosystem_maturity": (
            "passed" if ecosystem_maturity.ready_for_provider_exceedance else "blocked"
        ),
        "cloud_validation_batch": (
            "scheduled" if cloud_validation_batch.ready_for_cloud_dispatch else "not_required"
        ),
        "live_overlay_execution": "passed" if live_overlay is not None else "blocked",
        "finite_shot_provider_hardware_torch_phase_qnode_lowering": "blocked",
        "full_compiler_autograd_integration": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    required_capabilities.update(
        {
            f"phase_qnode_lowering:{route.name}": route.status
            for route in lowering_matrix.routes
            if route.status != "passed"
        }
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PhaseTorchMaturityAuditResult(
        bounded_model_ready=bounded_model_ready,
        ready_for_provider_exceedance=bounded_model_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        open_gaps=open_gaps,
    )


def _load_torch_live_overlay_evidence(
    artifact_path: str | Path,
) -> PhaseTorchLiveOverlayEvidence:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PyTorch live overlay artefact must be a JSON object")
    classification = _required_str(payload, "classification")
    if classification != "functional_non_isolated":
        raise ValueError("PyTorch live overlay artefact must be functional_non_isolated")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("PyTorch live overlay artefact must include rows")
    pytorch_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("backend") == "pytorch"
        and row.get("status") == "success"
    ]
    if not pytorch_rows:
        raise ValueError("PyTorch live overlay artefact requires a successful PyTorch row")
    row = pytorch_rows[0]
    dependency_versions = row.get("dependency_versions")
    if not isinstance(dependency_versions, dict):
        raise ValueError("successful PyTorch row must include dependency_versions")
    torch_version = dependency_versions.get("torch")
    if not isinstance(torch_version, str) or not torch_version:
        raise ValueError("successful PyTorch row must include a torch dependency version")
    return PhaseTorchLiveOverlayEvidence(
        artifact_id=_required_str(payload, "artifact_id"),
        artifact_path=str(path),
        classification=classification,
        torch_version=torch_version,
        value_error=_required_float(row, "value_error"),
        gradient_error=_required_float(row, "gradient_error"),
        runtime_seconds=_required_float(row, "runtime_seconds"),
        memory_peak_bytes=_required_int(row, "memory_peak_bytes"),
        batching_support=_required_str(row, "batching_support"),
        transform_support=_required_str(row, "transform_support"),
        claim_boundary=_required_str(row, "claim_boundary"),
        promotion_ready=bool(payload.get("promotion_ready", False)),
    )


def _required_str(payload: dict[Any, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(payload: dict[Any, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_int(payload: dict[Any, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


__all__ = [
    "PhaseTorchAutogradQNNGradientResult",
    "PhaseTorchCloudValidationRunSpec",
    "PhaseTorchCompileBoundaryAuditResult",
    "PhaseTorchCompileBoundaryRoute",
    "PhaseTorchCompileCompatibilityResult",
    "PhaseTorchEcosystemMaturityAuditResult",
    "PhaseTorchEcosystemMaturityRoute",
    "PhaseTorchFuncCompatibilityResult",
    "PhaseTorchLiveOverlayEvidence",
    "PhaseTorchMaturityAuditResult",
    "PhaseTorchModuleWrapperAuditResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchPhaseQNodeCompileResult",
    "PhaseTorchPhaseQNodeLoweringMatrixResult",
    "PhaseTorchPhaseQNodeLoweringRoute",
    "PhaseTorchPhaseQNodeStatevectorResult",
    "PhaseTorchPhaseQNodeTransformResult",
    "PhaseTorchQNNGradientResult",
    "PhaseTorchTrainingLoopAuditResult",
    "is_phase_torch_available",
    "plan_torch_cloud_validation_batch",
    "run_torch_compile_compatibility_audit",
    "run_torch_ecosystem_maturity_audit",
    "run_torch_func_compatibility_audit",
    "run_torch_maturity_audit",
    "run_torch_module_wrapper_audit",
    "run_torch_phase_qnode_lowering_matrix",
    "run_torch_training_loop_audit",
    "torch_autograd_qnn_value_and_grad",
    "torch_bounded_qnn_value_and_grad",
    "torch_bounded_qnn_layer",
    "torch_bounded_qnn_module",
    "torch_parameter_shift_value_and_grad",
    "torch_phase_qnode_compile_boundary_audit",
    "torch_phase_qnode_compile_audit",
    "torch_phase_qnode_transform_audit",
    "torch_phase_qnode_value_and_grad",
]
