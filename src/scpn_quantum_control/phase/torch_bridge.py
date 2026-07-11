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
:mod:`.torch_gradients`. Registered-QNode statevector and compiler diagnostics
live in :mod:`.torch_qnode_transforms`; bounded transform compatibility, module,
and training routes live in :mod:`.torch_compatibility`. This module retains
lowering, cloud, and maturity orchestration while those responsibilities undergo
bounded decomposition.
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
from .torch_compatibility import (
    _as_positive_learning_rate as _as_positive_learning_rate,
)
from .torch_compatibility import (
    _as_positive_step_count as _as_positive_step_count,
)
from .torch_compatibility import (
    _torch_bounded_qnn_loss_tensor as _torch_bounded_qnn_loss_tensor,
)
from .torch_compatibility import (
    _torch_nn_module_and_parameter as _torch_nn_module_and_parameter,
)
from .torch_compatibility import (
    _torch_parameter_count as _torch_parameter_count,
)
from .torch_compatibility import (
    run_torch_compile_compatibility_audit as _run_torch_compile_compatibility_audit,
)
from .torch_compatibility import (
    run_torch_func_compatibility_audit as _run_torch_func_compatibility_audit,
)
from .torch_compatibility import (
    run_torch_module_wrapper_audit as _run_torch_module_wrapper_audit,
)
from .torch_compatibility import (
    run_torch_training_loop_audit as _run_torch_training_loop_audit,
)
from .torch_compatibility import (
    torch_bounded_qnn_layer as _torch_bounded_qnn_layer,
)
from .torch_compatibility import (
    torch_bounded_qnn_module as _torch_bounded_qnn_module,
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
    return _run_torch_func_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=tolerance,
        _torch_loader=_load_torch,
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
    return _run_torch_compile_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        fullgraph=fullgraph,
        dynamic=dynamic,
        _torch_loader=_load_torch,
    )


def torch_bounded_qnn_module(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
) -> Any:
    """Return a PyTorch ``nn.Module`` wrapper for the bounded phase-QNN loss."""
    return _torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
        _torch_loader=_load_torch,
    )


def torch_bounded_qnn_layer(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
) -> Any:
    """Return the bounded phase-QNN wrapper using layer-oriented naming."""
    return _torch_bounded_qnn_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
        _torch_loader=_load_torch,
    )


def run_torch_module_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchModuleWrapperAuditResult:
    """Audit bounded phase-QNN PyTorch module/layer wrapper gradients."""
    return _run_torch_module_wrapper_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        tolerance=tolerance,
        _torch_loader=_load_torch,
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
    return _run_torch_training_loop_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        learning_rate=learning_rate,
        steps=steps,
        tolerance=tolerance,
        fullgraph=fullgraph,
        dynamic=dynamic,
        _torch_loader=_load_torch,
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
