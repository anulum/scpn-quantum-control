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
and training routes live in :mod:`.torch_compatibility`. Lowering, ecosystem,
cloud, overlay, and maturity orchestration live in :mod:`.torch_maturity`. This
module is the signature-stable public facade over those one-way leaves.
"""

from __future__ import annotations

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
from .torch_maturity import (
    _load_torch_live_overlay_evidence as _load_torch_live_overlay_evidence,
)
from .torch_maturity import (
    _required_float as _required_float,
)
from .torch_maturity import (
    _required_int as _required_int,
)
from .torch_maturity import (
    _required_str as _required_str,
)
from .torch_maturity import (
    _torch_cuda_metadata as _torch_cuda_metadata,
)
from .torch_maturity import (
    plan_torch_cloud_validation_batch as _plan_torch_cloud_validation_batch,
)
from .torch_maturity import (
    run_torch_ecosystem_maturity_audit as _run_torch_ecosystem_maturity_audit,
)
from .torch_maturity import (
    run_torch_maturity_audit as _run_torch_maturity_audit,
)
from .torch_maturity import (
    run_torch_phase_qnode_lowering_matrix as _run_torch_phase_qnode_lowering_matrix,
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
    return _run_torch_phase_qnode_lowering_matrix()


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
    return _run_torch_ecosystem_maturity_audit(
        _torch_loader=_load_torch,
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
    return _plan_torch_cloud_validation_batch(
        runner=runner,
        accelerator_backend=accelerator_backend,
        _ecosystem_runner=run_torch_ecosystem_maturity_audit,
        _lowering_runner=run_torch_phase_qnode_lowering_matrix,
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
    return _run_torch_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=tolerance,
        fullgraph=fullgraph,
        dynamic=dynamic,
        live_overlay_artifact_path=live_overlay_artifact_path,
        _analytic_tensor_runner=torch_bounded_qnn_value_and_grad,
        _custom_autograd_runner=torch_autograd_qnn_value_and_grad,
        _func_runner=run_torch_func_compatibility_audit,
        _compile_runner=run_torch_compile_compatibility_audit,
        _module_wrapper_runner=run_torch_module_wrapper_audit,
        _training_loop_runner=run_torch_training_loop_audit,
        _ecosystem_runner=run_torch_ecosystem_maturity_audit,
        _cloud_runner=plan_torch_cloud_validation_batch,
        _lowering_runner=run_torch_phase_qnode_lowering_matrix,
        _overlay_loader=_load_torch_live_overlay_evidence,
    )


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
