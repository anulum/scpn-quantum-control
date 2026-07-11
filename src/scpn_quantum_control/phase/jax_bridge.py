# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase JAX Bridge
"""Optional JAX execution and compatibility facade for phase gradients.

Immutable JAX result records live in :mod:`.jax_bridge_contracts`, and bounded
gradient implementations live in :mod:`.jax_gradients`. Registered-QNode
statevector, transform, sharding, and AOT routes live in
:mod:`.jax_qnode_transforms`. Bounded framework-compatibility and nested
transform audits live in :mod:`.jax_compatibility`. Lowering declarations,
cloud planning, and maturity aggregation live in :mod:`.jax_maturity`. This
module is the signature-stable public wrapper facade.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
)
from .jax_bridge_contracts import (
    PhaseJAXCloudValidationRunSpec,
    PhaseJAXCustomVJPQNNGradientResult,
    PhaseJAXGradientAgreementResult,
    PhaseJAXJITCompatibilityResult,
    PhaseJAXMaturityAuditResult,
    PhaseJAXNativeQNNGradientResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXNestedTransformRoute,
    PhaseJAXParameterShiftResult,
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseJAXPhaseQNodeLoweringMatrixResult,
    PhaseJAXPhaseQNodeLoweringRoute,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeShardingTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
)
from .jax_compatibility import (
    run_jax_jit_compatibility_audit as _run_jax_jit_compatibility_audit,
)
from .jax_compatibility import (
    run_jax_nested_transform_algebra_audit as _run_jax_nested_transform_algebra_audit,
)
from .jax_compatibility import (
    run_jax_pytree_compatibility_audit as _run_jax_pytree_compatibility_audit,
)
from .jax_compatibility import (
    run_jax_sharding_compatibility_audit as _run_jax_sharding_compatibility_audit,
)
from .jax_compatibility import (
    run_jax_vmap_compatibility_audit as _run_jax_vmap_compatibility_audit,
)
from .jax_gradients import (
    _load_jax,
)
from .jax_gradients import (
    check_jax_parameter_shift_agreement as _check_jax_parameter_shift_agreement,
)
from .jax_gradients import (
    jax_custom_vjp_qnn_value_and_grad as _jax_custom_vjp_qnn_value_and_grad,
)
from .jax_gradients import (
    jax_native_qnn_value_and_grad as _jax_native_qnn_value_and_grad,
)
from .jax_gradients import (
    jax_parameter_shift_value_and_grad as _jax_parameter_shift_value_and_grad,
)
from .jax_maturity import (
    plan_jax_cloud_validation_batch as _plan_jax_cloud_validation_batch,
)
from .jax_maturity import (
    run_jax_maturity_audit as _run_jax_maturity_audit,
)
from .jax_maturity import (
    run_jax_phase_qnode_lowering_matrix as _run_jax_phase_qnode_lowering_matrix,
)
from .jax_qnode_transforms import (
    jax_phase_qnode_aot_export_audit as _jax_phase_qnode_aot_export_audit,
)
from .jax_qnode_transforms import (
    jax_phase_qnode_native_transform_audit as _jax_phase_qnode_native_transform_audit,
)
from .jax_qnode_transforms import (
    jax_phase_qnode_pytree_transform_audit as _jax_phase_qnode_pytree_transform_audit,
)
from .jax_qnode_transforms import (
    jax_phase_qnode_sharding_transform_audit as _jax_phase_qnode_sharding_transform_audit,
)
from .jax_qnode_transforms import (
    jax_phase_qnode_value_and_grad as _jax_phase_qnode_value_and_grad,
)
from .qnode_circuit import (
    PhaseQNodeCircuit,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]


def is_phase_jax_available() -> bool:
    """Return whether the optional phase JAX adapter can import JAX."""
    try:
        _load_jax()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return False
    return True


def jax_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    jit: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXParameterShiftResult:
    """Return parameter-shift value and gradient through the optional JAX bridge.

    This adapter is an interop boundary, not a native quantum-kernel compiler.
    Without ``jit`` it imports JAX, accepts JAX-like inputs, and executes the
    repository's deterministic parameter-shift rule on the host. With ``jit`` it
    wraps that host execution in ``jax.pure_callback`` and ``jax.jit`` so JAX
    workflows can stage the call explicitly while provenance still reports
    ``host_callback=True``.
    """
    return _jax_parameter_shift_value_and_grad(
        objective,
        values,
        jit=jit,
        parameters=parameters,
        rule=rule,
        _jax_loader=_load_jax,
    )


def check_jax_parameter_shift_agreement(
    objective: ScalarObjective,
    jax_gradient: GradientCallable,
    values: ArrayLike,
    *,
    tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXGradientAgreementResult:
    """Compare SCPN parameter-shift gradients with a JAX-derived gradient callable.

    ``jax_gradient`` is caller-supplied so the bridge can verify agreement with
    ``jax.grad`` or equivalent JAX code without claiming automatic conversion of
    every SCPN objective into a native JAX quantum kernel.
    """
    return _check_jax_parameter_shift_agreement(
        objective,
        jax_gradient,
        values,
        tolerance=tolerance,
        parameters=parameters,
        rule=rule,
        _jax_loader=_load_jax,
    )


def jax_native_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXNativeQNNGradientResult:
    """Differentiate the bounded phase-QNN loss with native JAX autodiff.

    This route is intentionally narrower than arbitrary simulator autodiff. It
    expresses the repository's bounded phase-QNN classifier loss directly in
    JAX tensor operations, obtains a native ``value_and_grad`` result, and
    records agreement against the existing multi-frequency parameter-shift
    gradient. It does not use ``pure_callback`` and does not claim conversion of
    arbitrary quantum programs into JAX kernels.
    """
    return _jax_native_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        jit=jit,
        _jax_loader=_load_jax,
    )


def jax_custom_vjp_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXCustomVJPQNNGradientResult:
    """Differentiate the bounded phase-QNN loss through a JAX custom VJP.

    The primal is the same bounded phase-QNN MSE loss used by
    ``jax_native_qnn_value_and_grad``. The VJP rule is registered explicitly
    and returns the mathematically equivalent bounded-QNN derivative, then the
    result is checked against the repository's multi-frequency parameter-shift
    reference. This route is still intentionally narrow: it does not expose
    arbitrary simulator autodiff, provider callbacks, or hardware execution.
    """
    return _jax_custom_vjp_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        jit=jit,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into JAX.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It deliberately excludes finite-shot sampling, provider
    callbacks, hardware execution, density/noise channels, and dynamic circuits.
    """
    return _jax_phase_qnode_value_and_grad(
        circuit,
        params,
        tolerance=tolerance,
        jit=jit,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_native_transform_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Audit native JAX transforms for a registered local Phase-QNode.

    The executable route is the same deterministic statevector lowering used by
    :func:`jax_phase_qnode_value_and_grad`, but it is exercised through JAX
    ``grad``, ``value_and_grad``, ``jacfwd``, ``jacrev``, ``hessian``, ``jvp``,
    ``vjp``, ``vmap``, and ``jit``. It deliberately does not use host callbacks
    and does not promote finite-shot, provider, hardware, density/noise, or
    dynamic-circuit lowering.
    """
    return _jax_phase_qnode_native_transform_audit(
        circuit,
        params,
        tangent=tangent,
        batch_offsets=batch_offsets,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_pytree_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_pytree: object,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodePyTreeTransformResult:
    """Audit native JAX PyTree transforms for a registered local Phase-QNode.

    The route accepts nested numeric PyTree parameter containers, lowers them
    into the registered deterministic Phase-QNode statevector value function,
    and validates native JAX ``grad``, ``value_and_grad``, ``jacfwd``,
    ``jacrev``, ``hessian``, ``jvp``, ``vjp``, ``vmap``, and ``jit`` against the
    canonical SCPN parameter-shift gradient. It keeps the same fail-closed
    boundary as the flat transform audit: no host callbacks, no finite-shot
    lowering, no provider execution, no hardware submission, and no
    dynamic-circuit claim.
    """
    return _jax_phase_qnode_pytree_transform_audit(
        circuit,
        params_pytree,
        tangent=tangent,
        batch_offsets=batch_offsets,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_sharding_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_batch: ArrayLike,
    *,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Audit native JAX PMAP lowering for registered local Phase-QNode batches.

    The audit maps one deterministic statevector value-and-gradient row per
    local JAX device via ``jax.pmap`` and compares every row against the SCPN
    gate-aware parameter-shift reference. It is local sharding evidence only:
    no host callbacks, no finite-shot lowering, no provider execution, no
    hardware submission, and no wall-clock performance promotion.
    """
    return _jax_phase_qnode_sharding_transform_audit(
        circuit,
        params_batch,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_aot_export_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeAOTExportResult:
    """Audit JAX AOT lowering and export metadata for a registered Phase-QNode.

    The audit stages the deterministic local statevector value route through
    ``jax.jit(...).lower(...)``, records StableHLO/compiler metadata, exports
    the same jitted route through ``jax.export.export(...)``, serializes and
    deserializes it, and compares the compiled/exported values against the
    canonical SCPN parameter-shift value. The result is diagnostic evidence
    only: it does not promote persistent cross-platform execution, VJP export,
    provider callbacks, hardware execution, or performance claims.
    """
    return _jax_phase_qnode_aot_export_audit(
        circuit,
        params,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_jit_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXJITCompatibilityResult:
    """Audit bounded JAX JIT support without promoting host callbacks.

    The audit exercises three JIT-facing routes:

    - bounded native QNN ``value_and_grad`` with ``host_callback=False``;
    - bounded QNN ``custom_vjp`` with ``host_callback=False``;
    - parameter-shift interop under ``jax.pure_callback``, recorded as a
      host-callback route and therefore excluded from native-JIT promotion.
    """
    return _run_jax_jit_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_vmap_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXVMAPCompatibilityResult:
    """Audit bounded JAX VMAP support for parameter batches.

    The audit vectorises the bounded phase-QNN native and custom-VJP loss routes
    over a two-dimensional parameter batch. SCPN parameter-shift results are
    used as host-side references only and are reported as such rather than
    promoted as native VMAP.
    """
    return _run_jax_vmap_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_sharding_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXShardingCompatibilityResult:
    """Audit bounded JAX PMAP/sharding support for local-device batches.

    The audit maps one parameter row per local JAX device with ``jax.pmap``.
    It promotes only the bounded native and custom-VJP phase-QNN loss routes.
    SCPN parameter-shift references stay host-side validation rows and are not
    reported as sharded/native execution.
    """
    return _run_jax_sharding_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_pytree_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXPyTreeCompatibilityResult:
    """Audit bounded JAX PyTree parameter support.

    The audit accepts a JAX PyTree of numeric parameter leaves, flattens it into
    the bounded phase-QNN parameter vector, and restores gradients into the same
    PyTree structure. It does not claim arbitrary simulator PyTree lowering.
    """
    return _run_jax_pytree_compatibility_audit(
        features=features,
        labels=labels,
        params_pytree=params_pytree,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_nested_transform_algebra_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXNestedTransformAlgebraResult:
    """Audit bounded JAX nested-transform algebra for the phase-QNN route.

    This verifies only the implemented bounded classifier path. It does not
    promote arbitrary Phase-QNode lowering, full `jacfwd`/`jacrev` coverage,
    hardware/provider callbacks, or isolated benchmark evidence.
    """
    return _run_jax_nested_transform_algebra_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        params_pytree=params_pytree,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


def run_jax_phase_qnode_lowering_matrix() -> PhaseJAXPhaseQNodeLoweringMatrixResult:
    """Return the JAX parity matrix for registered Phase-QNode lowering.

    The bounded phase-QNN JAX routes are no-host-callback native framework
    evidence. Arbitrary registered Phase-QNode circuit lowering remains blocked
    until native JAX lowering rules and parity artefacts exist.
    """
    return _run_jax_phase_qnode_lowering_matrix()


def plan_jax_cloud_validation_batch(
    *,
    runner: str = "jarvislabs",
    accelerator_backend: str = "cuda",
) -> PhaseJAXCloudValidationRunSpec:
    """Plan the JAX cloud validation batch for blocked local accelerator routes.

    Parameters
    ----------
    runner:
        Human-readable runner label used in the downstream validation queue.
    accelerator_backend:
        Accelerator runtime requested for the cloud rerun. The differentiable
        lane currently accepts ``"cuda"`` and ``"rocm"`` plans.

    Returns
    -------
    PhaseJAXCloudValidationRunSpec
        JSON-ready scheduling metadata with local skip status, required cloud
        artefacts, environment constraints, and reproduction commands.
    """
    return _plan_jax_cloud_validation_batch(
        runner=runner,
        accelerator_backend=accelerator_backend,
        _jax_loader=_load_jax,
    )


def run_jax_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    params_batch: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXMaturityAuditResult:
    """Aggregate bounded JAX evidence and provider-level parity blockers.

    The audit intentionally separates the bounded phase-QNN evidence that is
    implemented today from the larger JAX ecosystem maturity target. It does
    not promote arbitrary quantum kernels, provider callbacks, hardware
    gradients, or benchmark claims until those routes have their own artefacts.
    """
    return _run_jax_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        params_pytree=params_pytree,
        tolerance=tolerance,
        _jax_loader=_load_jax,
    )


__all__ = [
    "PhaseJAXCloudValidationRunSpec",
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXJITCompatibilityResult",
    "PhaseJAXMaturityAuditResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXNestedTransformAlgebraResult",
    "PhaseJAXNestedTransformRoute",
    "PhaseJAXParameterShiftResult",
    "PhaseJAXPhaseQNodeLoweringMatrixResult",
    "PhaseJAXPhaseQNodeLoweringRoute",
    "PhaseJAXPhaseQNodeAOTExportResult",
    "PhaseJAXPhaseQNodeNativeTransformResult",
    "PhaseJAXPhaseQNodePyTreeTransformResult",
    "PhaseJAXPhaseQNodeShardingTransformResult",
    "PhaseJAXPhaseQNodeStatevectorResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
    "check_jax_parameter_shift_agreement",
    "is_phase_jax_available",
    "jax_custom_vjp_qnn_value_and_grad",
    "jax_native_qnn_value_and_grad",
    "jax_parameter_shift_value_and_grad",
    "jax_phase_qnode_aot_export_audit",
    "jax_phase_qnode_native_transform_audit",
    "jax_phase_qnode_pytree_transform_audit",
    "jax_phase_qnode_sharding_transform_audit",
    "jax_phase_qnode_value_and_grad",
    "plan_jax_cloud_validation_batch",
    "run_jax_jit_compatibility_audit",
    "run_jax_maturity_audit",
    "run_jax_nested_transform_algebra_audit",
    "run_jax_phase_qnode_lowering_matrix",
    "run_jax_pytree_compatibility_audit",
    "run_jax_sharding_compatibility_audit",
    "run_jax_vmap_compatibility_audit",
]
