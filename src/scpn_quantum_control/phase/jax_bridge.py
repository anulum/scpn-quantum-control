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
transform audits live in :mod:`.jax_compatibility`. This module retains public
wrappers, lowering matrices, cloud planning, and maturity orchestration.
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
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
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
from .jax_qnode_transforms import (
    _as_parameter_batch,
    _jax_local_devices,
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
    routes = (
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_native_value_and_grad",
            status="passed",
            reason="bounded phase-QNN loss is expressed directly in JAX operations",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_custom_vjp",
            status="passed",
            reason="bounded phase-QNN custom VJP route is registered without host callbacks",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_jit_value_and_grad",
            status="passed",
            reason="bounded phase-QNN JIT value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_vmap_value_and_grad",
            status="passed",
            reason="bounded phase-QNN VMAP value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_pytree_value_and_grad",
            status="passed",
            reason="bounded phase-QNN PyTree value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_statevector_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits lower into "
                "native JAX value-and-gradient execution without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_native_transform_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute "
                "through native JAX grad, value_and_grad, jacfwd, jacrev, hessian, "
                "jvp, vjp, vmap, and jit routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_pytree_transform_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute "
                "structured PyTree parameters through native JAX grad, value_and_grad, "
                "jacfwd, jacrev, jvp, vjp, vmap, and jit routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_pmap_sharding_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuit batches execute "
                "one row per local JAX device through native pmap value-and-gradient "
                "routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_aot_export_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector value routes can be "
                "staged through JAX AOT lowering and jax.export serialization diagnostics "
                "without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_finite_shot_lowering",
            status="blocked",
            reason="finite-shot JAX lowering needs sampler, seed, and uncertainty provenance",
            host_callback=False,
            requires=(
                "shot_policy",
                "rng_seed_provenance",
                "uncertainty_artifact",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_provider_lowering",
            status="blocked",
            reason="provider callbacks are not native JAX transform-safe routes",
            host_callback=False,
            requires=(
                "provider_allowlist",
                "callback_transform_safety_audit",
                "provider_execution_artifact",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_hardware_lowering",
            status="blocked",
            reason="live hardware JAX lowering requires ticketed execution evidence",
            host_callback=False,
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_dynamic_circuit_lowering",
            status="blocked",
            reason="mid-circuit measurement and feedback are outside the native JAX lowering boundary",
            host_callback=False,
            requires=(
                "dynamic_circuit_semantics",
                "classical_feedback_contract",
                "gradient_policy",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion requires isolated benchmark evidence",
            host_callback=False,
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseJAXPhaseQNodeLoweringMatrixResult(routes=routes)


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
    clean_runner = runner.strip()
    if not clean_runner:
        raise ValueError("runner must be a non-empty string")
    clean_backend = accelerator_backend.strip().lower()
    if clean_backend not in {"cuda", "rocm"}:
        raise ValueError("accelerator_backend must be 'cuda' or 'rocm'")

    jax_module, _ = _load_jax()
    local_device_count_fn = getattr(jax_module, "local_device_count", None)
    local_device_count = int(local_device_count_fn()) if callable(local_device_count_fn) else 0
    device_descriptions = _jax_local_devices(jax_module, max(local_device_count, 0))
    local_skip_reason = _jax_cloud_local_skip_reason(
        accelerator_backend=clean_backend,
        local_device_count=local_device_count,
        device_descriptions=device_descriptions,
    )
    blocked_local_routes = _jax_cloud_blocked_routes(
        accelerator_backend=clean_backend,
        local_skip_reason=local_skip_reason,
        local_device_count=local_device_count,
    )
    local_execution_status = (
        "local_accelerator_ready"
        if not blocked_local_routes
        else "skipped_incompatible_local_hardware"
    )
    commands = (
        ".venv/bin/python -m pytest "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_registered_qnode_sharding_transform_audit_uses_no_callback "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_sharding_compatibility_audit_batches_native_and_custom_vjp "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes -q",
        ".venv/bin/python -m pytest "
        "tests/test_differentiable_programming_benchmarks.py::"
        "test_quantum_gradient_benchmark_suite_matches_analytic_references -q",
        ".venv/bin/python - <<'PY'\n"
        "from scpn_quantum_control.phase import plan_jax_cloud_validation_batch\n"
        "print(plan_jax_cloud_validation_batch().to_dict())\n"
        "PY",
    )
    return PhaseJAXCloudValidationRunSpec(
        runner=clean_runner,
        local_execution_status=local_execution_status,
        local_skip_reason=local_skip_reason,
        accelerator_backend=clean_backend,
        local_device_count=local_device_count,
        device_descriptions=device_descriptions,
        blocked_local_routes=blocked_local_routes,
        required_artifacts=(
            "jax_cuda_device_metadata_artifact",
            "jax_xla_gpu_compile_artifact",
            "registered_phase_qnode_jax_pmap_sharding_artifact",
            "jax_multi_device_value_and_gradient_artifact",
            "isolated_benchmark_artifact",
            "host_load_and_affinity_metadata",
        ),
        required_environment={
            "accelerator_backend": clean_backend,
            "minimum_cuda_compute_capability": "7.5" if clean_backend == "cuda" else None,
            "minimum_visible_device_count": 2,
            "blocked_local_device_patterns": ("GTX 1060",),
            "visible_device_metadata_required": True,
            "host_load_metadata_required": True,
            "isolated_affinity_required_for_promotion": True,
            "network_required": False,
            "hardware_submission_allowed": False,
        },
        commands=commands,
        ready_for_cloud_dispatch=bool(blocked_local_routes),
    )


def _jax_cloud_local_skip_reason(
    *,
    accelerator_backend: str,
    local_device_count: int,
    device_descriptions: tuple[str, ...],
) -> str:
    joined_devices = " ".join(device_descriptions)
    lower_devices = joined_devices.lower()
    reasons: list[str] = []
    if local_device_count < 2:
        reasons.append(
            "JAX PMAP promotion requires at least two visible accelerator devices; "
            f"local_device_count={local_device_count}"
        )
    if accelerator_backend == "cuda":
        if "gtx 1060" in lower_devices:
            reasons.append(
                "local GTX 1060 does not satisfy the CUDA cloud validation floor "
                "or current JAX CUDA wheel route"
            )
        elif not any(token in lower_devices for token in ("cuda", "gpu", "nvidia")):
            reasons.append("no CUDA/GPU JAX device metadata is visible locally")
    elif not any(token in lower_devices for token in ("rocm", "amd", "gpu")):
        reasons.append("no ROCm/GPU JAX device metadata is visible locally")
    return "; ".join(reasons)


def _jax_cloud_blocked_routes(
    *,
    accelerator_backend: str,
    local_skip_reason: str,
    local_device_count: int,
) -> tuple[str, ...]:
    routes: list[str] = []
    if local_skip_reason:
        routes.append(f"jax_{accelerator_backend}_accelerator_device")
    if local_device_count < 2 or local_skip_reason:
        routes.append("registered_phase_qnode_pmap_multi_device_lowering")
    if routes:
        routes.append("isolated_benchmark_artifact")
    return tuple(dict.fromkeys(routes))


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
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    parameter_batch = _as_parameter_batch(
        "params_batch",
        params_batch,
        width=feature_matrix.shape[1],
    )

    custom_vjp = jax_custom_vjp_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    jit = run_jax_jit_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
    )
    vmap = run_jax_vmap_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    sharding = run_jax_sharding_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    pytree = run_jax_pytree_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
    )
    nested_transform_algebra = run_jax_nested_transform_algebra_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
    )
    phase_qnode_lowering_matrix = run_jax_phase_qnode_lowering_matrix()
    cloud_validation_batch = plan_jax_cloud_validation_batch()

    evidence: dict[str, object] = {
        "custom_vjp": custom_vjp,
        "jit": jit,
        "vmap": vmap,
        "pmap_sharding": sharding,
        "pytree": pytree,
        "nested_transform_algebra": nested_transform_algebra,
        "phase_qnode_lowering_matrix": phase_qnode_lowering_matrix,
        "cloud_validation_batch": cloud_validation_batch,
    }
    bounded_model_ready = all(
        bool(getattr(result, "passed", False))
        for name, result in evidence.items()
        if name
        not in {
            "phase_qnode_lowering_matrix",
            "cloud_validation_batch",
        }
    )
    required_capabilities = {
        "custom_vjp": "passed" if custom_vjp.passed else "failed",
        "jit": "passed" if jit.passed else "failed",
        "vmap": "passed" if vmap.passed else "failed",
        "pmap_sharding": "passed" if sharding.passed else "failed",
        "pytree": "passed" if pytree.passed else "failed",
        "nested_transform_algebra": "passed" if nested_transform_algebra.passed else "failed",
        "phase_qnode_lowering_matrix": (
            "passed" if phase_qnode_lowering_matrix.ready_for_provider_exceedance else "blocked"
        ),
        "cloud_validation_batch": (
            "scheduled" if cloud_validation_batch.ready_for_cloud_dispatch else "not_required"
        ),
        "arbitrary_quantum_kernel_jax_lowering": "blocked",
        "hardware_or_provider_callback_transform_safety": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    required_capabilities.update(
        {
            f"nested_transform:{route.name}": route.status
            for route in nested_transform_algebra.routes
            if route.status != "passed"
        }
    )
    required_capabilities.update(
        {
            f"phase_qnode_lowering:{route.name}": route.status
            for route in phase_qnode_lowering_matrix.routes
            if route.status != "passed"
        }
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PhaseJAXMaturityAuditResult(
        bounded_model_ready=bounded_model_ready,
        ready_for_provider_exceedance=bounded_model_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        open_gaps=open_gaps,
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
