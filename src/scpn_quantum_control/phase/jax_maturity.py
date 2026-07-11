# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Maturity Orchestration
"""JAX lowering declarations, cloud planning, and maturity aggregation.

This one-way leaf composes the bounded gradient, registered-QNode, and
compatibility evidence surfaces. The public facade injects its active
optional-JAX loader and preserves established signatures.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from .jax_bridge_contracts import (
    PhaseJAXCloudValidationRunSpec,
    PhaseJAXMaturityAuditResult,
    PhaseJAXPhaseQNodeLoweringMatrixResult,
    PhaseJAXPhaseQNodeLoweringRoute,
)
from .jax_compatibility import (
    run_jax_jit_compatibility_audit,
    run_jax_nested_transform_algebra_audit,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
)
from .jax_gradients import (
    JAXLoader,
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_jax,
    jax_custom_vjp_qnn_value_and_grad,
)
from .jax_qnode_transforms import (
    _as_parameter_batch,
    _jax_local_devices,
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
    _jax_loader: JAXLoader = _load_jax,
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

    jax_module, _ = _jax_loader()
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
    _jax_loader: JAXLoader = _load_jax,
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
        _jax_loader=_jax_loader,
    )
    jit = run_jax_jit_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        _jax_loader=_jax_loader,
    )
    vmap = run_jax_vmap_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
        _jax_loader=_jax_loader,
    )
    sharding = run_jax_sharding_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
        _jax_loader=_jax_loader,
    )
    pytree = run_jax_pytree_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
        _jax_loader=_jax_loader,
    )
    nested_transform_algebra = run_jax_nested_transform_algebra_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
        _jax_loader=_jax_loader,
    )
    phase_qnode_lowering_matrix = run_jax_phase_qnode_lowering_matrix()
    cloud_validation_batch = plan_jax_cloud_validation_batch(_jax_loader=_jax_loader)

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
    "plan_jax_cloud_validation_batch",
    "run_jax_maturity_audit",
    "run_jax_phase_qnode_lowering_matrix",
]
