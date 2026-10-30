# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Registered-QNode Transforms
"""Native JAX execution and transforms for registered local Phase-QNodes.

This one-way leaf owns deterministic statevector, flat/PyTree transform,
PMAP-sharding, and AOT/export routes. The compatibility facade injects its
optional-JAX loader and retains later compatibility and maturity orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from functools import partial
from importlib import import_module
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable_axis_operators import jax_binary_axis_matmul as _jax_binary_axis_matmul
from ..differentiable_axis_operators import restore_binary_axis_order as _restore_binary_axis_order
from ..differentiable_tree_vectors import (
    batched_gradient_tree_matrix as _batched_gradient_tree_matrix,
)
from ..differentiable_tree_vectors import gradient_tree_vector as _gradient_tree_vector
from ..differentiable_tree_vectors import hessian_tree_matrix as _hessian_tree_matrix
from ..differentiable_tree_vectors import jax_batch_tree as _jax_batch_tree
from ..differentiable_tree_vectors import jax_tree_vector as _jax_tree_vector
from ..differentiable_tree_vectors import jax_vector_tree as _jax_vector_tree
from ..differentiable_tree_vectors import parameter_tree_vector as _parameter_tree_vector
from ..phase_qnode_audit_admission import (
    admitted_parameter_batch as _admitted_parameter_batch,
)
from ..phase_qnode_gate_dispatch import GateOperatorFactories as _GateOperatorFactories
from ..phase_qnode_gate_dispatch import dispatch_gate_operator as _dispatch_gate_operator
from ..phase_qnode_gate_numerics import (
    controlled_z_operator as _controlled_z_operator,
)
from ..phase_qnode_gate_numerics import (
    hadamard_operator as _hadamard_operator,
)
from ..phase_qnode_gate_numerics import (
    jax_phase_operator as _jax_phase_operator,
)
from ..phase_qnode_gate_numerics import (
    pauli_rotation as _pauli_rotation,
)
from ..phase_qnode_gate_numerics import (
    quarter_turn_phase as _quarter_turn_phase,
)
from ..phase_qnode_gate_numerics import (
    sqrt_x_operator as _sqrt_x_operator,
)
from ..phase_qnode_jax_audits import (
    JAXAuditOperations as _JAXAuditOperations,
)
from ..phase_qnode_jax_audits import (
    enable_jax_x64 as _lower_enable_jax_x64,
)
from ..phase_qnode_jax_audits import (
    jax_aot_export_audit as _jax_aot_export_audit,
)
from ..phase_qnode_jax_audits import (
    jax_export_version as _lower_jax_export_version,
)
from ..phase_qnode_jax_audits import (
    jax_local_devices as _lower_jax_local_devices,
)
from ..phase_qnode_jax_audits import (
    jax_native_transform_audit as _jax_native_transform_audit,
)
from ..phase_qnode_jax_audits import (
    jax_pytree_transform_audit as _jax_pytree_transform_audit,
)
from ..phase_qnode_jax_audits import (
    jax_sharding_transform_audit as _jax_sharding_transform_audit,
)
from ..phase_qnode_jax_audits import (
    require_jax_aot_export_support as _require_jax_aot_export_support,
)
from ..phase_qnode_jax_audits import (
    require_jax_pmap_support as _lower_require_jax_pmap_support,
)
from ..phase_qnode_jax_audits import (
    require_jax_pytree_support as _lower_require_jax_pytree_support,
)
from ..phase_qnode_jax_audits import (
    require_jax_transform_support as _require_jax_transform_support,
)
from ..phase_qnode_native_operators import (
    jax_basis_transposition as _basis_transposition,
)
from ..phase_qnode_native_operators import (
    jax_controlled_operator as _controlled_operator,
)
from ..phase_qnode_native_operators import (
    pauli_operator_entries as _pauli_operator_entries,
)
from ..phase_qnode_native_operators import (
    phase_operator_entries as _phase_operator_entries,
)
from ..phase_qnode_native_operators import (
    two_qubit_operator_entries as _two_qubit_operator_entries,
)
from ..phase_qnode_native_statevector import (
    jax_operation_theta as _native_jax_operation_theta,
)
from ..phase_qnode_state_evolution import jax_zero_state as _jax_zero_state
from ..phase_qnode_statevector_engine import (
    StatevectorKernels as _StatevectorKernels,
)
from ..phase_qnode_statevector_engine import (
    apply_term_operator as _apply_term_operator,
)
from ..phase_qnode_statevector_engine import (
    observable_expectation as _observable_expectation,
)
from ..phase_qnode_statevector_engine import (
    statevector_value_and_state as _statevector_value_and_state,
)
from ..phase_qnode_statevector_engine import (
    symmetrized_product_expectation as _symmetrized_product_expectation,
)
from ..phase_qnode_statevector_engine import (
    term_expectation as _term_expectation,
)
from ..phase_qnode_statevector_engine import (
    term_product_expectation as _term_product_expectation,
)
from .jax_bridge_contracts import (
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeShardingTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
)
from .jax_gradients import (
    JAXLoader,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _as_scalar,
    _load_jax,
)
from .qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
)

FloatArray: TypeAlias = NDArray[np.float64]
JAXCallable = Callable[[object], object]


def _as_parameter_batch(name: str, values: object, *, width: int) -> FloatArray:
    return _admitted_parameter_batch(name, values, width=width)


def _require_jax_pmap_support(jax_module: Any) -> None:
    _lower_require_jax_pmap_support(jax_module)


def _require_jax_phase_qnode_transform_support(jax_module: Any) -> None:
    _require_jax_transform_support(jax_module, surface="native transforms")


def _require_jax_phase_qnode_aot_export_support(jax_module: Any) -> Any:
    return _require_jax_aot_export_support(
        jax_module, importer=lambda: import_module, suppressor=lambda: suppress
    )


def _jax_export_version(export_module: Any, name: str) -> int:
    """Return a JAX export calling-convention version from attr or function APIs."""
    return _lower_jax_export_version(export_module, name)


def _require_jax_phase_qnode_pytree_transform_support(jax_module: Any) -> None:
    _require_jax_transform_support(jax_module, surface="PyTree transforms")


def _require_jax_pytree_support(jax_module: Any) -> None:
    _lower_require_jax_pytree_support(jax_module)


def _jax_local_devices(jax_module: Any, local_device_count: int) -> tuple[str, ...]:
    return _lower_jax_local_devices(jax_module, local_device_count)


def _as_pytree_parameter_vector(
    jax_module: Any,
    name: str,
    params_pytree: object,
) -> tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate and flatten a numeric parameter PyTree with reconstruction metadata."""
    return _parameter_tree_vector(
        jax_module,
        name,
        params_pytree,
        numpy=lambda: np,
        scalar=lambda: float,
        integer=lambda: int,
        sequence=lambda: tuple,
        indexed=lambda: enumerate,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _jax_flatten_pytree(jax_module: Any, jnp: Any, params_pytree: object) -> object:
    return _jax_tree_vector(jax_module, jnp, params_pytree, length=lambda: len)


def _jax_unflatten_vector_to_pytree(
    jax_module: Any,
    jnp: Any,
    vector: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    return _jax_vector_tree(
        jax_module, jnp, vector, treedef, leaf_shapes, leaf_sizes, paired=lambda: zip
    )


def _jax_unflatten_batch_to_pytree(
    jax_module: Any,
    jnp: Any,
    batch: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    return _jax_batch_tree(
        jax_module, jnp, batch, treedef, leaf_shapes, leaf_sizes, paired=lambda: zip
    )


def _flatten_runtime_pytree_gradient(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    width: int,
) -> FloatArray:
    return _gradient_tree_vector(
        jax_module,
        name,
        gradient_pytree,
        width=width,
        numpy=lambda: np,
        scalar=lambda: float,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _flatten_batched_runtime_pytree_gradient(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    batch_size: int,
    width: int,
) -> FloatArray:
    return _batched_gradient_tree_matrix(
        jax_module,
        name,
        gradient_pytree,
        batch_size=batch_size,
        width=width,
        numpy=lambda: np,
        scalar=lambda: float,
        indexed=lambda: enumerate,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _flatten_runtime_pytree_hessian(
    jax_module: Any,
    name: str,
    hessian_pytree: object,
    *,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> FloatArray:
    return _hessian_tree_matrix(
        jax_module,
        name,
        hessian_pytree,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
        numpy=lambda: np,
        scalar=lambda: float,
        integer=lambda: int,
        length=lambda: len,
        paired=lambda: zip,
        accumulate=lambda: sum,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _jax_audit_routes() -> _JAXAuditOperations:
    """Return audit operation resolvers reading this module's live names."""
    return _JAXAuditOperations(
        support_report=lambda: phase_qnode_support_report,
        parameter_shift=lambda: parameter_shift_phase_qnode_gradient,
        support_error=lambda: PhaseQNodeSupportError,
        parameter_vector=lambda: _as_parameter_vector,
        parameter_batch=lambda: _as_parameter_batch,
        scalar=lambda: _as_scalar,
        unflatten_vector=lambda: _jax_unflatten_vector_to_pytree,
        unflatten_batch=lambda: _jax_unflatten_batch_to_pytree,
        flatten_gradient=lambda: _flatten_runtime_pytree_gradient,
        flatten_batched_gradient=lambda: _flatten_batched_runtime_pytree_gradient,
        flatten_hessian=lambda: _flatten_runtime_pytree_hessian,
        local_devices=lambda: _jax_local_devices,
        export_version=lambda: _jax_export_version,
    )


def jax_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into JAX.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It deliberately excludes finite-shot sampling, provider
    callbacks, hardware execution, density/noise channels, and dynamic circuits.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    value_and_grad = jax_module.value_and_grad(value_function)
    if jit:
        value_and_grad = jax_module.jit(value_and_grad)
        state_function = jax_module.jit(
            lambda raw_params: _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)[1]
        )
        jitted = True
    else:

        def state_function(raw_params: object) -> object:
            return _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)[1]

        jitted = False

    value_obj, gradient_obj = value_and_grad(jnp.asarray(parameter_values))
    state_obj = state_function(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX Phase-QNode gradient", gradient_obj, width=parameter_values.size
    )
    state = np.asarray(state_obj, dtype=np.complex128)
    value = _as_scalar("JAX Phase-QNode value", value_obj)
    max_abs_error = float(np.max(np.abs(gradient - parameter_shift.gradient), initial=0.0))
    l2_error = float(np.linalg.norm(gradient - parameter_shift.gradient))
    passed = bool(
        abs(value - parameter_shift.value) <= tolerance_value and max_abs_error <= tolerance_value
    )
    return PhaseJAXPhaseQNodeStatevectorResult(
        value=value,
        gradient=gradient,
        state=state,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


def jax_phase_qnode_native_transform_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Audit native JAX transforms for a registered local Phase-QNode.

    The executable route is the same deterministic statevector lowering used by
    :func:`jax_phase_qnode_value_and_grad`, but it is exercised through JAX
    ``grad``, ``value_and_grad``, ``jacfwd``, ``jacrev``, ``hessian``, ``jvp``,
    ``vjp``, ``vmap``, and ``jit``. It deliberately does not use host callbacks
    and does not promote finite-shot, provider, hardware, density/noise, or
    dynamic-circuit lowering.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_phase_qnode_transform_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    return _jax_native_transform_audit(
        jax_module,
        jnp,
        circuit,
        parameter_values,
        tolerance_value,
        value_function,
        _jax_audit_routes(),
        tangent=tangent,
        batch_offsets=batch_offsets,
    )


def jax_phase_qnode_pytree_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_pytree: object,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
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
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_pytree_support(jax_module)
    _require_jax_phase_qnode_pytree_transform_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    flattened = _as_pytree_parameter_vector(
        jax_module,
        "params_pytree",
        params_pytree,
    )

    def value_function(raw_tree: object) -> object:
        vector = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, vector)
        return value

    return _jax_pytree_transform_audit(
        jax_module,
        jnp,
        circuit,
        flattened,
        tolerance_value,
        value_function,
        _jax_audit_routes(),
        tangent=tangent,
        batch_offsets=batch_offsets,
    )


def jax_phase_qnode_sharding_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_batch: ArrayLike,
    *,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Audit native JAX PMAP lowering for registered local Phase-QNode batches.

    The audit maps one deterministic statevector value-and-gradient row per
    local JAX device via ``jax.pmap`` and compares every row against the SCPN
    gate-aware parameter-shift reference. It is local sharding evidence only:
    no host callbacks, no finite-shot lowering, no provider execution, no
    hardware submission, and no wall-clock performance promotion.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_pmap_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    return _jax_sharding_transform_audit(
        jax_module,
        jnp,
        circuit,
        params_batch,
        tolerance_value,
        value_function,
        _jax_audit_routes(),
    )


def jax_phase_qnode_aot_export_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
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
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    export_module = _require_jax_phase_qnode_aot_export_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    return _jax_aot_export_audit(
        jax_module,
        jnp,
        export_module,
        circuit,
        parameter_values,
        tolerance_value,
        value_function,
        _jax_audit_routes(),
    )


def _enable_jax_x64(jax_module: Any) -> None:
    _lower_enable_jax_x64(jax_module)


def _jax_phase_qnode_value_and_state(
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
) -> tuple[object, object]:
    return _statevector_value_and_state(_jax_statevector_routes(jnp), circuit, params)


def _jax_statevector_routes(jnp: Any) -> _StatevectorKernels:
    """Return engine kernels whose routes are this module's named functions."""
    return _StatevectorKernels(
        namespace=jnp,
        parameter_tensor=lambda params: jnp.asarray(params),
        zero_state=lambda namespace, width: _jax_zero_state(namespace, width),
        circuit_operations=lambda circuit: cast(
            tuple[PhaseQNodeOperation, ...], circuit.operations
        ),
        operation_theta=lambda operation, tensor: _jax_operation_theta(operation, tensor),
        gate_operator=lambda gate, theta: _jax_gate_matrix(jnp, gate, theta),
        apply_operator=lambda namespace, state, n_qubits, qubits, matrix: _jax_apply_gate_matrix(
            namespace, state, n_qubits, qubits, matrix
        ),
        pauli_operator=lambda label: _jax_pauli_matrix(jnp, label),
        dense_operator=lambda matrix: jnp.asarray(matrix, dtype=jnp.complex128),
        real_zero=lambda: jnp.asarray(0.0),
        other_observable=lambda state, n_qubits, observable: _jax_term_expectation(
            jnp, state, n_qubits, cast(PauliTerm, observable)
        ),
        dense_class=lambda: DenseHermitianObservable,
        covariance_class=lambda: PauliCovarianceObservable,
        hamiltonian_class=lambda: SparsePauliHamiltonian,
        term_class=lambda: PauliTerm,
        observable_expectation=lambda state, n_qubits, observable, _dense: _jax_expectation_value(
            jnp, state, n_qubits, observable
        ),
        term_expectation=lambda state, n_qubits, term: _jax_term_expectation(
            jnp, state, n_qubits, term
        ),
        symmetrized_product_expectation=lambda state, n_qubits, left, right: (
            _jax_symmetrized_product_expectation(jnp, state, n_qubits, left, right)
        ),
        term_product_expectation=lambda state, n_qubits, left, right: (
            _jax_term_product_expectation(jnp, state, n_qubits, left, right)
        ),
        apply_term_operator=lambda state, n_qubits, term: _jax_apply_term_operator(
            jnp, state, n_qubits, term
        ),
    )


def _jax_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    return _native_jax_operation_theta(operation, parameter_tensor)


def _jax_gate_matrix(jnp: Any, gate: str, theta: Any) -> Any:
    complex_dtype = jnp.complex128
    one = jnp.asarray(1.0, dtype=complex_dtype)
    zero = jnp.asarray(0.0, dtype=complex_dtype)
    imag = jnp.asarray(1.0j, dtype=complex_dtype)
    identity = jnp.eye(2, dtype=complex_dtype)
    x_matrix = jnp.asarray(_pauli_operator_entries("x"), dtype=complex_dtype)
    y_matrix = jnp.asarray(_pauli_operator_entries("y"), dtype=complex_dtype)
    z_matrix = jnp.asarray(_pauli_operator_entries("z"), dtype=complex_dtype)
    h_matrix = _hadamard_operator(
        jnp,
        real_tensor=lambda values: jnp.asarray(values),
        complex_tensor=lambda values: jnp.asarray(values, dtype=complex_dtype),
    )
    s_matrix = jnp.asarray(_phase_operator_entries(1.0j), dtype=complex_dtype)
    t_matrix = jnp.asarray(
        _phase_operator_entries(_quarter_turn_phase(lambda: np)), dtype=complex_dtype
    )
    sx_matrix = _sqrt_x_operator(
        complex_tensor=lambda values: jnp.asarray(values, dtype=complex_dtype)
    )

    return _dispatch_gate_operator(
        gate,
        theta,
        _GateOperatorFactories(
            identity=lambda: identity,
            x=lambda: x_matrix,
            y=lambda: y_matrix,
            z=lambda: z_matrix,
            h=lambda: h_matrix,
            s=lambda: s_matrix,
            t=lambda: t_matrix,
            sx=lambda: sx_matrix,
            rotation=lambda angle, identity_factory, operator: _pauli_rotation(
                jnp, angle, imag, identity=identity_factory, operator=operator
            ),
            phase=lambda angle: _jax_phase_operator(
                jnp, angle, one, zero, imag, dtype=complex_dtype
            ),
            two_qubit=lambda name: jnp.asarray(
                _two_qubit_operator_entries(name), dtype=complex_dtype
            ),
            controlled_z=lambda controls: _controlled_z_operator(
                jnp,
                controls,
                complex_tensor=lambda values: jnp.asarray(values, dtype=complex_dtype),
            ),
            controlled=lambda: partial(_jax_controlled, jnp),
            ccnot=lambda: _jax_ccnot_matrix(jnp),
            cswap=lambda: _jax_cswap_matrix(jnp),
            two_qubit_identity=lambda: jnp.eye(4, dtype=complex_dtype),
            kron=lambda: jnp.kron,
            gate=lambda name, angle: _jax_gate_matrix(jnp, name, angle),
            unsupported=lambda name: {
                "rzz": _pauli_rotation(
                    jnp,
                    theta,
                    imag,
                    identity=lambda: jnp.eye(4, dtype=complex_dtype),
                    operator=lambda: jnp.kron(z_matrix, z_matrix),
                )
            }[name],
        ),
    )


def _jax_controlled(jnp: Any, target: Any) -> Any:
    return _controlled_operator(jnp, target)


def _jax_ccnot_matrix(jnp: Any) -> Any:
    return _basis_transposition(jnp, 6, 7)


def _jax_cswap_matrix(jnp: Any) -> Any:
    return _basis_transposition(jnp, 5, 6)


def _jax_apply_gate_matrix(
    jnp: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
) -> Any:
    width = len(qubits)
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = np.argsort(axes)
    updated = _jax_binary_axis_matmul(
        jnp, state, matrix, dimensions=n_qubits, width=width, axes=axes
    )
    reshape = jnp.reshape
    permute = jnp.transpose
    return _restore_binary_axis_order(
        updated, tuple(int(axis) for axis in inverse), reshape=reshape, permute=permute
    )


def _jax_expectation_value(
    jnp: Any,
    state: Any,
    n_qubits: int,
    observable: object,
) -> Any:
    return _observable_expectation(_jax_statevector_routes(jnp), state, n_qubits, observable)


def _jax_term_expectation(jnp: Any, state: Any, n_qubits: int, term: PauliTerm) -> Any:
    return _term_expectation(_jax_statevector_routes(jnp), state, n_qubits, term)


def _jax_symmetrized_product_expectation(
    jnp: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    return _symmetrized_product_expectation(
        _jax_statevector_routes(jnp), state, n_qubits, left, right
    )


def _jax_term_product_expectation(
    jnp: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    return _term_product_expectation(_jax_statevector_routes(jnp), state, n_qubits, left, right)


def _jax_apply_term_operator(jnp: Any, state: Any, n_qubits: int, term: PauliTerm) -> Any:
    return _apply_term_operator(_jax_statevector_routes(jnp), state, n_qubits, term)


def _jax_pauli_matrix(jnp: Any, label: str) -> Any:
    if label == "x":
        return jnp.asarray(_pauli_operator_entries("x"), dtype=jnp.complex128)
    if label == "y":
        return jnp.asarray(_pauli_operator_entries("y"), dtype=jnp.complex128)
    z_matrix = jnp.asarray(_pauli_operator_entries("z"), dtype=jnp.complex128)
    return {"z": z_matrix}[label]


__all__ = [
    "jax_phase_qnode_aot_export_audit",
    "jax_phase_qnode_native_transform_audit",
    "jax_phase_qnode_pytree_transform_audit",
    "jax_phase_qnode_sharding_transform_audit",
    "jax_phase_qnode_value_and_grad",
]
