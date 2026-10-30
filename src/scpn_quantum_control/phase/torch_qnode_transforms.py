# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Registered-QNode Transforms
"""Native Torch execution and compiler diagnostics for registered Phase-QNodes.

This one-way leaf owns deterministic statevector, ``torch.func`` transform,
``torch.compile``, and compiler-boundary routes. The public facade injects its
active optional-Torch loader and retains later compatibility and maturity work.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from functools import partial
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable_axis_compilation import numpy_axis_order as _numpy_axis_order
from ..differentiable_axis_compilation import (
    register_torch_axis_compilation as _register_torch_axis_compilation,
)
from ..differentiable_tensor_materialization import (
    TensorMaterializationPrimitives as _TensorMaterializationPrimitives,
)
from ..differentiable_tensor_materialization import (
    materialize_torch_host_values as _materialize_torch_host_values,
)
from ..differentiable_torch_compile_specialisation import (
    torch_static_integer_specialisation as _torch_static_integer_specialisation,
)
from ..phase_qnode_audit_admission import (
    admitted_parameter_matrix as _admitted_parameter_matrix,
)
from ..phase_qnode_gate_dispatch import (
    GateOperatorFactories as _GateOperatorFactories,
)
from ..phase_qnode_gate_dispatch import (
    dispatch_gate_operator as _dispatch_gate_operator,
)
from ..phase_qnode_gate_dispatch import (
    refuse_unregistered_gate as _refuse_unregistered_gate,
)
from ..phase_qnode_gate_numerics import (
    controlled_z_operator as _controlled_z_operator,
)
from ..phase_qnode_gate_numerics import (
    hadamard_operator as _hadamard_operator,
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
from ..phase_qnode_gate_numerics import (
    torch_phase_operator as _torch_phase_operator,
)
from ..phase_qnode_native_operators import (
    pauli_operator_entries as _pauli_operator_entries,
)
from ..phase_qnode_native_operators import (
    phase_operator_entries as _phase_operator_entries,
)
from ..phase_qnode_native_operators import (
    torch_basis_transposition as _basis_transposition,
)
from ..phase_qnode_native_operators import (
    torch_controlled_operator_rows as _controlled_operator_rows,
)
from ..phase_qnode_native_operators import (
    two_qubit_operator_entries as _two_qubit_operator_entries,
)
from ..phase_qnode_native_statevector import (
    torch_apply_local_operator as _torch_apply_local_operator,
)
from ..phase_qnode_native_statevector import (
    torch_operation_theta as _native_torch_operation_theta,
)
from ..phase_qnode_state_evolution import torch_zero_state as _torch_zero_state
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
    refuse_unregistered_observable as _refuse_unregistered_observable,
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
from ..phase_qnode_torch_audits import (
    TorchAuditOperations as _TorchAuditOperations,
)
from ..phase_qnode_torch_audits import (
    compile_boundary_exception_reason as _lower_compile_boundary_exception_reason,
)
from ..phase_qnode_torch_audits import (
    torch_aot_autograd_boundary_route as _lower_torch_aot_autograd_boundary_route,
)
from ..phase_qnode_torch_audits import (
    torch_compile_audit as _torch_compile_audit,
)
from ..phase_qnode_torch_audits import (
    torch_compile_boundary_audit as _torch_compile_boundary_audit,
)
from ..phase_qnode_torch_audits import (
    torch_compile_boundary_route as _torch_compile_boundary_route,
)
from ..phase_qnode_torch_audits import (
    torch_compile_dense_matrix as _lower_torch_compile_dense_matrix,
)
from ..phase_qnode_torch_audits import (
    torch_compile_function as _torch_compile_function,
)
from ..phase_qnode_torch_audits import (
    torch_func_transform_audit as _torch_func_transform_audit,
)
from ..phase_qnode_torch_audits import (
    torch_func_transforms as _lower_torch_func_transforms,
)
from ..phase_qnode_torch_audits import (
    torch_scalar_to_float as _lower_torch_scalar_to_float,
)
from ..phase_qnode_torch_audits import (
    torch_statevector_gradient_audit as _torch_statevector_gradient_audit,
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
from .torch_bridge_contracts import (
    PhaseTorchCompileBoundaryAuditResult,
    PhaseTorchCompileBoundaryRoute,
    PhaseTorchPhaseQNodeCompileResult,
    PhaseTorchPhaseQNodeStatevectorResult,
    PhaseTorchPhaseQNodeTransformResult,
)
from .torch_gradients import (
    TorchLoader,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_torch,
    _torch_tensor,
    _torch_trainable_tensor,
    _torch_values_to_numpy,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _as_parameter_matrix(name: str, values: object, *, width: int | None = None) -> FloatArray:
    return _admitted_parameter_matrix(name, values, width=width)


def _torch_to_numpy_compatible(values: object) -> object:
    candidate = _materialize_torch_host_values(
        values,
        primitives=_TensorMaterializationPrimitives(
            attribute=lambda: getattr, predicate=lambda: callable
        ),
    )
    return candidate


def _torch_batch_to_numpy(values: object) -> FloatArray:
    return _as_parameter_matrix("values", _torch_to_numpy_compatible(values))


def _torch_matrix_to_numpy(name: str, values: object) -> FloatArray:
    return _as_parameter_matrix(name, _torch_to_numpy_compatible(values))


def _torch_scalar_to_float(values: object) -> float:
    return _lower_torch_scalar_to_float(values, host_values=_torch_to_numpy_compatible)


def _torch_func_transforms(torch_module: Any) -> tuple[Any, Any, Any]:
    return _lower_torch_func_transforms(torch_module)


def _torch_compile(torch_module: Any) -> Any:
    return _torch_compile_function(torch_module, register=lambda: _register_torch_axis_compilation)


def _torch_complex_tensor(torch_module: Any, values: object) -> Any:
    return torch_module.as_tensor(values, dtype=torch_module.complex128)


def _torch_real_tensor(torch_module: Any, values: object) -> Any:
    return torch_module.as_tensor(values, dtype=torch_module.float64)


def _torch_phase_qnode_value_and_state(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
) -> tuple[Any, Any]:
    return _statevector_value_and_state(_torch_statevector_routes(torch_module), circuit, params)


def _torch_statevector_routes(torch_module: Any) -> _StatevectorKernels:
    """Return engine kernels whose routes are this module's named functions."""
    return _StatevectorKernels(
        namespace=torch_module,
        parameter_tensor=lambda params: torch_module.as_tensor(params, dtype=torch_module.float64),
        zero_state=lambda namespace, width: _torch_zero_state(namespace, width),
        circuit_operations=lambda circuit: cast(
            tuple[PhaseQNodeOperation, ...], circuit.operations
        ),
        operation_theta=lambda operation, tensor: _torch_operation_theta(operation, tensor),
        gate_operator=lambda gate, theta: _torch_gate_matrix(torch_module, gate, theta),
        apply_operator=lambda namespace, state, n_qubits, qubits, matrix: _torch_apply_gate_matrix(
            namespace, state, n_qubits, qubits, matrix
        ),
        pauli_operator=lambda label: _torch_pauli_matrix(torch_module, label),
        dense_operator=lambda matrix: _torch_complex_tensor(torch_module, matrix),
        real_zero=lambda: _torch_real_tensor(torch_module, 0.0),
        other_observable=_refuse_unregistered_observable(
            "unsupported PyTorch Phase-QNode observable"
        ),
        dense_class=lambda: DenseHermitianObservable,
        covariance_class=lambda: PauliCovarianceObservable,
        hamiltonian_class=lambda: SparsePauliHamiltonian,
        term_class=lambda: PauliTerm,
        observable_expectation=lambda state, n_qubits, observable, dense: (
            _torch_expectation_value(torch_module, state, n_qubits, observable)
            if dense is None
            else _observable_expectation(
                _torch_statevector_routes(torch_module), state, n_qubits, observable, dense
            )
        ),
        term_expectation=lambda state, n_qubits, term: _torch_term_expectation(
            torch_module, state, n_qubits, term
        ),
        symmetrized_product_expectation=lambda state, n_qubits, left, right: (
            _torch_symmetrized_product_expectation(torch_module, state, n_qubits, left, right)
        ),
        term_product_expectation=lambda state, n_qubits, left, right: (
            _torch_term_product_expectation(torch_module, state, n_qubits, left, right)
        ),
        apply_term_operator=lambda state, n_qubits, term: _torch_apply_term_operator(
            torch_module, state, n_qubits, term
        ),
    )


def _torch_phase_qnode_scalar_value(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    parameter_tensor: Any,
    dense_matrix: Any = None,
) -> Any:
    if dense_matrix is None:
        value, _state = _torch_phase_qnode_value_and_state(torch_module, circuit, parameter_tensor)
        return value
    value, _state = _statevector_value_and_state(
        _torch_statevector_routes(torch_module), circuit, parameter_tensor, dense_matrix
    )
    return value


_DYNAMIC_AUDIT_QUALIFIED_TORCH: frozenset[tuple[int, int]] = frozenset({(2, 13), (2, 15)})
"""PyTorch minor releases whose scoped integer specialisation was qualified."""


def _torch_static_circuit_integers(
    torch_module: Any, dynamic: bool
) -> AbstractContextManager[object]:
    """Keep circuit-structure integers static while dynamic shapes are traced.

    With ``dynamic=True`` Dynamo treats integers read from object attributes,
    such as ``circuit.n_qubits`` and operation wires, as symbolic. The circuit
    structure is fixed for one audit, so symbolic wires only make tracing
    fail; that failed attempt marks shared helper frames to be skipped for the
    rest of the process, which later breaks fullgraph ``torch.func.grad`` in
    other public routes. PyTorch 2.13 and 2.15 expose the integer specialisation only
    through ``torch._dynamo.config``; tensors stay dynamic. The patch restores
    the caller's value on exit. A runtime without that setting is refused
    before anything is compiled, so the damaging attempt never starts.

    Raises
    ------
    RuntimeError
        If ``dynamic`` is requested on a PyTorch release outside the qualified
        set, or the runtime cannot keep circuit integers static.

    """
    return _torch_static_integer_specialisation(
        torch_module,
        dynamic,
        purpose="dynamic Phase-QNode compile audit",
        qualified_releases=_DYNAMIC_AUDIT_QUALIFIED_TORCH,
    )


def _torch_compile_dense_matrix(torch_module: Any, circuit: PhaseQNodeCircuit) -> Any:
    """Convert a dense observable before compilation, or return ``None``.

    Dynamo turns a NumPy constant inside a compiled ``torch.func.grad`` into a
    ``from_numpy`` guard whose dispatch keys differ under the functorch
    transform, so PyTorch 2.13 rejects the frame it has just compiled. A tensor
    converted outside the compiled region carries the same values without that
    guard.
    """
    return _lower_torch_compile_dense_matrix(
        torch_module, circuit, complex_tensor=_torch_complex_tensor
    )


def _torch_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    return _native_torch_operation_theta(operation, parameter_tensor)


def _torch_gate_matrix(torch_module: Any, gate: str, theta: Any) -> Any:
    complex_dtype = torch_module.complex128

    def identity() -> Any:
        return torch_module.eye(2, dtype=complex_dtype)

    def x_matrix() -> Any:
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("x"))

    def y_matrix() -> Any:
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("y"))

    def z_matrix() -> Any:
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("z"))

    def h_matrix() -> Any:
        return _hadamard_operator(
            torch_module,
            real_tensor=lambda values: _torch_real_tensor(torch_module, values),
            complex_tensor=lambda values: _torch_complex_tensor(torch_module, values),
        )

    def s_matrix() -> Any:
        return _torch_complex_tensor(torch_module, _phase_operator_entries(1.0j))

    def t_matrix() -> Any:
        return _torch_complex_tensor(
            torch_module, _phase_operator_entries(_quarter_turn_phase(lambda: np))
        )

    def sx_matrix() -> Any:
        return _sqrt_x_operator(
            complex_tensor=lambda values: _torch_complex_tensor(torch_module, values)
        )

    return _dispatch_gate_operator(
        gate,
        theta,
        _GateOperatorFactories(
            identity=identity,
            x=x_matrix,
            y=y_matrix,
            z=z_matrix,
            h=h_matrix,
            s=s_matrix,
            t=t_matrix,
            sx=sx_matrix,
            rotation=lambda angle, identity_factory, operator: _pauli_rotation(
                torch_module,
                angle,
                _torch_complex_tensor(torch_module, 1.0j),
                identity=identity_factory,
                operator=operator,
            ),
            phase=lambda angle: _torch_phase_operator(
                torch_module,
                angle,
                _torch_complex_tensor(torch_module, 1.0 + 0.0j),
                _torch_complex_tensor(torch_module, 0.0 + 0.0j),
                _torch_complex_tensor(torch_module, 1.0j),
            ),
            two_qubit=lambda name: _torch_complex_tensor(
                torch_module, _two_qubit_operator_entries(name)
            ),
            controlled_z=lambda controls: _controlled_z_operator(
                torch_module,
                controls,
                complex_tensor=lambda values: _torch_complex_tensor(torch_module, values),
            ),
            controlled=lambda: partial(_torch_controlled, torch_module),
            ccnot=lambda: _torch_ccnot_matrix(torch_module),
            cswap=lambda: _torch_cswap_matrix(torch_module),
            two_qubit_identity=lambda: torch_module.eye(4, dtype=complex_dtype),
            kron=lambda: torch_module.kron,
            gate=lambda name, angle: _torch_gate_matrix(torch_module, name, angle),
            unsupported=_refuse_unregistered_gate("unsupported PyTorch Phase-QNode gate"),
        ),
    )


def _torch_controlled(torch_module: Any, target: Any) -> Any:
    rows = _controlled_operator_rows(
        torch_module,
        target,
        complex_tensor=lambda values: _torch_complex_tensor(torch_module, values),
    )
    return torch_module.stack(tuple(rows))


def _torch_ccnot_matrix(torch_module: Any) -> Any:
    return _basis_transposition(torch_module, 6, 7)


def _torch_cswap_matrix(torch_module: Any) -> Any:
    return _basis_transposition(torch_module, 5, 6)


def _torch_apply_gate_matrix(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
) -> Any:
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = tuple(int(axis) for axis in _numpy_axis_order(np.argsort, axes))
    return _torch_apply_local_operator(
        torch_module, state, n_qubits, qubits, matrix, inverse_order=lambda _axes: inverse
    )


def _torch_expectation_value(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    observable: object,
) -> Any:
    return _observable_expectation(
        _torch_statevector_routes(torch_module), state, n_qubits, observable
    )


def _torch_term_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    term: PauliTerm,
) -> Any:
    return _term_expectation(_torch_statevector_routes(torch_module), state, n_qubits, term)


def _torch_symmetrized_product_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    return _symmetrized_product_expectation(
        _torch_statevector_routes(torch_module), state, n_qubits, left, right
    )


def _torch_term_product_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    return _term_product_expectation(
        _torch_statevector_routes(torch_module), state, n_qubits, left, right
    )


def _torch_apply_term_operator(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    term: PauliTerm,
) -> Any:
    return _apply_term_operator(_torch_statevector_routes(torch_module), state, n_qubits, term)


def _torch_pauli_matrix(torch_module: Any, label: str) -> Any:
    if label == "x":
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("x"))
    if label == "y":
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("y"))
    if label == "z":
        return _torch_complex_tensor(torch_module, _pauli_operator_entries("z"))
    raise ValueError(f"unsupported PyTorch Pauli label: {label}")


def _torch_audit_routes() -> _TorchAuditOperations:
    """Return audit operation resolvers reading this module's live names."""
    return _TorchAuditOperations(
        support_report=lambda: phase_qnode_support_report,
        parameter_shift=lambda: parameter_shift_phase_qnode_gradient,
        support_error=lambda: PhaseQNodeSupportError,
        value_and_state=lambda: _torch_phase_qnode_value_and_state,
        scalar_value=lambda: _torch_phase_qnode_scalar_value,
        compile_dense_matrix=lambda: _torch_compile_dense_matrix,
        static_integers=lambda: _torch_static_circuit_integers,
        tensor=lambda: _torch_tensor,
        trainable_tensor=lambda: _torch_trainable_tensor,
        scalar_to_float=lambda: _torch_scalar_to_float,
        values_to_numpy=lambda: _torch_values_to_numpy,
        batch_to_numpy=lambda: _torch_batch_to_numpy,
        parameter_vector=lambda: _as_parameter_vector,
        parameter_matrix=lambda: _as_parameter_matrix,
    )


def torch_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into PyTorch.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It excludes finite-shot sampling, provider callbacks,
    hardware execution, density/noise channels, dynamic circuits, and
    promotion-grade performance evidence.
    """
    torch_module = _torch_loader()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    return _torch_statevector_gradient_audit(
        torch_module, circuit, parameter_values, tolerance_value, _torch_audit_routes()
    )


def torch_phase_qnode_transform_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchPhaseQNodeTransformResult:
    """Audit registered Phase-QNode execution through ``torch.func`` transforms.

    The accepted surface is deterministic local statevector execution for the
    registered ``PhaseQNodeCircuit`` gate and observable family. The audit
    verifies ``torch.func.grad``, ``torch.func.jacrev``, and ``torch.func.vmap``
    against SCPN parameter-shift references. It does not promote finite-shot,
    provider, hardware, CUDA, ``torch.compile``, or performance claims.
    """
    torch_module = _torch_loader()
    transforms = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        _torch_matrix_to_numpy("params_batch", params_batch),
        width=parameter_values.size,
    )
    return _torch_func_transform_audit(
        torch_module,
        transforms,
        circuit,
        parameter_values,
        parameter_batch,
        tolerance_value,
        _torch_audit_routes(),
    )


def torch_phase_qnode_compile_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
    fullgraph: bool = False,
    dynamic: bool = False,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchPhaseQNodeCompileResult:
    """Audit registered Phase-QNode execution through ``torch.compile``.

    The audit compiles the deterministic local statevector value function and
    the corresponding ``torch.func.grad`` gradient function for the registered
    ``PhaseQNodeCircuit`` gate and observable family. It excludes finite-shot,
    provider, hardware, CUDA promotion, and performance claims.

    Each distinct circuit is a separate compiler specialisation of the same
    functions, so PyTorch's per-function recompilation limit applies across
    all circuits audited in one process; with ``fullgraph=True`` exceeding it
    raises rather than falling back to eager execution.

    ``dynamic=True`` keeps circuit-structure integers static and is admitted
    only on qualified PyTorch releases (currently 2.13). On other releases it
    raises ``RuntimeError`` before anything is compiled; ``dynamic=False`` is
    unaffected.
    """
    torch_module = _torch_loader()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    return _torch_compile_audit(
        torch_module,
        compile_fn,
        torch_func_grad,
        circuit,
        parameter_values,
        tolerance_value,
        _torch_audit_routes(),
        fullgraph=fullgraph,
        dynamic=dynamic,
    )


def _compile_boundary_exception_reason(exc: Exception) -> str:
    return _lower_compile_boundary_exception_reason(exc)


def _torch_compile_boundary_execution_route(
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
    _torch_loader: TorchLoader = _load_torch,
) -> tuple[PhaseTorchCompileBoundaryRoute, PhaseTorchPhaseQNodeCompileResult | None]:
    return _torch_compile_boundary_route(
        name=name,
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance,
        fullgraph=fullgraph,
        dynamic=dynamic,
        requires=requires,
        passed_reason=passed_reason,
        blocked_reason_after_pass=blocked_reason_after_pass,
        compile_audit=lambda: partial(
            torch_phase_qnode_compile_audit, _torch_loader=_torch_loader
        ),
        exception_reason=lambda: _compile_boundary_exception_reason,
    )


def _torch_aot_autograd_boundary_route(
    torch_module: Any,
    *,
    fullgraph_route: PhaseTorchCompileBoundaryRoute,
) -> PhaseTorchCompileBoundaryRoute:
    return _lower_torch_aot_autograd_boundary_route(torch_module, fullgraph_route=fullgraph_route)


def torch_phase_qnode_compile_boundary_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchCompileBoundaryAuditResult:
    """Classify PyTorch compiler boundaries for registered Phase-QNode lowering.

    The audit executes the deterministic local Phase-QNode ``torch.compile``
    route in non-fullgraph, dynamic, and fullgraph modes against SCPN
    parameter-shift references. It deliberately reports dynamic-shape,
    fullgraph, AOTAutograd/export, provider, hardware, CUDA, isolated benchmark,
    and performance surfaces as blocked until their promotion artefacts exist.
    """
    torch_module = _torch_loader()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    return _torch_compile_boundary_audit(
        torch_module,
        circuit,
        parameter_values,
        tolerance_value,
        _torch_audit_routes(),
        execute_route=lambda: partial(
            _torch_compile_boundary_execution_route, _torch_loader=_torch_loader
        ),
        aot_route=lambda: _torch_aot_autograd_boundary_route,
    )


__all__ = [
    "torch_phase_qnode_compile_audit",
    "torch_phase_qnode_compile_boundary_audit",
    "torch_phase_qnode_transform_audit",
    "torch_phase_qnode_value_and_grad",
]
