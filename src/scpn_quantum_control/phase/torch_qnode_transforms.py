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

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{name} width must be {width}, got {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _torch_batch_to_numpy(values: object) -> FloatArray:
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return _as_parameter_matrix("values", candidate)


def _torch_matrix_to_numpy(name: str, values: object) -> FloatArray:
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return _as_parameter_matrix(name, candidate)


def _torch_scalar_to_float(values: object) -> float:
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    scalar = np.asarray(candidate, dtype=float)
    if scalar.shape not in ((), (1,)):
        raise ValueError(f"PyTorch scalar value must be scalar-like, got {scalar.shape}")
    value = float(scalar.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("PyTorch scalar value must be finite")
    return value


def _torch_func_transforms(torch_module: Any) -> tuple[Any, Any, Any]:
    torch_func = getattr(torch_module, "func", None)
    grad = getattr(torch_func, "grad", None)
    vmap = getattr(torch_func, "vmap", None)
    jacrev = getattr(torch_func, "jacrev", None)
    if torch_func is None or not callable(grad) or not callable(vmap) or not callable(jacrev):
        raise RuntimeError("PyTorch module does not expose torch.func.grad/vmap/jacrev")
    return grad, vmap, jacrev


def _torch_compile(torch_module: Any) -> Any:
    compile_fn = getattr(torch_module, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError("PyTorch module does not expose torch.compile")
    return compile_fn


def _torch_complex_tensor(torch_module: Any, values: object) -> Any:
    return torch_module.as_tensor(values, dtype=torch_module.complex128)


def _torch_real_tensor(torch_module: Any, values: object) -> Any:
    return torch_module.as_tensor(values, dtype=torch_module.float64)


def _torch_phase_qnode_value_and_state(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
) -> tuple[Any, Any]:
    parameter_tensor = torch_module.as_tensor(params, dtype=torch_module.float64)
    state = torch_module.zeros((2**circuit.n_qubits,), dtype=torch_module.complex128)
    state[0] = 1.0 + 0.0j
    for operation in cast(tuple[PhaseQNodeOperation, ...], circuit.operations):
        matrix = _torch_gate_matrix(
            torch_module,
            operation.gate,
            _torch_operation_theta(operation, parameter_tensor),
        )
        state = _torch_apply_gate_matrix(
            torch_module,
            state,
            circuit.n_qubits,
            operation.qubits,
            matrix,
        )
    return _torch_expectation_value(
        torch_module, state, circuit.n_qubits, circuit.observable
    ), state


def _torch_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    if operation.parameter_index is None:
        return parameter_tensor.new_tensor(0.0)
    return parameter_tensor[operation.parameter_index]


def _torch_gate_matrix(torch_module: Any, gate: str, theta: Any) -> Any:
    complex_dtype = torch_module.complex128

    def identity() -> Any:
        return torch_module.eye(2, dtype=complex_dtype)

    def x_matrix() -> Any:
        return _torch_complex_tensor(torch_module, [[0.0, 1.0], [1.0, 0.0]])

    def y_matrix() -> Any:
        return _torch_complex_tensor(torch_module, [[0.0, -1.0j], [1.0j, 0.0]])

    def z_matrix() -> Any:
        return _torch_complex_tensor(torch_module, [[1.0, 0.0], [0.0, -1.0]])

    def h_matrix() -> Any:
        return (1.0 / torch_module.sqrt(_torch_real_tensor(torch_module, 2.0))) * (
            _torch_complex_tensor(torch_module, [[1.0, 1.0], [1.0, -1.0]])
        )

    def s_matrix() -> Any:
        return _torch_complex_tensor(torch_module, [[1.0, 0.0], [0.0, 1.0j]])

    def t_matrix() -> Any:
        return _torch_complex_tensor(
            torch_module,
            [[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]],
        )

    def sx_matrix() -> Any:
        return 0.5 * _torch_complex_tensor(
            torch_module,
            [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
        )

    if gate == "h":
        return h_matrix()
    if gate == "x":
        return x_matrix()
    if gate == "y":
        return y_matrix()
    if gate == "z":
        return z_matrix()
    if gate == "s":
        return s_matrix()
    if gate == "t":
        return t_matrix()
    if gate == "sx":
        return sx_matrix()
    if gate == "rx":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return (
            torch_module.cos(theta / 2.0) * identity()
            - imag * torch_module.sin(theta / 2.0) * x_matrix()
        )
    if gate == "ry":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return (
            torch_module.cos(theta / 2.0) * identity()
            - imag * torch_module.sin(theta / 2.0) * y_matrix()
        )
    if gate == "rz":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return (
            torch_module.cos(theta / 2.0) * identity()
            - imag * torch_module.sin(theta / 2.0) * z_matrix()
        )
    if gate == "phase":
        one = _torch_complex_tensor(torch_module, 1.0 + 0.0j)
        zero = _torch_complex_tensor(torch_module, 0.0 + 0.0j)
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return torch_module.stack(
            (
                torch_module.stack((one, zero)),
                torch_module.stack((zero, torch_module.exp(imag * theta))),
            )
        )
    if gate == "cnot":
        return _torch_complex_tensor(
            torch_module,
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        )
    if gate == "cz":
        return torch_module.diag(_torch_complex_tensor(torch_module, [1.0, 1.0, 1.0, -1.0]))
    if gate == "cy":
        return _torch_complex_tensor(
            torch_module,
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1.0j], [0, 0, 1.0j, 0]],
        )
    if gate == "swap":
        return _torch_complex_tensor(
            torch_module,
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        )
    if gate == "ch":
        return _torch_controlled(torch_module, h_matrix())
    if gate == "cs":
        return _torch_controlled(torch_module, s_matrix())
    if gate == "ct":
        return _torch_controlled(torch_module, t_matrix())
    if gate == "ccnot":
        return _torch_ccnot_matrix(torch_module)
    if gate == "ccz":
        return torch_module.diag(
            _torch_complex_tensor(torch_module, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
        )
    if gate == "cswap":
        return _torch_cswap_matrix(torch_module)
    if gate == "crx":
        return _torch_controlled(torch_module, _torch_gate_matrix(torch_module, "rx", theta))
    if gate == "cry":
        return _torch_controlled(torch_module, _torch_gate_matrix(torch_module, "ry", theta))
    if gate == "crz":
        return _torch_controlled(torch_module, _torch_gate_matrix(torch_module, "rz", theta))
    if gate == "rxx":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return torch_module.cos(theta / 2.0) * torch_module.eye(4, dtype=complex_dtype) - imag * (
            torch_module.sin(theta / 2.0)
        ) * torch_module.kron(x_matrix(), x_matrix())
    if gate == "ryy":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return torch_module.cos(theta / 2.0) * torch_module.eye(4, dtype=complex_dtype) - imag * (
            torch_module.sin(theta / 2.0)
        ) * torch_module.kron(y_matrix(), y_matrix())
    if gate == "rzz":
        imag = _torch_complex_tensor(torch_module, 1.0j)
        return torch_module.cos(theta / 2.0) * torch_module.eye(4, dtype=complex_dtype) - imag * (
            torch_module.sin(theta / 2.0)
        ) * torch_module.kron(z_matrix(), z_matrix())
    raise ValueError(f"unsupported PyTorch Phase-QNode gate: {gate}")


def _torch_controlled(torch_module: Any, target: Any) -> Any:
    rows = [
        _torch_complex_tensor(torch_module, [1.0, 0.0, 0.0, 0.0]),
        _torch_complex_tensor(torch_module, [0.0, 1.0, 0.0, 0.0]),
        torch_module.cat(
            (
                _torch_complex_tensor(torch_module, [0.0, 0.0]),
                target[0],
            )
        ),
        torch_module.cat(
            (
                _torch_complex_tensor(torch_module, [0.0, 0.0]),
                target[1],
            )
        ),
    ]
    return torch_module.stack(tuple(rows))


def _torch_ccnot_matrix(torch_module: Any) -> Any:
    matrix = torch_module.eye(8, dtype=torch_module.complex128)
    matrix[6, 6] = 0.0
    matrix[7, 7] = 0.0
    matrix[6, 7] = 1.0
    matrix[7, 6] = 1.0
    return matrix


def _torch_cswap_matrix(torch_module: Any) -> Any:
    matrix = torch_module.eye(8, dtype=torch_module.complex128)
    matrix[5, 5] = 0.0
    matrix[6, 6] = 0.0
    matrix[5, 6] = 1.0
    matrix[6, 5] = 1.0
    return matrix


def _torch_apply_gate_matrix(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
) -> Any:
    width = len(qubits)
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = tuple(int(axis) for axis in np.argsort(axes))
    tensor = torch_module.permute(torch_module.reshape(state, (2,) * n_qubits), tuple(axes))
    front = torch_module.reshape(tensor, (2**width, -1))
    updated = torch_module.reshape(matrix @ front, (2,) * n_qubits)
    return torch_module.reshape(torch_module.permute(updated, inverse), (-1,))


def _torch_expectation_value(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    observable: object,
) -> Any:
    if isinstance(observable, DenseHermitianObservable):
        matrix = _torch_complex_tensor(torch_module, observable.matrix)
        return torch_module.real(torch_module.vdot(state, matrix @ state))
    if isinstance(observable, PauliCovarianceObservable):
        symmetrized = _torch_symmetrized_product_expectation(
            torch_module, state, n_qubits, observable.left, observable.right
        )
        left_mean = _torch_term_expectation(torch_module, state, n_qubits, observable.left)
        right_mean = _torch_term_expectation(torch_module, state, n_qubits, observable.right)
        return symmetrized - left_mean * right_mean
    if isinstance(observable, SparsePauliHamiltonian):
        total = _torch_real_tensor(torch_module, 0.0)
        for term in observable.terms:
            total = total + _torch_term_expectation(torch_module, state, n_qubits, term)
        return total
    if isinstance(observable, PauliTerm):
        return _torch_term_expectation(torch_module, state, n_qubits, observable)
    raise ValueError(f"unsupported PyTorch Phase-QNode observable: {observable}")


def _torch_term_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    term: PauliTerm,
) -> Any:
    transformed = _torch_apply_term_operator(torch_module, state, n_qubits, term)
    return term.coefficient * torch_module.real(torch_module.vdot(state, transformed))


def _torch_symmetrized_product_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    left_right = _torch_term_product_expectation(torch_module, state, n_qubits, left, right)
    right_left = _torch_term_product_expectation(torch_module, state, n_qubits, right, left)
    return torch_module.real(0.5 * (left_right + right_left))


def _torch_term_product_expectation(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    transformed = _torch_apply_term_operator(torch_module, state, n_qubits, right)
    transformed = _torch_apply_term_operator(torch_module, transformed, n_qubits, left)
    return left.coefficient * right.coefficient * torch_module.vdot(state, transformed)


def _torch_apply_term_operator(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    term: PauliTerm,
) -> Any:
    transformed = state
    for qubit, label in term.factors:
        transformed = _torch_apply_gate_matrix(
            torch_module,
            transformed,
            n_qubits,
            (qubit,),
            _torch_pauli_matrix(torch_module, label),
        )
    return transformed


def _torch_pauli_matrix(torch_module: Any, label: str) -> Any:
    if label == "x":
        return _torch_complex_tensor(torch_module, [[0.0, 1.0], [1.0, 0.0]])
    if label == "y":
        return _torch_complex_tensor(torch_module, [[0.0, -1.0j], [1.0j, 0.0]])
    if label == "z":
        return _torch_complex_tensor(torch_module, [[1.0, 0.0], [0.0, -1.0]])
    raise ValueError(f"unsupported PyTorch Pauli label: {label}")


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
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)
    trainable_params = _torch_trainable_tensor(torch_module, parameter_values)
    value, state_obj = _torch_phase_qnode_value_and_state(torch_module, circuit, trainable_params)
    value.backward()
    gradient_obj = trainable_params.grad
    if gradient_obj is None:
        raise RuntimeError("PyTorch Phase-QNode autograd did not produce a gradient")
    value_scalar = _torch_scalar_to_float(value)
    gradient = _as_parameter_vector(
        "PyTorch Phase-QNode gradient",
        _torch_values_to_numpy(gradient_obj),
        width=parameter_values.size,
    )
    state = np.asarray(state_obj.detach().cpu().numpy(), dtype=np.complex128)
    max_abs_error = float(np.max(np.abs(gradient - parameter_shift.gradient), initial=0.0))
    l2_error = float(np.linalg.norm(gradient - parameter_shift.gradient))
    passed = bool(
        abs(value_scalar - parameter_shift.value) <= tolerance_value
        and max_abs_error <= tolerance_value
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
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_boundary=False,
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
    torch_func_grad, torch_func_vmap, torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        _torch_matrix_to_numpy("params_batch", params_batch),
        width=parameter_values.size,
    )
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    for row in parameter_batch:
        row_report = phase_qnode_support_report(circuit, row)
        if not row_report.supported:
            raise PhaseQNodeSupportError(row_report)

    def value_function(parameter_tensor: Any) -> Any:
        value, _state = _torch_phase_qnode_value_and_state(
            torch_module,
            circuit,
            parameter_tensor,
        )
        return value

    grad_fn = torch_func_grad(value_function)
    vmap_grad_fn = torch_func_vmap(grad_fn)
    jacrev_fn = torch_func_jacrev(value_function)
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_params_batch = _torch_tensor(torch_module, parameter_batch)
    torch_value, _torch_state = _torch_phase_qnode_value_and_state(
        torch_module,
        circuit,
        torch_params,
    )
    torch_gradient = grad_fn(torch_params)
    torch_jacrev_gradient = jacrev_fn(torch_params)
    torch_vmap_gradients = vmap_grad_fn(torch_params_batch)
    value = _torch_scalar_to_float(torch_value)
    gradient = _as_parameter_vector(
        "PyTorch Phase-QNode torch.func gradient",
        _torch_values_to_numpy(torch_gradient),
        width=parameter_values.size,
    )
    jacrev_gradient = _as_parameter_vector(
        "PyTorch Phase-QNode torch.func jacrev gradient",
        _torch_values_to_numpy(torch_jacrev_gradient),
        width=parameter_values.size,
    )
    vmap_gradients = _as_parameter_matrix(
        "PyTorch Phase-QNode torch.func vmap gradients",
        _torch_batch_to_numpy(torch_vmap_gradients),
        width=parameter_values.size,
    )
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)
    parameter_shift_batch_gradients = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in parameter_batch]
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
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        func_grad_supported=True,
        func_vmap_supported=True,
        func_jacrev_supported=True,
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
    """
    torch_module = _torch_loader()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)

    def value_function(parameter_tensor: Any) -> Any:
        value, _state = _torch_phase_qnode_value_and_state(
            torch_module,
            circuit,
            parameter_tensor,
        )
        return value

    grad_fn = torch_func_grad(value_function)
    compiled_value_fn = compile_fn(
        value_function,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )
    compiled_grad_fn = compile_fn(
        grad_fn,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_value = compiled_value_fn(torch_params)
    torch_gradient = compiled_grad_fn(torch_params)
    value = _torch_scalar_to_float(torch_value)
    gradient = _as_parameter_vector(
        "PyTorch Phase-QNode torch.compile gradient",
        _torch_values_to_numpy(torch_gradient),
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
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        torch_compile_supported=True,
        compiled_value_supported=True,
        compiled_gradient_supported=True,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )


def _compile_boundary_exception_reason(exc: Exception) -> str:
    detail = str(exc).splitlines()[0] if str(exc).splitlines() else repr(exc)
    return f"{type(exc).__name__}: {detail[:240]}"


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
    try:
        result = torch_phase_qnode_compile_audit(
            circuit,
            parameter_values,
            tolerance=tolerance,
            fullgraph=fullgraph,
            dynamic=dynamic,
            _torch_loader=_torch_loader,
        )
    except Exception as exc:
        reason = f"{name} execution is blocked: {_compile_boundary_exception_reason(exc)}"
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


def _torch_aot_autograd_boundary_route(
    torch_module: Any,
    *,
    fullgraph_route: PhaseTorchCompileBoundaryRoute,
) -> PhaseTorchCompileBoundaryRoute:
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
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)
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
    non_fullgraph_route, non_fullgraph_result = _torch_compile_boundary_execution_route(
        name="non_fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance_value,
        fullgraph=False,
        dynamic=False,
        requires=(),
        passed_reason=(
            "non-fullgraph torch.compile value and gradient execution matches the "
            "SCPN parameter-shift reference on the registered CPU statevector route"
        ),
        blocked_reason_after_pass=None,
        _torch_loader=_torch_loader,
    )
    dynamic_route, _dynamic_result = _torch_compile_boundary_execution_route(
        name="dynamic_non_fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance_value,
        fullgraph=False,
        dynamic=True,
        requires=(
            "variable_shape_compile_artifact",
            "dynamic_shape_guard_report",
        ),
        passed_reason="",
        blocked_reason_after_pass=dynamic_compile_blocker,
        _torch_loader=_torch_loader,
    )
    fullgraph_route, _fullgraph_result = _torch_compile_boundary_execution_route(
        name="fullgraph_compile",
        circuit=circuit,
        parameter_values=parameter_values,
        tolerance=tolerance_value,
        fullgraph=True,
        dynamic=False,
        requires=(
            "graph_break_free_fullgraph_artifact",
            "compiled_frame_evidence",
            "static_shape_symbolic_integer_guards",
        ),
        passed_reason="",
        blocked_reason_after_pass=fullgraph_compile_blocker,
        _torch_loader=_torch_loader,
    )
    aot_route = _torch_aot_autograd_boundary_route(torch_module, fullgraph_route=fullgraph_route)
    routes = (
        non_fullgraph_route,
        dynamic_route,
        fullgraph_route,
        aot_route,
    )
    if non_fullgraph_result is None:
        non_fullgraph_value = parameter_shift.value
        non_fullgraph_gradient = parameter_shift.gradient.copy()
        max_abs_reference_error = tolerance_value + 1.0
    else:
        non_fullgraph_value = non_fullgraph_result.value
        non_fullgraph_gradient = non_fullgraph_result.gradient.copy()
        max_abs_reference_error = non_fullgraph_result.max_abs_error
    passed = bool(
        non_fullgraph_route.status == "passed"
        and dynamic_route.status == "blocked"
        and fullgraph_route.status == "blocked"
        and aot_route.status == "blocked"
    )
    return PhaseTorchCompileBoundaryAuditResult(
        routes=routes,
        non_fullgraph_value=non_fullgraph_value,
        non_fullgraph_gradient=non_fullgraph_gradient,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        max_abs_reference_error=max_abs_reference_error,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        passed=passed,
        persistent_export_claim=False,
        provider_claim=False,
        performance_claim=False,
    )


__all__ = [
    "torch_phase_qnode_compile_audit",
    "torch_phase_qnode_compile_boundary_audit",
    "torch_phase_qnode_transform_audit",
    "torch_phase_qnode_value_and_grad",
]
