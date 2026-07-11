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
:mod:`.torch_gradients`. This module retains registered-QNode, compatibility,
module, training, cloud, and maturity routes while those responsibilities
undergo bounded decomposition.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

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
    torch_module = _load_torch()
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
) -> PhaseTorchPhaseQNodeTransformResult:
    """Audit registered Phase-QNode execution through ``torch.func`` transforms.

    The accepted surface is deterministic local statevector execution for the
    registered ``PhaseQNodeCircuit`` gate and observable family. The audit
    verifies ``torch.func.grad``, ``torch.func.jacrev``, and ``torch.func.vmap``
    against SCPN parameter-shift references. It does not promote finite-shot,
    provider, hardware, CUDA, ``torch.compile``, or performance claims.
    """
    torch_module = _load_torch()
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
) -> PhaseTorchPhaseQNodeCompileResult:
    """Audit registered Phase-QNode execution through ``torch.compile``.

    The audit compiles the deterministic local statevector value function and
    the corresponding ``torch.func.grad`` gradient function for the registered
    ``PhaseQNodeCircuit`` gate and observable family. It excludes finite-shot,
    provider, hardware, CUDA promotion, and performance claims.
    """
    torch_module = _load_torch()
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
) -> tuple[PhaseTorchCompileBoundaryRoute, PhaseTorchPhaseQNodeCompileResult | None]:
    try:
        result = torch_phase_qnode_compile_audit(
            circuit,
            parameter_values,
            tolerance=tolerance,
            fullgraph=fullgraph,
            dynamic=dynamic,
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
) -> PhaseTorchCompileBoundaryAuditResult:
    """Classify PyTorch compiler boundaries for registered Phase-QNode lowering.

    The audit executes the deterministic local Phase-QNode ``torch.compile``
    route in non-fullgraph, dynamic, and fullgraph modes against SCPN
    parameter-shift references. It deliberately reports dynamic-shape,
    fullgraph, AOTAutograd/export, provider, hardware, CUDA, isolated benchmark,
    and performance surfaces as blocked until their promotion artefacts exist.
    """
    torch_module = _load_torch()
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
