# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded phase-QNN framework gradients
"""Differentiate the bounded phase classifier in PyTorch, JAX and TensorFlow.

Each audit computes the classifier loss and gradient with one real framework
route and compares it with the canonical parameter-shift reference: a
``torch.autograd.Function`` whose backward is the audited analytic gradient,
native JAX ``value_and_grad`` (optionally under ``jit``), and the analytic
gradient returned as TensorFlow tensors. The caller supplies the framework
modules; this module never loads one. Admission and conversion operations are
reached through resolvers called where each audit needs them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_backend_capabilities import (
    BackendCapabilityPrimitives,
    resolve_torch_autograd_function,
    resolve_torch_autograd_grad,
)
from .differentiable_tensor_construction import (
    TensorConstructionPrimitives,
    construct_tensorflow_tensor,
    construct_torch_tensor,
    prepare_torch_trainable_tensor,
)
from .differentiable_tensor_materialization import (
    TensorMaterializationPrimitives,
    materialize_tensorflow_host_values,
    materialize_torch_host_values,
)
from .phase.jax_bridge_contracts import PhaseJAXNativeQNNGradientResult
from .phase.qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)
from .phase.tensorflow_bridge_contracts import PhaseTensorFlowQNNGradientResult
from .phase.torch_bridge_contracts import PhaseTorchAutogradQNNGradientResult
from .phase_qnn_model_numerics import PhaseModelNumerics
from .phase_qnn_native_model import jax_phase_classifier_loss
from .phase_qnn_numpy_model import numpy_phase_classifier_loss_and_gradient
from .phase_qnode_audit_admission import (
    admitted_jax_tolerance,
    admitted_parameter_vector,
    admitted_scalar,
    admitted_torch_tolerance,
)

FloatArray: TypeAlias = NDArray[np.float64]
Resolver: TypeAlias = Callable[[], Callable[..., Any]]
"""Zero-argument callable returning the operation to call at that moment."""
AuditRecord: TypeAlias = tuple[float, FloatArray, FloatArray, float, float, bool]
"""Loss, gradient, reference gradient, maximum and L2 error, and verdict."""


@dataclass(frozen=True)
class QNNGradientOperations:
    """Resolvers for the admission, model and conversion operations of the audits.

    Unset operations default to this module's PyTorch/TensorFlow-route
    admission, NumPy classifier model and PyTorch tensor construction.

    Parameters
    ----------
    feature_matrix, label_vector, parameter_vector, tolerance
        Admission of features, labels, parameters and the tolerance.
    analytic_model
        ``(features, labels, params) -> (loss, gradient)`` of the classifier.
    reference_loss, reference_gradient
        Parameter-shift classifier loss and gradient.
    tensor, trainable_tensor
        ``(runtime, values) -> tensor`` construction, without and with gradients.
    scalar
        Admission of a native scalar loss.

    """

    parameter_vector: Resolver
    reference_loss: Resolver
    reference_gradient: Resolver
    feature_matrix: Resolver = lambda: admitted_feature_matrix
    label_vector: Resolver = lambda: admitted_label_vector
    tolerance: Resolver = lambda: admitted_torch_tolerance
    analytic_model: Resolver = lambda: _numpy_analytic_model
    tensor: Resolver = lambda: _torch_tensor
    trainable_tensor: Resolver = lambda: _torch_trainable_tensor
    scalar: Resolver = lambda: admitted_scalar


def admitted_feature_matrix(features: object) -> FloatArray:
    """Return a finite, non-empty float64 feature matrix (PyTorch/TensorFlow routes).

    Parameters
    ----------
    features
        Host matrix-like features.

    Returns
    -------
    FloatArray
        Admitted ``(samples, features)`` copy.

    Raises
    ------
    ValueError
        If the features are not a finite, non-empty matrix.

    """
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def admitted_label_vector(labels: object, *, n_samples: int) -> FloatArray:
    """Return a finite float64 label vector of ``n_samples`` (PyTorch/TensorFlow routes).

    Parameters
    ----------
    labels
        Host vector-like labels.
    n_samples
        Required number of labels.

    Returns
    -------
    FloatArray
        Admitted label copy.

    Raises
    ------
    ValueError
        If the labels are not a finite vector of the given length.

    """
    vector = np.asarray(labels, dtype=float)
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if vector.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    return vector.astype(np.float64, copy=True)


def jax_admitted_feature_matrix(features: object) -> FloatArray:
    """Return a finite float64 feature matrix with the JAX route's refusals.

    Parameters
    ----------
    features
        Host matrix-like features.

    Returns
    -------
    FloatArray
        Admitted ``(samples, features)`` copy.

    Raises
    ------
    ValueError
        If the features are not a finite matrix with samples and columns.

    """
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def jax_admitted_label_vector(labels: object, *, n_samples: int) -> FloatArray:
    """Return finite probability labels with the JAX route's refusals.

    Parameters
    ----------
    labels
        Host vector-like or single-column labels.
    n_samples
        Required number of labels.

    Returns
    -------
    FloatArray
        Admitted label copy in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the labels are not finite, of the given length and in ``[0, 1]``.

    """
    vector = np.asarray(labels, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array or a single-column matrix")
    if vector.shape != (n_samples,):
        raise ValueError("features and labels must have the same sample count")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in the closed interval [0, 1]")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def admitted_bounded_qnn_reference(
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameter_values: FloatArray,
    tolerance: float,
    operations: QNNGradientOperations,
    *,
    runtime_label: str,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float, float, bool]:
    """Compare the analytic classifier gradient with the parameter-shift reference.

    Parameters
    ----------
    feature_matrix, label_vector, parameter_values
        Admitted classifier inputs with matching widths.
    tolerance
        Admitted non-negative tolerance.
    operations
        Model and admission resolvers.
    runtime_label
        Framework name used in messages, such as ``"PyTorch"``.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, float, float, float, bool]
        Loss, gradient, reference gradient, maximum and L2 error, tolerance and
        verdict.

    Raises
    ------
    RuntimeError
        If the analytic loss disagrees with the parameter-shift loss.

    """
    loss, gradient = operations.analytic_model()(feature_matrix, label_vector, parameter_values)
    gradient = operations.parameter_vector()(
        f"{runtime_label} bounded phase-QNN gradient",
        gradient,
        width=parameter_values.size,
    )
    reference_loss = operations.reference_loss()(feature_matrix, label_vector, parameter_values)
    if abs(loss - reference_loss) > tolerance:
        raise RuntimeError(
            f"{runtime_label} bounded phase-QNN tensor loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient = operations.reference_gradient()(
        feature_matrix, label_vector, parameter_values
    )
    reference_gradient = operations.parameter_vector()(
        "SCPN bounded phase-QNN parameter-shift gradient",
        reference_gradient,
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    passed = bool(max_abs_error <= tolerance)
    return (
        np.asarray(loss, dtype=np.float64),
        gradient,
        reference_gradient,
        max_abs_error,
        l2_error,
        tolerance,
        passed,
    )


def bounded_qnn_reference(
    features: object,
    labels: object,
    params: object,
    *,
    tolerance: float,
    operations: QNNGradientOperations,
    runtime_label: str,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float, float, bool]:
    """Admit classifier inputs and compare the analytic and reference gradients.

    Parameters
    ----------
    features, labels, params
        Host classifier inputs; the parameter width must match the features.
    tolerance
        Candidate non-negative tolerance.
    operations
        Model and admission resolvers.
    runtime_label
        Framework name used in messages.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, float, float, float, bool]
        As :func:`admitted_bounded_qnn_reference`.

    Raises
    ------
    ValueError
        If the parameter width does not match the feature width.

    """
    feature_matrix = operations.feature_matrix()(features)
    label_vector = operations.label_vector()(labels, n_samples=feature_matrix.shape[0])
    parameter_values = operations.parameter_vector()("params", params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = operations.tolerance()(tolerance)
    return admitted_bounded_qnn_reference(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance_value,
        operations,
        runtime_label=runtime_label,
    )


def torch_autograd_qnn_audit(
    torch_module: Any,
    autograd_grad: Any,
    function_class: Any,
    audit: dict[str, AuditRecord],
    parameter_values: FloatArray,
    tolerance: float,
    operations: QNNGradientOperations,
) -> PhaseTorchAutogradQNNGradientResult:
    """Differentiate through a ``torch.autograd.Function`` and read its audit record.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    autograd_grad
        ``torch.autograd.grad``.
    function_class
        ``torch.autograd.Function`` subclass whose forward stores the audit record.
    audit
        Mapping the forward fills under ``"record"``.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    operations
        Tensor construction and admission resolvers.

    Returns
    -------
    PhaseTorchAutogradQNNGradientResult
        Native loss and gradient with the forward's reference comparison.

    Raises
    ------
    RuntimeError
        If the forward did not record its audit evidence.

    """
    trainable_params = operations.trainable_tensor()(torch_module, parameter_values)
    torch_loss = function_class.apply(trainable_params)
    torch_gradient = autograd_grad(
        torch_loss,
        trainable_params,
        retain_graph=False,
        create_graph=False,
    )[0]
    try:
        loss, gradient, reference_gradient, max_abs_error, l2_error, passed = audit["record"]
    except KeyError as exc:
        raise RuntimeError("PyTorch autograd forward did not produce audit evidence") from exc
    return PhaseTorchAutogradQNNGradientResult(
        loss=loss,
        gradient=operations.parameter_vector()(
            "PyTorch autograd bounded phase-QNN gradient", gradient
        ),
        parameter_shift_gradient=reference_gradient,
        torch_loss=torch_loss,
        torch_gradient=torch_gradient,
        torch_parameter_shift_gradient=operations.tensor()(torch_module, reference_gradient),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=passed,
    )


def jax_native_qnn_audit(
    jax_module: Any,
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameter_values: FloatArray,
    tolerance: float,
    loss_fn: Callable[[object], object],
    operations: QNNGradientOperations,
    *,
    jnp: Any,
    jit: bool,
) -> PhaseJAXNativeQNNGradientResult:
    """Differentiate the classifier loss with native JAX ``value_and_grad``.

    Parameters
    ----------
    jax_module
        Supplied JAX module.
    feature_matrix, label_vector, parameter_values
        Admitted classifier inputs.
    tolerance
        Admitted non-negative tolerance.
    loss_fn
        Differentiable ``parameters -> loss``.
    operations
        Reference and admission resolvers.
    jnp
        Supplied ``jax.numpy`` namespace.
    jit
        Whether to compile the value-and-gradient function with ``jax.jit``.

    Returns
    -------
    PhaseJAXNativeQNNGradientResult
        Native loss and gradient with the reference comparison.

    Raises
    ------
    RuntimeError
        If ``value_and_grad`` or a requested ``jit`` is unavailable, or the
        native loss disagrees with the parameter-shift loss.

    """
    value_and_grad_fn = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad_fn):
        raise RuntimeError("JAX value_and_grad is required for native QNN autodiff")
    executable = value_and_grad_fn(loss_fn)
    jitted = False
    if jit:
        jit_fn = getattr(jax_module, "jit", None)
        if not callable(jit_fn):
            raise RuntimeError("JAX JIT is unavailable in the active JAX module")
        executable = jit_fn(executable)
        jitted = True

    loss_obj, gradient_obj = executable(jnp.asarray(parameter_values))
    gradient = operations.parameter_vector()(
        "JAX native QNN gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    reference_gradient = operations.reference_gradient()(
        feature_matrix, label_vector, parameter_values
    )
    reference_loss = operations.reference_loss()(feature_matrix, label_vector, parameter_values)
    loss = operations.scalar()("JAX native QNN loss", loss_obj)
    if abs(loss - reference_loss) > max(tolerance, 1e-12):
        raise RuntimeError("JAX native QNN loss disagrees with parameter-shift loss")
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXNativeQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=max_abs_error <= tolerance,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


def tensorflow_bounded_qnn_audit(
    tensorflow_module: Any,
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameter_values: FloatArray,
    tolerance: float,
    operations: QNNGradientOperations,
) -> PhaseTensorFlowQNNGradientResult:
    """Return the checked analytic classifier gradient as TensorFlow tensors.

    Parameters
    ----------
    tensorflow_module
        Supplied TensorFlow module.
    feature_matrix, label_vector, parameter_values
        Admitted classifier inputs with matching widths.
    tolerance
        Admitted non-negative tolerance.
    operations
        Model, tensor and admission resolvers.

    Returns
    -------
    PhaseTensorFlowQNNGradientResult
        Host and TensorFlow loss and gradients with the reference comparison.

    """
    loss, gradient, reference_gradient, max_abs_error, l2_error, _tolerance, passed = (
        admitted_bounded_qnn_reference(
            feature_matrix,
            label_vector,
            parameter_values,
            tolerance,
            operations,
            runtime_label="TensorFlow",
        )
    )
    tensor = operations.tensor
    return PhaseTensorFlowQNNGradientResult(
        loss=float(loss),
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        tensorflow_loss=tensor()(tensorflow_module, np.asarray(loss, dtype=np.float64)),
        tensorflow_gradient=tensor()(tensorflow_module, gradient),
        tensorflow_parameter_shift_gradient=tensor()(tensorflow_module, reference_gradient),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance,
        passed=passed,
    )


def _capability_primitives() -> BackendCapabilityPrimitives:
    return BackendCapabilityPrimitives(
        attribute=lambda: getattr,
        predicate=lambda: callable,
        presence=lambda: hasattr,
        invalid=lambda: RuntimeError,
    )


def _construction_primitives() -> TensorConstructionPrimitives:
    return TensorConstructionPrimitives(
        attribute=lambda: getattr, predicate=lambda: callable, invalid=lambda: RuntimeError
    )


def _materialization_primitives() -> TensorMaterializationPrimitives:
    return TensorMaterializationPrimitives(attribute=lambda: getattr, predicate=lambda: callable)


def _numpy_analytic_model(
    features: FloatArray, labels: FloatArray, params: FloatArray
) -> tuple[float, FloatArray]:
    return numpy_phase_classifier_loss_and_gradient(
        features,
        labels,
        params,
        numerics=PhaseModelNumerics(numpy=lambda: np, scalar=lambda: float),
    )


def _torch_tensor(torch_module: Any, values: object) -> Any:
    return construct_torch_tensor(torch_module, values, primitives=_construction_primitives())


def _torch_trainable_tensor(torch_module: Any, values: FloatArray) -> Any:
    return prepare_torch_trainable_tensor(
        _torch_tensor(torch_module, values), primitives=_construction_primitives()
    )


def _tensorflow_tensor(tensorflow_module: Any, values: object) -> Any:
    return construct_tensorflow_tensor(
        tensorflow_module, values, primitives=_construction_primitives()
    )


def _native_operations(
    *,
    feature_matrix: Callable[..., Any],
    label_vector: Callable[..., Any],
    tolerance: Callable[..., Any],
    tensor: Callable[..., Any],
) -> QNNGradientOperations:
    return QNNGradientOperations(
        feature_matrix=lambda: feature_matrix,
        label_vector=lambda: label_vector,
        parameter_vector=lambda: admitted_parameter_vector,
        tolerance=lambda: tolerance,
        analytic_model=lambda: _numpy_analytic_model,
        reference_loss=lambda: parameter_shift_qnn_classifier_loss,
        reference_gradient=lambda: parameter_shift_qnn_classifier_gradient,
        tensor=lambda: tensor,
        trainable_tensor=lambda: _torch_trainable_tensor,
        scalar=lambda: admitted_scalar,
    )


NATIVE_TORCH_QNN_OPERATIONS = _native_operations(
    feature_matrix=admitted_feature_matrix,
    label_vector=admitted_label_vector,
    tolerance=admitted_torch_tolerance,
    tensor=_torch_tensor,
)
"""Lower admission, model and PyTorch tensor operations."""

NATIVE_JAX_QNN_OPERATIONS = _native_operations(
    feature_matrix=jax_admitted_feature_matrix,
    label_vector=jax_admitted_label_vector,
    tolerance=admitted_jax_tolerance,
    tensor=_torch_tensor,
)
"""Lower admission and reference operations of the JAX route."""

NATIVE_TENSORFLOW_QNN_OPERATIONS = _native_operations(
    feature_matrix=admitted_feature_matrix,
    label_vector=admitted_label_vector,
    tolerance=admitted_torch_tolerance,
    tensor=_tensorflow_tensor,
)
"""Lower admission, model and TensorFlow tensor operations."""


def native_torch_autograd_qnn_value_and_grad(
    torch_module: Any,
    features: object,
    labels: object,
    params: object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchAutogradQNNGradientResult:
    """Differentiate the classifier through a lower ``torch.autograd.Function``.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module; ``torch.autograd.Function`` and
        ``torch.autograd.grad`` are required.
    features, labels, params
        Host classifier inputs; the parameter width must match the features.
    tolerance
        Non-negative tolerance.

    Returns
    -------
    PhaseTorchAutogradQNNGradientResult
        Native loss and gradient with the reference comparison.

    Raises
    ------
    ValueError
        If the parameter width does not match the feature width.

    """
    operations = NATIVE_TORCH_QNN_OPERATIONS
    function_base = resolve_torch_autograd_function(
        torch_module, primitives=_capability_primitives()
    )
    autograd_grad = resolve_torch_autograd_grad(torch_module, primitives=_capability_primitives())
    feature_matrix = admitted_feature_matrix(features)
    label_vector = admitted_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = admitted_parameter_vector(
        "values",
        materialize_torch_host_values(params, primitives=_materialization_primitives()),
    )
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = admitted_torch_tolerance(tolerance)
    audit: dict[str, AuditRecord] = {}

    def forward(ctx: Any, parameter_tensor: Any) -> Any:
        """Record the reference comparison and return the native loss.

        Parameters
        ----------
        ctx
            Autograd context receiving the analytic gradient.
        parameter_tensor
            Native parameter tensor.

        Returns
        -------
        Any
            Native scalar loss.

        """
        raw_params = admitted_parameter_vector(
            "values",
            materialize_torch_host_values(
                parameter_tensor, primitives=_materialization_primitives()
            ),
        )
        loss, gradient, reference_gradient, max_abs_error, l2_error, _checked, passed = (
            bounded_qnn_reference(
                feature_matrix,
                label_vector,
                raw_params,
                tolerance=tolerance_value,
                operations=operations,
                runtime_label="PyTorch",
            )
        )
        ctx.gradient = _torch_tensor(torch_module, gradient)
        audit["record"] = (
            float(loss),
            gradient,
            reference_gradient,
            max_abs_error,
            l2_error,
            passed,
        )
        return _torch_tensor(torch_module, loss)

    def backward(ctx: Any, grad_output: Any) -> tuple[Any]:
        """Scale the stored analytic gradient by the output cotangent.

        Parameters
        ----------
        ctx
            Autograd context holding the analytic gradient.
        grad_output
            Upstream cotangent.

        Returns
        -------
        tuple[Any]
            Parameter cotangent.

        """
        return (ctx.gradient * grad_output,)

    function_class = type(
        "BoundedPhaseQNNFunction",
        (cast(type, function_base),),
        {
            "__doc__": "Bounded classifier loss whose backward is the audited analytic gradient.",
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )
    return torch_autograd_qnn_audit(
        torch_module,
        autograd_grad,
        function_class,
        audit,
        parameter_values,
        tolerance_value,
        operations,
    )


def native_jax_qnn_value_and_grad(
    jax_module: Any,
    jnp: Any,
    features: object,
    labels: object,
    params: object,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXNativeQNNGradientResult:
    """Differentiate the classifier loss with native JAX ``value_and_grad``.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    features, labels, params
        Host classifier inputs; the parameter width must match the features.
    tolerance
        Non-negative tolerance.
    jit
        Whether to compile the value-and-gradient function.

    Returns
    -------
    PhaseJAXNativeQNNGradientResult
        Native loss and gradient with the reference comparison.

    """
    tolerance_value = admitted_jax_tolerance(tolerance)
    feature_matrix = jax_admitted_feature_matrix(features)
    label_vector = jax_admitted_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = admitted_parameter_vector("params", params, width=feature_matrix.shape[1])
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def loss_fn(raw_params: object) -> object:
        return cast(
            object,
            jax_phase_classifier_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params)),
        )

    return jax_native_qnn_audit(
        jax_module,
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance_value,
        loss_fn,
        NATIVE_JAX_QNN_OPERATIONS,
        jnp=jnp,
        jit=jit,
    )


def native_tensorflow_bounded_qnn_value_and_grad(
    tensorflow_module: Any,
    features: object,
    labels: object,
    params: object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowQNNGradientResult:
    """Return the checked analytic classifier gradient as TensorFlow tensors.

    Parameters
    ----------
    tensorflow_module
        Supplied TensorFlow module.
    features, labels, params
        Host or TensorFlow classifier inputs; the parameter width must match
        the features.
    tolerance
        Non-negative tolerance.

    Returns
    -------
    PhaseTensorFlowQNNGradientResult
        Host and TensorFlow loss and gradients with the reference comparison.

    Raises
    ------
    ValueError
        If the parameter width does not match the feature width.

    """
    feature_matrix = admitted_feature_matrix(features)
    label_vector = admitted_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = admitted_parameter_vector(
        "values",
        materialize_tensorflow_host_values(params, primitives=_materialization_primitives()),
    )
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = admitted_torch_tolerance(tolerance)
    return tensorflow_bounded_qnn_audit(
        tensorflow_module,
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance_value,
        NATIVE_TENSORFLOW_QNN_OPERATIONS,
    )


__all__ = [
    "NATIVE_JAX_QNN_OPERATIONS",
    "NATIVE_TENSORFLOW_QNN_OPERATIONS",
    "NATIVE_TORCH_QNN_OPERATIONS",
    "QNNGradientOperations",
    "admitted_bounded_qnn_reference",
    "admitted_feature_matrix",
    "admitted_label_vector",
    "bounded_qnn_reference",
    "jax_admitted_feature_matrix",
    "jax_admitted_label_vector",
    "jax_native_qnn_audit",
    "native_jax_qnn_value_and_grad",
    "native_tensorflow_bounded_qnn_value_and_grad",
    "native_torch_autograd_qnn_value_and_grad",
    "tensorflow_bounded_qnn_audit",
    "torch_autograd_qnn_audit",
]
