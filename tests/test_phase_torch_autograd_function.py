# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch autograd-function audit tests
"""Tests for the bounded PyTorch custom-autograd Function audit."""

from __future__ import annotations

import builtins
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.torch_autograd_function as torch_autograd_function
from scpn_quantum_control.phase import (
    TORCH_AUTOGRAD_FUNCTION_SCHEMA,
    PhaseTorchAutogradFunctionResult,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    run_torch_autograd_function_audit,
    torch_autograd_function_qnn_loss,
    torch_autograd_qnn_value_and_grad,
)

torch = pytest.importorskip("torch")


class _ForwardContext:
    """Minimal context object for direct custom-Function forward tests."""

    def __init__(self) -> None:
        self.saved_tensors: tuple[object, ...] = ()

    def save_for_backward(self, *tensors: object) -> None:
        """Record tensors saved for backward replay."""
        self.saved_tensors = tensors


class _DetachOnly:
    """Fake tensor exposing only ``detach`` for tensor-detection tests."""

    def detach(self) -> _DetachOnly:
        """Return this object as a detached fake tensor."""
        return self


class _RequiresGradOnly:
    """Fake tensor exposing only ``requires_grad_`` for trainable paths."""

    def __init__(self) -> None:
        self.requires_grad_value = False

    def requires_grad_(self, requires_grad: bool = True) -> _RequiresGradOnly:
        """Record the requested trainable flag."""
        self.requires_grad_value = requires_grad
        return self


class _FakeAsTensorNoDtype:
    """Fake torch module with ``as_tensor`` but no dtype constant."""

    @staticmethod
    def as_tensor(values: object) -> tuple[str, object]:
        """Return the requested values through an as-tensor marker."""
        return ("as_tensor", values)


class _FakeTensorNoDtype:
    """Fake torch module with ``tensor`` but no dtype constant."""

    @staticmethod
    def tensor(values: object) -> tuple[str, object]:
        """Return the requested values through a tensor marker."""
        return ("tensor", values)


class _FakeTensorWithDtype:
    """Fake torch module with ``tensor`` and a float64 dtype constant."""

    float64 = "float64"

    @staticmethod
    def tensor(values: object, *, dtype: object) -> tuple[str, object, object]:
        """Return the requested values and dtype through a tensor marker."""
        return ("tensor", values, dtype)


class _NoZeroGrad:
    """Fake optimizer without ``zero_grad``."""

    pass


class _NoStep:
    """Fake optimizer without ``step``."""

    def zero_grad(self) -> None:
        """Expose zero_grad while omitting step."""


def _features() -> NDArray[np.float64]:
    """Return deterministic two-parameter bounded phase-QNN features."""
    return np.array(
        [
            [0.0, 1.0],
            [np.pi / 2.0, -0.4],
            [np.pi, 0.25],
            [3.0 * np.pi / 2.0, 0.75],
        ],
        dtype=np.float64,
    )


def _labels() -> NDArray[np.float64]:
    """Return deterministic bounded phase-QNN labels."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic trainable bounded phase-QNN parameters."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_autograd_function_loss_populates_backward_gradient() -> None:
    """The public loss helper should populate ``Tensor.grad`` through backward."""
    params = torch.tensor(_params(), dtype=torch.float64, requires_grad=True)

    loss = torch_autograd_function_qnn_loss(_features(), _labels(), params, tolerance=1.0e-8)
    loss.backward()

    assert params.grad is not None
    assert float(loss.detach().cpu().numpy()) == pytest.approx(
        parameter_shift_qnn_classifier_loss(_features(), _labels(), _params()),
        abs=1.0e-8,
    )
    np.testing.assert_allclose(
        params.grad.detach().cpu().numpy(),
        parameter_shift_qnn_classifier_gradient(_features(), _labels(), _params()),
        atol=1.0e-8,
    )


def test_torch_autograd_function_audit_records_optimizer_integration() -> None:
    """The audit should record backward parity and optimizer-step evidence."""
    result = run_torch_autograd_function_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        learning_rate=0.05,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchAutogradFunctionResult)
    assert result.passed
    assert result.matrix_schema == TORCH_AUTOGRAD_FUNCTION_SCHEMA
    assert result.feature_shape == (4, 2)
    assert result.gradient_shape == (2,)
    assert result.route_status("custom_autograd_backward") == "passed"
    assert result.route_status("tensor_backward_integration") == "passed"
    assert result.route_status("optimizer_step_integration") == "passed"
    assert result.route_status("higher_order_autograd") == "blocked"
    assert result.route_status("cuda_autograd_function") == "blocked"
    assert result.route_status("provider_hardware_autograd_function") == "blocked"
    assert result.route_status("isolated_benchmark_autograd_function") == "blocked"
    assert result.open_gaps == (
        "higher_order_autograd",
        "cuda_autograd_function",
        "provider_hardware_autograd_function",
        "isolated_benchmark_autograd_function",
        "arbitrary_simulator_autograd_function",
    )
    assert result.optimizer_step_delta_norm > 0.0
    assert result.optimizer_loss_after <= result.optimizer_loss_before + result.tolerance
    assert result.max_abs_error <= result.tolerance
    assert result.custom_autograd_function_claim is True
    assert result.higher_order_claim is False
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert payload["passed"] is True
    assert payload["gradient_shape"] == [2]
    assert routes["custom_autograd_backward"]["status"] == "passed"
    assert routes["arbitrary_simulator_autograd_function"]["status"] == "blocked"
    assert "torch.autograd.Function" in str(payload["claim_boundary"])


def test_torch_autograd_function_audit_rejects_invalid_inputs() -> None:
    """The audit should reject invalid shapes and training controls."""
    with pytest.raises(ValueError, match="initial_params"):
        run_torch_autograd_function_audit(
            features=_features(),
            labels=_labels(),
            initial_params=np.array([0.25], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="learning_rate"):
        run_torch_autograd_function_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            learning_rate=0.0,
        )


def test_torch_autograd_function_route_lookup_fails_closed() -> None:
    """Unknown route lookups should fail closed."""
    result = run_torch_autograd_function_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
    )

    with pytest.raises(KeyError, match="unknown PyTorch autograd-function route"):
        result.route_status("missing")


def test_existing_torch_bridge_uses_the_promoted_autograd_function_route() -> None:
    """The legacy bridge API should keep returning custom-autograd evidence."""
    result = torch_autograd_qnn_value_and_grad(
        _features(),
        _labels(),
        _params(),
        tolerance=1.0e-8,
    )

    assert result.passed
    assert result.custom_autograd_function
    assert result.method == "torch_bounded_phase_qnn_custom_autograd_function"
    np.testing.assert_allclose(result.gradient, result.parameter_shift_gradient, atol=1.0e-8)


def test_torch_autograd_function_forward_helper_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct custom-Function forward calls should cover fail-closed edges."""
    function_cls = torch_autograd_function._autograd_function_class(torch, tolerance=1.0e-8)
    context = _ForwardContext()
    loss = function_cls.forward(
        context,
        torch.tensor(_features(), dtype=torch.float64),
        torch.tensor(_labels(), dtype=torch.float64),
        torch.tensor(_params(), dtype=torch.float64),
    )

    assert context.saved_tensors
    assert float(loss.detach().cpu().numpy()) == pytest.approx(
        parameter_shift_qnn_classifier_loss(_features(), _labels(), _params()),
        abs=1.0e-8,
    )

    def _bad_gradient(
        feature_matrix: NDArray[np.float64],
        label_vector: NDArray[np.float64],
        parameter_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del feature_matrix, label_vector
        return np.ones_like(parameter_values, dtype=np.float64)

    monkeypatch.setattr(torch_autograd_function, "_bounded_analytic_qnn_gradient", _bad_gradient)
    with pytest.raises(RuntimeError, match="diverged"):
        function_cls.forward(
            _ForwardContext(),
            torch.tensor(_features(), dtype=torch.float64),
            torch.tensor(_labels(), dtype=torch.float64),
            torch.tensor(_params(), dtype=torch.float64),
        )


def test_torch_autograd_function_route_failure_classification() -> None:
    """Route classification should expose failed local evidence rows."""
    routes = torch_autograd_function._classify_routes(
        gradient=np.array([1.0], dtype=np.float64),
        reference_gradient=np.array([0.0, 0.0], dtype=np.float64),
        optimizer_loss_before=1.0,
        optimizer_loss_after=2.0,
        optimizer_step_delta_norm=0.0,
        tolerance=1.0e-8,
    )
    statuses = {route.name: route.status for route in routes}

    assert statuses["custom_autograd_backward"] == "failed"
    assert statuses["tensor_backward_integration"] == "failed"
    assert statuses["optimizer_step_integration"] == "failed"


def test_torch_autograd_function_private_helper_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private helpers should fail closed for malformed runtime surfaces."""
    real_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "torch":
            raise ImportError("blocked torch")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match="PyTorch is unavailable"):
        torch_autograd_function._load_torch()
    monkeypatch.setattr(builtins, "__import__", real_import)

    with pytest.raises(RuntimeError, match="autograd.Function"):
        torch_autograd_function._torch_autograd_function(object())
    with pytest.raises(RuntimeError, match="as_tensor or tensor"):
        torch_autograd_function._torch_tensor_like(object(), 1.0, object())
    assert torch_autograd_function._torch_tensor_like(
        _FakeAsTensorNoDtype(),
        [1.0],
        object(),
    ) == ("as_tensor", [1.0])
    assert torch_autograd_function._torch_tensor_like(
        _FakeTensorNoDtype(),
        [1.0],
        object(),
    ) == ("tensor", [1.0])
    assert torch_autograd_function._torch_tensor_like(
        _FakeTensorWithDtype(),
        [1.0],
        object(),
    ) == ("tensor", [1.0], "float64")

    assert not torch_autograd_function._looks_like_torch_tensor(_DetachOnly())
    assert torch_autograd_function._parameter_tensor(
        torch,
        np.array([0.1], dtype=np.float64),
        np.array([0.1], dtype=np.float64),
    ).shape == (1,)

    fake_trainable = _RequiresGradOnly()
    monkeypatch.setattr(
        torch_autograd_function,
        "_torch_tensor_like",
        lambda _torch_module, _values, _like: fake_trainable,
    )
    assert (
        torch_autograd_function._trainable_parameter_tensor(
            object(),
            np.array([0.1], dtype=np.float64),
        )
        is fake_trainable
    )
    assert fake_trainable.requires_grad_value is True
    monkeypatch.setattr(
        torch_autograd_function,
        "_torch_tensor_like",
        lambda _torch_module, _values, _like: object(),
    )
    with pytest.raises(RuntimeError, match="requires_grad_"):
        torch_autograd_function._trainable_parameter_tensor(
            object(),
            np.array([0.1], dtype=np.float64),
        )

    with pytest.raises(RuntimeError, match="backward"):
        torch_autograd_function._torch_backward(object())
    with pytest.raises(RuntimeError, match="params.grad"):
        torch_autograd_function._require_tensor_gradient(object())
    with pytest.raises(RuntimeError, match="torch.optim.SGD"):
        torch_autograd_function._torch_sgd_optimizer(object(), (), learning_rate=0.1)
    with pytest.raises(RuntimeError, match="zero_grad"):
        torch_autograd_function._torch_optimizer_zero_grad(_NoZeroGrad())
    with pytest.raises(RuntimeError, match="step"):
        torch_autograd_function._torch_optimizer_step(_NoStep())


def test_torch_autograd_function_validation_helper_edges() -> None:
    """Validation helpers should reject malformed numeric inputs."""
    with pytest.raises(ValueError, match="finite"):
        torch_autograd_function._values_to_numpy([np.inf])
    with pytest.raises(ValueError, match="two-dimensional"):
        torch_autograd_function._as_feature_matrix(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="must not be empty"):
        torch_autograd_function._as_feature_matrix(np.empty((0, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        torch_autograd_function._as_feature_matrix(
            np.array([[np.nan]], dtype=np.float64),
        )

    single_column = torch_autograd_function._as_label_vector(
        np.array([[0.0], [1.0]], dtype=np.float64),
        n_samples=2,
    )
    np.testing.assert_allclose(single_column, np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        torch_autograd_function._as_label_vector(
            np.zeros((2, 2), dtype=np.float64),
            n_samples=2,
        )
    with pytest.raises(ValueError, match="shape"):
        torch_autograd_function._as_label_vector(
            np.array([0.0], dtype=np.float64),
            n_samples=2,
        )
    with pytest.raises(ValueError, match="finite"):
        torch_autograd_function._as_label_vector(
            np.array([np.nan], dtype=np.float64),
            n_samples=1,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        torch_autograd_function._as_label_vector(
            np.array([2.0], dtype=np.float64),
            n_samples=1,
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        torch_autograd_function._as_parameter_vector(
            "params",
            np.zeros((1, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        torch_autograd_function._as_parameter_vector(
            "params",
            np.array([np.inf], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        torch_autograd_function._as_non_negative_tolerance(-1.0)
    with pytest.raises(ValueError, match="positive"):
        torch_autograd_function._as_positive_learning_rate(float("nan"))
    with pytest.raises(ValueError, match="scalar-like"):
        torch_autograd_function._scalar_to_float(np.array([1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        torch_autograd_function._scalar_to_float(np.array([np.nan], dtype=np.float64))

    assert torch_autograd_function._max_abs_error(
        np.array([1.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    ) == float("inf")
    assert (
        torch_autograd_function._max_abs_error(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        == 0.0
    )
