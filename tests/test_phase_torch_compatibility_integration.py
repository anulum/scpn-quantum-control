# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Compatibility Integration Tests
"""Integration tests for Torch transforms, modules, compile, and training routes."""

from __future__ import annotations

import json

import numpy as np
import pytest
from _phase_torch_bridge_test_helpers import (
    _FakeTorch,
    _FakeTorchTensor,
    _FakeTorchWithoutCompile,
    _FakeTorchWithoutFunc,
    _FakeTorchWithoutNN,
)

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTorchCompileCompatibilityResult,
    PhaseTorchFuncCompatibilityResult,
    PhaseTorchModuleWrapperAuditResult,
    PhaseTorchTrainingLoopAuditResult,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    run_torch_compile_compatibility_audit,
    run_torch_func_compatibility_audit,
    run_torch_module_wrapper_audit,
    run_torch_training_loop_audit,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
)


def test_torch_func_compatibility_audit_checks_grad_vmap_and_jacrev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch func compatibility audit checks grad vmap and jacrev."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)

    result = run_torch_func_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    expected_batch = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch],
    )
    assert isinstance(result, PhaseTorchFuncCompatibilityResult)
    assert result.passed
    assert result.func_grad_supported
    assert result.func_vmap_supported
    assert result.func_jacrev_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_func_compatibility"
    np.testing.assert_allclose(result.grad_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.jacrev_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.vmap_gradients, expected_batch, atol=1e-12)
    assert fake_torch.func.grad_calls == 1
    assert fake_torch.func.vmap_calls == 1
    assert fake_torch.func.jacrev_calls == 1
    assert result.to_dict()["func_vmap_supported"] is True


def test_torch_func_compatibility_audit_fails_closed_without_torch_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch func compatibility audit fails closed without PyTorch func."""
    fake_torch = _FakeTorchWithoutFunc()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.func"):
        run_torch_func_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.45]], dtype=float),
        )


def test_torch_compile_compatibility_audit_checks_compiled_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch compile compatibility audit checks compiled grad."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = run_torch_compile_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTorchCompileCompatibilityResult)
    assert result.passed
    assert result.torch_compile_supported
    assert result.compiled_loss_supported
    assert result.compiled_gradient_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_compile_compatibility"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_torch.func.grad_calls == 1
    assert fake_torch.compile_calls == [{"fullgraph": True, "dynamic": False}]
    assert result.to_dict()["compiled_gradient_supported"] is True


def test_torch_compile_compatibility_audit_fails_closed_without_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch compile compatibility audit fails closed without compile."""
    fake_torch = _FakeTorchWithoutCompile()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.compile"):
        run_torch_compile_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_torch_bounded_qnn_module_and_layer_wrap_bounded_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch bounded QNN module and layer wrap bounded loss."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
    )
    layer = torch_bounded_qnn_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=False,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(features, labels, initial_params)
    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert module.claim_boundary == "bounded_torch_module_layer_wrapper"
    assert module.feature_width == 1
    assert module.host_boundary is False
    np.testing.assert_allclose(module().numpy(), expected_loss, atol=1e-12)
    np.testing.assert_allclose(module.parameter_shift_gradient(), expected_gradient, atol=1e-12)
    np.testing.assert_allclose(layer().numpy(), expected_loss, atol=1e-12)


def test_torch_module_wrapper_audit_checks_module_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch module wrapper audit checks module grad."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    result = run_torch_module_wrapper_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert isinstance(result, PhaseTorchModuleWrapperAuditResult)
    assert result.passed
    assert result.module_wrapper_supported
    assert result.layer_wrapper_supported
    assert result.trainable_parameters == 1
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_module_layer_wrapper"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["module_wrapper_supported"] is True
    assert fake_torch.func.grad_calls == 1


def test_torch_training_loop_audit_updates_module_with_compile_and_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch training loop audit updates module with compile and func."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    result = run_torch_training_loop_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        learning_rate=0.2,
        steps=4,
        tolerance=1e-12,
        fullgraph=True,
    )

    initial_loss = parameter_shift_qnn_classifier_loss(features, labels, initial_params)
    final_reference_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        result.final_params,
    )
    assert isinstance(result, PhaseTorchTrainingLoopAuditResult)
    assert result.passed
    assert result.steps == 4
    assert result.learning_rate == 0.2
    assert result.initial_loss == pytest.approx(initial_loss)
    assert result.final_loss < result.initial_loss
    assert result.loss_history.shape == (5,)
    assert np.all(np.diff(result.loss_history) <= 1e-12)
    assert result.gradient_history.shape == (4, 1)
    assert result.module_wrapper_supported
    assert result.func_grad_supported
    assert result.torch_compile_supported
    assert result.compiled_loss_supported
    assert result.parameter_update_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.final_gradient, final_reference_gradient, atol=1e-12)
    payload = result.to_dict()
    assert payload["claim_boundary"] == "bounded_torch_training_loop_parity"
    assert payload["compiled_loss_supported"] is True
    json.dumps(payload)
    assert fake_torch.func.grad_calls >= 1
    assert len(fake_torch.compile_calls) >= 1


@pytest.mark.parametrize(
    ("learning_rate", "steps", "tolerance", "match"),
    [
        (0.0, 4, 1e-12, "learning_rate"),
        (-0.1, 4, 1e-12, "learning_rate"),
        (0.2, 0, 1e-12, "steps"),
        (0.2, True, 1e-12, "steps"),
        (0.2, 4, -1e-12, "tolerance"),
    ],
)
def test_torch_training_loop_audit_fails_closed_on_invalid_controls(
    monkeypatch: pytest.MonkeyPatch,
    learning_rate: float,
    steps: int,
    tolerance: float,
    match: str,
) -> None:
    """Verify that PyTorch training loop audit fails closed on invalid controls."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match=match):
        run_torch_training_loop_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
            learning_rate=learning_rate,
            steps=steps,
            tolerance=tolerance,
        )


def test_torch_training_loop_audit_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch training loop audit fails closed on shape mismatch."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match=r"initial_params must have shape \(2,\)"):
        run_torch_training_loop_audit(
            features=np.array([[0.0, np.pi]], dtype=float),
            labels=np.array([1.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
        )


def test_torch_module_wrapper_fails_closed_without_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch module wrapper fails closed without nn."""
    fake_torch = _FakeTorchWithoutNN()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.nn.Module"):
        torch_bounded_qnn_module(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
        )
