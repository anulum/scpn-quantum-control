# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Gradient Leaf Tests
"""Direct leaf and compatibility-wrapper tests for bounded JAX gradients."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.jax_bridge as jax_bridge
import scpn_quantum_control.phase.jax_gradients as jax_gradients

FloatArray = NDArray[np.float64]


class _NumPyJAXArrayAPI:
    """Minimal JAX NumPy array conversion surface for host-gradient tests."""

    @staticmethod
    def asarray(values: object) -> FloatArray:
        """Convert values to a float64 NumPy array."""
        return np.asarray(values, dtype=np.float64)


def _objective(values: FloatArray) -> float:
    """Evaluate a one-parameter cosine objective in radians."""
    return float(np.cos(values[0]))


def _leaf_loader() -> tuple[object, _NumPyJAXArrayAPI]:
    """Return the minimal framework pair required by host-gradient routes."""
    return object(), _NumPyJAXArrayAPI()


def test_gradient_leaf_has_no_facade_back_edge() -> None:
    """Keep the gradient implementation independent from its compatibility facade."""
    tree = ast.parse(inspect.getsource(jax_gradients))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "jax_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_parameter_shift_leaf_executes_with_injected_loader() -> None:
    """Execute the host parameter-shift route directly through the gradient leaf."""
    values = np.array([0.4], dtype=np.float64)

    result = jax_gradients.jax_parameter_shift_value_and_grad(
        _objective,
        values,
        _jax_loader=_leaf_loader,
    )

    assert result.value == pytest.approx(np.cos(0.4))
    np.testing.assert_allclose(result.gradient, [-np.sin(0.4)], atol=1e-8)
    assert not result.jitted
    assert not result.host_callback


def test_gradient_agreement_leaf_executes_without_framework_arrays() -> None:
    """Compare a caller gradient directly while retaining the optional-JAX gate."""
    values = np.array([0.4], dtype=np.float64)

    result = jax_gradients.check_jax_parameter_shift_agreement(
        _objective,
        lambda raw: -np.sin(raw),
        values,
        _jax_loader=_leaf_loader,
    )

    assert result.passed
    assert result.max_abs_error == pytest.approx(0.0, abs=1e-8)


def test_phase_exports_keep_signature_compatible_facade_wrappers() -> None:
    """Expose facade wrappers publicly while keeping loader injection private."""
    names = (
        "jax_parameter_shift_value_and_grad",
        "check_jax_parameter_shift_agreement",
        "jax_native_qnn_value_and_grad",
        "jax_custom_vjp_qnn_value_and_grad",
    )
    for name in names:
        facade_function = getattr(jax_bridge, name)
        leaf_function = getattr(jax_gradients, name)
        assert getattr(phase, name) is facade_function
        assert facade_function is not leaf_function
        assert "_jax_loader" not in inspect.signature(facade_function).parameters
        assert "_jax_loader" in inspect.signature(leaf_function).parameters


def test_parameter_shift_facade_injects_its_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into the parameter-shift implementation."""
    sentinel = object()
    captured: dict[str, object] = {}

    def active_loader() -> tuple[object, object]:
        """Return the sentinel framework pair bound at the facade."""
        return sentinel, sentinel

    def implementation(*args: object, **kwargs: object) -> object:
        """Capture wrapper delegation arguments at the dynamic test boundary."""
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(jax_bridge, "_load_jax", active_loader)
    monkeypatch.setattr(jax_bridge, "_jax_parameter_shift_value_and_grad", implementation)

    result = jax_bridge.jax_parameter_shift_value_and_grad(_objective, np.array([0.2]))

    assert result is sentinel
    assert captured["_jax_loader"] is active_loader


def test_agreement_facade_injects_its_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into the agreement implementation."""
    sentinel = object()
    captured: dict[str, object] = {}

    def active_loader() -> tuple[object, object]:
        """Return the sentinel framework pair bound at the facade."""
        return sentinel, sentinel

    def implementation(*args: object, **kwargs: object) -> object:
        """Capture wrapper delegation arguments at the dynamic test boundary."""
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(jax_bridge, "_load_jax", active_loader)
    monkeypatch.setattr(jax_bridge, "_check_jax_parameter_shift_agreement", implementation)

    result = jax_bridge.check_jax_parameter_shift_agreement(
        _objective,
        lambda raw: raw,
        np.array([0.2]),
    )

    assert result is sentinel
    assert captured["_jax_loader"] is active_loader


@pytest.mark.parametrize(
    ("facade_name", "implementation_name"),
    [
        ("jax_native_qnn_value_and_grad", "_jax_native_qnn_value_and_grad"),
        ("jax_custom_vjp_qnn_value_and_grad", "_jax_custom_vjp_qnn_value_and_grad"),
    ],
)
def test_qnn_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
    facade_name: str,
    implementation_name: str,
) -> None:
    """Pass the facade loader binding into both bounded-QNN implementations."""
    sentinel = object()
    captured: dict[str, object] = {}

    def active_loader() -> tuple[object, object]:
        """Return the sentinel framework pair bound at the facade."""
        return sentinel, sentinel

    def implementation(*args: object, **kwargs: object) -> object:
        """Capture wrapper delegation arguments at the dynamic test boundary."""
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(jax_bridge, "_load_jax", active_loader)
    monkeypatch.setattr(jax_bridge, implementation_name, implementation)
    facade_function = getattr(jax_bridge, facade_name)

    result = facade_function(
        np.array([[0.1]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.2], dtype=np.float64),
    )

    assert result is sentinel
    assert captured["_jax_loader"] is active_loader
