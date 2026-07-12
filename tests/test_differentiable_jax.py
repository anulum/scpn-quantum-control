# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable JAX tests
# scpn-quantum-control -- differentiable JAX bridge tests
"""Tests for the optional differentiable JAX autodiff bridge."""

from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    is_jax_autodiff_available,
    jax_value_and_grad,
)
from scpn_quantum_control.differentiable_jax_adapter import (
    is_jax_autodiff_available as direct_is_jax_autodiff_available,
)
from scpn_quantum_control.differentiable_jax_adapter import (
    jax_value_and_grad as direct_jax_value_and_grad,
)

_ImportFn = Callable[..., object]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across optional JAX result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _install_fake_jax(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gradient: object,
    value: object | None = None,
) -> None:
    """Install a minimal JAX module pair for deterministic adapter tests."""

    jax_module = types.ModuleType("jax")
    jax_module.__path__ = []
    jax_numpy = types.ModuleType("jax.numpy")
    jax_numpy.asarray = np.asarray  # type: ignore[attr-defined]

    def value_and_grad(function: Callable[[Any], Any]) -> Callable[[Any], tuple[object, object]]:
        def evaluate(raw_values: Any) -> tuple[object, object]:
            objective_value = function(raw_values) if value is None else value
            return objective_value, gradient

        return evaluate

    jax_module.value_and_grad = value_and_grad  # type: ignore[attr-defined]
    jax_module.numpy = jax_numpy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy)


def _block_jax_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch import resolution so JAX behaves as unavailable."""

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "jax" or name.startswith("jax."):
            raise ImportError("blocked JAX import")
        return cast(_ImportFn, real_import)(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_jax_adapter_helpers_preserve_facade_identity() -> None:
    """Extracted JAX adapter helpers should keep facade and package-root imports stable."""

    import scpn_quantum_control as scpn
    from scpn_quantum_control import differentiable as differentiable_facade

    assert differentiable_facade.is_jax_autodiff_available is direct_is_jax_autodiff_available
    assert differentiable_facade.jax_value_and_grad is direct_jax_value_and_grad
    assert scpn.is_jax_autodiff_available is direct_is_jax_autodiff_available
    assert scpn.jax_value_and_grad is direct_jax_value_and_grad


def test_jax_adapter_reports_unavailable_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional JAX imports should remain fail-closed and explicit."""

    _block_jax_imports(monkeypatch)

    assert not is_jax_autodiff_available()
    with pytest.raises(ImportError, match="JAX autodiff is unavailable"):
        jax_value_and_grad(lambda values: values[0] ** 2, [2.0])


def test_jax_adapter_validates_fake_jax_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """JAX adapter should accept finite scalar values and matching gradients."""

    _install_fake_jax(monkeypatch, gradient=np.array([4.0], dtype=np.float64))

    assert is_jax_autodiff_available()
    value, gradient = jax_value_and_grad(lambda values: values[0] ** 2, [2.0])

    assert value == pytest.approx(4.0)
    _assert_allclose(gradient, [4.0])


def test_jax_adapter_rejects_malformed_fake_jax_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JAX adapter should reject malformed value and gradient payloads."""

    _install_fake_jax(monkeypatch, gradient=np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="gradient shape"):
        jax_value_and_grad(lambda values: values[0] ** 2, [2.0])

    _install_fake_jax(monkeypatch, gradient=np.array([np.inf], dtype=np.float64))
    with pytest.raises(ValueError, match="finite values"):
        jax_value_and_grad(lambda values: values[0] ** 2, [2.0])

    _install_fake_jax(
        monkeypatch,
        gradient=np.array([4.0], dtype=np.float64),
        value=np.array([4.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="real numeric scalar"):
        jax_value_and_grad(lambda values: values[0] ** 2, [2.0])


def test_jax_value_and_grad_matches_quadratic_when_available() -> None:
    """Optional JAX bridge should expose real autodiff when JAX is installed."""

    if not is_jax_autodiff_available():
        with pytest.raises(ImportError, match="JAX"):
            jax_value_and_grad(lambda values: values[0] ** 2, [2.0])
        return

    value, gradient = jax_value_and_grad(lambda values: values[0] ** 2, [2.0])
    assert value == pytest.approx(4.0)
    _assert_allclose(gradient, [4.0], rtol=1.0e-6, atol=1.0e-6)


def test_jax_value_and_grad_rejects_implicit_parameter_coercion() -> None:
    """JAX bridge input validation should match the native differentiable path."""

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        jax_value_and_grad(lambda values: values[0] ** 2, ["2.0"])
