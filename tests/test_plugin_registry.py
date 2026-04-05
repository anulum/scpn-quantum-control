# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Plugin Registry
"""Tests for extensible hardware backend plugin registry.

Covers:
    - PluginRegistry init, default lazy loaders
    - register (decorator), register_class (programmatic)
    - list_backends, available_backends
    - is_available for registered, lazy, and unknown
    - get_runner for registered, lazy, qiskit special-case, and unknown
    - Lazy loading caches class after first load
    - Error handling for unavailable backends
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.plugin_registry import PluginRegistry, registry


class TestPluginRegistryInit:
    def test_default_lazy_loaders(self):
        r = PluginRegistry()
        assert "qiskit" in r._lazy_loaders
        assert "pennylane" in r._lazy_loaders
        assert "cirq" in r._lazy_loaders

    def test_empty_backends_at_init(self):
        r = PluginRegistry()
        assert len(r._backends) == 0

    def test_list_includes_lazy(self):
        r = PluginRegistry()
        backends = r.list_backends()
        assert "qiskit" in backends
        assert "pennylane" in backends
        assert "cirq" in backends


class TestRegister:
    def test_decorator_register(self):
        r = PluginRegistry()

        @r.register("test_backend")
        class TestRunner:
            def __init__(self, K, omega, **kwargs):
                self.K = K

        assert "test_backend" in r._backends
        assert r._backends["test_backend"] is TestRunner

    def test_register_class(self):
        r = PluginRegistry()

        class MyRunner:
            pass

        r.register_class("my_runner", MyRunner)
        assert "my_runner" in r._backends

    def test_registered_appears_in_list(self):
        r = PluginRegistry()
        r.register_class("custom", type("Custom", (), {}))
        assert "custom" in r.list_backends()


class TestIsAvailable:
    def test_registered_is_available(self):
        r = PluginRegistry()
        r.register_class("mock_be", type("MockBE", (), {}))
        assert r.is_available("mock_be") is True

    def test_qiskit_lazy_available(self):
        r = PluginRegistry()
        assert r.is_available("qiskit") is True

    def test_unknown_not_available(self):
        r = PluginRegistry()
        assert r.is_available("nonexistent_backend") is False

    def test_lazy_import_failure(self):
        r = PluginRegistry()
        r._lazy_loaders["broken"] = ("nonexistent.module.path", "Cls")
        assert r.is_available("broken") is False

    def test_lazy_attribute_error(self):
        r = PluginRegistry()
        r._lazy_loaders["bad_attr"] = (
            "scpn_quantum_control.phase.xy_kuramoto",
            "NonExistentClass",
        )
        assert r.is_available("bad_attr") is False


class TestGetRunner:
    def test_registered_runner(self):
        r = PluginRegistry()

        class FakeRunner:
            def __init__(self, K, omega, **kwargs):
                self.K = K
                self.omega = omega

        r.register_class("fake", FakeRunner)
        K = np.eye(4)
        omega = np.ones(4)
        runner = r.get_runner("fake", K, omega)
        assert isinstance(runner, FakeRunner)
        np.testing.assert_array_equal(runner.K, K)

    def test_qiskit_lazy_runner(self):
        """Qiskit lazy loading uses n_qubits as first arg."""
        r = PluginRegistry()
        K = np.eye(3) * 0.5
        omega = np.ones(3)
        runner = r.get_runner("qiskit", K, omega)
        assert runner is not None

    def test_lazy_caches_class(self):
        """After lazy load, class is cached in _backends."""
        r = PluginRegistry()
        K = np.eye(3) * 0.5
        omega = np.ones(3)
        r.get_runner("qiskit", K, omega)
        assert "qiskit" in r._backends

    def test_non_qiskit_lazy_runner(self):
        """Non-qiskit lazy loading passes (K, omega) directly via actual lazy path."""
        r = PluginRegistry()
        # Register a custom lazy loader pointing to a real importable class
        # PluginRegistry itself takes no args, but we need (K, omega) signature.
        # Use a module with a known class that accepts (K, omega).
        from unittest.mock import MagicMock, patch

        mock_mod = MagicMock()
        mock_cls = MagicMock()
        mock_mod.FakeRunner = mock_cls
        r._lazy_loaders["fake_lazy"] = (
            "fake_lazy_module",
            "FakeRunner",
        )
        with patch("importlib.import_module", return_value=mock_mod):
            r.get_runner("fake_lazy", np.eye(2), np.ones(2))
            mock_cls.assert_called_once()
            assert "fake_lazy" in r._backends

    def test_unknown_raises(self):
        r = PluginRegistry()
        with pytest.raises(ValueError, match="Unknown backend"):
            r.get_runner("nonexistent", np.eye(2), np.ones(2))


class TestAvailableBackends:
    def test_returns_subset_of_list(self):
        r = PluginRegistry()
        available = r.available_backends()
        all_backends = r.list_backends()
        for b in available:
            assert b in all_backends

    def test_qiskit_available(self):
        r = PluginRegistry()
        assert "qiskit" in r.available_backends()


class TestGlobalRegistry:
    def test_singleton_exists(self):
        assert isinstance(registry, PluginRegistry)

    def test_list_backends_works(self):
        backends = registry.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
