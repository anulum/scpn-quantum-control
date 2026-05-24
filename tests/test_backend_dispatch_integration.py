# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend dispatch integration contract tests
"""Integration tests for backend selection, array conversion, plugin registration, and solver dispatch."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return n, K, omega


class TestBackendDispatchIntegration:
    """Backend dispatch should work transparently with solver modules."""

    def test_set_numpy_then_solve(self):
        """Setting numpy backend, then solving should work."""
        from scpn_quantum_control.backend_dispatch import get_backend, set_backend
        from scpn_quantum_control.phase.backend_selector import auto_solve

        set_backend("numpy")
        assert get_backend() == "numpy"

        _, K, omega = _system(4)
        result = auto_solve(K, omega)
        assert result["result"]["ground_energy"] < 0

        # Clean up
        set_backend("numpy")

    def test_available_backends_are_usable(self):
        """All reported available backends should be settable."""
        from scpn_quantum_control.backend_dispatch import (
            available_backends,
            get_backend,
            set_backend,
        )

        for backend in available_backends():
            set_backend(backend)
            assert get_backend() == backend

        # Reset to numpy
        set_backend("numpy")

    def test_to_from_numpy_roundtrip(self):
        """to_numpy(from_numpy(arr)) should be identity."""
        from scpn_quantum_control.backend_dispatch import (
            from_numpy,
            set_backend,
            to_numpy,
        )

        set_backend("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        roundtripped = to_numpy(from_numpy(arr))
        np.testing.assert_array_equal(arr, roundtripped)


class TestPluginRegistryIntegration:
    """Plugin registry should instantiate runners that work with real data."""

    def test_qiskit_runner_available(self):
        """Qiskit should be listed as an available backend."""
        from scpn_quantum_control.hardware.plugin_registry import registry

        assert registry.is_available("qiskit")
        assert "qiskit" in registry.list_backends()

    def test_custom_backend_registration_and_use(self):
        """Register a custom backend, get runner, and call it."""
        from scpn_quantum_control.hardware.plugin_registry import registry

        @registry.register("e2e_test_backend")
        class E2ETestRunner:
            def __init__(self, K, omega, **kwargs):
                self.n = K.shape[0]
                self.K = K
                self.omega = omega

            def run_trotter(self, t=0.1, reps=5):
                return {"energy": -1.0, "n": self.n}

        _, K, omega = _system(4)
        runner = registry.get_runner("e2e_test_backend", K, omega)
        result = runner.run_trotter(t=0.1, reps=3)

        assert result["energy"] == -1.0
        assert result["n"] == 4
