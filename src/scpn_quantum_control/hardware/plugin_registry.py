# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Plugin Registry for Hardware Backends
"""Extensible plugin architecture for quantum backends.

Register and discover hardware backends at runtime instead of hard-coding imports.
Inspired by OpenFermion's plugin system (Google Quantum AI).

Usage:
    from scpn_quantum_control.hardware.plugin_registry import registry

    # List available backends
    registry.list_backends()
    # → ['qiskit', 'pennylane', 'cirq']

    # Get a runner for a specific backend
    runner = registry.get_runner("pennylane", K, omega, device="default.qubit")

    # Register a custom backend
    @registry.register("my_backend")
    class MyRunner:
        def __init__(self, K, omega, **kwargs): ...
        def run_trotter(self, t, reps): ...
"""

from __future__ import annotations

from typing import Any


class PluginRegistry:
    """Registry for quantum hardware backend plugins."""

    def __init__(self) -> None:
        self._backends: dict[str, type] = {}
        self._lazy_loaders: dict[str, tuple[str, str]] = {
            "qiskit": (
                "scpn_quantum_control.phase.xy_kuramoto",
                "QuantumKuramotoSolver",  # init: (n_oscillators, K, omega)
            ),
            "pennylane": (
                "scpn_quantum_control.hardware.pennylane_adapter",
                "PennyLaneRunner",
            ),
            "cirq": (
                "scpn_quantum_control.hardware.cirq_adapter",
                "CirqRunner",
            ),
        }

    def register(self, name: str) -> Any:
        """Decorator to register a backend class.

        @registry.register("my_backend")
        class MyRunner:
            def __init__(self, K, omega, **kwargs): ...
        """

        def decorator(cls: type) -> type:
            self._backends[name] = cls
            return cls

        return decorator

    def register_class(self, name: str, cls: type) -> None:
        """Programmatically register a backend class."""
        self._backends[name] = cls

    def list_backends(self) -> list[str]:
        """List all registered + lazy-loadable backend names."""
        return sorted(set(list(self._backends.keys()) + list(self._lazy_loaders.keys())))

    def is_available(self, name: str) -> bool:
        """Check if a backend is installed and importable."""
        if name in self._backends:
            return True
        if name in self._lazy_loaders:
            module_path, class_name = self._lazy_loaders[name]
            try:
                import importlib

                mod = importlib.import_module(module_path)
                getattr(mod, class_name)
                return True
            except (ImportError, AttributeError):
                return False
        return False

    def get_runner(self, name: str, K: Any, omega: Any, **kwargs: Any) -> Any:
        """Get a runner instance for the specified backend.

        Parameters
        ----------
        name : str
            Backend name ("qiskit", "pennylane", "cirq", or custom).
        K, omega : coupling matrix and frequencies
        **kwargs : backend-specific arguments (e.g. device, shots)

        Returns
        -------
        Runner instance with .run_trotter() and/or .run_vqe() methods.
        """
        if name in self._backends:
            return self._backends[name](K, omega, **kwargs)

        if name in self._lazy_loaders:
            module_path, class_name = self._lazy_loaders[name]
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            self._backends[name] = cls  # cache for next call
            # QuantumKuramotoSolver takes (n, K, omega); others take (K, omega)
            if name == "qiskit":
                return cls(K.shape[0], K, omega, **kwargs)
            return cls(K, omega, **kwargs)

        available = self.list_backends()
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")

    def available_backends(self) -> list[str]:
        """List only backends that are actually importable."""
        return [name for name in self.list_backends() if self.is_available(name)]


# Global singleton
registry = PluginRegistry()
