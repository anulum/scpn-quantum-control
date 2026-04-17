# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend plugin registry
"""Plugin / backend extension API.

Closes audit item C10. Third parties may now register additional quantum
backends without editing this repository by declaring an entry point in
the ``scpn_quantum_control.backends`` group:

.. code-block:: toml

    [project.entry-points."scpn_quantum_control.backends"]
    acme_trapped_ion = "acme_plugin:AcmeBackend"

The entry-point target must be a zero-argument callable or a class that
returns an object satisfying the :class:`BackendProtocol` interface when
instantiated.

Internal backends (Qiskit runtime, PennyLane) register themselves via
this repository's own ``pyproject.toml`` entry points; they are loaded
the same way every third-party backend is, which means a broken plugin
cannot take down the rest of the registry.

Discovery is lazy: call :func:`discover_backends` once per process. The
module also exposes a manual :func:`register_backend` hatch for tests
and notebooks that want to exercise a specific class without round-trip
through entry points.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "scpn_quantum_control.backends"

logger = logging.getLogger(__name__)


@runtime_checkable
class BackendProtocol(Protocol):
    """Minimum contract every registered backend must satisfy.

    The registry intentionally asks for very little — enough to identify
    the backend and check whether it is runnable in the current
    environment. Anything beyond that (circuit submission, retrieval
    semantics) is delegated to adapter-specific subtypes.
    """

    name: str

    def is_available(self) -> bool:
        """True iff the backend can run in the current environment."""
        ...


BackendFactory = Callable[[], BackendProtocol]


class BackendRegistrationError(RuntimeError):
    """Raised when a backend fails to load or violates the contract."""


class BackendRegistry:
    """In-memory backend registry.

    Backends are keyed by their ``name`` attribute. Registration is
    idempotent (same-class re-registration is a no-op); registering a
    different factory under an existing name raises
    :class:`BackendRegistrationError` so silent overrides do not happen
    in production.
    """

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._discovered: bool = False

    # -- public API ---------------------------------------------------------

    def register(self, name: str, factory: BackendFactory) -> None:
        """Register a backend factory under ``name``."""
        if not isinstance(name, str) or not name:
            raise BackendRegistrationError(f"Backend name must be a non-empty str, got {name!r}")
        if not callable(factory):
            raise BackendRegistrationError(
                f"Backend factory for {name!r} is not callable",
            )
        existing = self._factories.get(name)
        if existing is not None and existing is not factory:
            raise BackendRegistrationError(
                f"Backend {name!r} already registered with a different factory",
            )
        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        """Remove a backend from the registry (useful for tests)."""
        self._factories.pop(name, None)

    def get(self, name: str) -> BackendProtocol:
        """Instantiate the backend registered under ``name``.

        Raises :class:`KeyError` if no such backend is registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"Unknown backend {name!r}; known backends: {sorted(self._factories)}",
            )
        backend = factory()
        if not isinstance(backend, BackendProtocol):
            raise BackendRegistrationError(
                f"Backend {name!r} factory returned {type(backend).__name__}, "
                "which does not satisfy BackendProtocol",
            )
        return backend

    def names(self) -> list[str]:
        """All registered backend names in insertion order."""
        return list(self._factories)

    def clear(self) -> None:
        """Wipe the registry (tests only)."""
        self._factories.clear()
        self._discovered = False

    # -- discovery ----------------------------------------------------------

    def discover(self, *, force: bool = False) -> list[str]:
        """Load every backend advertised under the entry-point group.

        Returns the list of names successfully registered during this
        call. A plugin that raises at load time is logged and skipped —
        one broken plugin never blocks the rest.
        """
        if self._discovered and not force:
            return []

        loaded: list[str] = []
        # ``requires-python = ">=3.10"`` — the ``group=`` kwarg is always
        # available, no legacy fallback needed.
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)

        for ep in entries:
            try:
                factory = ep.load()
            except Exception as exc:  # pragma: no cover - diagnostic branch
                logger.warning(
                    "Failed to load backend plugin %r from %s: %s",
                    ep.name,
                    getattr(ep, "value", "<unknown>"),
                    exc,
                )
                continue
            try:
                self.register(ep.name, factory)
                loaded.append(ep.name)
            except BackendRegistrationError as exc:
                logger.warning("Rejected backend %r: %s", ep.name, exc)

        self._discovered = True
        return loaded


# Module-level singleton — tests can reach in via :func:`get_registry`.
_registry = BackendRegistry()


def get_registry() -> BackendRegistry:
    """Return the process-wide backend registry."""
    return _registry


def register_backend(name: str, factory: BackendFactory) -> None:
    """Convenience wrapper for :meth:`BackendRegistry.register`."""
    _registry.register(name, factory)


def unregister_backend(name: str) -> None:
    """Convenience wrapper for :meth:`BackendRegistry.unregister`."""
    _registry.unregister(name)


def get_backend(name: str) -> BackendProtocol:
    """Convenience wrapper for :meth:`BackendRegistry.get`."""
    return _registry.get(name)


def discover_backends(*, force: bool = False) -> list[str]:
    """Convenience wrapper for :meth:`BackendRegistry.discover`."""
    return _registry.discover(force=force)


def list_backends(*, auto_discover: bool = True) -> list[str]:
    """Return every known backend name.

    When ``auto_discover`` is True (default) this triggers a one-shot
    entry-point scan so users do not have to remember to call
    :func:`discover_backends` first.
    """
    if auto_discover:
        discover_backends()
    return _registry.names()


# ---------------------------------------------------------------------------
# Built-in backend entries — zero-argument factories that the repository
# registers via its own pyproject.toml entry points. Third-party plugins
# follow the same pattern.
# ---------------------------------------------------------------------------


class _QiskitIBMBackend:
    """Built-in backend for IBM Quantum hardware via qiskit-ibm-runtime.

    ``is_available`` checks whether ``qiskit_ibm_runtime`` is importable
    — it deliberately does not authenticate because that would require a
    token and network access at registry-lookup time.
    """

    name = "qiskit_ibm"

    def is_available(self) -> bool:
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except Exception:
            return False
        return True


class _PennyLaneBackend:
    """Built-in backend for the PennyLane adapter."""

    name = "pennylane"

    def is_available(self) -> bool:
        from .pennylane_adapter import is_pennylane_available

        return is_pennylane_available()


def qiskit_ibm_factory() -> BackendProtocol:
    """Entry-point target for the IBM runtime backend."""
    return _QiskitIBMBackend()


def pennylane_factory() -> BackendProtocol:
    """Entry-point target for the PennyLane backend."""
    return _PennyLaneBackend()


# Pre-register the built-ins so the registry is useful even on a source
# checkout where the entry points haven't been installed.
_registry.register("qiskit_ibm", qiskit_ibm_factory)
_registry.register("pennylane", pennylane_factory)


__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendFactory",
    "BackendProtocol",
    "BackendRegistrationError",
    "BackendRegistry",
    "discover_backends",
    "get_backend",
    "get_registry",
    "list_backends",
    "pennylane_factory",
    "qiskit_ibm_factory",
    "register_backend",
    "unregister_backend",
]
