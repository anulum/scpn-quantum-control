# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend registry tests
"""Tests for the entry-points-based backend registry (audit C10)."""

from __future__ import annotations

import builtins
import importlib.metadata
from collections.abc import Mapping, Sequence
from types import ModuleType
from unittest.mock import patch

import pytest

from scpn_quantum_control.hardware import backends as be

# ---------------------------------------------------------------------------
# Fixtures — isolated registry instance per test
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> be.BackendRegistry:
    """Return a fresh registry. Avoids polluting the process-wide one."""
    return be.BackendRegistry()


class _FakeBackend:
    """Minimal backend that satisfies :class:`BackendProtocol`."""

    def __init__(self, name: str = "fake", available: bool = True) -> None:
        self.name = name
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_factory() -> _FakeBackend:
    return _FakeBackend()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    """The backend protocol."""

    def test_fake_backend_satisfies_protocol(self) -> None:
        """A fake backend satisfies the protocol."""
        assert isinstance(_FakeBackend(), be.BackendProtocol)

    def test_object_without_name_does_not_satisfy(self) -> None:
        """An object without a name does not satisfy it."""

        class Headless:
            def is_available(self) -> bool:
                return True

        assert not isinstance(Headless(), be.BackendProtocol)

    def test_object_without_is_available_does_not_satisfy(self) -> None:
        """An object without is_available does not satisfy it."""

        class Silent:
            name = "silent"

        assert not isinstance(Silent(), be.BackendProtocol)


# ---------------------------------------------------------------------------
# Registration semantics
# ---------------------------------------------------------------------------


class TestRegistration:
    """Registering backends."""

    def test_register_and_get_roundtrip(self, registry: be.BackendRegistry) -> None:
        """A registered backend is returned by get."""
        registry.register("fake", _fake_factory)
        b = registry.get("fake")
        assert b.name == "fake"
        assert b.is_available() is True

    def test_register_rejects_empty_name(self, registry: be.BackendRegistry) -> None:
        """An empty name is refused."""
        with pytest.raises(be.BackendRegistrationError):
            registry.register("", _fake_factory)

    def test_register_rejects_non_callable(self, registry: be.BackendRegistry) -> None:
        """A factory that is not callable is refused."""
        with pytest.raises(be.BackendRegistrationError):
            registry.register("bad", "not a callable")  # type: ignore[arg-type]

    def test_register_same_factory_is_idempotent(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """Registering the same factory twice is harmless."""
        registry.register("fake", _fake_factory)
        registry.register("fake", _fake_factory)  # must not raise
        assert registry.names() == ["fake"]

    def test_register_different_factory_raises(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """Registering another factory under a taken name is refused."""
        registry.register("fake", _fake_factory)
        with pytest.raises(be.BackendRegistrationError, match="already registered"):
            registry.register("fake", lambda: _FakeBackend(name="other"))

    def test_get_unknown_raises(self, registry: be.BackendRegistry) -> None:
        """An unknown backend raises and lists the known ones."""
        registry.register("z_backend", _fake_factory)
        registry.register("a_backend", _fake_factory)

        with pytest.raises(KeyError, match="Unknown backend"):
            registry.get("no-such-thing")
        with pytest.raises(KeyError) as exc_info:
            registry.get("still-missing")

        assert "['a_backend', 'z_backend']" in str(exc_info.value)

    def test_get_rejects_protocol_violating_factory(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """A factory that breaks the protocol is refused."""

        class NotABackend:
            pass

        registry.register("broken", NotABackend)  # type: ignore[arg-type] # deliberate protocol violation, refusal asserted below
        with pytest.raises(be.BackendRegistrationError, match="does not satisfy"):
            registry.get("broken")

    def test_unregister_removes(self, registry: be.BackendRegistry) -> None:
        """Unregistering removes the backend."""
        registry.register("fake", _fake_factory)
        assert "fake" in registry.names()
        registry.unregister("fake")
        assert "fake" not in registry.names()

    def test_unregister_missing_is_silent(self, registry: be.BackendRegistry) -> None:
        """Unregistering an unknown name is harmless."""
        before = registry.names()
        registry.unregister("never-there")  # must not raise
        assert registry.names() == before

    def test_clear_empties(self, registry: be.BackendRegistry) -> None:
        """Clearing empties the registry."""
        registry.register("a", _fake_factory)
        registry.register("b", _fake_factory)
        with patch.object(importlib.metadata, "entry_points", return_value=[]):
            registry.discover()
        registry.clear()
        assert registry.names() == []
        with patch.object(importlib.metadata, "entry_points", return_value=[]):
            assert registry.discover() == []

    def test_names_preserve_insertion_order(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """Names keep their insertion order."""
        registry.register("b", _fake_factory)
        registry.register("a", _fake_factory)
        assert registry.names() == ["b", "a"]


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


def _make_ep(name: str, target: object) -> object:
    """Build a stub EntryPoint-like object."""

    class _EP:
        def __init__(self) -> None:
            self.name = name
            self.value = f"<test:{name}>"

        def load(self) -> object:
            return target

    return _EP()


class TestDiscovery:
    """Discovering backends through entry points."""

    def test_discover_loads_entry_points(self, registry: be.BackendRegistry) -> None:
        """Discovery loads entry-point plugins."""
        eps = [_make_ep("plugin_a", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            loaded = registry.discover()
        assert "plugin_a" in loaded
        assert "plugin_a" in registry.names()

    def test_discover_is_one_shot_unless_forced(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """Discovery runs once unless forced."""
        eps = [_make_ep("plugin_a", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            first = registry.discover()
            second = registry.discover()
            assert first == ["plugin_a"]
            assert second == []

    def test_discover_force_reloads(self, registry: be.BackendRegistry) -> None:
        """Forced discovery reloads the plugins."""
        eps = [_make_ep("plugin_a", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            registry.discover()
            # Already registered — force=True should skip silent-collision.
            forced = registry.discover(force=True)
        assert forced == ["plugin_a"]

    def test_discover_skips_broken_plugin(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """A broken plugin does not block the good ones."""

        def _boom() -> None:
            raise RuntimeError("third-party plugin crashed at import")

        class _BrokenEP:
            name = "broken"
            value = "<test:broken>"

            def load(self) -> object:
                raise RuntimeError("load() failed")

        eps = [_BrokenEP(), _make_ep("good", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            loaded = registry.discover()
        assert loaded == ["good"], "broken plugin must not block the good one"
        assert "broken" not in registry.names()

    def test_failed_discovery_still_marks_pass_complete(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """A failed discovery still counts as done."""

        class _BrokenEP:
            name = "broken"
            value = "<test:broken>"

            def load(self) -> object:
                raise RuntimeError("load() failed")

        with patch.object(importlib.metadata, "entry_points", return_value=[_BrokenEP()]):
            assert registry.discover() == []
            assert registry.discover() == []

    def test_discover_skips_invalid_name(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """A plugin whose factory is not callable is logged and skipped, not registered.

        Other plugins in the same discovery pass still land in the registry.
        """

        class _NonCallableEP:
            name = "nocall"
            value = "<test:nocall>"

            def load(self) -> object:
                return "not callable"

        eps = [_NonCallableEP(), _make_ep("good", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            loaded = registry.discover()
        assert "good" in loaded
        assert "nocall" not in loaded

    def test_module_discover_wrapper_uses_process_registry(self) -> None:
        """Module wrapper discovers plugins through the singleton registry."""
        name = "xwrapper_plugin"
        eps = [_make_ep(name, _fake_factory)]
        reg = be.get_registry()
        reg.unregister(name)
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            loaded = be.discover_backends(force=True)
        try:
            assert loaded == [name]
            assert be.get_backend(name).is_available() is True
        finally:
            reg.unregister(name)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers + built-ins
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """The module's global registry."""

    def test_built_in_backends_are_present(self) -> None:
        """The built-in backends are registered."""
        reg = be.get_registry()
        assert "qiskit_ibm" in reg.names()
        assert "qiskit_aer" in reg.names()
        assert "cirq" in reg.names()
        assert "braket" in reg.names()
        assert "pennylane" in reg.names()
        assert "iqm" in reg.names()

    def test_qiskit_ibm_backend_satisfies_protocol(self) -> None:
        """The IBM backend satisfies the protocol."""
        b = be.get_backend("qiskit_ibm")
        assert b.name == "qiskit_ibm"
        assert isinstance(b.is_available(), bool)

    def test_pennylane_backend_satisfies_protocol(self) -> None:
        """The PennyLane backend satisfies the protocol."""
        b = be.get_backend("pennylane")
        assert b.name == "pennylane"
        assert isinstance(b.is_available(), bool)

    @pytest.mark.parametrize(
        ("name", "provider", "requires_approval"),
        [
            ("qiskit_ibm", "ibm_quantum", True),
            ("qiskit_aer", "local_qiskit_aer", False),
            ("cirq", "google_cirq", False),
            ("braket", "aws_braket", True),
            ("pennylane", "pennylane", False),
            ("iqm", "iqm", True),
        ],
    )
    def test_builtin_quantum_backend_descriptors(
        self,
        name: str,
        provider: str,
        requires_approval: bool,
    ) -> None:
        """Built-in backends carry their documented descriptors."""
        descriptor = be.describe_backend(name)
        assert descriptor.name == name
        assert descriptor.provider == provider
        assert descriptor.submit_requires_approval is requires_approval
        assert descriptor.available == be.get_backend(name).is_available()
        assert isinstance(descriptor.capabilities, tuple)
        assert "kuramoto_xy" in descriptor.workloads

    def test_cloud_descriptors_never_mark_auto_submit_safe(self) -> None:
        """Cloud backends always require approval to submit."""
        cloud = [
            be.describe_backend("qiskit_ibm"),
            be.describe_backend("braket"),
            be.describe_backend("iqm"),
        ]
        assert all(d.can_submit for d in cloud)
        assert all(d.submit_requires_approval for d in cloud)

    def test_local_simulator_descriptors_are_non_submit_paths(self) -> None:
        """Local simulators simulate and never submit."""
        local = [be.describe_backend("qiskit_aer"), be.describe_backend("cirq")]
        assert all(d.can_simulate for d in local)
        assert all(not d.can_submit for d in local)
        assert all(not d.submit_requires_approval for d in local)

    def test_list_quantum_backends_returns_sorted_descriptors(self) -> None:
        """Backend descriptors are listed in name order."""
        descriptors = be.list_quantum_backends(auto_discover=False)
        names = [d.name for d in descriptors]
        assert names == sorted(names)
        assert {"qiskit_ibm", "qiskit_aer", "cirq", "braket", "pennylane", "iqm"} <= set(names)

    def test_legacy_plugin_gets_conservative_descriptor(self) -> None:
        """A legacy plugin gets a conservative descriptor."""
        name = "legacy_descriptor_test"
        be.register_backend(name, _fake_factory)
        try:
            descriptor = be.describe_backend(name)
            assert descriptor.name == name
            assert descriptor.provider == "external"
            assert descriptor.execution_mode == "unknown"
            assert descriptor.capabilities == ()
            assert descriptor.can_submit is False
            assert descriptor.submit_requires_approval is True
        finally:
            be.unregister_backend(name)

    def test_list_backends_auto_discover_false(self) -> None:
        """Listing without discovery still shows the built-ins."""
        names = be.list_backends(auto_discover=False)
        assert "qiskit_ibm" in names

    def test_list_backends_auto_discovers_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Listing discovers plugins by default."""
        calls: list[bool] = []

        def mark_discovered(*, force: bool = False) -> list[str]:
            calls.append(force)
            return []

        monkeypatch.setattr(be, "discover_backends", mark_discovered)
        names = be.list_backends()
        assert calls == [False]
        assert "qiskit_ibm" in names

    def test_register_backend_wrapper(self) -> None:
        """The module-level register wrapper registers a backend."""
        unique = "xvendor_backend_for_test"
        be.register_backend(unique, _fake_factory)
        try:
            assert be.get_backend(unique).name == "fake"
        finally:
            be.unregister_backend(unique)

    def test_qiskit_ibm_backend_unavailable_when_sdk_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the IBM SDK the IBM backend is unavailable."""
        original_import = builtins.__import__

        def blocked_import(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == "qiskit_ibm_runtime":
                raise ImportError("runtime SDK absent")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        assert be.qiskit_ibm_factory().is_available() is False


# ---------------------------------------------------------------------------
# Pipeline smoke — registry should be the discovery entry point for
# hardware code paths. This test proves the mechanism exists and is
# hooked up to the advertised entry-point group.
# ---------------------------------------------------------------------------


class TestPipelineBackendDiscovery:
    """Backend discovery in a pipeline."""

    def test_entry_point_group_exact(self) -> None:
        """The entry-point group name is fixed."""
        assert be.ENTRY_POINT_GROUP == "scpn_quantum_control.backends"

    def test_pipeline_register_then_run(self) -> None:
        """Run a third-party plugin round trip: register, fetch, check availability, unregister.

        This is the exact contract a downstream integrator depends on.
        """
        name = "xplugin_roundtrip"
        be.register_backend(name, _fake_factory)
        try:
            b = be.get_backend(name)
            assert b.is_available()
            assert name in be.list_backends(auto_discover=False)
        finally:
            be.unregister_backend(name)
        assert name not in be.list_backends(auto_discover=False)
