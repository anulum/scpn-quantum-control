# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend registry tests
"""Tests for the entry-points-based backend registry (audit C10)."""

from __future__ import annotations

import importlib.metadata
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
    def test_fake_backend_satisfies_protocol(self) -> None:
        assert isinstance(_FakeBackend(), be.BackendProtocol)

    def test_object_without_name_does_not_satisfy(self) -> None:
        class Headless:
            def is_available(self) -> bool:
                return True

        assert not isinstance(Headless(), be.BackendProtocol)

    def test_object_without_is_available_does_not_satisfy(self) -> None:
        class Silent:
            name = "silent"

        assert not isinstance(Silent(), be.BackendProtocol)


# ---------------------------------------------------------------------------
# Registration semantics
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_get_roundtrip(self, registry: be.BackendRegistry) -> None:
        registry.register("fake", _fake_factory)
        b = registry.get("fake")
        assert b.name == "fake"
        assert b.is_available() is True

    def test_register_rejects_empty_name(self, registry: be.BackendRegistry) -> None:
        with pytest.raises(be.BackendRegistrationError):
            registry.register("", _fake_factory)

    def test_register_rejects_non_callable(self, registry: be.BackendRegistry) -> None:
        with pytest.raises(be.BackendRegistrationError):
            registry.register("bad", "not a callable")  # type: ignore[arg-type]

    def test_register_same_factory_is_idempotent(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        registry.register("fake", _fake_factory)
        registry.register("fake", _fake_factory)  # must not raise
        assert registry.names() == ["fake"]

    def test_register_different_factory_raises(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        registry.register("fake", _fake_factory)
        with pytest.raises(be.BackendRegistrationError, match="already registered"):
            registry.register("fake", lambda: _FakeBackend(name="other"))

    def test_get_unknown_raises(self, registry: be.BackendRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown backend"):
            registry.get("no-such-thing")

    def test_get_rejects_protocol_violating_factory(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        class NotABackend:
            pass

        registry.register("broken", NotABackend)
        with pytest.raises(be.BackendRegistrationError, match="does not satisfy"):
            registry.get("broken")

    def test_unregister_removes(self, registry: be.BackendRegistry) -> None:
        registry.register("fake", _fake_factory)
        assert "fake" in registry.names()
        registry.unregister("fake")
        assert "fake" not in registry.names()

    def test_unregister_missing_is_silent(self, registry: be.BackendRegistry) -> None:
        registry.unregister("never-there")  # must not raise

    def test_clear_empties(self, registry: be.BackendRegistry) -> None:
        registry.register("a", _fake_factory)
        registry.register("b", _fake_factory)
        registry.clear()
        assert registry.names() == []

    def test_names_preserve_insertion_order(
        self,
        registry: be.BackendRegistry,
    ) -> None:
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
    def test_discover_loads_entry_points(self, registry: be.BackendRegistry) -> None:
        eps = [_make_ep("plugin_a", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            loaded = registry.discover()
        assert "plugin_a" in loaded
        assert "plugin_a" in registry.names()

    def test_discover_is_one_shot_unless_forced(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        eps = [_make_ep("plugin_a", _fake_factory)]
        with patch.object(importlib.metadata, "entry_points", return_value=eps):
            first = registry.discover()
            second = registry.discover()
            assert first == ["plugin_a"]
            assert second == []

    def test_discover_force_reloads(self, registry: be.BackendRegistry) -> None:
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

    def test_discover_skips_invalid_name(
        self,
        registry: be.BackendRegistry,
    ) -> None:
        """A plugin whose factory is not callable must be logged-and-skipped,
        not registered; other plugins in the same discovery pass must
        still land in the registry."""

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


# ---------------------------------------------------------------------------
# Module-level convenience wrappers + built-ins
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_built_in_backends_are_present(self) -> None:
        reg = be.get_registry()
        assert "qiskit_ibm" in reg.names()
        assert "pennylane" in reg.names()

    def test_qiskit_ibm_backend_satisfies_protocol(self) -> None:
        b = be.get_backend("qiskit_ibm")
        assert b.name == "qiskit_ibm"
        assert isinstance(b.is_available(), bool)

    def test_pennylane_backend_satisfies_protocol(self) -> None:
        b = be.get_backend("pennylane")
        assert b.name == "pennylane"
        assert isinstance(b.is_available(), bool)

    def test_list_backends_auto_discover_false(self) -> None:
        names = be.list_backends(auto_discover=False)
        assert "qiskit_ibm" in names

    def test_register_backend_wrapper(self) -> None:
        unique = "xvendor_backend_for_test"
        be.register_backend(unique, _fake_factory)
        try:
            assert be.get_backend(unique).name == "fake"
        finally:
            be.unregister_backend(unique)


# ---------------------------------------------------------------------------
# Pipeline smoke — registry should be the discovery entry point for
# hardware code paths. This test proves the mechanism exists and is
# hooked up to the advertised entry-point group.
# ---------------------------------------------------------------------------


class TestPipelineBackendDiscovery:
    def test_entry_point_group_exact(self) -> None:
        assert be.ENTRY_POINT_GROUP == "scpn_quantum_control.backends"

    def test_pipeline_register_then_run(self) -> None:
        """Simulate a third-party plugin round-trip: register, fetch,
        check availability, unregister. This is the exact contract a
        downstream integrator depends on."""
        name = "xplugin_roundtrip"
        be.register_backend(name, _fake_factory)
        try:
            b = be.get_backend(name)
            assert b.is_available()
            assert name in be.list_backends(auto_discover=False)
        finally:
            be.unregister_backend(name)
        assert name not in be.list_backends(auto_discover=False)
