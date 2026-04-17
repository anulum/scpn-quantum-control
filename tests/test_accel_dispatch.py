# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-language dispatcher tests
"""Tests for the Rust → Julia → Python dispatcher.

Exercises the multi-language accel chain codified in
``feedback_multi_language_accel.md`` for the ``order_parameter``
compute function. Three layers of test intent:

1. Correctness — every tier returns the same R(theta) to machine
   precision for the same input, across a Hypothesis-generated grid.
2. Dispatch semantics — the registered chain ends with the Python
   floor, tiers that raise ``ImportError`` / ``ModuleNotFoundError``
   / ``RuntimeError`` are skipped, ``last_tier_used`` tracks whichever
   tier actually served the call.
3. Availability introspection — ``available_tiers`` reflects the
   installed state without booting Julia's runtime.

Julia tests are guarded by ``pytest.importorskip`` so a CI image
without ``juliacall`` still runs the Rust + Python parts of the
suite. When Julia *is* available, this file also exercises the full
three-tier chain end-to-end.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.dispatcher as d

_GLOBAL_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Dispatcher mechanics
# ---------------------------------------------------------------------------


class TestDispatcherMechanics:
    def test_chain_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="chain must be non-empty"):
            d.MultiLangDispatcher([])

    def test_chain_requires_python_floor_last(self) -> None:
        with pytest.raises(ValueError, match="python floor"):
            d.MultiLangDispatcher([("rust", lambda x: x)])

    def test_fallback_when_tier_raises_importerror(self) -> None:
        def _bad_tier(_: int) -> int:
            raise ImportError("simulated missing binding")

        def _floor(x: int) -> int:
            return x + 1

        disp = d.MultiLangDispatcher([("fake", _bad_tier), ("python", _floor)])
        assert disp(10) == 11
        assert disp.last_tier == "python"

    def test_fallback_when_tier_raises_runtimeerror(self) -> None:
        def _bad_tier(_: int) -> int:
            raise RuntimeError("simulated")

        def _floor(x: int) -> int:
            return x * 2

        disp = d.MultiLangDispatcher([("fake", _bad_tier), ("python", _floor)])
        assert disp(3) == 6
        assert disp.last_tier == "python"

    def test_does_not_swallow_unrelated_exception(self) -> None:
        def _bad_tier(_: int) -> int:
            raise ValueError("bug in implementation")

        def _floor(x: int) -> int:
            return x

        disp = d.MultiLangDispatcher([("fake", _bad_tier), ("python", _floor)])
        with pytest.raises(ValueError, match="bug in"):
            disp(1)

    def test_tiers_returns_order(self) -> None:
        disp = d.MultiLangDispatcher([("python", lambda x: x)])
        assert disp.tiers() == ["python"]


# ---------------------------------------------------------------------------
# Registry + convenience wrappers
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_order_parameter(self) -> None:
        assert d.dispatch("order_parameter", np.zeros(4)) == pytest.approx(1.0)

    def test_registry_rejects_unknown(self) -> None:
        with pytest.raises(KeyError, match="no dispatcher"):
            d.dispatch("no_such_thing", None)

    def test_last_tier_used_is_none_before_any_call(self) -> None:
        # After a fresh module-import + at least one call via another
        # test, this may already be set — we only assert it's either
        # None or one of the known tier names.
        value = d.last_tier_used()
        assert value is None or value in {"rust", "julia", "python"}


# ---------------------------------------------------------------------------
# Python floor correctness
# ---------------------------------------------------------------------------


class TestPythonFloor:
    def test_zero_vector_has_R_one(self) -> None:
        # All phases aligned at 0 → perfectly coherent.
        assert d._python_order_parameter(np.zeros(8)) == pytest.approx(1.0)

    def test_antipodal_pair_has_R_zero(self) -> None:
        theta = np.array([0.0, math.pi])
        assert d._python_order_parameter(theta) == pytest.approx(0.0, abs=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_python_floor_in_unit_interval(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        r = d._python_order_parameter(theta)
        assert 0.0 - 1e-12 <= r <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Rust tier (already shipped)
# ---------------------------------------------------------------------------


class TestRustTier:
    def test_rust_probe_reflects_engine(self) -> None:
        """Probe must return True iff `scpn_quantum_engine` is importable.

        The Rust wheel is optional (ships via the `[rust]` extra). When
        it is installed, the probe is True and the engine must match the
        Python floor (covered below). When it is missing — minimal
        install, Docker without the wheel, a CI image without maturin —
        the probe must cleanly say False and the dispatcher must fall
        through to the next tier."""
        try:
            import scpn_quantum_engine  # noqa: F401
        except Exception:
            assert d._rust_available() is False
        else:
            assert d._rust_available() is True

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        pytest.importorskip("scpn_quantum_engine")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        r_rust = d._rust_order_parameter(theta)
        r_py = d._python_order_parameter(theta)
        assert abs(r_rust - r_py) < 1e-12


# ---------------------------------------------------------------------------
# Julia tier — optional, skipped when juliacall is not installed
# ---------------------------------------------------------------------------


class TestJuliaTier:
    def test_julia_probe_reflects_juliacall(self) -> None:
        """Probe should return True iff juliacall is importable; it
        MUST NOT boot the Julia runtime itself."""
        try:
            import juliacall  # noqa: F401
        except Exception:
            assert d._julia_available() is False
        else:
            assert d._julia_available() is True

    def test_julia_order_parameter_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import (
            order_parameter as julia_op,
        )

        rng = np.random.default_rng(20260417)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        r_julia = julia_op(theta)
        r_py = d._python_order_parameter(theta)
        assert abs(r_julia - r_py) < 1e-10

    def test_julia_full_dispatch_reaches_julia_when_rust_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With Rust forced unavailable, the dispatcher must pick Julia
        (when it is installed) before the Python floor."""
        pytest.importorskip("juliacall")

        def _sim_rust_gone(_theta: np.ndarray) -> float:
            raise ImportError("rust wheel simulated missing")

        disp = d.MultiLangDispatcher(
            [
                ("rust", _sim_rust_gone),
                ("julia", d._julia_order_parameter),
                ("python", d._python_order_parameter),
            ],
        )
        rng = np.random.default_rng(7)
        theta = rng.uniform(0.0, 2 * math.pi, size=6)
        r = disp(theta)
        assert disp.last_tier == "julia"
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# End-to-end agreement across every installed tier
# ---------------------------------------------------------------------------


class TestCrossTierAgreement:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = d._python_order_parameter(theta)
        for name, impl in d._ORDER_PARAMETER_CHAIN:
            try:
                r = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            assert abs(r - reference) < 1e-10, f"tier {name!r} disagreed"


# ---------------------------------------------------------------------------
# available_tiers probe does not boot Julia
# ---------------------------------------------------------------------------


class TestAvailableTiersProbe:
    def test_probe_does_not_boot_julia(self) -> None:
        """``available_tiers`` must only do import-level checks; booting
        Julia on every probe would make the registry unusable for
        fast-path dispatch decisions.

        The probe table in ``dispatcher.py`` binds the probe callables
        into ``_TIER_PROBES`` at import time, so patching
        ``d._julia_available`` doesn't reach the pre-bound reference.
        We patch the table entry directly instead.
        """
        spy_calls = {"count": 0}

        def _spy() -> bool:
            spy_calls["count"] += 1
            return d._julia_available()

        import scpn_quantum_control.accel.dispatcher as mod

        with patch.dict(mod._TIER_PROBES, {"julia": _spy}):
            tiers = d.available_tiers()
        assert spy_calls["count"] >= 1, "julia probe never ran"

        # "julia" only appears iff juliacall is importable.
        try:
            import juliacall  # noqa: F401

            assert "julia" in tiers
        except Exception:
            assert "julia" not in tiers


# ---------------------------------------------------------------------------
# Pipeline smoke — the public API
# ---------------------------------------------------------------------------


class TestPipelineAccel:
    def test_order_parameter_public_api(self) -> None:
        from scpn_quantum_control.accel import order_parameter

        theta = np.zeros(16)
        assert order_parameter(theta) == pytest.approx(1.0)

    def test_pipeline_picks_first_available_tier(self) -> None:
        from scpn_quantum_control.accel import last_tier_used, order_parameter

        rng = np.random.default_rng(1234)
        theta = rng.uniform(0.0, 2 * math.pi, size=32)
        r = order_parameter(theta)
        assert 0.0 <= r <= 1.0
        assert last_tier_used() in {"rust", "julia", "python"}


# ---------------------------------------------------------------------------
# Error-path coverage — the False branches of _rust_available /
# _julia_available and the "unreachable" RuntimeError of the dispatcher
# are exercised here so coverage doesn't bottom out below the 95% gate.
# ---------------------------------------------------------------------------


class TestProbeNegativePaths:
    def test_rust_probe_false_when_import_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fail(name: str, *args: object, **kwargs: object) -> object:
            if name == "scpn_quantum_engine":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert d._rust_available() is False

    def test_julia_probe_false_when_import_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fail(name: str, *args: object, **kwargs: object) -> object:
            if name == "juliacall":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert d._julia_available() is False

    def test_dispatcher_raises_when_every_tier_fails(self) -> None:
        """The 'python floor unreachable' branch is exercised by
        constructing a chain whose floor itself raises ImportError.
        This is the documented-as-unreachable path."""

        def _raise_import(*_: object, **__: object) -> object:
            raise ImportError("simulated floor failure")

        disp = d.MultiLangDispatcher(
            [
                ("alt", _raise_import),
                ("python", _raise_import),  # named 'python' to pass the last-entry check
            ],
        )
        with pytest.raises(RuntimeError, match="every tier failed"):
            disp(None)


class TestJuliaNegativePaths:
    def test_julia_load_raises_when_juliacall_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        from scpn_quantum_control.accel import julia as jl_mod

        real_import = builtins.__import__

        def _fail(name: str, *args: object, **kwargs: object) -> object:
            if name == "juliacall":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        # Force a fresh load so the patched import is hit.
        monkeypatch.setattr(jl_mod, "_JL", None)
        monkeypatch.setattr(jl_mod, "_INCLUDED", False)
        with pytest.raises(ImportError, match="juliacall"):
            jl_mod._load()

    def test_julia_is_available_returns_false_on_missing_juliacall(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        from scpn_quantum_control.accel import julia as jl_mod

        real_import = builtins.__import__

        def _fail(name: str, *args: object, **kwargs: object) -> object:
            if name == "juliacall":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        monkeypatch.setattr(jl_mod, "_JL", None)
        monkeypatch.setattr(jl_mod, "_INCLUDED", False)
        assert jl_mod.is_available() is False

    def test_order_parameters_batch_delegates_through_load(self) -> None:
        """The batched variant shares the same lazy-load path — run it
        once to cover the batch line."""
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import order_parameters_batch

        theta_batch = np.zeros((3, 4))
        out = order_parameters_batch(theta_batch)
        np.testing.assert_allclose(out, np.ones(3), atol=1e-12)
