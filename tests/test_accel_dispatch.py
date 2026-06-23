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
import sys
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.daido_observables as da_obs
import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.mean_phase_observables as mp_obs
import scpn_quantum_control.accel.order_parameter_observables as op_obs

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
        value = op_obs.last_tier_used()
        assert value is None or value in {"rust", "julia", "python"}


class TestAccelerationPackageContracts:
    def test_random_state_seed_is_normalised_and_reproducible(self) -> None:
        from scpn_quantum_control.accel import rust_random_state

        first = rust_random_state(3, seed=123)
        second = rust_random_state(3, seed=123)
        different_seed = rust_random_state(3, seed=124)

        assert first.shape == (8,)
        assert np.iscomplexobj(first)
        assert np.linalg.norm(first) == pytest.approx(1.0)
        np.testing.assert_allclose(first, second)
        assert not np.allclose(first, different_seed)

    def test_feedback_correction_scales_matrix_without_mutating_input(self) -> None:
        from scpn_quantum_control.accel.rust_kuramoto_classical import apply_feedback_correction

        K_nm = np.array([[0.0, 0.2], [0.2, 0.0]])
        corrected = apply_feedback_correction(K_nm, asymmetry=0.5, sync_order=0.4)

        np.testing.assert_allclose(corrected, K_nm * 1.02)
        np.testing.assert_allclose(K_nm, np.array([[0.0, 0.2], [0.2, 0.0]]))

    def test_large_n_proxy_records_inputs_and_sync_response(self) -> None:
        from scpn_quantum_control.accel.rust_kuramoto_classical import run_large_n

        inactive = run_large_n(N=64, K=0.3, lambda_fim=0.0, delta=0.1, steps=20)
        active = run_large_n(N=64, K=0.3, lambda_fim=0.2, delta=0.1, steps=20)

        assert inactive == {"sync_order": 0.1, "lambda_fim": 0.0, "K": 0.3, "N": 64}
        assert active == {"sync_order": 0.85, "lambda_fim": 0.2, "K": 0.3, "N": 64}
        assert active["sync_order"] > inactive["sync_order"]


# ---------------------------------------------------------------------------
# Python floor correctness
# ---------------------------------------------------------------------------


class TestPythonFloor:
    def test_zero_vector_has_R_one(self) -> None:
        # All phases aligned at 0 → perfectly coherent.
        assert op_obs._python_order_parameter(np.zeros(8)) == pytest.approx(1.0)

    def test_antipodal_pair_has_R_zero(self) -> None:
        theta = np.array([0.0, math.pi])
        assert op_obs._python_order_parameter(theta) == pytest.approx(0.0, abs=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_python_floor_in_unit_interval(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        r = op_obs._python_order_parameter(theta)
        assert 0.0 - 1e-12 <= r <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Rust tier (already shipped)
# ---------------------------------------------------------------------------


class TestRustTier:
    def test_optional_rust_engine_absence_is_optional(self) -> None:
        from scpn_quantum_control.accel.rust_import import optional_rust_engine

        with patch.dict(sys.modules, {"scpn_quantum_engine": None}):
            assert optional_rust_engine() is None

    def test_optional_rust_engine_broken_dependency_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scpn_quantum_control.accel.rust_import import optional_rust_engine

        real_import = __import__

        def broken_engine_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scpn_quantum_engine":
                raise ModuleNotFoundError("missing native dependency", name="libstdc++")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", broken_engine_import)
        with pytest.raises(ModuleNotFoundError, match="missing native dependency"):
            optional_rust_engine()

    def test_rust_probe_propagates_broken_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_optional = d.optional_rust_engine

        def broken_engine() -> object:
            raise ModuleNotFoundError("broken installed engine", name="libstdc++")

        monkeypatch.setattr(d, "optional_rust_engine", broken_engine)
        with pytest.raises(ModuleNotFoundError, match="broken installed engine"):
            d._rust_available()
        monkeypatch.setattr(d, "optional_rust_engine", real_optional)

    def test_rust_probe_reflects_usable_engine(self) -> None:
        """Probe must return True iff `order_parameter` is callable.

        The Rust wheel is optional (ships via the `[rust]` extra). When
        it is installed, the probe is True and the engine must match the
        Python floor (covered below). When it is missing — minimal
        install, Docker without the wheel, a CI image without maturin —
        or present without this symbol, the probe must cleanly say False
        and the dispatcher must fall through to the next tier."""
        try:
            import scpn_quantum_engine
        except Exception:
            assert d._rust_available() is False
        else:
            assert d._rust_available() is callable(
                getattr(scpn_quantum_engine, "order_parameter", None)
            )

    def test_partial_rust_engine_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", op_obs._rust_order_parameter),
                ("python", op_obs._python_order_parameter),
            ],
        )
        assert disp(np.zeros(4)) == pytest.approx(1.0)
        assert disp.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "order_parameter", None)):
            pytest.skip("scpn_quantum_engine.order_parameter unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        r_rust = op_obs._rust_order_parameter(theta)
        r_py = op_obs._python_order_parameter(theta)
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
        r_py = op_obs._python_order_parameter(theta)
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
                ("julia", op_obs._julia_order_parameter),
                ("python", op_obs._python_order_parameter),
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
        reference = op_obs._python_order_parameter(theta)
        for name, impl in op_obs._ORDER_PARAMETER_CHAIN:
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
                raise ModuleNotFoundError("simulated absence", name="scpn_quantum_engine")
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


# ---------------------------------------------------------------------------
# Order parameter gradient — analytic floor, parity, and dispatch
# ---------------------------------------------------------------------------


def _order_parameter_value(theta: np.ndarray) -> float:
    """Reference scalar order parameter ``R = |<exp(i theta)>|``."""
    return float(abs(np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))))


def _finite_difference_gradient(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the order parameter for cross-checking."""
    grad = np.zeros(theta.size, dtype=np.float64)
    for j in range(theta.size):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        grad[j] = (_order_parameter_value(plus) - _order_parameter_value(minus)) / (2.0 * step)
    return grad


class TestPythonGradientFloor:
    def test_matches_synchronisation_force_identity(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=17)
        grad = op_obs._python_order_parameter_gradient(theta)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        psi = math.atan2(sin_mean, cos_mean)
        identity = np.sin(psi - theta) / theta.size
        np.testing.assert_allclose(grad, identity, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-3:
            return  # near-incoherent: ill-conditioned, excluded from the FD check
        grad = op_obs._python_order_parameter_gradient(theta)
        np.testing.assert_allclose(grad, _finite_difference_gradient(theta), atol=1e-6)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_sums_to_zero(self, n: int, seed: int) -> None:
        # A global phase shift leaves R invariant, so the gradient sums to zero.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert abs(float(np.sum(op_obs._python_order_parameter_gradient(theta)))) < 1e-12

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_is_bounded_by_inverse_n(self, n: int, seed: int) -> None:
        # |partial R / partial theta_j| = |sin(psi - theta_j)| / N <= 1/N everywhere,
        # including arbitrarily close to the incoherent state.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        grad = op_obs._python_order_parameter_gradient(theta)
        assert np.all(np.abs(grad) <= 1.0 / n + 1e-12)

    def test_aligned_state_has_zero_gradient(self) -> None:
        grad = op_obs._python_order_parameter_gradient(np.full(8, 0.7))
        np.testing.assert_allclose(grad, np.zeros(8), atol=1e-15)

    def test_single_oscillator_has_zero_gradient(self) -> None:
        grad = op_obs._python_order_parameter_gradient(np.array([2.7]))
        assert grad.shape == (1,)
        assert abs(float(grad[0])) < 1e-15

    def test_empty_input_returns_empty(self) -> None:
        assert op_obs._python_order_parameter_gradient(np.array([])).shape == (0,)

    def test_exact_incoherent_state_returns_zero_subgradient(self) -> None:
        # [0, pi, 0, -pi] gives C = S = 0 exactly (cos(+-pi) = -1; sin(pi), sin(-pi)
        # are IEEE-exact negatives that cancel), so R = 0 and the zero subgradient
        # is returned rather than a 0/0 NaN.
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        assert float(np.hypot(np.mean(np.cos(theta)), np.mean(np.sin(theta)))) == 0.0
        grad = op_obs._python_order_parameter_gradient(theta)
        np.testing.assert_array_equal(grad, np.zeros(4))


class TestRustGradientTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "order_parameter_gradient", None)):
            pytest.skip("scpn_quantum_engine.order_parameter_gradient unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        rust = op_obs._rust_order_parameter_gradient(theta)
        floor = op_obs._python_order_parameter_gradient(theta)
        np.testing.assert_allclose(rust, floor, atol=1e-12)

    def test_partial_rust_engine_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", op_obs._rust_order_parameter_gradient),
                ("python", op_obs._python_order_parameter_gradient),
            ],
        )
        out = disp(np.full(4, 0.7))
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-15)
        assert disp.last_tier == "python"

    def test_rust_engine_absence_raises_module_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            op_obs._rust_order_parameter_gradient(np.zeros(3))


class TestJuliaGradientTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import order_parameter_gradient as julia_grad

        rng = np.random.default_rng(20260622)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        np.testing.assert_allclose(
            julia_grad(theta),
            op_obs._python_order_parameter_gradient(theta),
            atol=1e-10,
        )

    def test_julia_full_dispatch_reaches_julia_when_rust_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("juliacall")

        def _sim_rust_gone(_theta: np.ndarray) -> np.ndarray:
            raise ImportError("rust wheel simulated missing")

        disp = d.MultiLangDispatcher(
            [
                ("rust", _sim_rust_gone),
                ("julia", op_obs._julia_order_parameter_gradient),
                ("python", op_obs._python_order_parameter_gradient),
            ],
        )
        rng = np.random.default_rng(7)
        theta = rng.uniform(0.0, 2 * math.pi, size=6)
        out = disp(theta)
        assert disp.last_tier == "julia"
        np.testing.assert_allclose(out, op_obs._python_order_parameter_gradient(theta), atol=1e-10)


class TestGradientCrossTierAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = op_obs._python_order_parameter_gradient(theta)
        for name, impl in op_obs._ORDER_PARAMETER_GRADIENT_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=f"tier {name!r}")

    def test_registry_contains_gradient(self) -> None:
        out = d.dispatch("order_parameter_gradient", np.full(4, 0.3))
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-15)

    def test_public_api_and_last_gradient_tier(self) -> None:
        from scpn_quantum_control.accel import (
            last_gradient_tier_used,
            order_parameter_gradient,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=24)
        grad = order_parameter_gradient(theta)
        assert grad.shape == (24,)
        assert last_gradient_tier_used() in {"rust", "julia", "python"}

    def test_gradient_chain_ends_with_python_floor(self) -> None:
        assert op_obs._ORDER_PARAMETER_GRADIENT_CHAIN[-1][0] == "python"


class TestRustValueTierAbsence:
    def test_rust_order_parameter_absence_raises_module_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mirrors the gradient-tier absence test so both Rust entry points cover
        # the engine-missing branch deterministically on an engine-present host.
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            op_obs._rust_order_parameter(np.zeros(3))


# ---------------------------------------------------------------------------
# Order parameter Hessian — analytic floor, invariants, parity, and dispatch
# ---------------------------------------------------------------------------


def _finite_difference_hessian(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference Hessian from the analytic gradient floor."""
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            op_obs._python_order_parameter_gradient(plus)
            - op_obs._python_order_parameter_gradient(minus)
        ) / (2.0 * step)
    return out


class TestPythonHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(13)
        theta = rng.uniform(-math.pi, math.pi, size=11)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        aligned = (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)) / magnitude
        expected = np.outer(aligned, aligned) / (theta.size**2 * magnitude) - np.diag(
            aligned / theta.size
        )
        np.testing.assert_allclose(
            op_obs._python_order_parameter_hessian(theta), expected, atol=1e-15
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_is_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = op_obs._python_order_parameter_hessian(theta)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rows_sum_to_zero(self, n: int, seed: int) -> None:
        # A global phase shift leaves r invariant, so each Hessian row sums to zero.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = op_obs._python_order_parameter_hessian(theta)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-2:
            return  # near-incoherent: the second derivative is ill-conditioned
        hessian = op_obs._python_order_parameter_hessian(theta)
        np.testing.assert_allclose(hessian, _finite_difference_hessian(theta), atol=1e-5)

    def test_fully_synchronised_curvature(self) -> None:
        # r = 1: the gradient is zero but the Hessian is 11^T/N^2 - I/N (negative
        # semidefinite, since perfect synchronisation is the maximum of r).
        n = 6
        hessian = op_obs._python_order_parameter_hessian(np.full(n, 0.4))
        expected = np.full((n, n), 1.0 / n**2) - np.eye(n) / n
        np.testing.assert_allclose(hessian, expected, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert np.all(eigenvalues <= 1e-12)

    def test_single_oscillator_is_zero(self) -> None:
        hessian = op_obs._python_order_parameter_hessian(np.array([2.7]))
        assert hessian.shape == (1, 1)
        assert abs(float(hessian[0, 0])) < 1e-15

    def test_empty_input_returns_empty_matrix(self) -> None:
        assert op_obs._python_order_parameter_hessian(np.array([])).shape == (0, 0)

    def test_exact_incoherent_returns_zero_matrix(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        assert float(np.hypot(np.mean(np.cos(theta)), np.mean(np.sin(theta)))) == 0.0
        np.testing.assert_array_equal(
            op_obs._python_order_parameter_hessian(theta), np.zeros((4, 4))
        )


class TestRustHessianTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "order_parameter_hessian", None)):
            pytest.skip("scpn_quantum_engine.order_parameter_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            op_obs._rust_order_parameter_hessian(theta),
            op_obs._python_order_parameter_hessian(theta),
            atol=1e-12,
        )

    def test_rust_engine_absence_raises_module_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            op_obs._rust_order_parameter_hessian(np.zeros(3))

    def test_partial_rust_engine_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", op_obs._rust_order_parameter_hessian),
                ("python", op_obs._python_order_parameter_hessian),
            ],
        )
        out = disp(np.full(4, 0.7))
        np.testing.assert_allclose(
            out, op_obs._python_order_parameter_hessian(np.full(4, 0.7)), atol=0
        )
        assert disp.last_tier == "python"


class TestJuliaHessianTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import order_parameter_hessian as julia_hessian

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        np.testing.assert_allclose(
            julia_hessian(theta),
            op_obs._python_order_parameter_hessian(theta),
            atol=1e-10,
        )


class TestHessianCrossTierAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = op_obs._python_order_parameter_hessian(theta)
        for name, impl in op_obs._ORDER_PARAMETER_HESSIAN_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=f"tier {name!r}")

    def test_registry_contains_hessian(self) -> None:
        out = d.dispatch("order_parameter_hessian", np.full(4, 0.3))
        assert out.shape == (4, 4)

    def test_public_api_and_last_hessian_tier(self) -> None:
        from scpn_quantum_control.accel import last_hessian_tier_used, order_parameter_hessian

        rng = np.random.default_rng(77)
        theta = rng.uniform(0.0, 2 * math.pi, size=20)
        hessian = order_parameter_hessian(theta)
        assert hessian.shape == (20, 20)
        assert last_hessian_tier_used() in {"rust", "julia", "python"}

    def test_hessian_chain_ends_with_python_floor(self) -> None:
        assert op_obs._ORDER_PARAMETER_HESSIAN_CHAIN[-1][0] == "python"


# ---------------------------------------------------------------------------
# Mean phase and its gradient — analytic floor, invariants, parity, dispatch
# ---------------------------------------------------------------------------


def _finite_difference_mean_phase_gradient(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the circular mean phase, unwrapped at ±π."""
    grad = np.zeros(theta.size, dtype=np.float64)
    for j in range(theta.size):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        delta = mp_obs._python_mean_phase(plus) - mp_obs._python_mean_phase(minus)
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        grad[j] = delta / (2.0 * step)
    return grad


class TestPythonMeanPhaseFloor:
    def test_value_matches_atan2(self) -> None:
        rng = np.random.default_rng(31)
        theta = rng.uniform(-math.pi, math.pi, size=13)
        expected = math.atan2(float(np.mean(np.sin(theta))), float(np.mean(np.cos(theta))))
        assert mp_obs._python_mean_phase(theta) == pytest.approx(expected, abs=1e-12)

    def test_single_oscillator_is_identity(self) -> None:
        assert mp_obs._python_mean_phase(np.array([2.7])) == pytest.approx(2.7, abs=1e-12)

    def test_empty_input_is_zero(self) -> None:
        assert mp_obs._python_mean_phase(np.array([])) == 0.0

    def test_gradient_matches_closed_form(self) -> None:
        rng = np.random.default_rng(17)
        theta = rng.uniform(-math.pi, math.pi, size=15)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        expected = (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)) / (
            theta.size * magnitude**2
        )
        np.testing.assert_allclose(mp_obs._python_mean_phase_gradient(theta), expected, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_sums_to_one(self, n: int, seed: int) -> None:
        # A global phase shift advances ψ identically, so the gradient sums to one.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert float(np.sum(mp_obs._python_mean_phase_gradient(theta))) == pytest.approx(
            1.0, abs=1e-12
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_matches_finite_difference(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-2:
            return  # near-incoherent: ψ is ill-conditioned
        np.testing.assert_allclose(
            mp_obs._python_mean_phase_gradient(theta),
            _finite_difference_mean_phase_gradient(theta),
            atol=1e-6,
        )

    def test_single_oscillator_gradient_is_one(self) -> None:
        np.testing.assert_allclose(
            mp_obs._python_mean_phase_gradient(np.array([2.7])), [1.0], atol=1e-15
        )

    def test_aligned_gradient_is_uniform(self) -> None:
        # All oscillators aligned: ψ = θ and ∂ψ/∂θ_j = 1/N for every j.
        grad = mp_obs._python_mean_phase_gradient(np.full(8, 0.7))
        np.testing.assert_allclose(grad, np.full(8, 1.0 / 8), atol=1e-15)

    def test_empty_gradient_is_empty(self) -> None:
        assert mp_obs._python_mean_phase_gradient(np.array([])).shape == (0,)

    def test_exact_incoherent_gradient_is_zero(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(mp_obs._python_mean_phase_gradient(theta), np.zeros(4))


class TestRustMeanPhaseTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "mean_phase_gradient", None)):
            pytest.skip("scpn_quantum_engine.mean_phase_gradient unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        assert mp_obs._rust_mean_phase(theta) == pytest.approx(
            mp_obs._python_mean_phase(theta), abs=1e-12
        )
        np.testing.assert_allclose(
            mp_obs._rust_mean_phase_gradient(theta),
            mp_obs._python_mean_phase_gradient(theta),
            atol=1e-12,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase(np.zeros(3))

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase_gradient(np.zeros(3))


class TestJuliaMeanPhaseTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import mean_phase as julia_value
        from scpn_quantum_control.accel.julia import mean_phase_gradient as julia_grad

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        assert julia_value(theta) == pytest.approx(mp_obs._python_mean_phase(theta), abs=1e-10)
        np.testing.assert_allclose(
            julia_grad(theta), mp_obs._python_mean_phase_gradient(theta), atol=1e-10
        )


class TestMeanPhaseDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        value_reference = mp_obs._python_mean_phase(theta)
        grad_reference = mp_obs._python_mean_phase_gradient(theta)
        for name, impl in mp_obs._MEAN_PHASE_CHAIN:
            try:
                assert impl(theta) == pytest.approx(value_reference, abs=1e-10), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        for name, impl in mp_obs._MEAN_PHASE_GRADIENT_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, grad_reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_mean_phase_gradient_tier_used,
            last_mean_phase_tier_used,
            mean_phase,
            mean_phase_gradient,
        )

        assert d.dispatch("mean_phase", np.zeros(4)) == pytest.approx(0.0)
        assert d.dispatch("mean_phase_gradient", np.full(4, 0.3)).shape == (4,)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=18)
        assert isinstance(mean_phase(theta), float)
        assert mean_phase_gradient(theta).shape == (18,)
        assert last_mean_phase_tier_used() in {"rust", "julia", "python"}
        assert last_mean_phase_gradient_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert mp_obs._MEAN_PHASE_CHAIN[-1][0] == "python"
        assert mp_obs._MEAN_PHASE_GRADIENT_CHAIN[-1][0] == "python"


class TestMeanPhasePartialEngine:
    def test_partial_engine_value_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [("rust", mp_obs._rust_mean_phase), ("python", mp_obs._python_mean_phase)],
        )
        assert disp(np.zeros(4)) == pytest.approx(0.0)
        assert disp.last_tier == "python"

    def test_partial_engine_gradient_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", mp_obs._rust_mean_phase_gradient),
                ("python", mp_obs._python_mean_phase_gradient),
            ],
        )
        np.testing.assert_allclose(
            disp(np.full(4, 0.5)), mp_obs._python_mean_phase_gradient(np.full(4, 0.5))
        )
        assert disp.last_tier == "python"


# ---------------------------------------------------------------------------
# Mean phase Hessian — analytic floor, invariants, parity, and dispatch
# ---------------------------------------------------------------------------


def _finite_difference_mean_phase_hessian(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference Hessian from the analytic mean-phase gradient floor."""
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            mp_obs._python_mean_phase_gradient(plus) - mp_obs._python_mean_phase_gradient(minus)
        ) / (2.0 * step)
    return out


class TestPythonMeanPhaseHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(41)
        theta = rng.uniform(-math.pi, math.pi, size=11)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        aligned_cos = (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)) / magnitude
        aligned_sin = (sin_mean * np.cos(theta) - cos_mean * np.sin(theta)) / magnitude
        expected = -(np.outer(aligned_sin, aligned_cos) + np.outer(aligned_cos, aligned_sin)) / (
            theta.size**2 * magnitude**2
        ) + np.diag(aligned_sin / (theta.size * magnitude))
        np.testing.assert_allclose(mp_obs._python_mean_phase_hessian(theta), expected, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_is_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rows_sum_to_zero(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-2:
            return
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(
            hessian, _finite_difference_mean_phase_hessian(theta), atol=1e-5
        )

    def test_single_oscillator_is_zero(self) -> None:
        hessian = mp_obs._python_mean_phase_hessian(np.array([2.7]))
        assert hessian.shape == (1, 1)
        assert abs(float(hessian[0, 0])) < 1e-15

    def test_empty_input_returns_empty_matrix(self) -> None:
        assert mp_obs._python_mean_phase_hessian(np.array([])).shape == (0, 0)

    def test_exact_incoherent_returns_zero_matrix(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(mp_obs._python_mean_phase_hessian(theta), np.zeros((4, 4)))


class TestRustMeanPhaseHessianTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "mean_phase_hessian", None)):
            pytest.skip("scpn_quantum_engine.mean_phase_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            mp_obs._rust_mean_phase_hessian(theta),
            mp_obs._python_mean_phase_hessian(theta),
            atol=1e-11,
        )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase_hessian(np.zeros(3))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", mp_obs._rust_mean_phase_hessian),
                ("python", mp_obs._python_mean_phase_hessian),
            ],
        )
        np.testing.assert_allclose(
            disp(np.full(4, 0.5)), mp_obs._python_mean_phase_hessian(np.full(4, 0.5))
        )
        assert disp.last_tier == "python"


class TestJuliaMeanPhaseHessianTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import mean_phase_hessian as julia_hessian

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        np.testing.assert_allclose(
            julia_hessian(theta), mp_obs._python_mean_phase_hessian(theta), atol=1e-10
        )


class TestMeanPhaseHessianDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = mp_obs._python_mean_phase_hessian(theta)
        for name, impl in mp_obs._MEAN_PHASE_HESSIAN_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_mean_phase_hessian_tier_used,
            mean_phase_hessian,
        )

        assert d.dispatch("mean_phase_hessian", np.full(4, 0.3)).shape == (4, 4)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=16)
        assert mean_phase_hessian(theta).shape == (16, 16)
        assert last_mean_phase_hessian_tier_used() in {"rust", "julia", "python"}

    def test_chain_ends_with_python_floor(self) -> None:
        assert mp_obs._MEAN_PHASE_HESSIAN_CHAIN[-1][0] == "python"


# ---------------------------------------------------------------------------
# Daido higher-order order parameters — physics, reduction, parity, dispatch
# ---------------------------------------------------------------------------


def _daido_value(theta: np.ndarray, m: int) -> float:
    return float(abs(np.mean(np.exp(1j * m * theta))))


def _finite_difference_daido_gradient(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros(theta.size, dtype=np.float64)
    for j in range(theta.size):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        grad[j] = (_daido_value(plus, m) - _daido_value(minus, m)) / (2.0 * step)
    return grad


class TestPythonDaidoFloor:
    def test_two_cluster_state_is_detected(self) -> None:
        # Two antipodal clusters: r_1 = 0 (first harmonic cancels), r_2 = 1.
        theta = np.array([0.0, 0.0, math.pi, math.pi])
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 2) == pytest.approx(1.0, abs=1e-10)

    def test_three_cluster_state_is_detected(self) -> None:
        # Three evenly spaced clusters: r_1 = r_2 = 0 but r_3 = 1.
        theta = np.repeat([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0], 3)
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 2) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 3) == pytest.approx(1.0, abs=1e-10)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(
            op_obs._python_order_parameter(theta), abs=1e-12
        )
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_gradient(theta, 1),
            op_obs._python_order_parameter_gradient(theta),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_matches_finite_difference(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _daido_value(theta, m) < 1e-2:
            return
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_gradient(theta, m),
            _finite_difference_daido_gradient(theta, m),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_sums_to_zero(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert abs(float(np.sum(da_obs._python_daido_order_parameter_gradient(theta, m)))) < 1e-12

    def test_value_in_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.uniform(-math.pi, math.pi, size=40)
        for m in (1, 2, 3, 4):
            assert 0.0 - 1e-12 <= da_obs._python_daido_order_parameter(theta, m) <= 1.0 + 1e-12

    def test_rejects_non_positive_harmonic(self) -> None:
        theta = np.zeros(4)
        for bad in (0, -1, -3):
            with pytest.raises(ValueError, match="positive integer"):
                da_obs._python_daido_order_parameter(theta, bad)
            with pytest.raises(ValueError, match="positive integer"):
                da_obs._python_daido_order_parameter_gradient(theta, bad)

    def test_empty_input(self) -> None:
        assert da_obs._python_daido_order_parameter(np.array([]), 2) == 0.0
        assert da_obs._python_daido_order_parameter_gradient(np.array([]), 2).shape == (0,)

    def test_exact_incoherent_gradient_is_zero(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(
            da_obs._python_daido_order_parameter_gradient(theta, 1), np.zeros(4)
        )


class TestRustDaidoTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_order_parameter_gradient", None)):
            pytest.skip("scpn_quantum_engine.daido_order_parameter_gradient unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        assert da_obs._rust_daido_order_parameter(theta, m) == pytest.approx(
            da_obs._python_daido_order_parameter(theta, m), abs=1e-12
        )
        np.testing.assert_allclose(
            da_obs._rust_daido_order_parameter_gradient(theta, m),
            da_obs._python_daido_order_parameter_gradient(theta, m),
            atol=1e-12,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter(np.zeros(3), 2)

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter_gradient(np.zeros(3), 2)

    def test_rust_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            da_obs._rust_daido_order_parameter(np.zeros(3), 0)


class TestJuliaDaidoTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import daido_order_parameter as julia_value
        from scpn_quantum_control.accel.julia import daido_order_parameter_gradient as julia_grad

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        for m in (1, 2, 3):
            assert julia_value(theta, m) == pytest.approx(
                da_obs._python_daido_order_parameter(theta, m), abs=1e-10
            )
            np.testing.assert_allclose(
                julia_grad(theta, m),
                da_obs._python_daido_order_parameter_gradient(theta, m),
                atol=1e-10,
            )


class TestDaidoDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        value_ref = da_obs._python_daido_order_parameter(theta, m)
        grad_ref = da_obs._python_daido_order_parameter_gradient(theta, m)
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_CHAIN:
            try:
                assert impl(theta, m) == pytest.approx(value_ref, abs=1e-10), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, grad_ref, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            daido_order_parameter,
            daido_order_parameter_gradient,
            last_daido_gradient_tier_used,
            last_daido_tier_used,
        )

        assert d.dispatch("daido_order_parameter", np.zeros(4), 2) == pytest.approx(1.0)
        assert d.dispatch("daido_order_parameter_gradient", np.full(4, 0.3), 2).shape == (4,)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=18)
        assert isinstance(daido_order_parameter(theta, 3), float)
        assert daido_order_parameter_gradient(theta, 3).shape == (18,)
        assert last_daido_tier_used() in {"rust", "julia", "python"}
        assert last_daido_gradient_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert da_obs._DAIDO_ORDER_PARAMETER_CHAIN[-1][0] == "python"
        assert da_obs._DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN[-1][0] == "python"


class TestDaidoPartialEngine:
    def test_partial_engine_value_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter),
                ("python", da_obs._python_daido_order_parameter),
            ],
        )
        assert disp(np.zeros(4), 2) == pytest.approx(1.0)
        assert disp.last_tier == "python"

    def test_partial_engine_gradient_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter_gradient),
                ("python", da_obs._python_daido_order_parameter_gradient),
            ],
        )
        out = disp(np.full(4, 0.5), 2)
        np.testing.assert_allclose(
            out, da_obs._python_daido_order_parameter_gradient(np.full(4, 0.5), 2)
        )
        assert disp.last_tier == "python"


# ---------------------------------------------------------------------------
# Daido Hessian — analytic floor, reduction, invariants, parity, dispatch
# ---------------------------------------------------------------------------


def _finite_difference_daido_hessian(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            da_obs._python_daido_order_parameter_gradient(plus, m)
            - da_obs._python_daido_order_parameter_gradient(minus, m)
        ) / (2.0 * step)
    return out


class TestPythonDaidoHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(43)
        theta = rng.uniform(-math.pi, math.pi, size=11)
        m = 2
        scaled = m * theta
        cos_mean = float(np.mean(np.cos(scaled)))
        sin_mean = float(np.mean(np.sin(scaled)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        aligned = (cos_mean * np.cos(scaled) + sin_mean * np.sin(scaled)) / magnitude
        expected = (m * m) * (
            np.outer(aligned, aligned) / (theta.size**2 * magnitude)
            - np.diag(aligned / theta.size)
        )
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, m), expected, atol=1e-15
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_order_parameter_hessian(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, 1),
            op_obs._python_order_parameter_hessian(theta),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_and_rows_sum_to_zero(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = da_obs._python_daido_order_parameter_hessian(theta, m)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-11)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _daido_value(theta, m) < 1e-2:
            return
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, m),
            _finite_difference_daido_hessian(theta, m),
            atol=1e-4,
        )

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            da_obs._python_daido_order_parameter_hessian(np.zeros(4), 0)

    def test_empty_and_incoherent(self) -> None:
        assert da_obs._python_daido_order_parameter_hessian(np.array([]), 2).shape == (0, 0)
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(
            da_obs._python_daido_order_parameter_hessian(theta, 1), np.zeros((4, 4))
        )


class TestRustDaidoHessianTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_order_parameter_hessian", None)):
            pytest.skip("scpn_quantum_engine.daido_order_parameter_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            da_obs._rust_daido_order_parameter_hessian(theta, m),
            da_obs._python_daido_order_parameter_hessian(theta, m),
            atol=1e-11,
        )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter_hessian(np.zeros(3), 2)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = da_obs.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter_hessian),
                ("python", da_obs._python_daido_order_parameter_hessian),
            ],
        )
        out = disp(np.full(4, 0.5), 2)
        np.testing.assert_allclose(
            out, da_obs._python_daido_order_parameter_hessian(np.full(4, 0.5), 2)
        )
        assert disp.last_tier == "python"


class TestJuliaDaidoHessianTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import (
            daido_order_parameter_hessian as julia_hessian,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        for m in (1, 2, 3):
            np.testing.assert_allclose(
                julia_hessian(theta, m),
                da_obs._python_daido_order_parameter_hessian(theta, m),
                atol=1e-10,
            )


class TestDaidoHessianDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = da_obs._python_daido_order_parameter_hessian(theta, m)
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            daido_order_parameter_hessian,
            last_daido_hessian_tier_used,
        )

        assert d.dispatch("daido_order_parameter_hessian", np.full(4, 0.3), 2).shape == (4, 4)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=16)
        assert daido_order_parameter_hessian(theta, 3).shape == (16, 16)
        assert last_daido_hessian_tier_used() in {"rust", "julia", "python"}

    def test_chain_ends_with_python_floor(self) -> None:
        assert da_obs._DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN[-1][0] == "python"
