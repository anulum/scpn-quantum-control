# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Property-based fuzz for hardware/classical.py
"""Property-based fuzz tests for the classical reference solvers.

Continues audit item B8 after the phase_artifact fuzz template. The
``classical_kuramoto_reference`` driver validates ``dt`` and ``t_max``
but then happily integrates whatever (K, omega, theta0) the caller
hands it. Hypothesis verifies:

* Boundary validators reject exactly the invalid (dt, t_max) pairs and
  no others.
* Valid integrations return dict-shaped output with matching array
  lengths for every random dimension 2 ≤ n_osc ≤ 6.
* The order parameter ``R(t)`` stays in ``[0, 1]`` under every valid
  random input — a physical invariant of the Kuramoto model that would
  catch any future regression in the Euler integrator or the Rust
  fast-path.
* Shape contracts between theta history, times, and R history never
  drift.

``classical_exact_diag`` gets a smaller fuzz pass — the 2^n Hilbert
space cap keeps it at n_osc ≤ 4 to stay under a couple of seconds per
example.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_quantum_control.hardware.classical import (
    _order_param,
    classical_exact_diag,
    classical_kuramoto_reference,
)

_GLOBAL_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-50.0,
    max_value=50.0,
)

positive_dt = st.floats(min_value=1e-3, max_value=0.2, allow_nan=False)

small_n_osc = st.integers(min_value=2, max_value=6)


def _coupling_matrix(rng: np.random.Generator, n: int, scale: float = 0.5) -> np.ndarray:
    """Symmetric coupling matrix with zero diagonal."""
    K = rng.normal(0.0, scale, size=(n, n))
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


# ---------------------------------------------------------------------------
# Validator boundary fuzz — dt + t_max
# ---------------------------------------------------------------------------


class TestKuramotoValidatorFuzz:
    @_GLOBAL_SETTINGS
    @given(
        dt=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        t_max=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_rejects_non_positive_dt(self, dt: float, t_max: float) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            classical_kuramoto_reference(n_osc=3, t_max=t_max, dt=dt)

    @_GLOBAL_SETTINGS
    @given(
        t_max=st.floats(max_value=-1e-6, allow_nan=False, allow_infinity=False),
    )
    def test_rejects_negative_t_max(self, t_max: float) -> None:
        with pytest.raises(ValueError, match="t_max must be non-negative"):
            classical_kuramoto_reference(n_osc=3, t_max=t_max, dt=0.01)

    def test_boundary_dt_equals_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            classical_kuramoto_reference(n_osc=3, t_max=1.0, dt=0.0)

    def test_boundary_t_max_equals_zero_accepted(self) -> None:
        # t_max = 0 is explicitly allowed ("non-negative"), not strict.
        out = classical_kuramoto_reference(n_osc=3, t_max=0.0, dt=0.01)
        assert len(out["times"]) >= 1
        assert len(out["R"]) >= 1


# ---------------------------------------------------------------------------
# Integration result invariants
# ---------------------------------------------------------------------------


class TestKuramotoIntegrationFuzz:
    @_GLOBAL_SETTINGS
    @given(
        n_osc=small_n_osc,
        t_max=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
        dt=positive_dt,
    )
    def test_integration_returns_dict_with_matching_lengths(
        self,
        n_osc: int,
        t_max: float,
        dt: float,
    ) -> None:
        out = classical_kuramoto_reference(n_osc=n_osc, t_max=t_max, dt=dt)
        assert set(out) == {"times", "theta", "R"}
        n = len(out["times"])
        assert n >= 1
        assert len(out["R"]) == n
        assert out["theta"].shape[0] == n
        assert out["theta"].shape[1] == n_osc

    @_GLOBAL_SETTINGS
    @given(
        n_osc=small_n_osc,
        t_max=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
        dt=positive_dt,
    )
    def test_R_stays_in_unit_interval(
        self,
        n_osc: int,
        t_max: float,
        dt: float,
    ) -> None:
        """Kuramoto order parameter |<exp(i theta)>| ∈ [0, 1] at all times."""
        out = classical_kuramoto_reference(n_osc=n_osc, t_max=t_max, dt=dt)
        R = out["R"]
        assert np.all(np.isfinite(R))
        assert np.all(R >= -1e-12)
        assert np.all(R <= 1.0 + 1e-12)

    @_GLOBAL_SETTINGS
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        n_osc=small_n_osc,
    )
    def test_custom_coupling_stays_finite(
        self,
        seed: int,
        n_osc: int,
    ) -> None:
        """Random symmetric K with any omega must produce finite theta(t)
        and R(t) — no NaN / inf regressions under perturbation."""
        rng = np.random.default_rng(seed)
        K = _coupling_matrix(rng, n_osc)
        omega = rng.normal(1.0, 0.2, size=n_osc)
        theta0 = rng.uniform(0.0, 2 * math.pi, size=n_osc)
        out = classical_kuramoto_reference(
            n_osc=n_osc,
            t_max=1.0,
            dt=0.05,
            K=K,
            omega=omega,
            theta0=theta0,
        )
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(np.isfinite(out["R"]))


# ---------------------------------------------------------------------------
# _order_param invariants
# ---------------------------------------------------------------------------


class TestOrderParameterFuzz:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_order_param_in_unit_interval(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        r = _order_param(theta)
        assert 0.0 - 1e-12 <= r <= 1.0 + 1e-12

    @_GLOBAL_SETTINGS
    @given(n=st.integers(min_value=1, max_value=32))
    def test_order_param_phase_locked_is_one(self, n: int) -> None:
        """When every oscillator shares the same phase, R = 1 exactly."""
        theta = np.full(n, math.pi / 3)
        assert abs(_order_param(theta) - 1.0) < 1e-12

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        shift=finite_floats,
    )
    def test_order_param_is_rotation_invariant(
        self,
        n: int,
        shift: float,
    ) -> None:
        """R is invariant under uniform phase translation."""
        rng = np.random.default_rng(n)
        theta = rng.uniform(0.0, 2 * math.pi, size=n)
        r_original = _order_param(theta)
        r_shifted = _order_param(theta + shift)
        assert abs(r_original - r_shifted) < 1e-12


# ---------------------------------------------------------------------------
# classical_exact_diag — small Hilbert space only
# ---------------------------------------------------------------------------


class TestExactDiagFuzz:
    @_GLOBAL_SETTINGS
    @given(n_osc=st.integers(min_value=2, max_value=4))
    def test_exact_diag_returns_finite_spectrum(self, n_osc: int) -> None:
        out = classical_exact_diag(n_osc=n_osc)
        assert np.all(np.isfinite(out["eigenvalues"]))
        assert np.isfinite(out["ground_energy"])
        # Spectrum must be sorted ascending.
        assert np.all(np.diff(out["eigenvalues"]) >= -1e-10)

    @_GLOBAL_SETTINGS
    @given(n_osc=st.integers(min_value=2, max_value=4))
    def test_ground_state_is_normalised(self, n_osc: int) -> None:
        out = classical_exact_diag(n_osc=n_osc)
        gs = out["ground_state"]
        norm = float(np.linalg.norm(gs))
        assert abs(norm - 1.0) < 1e-8

    @_GLOBAL_SETTINGS
    @given(n_osc=st.integers(min_value=2, max_value=4))
    def test_spectral_gap_non_negative(self, n_osc: int) -> None:
        out = classical_exact_diag(n_osc=n_osc)
        assert out["spectral_gap"] >= -1e-10
