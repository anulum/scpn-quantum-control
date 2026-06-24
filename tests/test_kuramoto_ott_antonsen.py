# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Ott–Antonsen reduction tests
"""Multi-angle tests for the Ott–Antonsen reduction of the mean-field Kuramoto model.

Covers the reduced vector field against its analytic form and fixed point, the steady state and
its agreement with the mean-field self-consistency, the relaxation of the reduced flow onto the
synchronised branch from above and below, the match of the reduced ``r(t)`` against a full
Lorentzian ensemble integrated with the networked solver, the invariance of the modulus and the
linear phase drift under the mean frequency, the forward-sensitivity gradient against central
finite differences, and the input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.accel import (
    kuramoto_rk4_trajectory,
    lorentzian_order_parameter,
    ott_antonsen_field,
    ott_antonsen_order_parameter,
    ott_antonsen_steady_state,
    ott_antonsen_terminal_order_parameter_value_and_grad,
    ott_antonsen_trajectory,
)


class TestField:
    def test_matches_analytic_form(self) -> None:
        coupling, half_width, centre = 3.0, 0.5, 0.7
        for z in (0.3 + 0.1j, -0.4 + 0.6j, 0.8 - 0.2j):
            linear = 0.5 * coupling - half_width
            expected = (linear + 1j * centre) * z - 0.5 * coupling * abs(z) ** 2 * z
            assert ott_antonsen_field(z, coupling, half_width, centre=centre) == pytest.approx(
                expected
            )

    def test_steady_state_is_a_fixed_point(self) -> None:
        coupling, half_width = 3.0, 0.5  # K_c = 1.0, r* = sqrt(1 - 2*0.5/3)
        r_star = ott_antonsen_steady_state(coupling, half_width)
        # With zero mean frequency the synchronised state is a genuine fixed point.
        assert ott_antonsen_field(complex(r_star), coupling, half_width) == pytest.approx(0.0j)


class TestSteadyState:
    def test_matches_mean_field_self_consistency(self) -> None:
        for coupling, half_width in ((2.0, 0.5), (5.0, 1.0), (0.8, 0.5)):
            assert ott_antonsen_steady_state(coupling, half_width) == pytest.approx(
                lorentzian_order_parameter(coupling, half_width)
            )

    def test_relaxes_onto_branch_from_below_and_above(self) -> None:
        coupling, half_width, dt, n_steps = 3.0, 0.5, 0.02, 600
        r_star = ott_antonsen_steady_state(coupling, half_width)
        from_below = ott_antonsen_order_parameter(0.05 + 0j, coupling, half_width, dt, n_steps)
        from_above = ott_antonsen_order_parameter(0.99 + 0j, coupling, half_width, dt, n_steps)
        assert from_below[-1] == pytest.approx(r_star, abs=1e-4)
        assert from_above[-1] == pytest.approx(r_star, abs=1e-4)
        assert from_below[0] < from_below[-1]  # rises towards the branch
        assert from_above[0] > from_above[-1]  # falls towards the branch

    def test_incoherent_below_critical(self) -> None:
        coupling, half_width, dt, n_steps = 0.7, 0.5, 0.02, 800  # K < K_c = 1.0
        radii = ott_antonsen_order_parameter(0.4 + 0j, coupling, half_width, dt, n_steps)
        # Below the critical coupling the only stable state is incoherence: the order parameter
        # decays monotonically towards zero (the linear rate (K/2 − Δ) < 0 dominates).
        assert np.all(np.diff(radii) < 0.0)
        assert radii[-1] < 0.05
        assert ott_antonsen_steady_state(coupling, half_width) == 0.0


class TestEnsembleAgreement:
    def test_reduced_matches_full_lorentzian_ensemble(self) -> None:
        coupling, half_width, centre = 3.0, 0.5, 0.0
        dt, n_steps = 0.02, 200
        r0 = 0.1
        reduced = ott_antonsen_order_parameter(
            complex(r0), coupling, half_width, dt, n_steps, centre=centre
        )

        count = 2000
        rng = np.random.default_rng(0)
        quantiles = (np.arange(count) + 0.5) / count
        omega = centre + half_width * np.tan(math.pi * (quantiles - 0.5))
        rng.shuffle(omega)
        # Wrapped-Cauchy initial phases place the ensemble exactly on the Ott–Antonsen manifold
        # at order parameter r0.
        uniforms = rng.uniform(0.0, 1.0, count)
        theta0 = 2.0 * np.arctan(((1.0 - r0) / (1.0 + r0)) * np.tan(math.pi * (uniforms - 0.5)))
        matrix = np.full((count, count), coupling / count)
        np.fill_diagonal(matrix, 0.0)
        trajectory = kuramoto_rk4_trajectory(
            np.ascontiguousarray(theta0),
            np.ascontiguousarray(omega),
            np.ascontiguousarray(matrix),
            dt,
            n_steps,
        )
        ensemble_final = abs(np.mean(np.exp(1j * trajectory[-1])))
        assert ensemble_final == pytest.approx(reduced[-1], abs=2e-2)


class TestFrequencyInvariance:
    def test_modulus_invariant_and_phase_drifts(self) -> None:
        coupling, half_width, dt, n_steps = 3.0, 0.5, 0.01, 300
        centre = 0.9
        z0 = 0.3 + 0.0j
        without = ott_antonsen_order_parameter(z0, coupling, half_width, dt, n_steps)
        with_drift = ott_antonsen_order_parameter(
            z0, coupling, half_width, dt, n_steps, centre=centre
        )
        np.testing.assert_allclose(without, with_drift, atol=1e-6)
        trajectory = ott_antonsen_trajectory(z0, coupling, half_width, dt, n_steps, centre=centre)
        drift = math.atan2(trajectory[-1].imag, trajectory[-1].real)
        assert drift == pytest.approx(centre * dt * n_steps, abs=1e-3)


class TestDifferentiableForm:
    def test_gradient_matches_finite_difference(self) -> None:
        coupling, half_width, dt, n_steps = 3.0, 0.5, 0.02, 200
        z0 = 0.1 + 0.0j
        value, grad_coupling, grad_half_width = (
            ott_antonsen_terminal_order_parameter_value_and_grad(
                z0, coupling, half_width, dt, n_steps
            )
        )
        reference = ott_antonsen_order_parameter(z0, coupling, half_width, dt, n_steps)[-1]
        assert value == pytest.approx(reference)

        step = 1e-6
        fd_coupling = (
            ott_antonsen_order_parameter(z0, coupling + step, half_width, dt, n_steps)[-1]
            - ott_antonsen_order_parameter(z0, coupling - step, half_width, dt, n_steps)[-1]
        ) / (2.0 * step)
        fd_half_width = (
            ott_antonsen_order_parameter(z0, coupling, half_width + step, dt, n_steps)[-1]
            - ott_antonsen_order_parameter(z0, coupling, half_width - step, dt, n_steps)[-1]
        ) / (2.0 * step)
        assert grad_coupling == pytest.approx(fd_coupling, abs=1e-5)
        assert grad_half_width == pytest.approx(fd_half_width, abs=1e-5)

    def test_origin_is_not_differentiable(self) -> None:
        # z0 = 0 stays at the incoherent origin where r has no gradient.
        with pytest.raises(ValueError, match="not differentiable at the origin"):
            ott_antonsen_terminal_order_parameter_value_and_grad(0.0 + 0.0j, 0.8, 0.5, 0.02, 50)


class TestValidation:
    def test_trajectory_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="coupling must be positive"):
            ott_antonsen_trajectory(0.1 + 0j, 0.0, 0.5, 0.02, 10)
        with pytest.raises(ValueError, match="half_width must be positive"):
            ott_antonsen_trajectory(0.1 + 0j, 3.0, 0.0, 0.02, 10)
        with pytest.raises(ValueError, match="dt must be positive"):
            ott_antonsen_trajectory(0.1 + 0j, 3.0, 0.5, 0.0, 10)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            ott_antonsen_trajectory(0.1 + 0j, 3.0, 0.5, 0.02, 0)

    def test_steady_state_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="coupling must be positive"):
            ott_antonsen_steady_state(0.0, 0.5)
