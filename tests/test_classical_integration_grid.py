# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Classical Integration Grid Contract Tests
"""One integration grid, shared by the classical references and the Rust tier.

The reported time of a sample must be where the integrator actually put the
state. The previous grid spread ``linspace(0, t_max, n_steps + 1)`` over the
requested duration while the integrator advanced by ``dt``, so for a duration
that is not a multiple of ``dt`` every label disagreed with its own state: for
``t_max = 1`` and ``dt = 0.3`` an uncoupled oscillator with ``omega = 1``
reached phase 0.9 while the last time said 1.0. A zero duration was worse — it
still took one step, so the state at the reported time 0 was 0.3.

The oracle throughout is analytic. For ``K = 0`` and ``omega = 1`` the exact
solution is ``theta(t) = theta0 + t``, and the Euler step is exact for it, so
the phase after ``s`` steps must equal the reported time to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control._rust_accel import optional_rust_engine
from scpn_quantum_control.hardware import (
    integration_step_count as package_step_count,
)
from scpn_quantum_control.hardware.classical import (
    INTEGRATION_GRID_RELATIVE_TOLERANCE,
    classical_exact_evolution,
    classical_kuramoto_reference,
    integration_step_count,
    integration_times,
)


def _free_oscillator(
    t_max: float, dt: float, theta0: float = 0.0
) -> dict[str, NDArray[np.float64]]:
    """Integrate one uncoupled oscillator with unit frequency.

    Parameters
    ----------
    t_max
        Requested duration.
    dt
        Integration step.
    theta0
        Initial phase.

    Returns
    -------
    dict
        The reference result, whose exact solution is ``theta(t) = theta0 + t``.

    """
    return classical_kuramoto_reference(
        1,
        t_max=t_max,
        dt=dt,
        K=np.zeros((1, 1)),
        omega=np.ones(1),
        theta0=np.array([theta0]),
    )


class TestStepCount:
    """``integration_step_count`` never overshoots the requested duration."""

    @pytest.mark.parametrize(
        ("t_max", "dt", "expected"),
        [
            (1.0, 0.3, 3),
            (1.0, 0.4, 2),
            (1.0, 0.6, 1),
            (0.2, 0.3, 0),
            (0.0, 0.3, 0),
            (0.29999, 0.1, 2),
        ],
    )
    def test_non_divisible_durations_floor(self, t_max: float, dt: float, expected: int) -> None:
        """A partial final step is dropped rather than rounded up.

        Parameters
        ----------
        t_max, dt
            Grid inputs.
        expected
            Number of whole steps that fit.

        """
        assert integration_step_count(t_max, dt) == expected
        assert integration_step_count(t_max, dt) * dt <= t_max + 1e-12

    @pytest.mark.parametrize(
        ("t_max", "dt", "expected"),
        [
            (0.5, 0.1, 5),
            (2.0, 0.1, 20),
            (0.3, 0.1, 3),
            (0.1, 0.05, 2),
            (1.0, 0.1, 10),
            (0.7, 0.1, 7),
        ],
    )
    def test_divisible_durations_keep_every_step(
        self, t_max: float, dt: float, expected: int
    ) -> None:
        """A duration that is a whole number of steps keeps all of them.

        Parameters
        ----------
        t_max, dt
            Grid inputs.
        expected
            Number of steps the caller intends.

        """
        assert integration_step_count(t_max, dt) == expected

    @pytest.mark.parametrize(
        ("t_max", "dt", "expected"),
        [
            (0.3, 0.1, 3),
            (0.7, 0.1, 7),
            (0.6, 0.2, 3),
        ],
    )
    def test_quotients_that_are_inexact_in_binary_still_keep_every_step(
        self, t_max: float, dt: float, expected: int
    ) -> None:
        """These divide exactly in decimal but not in IEEE double.

        ``0.3 / 0.1`` is ``2.9999999999999996``, so a bare floor would return
        two steps and silently drop the caller's last one. This is the case the
        snapping tolerance exists for; the divisible cases above happen to come
        out exact and would survive a floor on their own.

        Parameters
        ----------
        t_max, dt
            Grid inputs.
        expected
            Number of steps the caller intends.

        """
        assert t_max / dt < expected
        assert integration_step_count(t_max, dt) == expected

    def test_the_snapping_tolerance_is_relative_and_narrow(self) -> None:
        """A duration genuinely short of a whole step is not snapped up."""
        dt = 0.1
        just_under = dt * (5 - 10 * INTEGRATION_GRID_RELATIVE_TOLERANCE)
        assert integration_step_count(just_under, dt) == 4
        within = dt * (5 - 0.01 * INTEGRATION_GRID_RELATIVE_TOLERANCE)
        assert integration_step_count(within, dt) == 5

    @pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
    def test_rejects_inadmissible_step(self, dt: float) -> None:
        """A non-positive or non-finite step has no grid.

        Parameters
        ----------
        dt
            Inadmissible step size.

        """
        with pytest.raises(ValueError, match="dt must be positive and finite"):
            integration_step_count(1.0, dt)

    @pytest.mark.parametrize("t_max", [-1.0, float("nan"), float("inf")])
    def test_rejects_inadmissible_duration(self, t_max: float) -> None:
        """A negative or non-finite duration has no grid.

        Parameters
        ----------
        t_max
            Inadmissible duration.

        """
        with pytest.raises(ValueError, match="t_max must be non-negative and finite"):
            integration_step_count(t_max, 0.1)


class TestTimes:
    """``integration_times`` reports where the integrator put each sample."""

    def test_times_are_multiples_of_the_step(self) -> None:
        """Sample ``s`` sits at ``s · dt``, exactly as the Rust kernel reports."""
        times = integration_times(4, 0.25)
        np.testing.assert_array_equal(times, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert times.dtype == np.float64

    def test_zero_steps_is_the_initial_sample_alone(self) -> None:
        """No evolution means one sample, at time zero."""
        np.testing.assert_array_equal(integration_times(0, 0.3), np.array([0.0]))

    def test_rejects_a_negative_step_count(self) -> None:
        """A negative step count is not a trajectory."""
        with pytest.raises(ValueError, match="n_steps must be non-negative"):
            integration_times(-1, 0.1)


class TestKuramotoGrid:
    """The Kuramoto reference reports truthful times."""

    def test_recorded_case_labels_match_their_own_states(self) -> None:
        """The card's case: t_max=1, dt=0.3, omega=1 reaches 0.9, not 1.0."""
        result = _free_oscillator(1.0, 0.3)

        np.testing.assert_allclose(result["times"], np.array([0.0, 0.3, 0.6, 0.9]), atol=1e-12)
        np.testing.assert_allclose(result["theta"][:, 0], result["times"], atol=1e-12)
        assert result["times"][-1] <= 1.0

    @pytest.mark.parametrize(("t_max", "dt"), [(1.0, 0.3), (0.5, 0.1), (1.0, 0.4), (0.7, 0.25)])
    def test_every_phase_equals_its_reported_time(self, t_max: float, dt: float) -> None:
        """For the free oscillator the analytic solution is theta(t) = t.

        Parameters
        ----------
        t_max, dt
            Grid inputs.

        """
        result = _free_oscillator(t_max, dt)

        np.testing.assert_allclose(result["theta"][:, 0], result["times"], atol=1e-12)

    def test_zero_duration_does_not_evolve(self) -> None:
        """A zero duration used to take one step and report it as time zero."""
        result = _free_oscillator(0.0, 0.3, theta0=0.4)

        assert result["times"].shape == (1,)
        assert result["theta"].shape == (1, 1)
        assert result["theta"][0, 0] == pytest.approx(0.4)
        assert result["times"][0] == 0.0

    def test_duration_shorter_than_one_step_does_not_evolve(self) -> None:
        """No whole step fits, so the trajectory is the initial condition."""
        result = _free_oscillator(0.2, 0.3, theta0=0.4)

        assert result["times"].shape == (1,)
        assert result["theta"][0, 0] == pytest.approx(0.4)

    def test_divisible_duration_keeps_its_final_sample(self) -> None:
        """0.5 / 0.1 is inexact in binary; the last step must survive it."""
        result = _free_oscillator(0.5, 0.1)

        assert result["times"].shape == (6,)
        assert result["times"][-1] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"t_max": 1.0, "dt": 0.0}, "dt must be positive and finite"),
            ({"t_max": -1.0, "dt": 0.1}, "t_max must be non-negative and finite"),
            ({"t_max": 1.0, "dt": float("nan")}, "dt must be positive and finite"),
        ],
    )
    def test_rejections_reach_the_public_entry_point(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        """The grid contract is enforced through the public function.

        Parameters
        ----------
        kwargs
            Inadmissible grid arguments.
        message
            Expected refusal text.

        """
        with pytest.raises(ValueError, match=message):
            classical_kuramoto_reference(
                1, K=np.zeros((1, 1)), omega=np.ones(1), theta0=np.zeros(1), **kwargs
            )


class TestExactEvolutionGrid:
    """The exact-evolution sibling uses the same grid."""

    def test_times_are_step_multiples(self) -> None:
        """A non-divisible duration ends short rather than mislabelling."""
        result = classical_exact_evolution(2, t_max=1.0, dt=0.3)

        np.testing.assert_allclose(result["times"], np.array([0.0, 0.3, 0.6, 0.9]), atol=1e-12)
        assert result["R"].shape == (4,)

    def test_zero_duration_returns_the_initial_state_alone(self) -> None:
        """It used to apply the propagator once for a zero-length evolution."""
        result = classical_exact_evolution(2, t_max=0.0, dt=0.1)

        assert result["times"].shape == (1,)
        assert result["R"].shape == (1,)
        assert result["times"][0] == 0.0

    def test_divisible_duration_keeps_its_final_sample(self) -> None:
        """0.3 / 0.1 is inexact in binary; the pinned corpus depends on this."""
        result = classical_exact_evolution(2, t_max=0.3, dt=0.1)

        assert result["times"].shape == (4,)
        assert result["times"][-1] == pytest.approx(0.3)

    def test_rejects_an_inadmissible_step(self) -> None:
        """The sibling had no grid validation of its own before this contract."""
        with pytest.raises(ValueError, match="dt must be positive and finite"):
            classical_exact_evolution(2, t_max=1.0, dt=0.0)


class TestTierParity:
    """Python and the compiled Rust tier report the same grid."""

    def test_python_fallback_matches_the_native_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same inputs, same times and same order parameter across tiers.

        Parameters
        ----------
        monkeypatch
            Used to force the Python fallback for the comparison arm.

        """
        engine = optional_rust_engine()
        if engine is None:
            pytest.skip("no native extension is installed in this environment")

        import scpn_quantum_control.hardware.classical as module

        coupling = np.full((4, 4), 0.2) - np.diag(np.full(4, 0.2))
        omega = np.linspace(0.5, 1.5, 4)
        theta0 = np.linspace(0.0, 1.0, 4)

        native = classical_kuramoto_reference(
            4, t_max=1.0, dt=0.3, K=coupling, omega=omega, theta0=theta0
        )
        monkeypatch.setattr(module, "optional_rust_engine", lambda: None)
        fallback = classical_kuramoto_reference(
            4, t_max=1.0, dt=0.3, K=coupling, omega=omega, theta0=theta0
        )

        np.testing.assert_array_equal(native["times"], fallback["times"])
        np.testing.assert_allclose(native["theta"], fallback["theta"], atol=1e-12)
        np.testing.assert_allclose(native["R"], fallback["R"], atol=1e-12)

    def test_both_tiers_report_the_expected_grid(self) -> None:
        """Whichever tier runs, the times are the shared expression."""
        result = classical_kuramoto_reference(
            2,
            t_max=1.0,
            dt=0.3,
            K=np.zeros((2, 2)),
            omega=np.ones(2),
            theta0=np.zeros(2),
        )

        expected = integration_times(integration_step_count(1.0, 0.3), 0.3)
        np.testing.assert_array_equal(result["times"], expected)


class TestPackageSurface:
    """The shared grid rule is reachable from the hardware package."""

    def test_package_export_is_the_module_function(self) -> None:
        """``scpn_quantum_control.hardware`` exposes the rule it documents."""
        assert package_step_count is integration_step_count
