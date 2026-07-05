# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — tests for the learned neural Lyapunov certificate tier
r"""Contract tests for the learned neural Lyapunov certificate.

These exercise real JAX and skip without the optional ``[jax]`` extra. The load-bearing claims: the
learned ``V_ψ`` vanishes and has a vanishing Lie derivative at the relaxed equilibrium and is invariant
to a global phase shift by construction; on the symmetric, cohesive regime the sampled verdict agrees
with the closed-form certificate; the learned certificate certifies an *asymmetric* coupling that the
closed-form certificate cannot even accept; a repulsive coupling is honestly refused and its violation
is surfaced by the falsifier; and every validation and residency contract holds. The verdict is a
sampling certificate, never a formal proof — the tests assert only what a finite sample can support.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import neural_lyapunov_certificate as nlc  # noqa: E402
from oscillatools.accel.neural_lyapunov_certificate import (  # noqa: E402
    LyapunovCertificateReport,
    LyapunovCounterexample,
    NeuralLyapunovCertificate,
    certify_neural_lyapunov,
    falsify_neural_lyapunov,
    fit_neural_lyapunov_certificate,
    neural_lyapunov_decrease,
    neural_lyapunov_value,
)
from oscillatools.accel.synchronisation_certificate import (  # noqa: E402
    certify_synchronisation,
)

_N = 3
# A single fit configuration reused across every JAX-touching test: keeping N, the hidden widths and
# the sample size fixed means the JIT-compiled kernels are shared, so only the first fit pays the
# compilation cost and the remaining fits reuse it.
_CONFIG = dict(
    region_radius=0.5,
    hidden_layers=(32,),
    epsilon=4e-2,
    learning_rate=1e-2,
    iterations=250,
    sample_size=128,
    warm_start_iterations=200,
    falsifier_rounds=2,
    falsifier_restarts=8,
    falsifier_steps=25,
    relaxation_steps=100,
)


def _symmetric_coupling(strength: float) -> np.ndarray:
    """A symmetric all-to-all coupling of the given strength (zero diagonal)."""
    return (np.ones((_N, _N)) - np.eye(_N)) * strength


@pytest.fixture(scope="module")
def certified() -> NeuralLyapunovCertificate:
    """A certificate for identical oscillators under symmetric attractive coupling (certifiable)."""
    return fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(0.9), seed=1, **_CONFIG
    )


def test_value_is_zero_at_the_equilibrium(certified: NeuralLyapunovCertificate) -> None:
    """``V_ψ(θ*) = 0`` exactly — the pinning subtraction guarantees it."""
    assert neural_lyapunov_value(certified, certified.phases_star) == 0.0


def test_decrease_is_zero_at_the_equilibrium(certified: NeuralLyapunovCertificate) -> None:
    """``\\dot V_ψ(θ*) = 0`` — the relaxed state is a relative equilibrium and ``∇V ⟂ 1``."""
    assert abs(neural_lyapunov_decrease(certified, certified.phases_star)) < 1e-8


def test_value_is_invariant_to_a_global_phase_shift(
    certified: NeuralLyapunovCertificate,
) -> None:
    """``V_ψ`` depends only on phase differences, so a rigid rotation leaves it unchanged."""
    theta = certified.phases_star + np.array([0.2, -0.15, 0.05])
    shifted = theta + 1.3
    assert neural_lyapunov_value(certified, theta) == pytest.approx(
        neural_lyapunov_value(certified, shifted), abs=1e-9
    )


def test_decrease_is_invariant_to_a_global_phase_shift(
    certified: NeuralLyapunovCertificate,
) -> None:
    """The Lie derivative is blind to the uniform drift: ``∇V ⟂ 1`` and ``f`` is shift-equivariant."""
    theta = certified.phases_star + np.array([0.1, 0.25, -0.2])
    shifted = theta + 0.8
    assert neural_lyapunov_decrease(certified, theta) == pytest.approx(
        neural_lyapunov_decrease(certified, shifted), abs=1e-9
    )


def test_symmetric_regime_is_certified_on_sample(
    certified: NeuralLyapunovCertificate,
) -> None:
    """The Lyapunov conditions hold across the verification annulus."""
    report = certify_neural_lyapunov(certified, seed=7)
    assert report.worst_decrease < 0.0
    assert report.minimum_value > 0.0
    assert report.is_certified_on_sample


def test_sampled_verdict_agrees_with_the_closed_form_certificate(
    certified: NeuralLyapunovCertificate,
) -> None:
    """Where the analytic gradient-flow certificate is valid, the learned one must agree with it."""
    analytic = certify_synchronisation(certified.phases_star, certified.omega, certified.coupling)
    learned = certify_neural_lyapunov(certified, seed=3)
    assert analytic.is_certified
    assert learned.is_certified_on_sample


def test_falsifier_finds_no_significant_violation_when_certified(
    certified: NeuralLyapunovCertificate,
) -> None:
    """A certified region exposes no positive Lie derivative — the flow never increases ``V_ψ``."""
    counterexample = falsify_neural_lyapunov(certified, seed=5, restarts=24, steps=60)
    assert counterexample.decrease < 1e-3


def test_asymmetric_coupling_is_learned_where_the_closed_form_cannot_apply() -> None:
    """The closed-form certificate rejects asymmetric coupling; the learned certificate handles it."""
    asymmetric = np.array([[0.0, 1.0, 0.3], [0.8, 0.0, 1.0], [0.25, 0.9, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        certify_synchronisation(np.zeros(_N), np.zeros(_N), asymmetric)
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), asymmetric, seed=1, **_CONFIG
    )
    report = certify_neural_lyapunov(certificate, seed=7)
    assert report.is_certified_on_sample


def test_repulsive_coupling_is_not_certified() -> None:
    """Repulsive coupling has no stable phase-locked state; the certificate is honestly refused."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(-0.9), seed=1, **_CONFIG
    )
    report = certify_neural_lyapunov(certificate, seed=7)
    assert not report.is_certified_on_sample
    assert certificate.counterexamples_added == _CONFIG["falsifier_rounds"]


def test_repulsive_falsifier_surfaces_a_counterexample() -> None:
    """The falsifier locates a genuine Lyapunov violation for the repulsive regime."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(-0.9), seed=1, **_CONFIG
    )
    counterexample = falsify_neural_lyapunov(certificate, seed=5, restarts=32, steps=80)
    assert counterexample.decrease > 1e-3


def test_relaxation_moves_the_guess_onto_the_relative_equilibrium() -> None:
    """A crude guess is relaxed onto the locked configuration, where the Lie derivative vanishes."""
    omega = np.array([-0.3, 0.0, 0.3])
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        omega,
        _symmetric_coupling(1.5),
        seed=1,
        **{**_CONFIG, "relaxation_steps": 300, "relaxation_rate": 2e-2},
    )
    assert not np.allclose(certificate.phases_star, np.zeros(_N))
    assert abs(neural_lyapunov_decrease(certificate, certificate.phases_star)) < 1e-6


def test_relaxation_steps_zero_leaves_a_true_equilibrium_untouched() -> None:
    """With no relaxation an exact equilibrium guess is carried through unchanged."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        np.zeros(_N),
        _symmetric_coupling(0.9),
        seed=1,
        **{**_CONFIG, "relaxation_steps": 0},
    )
    assert np.array_equal(certificate.phases_star, np.zeros(_N))


def test_fit_is_deterministic() -> None:
    """The same seed reproduces identical network parameters bit for bit."""
    kwargs = dict(seed=2, **_CONFIG)
    first = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(0.9), **kwargs
    )
    second = fit_neural_lyapunov_certificate(
        np.zeros(_N), np.zeros(_N), _symmetric_coupling(0.9), **kwargs
    )
    for (weight_a, bias_a), (weight_b, bias_b) in zip(
        first.parameters, second.parameters, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
        assert np.array_equal(bias_a, bias_b)


def test_warm_start_disabled_still_fits() -> None:
    """Skipping the warm start still produces a pinned certificate (``V_ψ(θ*) = 0``)."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        np.zeros(_N),
        _symmetric_coupling(0.9),
        seed=1,
        **{**_CONFIG, "warm_start": False},
    )
    assert neural_lyapunov_value(certificate, certificate.phases_star) == 0.0


def test_zero_warm_start_iterations_skip_the_warm_start() -> None:
    """``warm_start`` set but zero iterations bypasses the warm-start block."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        np.zeros(_N),
        _symmetric_coupling(0.9),
        seed=1,
        **{**_CONFIG, "warm_start_iterations": 0},
    )
    assert isinstance(certificate, NeuralLyapunovCertificate)


def test_generous_tolerance_ends_the_loop_after_one_round() -> None:
    """A tolerance above every violation halts the counterexample loop immediately."""
    certificate = fit_neural_lyapunov_certificate(
        np.zeros(_N),
        np.zeros(_N),
        _symmetric_coupling(0.9),
        seed=1,
        tolerance=1e9,
        **_CONFIG,
    )
    assert certificate.rounds == 1
    assert certificate.counterexamples_added == 0


def test_certificate_carries_its_training_diagnostics(
    certified: NeuralLyapunovCertificate,
) -> None:
    """The record exposes the configuration and the training summary."""
    assert certified.epsilon == _CONFIG["epsilon"]
    assert certified.region_radius == _CONFIG["region_radius"]
    assert certified.phases_star.shape == (_N,)
    assert certified.coupling.shape == (_N, _N)
    assert certified.training_risk >= 0.0
    assert 1 <= certified.rounds <= _CONFIG["falsifier_rounds"]


def test_report_reports_the_sample_size(certified: NeuralLyapunovCertificate) -> None:
    """The report echoes the verification sample size and carries a boolean verdict."""
    report = certify_neural_lyapunov(certified, sample_size=256, seed=1)
    assert isinstance(report, LyapunovCertificateReport)
    assert report.sample_size == 256
    assert isinstance(report.is_certified_on_sample, bool)


def test_counterexample_violation_is_the_worse_of_the_two_conditions(
    certified: NeuralLyapunovCertificate,
) -> None:
    """``violation = max(\\dot V_ψ, -V_ψ)`` at the returned state, of the right shape."""
    counterexample = falsify_neural_lyapunov(certified, seed=9, restarts=8, steps=20)
    assert isinstance(counterexample, LyapunovCounterexample)
    assert counterexample.state.shape == (_N,)
    assert counterexample.violation == pytest.approx(
        max(counterexample.decrease, -counterexample.value), abs=1e-9
    )


def test_backend_is_cached() -> None:
    """The JAX backend is built once and reused."""
    assert nlc._load_backend() is nlc._load_backend()


@pytest.mark.parametrize(
    "override",
    [
        {"region_radius": 0.0},
        {"hidden_layers": ()},
        {"hidden_layers": (0,)},
        {"epsilon": -1e-3},
        {"learning_rate": 0.0},
        {"iterations": 0},
        {"sample_size": 0},
        {"positivity_margin": -1e-3},
        {"decrease_margin": -1e-3},
        {"warm_start_iterations": -1},
        {"falsifier_rounds": 0},
        {"falsifier_restarts": 0},
        {"falsifier_steps": 0},
        {"falsifier_step_size": 0.0},
        {"relaxation_steps": -1},
        {"relaxation_rate": 0.0},
        {"tolerance": -1e-3},
    ],
)
def test_fit_rejects_out_of_bound_hyperparameters(override: dict[str, object]) -> None:
    """Every hyperparameter bound is enforced before any JAX work begins."""
    kwargs = {**_CONFIG, **override}
    with pytest.raises(ValueError):
        fit_neural_lyapunov_certificate(
            np.zeros(_N), np.zeros(_N), _symmetric_coupling(0.9), seed=1, **kwargs
        )


@pytest.mark.parametrize(
    ("phases_star", "omega", "coupling"),
    [
        (np.zeros((2, 2)), np.zeros(_N), _symmetric_coupling(0.9)),
        (np.zeros(1), np.zeros(1), np.zeros((1, 1))),
        (np.zeros(_N), np.zeros(_N + 1), _symmetric_coupling(0.9)),
        (np.zeros(_N), np.zeros(_N), np.zeros((_N, _N + 1))),
        (np.array([0.0, np.nan, 0.0]), np.zeros(_N), _symmetric_coupling(0.9)),
        (np.zeros(_N), np.array([0.0, np.inf, 0.0]), _symmetric_coupling(0.9)),
    ],
)
def test_fit_rejects_malformed_problems(
    phases_star: np.ndarray, omega: np.ndarray, coupling: np.ndarray
) -> None:
    """The equilibrium, frequencies and coupling shapes and finiteness are validated."""
    with pytest.raises(ValueError):
        fit_neural_lyapunov_certificate(phases_star, omega, coupling, region_radius=0.5)


@pytest.mark.parametrize("override", [{"sample_size": 0}, {"tolerance": -1e-3}])
def test_certify_rejects_out_of_bound_arguments(
    certified: NeuralLyapunovCertificate, override: dict[str, object]
) -> None:
    """The verification sample size and tolerance are validated."""
    with pytest.raises(ValueError):
        certify_neural_lyapunov(certified, **override)


@pytest.mark.parametrize("override", [{"restarts": 0}, {"steps": 0}, {"step_size": 0.0}])
def test_falsify_rejects_out_of_bound_arguments(
    certified: NeuralLyapunovCertificate, override: dict[str, object]
) -> None:
    """The falsifier restart, step and step-size bounds are enforced."""
    with pytest.raises(ValueError):
        falsify_neural_lyapunov(certified, **override)


@pytest.mark.parametrize("bad", [np.zeros(_N + 1), np.array([0.0, np.nan, 0.0])])
def test_value_rejects_malformed_states(
    certified: NeuralLyapunovCertificate, bad: np.ndarray
) -> None:
    """A phase configuration of the wrong shape or with non-finite entries is rejected."""
    with pytest.raises(ValueError):
        neural_lyapunov_value(certified, bad)


@pytest.mark.parametrize("bad", [np.zeros(_N + 1), np.array([np.inf, 0.0, 0.0])])
def test_decrease_rejects_malformed_states(
    certified: NeuralLyapunovCertificate, bad: np.ndarray
) -> None:
    """The Lie-derivative evaluation validates the state shape and finiteness."""
    with pytest.raises(ValueError):
        neural_lyapunov_decrease(certified, bad)
