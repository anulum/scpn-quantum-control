# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — independent local conformance tests
"""Exercise source-bound conformance using a real finite-difference owner."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import sparse
from scipy.integrate import solve_ivp

from oscillatools.accel.kuramoto_delayed import (
    delayed_mean_field_force,
    integrate_delayed_kuramoto,
)
from oscillatools.accel.kuramoto_noisy import integrate_noisy_kuramoto, noisy_kuramoto_step
from oscillatools.accel.kuramoto_system import KuramotoSystem
from oscillatools.accel.sparse_kuramoto import (
    sparse_coupling_from_scipy,
    sparse_kuramoto_rk4_trajectory,
    sparse_networked_kuramoto_force,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    check_custom_derivative_consistency,
    value_and_hessian,
    whole_program_value_and_grad,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_grad,
)
from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobRef,
    QuantumWorkload,
)
from scpn_quantum_control.metamorphic_ad_verification import (
    IndependentConformanceProtocol,
    evaluate_chain_rule_residual,
    evaluate_independent_scalar_conformance,
    evaluate_linearity_residual,
    probe_metamorphic_law,
    require_current_conformance,
)
from scpn_quantum_control.phase.qnode_circuit import (
    PhaseQNodeCircuit,
    PhaseQNodeSupportError,
    execute_phase_qnode_circuit,
    phase_qnode_support_report,
)
from scpn_quantum_control.wirtinger_calculus import (
    holomorphic_gradient,
    real_objective_gradient,
    wirtinger_partials,
)


def _digest(value: bytes) -> str:
    """Return an exact fixture identity without borrowing product code."""
    return sha256(value).hexdigest()


def _protocol() -> IndependentConformanceProtocol:
    """Declare a bounded analytic comparison before observing the result."""
    return IndependentConformanceProtocol(
        estimand="scalar cubic primal and d/dx at x=2",
        domain="real float64, central difference step 1e-6",
        oracle_kind="analytic",
        oracle_reference="hand-derived x^3 and 3*x^2",
        oracle_digest=_digest(Path(__file__).read_bytes()),
        input_digest=_digest(b"x=2;float64;step=1e-6"),
        source_digest=_digest(
            (
                Path(__file__).parents[1]
                / "src/scpn_quantum_control/differentiable_finite_difference.py"
            ).read_bytes()
        ),
        dataset_digest=_digest(b"deterministic-cubic-fixture-v1"),
        comparator_version="analytic-cubic-v1",
        runtime="Python finite_difference_central",
        absolute_tolerance=1e-8,
    )


def _observed() -> tuple[float, float]:
    """Run the existing public numerical owner, including its actual gradient."""
    result = value_and_finite_difference_grad(
        lambda values: float(values[0] ** 3), np.array([2.0]), step=1e-6
    )
    return result.value, float(result.gradient[0])


def test_independent_conformance_oracles_01() -> None:
    """Compare public runtime values to a separately derived cubic oracle."""
    value, gradient = _observed()
    oracle_value = 2.0**3
    oracle_gradient = 3.0 * 2.0**2
    assert oracle_value == 8.0
    assert oracle_gradient == 12.0
    result = evaluate_independent_scalar_conformance(
        _protocol(),
        observed_primal=value,
        observed_gradient=gradient,
        oracle_primal=oracle_value,
        oracle_gradient=oracle_gradient,
    )
    assert result.passed
    assert result.primal_residual == 0.0
    assert result.gradient_residual <= 1e-8
    require_current_conformance(result, _protocol())


def test_dynamic_semantic_fixture_refuses_unsupported_local_execution() -> None:
    """A local unitary route must refuse a partial-measurement challenge."""
    alpha = np.pi / 3.0
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("h", (1,)),
            ("rz", (0,), 1),
            ("rz", (1,), 1),
            ("measure", (0,)),
        ),
        observable="pauli_z",
    )
    report = phase_qnode_support_report(circuit, [alpha, alpha])
    assert report.supported is False
    assert report.unsupported_gates == ("measure",)
    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: measure"):
        execute_phase_qnode_circuit(circuit, [alpha, alpha])


def test_dynamic_semantic_reference_preserves_labelled_aer_distribution() -> None:
    """Execute the frozen conditional circuit on Aer with an analytic label oracle."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(2, 2)
    alpha = np.pi / 3.0
    circuit.ry(alpha, 0)
    circuit.h(1)
    circuit.rz(alpha, 0)
    circuit.rz(alpha, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.z(1)
    circuit.h(1)
    circuit.measure(1, 1)

    shots = 16_384
    simulator = AerSimulator(seed_simulator=211, max_parallel_threads=1)
    result = simulator.run(circuit, shots=shots).result()
    assert result.success
    counts = result.get_counts(circuit)
    assert set(counts) == {"00", "10", "01", "11"}
    assert sum(counts.values()) == shots

    expected = {
        "00": Fraction(9, 16),
        "10": Fraction(3, 16),
        "01": Fraction(1, 16),
        "11": Fraction(3, 16),
    }
    z = NormalDist().inv_cdf(1.0 - 0.01 / (2.0 * len(expected)))
    intervals: dict[str, tuple[float, float]] = {}
    for label, probability in expected.items():
        observed = counts[label] / shots
        denominator = 1.0 + z * z / shots
        centre = (observed + z * z / (2.0 * shots)) / denominator
        margin = (
            z
            * np.sqrt(observed * (1.0 - observed) / shots + z * z / (4.0 * shots * shots))
            / denominator
        )
        intervals[label] = (float(centre - margin), float(centre + margin))
        assert intervals[label][0] <= float(probability) <= intervals[label][1]
    assert not (intervals["10"][0] <= float(expected["01"]) <= intervals["10"][1])


@pytest.mark.parametrize("primal,gradient", [(9.0, 12.0), (8.0, 8.0)])
def test_independent_conformance_oracles_02(primal: float, gradient: float) -> None:
    """Catch wrong values even when a catalogue law exists and import works."""
    result = evaluate_independent_scalar_conformance(
        _protocol(),
        observed_primal=primal,
        observed_gradient=gradient,
        oracle_primal=8.0,
        oracle_gradient=12.0,
    )
    assert not result.passed
    with pytest.raises(ValueError, match="failed"):
        require_current_conformance(result, _protocol())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_digest", _digest(b"new-source")),
        ("oracle_digest", _digest(b"new-oracle")),
        ("dataset_digest", _digest(b"new-dataset")),
        ("input_digest", _digest(b"new-input")),
        ("comparator_version", "analytic-cubic-v2"),
    ],
)
def test_independent_conformance_oracles_03(field: str, value: str) -> None:
    """Reject a stale result when its source, oracle, data, or version changes."""
    protocol = _protocol()
    result = evaluate_independent_scalar_conformance(
        protocol,
        observed_primal=8.0,
        observed_gradient=12.0,
        oracle_primal=8.0,
        oracle_gradient=12.0,
    )
    if field == "source_digest":
        changed = replace(protocol, source_digest=value)
    elif field == "oracle_digest":
        changed = replace(protocol, oracle_digest=value)
    elif field == "dataset_digest":
        changed = replace(protocol, dataset_digest=value)
    elif field == "input_digest":
        changed = replace(protocol, input_digest=value)
    else:
        changed = replace(protocol, comparator_version=value)
    with pytest.raises(ValueError, match="identity"):
        require_current_conformance(result, changed)


def test_protocol_identity_is_deterministic_and_budget_bound() -> None:
    """Treat an altered acceptance band as a new protocol, not a prior pass."""
    protocol = _protocol()
    assert protocol.identity == _protocol().identity
    assert protocol.identity != replace(protocol, absolute_tolerance=1e-4).identity


@pytest.mark.parametrize("field", ["observed_primal", "oracle_gradient"])
def test_nonfinite_observation_or_oracle_refuses(field: str) -> None:
    """Do not qualify undefined numerical evidence."""
    values = {
        "observed_primal": 8.0,
        "observed_gradient": 12.0,
        "oracle_primal": 8.0,
        "oracle_gradient": 12.0,
    }
    values[field] = float("nan")
    with pytest.raises(ValueError, match=field):
        evaluate_independent_scalar_conformance(_protocol(), **values)


@pytest.mark.parametrize(
    "field",
    ["estimand", "domain", "oracle_reference", "comparator_version", "runtime"],
)
def test_protocol_requires_declared_comparison_context(field: str) -> None:
    """Keep a nominally passing number from qualifying an anonymous claim."""
    protocol = _protocol()
    replacements = {
        "estimand": lambda: replace(protocol, estimand=" "),
        "domain": lambda: replace(protocol, domain=" "),
        "oracle_reference": lambda: replace(protocol, oracle_reference=" "),
        "comparator_version": lambda: replace(protocol, comparator_version=" "),
        "runtime": lambda: replace(protocol, runtime=" "),
    }
    with pytest.raises(ValueError, match=field):
        replacements[field]()


def test_protocol_refuses_unsupported_oracle_and_unbound_digest() -> None:
    """Reject an invented evidence class or non-identity before execution."""
    protocol = _protocol()
    with pytest.raises(ValueError, match="oracle_kind"):
        replace(
            protocol,
            oracle_kind=cast(
                Literal["analytic", "numerical", "metamorphic", "empirical", "formal"],
                "import_only",
            ),
        )
    with pytest.raises(ValueError, match="source_digest"):
        replace(protocol, source_digest="not-a-sha256")


@pytest.mark.parametrize(
    "budget", [0.0, -1.0, float("inf"), float("nan"), True, "1", 1 + 0j, 10**400]
)
def test_protocol_requires_finite_positive_predeclared_budget(budget: Any) -> None:
    """Refuse a vacuous, negative or undefined numerical acceptance band."""
    with pytest.raises(ValueError, match="absolute_tolerance"):
        replace(_protocol(), absolute_tolerance=budget)


def test_integer_budget_is_normalized_before_protocol_identity_and_evaluation() -> None:
    """A supported numeric budget has one stable identity and can be evaluated."""
    protocol = replace(_protocol(), absolute_tolerance=1)
    assert protocol.absolute_tolerance == 1.0
    assert protocol.identity == replace(_protocol(), absolute_tolerance=1.0).identity
    observed_primal, observed_gradient = _observed()
    result = evaluate_independent_scalar_conformance(
        protocol,
        observed_primal=observed_primal,
        observed_gradient=observed_gradient,
        oracle_primal=8.0,
        oracle_gradient=12.0,
    )
    require_current_conformance(result, replace(_protocol(), absolute_tolerance=1.0))


def test_catalogue_registration_is_not_numerical_execution() -> None:
    """Keep an imported executable law from masquerading as a run result."""
    probe = probe_metamorphic_law("law:metamorphic.linearity")
    assert not probe.passed
    assert not probe.refused
    assert probe.residual is None


def test_linearity_law_uses_real_runtime_and_analytic_values() -> None:
    """Bind the registered linearity law to actual public function evaluations."""

    def linear(values: NDArray[np.float64]) -> float:
        """Provide a concrete linear objective to the numerical owner."""
        return float(3.0 * values[0])

    f_a = value_and_finite_difference_grad(linear, np.array([1.0])).value
    f_b = value_and_finite_difference_grad(linear, np.array([2.0])).value
    f_ab = value_and_finite_difference_grad(linear, np.array([3.0])).value
    assert (f_a, f_b, f_ab) == (3.0, 6.0, 9.0)
    assert evaluate_linearity_residual(f_a, f_b, f_ab).passed
    assert not evaluate_linearity_residual(f_a, f_b, f_ab + 1.0).passed


def test_chain_rule_law_uses_real_runtime_and_analytic_values() -> None:
    """Compare a composed public derivative to separately derived x^6 values."""
    inner = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), np.array([2.0]))
    outer = value_and_finite_difference_grad(
        lambda values: float(values[0] ** 3), np.array([inner.value])
    )
    composite = value_and_finite_difference_grad(
        lambda values: float(values[0] ** 6), np.array([2.0])
    )
    assert inner.value == 4.0
    assert outer.value == 64.0
    assert composite.value == 64.0
    assert abs(float(inner.gradient[0]) - 4.0) < 1e-4
    assert abs(float(outer.gradient[0]) - 48.0) < 1e-4
    law = evaluate_chain_rule_residual(
        float(outer.gradient[0]),
        float(inner.gradient[0]),
        float(composite.gradient[0]),
        tolerance=1e-4,
    )
    assert law.passed
    assert not evaluate_chain_rule_residual(
        float(outer.gradient[0]), 4.0, 8.0, tolerance=1e-4
    ).passed
    protocol = replace(
        _protocol(),
        estimand="composed scalar x^6 primal and derivative at x=2",
        domain="real float64, central difference step 1e-6",
        input_digest=_digest(b"x=2;composition=x^2 then y^3;step=1e-6"),
        comparator_version="analytic-composition-v1",
        absolute_tolerance=1e-4,
    )
    result = evaluate_independent_scalar_conformance(
        protocol,
        observed_primal=composite.value,
        observed_gradient=float(composite.gradient[0]),
        oracle_primal=2.0**6,
        oracle_gradient=6.0 * 2.0**5,
    )
    require_current_conformance(result, protocol)


@pytest.mark.parametrize("kind", ["metamorphic", "empirical", "formal"])
def test_scalar_numeric_comparison_cannot_promote_other_evidence_classes(
    kind: Literal["metamorphic", "empirical", "formal"],
) -> None:
    """Keep relation, sampling and proof claims outside scalar equality."""
    protocol = replace(_protocol(), oracle_kind=kind)
    with pytest.raises(ValueError, match="oracle_kind"):
        evaluate_independent_scalar_conformance(
            protocol,
            observed_primal=8.0,
            observed_gradient=12.0,
            oracle_primal=8.0,
            oracle_gradient=12.0,
        )


def test_consumer_rechecks_residual_instead_of_trusting_pass_flag() -> None:
    """A forged status bit cannot promote an observed wrong derivative."""
    protocol = _protocol()
    failed = evaluate_independent_scalar_conformance(
        protocol,
        observed_primal=8.0,
        observed_gradient=8.0,
        oracle_primal=8.0,
        oracle_gradient=12.0,
    )
    assert not failed.passed
    with pytest.raises(ValueError, match="failed"):
        require_current_conformance(replace(failed, passed=True), protocol)


def test_kuramoto_uncoupled_runtime_matches_independent_phase_formula() -> None:
    """Compare the public two-oscillator flow to exact uncoupled drift."""
    initial = np.array([-0.3, 0.7], dtype=np.float64)
    frequencies = np.array([0.2, -0.1], dtype=np.float64)
    system = KuramotoSystem.mean_field(initial, frequencies, 0.0, dt=0.05)
    observed = system.step(n=20)
    expected = initial + frequencies * 1.0
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    assert not np.allclose(observed, initial - frequencies, rtol=0.0, atol=1e-8)


def test_kuramoto_coupled_runtime_refines_toward_two_body_solution() -> None:
    """Use an analytic phase-separation trajectory as an independent oracle."""
    initial = np.array([-0.4, 0.4], dtype=np.float64)
    frequencies = np.zeros(2, dtype=np.float64)
    coupling = 0.7
    initial_gap = initial[1] - initial[0]
    expected_gap = 2.0 * np.arctan(np.tan(initial_gap / 2.0) * np.exp(-coupling))
    expected = np.array([-expected_gap / 2.0, expected_gap / 2.0])
    coarse = KuramotoSystem.mean_field(initial, frequencies, coupling, dt=0.2).step(n=5)
    fine = KuramotoSystem.mean_field(initial, frequencies, coupling, dt=0.1).step(n=10)
    coarse_error = float(np.max(np.abs(coarse - expected)))
    fine_error = float(np.max(np.abs(fine - expected)))
    assert fine_error < coarse_error
    assert fine_error < 1e-5
    assert not np.allclose(fine, -expected, rtol=0.0, atol=1e-5)


def test_kuramoto_network_and_rotating_frame_match_declared_equations() -> None:
    """Check two public topologies and a common-frequency frame shift."""
    initial = np.array([-0.4, 0.4], dtype=np.float64)
    frequencies = np.array([0.2, -0.1], dtype=np.float64)
    coupling = 0.7
    dt = 0.05
    steps = 20
    all_to_all = np.full((2, 2), coupling / 2.0, dtype=np.float64)
    mean_field = KuramotoSystem.mean_field(initial, frequencies, coupling, dt=dt).step(n=steps)
    networked = KuramotoSystem.networked(initial, frequencies, all_to_all, dt=dt).step(n=steps)
    np.testing.assert_allclose(networked, mean_field, rtol=0.0, atol=1e-13)

    common_frequency = 0.3
    shifted = KuramotoSystem.mean_field(
        initial, frequencies + common_frequency, coupling, dt=dt
    ).step(n=steps)
    np.testing.assert_allclose(
        shifted, mean_field + common_frequency * dt * steps, rtol=0.0, atol=1e-13
    )
    assert not np.allclose(shifted, mean_field, rtol=0.0, atol=1e-6)


def test_kuramoto_public_flow_converges_to_independent_scipy_reference() -> None:
    """Compare the public RK4 flow with a separately stated two-body ODE."""
    initial = np.array([-0.4, 0.4], dtype=np.float64)
    frequencies = np.array([0.2, -0.1], dtype=np.float64)
    coupling = 0.7

    def rhs(_time: float, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the two-body equation without the production force owner."""
        interaction = 0.5 * coupling * np.sin(phase[1] - phase[0])
        return np.array(
            [frequencies[0] + interaction, frequencies[1] - interaction],
            dtype=np.float64,
        )

    reference = solve_ivp(rhs, (0.0, 1.0), initial, method="DOP853", rtol=1e-12, atol=1e-14)
    assert reference.success
    expected = reference.y[:, -1]
    errors = [
        float(
            np.max(
                np.abs(
                    KuramotoSystem.mean_field(initial, frequencies, coupling, dt=1.0 / steps).step(
                        n=steps
                    )
                    - expected
                )
            )
        )
        for steps in (16, 32, 64)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] <= 1e-10 + 1e-8 * float(np.max(np.abs(expected)))


@pytest.mark.parametrize("size", [4, 8, 16])
@pytest.mark.parametrize("edge_factor", [0, 1, 2])
def test_sparse_kuramoto_size_and_edge_sweep_matches_dense_and_direct_force(
    size: int, edge_factor: int
) -> None:
    """Sweep oscillator and directed-edge counts through both public RK4 paths."""
    rng = np.random.default_rng(73)
    candidates = np.array(
        [(row, col) for row in range(size) for col in range(size) if row != col],
        dtype=np.intp,
    )
    selected = candidates[rng.permutation(len(candidates))[: size * edge_factor]]
    weights = rng.uniform(0.1, 0.8, size=len(selected)) * np.where(
        np.arange(len(selected)) % 2 == 0, 1.0, -1.0
    )
    matrix = np.zeros((size, size), dtype=np.float64)
    for (row, col), weight in zip(selected, weights, strict=True):
        matrix[row, col] = weight
    initial = np.linspace(-0.7, 0.8, size, dtype=np.float64)
    frequencies = np.linspace(-0.12, 0.14, size, dtype=np.float64)
    coupling = sparse_coupling_from_scipy(sparse.csr_array(matrix))
    assert coupling.nnz == size * edge_factor
    observed_force = sparse_networked_kuramoto_force(initial, coupling)
    direct_force = np.array(
        [
            sum(matrix[row, col] * np.sin(initial[col] - initial[row]) for col in range(size))
            for row in range(size)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed_force, direct_force, rtol=0.0, atol=2e-14)
    sparse_path = sparse_kuramoto_rk4_trajectory(initial, frequencies, coupling, 0.02, 10)
    dense_system = KuramotoSystem.networked(initial, frequencies, matrix, dt=0.02)
    dense_path = np.stack((initial, *(dense_system.step() for _ in range(10))))
    np.testing.assert_allclose(sparse_path, dense_path, rtol=0.0, atol=2e-12)
    if edge_factor:
        transposed_force = np.array(
            [
                sum(matrix[col, row] * np.sin(initial[col] - initial[row]) for col in range(size))
                for row in range(size)
            ],
            dtype=np.float64,
        )
        assert float(np.max(np.abs(observed_force - transposed_force))) > 1e-6


def test_delayed_public_integrator_preserves_analytic_zero_force_drift() -> None:
    """Check a history-backed method-of-steps run against direct arithmetic."""
    history = np.array([[-0.2, 0.7], [-0.1, 0.65], [0.0, 0.6]])
    frequencies = np.array([0.2, -0.1])
    trajectory = integrate_delayed_kuramoto(
        history,
        frequencies,
        lambda current, delayed: np.zeros_like(current),
        delay=0.2,
        dt=0.1,
        n_steps=4,
    )
    np.testing.assert_allclose(
        trajectory.terminal_phases,
        history[-1] + frequencies * 0.4,
        rtol=0.0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="integer multiple"):
        integrate_delayed_kuramoto(
            history,
            frequencies,
            lambda current, delayed: np.zeros_like(current),
            delay=0.25,
            dt=0.1,
            n_steps=4,
        )


def test_delayed_public_integrator_matches_two_window_method_of_steps() -> None:
    """Resolve the delay boundary against a hand-derived piecewise trajectory."""
    initial = np.array([1.0, -0.5])
    history = np.repeat(initial[np.newaxis, :], 5, axis=0)
    trajectory = integrate_delayed_kuramoto(
        history,
        np.zeros(2),
        lambda current, delayed: delayed,
        delay=1.0,
        dt=0.25,
        n_steps=8,
    )
    times = trajectory.times
    expected_factor = 1.0 + times + 0.5 * np.maximum(times - 1.0, 0.0) ** 2
    expected = expected_factor[:, np.newaxis] * initial[np.newaxis, :]
    np.testing.assert_allclose(trajectory.phases, expected, rtol=0.0, atol=1e-14)
    assert trajectory.delay_steps == 4
    assert not np.allclose(trajectory.phases[-1], 3.0 * initial, rtol=0.0, atol=1e-14)


def test_delayed_mean_field_refines_toward_independent_two_window_scipy() -> None:
    """Compare nonlinear delayed coupling with a separate method-of-steps ODE."""
    initial = np.array([0.15, -0.35])
    frequencies = np.array([0.2, -0.1])
    coupling = 0.7
    delay = 0.2

    def rhs(current: NDArray[np.float64], lagged: NDArray[np.float64]) -> NDArray[np.float64]:
        """State the two-body delayed equation independently of the owner."""
        interaction = coupling * np.sin(lagged[np.newaxis, :] - current[:, np.newaxis])
        return np.array(frequencies + interaction.mean(axis=1), dtype=np.float64)

    def first_rhs(_time: float, current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use the constant supplied history during the first delay window."""
        return rhs(current, initial)

    first = solve_ivp(
        first_rhs,
        (0.0, delay),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    assert first.success and first.sol is not None
    first_solution = first.sol

    def second_rhs(time: float, current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Read the earlier independently integrated dense trajectory."""
        return rhs(current, np.asarray(first_solution(time - delay), dtype=np.float64))

    second = solve_ivp(
        second_rhs,
        (delay, 2.0 * delay),
        first.y[:, -1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert second.success
    expected = second.y[:, -1]
    errors = []
    for dt in (0.1, 0.05, 0.025):
        history = np.repeat(initial[np.newaxis, :], int(round(delay / dt)) + 1, axis=0)
        trajectory = integrate_delayed_kuramoto(
            history,
            frequencies,
            lambda current, lagged: delayed_mean_field_force(current, lagged, coupling),
            delay=delay,
            dt=dt,
            n_steps=int(round(2.0 * delay / dt)),
        )
        errors.append(float(np.max(np.abs(trajectory.terminal_phases - expected))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] <= 1e-6


def test_noisy_public_step_uses_exact_supplied_increment() -> None:
    """Match one stochastic step to independently calculated fixed-noise values."""
    phases = np.array([0.0, 1.0])
    frequencies = np.array([1.0, -1.0])
    noise = np.array([1.0, -2.0])

    def force(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the deliberately absent coupling term."""
        return np.zeros_like(theta)

    observed = noisy_kuramoto_step(phases, frequencies, force, 0.5, 0.25, noise)
    np.testing.assert_allclose(observed, [0.75, -0.25], rtol=0.0, atol=1e-14)
    assert not np.allclose(observed, [0.75, 1.75], rtol=0.0, atol=1e-14)
    deterministic = noisy_kuramoto_step(phases, frequencies, force, 0.0, 0.25, noise)
    np.testing.assert_allclose(deterministic, [0.25, 0.75], rtol=0.0, atol=1e-14)


def test_noisy_public_ensemble_matches_paths_and_gaussian_coverage() -> None:
    """Exercise 200 seeded public runs against matched paths and a binomial oracle."""
    count = 1000
    steps = 20
    dt = 0.05
    diffusion = 0.2
    initial = np.zeros(count)
    frequencies = np.full(count, 0.1)
    scale = float(np.sqrt(2.0 * diffusion * dt))

    def no_coupling(current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Give each oscillator an independent zero-coupling Brownian path."""
        return np.zeros_like(current)

    samples = []
    for seed in range(200):
        observed = integrate_noisy_kuramoto(
            initial,
            frequencies,
            no_coupling,
            diffusion=diffusion,
            dt=dt,
            n_steps=steps,
            seed=seed,
        )
        generator = np.random.default_rng(seed)
        increments = np.stack([generator.standard_normal(count) for _ in range(steps)])
        expected = frequencies * (steps * dt) + scale * increments.sum(axis=0)
        np.testing.assert_allclose(observed.terminal_phases, expected, rtol=0.0, atol=1e-12)
        samples.append(observed.terminal_phases)
    replay = integrate_noisy_kuramoto(
        initial,
        frequencies,
        no_coupling,
        diffusion=diffusion,
        dt=dt,
        n_steps=steps,
        seed=0,
    )
    assert np.array_equal(samples[0], replay.terminal_phases)

    terminal = np.stack(samples).ravel()
    assert terminal.size == 200_000
    central_quantile = NormalDist().inv_cdf(0.975)
    covered = int(np.count_nonzero(np.abs(terminal - 0.1) <= central_quantile * np.sqrt(0.4)))
    proportion = covered / terminal.size
    z = NormalDist().inv_cdf(0.995)
    denominator = 1.0 + z * z / terminal.size
    centre = (proportion + z * z / (2.0 * terminal.size)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / terminal.size
            + z * z / (4.0 * terminal.size * terminal.size)
        )
        / denominator
    )
    assert centre - margin <= 0.95 <= centre + margin


@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
def test_inverse_matrix_gradient_matches_independent_diagonal_formula(eps: float) -> None:
    """Sweep public Program AD conditioning against a direct inverse formula."""

    def objective(values: Any) -> object:
        """Sum inverse entries through the public intercepted operation."""
        return np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), -1))

    result = whole_program_value_and_grad(
        objective, np.array([2.0, 0.0, 0.0, eps], dtype=np.float64)
    )
    assert result.value == pytest.approx(0.5 + 1.0 / eps, rel=1e-12)
    expected_gradient = np.array([-0.25, -1.0 / (2.0 * eps), -1.0 / (2.0 * eps), -1.0 / eps**2])
    np.testing.assert_allclose(result.gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert not np.allclose(result.gradient, [0.0, 0.0, 0.0, -1.0 / eps])


def test_repeated_spectrum_fixed_subspace_objective_and_primitive_refusal() -> None:
    """Separate a stable fixed-projector trace from an ambiguous eigensolver rule."""
    matrix = np.diag([1.0, 1.0, 3.0]).astype(np.float64)
    projector = np.diag([1.0, 1.0, 0.0]).astype(np.float64)

    def stable_objective(values: Any) -> object:
        """Measure the frozen two-dimensional cluster through its fixed projector."""
        return np.trace(np.reshape(values, (3, 3)) @ projector)

    def distinct_spectrum_objective(values: Any) -> object:
        """Route a repeated spectrum through the public eigenvalue primitive."""
        return np.sum(np.linalg.eigvalsh(np.reshape(values, (3, 3))))

    observed = whole_program_value_and_grad(stable_objective, matrix.reshape(-1))
    assert observed.value == pytest.approx(2.0, rel=0.0, abs=1e-12)
    np.testing.assert_allclose(observed.gradient, projector.reshape(-1), rtol=1e-12, atol=1e-12)
    assert not np.allclose(observed.gradient, np.eye(3).reshape(-1))

    with pytest.raises(ValueError, match="distinct eigenvalues"):
        whole_program_value_and_grad(distinct_spectrum_objective, matrix.reshape(-1))


@pytest.mark.parametrize("angle", [0.3, -0.6])
def test_moving_repeated_cluster_projector_matches_independent_eigh_family(
    angle: float,
) -> None:
    """Differentiate a known moving rank-two projector along an isospectral family."""

    def objective(values: Any) -> object:
        theta = values[0]
        isolated = np.stack((np.sin(theta), theta * 0.0, np.cos(theta)))
        lower_projector = np.eye(3) - np.outer(isolated, isolated)
        return lower_projector[0, 0]

    def eigenspace_value(theta: float) -> float:
        isolated = np.array([np.sin(theta), 0.0, np.cos(theta)])
        matrix = np.eye(3) + 2.0 * np.outer(isolated, isolated)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        np.testing.assert_allclose(eigenvalues, [1.0, 1.0, 3.0], atol=1e-12, rtol=0.0)
        lower_projector = eigenvectors[:, :2] @ eigenvectors[:, :2].T
        return float(lower_projector[0, 0])

    observed = whole_program_value_and_grad(objective, [angle])
    expected_value = float(np.cos(angle) ** 2)
    expected_gradient = float(-np.sin(2.0 * angle))
    step = 1e-5
    eigenspace_gradient = (eigenspace_value(angle + step) - eigenspace_value(angle - step)) / (
        2.0 * step
    )
    assert observed.value == pytest.approx(expected_value, abs=1e-12)
    assert observed.value == pytest.approx(eigenspace_value(angle), abs=1e-12)
    assert observed.gradient[0] == pytest.approx(expected_gradient, abs=1e-12)
    assert observed.gradient[0] == pytest.approx(eigenspace_gradient, abs=1e-6)
    isolated_value = float(np.sin(angle) ** 2)
    isolated_gradient = float(np.sin(2.0 * angle))
    assert abs(observed.value - isolated_value) > 0.1
    assert abs(observed.gradient[0] - isolated_gradient) > 0.5


@pytest.mark.parametrize("epsilon", [1e-2, 1e-4, 1e-6])
def test_non_diagonal_near_singular_solve_matches_adjugate_gradient(epsilon: float) -> None:
    """Compare the public matrix-solve adjoint with a hand-derived 2x2 identity."""
    rhs = np.array([1.0, 0.0], dtype=np.float64)
    weights = np.array([0.5, -0.75], dtype=np.float64)

    def objective(values: Any) -> object:
        """Differentiate the public solve with respect to the matrix entries."""
        return np.sum(np.linalg.solve(np.reshape(values, (2, 2)), rhs) * weights)

    matrix = np.array([[1.0, 1.0], [1.0, 1.0 + epsilon]], dtype=np.float64)
    observed = whole_program_value_and_grad(objective, matrix.reshape(-1))
    solution = np.array([(1.0 + epsilon) / epsilon, -1.0 / epsilon])
    adjoint = np.array([(1.25 + 0.5 * epsilon) / epsilon, -1.25 / epsilon])
    expected_value = (1.25 + 0.5 * epsilon) / epsilon
    expected_gradient = -np.outer(adjoint, solution).reshape(-1)
    np.testing.assert_allclose(observed.value, expected_value, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(observed.gradient, expected_gradient, rtol=1e-9, atol=1e-6)
    with pytest.raises(ValueError, match="from forward gradient"):
        replace(observed, gradient=observed.gradient * 1.000001)
    if epsilon < 1e-3:
        clipped_value = (1.25 + 0.5e-3) / 1e-3
        clipped_solution = np.array([(1.0 + 1e-3) / 1e-3, -1.0 / 1e-3])
        clipped_adjoint = np.array([(1.25 + 0.5e-3) / 1e-3, -1.25 / 1e-3])
        clipped_gradient = -np.outer(clipped_adjoint, clipped_solution).reshape(-1)
        assert not np.isclose(observed.value, clipped_value, rtol=1e-9, atol=1e-6)
        assert not np.allclose(observed.gradient, clipped_gradient, rtol=1e-9, atol=1e-6)


def test_studio_read_only_control_composition_preserves_frozen_revision(tmp_path: Path) -> None:
    """Exercise a bounded two-oscillator design through the real Studio AD spine."""
    if sys.version_info < (3, 12):
        pytest.importorskip("scpn_studio_platform", reason="Studio extra absent in 3.11 CI")

    from scpn_quantum_control.studio.executive import (
        ExecutiveRequest,
        preview_action,
        run_action,
    )
    from scpn_quantum_control.studio.executive_differentiate import default_registry

    program = {
        "inputs": [["u", 0.5]],
        "operations": [
            {"op": "mul", "inputs": ["u", "0.2"], "into": "shift"},
            {"op": "add", "inputs": ["0.4", "shift"], "into": "gap"},
            {"op": "mul", "inputs": ["gap", "gap"], "into": "gap_square"},
            {"op": "mul", "inputs": ["u", "u"], "into": "u_square"},
            {"op": "mul", "inputs": ["u_square", "0.1"], "into": "penalty"},
            {"op": "add", "inputs": ["gap_square", "penalty"], "into": "objective"},
        ],
        "output": "objective",
    }
    request = ExecutiveRequest(
        verb="differentiate",
        action_id="g03-two-oscillator-control",
        backend="python",
        parameters=program,
    )
    registry = default_registry()
    preview = preview_action(request, registry=registry)
    assert not preview.requires_approval
    record = run_action(request, registry=registry)
    assert record.result.status == "succeeded"
    assert record.plan.to_dict() == preview.to_dict()
    assert record.result.outputs["verified"] is True
    assert float(record.result.outputs["value"]) == pytest.approx(0.275, abs=1e-10)
    gradient = float(cast(list[float], record.result.outputs["gradient"])[0])
    assert gradient == pytest.approx(0.3, abs=1e-10)

    proposed_u = float(np.clip(0.5 - gradient, -0.5, 0.5))
    assert proposed_u == pytest.approx(0.2, abs=1e-10)
    assert -0.5 <= proposed_u <= 0.5
    held_out = KuramotoSystem.mean_field(
        np.array([-0.2, 0.2]),
        np.array([0.1 - proposed_u, -0.1 + proposed_u]),
        0.0,
        dt=0.1,
    ).step(n=1)
    gap = float(held_out[1] - held_out[0])
    held_out_objective = gap * gap + 0.1 * proposed_u * proposed_u
    assert held_out_objective == pytest.approx(0.1804, abs=1e-10)

    assert record.script is not None
    script_path = tmp_path / record.script.filename
    script_path.write_text(record.script.source)
    replay = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, check=True
    )
    assert "verified" in replay.stdout

    changed_program = dict(record.plan.parameters)
    changed_program["inputs"] = [("u", 0.4)]
    with pytest.raises(ValueError, match="digest must seal"):
        replace(record, plan=replace(record.plan, parameters=changed_program))
    assert record.plan.to_dict() == preview.to_dict()


def test_local_hal_late_cancel_preserves_repeated_completed_evidence() -> None:
    """Keep one completed job and its raw counts stable across retrieval and cancellation."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(backend)
    workload = QuantumWorkload("identity-custody", "mlir", "module {}", 2, shots=16)
    job = hal.submit(backend.backend_id, workload)
    first = hal.result(job)
    second = hal.result(job)
    assert first is second
    assert first.job is job
    assert sum(first.counts.values()) == 16
    assert hal.cancel(job) is job
    assert hal.status(job) == "completed"
    assert hal.result(job) is first
    with pytest.raises(ValueError, match="job lookup.*workload_id"):
        hal.result(replace(job, workload_id="wrong-workload"))
    assert hal.result(job) is first


def test_local_hal_lost_submit_response_recovers_one_exact_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a real local submission after its response is lost without retrying."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(backend)
    workload = QuantumWorkload("lost-response-custody", "mlir", "module {}", 2, shots=16)
    real_submit = backend.submit
    created: list[QuantumJobRef] = []
    attempts: list[str] = []

    def lose_response(
        submitted: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        attempts.append(submitted.workload_id)
        created.append(real_submit(submitted, approval_id=approval_id))
        raise TimeoutError("submit response lost")

    monkeypatch.setattr(backend, "submit", lose_response)
    with pytest.raises(TimeoutError, match="submit response lost"):
        hal.submit(backend.backend_id, workload)
    assert attempts == [workload.workload_id]
    assert len(created) == 1

    recovered = HardwareAbstractionLayer((backend.profile,))
    recovered.register_backend(backend)
    job = created[0]
    first = recovered.result(job)
    assert recovered.result(job) is first
    assert first.job is job
    assert sum(first.counts.values()) == workload.shots
    with pytest.raises(ValueError, match="job lookup.*workload_id"):
        recovered.result(replace(job, workload_id="wrong-workload"))
    assert recovered.result(job) is first
    assert attempts == [workload.workload_id]


@pytest.mark.parametrize(
    ("values", "expected_value", "expected_gradient"),
    [
        ([2.0, 3.0, 0.5], 13.0, [8.5, 3.0, 6.0]),
        ([-2.0, 3.0, 0.5], 5.5, [-2.5, 1.5, 1.0]),
    ],
)
def test_program_ad_branch_alias_and_indexed_writes_match_analytic_oracle(
    values: list[float], expected_value: float, expected_gradient: list[float]
) -> None:
    """Compare executed branch and alias mutations with hand-derived gradients."""

    def objective(parameters: Any) -> object:
        x, y, z = parameters
        scratch = parameters.copy()
        alias = scratch
        if x > 0.0:
            alias[1] = x * y
        else:
            alias[1] = x + y
        scratch[2] = alias[1] * z
        return scratch[1] + alias[2] + x * x

    observed = whole_program_value_and_grad(objective, np.array(values, dtype=np.float64))
    assert observed.value == pytest.approx(expected_value, abs=1e-12)
    np.testing.assert_allclose(observed.gradient, expected_gradient, atol=1e-12, rtol=0.0)
    assert observed.control_flow_observed
    assert observed.program_ir is not None
    assert any(edge.kind == "mutation_version" for edge in observed.program_ir.alias_edges)
    wrong_gradient = np.array(expected_gradient, dtype=np.float64)
    wrong_gradient[1] = 0.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(observed.gradient, wrong_gradient, atol=1e-12, rtol=0.0)


def test_batched_custom_derivatives_match_independent_block_jacobian() -> None:
    """Compare public batched JVP, VJP and Jacobian with hand-derived blocks."""
    rule = CustomDerivativeRule(
        name="g03_quadratic_pair",
        value_fn=lambda p: np.array([p[0] * p[1], p[0] * p[0] + p[1]]),
        jvp_rule=lambda p, t: np.array([p[1] * t[0] + p[0] * t[1], 2.0 * p[0] * t[0] + t[1]]),
        vjp_rule=lambda p, c: np.array([p[1] * c[0] + 2.0 * p[0] * c[1], p[0] * c[0] + c[1]]),
    )
    tangents = [[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]]
    cotangents = [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]
    jvp = batch_value_and_custom_jvp(rule, [2.0, 3.0], tangents)
    vjp = batch_value_and_custom_vjp(rule, [2.0, 3.0], cotangents)
    jacobian = batch_value_and_custom_jacobian(rule, [[2.0, 3.0], [-1.0, 4.0]])
    assert len(jvp) == len(tangents)
    assert len(vjp) == len(cotangents)
    np.testing.assert_array_equal([row.jvp for row in jvp], [[3.0, 4.0], [2.0, 1.0], [1.0, -2.0]])
    np.testing.assert_array_equal([row.vjp for row in vjp], [[3.0, 2.0], [4.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(jacobian[0].jacobian, [[3.0, 2.0], [4.0, 1.0]])
    np.testing.assert_array_equal(jacobian[1].jacobian, [[4.0, -1.0], [-2.0, 1.0]])
    np.testing.assert_array_equal(jvp[0].value, [6.0, 7.0])
    check = check_custom_derivative_consistency(
        rule, [2.0, 3.0], [-1.0, 2.0], [2.0, -1.0], tolerance=1e-6
    )
    assert check.passed
    wrong_rule = CustomDerivativeRule(
        name="g03_wrong_quadratic_pair",
        value_fn=rule.value_fn,
        jvp_rule=rule.jvp_rule,
        vjp_rule=lambda p, c: np.array([p[1] * c[0] + p[0] * c[1], p[0] * c[0] + c[1]]),
    )
    wrong_check = check_custom_derivative_consistency(
        wrong_rule, [2.0, 3.0], [-1.0, 2.0], [2.0, -1.0], tolerance=1e-6
    )
    assert not wrong_check.passed
    assert wrong_check.vjp_l2_error > wrong_check.tolerance


def test_program_ad_first_order_and_public_hessian_match_analytic_curvature() -> None:
    """Compare public first and second-order outputs against a hand Hessian."""

    def objective(values: Any) -> object:
        x, y = values
        return x * x * y + np.sin(x)

    values = np.array([0.5, 2.0], dtype=np.float64)
    first = whole_program_value_and_grad(objective, values)
    real_objective = cast(Callable[[NDArray[np.float64]], float], objective)
    second = value_and_hessian(real_objective, values, step=1e-4)
    expected_value = 0.5 + np.sin(0.5)
    expected_gradient = [2.0 + np.cos(0.5), 0.25]
    expected_hessian = [[4.0 - np.sin(0.5), 1.0], [1.0, 0.0]]
    assert first.value == pytest.approx(expected_value, abs=1e-12)
    assert second.value == pytest.approx(expected_value, abs=1e-12)
    np.testing.assert_allclose(first.gradient, expected_gradient, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(second.hessian, expected_hessian, atol=1e-6, rtol=0.0)
    np.testing.assert_array_equal(second.hessian, second.hessian.T)
    with pytest.raises(ValueError, match="Hessian method must be finite_difference"):
        value_and_hessian(real_objective, values, method="reverse")


def test_nonholomorphic_loss_uses_wirtinger_and_refuses_holomorphic_gradient() -> None:
    """Keep modulus-squared real-loss gradient separate from holomorphic AD."""
    point = np.array([1.0 + 2.0j], dtype=np.complex128)

    def loss(values: NDArray[np.complex128]) -> float:
        return float(np.abs(values[0]) ** 2)

    def complex_loss(values: NDArray[np.complex128]) -> complex:
        return complex(loss(values))

    partials = wirtinger_partials(complex_loss, point)
    np.testing.assert_allclose(partials.df_dz, np.conj(point), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(partials.df_dconj_z, point, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(real_objective_gradient(loss, point), point, atol=1e-6, rtol=0.0)
    assert partials.holomorphic_residual > 1.0
    with pytest.raises(ValueError, match="not holomorphic"):
        holomorphic_gradient(complex_loss, point)
