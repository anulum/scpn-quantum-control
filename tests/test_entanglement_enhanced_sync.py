# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Entangled Initial-State Synchronisation Study
"""Real-surface tests for the entanglement-sync initial-state comparison API."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis.entanglement_enhanced_sync import (
    InitialState,
    SyncTrajectory,
    compare_all_initial_states,
    compare_initial_states_with_dephased_controls,
    entanglement_advantage,
    local_phase_observables,
    mean_single_qubit_linear_entropy,
    prepare_initial_state,
    simulate_sync_trajectory,
    transverse_exchange_coherence,
)
from scpn_quantum_control.analysis.quantum_speed_limit import compute_qsl
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError


def _statevector(circuit: QuantumCircuit) -> NDArray[np.complex128]:
    """Return one circuit state as a typed complex array."""
    return np.asarray(Statevector.from_instruction(circuit), dtype=np.complex128)


@pytest.mark.parametrize("state_type", tuple(InitialState))
def test_prepare_initial_state_returns_normalised_state(state_type: InitialState) -> None:
    """Prepare every public state family through the real Qiskit entry point."""
    vector = _statevector(prepare_initial_state(4, state_type, OMEGA_N_16[:4]))
    assert vector.shape == (16,)
    assert np.vdot(vector, vector).real == pytest.approx(1.0)


def test_prepared_state_support_and_entanglement_are_correct() -> None:
    """Check exact GHZ/W support and pure-state one-versus-rest entropy."""
    omega = OMEGA_N_16[:4]
    product = _statevector(prepare_initial_state(4, InitialState.PRODUCT, omega))
    bell = _statevector(prepare_initial_state(4, InitialState.BELL_PAIRS, omega))
    ghz = _statevector(prepare_initial_state(4, InitialState.GHZ, omega))
    w_state = _statevector(prepare_initial_state(4, InitialState.W_STATE, omega))

    assert np.flatnonzero(abs(ghz) > 1e-12).tolist() == [0, 15]
    assert np.abs(ghz[[0, 15]]) ** 2 == pytest.approx([0.5, 0.5])
    assert len(np.flatnonzero(abs(w_state) > 1e-12)) == 4
    assert np.abs(w_state[np.flatnonzero(abs(w_state) > 1e-12)]) ** 2 == pytest.approx([0.25] * 4)
    assert mean_single_qubit_linear_entropy(product, 4) == pytest.approx(0.0, abs=1e-12)
    assert mean_single_qubit_linear_entropy(bell, 4) == pytest.approx(1.0)
    assert mean_single_qubit_linear_entropy(ghz, 4) == pytest.approx(1.0)
    assert mean_single_qubit_linear_entropy(w_state, 4) == pytest.approx(0.75)


def test_odd_bell_family_and_single_qubit_w_are_supported() -> None:
    """Exercise the unpaired Bell qubit and one-qubit W boundaries."""
    odd = _statevector(prepare_initial_state(3, InitialState.BELL_PAIRS, OMEGA_N_16[:3]))
    one = _statevector(prepare_initial_state(1, InitialState.W_STATE, OMEGA_N_16[:1]))
    assert odd.shape == (8,)
    assert one.tolist() == [0j, (1 + 0j)]
    assert transverse_exchange_coherence(one, 1) == 0.0


@pytest.mark.parametrize(
    ("n_qubits", "omega", "message"),
    [
        (0, np.zeros(0), "positive integer"),
        (True, np.zeros(1), "positive integer"),
        (2, np.zeros(1), "shape"),
        (2, np.array([0.0, np.nan]), "finite"),
    ],
)
def test_prepare_initial_state_rejects_invalid_dimensions(
    n_qubits: int,
    omega: NDArray[np.float64],
    message: str,
) -> None:
    """Reject invalid state-preparation dimensions and frequencies."""
    with pytest.raises(ValueError, match=message):
        prepare_initial_state(n_qubits, InitialState.PRODUCT, omega)


def test_prepare_initial_state_rejects_unknown_family() -> None:
    """Reject values outside the public initial-state enum."""
    with pytest.raises(TypeError, match="InitialState"):
        prepare_initial_state(2, "product", np.zeros(2))  # type: ignore[arg-type] # intentional invalid input


def test_local_phase_order_requires_transverse_visibility() -> None:
    """Prevent the historical atan2(0, 0) mapping of basis states to R=1."""
    basis_zero = np.array([1.0, 0.0], dtype=np.complex128)
    order, visibility, defined = local_phase_observables(basis_zero, 1)
    assert order == 0.0
    assert visibility == 0.0
    assert defined is False

    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    order, visibility, defined = local_phase_observables(plus, 1)
    assert order == pytest.approx(1.0)
    assert visibility == pytest.approx(1.0)
    assert defined is True


def test_local_phase_order_detects_opposite_visible_phases() -> None:
    """Distinguish visible antiphase from absent local phase."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    product = np.kron(minus, plus)
    order, visibility, defined = local_phase_observables(product, 2)
    assert order == pytest.approx(0.0, abs=1e-12)
    assert visibility == pytest.approx(1.0)
    assert defined is True


def test_exchange_coherence_distinguishes_w_state_from_dephased_control() -> None:
    """Measure pair exchange coherence without relabelling it Kuramoto R."""
    vector = _statevector(prepare_initial_state(4, InitialState.W_STATE, OMEGA_N_16[:4]))
    density = np.outer(vector, vector.conj())
    dephased = np.diag(np.diag(density)).astype(np.complex128)
    assert transverse_exchange_coherence(vector, 4) == pytest.approx(0.5)
    assert transverse_exchange_coherence(dephased, 4) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("state", "n_qubits", "message"),
    [
        (np.zeros(2, dtype=np.complex128), 1, "positive finite norm"),
        (np.zeros(3, dtype=np.complex128), 1, "shape"),
        (np.eye(2, dtype=np.complex128), 1, "unit trace"),
        (np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128), 1, "Hermitian"),
        (
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128),
            1,
            "finite",
        ),
    ],
)
def test_observables_reject_invalid_quantum_states(
    state: NDArray[np.complex128],
    n_qubits: int,
    message: str,
) -> None:
    """Fail closed on malformed statevectors and density matrices."""
    with pytest.raises(ValueError, match=message):
        local_phase_observables(state, n_qubits)


def test_local_phase_observables_reject_invalid_tolerance() -> None:
    """Require a finite non-negative visibility tolerance."""
    with pytest.raises(ValueError, match="visibility_tolerance"):
        local_phase_observables(np.array([1.0, 0.0]), 1, visibility_tolerance=-1.0)


def test_public_observables_reject_invalid_qubit_count_and_mixed_entropy_input() -> None:
    """Public observables reject invalid dimensions and mixed-state entropy use."""
    vector = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="positive integer"):
        local_phase_observables(vector, 0)
    with pytest.raises(ValueError, match="pure statevector"):
        mean_single_qubit_linear_entropy(np.diag([1.0, 0.0]), 1)


def test_simulation_returns_bounded_observables_and_defined_flags() -> None:
    """Exercise the exact public simulator without a fake Hamiltonian."""
    coupling = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    product = simulate_sync_trajectory(
        coupling,
        omega,
        InitialState.PRODUCT,
        t_max=0.5,
        n_steps=5,
    )
    ghz = simulate_sync_trajectory(
        coupling,
        omega,
        InitialState.GHZ,
        t_max=0.5,
        n_steps=5,
    )
    assert product.times == pytest.approx(np.linspace(0.0, 0.5, 6))
    assert len(product.R_values) == len(product.local_visibility_values) == 6
    assert len(product.phase_defined_values) == len(product.exchange_coherence_values) == 6
    assert all(0.0 <= value <= 1.0 for value in product.R_values)
    assert all(0.0 <= value <= 1.0 for value in product.exchange_coherence_values)
    assert product.final_phase_defined is True
    assert ghz.final_R == 0.0
    assert ghz.final_local_visibility == 0.0
    assert ghz.final_phase_defined is False


def test_simulation_budget_fails_before_dense_hamiltonian() -> None:
    """Keep the dense-allocation guard on the real entry point."""
    with pytest.raises(DenseAllocationError, match="entangled initial-state dense"):
        simulate_sync_trajectory(
            build_knm_paper27(L=12),
            OMEGA_N_16[:12],
            InitialState.PRODUCT,
            t_max=0.1,
            n_steps=1,
            max_dense_gib=1e-12,
        )


@pytest.mark.parametrize(
    ("coupling", "omega", "t_max", "n_steps", "message"),
    [
        (np.zeros((2, 3)), np.zeros(2), 1.0, 2, "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.zeros(2), 1.0, 2, "finite"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2), 1.0, 2, "symmetric"),
        (np.zeros((2, 2)), np.zeros(1), 1.0, 2, "shape"),
        (np.zeros((2, 2)), np.zeros(2), 0.0, 2, "t_max"),
        (np.zeros((2, 2)), np.zeros(2), np.inf, 2, "t_max"),
        (np.zeros((2, 2)), np.zeros(2), 1.0, 0, "n_steps"),
        (np.zeros((2, 2)), np.zeros(2), 1.0, True, "n_steps"),
    ],
)
def test_simulation_rejects_invalid_model_contracts(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float,
    n_steps: int,
    message: str,
) -> None:
    """Reject malformed exact-evolution contracts before allocation."""
    with pytest.raises(ValueError, match=message):
        simulate_sync_trajectory(
            coupling,
            omega,
            InitialState.PRODUCT,
            t_max=t_max,
            n_steps=n_steps,
        )


def test_compare_all_initial_states_uses_one_observable_contract() -> None:
    """Return all four real trajectories with no false entangled-state R."""
    results = compare_all_initial_states(
        build_knm_paper27(L=4),
        OMEGA_N_16[:4],
        t_max=0.5,
        n_steps=5,
        max_dense_gib=0.25,
    )
    assert tuple(results) == tuple(state.value for state in InitialState)
    assert results["product"].final_phase_defined is True
    assert results["bell_pairs"].final_phase_defined is False
    assert results["ghz"].final_phase_defined is False
    assert results["w_state"].final_phase_defined is False


def test_matched_control_study_refuses_entanglement_specific_attribution() -> None:
    """Retain the separable coherence control and BL-65 certificate."""
    comparisons = compare_initial_states_with_dephased_controls(
        build_knm_paper27(L=4),
        OMEGA_N_16[:4],
        t_max=2.0,
        n_steps=20,
    )
    assert tuple(comparisons) == tuple(state.value for state in InitialState)
    assert comparisons["product"].initial_mean_single_qubit_linear_entropy == pytest.approx(
        0.0, abs=1e-12
    )
    assert comparisons["product"].delta_mean_exchange_coherence > 0.2
    assert comparisons["bell_pairs"].delta_mean_exchange_coherence > 0.03
    assert comparisons["ghz"].delta_mean_exchange_coherence == pytest.approx(0.0, abs=1e-12)
    assert comparisons["w_state"].delta_mean_exchange_coherence > 0.3
    for comparison in comparisons.values():
        assert comparison.attribution_status == "coherence_effect_not_entanglement_specific"
        assert comparison.entanglement_specific_effect_supported is False
        assert comparison.language_status == "research_observation"
        certificate = comparison.no_advantage_certificate
        assert certificate["language_status"] == "no_advantage_default"
        assert certificate["protocol_id"] == "protocol:entanglement.initial_state_observation"
        assert comparison.to_dict()["entanglement_specific_effect_supported"] is False


def test_zero_coherence_scenario_keeps_attribution_unestablished() -> None:
    """Exercise the fail-closed attribution branch when every state is stationary."""
    comparisons = compare_initial_states_with_dephased_controls(
        np.zeros((1, 1)),
        np.zeros(1),
        t_max=0.2,
        n_steps=2,
    )
    assert comparisons["product"].attribution_status == (
        "entanglement_specificity_not_established"
    )
    assert comparisons["product"].delta_mean_exchange_coherence == 0.0


def test_legacy_helper_is_descriptive_and_no_advantage_only() -> None:
    """Preserve the old name while removing speedup and advantage semantics."""
    assert entanglement_advantage({}) == {}
    results = compare_all_initial_states(
        build_knm_paper27(L=3),
        OMEGA_N_16[:3],
        t_max=0.2,
        n_steps=2,
    )
    comparisons = entanglement_advantage(results)
    assert set(comparisons) == {"bell_pairs", "ghz", "w_state"}
    for comparison in comparisons.values():
        assert comparison["claim_status"] == "legacy_descriptive_comparison_only"
        assert comparison["entanglement_specific_effect_supported"] is False
        assert "convergence_speedup" not in comparison


def test_legacy_trajectory_defaults_are_explicitly_non_evidentiary() -> None:
    """Keep positional construction compatible without inventing new metrics."""
    trajectory = SyncTrajectory("legacy", [0.0], [0.5], 0.5, 2)
    assert trajectory.final_local_visibility == 0.0
    assert trajectory.final_phase_defined is False
    assert trajectory.final_exchange_coherence == 0.0
    payload = trajectory.to_dict()
    assert payload["final_local_phase_order"] == 0.5
    assert payload["phase_defined_values"] == []


def test_quantum_speed_limit_uses_real_visibility_aware_order_parameter() -> None:
    """Reach a public QSL threshold through the real local-phase-order consumer."""
    result = compute_qsl(
        np.zeros((1, 1), dtype=np.float64),
        np.array([np.pi / 2.0], dtype=np.float64),
        t_target=0.1,
        dt=0.05,
        R_threshold=0.9,
    )

    assert result.n_qubits == 1
    assert result.tau_actual == pytest.approx(0.05)
