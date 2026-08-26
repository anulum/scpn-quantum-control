# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — identity-observer identity observer product tests
"""Real-surface tests for fail-closed identity control observers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.identity_observer_product import (
    ATTESTED_RESULT_SEAL_POINTER,
    IDENTITY_OBSERVER_SCHEMA,
    IdentityObserverThresholds,
    evaluate_identity_safety,
    identity_metric_inventory,
    identity_observer_unsuitable_scenarios,
)


def _problem() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        np.array([-0.1, 0.1], dtype=np.float64),
    )


def _thresholds(**overrides: float) -> IdentityObserverThresholds:
    values = {
        "min_energy_gap": 0.1,
        "max_transition_probability": 0.1,
        "min_coherence_fidelity": 0.1,
        "min_chsh_when_observed": 2.0,
    }
    values.update(overrides)
    return IdentityObserverThresholds(**values)


def _bell_state() -> Statevector:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return Statevector.from_instruction(circuit)


def test_inventory_separates_loop_metrics_from_identity_products() -> None:
    rows = identity_metric_inventory()
    safe = {row.metric_id for row in rows if row.loop_safe}
    unsafe = {row.metric_id for row in rows if not row.loop_safe}

    assert safe == {"robustness_gap", "coherence_budget", "entanglement_witness"}
    assert unsafe == {"identity_key", "binding_spec"}
    assert all(row.claim_boundary for row in rows)


def test_unsuitable_scenarios_cover_overinterpretation() -> None:
    scenarios = identity_observer_unsuitable_scenarios()

    assert len(scenarios) == 5
    assert any("unbreakable" in scenario for scenario in scenarios)
    assert any("consciousness" in scenario for scenario in scenarios)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_energy_gap": np.inf}, "finite"),
        ({"min_energy_gap": 0.0}, "min_energy_gap"),
        ({"max_transition_probability": 1.1}, "max_transition_probability"),
        ({"min_coherence_fidelity": 0.0}, "min_coherence_fidelity"),
        ({"min_chsh_when_observed": 3.0}, "min_chsh_when_observed"),
    ],
)
def test_threshold_validation(overrides: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _thresholds(**overrides)


def test_real_robustness_and_coherence_observers_allow() -> None:
    K, omega = _problem()
    decision = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(),
        planned_depth=2,
        n_qubits=2,
        noise_strength=0.001,
        sweep_rate=0.001,
    )

    assert decision.allowed is True
    assert decision.action == "continue"
    assert decision.observer.energy_gap > 0.1
    assert decision.observer.coherence_fidelity > 0.1
    assert decision.observer.witness_status == "not_requested"
    assert decision.observer.seal_pointer == ATTESTED_RESULT_SEAL_POINTER
    payload = decision.to_dict()
    assert payload["schema"] == IDENTITY_OBSERVER_SCHEMA
    assert payload["blockers"] == []


def test_supported_bell_witness_allows() -> None:
    K, omega = _problem()
    decision = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(),
        planned_depth=2,
        n_qubits=2,
        noise_strength=0.001,
        sweep_rate=0.001,
        statevector=_bell_state(),
        witness_pair=(0, 1),
        require_witness=True,
    )

    assert decision.allowed is True
    assert decision.observer.witness_status == "supported"
    assert decision.observer.chsh_value is not None
    assert decision.observer.chsh_value > 2.0
    assert decision.observer.witness_pair == (0, 1)


def test_supported_but_below_threshold_witness_holds() -> None:
    K, omega = _problem()
    product_state = Statevector.from_int(0, 4)
    decision = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(),
        planned_depth=2,
        n_qubits=2,
        noise_strength=0.001,
        statevector=product_state,
        witness_pair=(0, 1),
    )

    assert decision.allowed is False
    assert decision.action == "hold"
    assert any("CHSH value" in blocker for blocker in decision.blockers)


def test_missing_or_invalid_requested_witness_aborts() -> None:
    K, omega = _problem()
    missing = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(),
        planned_depth=2,
        n_qubits=2,
        noise_strength=0.001,
        require_witness=True,
    )
    invalid = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(),
        planned_depth=2,
        n_qubits=2,
        noise_strength=0.001,
        statevector=_bell_state(),
        witness_pair=(0, 0),
    )

    assert missing.action == "abort"
    assert missing.observer.witness_status == "unsupported"
    assert invalid.action == "abort"
    assert invalid.observer.witness_pair is None
    assert any("unsupported" in blocker for blocker in invalid.blockers)


def test_numeric_threshold_and_depth_trips_hold() -> None:
    K, omega = _problem()
    decision = evaluate_identity_safety(
        K,
        omega,
        thresholds=_thresholds(
            min_energy_gap=0.9,
            max_transition_probability=0.0,
            min_coherence_fidelity=0.99,
        ),
        planned_depth=2000,
        n_qubits=2,
        noise_strength=0.01,
    )

    assert decision.allowed is False
    assert decision.action == "hold"
    assert any("energy_gap" in blocker for blocker in decision.blockers)
    assert any("transition_probability" in blocker for blocker in decision.blockers)
    assert any("coherence_fidelity" in blocker for blocker in decision.blockers)
    assert any("coherence budget" in blocker for blocker in decision.blockers)


def test_input_guards_precede_metric_evaluation() -> None:
    K, omega = _problem()
    with pytest.raises(ValueError, match="planned_depth"):
        evaluate_identity_safety(
            K,
            omega,
            thresholds=_thresholds(),
            planned_depth=-1,
            n_qubits=2,
        )
    with pytest.raises(ValueError, match="n_qubits"):
        evaluate_identity_safety(
            K,
            omega,
            thresholds=_thresholds(),
            planned_depth=1,
            n_qubits=0,
        )
