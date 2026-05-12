#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

import numpy as np
import pytest

from scpn_quantum_control.analysis.adaptive_fim_feedback import (
    AdaptiveFIMConfig,
    FIMWitness,
    adaptive_lambda_schedule,
    propose_next_lambda,
)
from scpn_quantum_control.analysis.dla_parity_witness import DLAParityWitness
from scpn_quantum_control.analysis.integrated_information_phi import IntegratedInformationPhi
from scpn_quantum_control.analysis.logical_sync_witness import LogicalSyncWitness
from scpn_quantum_control.analysis.quantum_fisher_information import QuantumFisherInformation
from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter
from scpn_quantum_control.analysis.thermodynamic_witness import ThermodynamicWitness


def test_all_observables_with_real_counts():
    # Realistic 8-qubit measurement counts (example from actual circuit)
    counts = {"00000000": 4523, "11111111": 3187, "01010101": 1740, "10101010": 550}

    dla = DLAParityWitness()
    sync = SyncOrderParameter()
    phi = IntegratedInformationPhi()
    qfi = QuantumFisherInformation()
    thermo = ThermodynamicWitness()
    logical = LogicalSyncWitness()

    dla_res = dla(counts)
    sync_res = sync(counts)
    phi_res = phi(counts, allow_entropy_proxy=True)
    qfi_res = qfi(counts, sync_order=0.95, dla_asymmetry=8.0, allow_proxy_estimate=True)
    thermo_res = thermo(counts, work_samples_joule=[1.19, 1.20, 1.21])
    logical_res = logical(counts, logical_fidelity=0.92)

    print("✅ DLAParityWitness:", dla_res)
    print("✅ SyncOrderParameter:", sync_res)
    print("✅ IntegratedInformationPhi:", phi_res)
    print("✅ QuantumFisherInformation:", qfi_res)
    print("✅ ThermodynamicWitness:", thermo_res)
    print("✅ LogicalSyncWitness:", logical_res)

    assert 0.0 <= sync_res["sync_order"] <= 1.0
    assert abs(dla_res["dla_asymmetry"]) <= 100  # ±100% is the physical maximum
    assert phi_res["is_integrated_information"] == 0.0
    assert "phi" not in phi_res
    assert qfi_res["qfi_available"] == 0.0
    assert qfi_res["is_quantum_fisher_information"] == 0.0
    assert "qfi" not in qfi_res

    print("\nAll observables passed integration test with real circuit counts.")


def test_thermodynamic_witness_requires_explicit_work_input():
    witness = ThermodynamicWitness()

    with pytest.raises(ValueError, match="work_samples_joule"):
        witness({"0": 10, "1": 6})


def test_thermodynamic_witness_reports_sample_statistics_and_jarzynski_residual():
    witness = ThermodynamicWitness()

    result = witness(
        {"0": 10, "1": 6},
        work_samples_joule=[1.0e-21, 1.2e-21, 0.8e-21],
        beta_per_joule=2.5e20,
        delta_free_energy_joule=0.9e-21,
    )

    assert result["mean_work_joule"] == pytest.approx(1.0e-21)
    assert result["work_variance_joule2"] > 0.0
    assert result["dissipated_work_joule"] == pytest.approx(0.1e-21)
    assert "jarzynski_delta_free_energy_joule" in result
    assert "work" not in result


def test_dla_parity_witness_reports_empty_and_balanced_negative_controls():
    witness = DLAParityWitness()

    assert witness({}) == {"dla_asymmetry": 0.0, "odd_robustness": 0.5, "even_robustness": 0.5}
    assert witness({"00": 0, "01": 0}) == {
        "dla_asymmetry": 0.0,
        "odd_robustness": 0.5,
        "even_robustness": 0.5,
    }
    balanced = witness({"00": 5, "01": 5})

    assert balanced["dla_asymmetry"] == pytest.approx(0.0)
    assert balanced["odd_robustness"] == pytest.approx(0.5)
    assert balanced["even_robustness"] == pytest.approx(0.5)
    assert balanced["total_shots"] == 10


def test_dla_parity_witness_keeps_physical_asymmetry_bounds():
    witness = DLAParityWitness()

    all_odd = witness({"01": 7})
    all_even = witness({"00": 7})

    assert all_odd["dla_asymmetry"] == pytest.approx(100.0)
    assert all_even["dla_asymmetry"] == pytest.approx(-100.0)


def test_integrated_information_does_not_report_entropy_proxy_as_phi():
    witness = IntegratedInformationPhi()

    with pytest.raises(NotImplementedError, match="integrated information"):
        witness({"00": 5, "11": 5})


def test_integrated_information_entropy_proxy_is_explicitly_labelled():
    witness = IntegratedInformationPhi()

    result = witness({"00": 5, "11": 5}, allow_entropy_proxy=True)

    assert result["phi_available"] == 0.0
    assert result["entropy_proxy"] > 0.0
    assert result["is_integrated_information"] == 0.0
    assert "phi" not in result


def test_integrated_information_entropy_proxy_handles_empty_counts_without_phi_claim():
    witness = IntegratedInformationPhi()

    result = witness({}, allow_entropy_proxy=True)

    assert result == {
        "phi_available": 0.0,
        "entropy_proxy": 0.0,
        "is_integrated_information": 0.0,
    }


def test_integrated_information_entropy_proxy_rejects_invalid_count_totals():
    witness = IntegratedInformationPhi()

    with pytest.raises(ValueError, match="positive total"):
        witness({"00": 0, "11": 0}, allow_entropy_proxy=True)
    with pytest.raises(ValueError, match="negative"):
        witness({"00": 2, "11": -1}, allow_entropy_proxy=True)


def test_integrated_information_routes_real_hamiltonian_inputs_to_quantum_phi_engine():
    witness = IntegratedInformationPhi()
    coupling_matrix = np.array(
        [
            [0.0, 0.4, 0.2],
            [0.4, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        dtype=float,
    )
    natural_frequencies = np.array([0.1, 0.2, 0.3], dtype=float)

    result = witness(
        coupling_matrix=coupling_matrix,
        natural_frequencies=natural_frequencies,
    )

    assert result["phi_available"] == 1.0
    assert result["is_integrated_information"] == 1.0
    assert result["phi"] >= 0.0
    assert result["phi_max"] >= result["phi"]
    assert result["n_qubits"] == 3.0
    assert result["n_bipartitions"] > 0.0
    assert "entropy_proxy" not in result


def test_integrated_information_rejects_partial_or_invalid_hamiltonian_inputs():
    witness = IntegratedInformationPhi()

    with pytest.raises(ValueError, match="coupling_matrix and natural_frequencies"):
        witness(coupling_matrix=np.eye(2))
    with pytest.raises(ValueError, match="square"):
        witness(coupling_matrix=np.ones((2, 3)), natural_frequencies=np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        witness(coupling_matrix=np.eye(2), natural_frequencies=np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="matching"):
        witness(coupling_matrix=np.eye(2), natural_frequencies=np.ones(3))
    with pytest.raises(ValueError, match="symmetric"):
        witness(
            coupling_matrix=np.array([[0.0, 0.1], [0.2, 0.0]]),
            natural_frequencies=np.array([0.0, 0.1]),
        )


def test_quantum_fisher_information_refuses_proxy_by_default():
    qfi = QuantumFisherInformation()

    with pytest.raises(NotImplementedError, match="coupling_matrix"):
        qfi({"00": 5, "11": 5}, sync_order=0.95, dla_asymmetry=8.0)


def test_quantum_fisher_information_requires_complete_production_inputs():
    qfi = QuantumFisherInformation()

    with pytest.raises(ValueError, match="both coupling_matrix and natural_frequencies"):
        qfi(coupling_matrix=np.eye(2))


def test_quantum_fisher_information_proxy_is_explicitly_labelled():
    qfi = QuantumFisherInformation()

    result = qfi({"00": 5, "11": 5}, sync_order=0.95, dla_asymmetry=8.0, allow_proxy_estimate=True)

    assert result["qfi_available"] == 0.0
    assert result["is_quantum_fisher_information"] == 0.0
    assert result["qfi_proxy"] > 0.0
    assert "qfi" not in result


def test_quantum_fisher_information_proxy_can_derive_inputs_from_counts():
    qfi = QuantumFisherInformation()

    result = qfi({"00": 6, "11": 4}, allow_proxy_estimate=True)

    assert result["qfi_available"] == 0.0
    assert result["is_quantum_fisher_information"] == 0.0
    assert result["sync_order_input"] >= 0.0
    assert abs(result["dla_asymmetry_input"]) <= 100.0
    assert "qfi" not in result


def test_quantum_fisher_information_proxy_requires_all_diagnostic_inputs():
    qfi = QuantumFisherInformation()

    with pytest.raises(ValueError, match="sync_order and dla_asymmetry"):
        qfi(allow_proxy_estimate=True, sync_order=0.5)


def test_quantum_fisher_information_routes_real_hamiltonian_inputs_to_qfi_engine():
    import numpy as np

    qfi = QuantumFisherInformation()
    coupling_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    natural_frequencies = np.array([0.2, -0.2])

    result = qfi(
        coupling_matrix=coupling_matrix,
        natural_frequencies=natural_frequencies,
        coupling_pairs=[(0, 1)],
        n_measurements=5000,
    )

    assert result["qfi_available"] == 1.0
    assert result["is_quantum_fisher_information"] == 1.0
    assert result["qfi"] > 0.0
    assert result["qfi_matrix_shape_0"] == 1.0
    assert result["qfi_matrix_shape_1"] == 1.0
    assert result["precision_bound_for_measurement_budget"] > 0.0
    assert result["n_measurements"] == 5000.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling_matrix": [[0.0, np.nan], [np.nan, 0.0]]}, "finite"),
        ({"coupling_matrix": [0.0, 1.0]}, "square"),
        ({"coupling_matrix": [[0.0, 1.0], [0.5, 0.0]]}, "symmetric"),
        ({"natural_frequencies": [0.2, np.inf]}, "finite"),
        ({"natural_frequencies": [0.2, 0.0, -0.2]}, "matching"),
        ({"n_measurements": 0}, "positive integer"),
        ({"n_measurements": 1.5}, "positive integer"),
        ({"n_measurements": True}, "positive integer"),
        ({"coupling_pairs": [(0,)]}, "coupling_pairs"),
        ({"coupling_pairs": [0]}, "coupling_pairs"),
        ({"coupling_pairs": [(0, 2)]}, "coupling_pairs"),
        ({"coupling_pairs": [(1, 1)]}, "coupling_pairs"),
        ({"coupling_pairs": 3}, "coupling_pairs"),
        ({"coupling_pairs": [(False, 1)]}, "integers"),
    ],
)
def test_quantum_fisher_information_rejects_invalid_production_inputs(kwargs, match):
    qfi = QuantumFisherInformation()
    base_kwargs = {
        "coupling_matrix": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "natural_frequencies": np.array([0.2, -0.2]),
        "coupling_pairs": [(0, 1)],
        "n_measurements": 5000,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        qfi(**base_kwargs)


def test_logical_sync_witness_refuses_fidelity_only_proxy_by_default():
    logical = LogicalSyncWitness()

    with pytest.raises(NotImplementedError, match="counts or probabilities"):
        logical(logical_fidelity=0.92)


def test_logical_sync_witness_fidelity_proxy_is_explicitly_labelled():
    logical = LogicalSyncWitness()

    result = logical(logical_fidelity=0.92, allow_fidelity_proxy=True)

    assert result["logical_sync_available"] == 0.0
    assert result["is_logical_sync_witness"] == 0.0
    assert result["logical_fidelity_proxy"] == pytest.approx(0.92)
    assert "logical_sync_order" not in result
    assert "passes" not in result


def test_logical_sync_witness_infers_spec_from_probability_vector():
    logical = LogicalSyncWitness()
    probabilities = np.array([1.0, 0.0], dtype=np.float64)

    result = logical(probabilities=probabilities)

    assert result["logical_fidelity"] == pytest.approx(1.0)
    assert result["logical_sync_order"] == pytest.approx(1.0)
    assert result["passes"] is True


def test_logical_sync_witness_rejects_missing_and_non_power_probability_inputs():
    logical = LogicalSyncWitness()

    with pytest.raises(ValueError, match="provided"):
        logical()
    with pytest.raises(ValueError, match="power of two"):
        logical(probabilities=np.ones(3, dtype=np.float64) / 3.0)


@pytest.mark.parametrize("logical_fidelity", [float("nan"), float("inf"), -0.1, 1.1])
def test_logical_sync_witness_fidelity_proxy_rejects_invalid_values(logical_fidelity):
    logical = LogicalSyncWitness()

    with pytest.raises(ValueError, match="logical_fidelity"):
        logical(logical_fidelity=logical_fidelity, allow_fidelity_proxy=True)


def test_adaptive_fim_feedback_reduces_lambda_for_bad_leakage_and_deadband_holds():
    witness = FIMWitness(leakage=0.18, retention=0.93, depth=12, shots=4096)
    config = AdaptiveFIMConfig(lambda_min=0.0, lambda_max=1.0, step_gain=2.0, deadband=0.01)

    step = propose_next_lambda(0.5, witness, config)
    held = propose_next_lambda(0.5, FIMWitness(leakage=0.005, retention=0.99), config)

    assert step.lambda_out == pytest.approx(0.14)
    assert step.error_signal == pytest.approx(0.18)
    assert step.clipped is False
    assert "leakage" in step.rationale
    assert held.lambda_out == pytest.approx(0.5)


def test_adaptive_fim_feedback_retention_mode_clips_and_preserves_provenance():
    config = AdaptiveFIMConfig(
        lambda_min=0.2,
        lambda_max=1.0,
        step_gain=2.0,
        target_retention=0.95,
        mode="retention_recovery",
    )
    witnesses = [
        FIMWitness(leakage=0.01, retention=0.4, depth=10, shots=1024),
        FIMWitness(leakage=0.02, retention=0.9, depth=11, shots=2048),
    ]

    schedule = adaptive_lambda_schedule(1.0, witnesses, config)

    assert [step.index for step in schedule] == [0, 1]
    assert schedule[0].lambda_out == pytest.approx(0.2)
    assert schedule[0].clipped is True
    assert schedule[0].witness.depth == 10
    assert schedule[0].witness.shots == 1024
    assert schedule[1].lambda_in == pytest.approx(schedule[0].lambda_out)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: AdaptiveFIMConfig(lambda_min=-0.1), "lambda_min"),
        (lambda: AdaptiveFIMConfig(lambda_min=1.0, lambda_max=0.5), "lambda_max"),
        (lambda: AdaptiveFIMConfig(step_gain=-1.0), "step_gain"),
        (lambda: AdaptiveFIMConfig(deadband=-1.0), "deadband"),
        (lambda: AdaptiveFIMConfig(mode="invalid"), "mode"),
        (lambda: FIMWitness(leakage=-0.1, retention=1.0), "leakage"),
        (lambda: FIMWitness(leakage=0.1, retention=1.2), "retention"),
        (lambda: FIMWitness(leakage=0.1, retention=0.9, depth=-1), "depth"),
        (lambda: FIMWitness(leakage=0.1, retention=0.9, shots=0), "shots"),
    ],
)
def test_adaptive_fim_feedback_rejects_invalid_configuration_and_witnesses(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_adaptive_fim_feedback_rejects_invalid_current_lambda():
    with pytest.raises(ValueError, match="current_lambda"):
        propose_next_lambda(-0.1, FIMWitness(leakage=0.0, retention=1.0))
    with pytest.raises(ValueError, match="current_lambda"):
        propose_next_lambda(float("nan"), FIMWitness(leakage=0.0, retention=1.0))


if __name__ == "__main__":
    test_all_observables_with_real_counts()
