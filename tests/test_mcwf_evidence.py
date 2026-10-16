# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-bound MCWF sampling evidence tests
"""Exercise actual seeded ensembles through the original product facade."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.open_system_objectives import BoundedOpenSystemObjectiveCase
from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble, mcwf_trajectory


def _case(
    count: int = 5, *, horizon: float = 0.2, dt: float = 0.1
) -> BoundedOpenSystemObjectiveCase:
    """Declare a real one-qubit fixture with explicit source rates and parameter scales."""
    return BoundedOpenSystemObjectiveCase(
        case_id="one_qubit_sampling",
        n_oscillators=1,
        coupling_matrix=np.zeros((1, 1)),
        omega=np.array([0.7]),
        gamma_amp=0.2,
        gamma_deph=0.1,
        initial_params=np.array([1.0, 2.0]),
        target_order_parameter=0.5,
        target_purity=0.8,
        t_max=horizon,
        dt=dt,
        n_trajectories=count,
        seed=7,
    )


def test_sampling_owner_is_registered_in_the_original_product_surface_map() -> None:
    """Expose the sampling companion through the product's existing discovery API."""
    from scpn_quantum_control.open_system_mcwf_product import (
        build_open_system_mcwf_product_registry,
        map_open_system_mcwf_public_surfaces,
    )

    rows = map_open_system_mcwf_public_surfaces()
    assert any(
        row["module_path"] == "scpn_quantum_control.phase.mcwf_evidence"
        and row["role"] == "source_bound_sampling_evidence"
        for row in rows
    )
    assert build_open_system_mcwf_product_registry()["public_surfaces"] == list(rows)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_companion_preserves_the_actual_ensemble_and_reports_sample_uncertainty(
    count: int,
) -> None:
    """Bind original source values, effective rates, child seeds and n-minus-one uncertainty."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    case = _case(count)
    evidence = materialise_mcwf_ensemble_evidence(case)
    coupling, amplitude, dephasing = case.scaled_inputs(case.initial_params)
    original = mcwf_ensemble(
        coupling, case.omega, amplitude, dephasing, case.t_max, case.dt, count, case.seed
    )
    np.testing.assert_array_equal(evidence.trajectory_values, original["R_trajectories"])
    np.testing.assert_array_equal(evidence.mean, original["R_mean"])
    np.testing.assert_array_equal(evidence.times, original["times"])
    for index, seed in enumerate(evidence.trajectory_seeds):
        replay = mcwf_trajectory(
            coupling, case.omega, amplitude, dephasing, case.t_max, case.dt, seed
        )
        np.testing.assert_array_equal(evidence.trajectory_values[index], replay["R"])
    record = evidence.to_dict()
    assert record["source_case"] == case.to_dict()
    assert record["effective_rates"] == {"gamma_amp": 0.4, "gamma_deph": 0.2}
    assert record["total_jumps"] == original["total_jumps"]
    assert record["model_domain"] == "cp_markovian_simulation"
    assert record["time_discretisation_error"] is None
    assert record["confidence_interval"] is None
    if count == 1:
        assert evidence.sample_variance is None
        assert evidence.standard_error is None
        assert record["sample_variance"] is None
        assert record["standard_error"] is None
    else:
        expected = np.var(original["R_trajectories"], axis=0, ddof=1)
        np.testing.assert_array_equal(evidence.sample_variance, expected)
        np.testing.assert_array_equal(evidence.standard_error, np.sqrt(expected / count))
        np.testing.assert_array_equal(record["sample_variance"], expected)
        np.testing.assert_array_equal(record["standard_error"], np.sqrt(expected / count))


def test_companion_zero_horizon_preserves_inputs_rng_and_detaches_exports() -> None:
    """Snapshot source and actual zero-time arrays without mutating caller state."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    case = _case(horizon=0.0)
    before = np.random.get_state(legacy=True)
    assert isinstance(before, tuple)
    coupling, omega, params = (
        case.coupling_matrix.copy(),
        case.omega.copy(),
        case.initial_params.copy(),
    )
    evidence = materialise_mcwf_ensemble_evidence(case)
    after = np.random.get_state(legacy=True)
    assert isinstance(after, tuple)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(case.coupling_matrix, coupling)
    np.testing.assert_array_equal(case.omega, omega)
    np.testing.assert_array_equal(case.initial_params, params)
    np.testing.assert_array_equal(evidence.times, [0.0])
    for values in (evidence.times, evidence.mean, evidence.trajectory_values):
        assert not values.flags.writeable
    first = evidence.to_dict()
    case.omega[0] = 2.0
    assert evidence.to_dict() == first
    exported_mean = first["mean"]
    assert isinstance(exported_mean, list)
    exported_mean[0] = -1.0
    assert evidence.to_dict()["mean"] != exported_mean


@pytest.mark.parametrize("domain", ["non_markovian", "non_cp", "hardware", ""])
def test_companion_refuses_an_unsupported_model_domain(domain: str) -> None:
    """Refuse unqualified model or physical-device claims at the public product boundary."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    with pytest.raises(ValueError, match="CP Markovian simulation"):
        materialise_mcwf_ensemble_evidence(_case(), model_domain=domain)


def test_companion_admits_complete_history_before_execution() -> None:
    """Refuse a real history request above the declared allowance before allocation."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    with pytest.raises(DenseAllocationError, match="execution memory"):
        materialise_mcwf_ensemble_evidence(_case(1000, dt=1e-5), max_dense_gib=1e-6)


def test_nonzero_sampling_variance_matches_an_independent_scalar_oracle() -> None:
    """Distinguish n-minus-one uncertainty on an actual ensemble with both jump outcomes."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    evidence = materialise_mcwf_ensemble_evidence(_case(64, horizon=2.0, dt=0.25))
    samples = [float(value) for value in evidence.trajectory_values[:, -1]]
    assert max(samples) > min(samples)
    mean = math.fsum(samples) / len(samples)
    variance = math.fsum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
    assert evidence.sample_variance is not None
    assert evidence.standard_error is not None
    assert evidence.sample_variance[-1] == pytest.approx(variance, rel=1e-13, abs=1e-14)
    assert evidence.standard_error[-1] == pytest.approx(math.sqrt(variance / 64), rel=1e-13)
    assert not evidence.sample_variance.flags.writeable
    assert not evidence.standard_error.flags.writeable


@pytest.mark.parametrize("step", [float(np.nextafter(0.0, 1.0)), 1e-30])
def test_companion_refuses_unaddressable_grid_before_constructing_history(step: float) -> None:
    """Reject a finite positive step whose requested interval count overflows."""
    from scpn_quantum_control.open_system_mcwf_product import materialise_mcwf_ensemble_evidence

    with pytest.raises(DenseAllocationError, match="addressability"):
        materialise_mcwf_ensemble_evidence(_case(dt=step))
