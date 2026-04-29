# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Real-Data Synchronisation Forecasting
"""Tests for held-out synchronisation forecasting benchmarks."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from scpn_quantum_control.forecasting import (
    ForecastModelRun,
    SynchronisationForecastBenchmarkResult,
    SynchronisationForecastDataset,
    load_hardware_kuramoto_4osc_trace,
    load_ieee5bus_sync_forecast_case,
    run_real_data_sync_forecast_benchmark,
    run_real_data_sync_forecast_suite,
)


def test_hardware_trace_loader_preserves_committed_measurements():
    dataset = load_hardware_kuramoto_4osc_trace()

    assert dataset.name == "ibm_heron_r2_kuramoto_4osc"
    assert dataset.source_kind == "qpu_hardware_measurement"
    assert dataset.source_path == "results/hw_kuramoto_4osc.json"
    assert "\\" not in dataset.source_path
    np.testing.assert_allclose(dataset.times, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(dataset.observed_order_parameter[-1], 0.3757)
    assert dataset.holdout_size == 2
    assert dataset.provenance["job_id"] == "d6h2qbf3o3rs73caft20"


def test_hardware_forecast_improves_held_out_mse_without_holdout_fit():
    result = run_real_data_sync_forecast_benchmark(load_hardware_kuramoto_4osc_trace())

    assert isinstance(result, SynchronisationForecastBenchmarkResult)
    assert result.passes
    assert result.improvement_fraction > 0.90
    assert result.baseline.holdout_mse > result.calibrated.holdout_mse
    assert result.calibrated.metadata["holdout_values_visible_to_fit"] is False
    np.testing.assert_allclose(
        result.calibrated.predictions[: result.dataset.train_size],
        result.dataset.observed_order_parameter[: result.dataset.train_size],
        atol=1e-12,
    )


def test_ieee5bus_case_uses_public_topology_and_rust_or_python_baseline():
    dataset = load_ieee5bus_sync_forecast_case()

    assert dataset.domain == "power-grid"
    assert dataset.source_kind == "public_topology_classical_replay"
    assert dataset.n_oscillators == 5
    assert dataset.baseline_order_parameter is not None
    assert dataset.provenance["baseline_backend"] in {
        "rust:kuramoto_trajectory",
        "python:euler",
    }
    assert np.all(np.diff(dataset.times) > 0)
    assert np.max(np.abs(dataset.observed_order_parameter - dataset.baseline_order_parameter)) > 0


def test_ieee5bus_forecast_passes_with_fixed_train_window():
    dataset = load_ieee5bus_sync_forecast_case()
    result = run_real_data_sync_forecast_benchmark(dataset, min_improvement_fraction=0.01)

    assert result.passes
    assert result.dataset.train_size == 4
    assert result.baseline.backend.startswith("dataset:baseline_R:")
    assert result.calibrated.holdout_mae < result.baseline.holdout_mae


def test_suite_runs_hardware_and_topology_cases():
    suite = run_real_data_sync_forecast_suite(include_topology_replay=True)

    assert [result.dataset.name for result in suite] == [
        "ibm_heron_r2_kuramoto_4osc",
        "ieee5bus_frequency_disturbance",
    ]
    assert all(result.passes for result in suite)
    assert all(result.as_dict()["passes"] for result in suite)


def test_result_serialisation_contains_plain_json_values():
    result = run_real_data_sync_forecast_benchmark(load_hardware_kuramoto_4osc_trace())
    payload = result.as_dict()

    encoded = json.dumps(payload, sort_keys=True)
    assert "ibm_heron_r2_kuramoto_4osc" in encoded
    assert isinstance(payload["baseline"]["predictions"], list)
    assert isinstance(payload["calibrated"]["elapsed_ms"], float)


def test_dataset_validation_rejects_no_holdout_window():
    good = load_hardware_kuramoto_4osc_trace()
    bad = SynchronisationForecastDataset(
        name=good.name,
        domain=good.domain,
        source_path=good.source_path,
        times=good.times,
        observed_order_parameter=good.observed_order_parameter,
        baseline_order_parameter=good.baseline_order_parameter,
        coupling=good.coupling,
        omega=good.omega,
        theta0=good.theta0,
        train_size=good.times.size,
        source_kind=good.source_kind,
    )

    with pytest.raises(ValueError, match="train_size"):
        run_real_data_sync_forecast_benchmark(bad)


def test_validation_rejects_unbounded_observed_trace():
    good = load_hardware_kuramoto_4osc_trace()
    bad = SynchronisationForecastDataset(
        name=good.name,
        domain=good.domain,
        source_path=good.source_path,
        times=good.times,
        observed_order_parameter=np.array([0.1, 0.2, 1.2, 0.3], dtype=np.float64),
        baseline_order_parameter=good.baseline_order_parameter,
        coupling=good.coupling,
        omega=good.omega,
        theta0=good.theta0,
        train_size=good.train_size,
        source_kind=good.source_kind,
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run_real_data_sync_forecast_benchmark(bad)


def test_forecast_model_run_is_public_result_envelope():
    result = run_real_data_sync_forecast_benchmark(load_hardware_kuramoto_4osc_trace())

    assert isinstance(result.baseline, ForecastModelRun)
    assert isinstance(result.calibrated, ForecastModelRun)
    assert result.baseline.predictions.shape == result.dataset.observed_order_parameter.shape


def test_cli_hardware_only_outputs_json():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_data_sync_forecast_benchmark.py",
            "--hardware-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["suite"] == "real_data_sync_forecasting"
    assert payload["passes"] is True
    assert payload["results"][0]["dataset"]["source_kind"] == "qpu_hardware_measurement"
