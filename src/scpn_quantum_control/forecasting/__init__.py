# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Forecasting
"""Forecasting benchmarks for observed synchronisation traces."""

from .real_data_sync import (
    ForecastModelRun,
    SynchronisationForecastBenchmarkResult,
    SynchronisationForecastDataset,
    load_hardware_kuramoto_4osc_trace,
    load_ieee5bus_sync_forecast_case,
    run_real_data_sync_forecast_benchmark,
    run_real_data_sync_forecast_suite,
)

__all__ = [
    "ForecastModelRun",
    "SynchronisationForecastBenchmarkResult",
    "SynchronisationForecastDataset",
    "load_hardware_kuramoto_4osc_trace",
    "load_ieee5bus_sync_forecast_case",
    "run_real_data_sync_forecast_benchmark",
    "run_real_data_sync_forecast_suite",
]
