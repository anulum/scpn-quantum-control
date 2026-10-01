# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Forecasting
"""Forecasting benchmarks for observed synchronisation traces.

Includes the optional PyTorch DeepONet neural-operator surrogate for Kuramoto dynamics
(:mod:`oscillatools.neural_operator`, re-exported here for backward compatibility); its dataset
builder is pure NumPy, while training and forecasting require ``oscillatools[torch]`` behind a lazy
import. The surrogate's honest advantage over direct simulation — held-out fidelity against a
persistence baseline plus the host-independent operation-count crossover — is quantified by
:mod:`.neural_operator_advantage`, whose arithmetic core lives in the pure-NumPy
:mod:`.neural_operator_cost_model`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscillatools.neural_operator import (
        KuramotoOperatorDataset,
        TrainedKuramotoOperator,
        simulate_operator_dataset,
        train_kuramoto_neural_operator,
    )

    from .multimodal_bridge import (
        ForecastActiveSensingBridge,
        ForecastControllerInitialisation,
        forecast_to_controller_initialisation,
        plan_forecast_active_sensing,
    )
    from .multimodal_forecaster import (
        DomainForecastAccuracy,
        ForecastAccuracyCertificate,
        MultimodalPointForecast,
        MultimodalRidgeForecaster,
        evaluate_point_forecast,
        fit_multimodal_ridge_forecaster,
    )
    from .multimodal_report import (
        MULTIMODAL_EVIDENCE_BOUNDARY,
        MULTIMODAL_EVIDENCE_SCHEMA,
        MultimodalForecastingEvidence,
        MultimodalSupportRow,
        render_multimodal_forecasting_markdown,
        write_multimodal_forecasting_evidence,
    )
    from .multimodal_schema import (
        MultimodalObservationBatch,
        SyntheticDomainTag,
        assert_disjoint_batches,
    )
    from .neural_operator_advantage import (
        HeldOutFidelity,
        NeuralOperatorAdvantage,
        evaluate_neural_operator_advantage,
    )
    from .neural_operator_cost_model import SurrogateCostModel, build_cost_model
    from .partial_observation import (
        PartialObservationBatchCertificate,
        PartialObservationScore,
        PartialObservationWeights,
        evaluate_partial_observation_batch,
        evaluate_partial_observation_objective,
    )
    from .real_data_sync import (
        ForecastModelRun,
        SynchronisationForecastBenchmarkResult,
        SynchronisationForecastDataset,
        load_hardware_kuramoto_4osc_trace,
        load_ieee5bus_sync_forecast_case,
        run_real_data_sync_forecast_benchmark,
        run_real_data_sync_forecast_suite,
    )
    from .synthetic_multimodal import (
        SYNTHETIC_MULTIMODAL_SOURCE,
        SyntheticMultimodalConfig,
        SyntheticMultimodalDataset,
        generate_synthetic_multimodal_dataset,
    )
    from .uncertainty import (
        DomainIntervalCoverage,
        IntervalCoverageCertificate,
        MultimodalIntervalForecast,
        ResidualIntervalCalibrator,
        apply_residual_interval,
        certify_interval_coverage,
        fit_residual_interval_calibrator,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "KuramotoOperatorDataset": ("oscillatools.neural_operator", "KuramotoOperatorDataset"),
    "TrainedKuramotoOperator": ("oscillatools.neural_operator", "TrainedKuramotoOperator"),
    "simulate_operator_dataset": ("oscillatools.neural_operator", "simulate_operator_dataset"),
    "train_kuramoto_neural_operator": (
        "oscillatools.neural_operator",
        "train_kuramoto_neural_operator",
    ),
    "ForecastActiveSensingBridge": (
        "scpn_quantum_control.forecasting.multimodal_bridge",
        "ForecastActiveSensingBridge",
    ),
    "ForecastControllerInitialisation": (
        "scpn_quantum_control.forecasting.multimodal_bridge",
        "ForecastControllerInitialisation",
    ),
    "forecast_to_controller_initialisation": (
        "scpn_quantum_control.forecasting.multimodal_bridge",
        "forecast_to_controller_initialisation",
    ),
    "plan_forecast_active_sensing": (
        "scpn_quantum_control.forecasting.multimodal_bridge",
        "plan_forecast_active_sensing",
    ),
    "DomainForecastAccuracy": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "DomainForecastAccuracy",
    ),
    "ForecastAccuracyCertificate": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "ForecastAccuracyCertificate",
    ),
    "MultimodalPointForecast": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "MultimodalPointForecast",
    ),
    "MultimodalRidgeForecaster": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "MultimodalRidgeForecaster",
    ),
    "evaluate_point_forecast": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "evaluate_point_forecast",
    ),
    "fit_multimodal_ridge_forecaster": (
        "scpn_quantum_control.forecasting.multimodal_forecaster",
        "fit_multimodal_ridge_forecaster",
    ),
    "MULTIMODAL_EVIDENCE_BOUNDARY": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "MULTIMODAL_EVIDENCE_BOUNDARY",
    ),
    "MULTIMODAL_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "MULTIMODAL_EVIDENCE_SCHEMA",
    ),
    "MultimodalForecastingEvidence": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "MultimodalForecastingEvidence",
    ),
    "MultimodalSupportRow": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "MultimodalSupportRow",
    ),
    "render_multimodal_forecasting_markdown": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "render_multimodal_forecasting_markdown",
    ),
    "write_multimodal_forecasting_evidence": (
        "scpn_quantum_control.forecasting.multimodal_report",
        "write_multimodal_forecasting_evidence",
    ),
    "MultimodalObservationBatch": (
        "scpn_quantum_control.forecasting.multimodal_schema",
        "MultimodalObservationBatch",
    ),
    "SyntheticDomainTag": (
        "scpn_quantum_control.forecasting.multimodal_schema",
        "SyntheticDomainTag",
    ),
    "assert_disjoint_batches": (
        "scpn_quantum_control.forecasting.multimodal_schema",
        "assert_disjoint_batches",
    ),
    "HeldOutFidelity": (
        "scpn_quantum_control.forecasting.neural_operator_advantage",
        "HeldOutFidelity",
    ),
    "NeuralOperatorAdvantage": (
        "scpn_quantum_control.forecasting.neural_operator_advantage",
        "NeuralOperatorAdvantage",
    ),
    "evaluate_neural_operator_advantage": (
        "scpn_quantum_control.forecasting.neural_operator_advantage",
        "evaluate_neural_operator_advantage",
    ),
    "SurrogateCostModel": (
        "scpn_quantum_control.forecasting.neural_operator_cost_model",
        "SurrogateCostModel",
    ),
    "build_cost_model": (
        "scpn_quantum_control.forecasting.neural_operator_cost_model",
        "build_cost_model",
    ),
    "PartialObservationBatchCertificate": (
        "scpn_quantum_control.forecasting.partial_observation",
        "PartialObservationBatchCertificate",
    ),
    "PartialObservationScore": (
        "scpn_quantum_control.forecasting.partial_observation",
        "PartialObservationScore",
    ),
    "PartialObservationWeights": (
        "scpn_quantum_control.forecasting.partial_observation",
        "PartialObservationWeights",
    ),
    "evaluate_partial_observation_batch": (
        "scpn_quantum_control.forecasting.partial_observation",
        "evaluate_partial_observation_batch",
    ),
    "evaluate_partial_observation_objective": (
        "scpn_quantum_control.forecasting.partial_observation",
        "evaluate_partial_observation_objective",
    ),
    "ForecastModelRun": ("scpn_quantum_control.forecasting.real_data_sync", "ForecastModelRun"),
    "SynchronisationForecastBenchmarkResult": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "SynchronisationForecastBenchmarkResult",
    ),
    "SynchronisationForecastDataset": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "SynchronisationForecastDataset",
    ),
    "load_hardware_kuramoto_4osc_trace": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "load_hardware_kuramoto_4osc_trace",
    ),
    "load_ieee5bus_sync_forecast_case": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "load_ieee5bus_sync_forecast_case",
    ),
    "run_real_data_sync_forecast_benchmark": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "run_real_data_sync_forecast_benchmark",
    ),
    "run_real_data_sync_forecast_suite": (
        "scpn_quantum_control.forecasting.real_data_sync",
        "run_real_data_sync_forecast_suite",
    ),
    "SYNTHETIC_MULTIMODAL_SOURCE": (
        "scpn_quantum_control.forecasting.synthetic_multimodal",
        "SYNTHETIC_MULTIMODAL_SOURCE",
    ),
    "SyntheticMultimodalConfig": (
        "scpn_quantum_control.forecasting.synthetic_multimodal",
        "SyntheticMultimodalConfig",
    ),
    "SyntheticMultimodalDataset": (
        "scpn_quantum_control.forecasting.synthetic_multimodal",
        "SyntheticMultimodalDataset",
    ),
    "generate_synthetic_multimodal_dataset": (
        "scpn_quantum_control.forecasting.synthetic_multimodal",
        "generate_synthetic_multimodal_dataset",
    ),
    "DomainIntervalCoverage": (
        "scpn_quantum_control.forecasting.uncertainty",
        "DomainIntervalCoverage",
    ),
    "IntervalCoverageCertificate": (
        "scpn_quantum_control.forecasting.uncertainty",
        "IntervalCoverageCertificate",
    ),
    "MultimodalIntervalForecast": (
        "scpn_quantum_control.forecasting.uncertainty",
        "MultimodalIntervalForecast",
    ),
    "ResidualIntervalCalibrator": (
        "scpn_quantum_control.forecasting.uncertainty",
        "ResidualIntervalCalibrator",
    ),
    "apply_residual_interval": (
        "scpn_quantum_control.forecasting.uncertainty",
        "apply_residual_interval",
    ),
    "certify_interval_coverage": (
        "scpn_quantum_control.forecasting.uncertainty",
        "certify_interval_coverage",
    ),
    "fit_residual_interval_calibrator": (
        "scpn_quantum_control.forecasting.uncertainty",
        "fit_residual_interval_calibrator",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "MULTIMODAL_EVIDENCE_BOUNDARY",
    "MULTIMODAL_EVIDENCE_SCHEMA",
    "DomainForecastAccuracy",
    "DomainIntervalCoverage",
    "ForecastAccuracyCertificate",
    "ForecastActiveSensingBridge",
    "ForecastControllerInitialisation",
    "ForecastModelRun",
    "HeldOutFidelity",
    "IntervalCoverageCertificate",
    "KuramotoOperatorDataset",
    "MultimodalForecastingEvidence",
    "MultimodalIntervalForecast",
    "MultimodalObservationBatch",
    "MultimodalPointForecast",
    "MultimodalRidgeForecaster",
    "MultimodalSupportRow",
    "NeuralOperatorAdvantage",
    "PartialObservationBatchCertificate",
    "PartialObservationScore",
    "PartialObservationWeights",
    "ResidualIntervalCalibrator",
    "SYNTHETIC_MULTIMODAL_SOURCE",
    "SurrogateCostModel",
    "SyntheticDomainTag",
    "SyntheticMultimodalConfig",
    "SyntheticMultimodalDataset",
    "SynchronisationForecastBenchmarkResult",
    "SynchronisationForecastDataset",
    "TrainedKuramotoOperator",
    "apply_residual_interval",
    "assert_disjoint_batches",
    "build_cost_model",
    "certify_interval_coverage",
    "evaluate_neural_operator_advantage",
    "evaluate_partial_observation_batch",
    "evaluate_partial_observation_objective",
    "evaluate_point_forecast",
    "fit_multimodal_ridge_forecaster",
    "fit_residual_interval_calibrator",
    "forecast_to_controller_initialisation",
    "generate_synthetic_multimodal_dataset",
    "load_hardware_kuramoto_4osc_trace",
    "load_ieee5bus_sync_forecast_case",
    "run_real_data_sync_forecast_benchmark",
    "run_real_data_sync_forecast_suite",
    "plan_forecast_active_sensing",
    "render_multimodal_forecasting_markdown",
    "simulate_operator_dataset",
    "train_kuramoto_neural_operator",
    "write_multimodal_forecasting_evidence",
]
