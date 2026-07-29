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
    BL37_EVIDENCE_BOUNDARY,
    BL37_EVIDENCE_SCHEMA,
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

__all__ = [
    "BL37_EVIDENCE_BOUNDARY",
    "BL37_EVIDENCE_SCHEMA",
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
