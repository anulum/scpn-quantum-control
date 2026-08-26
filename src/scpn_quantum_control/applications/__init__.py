# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Physical Applications
"""Physical system benchmarks and application modules."""

from .app_plugins import (
    ApplicationPluginBenchmark,
    ApplicationPluginRegistry,
    compile_application_problem,
    discover_application_plugins,
    get_application_plugin,
    get_application_plugin_registry,
    load_application_dataset,
    run_application_benchmark_suite,
)
from .cross_domain import CrossDomainResult, run_cross_domain_validation
from .dataset_catalog import (
    ApplicationBenchmarkDescriptor,
    ApplicationBenchmarkPrivacyAudit,
    artifact_to_kuramoto_problem,
    audit_application_benchmark_privacy,
    get_application_benchmark_descriptor,
    list_application_benchmark_descriptors,
    load_application_benchmark_artifact,
)
from .eeg_benchmark import EEGBenchmarkResult, eeg_benchmark
from .fmo_benchmark import FMOBenchmarkResult, fmo_benchmark, fmo_coupling_matrix
from .honesty_kits import (
    APPLICATION_HONESTY_CLAIM_BOUNDARY,
    APPLICATION_HONESTY_SCHEMA,
    FORECASTING_DOMAIN_TAGS,
    ApplicationDataOrigin,
    ApplicationHonestyAuditReport,
    ApplicationSupportStatus,
    DomainApplicationHonestyKit,
    ForecastingDomainTag,
    build_application_honesty_audit_report,
    get_domain_application_honesty_kit,
    get_domain_application_honesty_kit_for_dataset,
    list_domain_application_honesty_kits,
    render_application_honesty_audit_markdown,
)
from .iter_benchmark import ITERBenchmarkResult, iter_benchmark
from .josephson_array import JosephsonBenchmarkResult, josephson_benchmark
from .josephson_magnitude_study import (
    JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY,
    JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA,
    JosephsonKnmCandidate,
    JosephsonMagnitudeGate,
    JosephsonMagnitudeStudyDesign,
    build_josephson_knm_magnitude_study_design,
    render_josephson_knm_magnitude_study_markdown,
)
from .power_grid import PowerGridBenchmarkResult, power_grid_benchmark
from .qrc_baseline import (
    ClassicalESNReadoutResult,
    QRCBaselineComparison,
    QRCHoldoutComparison,
    classical_esn_feature_matrix,
    classical_esn_ridge_regression,
    compare_quantum_reservoir_to_esn,
    compare_quantum_reservoir_to_esn_holdout,
)
from .quantum_evs import QuantumEVSResult, quantum_evs_enhance
from .quantum_kernel import (
    QuantumKernelResult,
    canonical_edge_pairs,
    compute_kernel_matrix,
    encode_topology_edge_features,
)
from .quantum_reservoir import ReservoirResult, reservoir_features
from .quantum_reservoir_product import (
    QRC_PRODUCT_CLAIM_BOUNDARY,
    ReservoirLinearObjective,
    ReservoirTaskKind,
    ReservoirTrainingCertificate,
    SyntheticReservoirDataset,
    certify_reservoir_training,
    generate_synthetic_reservoir_task,
)

__all__ = [
    "ApplicationBenchmarkDescriptor",
    "ApplicationBenchmarkPrivacyAudit",
    "APPLICATION_HONESTY_CLAIM_BOUNDARY",
    "APPLICATION_HONESTY_SCHEMA",
    "FORECASTING_DOMAIN_TAGS",
    "ApplicationDataOrigin",
    "ApplicationHonestyAuditReport",
    "ApplicationPluginBenchmark",
    "ApplicationPluginRegistry",
    "ApplicationSupportStatus",
    "artifact_to_kuramoto_problem",
    "audit_application_benchmark_privacy",
    "build_application_honesty_audit_report",
    "ClassicalESNReadoutResult",
    "classical_esn_feature_matrix",
    "classical_esn_ridge_regression",
    "compile_application_problem",
    "CrossDomainResult",
    "compare_quantum_reservoir_to_esn",
    "compare_quantum_reservoir_to_esn_holdout",
    "discover_application_plugins",
    "DomainApplicationHonestyKit",
    "ForecastingDomainTag",
    "EEGBenchmarkResult",
    "eeg_benchmark",
    "FMOBenchmarkResult",
    "fmo_benchmark",
    "fmo_coupling_matrix",
    "ITERBenchmarkResult",
    "iter_benchmark",
    "JosephsonBenchmarkResult",
    "JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY",
    "JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA",
    "JosephsonKnmCandidate",
    "JosephsonMagnitudeGate",
    "JosephsonMagnitudeStudyDesign",
    "build_josephson_knm_magnitude_study_design",
    "josephson_benchmark",
    "PowerGridBenchmarkResult",
    "power_grid_benchmark",
    "get_application_benchmark_descriptor",
    "get_application_plugin",
    "get_application_plugin_registry",
    "get_domain_application_honesty_kit",
    "get_domain_application_honesty_kit_for_dataset",
    "list_application_benchmark_descriptors",
    "list_domain_application_honesty_kits",
    "load_application_benchmark_artifact",
    "load_application_dataset",
    "QuantumEVSResult",
    "quantum_evs_enhance",
    "QuantumKernelResult",
    "canonical_edge_pairs",
    "compute_kernel_matrix",
    "encode_topology_edge_features",
    "QRCBaselineComparison",
    "QRCHoldoutComparison",
    "QRC_PRODUCT_CLAIM_BOUNDARY",
    "render_josephson_knm_magnitude_study_markdown",
    "render_application_honesty_audit_markdown",
    "ReservoirResult",
    "ReservoirLinearObjective",
    "ReservoirTaskKind",
    "ReservoirTrainingCertificate",
    "SyntheticReservoirDataset",
    "certify_reservoir_training",
    "generate_synthetic_reservoir_task",
    "reservoir_features",
    "run_application_benchmark_suite",
    "run_cross_domain_validation",
]
