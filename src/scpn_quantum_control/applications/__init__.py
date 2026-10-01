# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Physical Applications
"""Physical system benchmarks and application modules."""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ApplicationPluginBenchmark": (
        "scpn_quantum_control.applications.app_plugins",
        "ApplicationPluginBenchmark",
    ),
    "ApplicationPluginRegistry": (
        "scpn_quantum_control.applications.app_plugins",
        "ApplicationPluginRegistry",
    ),
    "compile_application_problem": (
        "scpn_quantum_control.applications.app_plugins",
        "compile_application_problem",
    ),
    "discover_application_plugins": (
        "scpn_quantum_control.applications.app_plugins",
        "discover_application_plugins",
    ),
    "get_application_plugin": (
        "scpn_quantum_control.applications.app_plugins",
        "get_application_plugin",
    ),
    "get_application_plugin_registry": (
        "scpn_quantum_control.applications.app_plugins",
        "get_application_plugin_registry",
    ),
    "load_application_dataset": (
        "scpn_quantum_control.applications.app_plugins",
        "load_application_dataset",
    ),
    "run_application_benchmark_suite": (
        "scpn_quantum_control.applications.app_plugins",
        "run_application_benchmark_suite",
    ),
    "CrossDomainResult": ("scpn_quantum_control.applications.cross_domain", "CrossDomainResult"),
    "run_cross_domain_validation": (
        "scpn_quantum_control.applications.cross_domain",
        "run_cross_domain_validation",
    ),
    "ApplicationBenchmarkDescriptor": (
        "scpn_quantum_control.applications.dataset_catalog",
        "ApplicationBenchmarkDescriptor",
    ),
    "ApplicationBenchmarkPrivacyAudit": (
        "scpn_quantum_control.applications.dataset_catalog",
        "ApplicationBenchmarkPrivacyAudit",
    ),
    "artifact_to_kuramoto_problem": (
        "scpn_quantum_control.applications.dataset_catalog",
        "artifact_to_kuramoto_problem",
    ),
    "audit_application_benchmark_privacy": (
        "scpn_quantum_control.applications.dataset_catalog",
        "audit_application_benchmark_privacy",
    ),
    "get_application_benchmark_descriptor": (
        "scpn_quantum_control.applications.dataset_catalog",
        "get_application_benchmark_descriptor",
    ),
    "list_application_benchmark_descriptors": (
        "scpn_quantum_control.applications.dataset_catalog",
        "list_application_benchmark_descriptors",
    ),
    "load_application_benchmark_artifact": (
        "scpn_quantum_control.applications.dataset_catalog",
        "load_application_benchmark_artifact",
    ),
    "EEGBenchmarkResult": (
        "scpn_quantum_control.applications.eeg_benchmark",
        "EEGBenchmarkResult",
    ),
    "eeg_benchmark": ("scpn_quantum_control.applications.eeg_benchmark", "eeg_benchmark"),
    "FMOBenchmarkResult": (
        "scpn_quantum_control.applications.fmo_benchmark",
        "FMOBenchmarkResult",
    ),
    "fmo_benchmark": ("scpn_quantum_control.applications.fmo_benchmark", "fmo_benchmark"),
    "fmo_coupling_matrix": (
        "scpn_quantum_control.applications.fmo_benchmark",
        "fmo_coupling_matrix",
    ),
    "APPLICATION_HONESTY_CLAIM_BOUNDARY": (
        "scpn_quantum_control.applications.honesty_kits",
        "APPLICATION_HONESTY_CLAIM_BOUNDARY",
    ),
    "APPLICATION_HONESTY_SCHEMA": (
        "scpn_quantum_control.applications.honesty_kits",
        "APPLICATION_HONESTY_SCHEMA",
    ),
    "FORECASTING_DOMAIN_TAGS": (
        "scpn_quantum_control.applications.honesty_kits",
        "FORECASTING_DOMAIN_TAGS",
    ),
    "ApplicationDataOrigin": (
        "scpn_quantum_control.applications.honesty_kits",
        "ApplicationDataOrigin",
    ),
    "ApplicationHonestyAuditReport": (
        "scpn_quantum_control.applications.honesty_kits",
        "ApplicationHonestyAuditReport",
    ),
    "ApplicationSupportStatus": (
        "scpn_quantum_control.applications.honesty_kits",
        "ApplicationSupportStatus",
    ),
    "DomainApplicationHonestyKit": (
        "scpn_quantum_control.applications.honesty_kits",
        "DomainApplicationHonestyKit",
    ),
    "ForecastingDomainTag": (
        "scpn_quantum_control.applications.honesty_kits",
        "ForecastingDomainTag",
    ),
    "build_application_honesty_audit_report": (
        "scpn_quantum_control.applications.honesty_kits",
        "build_application_honesty_audit_report",
    ),
    "get_domain_application_honesty_kit": (
        "scpn_quantum_control.applications.honesty_kits",
        "get_domain_application_honesty_kit",
    ),
    "get_domain_application_honesty_kit_for_dataset": (
        "scpn_quantum_control.applications.honesty_kits",
        "get_domain_application_honesty_kit_for_dataset",
    ),
    "list_domain_application_honesty_kits": (
        "scpn_quantum_control.applications.honesty_kits",
        "list_domain_application_honesty_kits",
    ),
    "render_application_honesty_audit_markdown": (
        "scpn_quantum_control.applications.honesty_kits",
        "render_application_honesty_audit_markdown",
    ),
    "ITERBenchmarkResult": (
        "scpn_quantum_control.applications.iter_benchmark",
        "ITERBenchmarkResult",
    ),
    "iter_benchmark": ("scpn_quantum_control.applications.iter_benchmark", "iter_benchmark"),
    "JosephsonBenchmarkResult": (
        "scpn_quantum_control.applications.josephson_array",
        "JosephsonBenchmarkResult",
    ),
    "josephson_benchmark": (
        "scpn_quantum_control.applications.josephson_array",
        "josephson_benchmark",
    ),
    "JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "JOSEPHSON_KNM_MAGNITUDE_STUDY_BOUNDARY",
    ),
    "JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "JOSEPHSON_KNM_MAGNITUDE_STUDY_SCHEMA",
    ),
    "JosephsonKnmCandidate": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "JosephsonKnmCandidate",
    ),
    "JosephsonMagnitudeGate": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "JosephsonMagnitudeGate",
    ),
    "JosephsonMagnitudeStudyDesign": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "JosephsonMagnitudeStudyDesign",
    ),
    "build_josephson_knm_magnitude_study_design": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "build_josephson_knm_magnitude_study_design",
    ),
    "render_josephson_knm_magnitude_study_markdown": (
        "scpn_quantum_control.applications.josephson_magnitude_study",
        "render_josephson_knm_magnitude_study_markdown",
    ),
    "PowerGridBenchmarkResult": (
        "scpn_quantum_control.applications.power_grid",
        "PowerGridBenchmarkResult",
    ),
    "power_grid_benchmark": (
        "scpn_quantum_control.applications.power_grid",
        "power_grid_benchmark",
    ),
    "ClassicalESNReadoutResult": (
        "scpn_quantum_control.applications.qrc_baseline",
        "ClassicalESNReadoutResult",
    ),
    "QRCBaselineComparison": (
        "scpn_quantum_control.applications.qrc_baseline",
        "QRCBaselineComparison",
    ),
    "QRCHoldoutComparison": (
        "scpn_quantum_control.applications.qrc_baseline",
        "QRCHoldoutComparison",
    ),
    "classical_esn_feature_matrix": (
        "scpn_quantum_control.applications.qrc_baseline",
        "classical_esn_feature_matrix",
    ),
    "classical_esn_ridge_regression": (
        "scpn_quantum_control.applications.qrc_baseline",
        "classical_esn_ridge_regression",
    ),
    "compare_quantum_reservoir_to_esn": (
        "scpn_quantum_control.applications.qrc_baseline",
        "compare_quantum_reservoir_to_esn",
    ),
    "compare_quantum_reservoir_to_esn_holdout": (
        "scpn_quantum_control.applications.qrc_baseline",
        "compare_quantum_reservoir_to_esn_holdout",
    ),
    "QuantumEVSResult": ("scpn_quantum_control.applications.quantum_evs", "QuantumEVSResult"),
    "quantum_evs_enhance": (
        "scpn_quantum_control.applications.quantum_evs",
        "quantum_evs_enhance",
    ),
    "QuantumKernelResult": (
        "scpn_quantum_control.applications.quantum_kernel",
        "QuantumKernelResult",
    ),
    "canonical_edge_pairs": (
        "scpn_quantum_control.applications.quantum_kernel",
        "canonical_edge_pairs",
    ),
    "compute_kernel_matrix": (
        "scpn_quantum_control.applications.quantum_kernel",
        "compute_kernel_matrix",
    ),
    "encode_topology_edge_features": (
        "scpn_quantum_control.applications.quantum_kernel",
        "encode_topology_edge_features",
    ),
    "ReservoirResult": ("scpn_quantum_control.applications.quantum_reservoir", "ReservoirResult"),
    "reservoir_features": (
        "scpn_quantum_control.applications.quantum_reservoir",
        "reservoir_features",
    ),
    "QRC_PRODUCT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "QRC_PRODUCT_CLAIM_BOUNDARY",
    ),
    "ReservoirLinearObjective": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "ReservoirLinearObjective",
    ),
    "ReservoirTaskKind": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "ReservoirTaskKind",
    ),
    "ReservoirTrainingCertificate": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "ReservoirTrainingCertificate",
    ),
    "SyntheticReservoirDataset": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "SyntheticReservoirDataset",
    ),
    "certify_reservoir_training": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "certify_reservoir_training",
    ),
    "generate_synthetic_reservoir_task": (
        "scpn_quantum_control.applications.quantum_reservoir_product",
        "generate_synthetic_reservoir_task",
    ),
}


class _ExportModule(ModuleType):
    """Keep declared object exports when Python publishes child modules."""

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a module attribute while preserving same-named exports.

        Parameters
        ----------
        name
            Attribute assigned by the import system or a caller.
        value
            Value to publish in the package namespace.

        """
        target = _PUBLIC_EXPORTS.get(name)
        if (
            target is not None
            and target[1] is not None
            and isinstance(value, ModuleType)
            and value.__name__ in {target[0], f"{__name__}.{name}"}
        ):
            value = __getattr__(name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ExportModule


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
