# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Domain Benchmark Datasets
"""Exact-answer differentiable benchmark datasets for bounded phase models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    predict_parameter_shift_qnn_classifier,
)

FloatArray: TypeAlias = NDArray[np.float64]

EVIDENCE_CLASS = "synthetic_exact_answer"
CLAIM_BOUNDARY = (
    "synthetic exact-answer datasets validate bounded differentiable math and "
    "training fixtures; they are not hardware, external measured-system, or "
    "isolated performance benchmark evidence"
)
PUBLISHED_CLAIM_BOUNDARY = (
    "published domain benchmark cases reference committed public QPU data "
    "artifacts and validate their Kuramoto conversion path; they are not "
    "hardware execution, live provider, or performance benchmark evidence"
)


@dataclass(frozen=True)
class DifferentiableQNNExactAnswerCase:
    """Exact-answer dataset for the bounded phase-QNN classifier."""

    dataset_id: str
    features: FloatArray
    labels: FloatArray
    params: FloatArray
    expected_probabilities: FloatArray
    expected_loss: float
    expected_gradient: FloatArray
    source_kind: str
    analytic_contract: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def n_samples(self) -> int:
        """Return the number of labelled samples."""
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Return the number of trainable phase features."""
        return int(self.features.shape[1])

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready exact-answer QNN dataset metadata."""
        return {
            "dataset_id": self.dataset_id,
            "domain": "bounded_phase_qnn",
            "source_kind": self.source_kind,
            "features": self.features.tolist(),
            "labels": self.labels.tolist(),
            "params": self.params.tolist(),
            "expected_probabilities": self.expected_probabilities.tolist(),
            "expected_loss": self.expected_loss,
            "expected_gradient": self.expected_gradient.tolist(),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "analytic_contract": self.analytic_contract,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableKuramotoExactAnswerCase:
    """Exact-answer two-oscillator Kuramoto-XY phase dataset."""

    dataset_id: str
    phases: FloatArray
    coupling: FloatArray
    natural_frequencies: FloatArray
    expected_order_parameter: float
    expected_mean_phase: float
    expected_xy_energy: float
    expected_energy_gradient: FloatArray
    source_kind: str
    analytic_contract: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def n_oscillators(self) -> int:
        """Return the oscillator count."""
        return int(self.phases.size)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready exact-answer Kuramoto-XY dataset metadata."""
        return {
            "dataset_id": self.dataset_id,
            "domain": "kuramoto_xy_pair",
            "source_kind": self.source_kind,
            "phases": self.phases.tolist(),
            "coupling": self.coupling.tolist(),
            "natural_frequencies": self.natural_frequencies.tolist(),
            "expected_order_parameter": self.expected_order_parameter,
            "expected_mean_phase": self.expected_mean_phase,
            "expected_xy_energy": self.expected_xy_energy,
            "expected_energy_gradient": self.expected_energy_gradient.tolist(),
            "n_oscillators": self.n_oscillators,
            "analytic_contract": self.analytic_contract,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDomainBenchmarkDatasetSuite:
    """Bundle of exact-answer differentiable domain datasets."""

    qnn_cases: tuple[DifferentiableQNNExactAnswerCase, ...]
    kuramoto_cases: tuple[DifferentiableKuramotoExactAnswerCase, ...]
    evidence_class: str
    claim_boundary: str

    @property
    def case_count(self) -> int:
        """Return the total number of exact-answer cases."""
        return len(self.qnn_cases) + len(self.kuramoto_cases)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Return all dataset identifiers in deterministic order."""
        return tuple(case.dataset_id for case in self.qnn_cases) + tuple(
            case.dataset_id for case in self.kuramoto_cases
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dataset-suite metadata."""
        return {
            "case_count": self.case_count,
            "dataset_ids": list(self.dataset_ids),
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "qnn_cases": [case.to_dict() for case in self.qnn_cases],
            "kuramoto_cases": [case.to_dict() for case in self.kuramoto_cases],
        }


@dataclass(frozen=True)
class DifferentiableDomainBenchmarkValidationResult:
    """Validation result for synthetic exact-answer domain datasets."""

    dataset_id: str
    domain: str
    max_abs_error: float
    tolerance: float
    passed: bool
    checked_quantities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation evidence."""
        return {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "checked_quantities": list(self.checked_quantities),
        }


@dataclass(frozen=True)
class DifferentiableDomainBenchmarkValidationSuite:
    """Validation evidence for exact-answer differentiable domain datasets."""

    results: tuple[DifferentiableDomainBenchmarkValidationResult, ...]
    evidence_class: str
    claim_boundary: str

    @property
    def passed(self) -> bool:
        """Return whether every exact-answer dataset validated."""
        return all(result.passed for result in self.results)

    @property
    def case_count(self) -> int:
        """Return the number of validated datasets."""
        return len(self.results)

    @property
    def max_abs_error(self) -> float:
        """Return the worst validation error across the suite."""
        if not self.results:
            return 0.0
        return max(result.max_abs_error for result in self.results)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation-suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "max_abs_error": self.max_abs_error,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class DifferentiablePublishedDomainBenchmarkCase:
    """Published public-domain benchmark case backed by a QPU artifact."""

    dataset_id: str
    domain: str
    source_reference: str
    source_licence: str
    transform: str
    artifact_path: str
    artifact_sha256: str
    n_oscillators: int
    coupling_frobenius_norm: float
    omega_l2_norm: float
    source_mode: str
    publication_safe: bool
    evidence_class: str = "published_domain_artifact"
    claim_boundary: str = PUBLISHED_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready published-domain benchmark metadata."""
        return {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "source_reference": self.source_reference,
            "source_licence": self.source_licence,
            "transform": self.transform,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "n_oscillators": self.n_oscillators,
            "coupling_frobenius_norm": self.coupling_frobenius_norm,
            "omega_l2_norm": self.omega_l2_norm,
            "source_mode": self.source_mode,
            "publication_safe": self.publication_safe,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiablePublishedDomainBenchmarkSuite:
    """Published public-domain differentiable benchmark case index."""

    cases: tuple[DifferentiablePublishedDomainBenchmarkCase, ...]
    evidence_class: str
    claim_boundary: str

    @property
    def case_count(self) -> int:
        """Return the number of published-domain cases."""
        return len(self.cases)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Return published-domain dataset identifiers."""
        return tuple(case.dataset_id for case in self.cases)

    def case_by_id(self, dataset_id: str) -> DifferentiablePublishedDomainBenchmarkCase:
        """Return one published-domain case by dataset identifier."""
        for case in self.cases:
            if case.dataset_id == dataset_id:
                return case
        raise KeyError(f"unknown published differentiable domain dataset: {dataset_id}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready published-domain benchmark suite metadata."""
        return {
            "case_count": self.case_count,
            "dataset_ids": list(self.dataset_ids),
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class DifferentiablePublishedDomainBenchmarkValidationResult:
    """Validation result for one published public-domain benchmark case."""

    dataset_id: str
    domain: str
    artifact_sha256: str
    n_oscillators: int
    publication_safe: bool
    kuramoto_conversion_passed: bool
    metadata_roundtrip_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready published-domain validation evidence."""
        return {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "artifact_sha256": self.artifact_sha256,
            "n_oscillators": self.n_oscillators,
            "publication_safe": self.publication_safe,
            "kuramoto_conversion_passed": self.kuramoto_conversion_passed,
            "metadata_roundtrip_passed": self.metadata_roundtrip_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DifferentiablePublishedDomainBenchmarkValidationSuite:
    """Validation evidence for published public-domain benchmark cases."""

    results: tuple[DifferentiablePublishedDomainBenchmarkValidationResult, ...]
    evidence_class: str
    claim_boundary: str

    @property
    def passed(self) -> bool:
        """Return whether every published-domain case validated."""
        return all(result.passed for result in self.results)

    @property
    def case_count(self) -> int:
        """Return the number of validated cases."""
        return len(self.results)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready published-domain validation suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "results": [result.to_dict() for result in self.results],
        }


def _qnn_probabilities(features: FloatArray, params: FloatArray) -> FloatArray:
    probabilities = 0.5 * (1.0 - np.cos(features + params[None, :]))
    return cast(FloatArray, np.mean(probabilities, axis=1).astype(np.float64, copy=False))


def _qnn_case(
    *,
    dataset_id: str,
    features: FloatArray,
    labels: FloatArray,
    params: FloatArray,
    analytic_contract: str,
) -> DifferentiableQNNExactAnswerCase:
    probabilities = _qnn_probabilities(features, params)
    residual = probabilities - labels
    loss = float(np.mean(residual * residual))
    gradient = parameter_shift_qnn_classifier_gradient(features, labels, params)
    return DifferentiableQNNExactAnswerCase(
        dataset_id=dataset_id,
        features=features,
        labels=labels,
        params=params,
        expected_probabilities=probabilities,
        expected_loss=loss,
        expected_gradient=gradient,
        source_kind="synthetic_closed_form",
        analytic_contract=analytic_contract,
    )


def _kuramoto_pair_case(
    *,
    dataset_id: str,
    coupling_strength: float,
    phase_gap: float,
) -> DifferentiableKuramotoExactAnswerCase:
    k = float(coupling_strength)
    delta = float(phase_gap)
    phases = np.array([0.0, delta], dtype=np.float64)
    coupling = np.array([[0.0, k], [k, 0.0]], dtype=np.float64)
    natural_frequencies = np.zeros(2, dtype=np.float64)
    order_parameter = float(abs((1.0 + np.exp(1j * delta)) / 2.0))
    mean_phase = float(np.angle(1.0 + np.exp(1j * delta)))
    energy = float(-k * np.cos(delta))
    gradient = np.array([-k * np.sin(delta), k * np.sin(delta)], dtype=np.float64)
    return DifferentiableKuramotoExactAnswerCase(
        dataset_id=dataset_id,
        phases=phases,
        coupling=coupling,
        natural_frequencies=natural_frequencies,
        expected_order_parameter=order_parameter,
        expected_mean_phase=mean_phase,
        expected_xy_energy=energy,
        expected_energy_gradient=gradient,
        source_kind="synthetic_closed_form",
        analytic_contract=(
            "two-oscillator XY phase energy E=-K*cos(theta_1-theta_0), "
            "gradient=(-K*sin(delta), K*sin(delta)), and "
            "R=|(exp(i theta_0)+exp(i theta_1))/2|"
        ),
    )


def load_differentiable_domain_benchmark_datasets(
    *,
    dataset_ids: Sequence[str] | None = None,
) -> DifferentiableDomainBenchmarkDatasetSuite:
    """Return deterministic exact-answer datasets for differentiable checks."""
    qnn_cases = (
        _qnn_case(
            dataset_id="bounded_qnn_phase_separable_exact",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            params=np.array([0.0], dtype=np.float64),
            analytic_contract=(
                "single-feature bounded phase response p=0.5*(1-cos(x+theta)) "
                "has exact labels and zero MSE gradient at theta=0"
            ),
        ),
        _qnn_case(
            dataset_id="bounded_qnn_two_feature_mixed_exact",
            features=np.array([[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]], dtype=np.float64),
            labels=np.array([0.0, 1.0, 0.25], dtype=np.float64),
            params=np.array([0.4, -0.2], dtype=np.float64),
            analytic_contract=(
                "two-feature bounded phase response with full-batch MSE and "
                "multi-frequency parameter-shift gradient"
            ),
        ),
    )
    kuramoto_cases = (
        _kuramoto_pair_case(
            dataset_id="kuramoto_xy_two_oscillator_pi_over_3",
            coupling_strength=0.5,
            phase_gap=float(np.pi / 3.0),
        ),
    )
    suite = DifferentiableDomainBenchmarkDatasetSuite(
        qnn_cases=qnn_cases,
        kuramoto_cases=kuramoto_cases,
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
    )
    if dataset_ids is None:
        return suite
    selected = {str(dataset_id).strip() for dataset_id in dataset_ids}
    if not selected or any(not dataset_id for dataset_id in selected):
        raise ValueError("dataset_ids must contain non-empty dataset identifiers")
    known = set(suite.dataset_ids)
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown differentiable benchmark dataset: {', '.join(sorted(unknown))}")
    return DifferentiableDomainBenchmarkDatasetSuite(
        qnn_cases=tuple(case for case in qnn_cases if case.dataset_id in selected),
        kuramoto_cases=tuple(case for case in kuramoto_cases if case.dataset_id in selected),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
    )


def _as_non_negative_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return value


def _validate_qnn_case(
    case: DifferentiableQNNExactAnswerCase,
    *,
    tolerance: float,
) -> DifferentiableDomainBenchmarkValidationResult:
    prediction = predict_parameter_shift_qnn_classifier(case.features, case.params)
    loss = parameter_shift_qnn_classifier_loss(case.features, case.labels, case.params)
    gradient = parameter_shift_qnn_classifier_gradient(case.features, case.labels, case.params)
    errors = (
        np.max(np.abs(prediction.probabilities - case.expected_probabilities)),
        abs(loss - case.expected_loss),
        np.max(np.abs(gradient - case.expected_gradient)),
    )
    max_abs_error = float(max(errors))
    return DifferentiableDomainBenchmarkValidationResult(
        dataset_id=case.dataset_id,
        domain="bounded_phase_qnn",
        max_abs_error=max_abs_error,
        tolerance=tolerance,
        passed=bool(max_abs_error <= tolerance),
        checked_quantities=("probabilities", "mse_loss", "parameter_shift_gradient"),
    )


def _kuramoto_xy_pair_energy(phases: FloatArray, coupling: FloatArray) -> float:
    return float(-coupling[0, 1] * np.cos(phases[1] - phases[0]))


def _kuramoto_xy_pair_energy_gradient(phases: FloatArray, coupling: FloatArray) -> FloatArray:
    k = float(coupling[0, 1])
    delta = float(phases[1] - phases[0])
    return np.array([-k * np.sin(delta), k * np.sin(delta)], dtype=np.float64)


def _kuramoto_pair_order_parameter(phases: FloatArray) -> tuple[float, float]:
    z = np.mean(np.exp(1j * phases))
    return float(abs(z)), float(np.angle(z))


def _validate_kuramoto_case(
    case: DifferentiableKuramotoExactAnswerCase,
    *,
    tolerance: float,
) -> DifferentiableDomainBenchmarkValidationResult:
    order_parameter, mean_phase = _kuramoto_pair_order_parameter(case.phases)
    energy = _kuramoto_xy_pair_energy(case.phases, case.coupling)
    gradient = _kuramoto_xy_pair_energy_gradient(case.phases, case.coupling)
    errors = (
        abs(order_parameter - case.expected_order_parameter),
        abs(mean_phase - case.expected_mean_phase),
        abs(energy - case.expected_xy_energy),
        np.max(np.abs(gradient - case.expected_energy_gradient)),
    )
    max_abs_error = float(max(errors))
    return DifferentiableDomainBenchmarkValidationResult(
        dataset_id=case.dataset_id,
        domain="kuramoto_xy_pair",
        max_abs_error=max_abs_error,
        tolerance=tolerance,
        passed=bool(max_abs_error <= tolerance),
        checked_quantities=("order_parameter", "mean_phase", "xy_energy", "energy_gradient"),
    )


def run_differentiable_domain_benchmark_dataset_validation(
    *,
    dataset_ids: Sequence[str] | None = None,
    tolerance: float = 1e-12,
) -> DifferentiableDomainBenchmarkValidationSuite:
    """Validate exact-answer differentiable domain benchmark datasets."""
    tolerance_value = _as_non_negative_tolerance(tolerance)
    datasets = load_differentiable_domain_benchmark_datasets(dataset_ids=dataset_ids)
    results = [
        *(_validate_qnn_case(case, tolerance=tolerance_value) for case in datasets.qnn_cases),
        *(
            _validate_kuramoto_case(case, tolerance=tolerance_value)
            for case in datasets.kuramoto_cases
        ),
    ]
    return DifferentiableDomainBenchmarkValidationSuite(
        results=tuple(results),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
    )


def _selected_dataset_ids(
    *,
    dataset_ids: Sequence[str] | None,
    known: set[str],
    label: str,
) -> set[str] | None:
    if dataset_ids is None:
        return None
    selected = {str(dataset_id).strip() for dataset_id in dataset_ids}
    if not selected or any(not dataset_id for dataset_id in selected):
        raise ValueError(f"{label} must contain non-empty dataset identifiers")
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return selected


def load_differentiable_published_domain_benchmark_cases(
    *,
    dataset_ids: Sequence[str] | None = None,
) -> DifferentiablePublishedDomainBenchmarkSuite:
    """Return published public-domain benchmark cases backed by artifacts."""
    from scpn_quantum_control.applications.dataset_catalog import (
        artifact_to_kuramoto_problem,
        list_application_benchmark_descriptors,
        load_application_benchmark_artifact,
    )

    descriptors = list_application_benchmark_descriptors()
    known = {descriptor.dataset_id for descriptor in descriptors}
    selected = _selected_dataset_ids(dataset_ids=dataset_ids, known=known, label="dataset_ids")
    cases: list[DifferentiablePublishedDomainBenchmarkCase] = []
    for descriptor in descriptors:
        if selected is not None and descriptor.dataset_id not in selected:
            continue
        artifact = load_application_benchmark_artifact(descriptor.dataset_id)
        problem = artifact_to_kuramoto_problem(artifact)
        payload = artifact.to_dict()
        cases.append(
            DifferentiablePublishedDomainBenchmarkCase(
                dataset_id=descriptor.dataset_id,
                domain=descriptor.domain,
                source_reference=descriptor.source_reference,
                source_licence=descriptor.source_licence,
                transform=descriptor.transform,
                artifact_path=str(descriptor.path),
                artifact_sha256=str(payload["artifact_sha256"]),
                n_oscillators=problem.n_oscillators,
                coupling_frobenius_norm=float(np.linalg.norm(problem.K_nm, ord="fro")),
                omega_l2_norm=float(np.linalg.norm(problem.omega, ord=2)),
                source_mode=artifact.source_mode,
                publication_safe=True,
            )
        )
    return DifferentiablePublishedDomainBenchmarkSuite(
        cases=tuple(cases),
        evidence_class="published_domain_artifact",
        claim_boundary=PUBLISHED_CLAIM_BOUNDARY,
    )


def run_differentiable_published_domain_benchmark_validation(
    *,
    dataset_ids: Sequence[str] | None = None,
) -> DifferentiablePublishedDomainBenchmarkValidationSuite:
    """Validate published public-domain benchmark cases and conversions."""
    from scpn_quantum_control.applications.dataset_catalog import (
        artifact_to_kuramoto_problem,
        get_application_benchmark_descriptor,
        load_application_benchmark_artifact,
    )

    suite = load_differentiable_published_domain_benchmark_cases(dataset_ids=dataset_ids)
    results: list[DifferentiablePublishedDomainBenchmarkValidationResult] = []
    for case in suite.cases:
        descriptor = get_application_benchmark_descriptor(case.dataset_id)
        artifact = load_application_benchmark_artifact(case.dataset_id)
        artifact.require_publication_safe()
        problem = artifact_to_kuramoto_problem(artifact)
        payload = artifact.to_dict()
        metadata = dict(problem.metadata)
        metadata_roundtrip_passed = (
            metadata.get("source_name") == descriptor.dataset_id
            and metadata.get("domain") == descriptor.domain
            and problem.n_oscillators == artifact.n_oscillators
        )
        kuramoto_conversion_passed = (
            problem.K_nm.shape == artifact.K_nm.shape
            and problem.omega.shape == artifact.omega.shape
            and np.allclose(problem.K_nm, artifact.K_nm, atol=0.0, rtol=0.0)
            and np.allclose(problem.omega, artifact.omega, atol=0.0, rtol=0.0)
        )
        publication_safe = not artifact.is_synthetic
        results.append(
            DifferentiablePublishedDomainBenchmarkValidationResult(
                dataset_id=case.dataset_id,
                domain=case.domain,
                artifact_sha256=str(payload["artifact_sha256"]),
                n_oscillators=problem.n_oscillators,
                publication_safe=publication_safe,
                kuramoto_conversion_passed=kuramoto_conversion_passed,
                metadata_roundtrip_passed=metadata_roundtrip_passed,
                passed=bool(
                    publication_safe and kuramoto_conversion_passed and metadata_roundtrip_passed
                ),
            )
        )
    return DifferentiablePublishedDomainBenchmarkValidationSuite(
        results=tuple(results),
        evidence_class="published_domain_artifact",
        claim_boundary=PUBLISHED_CLAIM_BOUNDARY,
    )


__all__ = [
    "DifferentiableDomainBenchmarkDatasetSuite",
    "DifferentiableDomainBenchmarkValidationResult",
    "DifferentiableDomainBenchmarkValidationSuite",
    "DifferentiableKuramotoExactAnswerCase",
    "DifferentiablePublishedDomainBenchmarkCase",
    "DifferentiablePublishedDomainBenchmarkSuite",
    "DifferentiablePublishedDomainBenchmarkValidationResult",
    "DifferentiablePublishedDomainBenchmarkValidationSuite",
    "DifferentiableQNNExactAnswerCase",
    "load_differentiable_domain_benchmark_datasets",
    "load_differentiable_published_domain_benchmark_cases",
    "run_differentiable_domain_benchmark_dataset_validation",
    "run_differentiable_published_domain_benchmark_validation",
]
