# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-unifier validation fixtures
"""Executable simulator fixtures for Paper 0 EQ0115-EQ0118 anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_information_thermodynamics_validation_spec


@dataclass(frozen=True, slots=True)
class CyclicOperatorConfig:
    """Finite-dimensional cyclic-operator settings for the EQ0115 boundary."""

    dimension: int = 4
    period: int = 7
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or self.dimension < 2:
            raise ValueError("dimension must be at least two")
        if not isinstance(self.period, int) or self.period < 2:
            raise ValueError("period must be at least two")


@dataclass(frozen=True, slots=True)
class CyclicOperatorValidationResult:
    """Source-anchored cyclic-operator validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    unitarity_error: float
    cycle_closure_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class ABLBoundaryConfig:
    """Finite TSVF/ABL settings for the EQ0116 boundary-probability fixture."""

    pre_state: np.ndarray | None = None
    post_state: np.ndarray | None = None
    projectors: tuple[np.ndarray, ...] | None = None
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        pre = _normalised_state(
            "pre_state",
            self.pre_state
            if self.pre_state is not None
            else np.array([np.sqrt(0.7), np.sqrt(0.3)], dtype=np.complex128),
        )
        post = _normalised_state(
            "post_state",
            self.post_state
            if self.post_state is not None
            else np.array([np.sqrt(0.6), np.sqrt(0.4)], dtype=np.complex128),
        )
        if pre.shape != post.shape:
            raise ValueError("pre_state and post_state must have identical shape")
        projectors = (
            self.projectors if self.projectors is not None else _basis_projectors(pre.size)
        )
        validated_projectors = _validated_projectors(projectors, dimension=pre.size)
        object.__setattr__(self, "pre_state", pre)
        object.__setattr__(self, "post_state", post)
        object.__setattr__(self, "projectors", validated_projectors)


@dataclass(frozen=True, slots=True)
class ABLBoundaryValidationResult:
    """Source-anchored TSVF/ABL validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    probability_normalisation_error: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class InformationThermodynamicsConfig:
    """Finite entropy-budget settings for the EQ0117-EQ0118 fixture."""

    thermodynamic_entropy_rate: float = -0.12
    mutual_information_rate: float = 0.4
    landauer_cost_per_nat: float = 0.5
    proportionality: float = 0.3
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "thermodynamic_entropy_rate",
            "mutual_information_rate",
            "landauer_cost_per_nat",
            "proportionality",
        ):
            _require_finite(name, float(getattr(self, name)))
        if self.mutual_information_rate < 0.0:
            raise ValueError("mutual_information_rate must be non-negative")
        if self.landauer_cost_per_nat < 0.0:
            raise ValueError("landauer_cost_per_nat must be non-negative")
        if self.proportionality < 0.0:
            raise ValueError("proportionality must be non-negative")


@dataclass(frozen=True, slots=True)
class EntropyBudgetRates:
    """Computed finite-rate entropy budget for EQ0117-EQ0118."""

    negentropy_rate: float
    information_entropy_rate: float
    total_entropy_rate: float
    mutual_information_expected_negentropy: float
    mutual_information_negentropy_error: float


@dataclass(frozen=True, slots=True)
class InformationThermodynamicsValidationResult:
    """Source-anchored information-thermodynamics validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    negentropy_rate: float
    information_entropy_rate: float
    total_entropy_rate: float
    gsl_margin: float
    mutual_information_negentropy_error: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def build_cyclic_operator(config: CyclicOperatorConfig) -> np.ndarray:
    """Construct a unitary finite cyclic operator with ``U**period = I``."""
    phases = 2.0 * np.pi * np.arange(config.dimension, dtype=np.float64) / config.period
    return cast(np.ndarray, np.diag(np.exp(1j * phases)))


def validate_cyclic_operator_fixture(
    config: CyclicOperatorConfig | None = None,
) -> CyclicOperatorValidationResult:
    """Run the source-anchored EQ0115 cyclic-operator fixture."""
    cfg = config or CyclicOperatorConfig()
    spec = load_information_thermodynamics_validation_spec(
        "computational.cyclic_operator_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    operator = build_cyclic_operator(cfg)
    identity = np.eye(cfg.dimension, dtype=np.complex128)
    unitarity_error = float(np.linalg.norm(operator.conj().T @ operator - identity, ord=2))
    closure = np.linalg.matrix_power(operator, cfg.period)
    cycle_closure_residual = float(np.linalg.norm(closure - identity, ord=2))
    wrong_period = np.linalg.matrix_power(operator, cfg.period - 1)
    non_unitary = operator.copy()
    non_unitary[0, 0] *= 1.05
    controls = {
        "non_unitary_rejection_label": float(
            np.linalg.norm(non_unitary.conj().T @ non_unitary - identity, ord=2) > 0.0
        ),
        "wrong_period_residual": float(np.linalg.norm(wrong_period - identity, ord=2)),
    }
    metadata = {
        "paper0_spec_key": "computational.cyclic_operator_boundary",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "dimension": int(cfg.dimension),
        "period": int(cfg.period),
        "simulator_only_mechanism_evidence": True,
    }
    return CyclicOperatorValidationResult(
        spec_key="computational.cyclic_operator_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        unitarity_error=unitarity_error,
        cycle_closure_residual=cycle_closure_residual,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def abl_probabilities(
    pre_state: np.ndarray | None,
    post_state: np.ndarray | None,
    projectors: tuple[np.ndarray, ...] | None,
) -> tuple[float, ...]:
    """Return Aharonov-Bergmann-Lebowitz probabilities for finite projectors."""
    pre = _normalised_state("pre_state", pre_state)
    post = _normalised_state("post_state", post_state)
    if pre.shape != post.shape:
        raise ValueError("pre_state and post_state must have identical shape")
    if projectors is None:
        raise ValueError("projectors must not be None")
    raw_projectors = tuple(np.asarray(projector, dtype=np.complex128) for projector in projectors)
    weights = np.array(
        [abs(np.vdot(post, projector @ pre)) ** 2 for projector in raw_projectors],
        dtype=np.float64,
    )
    denominator = float(np.sum(weights))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("post_state denominator is zero for the supplied ABL boundary")
    validated_projectors = _validated_projectors(raw_projectors, dimension=pre.size)
    probabilities = weights / denominator
    if len(probabilities) != len(validated_projectors):
        raise AssertionError("projector validation changed projector count")
    return tuple(float(value) for value in probabilities)


def validate_tsvf_abl_fixture(
    config: ABLBoundaryConfig | None = None,
) -> ABLBoundaryValidationResult:
    """Run the source-anchored EQ0116 TSVF/ABL probability fixture."""
    cfg = config or ABLBoundaryConfig()
    spec = load_information_thermodynamics_validation_spec(
        "computational.tsvf_abl_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    probabilities = abl_probabilities(cfg.pre_state, cfg.post_state, cfg.projectors)
    normalisation_error = abs(sum(probabilities) - 1.0)
    born_probabilities = _born_probabilities(cast(np.ndarray, cfg.pre_state), cfg.projectors)
    marginal_abl_probabilities = _abl_probabilities_marginalised_over_post_basis(
        cast(np.ndarray, cfg.pre_state),
        cfg.projectors,
    )
    controls = {
        "born_rule_reduction_l1": float(
            np.sum(
                np.abs(
                    np.array(born_probabilities, dtype=np.float64)
                    - np.array(marginal_abl_probabilities, dtype=np.float64)
                )
            )
        ),
        "zero_denominator_rejection_label": _zero_denominator_rejection_label(),
        "projector_resolution_error": _projector_resolution_error(cfg.projectors),
    }
    metadata = {
        "paper0_spec_key": "computational.tsvf_abl_boundary",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "dimension": int(cast(np.ndarray, cfg.pre_state).size),
        "projector_count": int(len(cast(tuple[np.ndarray, ...], cfg.projectors))),
        "simulator_only_mechanism_evidence": True,
        "boundary_probability_rule": "Aharonov-Bergmann-Lebowitz",
    }
    return ABLBoundaryValidationResult(
        spec_key="computational.tsvf_abl_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        probabilities=probabilities,
        probability_normalisation_error=normalisation_error,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def entropy_budget_rates(config: InformationThermodynamicsConfig) -> EntropyBudgetRates:
    """Compute finite-rate entropy, information, and negentropy budget terms."""
    negentropy_rate = max(0.0, -config.thermodynamic_entropy_rate)
    information_entropy_rate = config.landauer_cost_per_nat * config.mutual_information_rate
    total_entropy_rate = config.thermodynamic_entropy_rate + information_entropy_rate
    expected_negentropy = config.proportionality * config.mutual_information_rate
    return EntropyBudgetRates(
        negentropy_rate=float(negentropy_rate),
        information_entropy_rate=float(information_entropy_rate),
        total_entropy_rate=float(total_entropy_rate),
        mutual_information_expected_negentropy=float(expected_negentropy),
        mutual_information_negentropy_error=float(abs(negentropy_rate - expected_negentropy)),
    )


def validate_information_thermodynamics_fixture(
    config: InformationThermodynamicsConfig | None = None,
) -> InformationThermodynamicsValidationResult:
    """Run the source-anchored EQ0117-EQ0118 entropy-budget fixture."""
    cfg = config or InformationThermodynamicsConfig()
    spec = load_information_thermodynamics_validation_spec(
        "computational.info_thermodynamics",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    rates = entropy_budget_rates(cfg)
    independent = entropy_budget_rates(
        InformationThermodynamicsConfig(
            thermodynamic_entropy_rate=0.0,
            mutual_information_rate=0.0,
            landauer_cost_per_nat=cfg.landauer_cost_per_nat,
            proportionality=cfg.proportionality,
            spec_bundle_path=cfg.spec_bundle_path,
        )
    )
    violated = entropy_budget_rates(
        InformationThermodynamicsConfig(
            thermodynamic_entropy_rate=cfg.thermodynamic_entropy_rate,
            mutual_information_rate=cfg.mutual_information_rate,
            landauer_cost_per_nat=0.0,
            proportionality=cfg.proportionality,
            spec_bundle_path=cfg.spec_bundle_path,
        )
    )
    controls = {
        "independent_channel_negentropy_abs": abs(independent.negentropy_rate),
        "landauer_violation_label": float(violated.total_entropy_rate < 0.0),
        "finite_gsl_margin_label": float(rates.total_entropy_rate >= 0.0),
    }
    metadata = {
        "paper0_spec_key": "computational.info_thermodynamics",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "thermodynamic_entropy_rate": float(cfg.thermodynamic_entropy_rate),
        "mutual_information_rate": float(cfg.mutual_information_rate),
        "landauer_cost_per_nat": float(cfg.landauer_cost_per_nat),
        "proportionality": float(cfg.proportionality),
        "simulator_only_mechanism_evidence": True,
    }
    return InformationThermodynamicsValidationResult(
        spec_key="computational.info_thermodynamics",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        negentropy_rate=rates.negentropy_rate,
        information_entropy_rate=rates.information_entropy_rate,
        total_entropy_rate=rates.total_entropy_rate,
        gsl_margin=rates.total_entropy_rate,
        mutual_information_negentropy_error=rates.mutual_information_negentropy_error,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _basis_projectors(dimension: int) -> tuple[np.ndarray, ...]:
    projectors: list[np.ndarray] = []
    for index in range(dimension):
        projector = np.zeros((dimension, dimension), dtype=np.complex128)
        projector[index, index] = 1.0
        projectors.append(projector)
    return tuple(projectors)


def _born_probabilities(
    state: np.ndarray,
    projectors: tuple[np.ndarray, ...] | None,
) -> tuple[float, ...]:
    if projectors is None:
        raise ValueError("projectors must not be None")
    probabilities = [float(np.real(np.vdot(state, projector @ state))) for projector in projectors]
    total = float(np.sum(probabilities))
    if total <= np.finfo(np.float64).eps:
        raise ValueError("Born-rule denominator is zero")
    return tuple(float(value / total) for value in probabilities)


def _abl_probabilities_marginalised_over_post_basis(
    pre_state: np.ndarray,
    projectors: tuple[np.ndarray, ...] | None,
) -> tuple[float, ...]:
    if projectors is None:
        raise ValueError("projectors must not be None")
    dimension = int(pre_state.size)
    basis = _basis_projectors(dimension)
    weights: list[float] = []
    for projector in projectors:
        weight = 0.0
        for post_projector in basis:
            post = cast(np.ndarray, post_projector @ np.ones(dimension, dtype=np.complex128))
            post_norm = float(np.linalg.norm(post))
            if post_norm <= np.finfo(np.float64).eps:
                continue
            post /= post_norm
            weight += abs(np.vdot(post, projector @ pre_state)) ** 2
        weights.append(float(weight))
    denominator = float(np.sum(weights))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("marginal ABL denominator is zero")
    return tuple(float(value / denominator) for value in weights)


def _normalised_state(name: str, values: np.ndarray | None) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    norm = float(np.linalg.norm(arr))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have non-zero norm")
    return cast(np.ndarray, arr / norm)


def _validated_projectors(
    projectors: tuple[np.ndarray, ...],
    *,
    dimension: int,
) -> tuple[np.ndarray, ...]:
    if len(projectors) < 1:
        raise ValueError("projectors must contain at least one projector")
    identity = np.eye(dimension, dtype=np.complex128)
    total = np.zeros((dimension, dimension), dtype=np.complex128)
    validated: list[np.ndarray] = []
    for index, projector in enumerate(projectors):
        arr = np.asarray(projector, dtype=np.complex128)
        if arr.shape != (dimension, dimension):
            raise ValueError(f"projector {index} must have shape {(dimension, dimension)}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"projector {index} must contain finite values")
        if not np.allclose(arr, arr.conj().T, atol=1.0e-12):
            raise ValueError(f"projector {index} must be Hermitian")
        if not np.allclose(arr @ arr, arr, atol=1.0e-12):
            raise ValueError(f"projector {index} must be idempotent")
        total += arr
        validated.append(cast(np.ndarray, arr))
    if not np.allclose(total, identity, atol=1.0e-12):
        raise ValueError("projectors must resolve identity")
    return tuple(validated)


def _projector_resolution_error(projectors: tuple[np.ndarray, ...] | None) -> float:
    if projectors is None:
        raise ValueError("projectors must not be None")
    dimension = int(projectors[0].shape[0])
    total = np.sum(np.stack(projectors, axis=0), axis=0)
    identity = np.eye(dimension, dtype=np.complex128)
    return float(np.linalg.norm(total - identity, ord=2))


def _zero_denominator_rejection_label() -> float:
    try:
        abl_probabilities(
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
            (np.diag([1.0, 0.0]).astype(np.complex128),),
        )
    except ValueError as exc:
        return float("post_state denominator" in str(exc))
    return 0.0


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "ABLBoundaryConfig",
    "ABLBoundaryValidationResult",
    "CyclicOperatorConfig",
    "CyclicOperatorValidationResult",
    "EntropyBudgetRates",
    "InformationThermodynamicsConfig",
    "InformationThermodynamicsValidationResult",
    "abl_probabilities",
    "build_cyclic_operator",
    "entropy_budget_rates",
    "validate_cyclic_operator_fixture",
    "validate_information_thermodynamics_fixture",
    "validate_tsvf_abl_fixture",
]
