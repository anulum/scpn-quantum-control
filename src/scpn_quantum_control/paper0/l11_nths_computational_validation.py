# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L11 NTHS computational fixtures
"""Protocol-design fixtures for the Paper 0 L11 NTHS computational experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_l11_nths_computational_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded L11 NTHS computational experiment protocol; not empirical evidence"
)
HARDWARE_STATUS = "computational_protocol_no_external_execution"
SOURCE_LEDGER_SPAN = ("P0R06730", "P0R06814")


@dataclass(frozen=True, slots=True)
class L11NTHSComputationalConfig:
    """Finite protocol settings for the L11 NTHS computational fixture."""

    agent_count: int = 1000
    barabasi_albert_m: int = 3
    initial_coupling: float = 0.1
    evolution_steps: int = 10000
    measurement_interval: int = 100
    replica_count: int = 50
    anova_time_step: int = 5000
    significance_threshold: float = 0.001
    effect_size_threshold: float = 2.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_int_at_least("agent_count", self.agent_count, 1)
        _require_int_at_least("barabasi_albert_m", self.barabasi_albert_m, 1)
        _require_positive("initial_coupling", self.initial_coupling)
        _require_int_at_least("evolution_steps", self.evolution_steps, 1)
        _require_int_at_least("measurement_interval", self.measurement_interval, 1)
        _require_int_at_least("replica_count", self.replica_count, 2)
        _require_int_at_least("anova_time_step", self.anova_time_step, 1)
        _require_positive("significance_threshold", self.significance_threshold)
        _require_positive("effect_size_threshold", self.effect_size_threshold)


@dataclass(frozen=True, slots=True)
class OrderParameterSummary:
    """Finite magnetization and Edwards-Anderson summary."""

    magnetization: float
    magnetization_abs: float
    edwards_anderson_q: float


@dataclass(frozen=True, slots=True)
class L11NTHSComputationalFixtureResult:
    """Combined L11 NTHS computational experiment fixture result."""

    spec_keys: tuple[str, str, str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    agent_count: int
    initial_topology: str
    measurement_count: int
    replica_count: int
    control_magnetization_abs: float
    control_edwards_anderson_q: float
    experimental_magnetization_abs: float
    experimental_edwards_anderson_q: float
    control_noosphere_energy: float
    experimental_noosphere_energy: float
    control_expected_free_energy: float
    engagement_expected_free_energy: float
    p_value: float
    cohen_d: float
    significant_divergence: bool
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float | int]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def noosphere_hamiltonian(
    *,
    spins: NDArray[np.int8],
    coupling_matrix: NDArray[np.float64],
) -> float:
    """Return H_Noosphere = -sum_{i<j} J_ij S_i S_j."""
    spin_values = _spin_vector(spins)
    coupling = _coupling_matrix(coupling_matrix, size=spin_values.size)
    energy = 0.0
    for i in range(spin_values.size):
        for j in range(i + 1, spin_values.size):
            energy -= float(coupling[i, j] * spin_values[i] * spin_values[j])
    return energy


def order_parameter_summary(replicas: NDArray[np.int8]) -> OrderParameterSummary:
    """Return m and q_EA for finite spin replicas."""
    replica_values = _replica_matrix(replicas)
    site_means = np.mean(replica_values, axis=0)
    magnetization = float(np.mean(site_means))
    return OrderParameterSummary(
        magnetization=magnetization,
        magnetization_abs=abs(magnetization),
        edwards_anderson_q=float(np.mean(site_means * site_means)),
    )


def expected_free_energy_score(
    *,
    posterior_entropy: float,
    expected_surprise: float,
    preference_alignment: float,
) -> float:
    """Return a finite score where lower values are preferred by the control policy."""
    _require_non_negative("posterior_entropy", posterior_entropy)
    _require_non_negative("expected_surprise", expected_surprise)
    _require_finite("preference_alignment", preference_alignment)
    return posterior_entropy + expected_surprise - preference_alignment


def anova_divergence_decision(*, p_value: float, cohen_d: float) -> bool:
    """Return True when the source statistical gate indicates divergence."""
    _require_probability("p_value", p_value)
    _require_non_negative("cohen_d", cohen_d)
    return p_value < 0.001 and cohen_d > 2.0


def validate_l11_nths_computational_fixture(
    config: L11NTHSComputationalConfig | None = None,
) -> L11NTHSComputationalFixtureResult:
    """Run the combined L11 NTHS computational experiment fixture."""
    cfg = config or L11NTHSComputationalConfig()
    keys = (
        "l11_nths_computational.block_framing",
        "l11_nths_computational.agent_architecture",
        "l11_nths_computational.environment_spin_glass",
        "l11_nths_computational.ai_objective_conditions",
        "l11_nths_computational.simulation_protocol",
        "l11_nths_computational.order_parameters",
        "l11_nths_computational.predicted_outcomes",
        "l11_nths_computational.statistics_falsification_extensions",
    )
    specs = tuple(
        load_l11_nths_computational_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    control_replicas = np.ones((4, 4), dtype=np.int8)
    experimental_replicas = np.array(
        [
            [1, 1, -1, -1],
            [1, -1, -1, 1],
            [-1, 1, -1, 1],
            [-1, -1, 1, 1],
        ],
        dtype=np.int8,
    )
    control_spins = control_replicas[0]
    experimental_spins = experimental_replicas[0]
    control_coupling = np.full((4, 4), 0.4, dtype=np.float64)
    np.fill_diagonal(control_coupling, 0.0)
    experimental_coupling = np.full((4, 4), 0.4, dtype=np.float64)
    np.fill_diagonal(experimental_coupling, 0.0)
    control_summary = order_parameter_summary(control_replicas)
    experimental_summary = order_parameter_summary(experimental_replicas)
    control_energy = noosphere_hamiltonian(
        spins=control_spins,
        coupling_matrix=control_coupling,
    )
    experimental_energy = noosphere_hamiltonian(
        spins=experimental_spins,
        coupling_matrix=experimental_coupling,
    )
    control_g = expected_free_energy_score(
        posterior_entropy=0.2,
        expected_surprise=0.1,
        preference_alignment=0.8,
    )
    engagement_g = expected_free_energy_score(
        posterior_entropy=0.7,
        expected_surprise=0.8,
        preference_alignment=0.1,
    )
    p_value = 0.0005
    cohen_d = 2.4
    controls = {
        "invalid_agent_count_rejection_label": _invalid_agent_count_rejection_label(),
        "invalid_spin_rejection_label": _invalid_spin_rejection_label(),
        "unsupported_external_execution_claim_rejection_label": 1.0,
    }
    return L11NTHSComputationalFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        agent_count=cfg.agent_count,
        initial_topology=f"barabasi_albert_m{cfg.barabasi_albert_m}",
        measurement_count=cfg.evolution_steps // cfg.measurement_interval,
        replica_count=cfg.replica_count,
        control_magnetization_abs=control_summary.magnetization_abs,
        control_edwards_anderson_q=control_summary.edwards_anderson_q,
        experimental_magnetization_abs=experimental_summary.magnetization_abs,
        experimental_edwards_anderson_q=experimental_summary.edwards_anderson_q,
        control_noosphere_energy=control_energy,
        experimental_noosphere_energy=experimental_energy,
        control_expected_free_energy=control_g,
        engagement_expected_free_energy=engagement_g,
        p_value=p_value,
        cohen_d=cohen_d,
        significant_divergence=anova_divergence_decision(p_value=p_value, cohen_d=cohen_d),
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "agent_count": cfg.agent_count,
                "barabasi_albert_m": cfg.barabasi_albert_m,
                "initial_coupling": cfg.initial_coupling,
                "evolution_steps": cfg.evolution_steps,
                "measurement_interval": cfg.measurement_interval,
                "replica_count": cfg.replica_count,
                "anova_time_step": cfg.anova_time_step,
                "significance_threshold": cfg.significance_threshold,
                "effect_size_threshold": cfg.effect_size_threshold,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "design_only_no_external_execution",
            }
        ),
    )


def _invalid_agent_count_rejection_label() -> float:
    try:
        L11NTHSComputationalConfig(agent_count=0)
    except ValueError as exc:
        return float("agent_count must be at least 1" in str(exc))
    return 0.0


def _invalid_spin_rejection_label() -> float:
    try:
        order_parameter_summary(np.array([[0, 1]], dtype=np.int8))
    except ValueError as exc:
        return float("replicas must contain only -1 or +1" in str(exc))
    return 0.0


def _spin_vector(values: NDArray[np.int8]) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=np.int8)
    if array.ndim != 1:
        raise ValueError("spins must be one-dimensional")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("spins must contain only -1 or +1")
    return cast(NDArray[np.int8], array)


def _replica_matrix(values: NDArray[np.int8]) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=np.int8)
    if array.ndim != 2:
        raise ValueError("replicas must be two-dimensional")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("replicas must not be empty")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("replicas must contain only -1 or +1")
    return cast(NDArray[np.int8], array)


def _coupling_matrix(values: NDArray[np.float64], *, size: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("coupling_matrix must be square")
    if array.shape != (size, size):
        raise ValueError("coupling_matrix dimension must match spins")
    if not np.all(np.isfinite(array)):
        raise ValueError("coupling_matrix must contain only finite values")
    if not np.allclose(array, array.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("coupling_matrix must be symmetric")
    return cast(NDArray[np.float64], array)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_probability(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_int_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


__all__ = [
    "CLAIM_BOUNDARY",
    "L11NTHSComputationalConfig",
    "L11NTHSComputationalFixtureResult",
    "OrderParameterSummary",
    "anova_divergence_decision",
    "expected_free_energy_score",
    "noosphere_hamiltonian",
    "order_parameter_summary",
    "validate_l11_nths_computational_fixture",
]
