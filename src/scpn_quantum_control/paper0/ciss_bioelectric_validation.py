# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 CISS-bioelectric fixtures
"""Simulator-only Layer 3 CISS-bioelectric feedback fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_ciss_bioelectric_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded CISS-bioelectric feedback simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06560", "P0R06581")


@dataclass(frozen=True, slots=True)
class CISSBioelectricConfig:
    """Finite simulator settings for CISS-bioelectric fixtures."""

    effective_field_scale_t_per_lambda: float = 80.0
    spin_orbit_lambda: float = 0.5
    nonlinear_yield_scale_t: float = 25.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive(
            "effective_field_scale_t_per_lambda",
            self.effective_field_scale_t_per_lambda,
        )
        _require_finite("spin_orbit_lambda", self.spin_orbit_lambda)
        _require_positive("nonlinear_yield_scale_t", self.nonlinear_yield_scale_t)


@dataclass(frozen=True, slots=True)
class CISSBioelectricFixtureResult:
    """Combined Layer 3 CISS-bioelectric fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    spin_filter_hamiltonian: float
    effective_field_t: float
    radical_pair_hamiltonian: float
    radical_pair_yield_modulation: float
    cascade_drive: float
    feedback_derivative: float
    feedback_hamiltonian_shift: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def ciss_spin_filter_hamiltonian(
    *,
    epsilon_0: float,
    delta: float,
    sigma_z: float,
    spin_orbit_lambda: float,
    length_scale: float,
    sigma_dot_l: float,
    g_factor: float,
    s_dot_sigma: float,
) -> float:
    """Return the source CISS spin-filter Hamiltonian sum."""
    _require_finite_values(
        epsilon_0,
        delta,
        sigma_z,
        spin_orbit_lambda,
        sigma_dot_l,
        g_factor,
        s_dot_sigma,
        message="CISS Hamiltonian inputs must be finite",
    )
    _require_positive("length_scale", length_scale)
    return float(
        epsilon_0
        + 0.5 * delta * sigma_z
        + (spin_orbit_lambda / (length_scale * length_scale)) * sigma_dot_l
        + g_factor * s_dot_sigma
    )


def ciss_effective_field_t(*, spin_orbit_lambda: float, scale_t_per_lambda: float) -> float:
    """Map source spin-orbit coupling to an effective field scale."""
    _require_finite("spin_orbit_lambda", spin_orbit_lambda)
    _require_positive("scale_t_per_lambda", scale_t_per_lambda)
    return abs(spin_orbit_lambda) * scale_t_per_lambda


def radical_pair_hamiltonian(
    *,
    zeeman_terms: NDArray[np.float64],
    hyperfine_terms: NDArray[np.float64],
    exchange_j: float,
    s1_dot_s2: float,
) -> float:
    """Return finite radical-pair Zeeman, hyperfine, and exchange terms."""
    zeeman = _finite_vector("zeeman_terms", zeeman_terms)
    hyperfine = _finite_matrix("hyperfine_terms", hyperfine_terms)
    _require_finite_values(
        exchange_j,
        s1_dot_s2,
        message="radical-pair scalar inputs must be finite",
    )
    return float(zeeman.sum() + hyperfine.sum() + exchange_j * (0.5 + 2.0 * s1_dot_s2))


def membrane_potential_derivative(*, ionic_current: float, pump_current: float) -> float:
    """Return the source membrane derivative -I_ion + I_pump."""
    _require_finite_values(ionic_current, pump_current, message="membrane currents must be finite")
    return -ionic_current + pump_current


def bioelectric_cascade_drive(
    *,
    target_potential_gradient: float,
    cav_activation_gain: float,
    camkii_gain: float,
    chromatin_gain: float,
) -> float:
    """Return signed drive from -grad V_target through the source cascade."""
    _require_finite_values(
        target_potential_gradient,
        cav_activation_gain,
        camkii_gain,
        chromatin_gain,
        message="bioelectric cascade inputs must be finite",
    )
    return -target_potential_gradient * cav_activation_gain * camkii_gain * chromatin_gain


def local_field_feedback_shift(
    *,
    membrane_potential: float,
    local_field_gain: float,
    spin_coupling_sum: float,
) -> float:
    """Return B_local(V_mem) dot (g_1 S_1 + g_2 S_2) as a scalar fixture."""
    _require_finite_values(
        membrane_potential,
        local_field_gain,
        spin_coupling_sum,
        message="local-field feedback inputs must be finite",
    )
    return membrane_potential * local_field_gain * spin_coupling_sum


def nonlinear_radical_pair_yield_modulation(
    *,
    effective_field_t: float,
    nonlinear_yield_scale_t: float,
) -> float:
    """Return bounded nonlinear modulation label for applied effective field."""
    _require_finite("effective_field_t", effective_field_t)
    _require_positive("nonlinear_yield_scale_t", nonlinear_yield_scale_t)
    return abs(tanh(effective_field_t / nonlinear_yield_scale_t))


def validate_ciss_bioelectric_fixture(
    config: CISSBioelectricConfig | None = None,
) -> CISSBioelectricFixtureResult:
    """Run the combined Layer 3 CISS-bioelectric fixture."""
    cfg = config or CISSBioelectricConfig()
    keys = (
        "ciss_bioelectric.layer3_framing",
        "ciss_bioelectric.ciss_spin_filter",
        "ciss_bioelectric.radical_pair_modulation",
        "ciss_bioelectric.bioelectric_cascade_feedback",
        "ciss_bioelectric.observable_predictions",
    )
    specs = tuple(
        load_ciss_bioelectric_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    spin_filter = ciss_spin_filter_hamiltonian(
        epsilon_0=0.2,
        delta=0.4,
        sigma_z=-1.0,
        spin_orbit_lambda=cfg.spin_orbit_lambda,
        length_scale=2.0,
        sigma_dot_l=3.0,
        g_factor=2.1,
        s_dot_sigma=0.25,
    )
    effective_field = ciss_effective_field_t(
        spin_orbit_lambda=cfg.spin_orbit_lambda,
        scale_t_per_lambda=cfg.effective_field_scale_t_per_lambda,
    )
    rp_hamiltonian = radical_pair_hamiltonian(
        zeeman_terms=np.array([0.2, 0.3], dtype=np.float64),
        hyperfine_terms=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64),
        exchange_j=0.4,
        s1_dot_s2=-0.25,
    )
    cascade = bioelectric_cascade_drive(
        target_potential_gradient=1.5,
        cav_activation_gain=0.8,
        camkii_gain=0.6,
        chromatin_gain=0.5,
    )
    feedback_derivative = membrane_potential_derivative(ionic_current=0.7, pump_current=0.2)
    controls = {
        "non_positive_length_rejection_label": _non_positive_length_rejection_label(),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "unsupported_morphogenesis_evidence_rejection_label": 1.0,
    }
    return CISSBioelectricFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        spin_filter_hamiltonian=spin_filter,
        effective_field_t=effective_field,
        radical_pair_hamiltonian=rp_hamiltonian,
        radical_pair_yield_modulation=nonlinear_radical_pair_yield_modulation(
            effective_field_t=effective_field,
            nonlinear_yield_scale_t=cfg.nonlinear_yield_scale_t,
        ),
        cascade_drive=cascade,
        feedback_derivative=feedback_derivative,
        feedback_hamiltonian_shift=local_field_feedback_shift(
            membrane_potential=-0.07,
            local_field_gain=0.4,
            spin_coupling_sum=1.3,
        ),
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "effective_field_scale_t_per_lambda": cfg.effective_field_scale_t_per_lambda,
                "spin_orbit_lambda": cfg.spin_orbit_lambda,
                "nonlinear_yield_scale_t": cfg.nonlinear_yield_scale_t,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "effective_field_source_range_t": (10.0, 100.0),
            }
        ),
    )


def _non_positive_length_rejection_label() -> float:
    try:
        ciss_spin_filter_hamiltonian(
            epsilon_0=0.0,
            delta=0.1,
            sigma_z=1.0,
            spin_orbit_lambda=0.2,
            length_scale=0.0,
            sigma_dot_l=1.0,
            g_factor=2.0,
            s_dot_sigma=0.1,
        )
    except ValueError as exc:
        return float("length_scale must be finite and positive" in str(exc))
    return 0.0


def _shape_mismatch_rejection_label() -> float:
    try:
        radical_pair_hamiltonian(
            zeeman_terms=np.ones((1, 2), dtype=np.float64),
            hyperfine_terms=np.ones((2, 2), dtype=np.float64),
            exchange_j=0.1,
            s1_dot_s2=0.0,
        )
    except ValueError as exc:
        return float("zeeman_terms must be one-dimensional" in str(exc))
    return 0.0


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _finite_matrix(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_values(*values: float, message: str) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError(message)


__all__ = [
    "CISSBioelectricConfig",
    "CISSBioelectricFixtureResult",
    "CLAIM_BOUNDARY",
    "bioelectric_cascade_drive",
    "ciss_effective_field_t",
    "ciss_spin_filter_hamiltonian",
    "local_field_feedback_shift",
    "membrane_potential_derivative",
    "nonlinear_radical_pair_yield_modulation",
    "radical_pair_hamiltonian",
    "validate_ciss_bioelectric_fixture",
]
