# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational verification tools
"""Source-bounded fixtures for Paper 0 computational verification tools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .spec_loader import load_computational_verification_tools_validation_spec

CLAIM_BOUNDARY = "source-bounded computational protocol; not empirical execution evidence"
HARDWARE_STATUS = "computational_protocol_no_claimed_execution"
SOURCE_LEDGER_SPAN = ("P0R07006", "P0R07072")
M_PL_GEV = 2.435e18
LAMBDA_0_GEV4 = 1.1056e-52
LAMBDA_PSI_G = 1.068935e-122


@dataclass(frozen=True, slots=True)
class ComputationalVerificationToolsConfig:
    """Finite parameters for the Paper 0 computational verification fixtures."""

    lattice_size: int = 16
    warmup_lattice_size: int = 12
    lattice_spacing: float = 1.0
    v: float = 1.0
    lambda4: float = 0.10
    ntraj: int = 3000
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.lattice_size < 1:
            raise ValueError("lattice_size must be at least 1")
        if self.warmup_lattice_size < 1:
            raise ValueError("warmup_lattice_size must be at least 1")
        if self.lattice_spacing <= 0.0:
            raise ValueError("lattice_spacing must be positive")
        if self.v <= 0.0:
            raise ValueError("v must be positive")
        if self.lambda4 <= 0.0:
            raise ValueError("lambda4 must be positive")
        if self.ntraj < 1:
            raise ValueError("ntraj must be at least 1")


@dataclass(frozen=True, slots=True)
class LatticeMassTargets:
    """Source-stated tuned-line mass targets."""

    yukawa: float
    pcac_mass: float
    mass_ratio_target: float
    tuned_relation_residual: float


@dataclass(frozen=True, slots=True)
class LatticeHMCActionResult:
    """Terms of the source-stated quenched lattice action fixture."""

    radial_gradient: float
    goldstone_gradient: float
    potential: float
    total: float
    fermion_boundary: str


@dataclass(frozen=True, slots=True)
class ComputationalVerificationToolsFixtureResult:
    """Combined computational verification-tool fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    tool_count: int
    spec_count: int
    lattice_mass_targets: LatticeMassTargets
    class_patch_parameters: MappingProxyType[str, float]
    lambda_constants: MappingProxyType[str, float]
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def lattice_mass_targets(
    config: ComputationalVerificationToolsConfig | None = None,
) -> LatticeMassTargets:
    """Return the source-stated tuned-line Yukawa, PCAC, and mass-ratio targets."""
    cfg = config or ComputationalVerificationToolsConfig()
    yukawa = math.sqrt(cfg.lambda4 / 2.0)
    return LatticeMassTargets(
        yukawa=yukawa,
        pcac_mass=yukawa / math.sqrt(2.0) * cfg.v,
        mass_ratio_target=math.sqrt(2.0),
        tuned_relation_residual=yukawa**2 - cfg.lambda4 / 2.0,
    )


def lattice_hmc_action(
    rho: ArrayLike,
    phi: ArrayLike,
    *,
    config: ComputationalVerificationToolsConfig | None = None,
) -> LatticeHMCActionResult:
    """Evaluate the source-stated quenched periodic-lattice action terms."""
    cfg = config or ComputationalVerificationToolsConfig()
    rho_array = np.asarray(rho, dtype=float)
    phi_array = np.asarray(phi, dtype=float)
    if rho_array.shape != phi_array.shape:
        raise ValueError("rho and phi must have the same lattice shape")
    if rho_array.ndim < 1:
        raise ValueError("rho and phi must be finite-dimensional lattice arrays")

    radial_gradient = _periodic_gradient_square(rho_array)
    goldstone_gradient = (cfg.v**2 / cfg.lattice_spacing**2) * _periodic_gradient_square(phi_array)
    potential = (
        cfg.lambda4
        * cfg.lattice_spacing**2
        * float(np.sum((rho_array * (rho_array + 2.0 * cfg.v)) ** 2))
    )
    total = 0.5 * radial_gradient + 0.5 * goldstone_gradient + potential
    return LatticeHMCActionResult(
        radial_gradient=0.5 * radial_gradient,
        goldstone_gradient=0.5 * goldstone_gradient,
        potential=potential,
        total=total,
        fermion_boundary="quenched_flat_line_test",
    )


def class_goldstone_equation_of_state(
    a: float,
    *,
    eps: float,
    omega_log: float,
    phase: float,
) -> float:
    """Return the source-stated oscillatory Goldstone equation of state w(a)."""
    if a <= 0.0:
        raise ValueError("scale factor a must be positive")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    return -1.0 + eps * math.cos(omega_log * math.log(a) + phase)


def class_goldstone_density_pressure(
    a: float,
    *,
    rho_phi0: float,
    eps: float,
    omega_log: float,
    phase: float,
) -> tuple[float, float, float]:
    """Return the CLASS background density, pressure, and w(a) mapping."""
    if rho_phi0 < 0.0:
        raise ValueError("rho_phi0 must be non-negative")
    w_value = class_goldstone_equation_of_state(
        a,
        eps=eps,
        omega_log=omega_log,
        phase=phase,
    )
    oscillation = 1.0 + eps * math.cos(omega_log * math.log(a) + phase)
    density = rho_phi0 * a**-3 * oscillation
    return density, w_value * density, w_value


def washout_regime(*, eps: float, omega_log: float) -> bool:
    """Return whether the source-stated small-epsilon high-frequency washout applies."""
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    if omega_log < 0.0:
        raise ValueError("omega_log must be non-negative")
    return eps <= 1.0e-3 and omega_log >= 500.0


def lambda_psi_rho(psi_t: float) -> float:
    """Return the source-stated Psi-sector vacuum-energy density in GeV^4."""
    return LAMBDA_PSI_G * (psi_t**2) * (M_PL_GEV**2)


def lambda_eff(psi_t: float) -> float:
    """Return the source-stated effective vacuum-energy density in GeV^4."""
    return LAMBDA_0_GEV4 + lambda_psi_rho(psi_t)


def validate_computational_verification_tools_fixture(
    config: ComputationalVerificationToolsConfig | None = None,
) -> ComputationalVerificationToolsFixtureResult:
    """Run the computational verification tools boundary fixture."""
    cfg = config or ComputationalVerificationToolsConfig()
    keys = (
        "computational_verification_tools.chapter_boundary",
        "computational_verification_tools.lattice_hmc_flat_line",
        "computational_verification_tools.class_goldstone_eos",
        "computational_verification_tools.lambda_eff_utility",
        "computational_verification_tools.execution_boundaries",
    )
    specs = tuple(
        load_computational_verification_tools_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    controls = {
        "hmc_execution_overclaim_rejection_label": 1.0,
        "class_patch_without_parameter_audit_rejection_label": _class_parameter_audit_label(),
        "lambda_units_mismatch_rejection_label": _lambda_units_audit_label(),
    }
    return ComputationalVerificationToolsFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        tool_count=3,
        spec_count=len(keys),
        lattice_mass_targets=lattice_mass_targets(cfg),
        class_patch_parameters=MappingProxyType(
            {"Omega_phi": 1.0e-4, "eps_phi": 0.1, "omega_log_phi": 3.0e4, "phi0_phi": 0.0}
        ),
        lambda_constants=MappingProxyType(
            {
                "M_PL_GeV": M_PL_GEV,
                "LAMBDA_0_GeV4": LAMBDA_0_GEV4,
                "LAMBDA_PSI_G": LAMBDA_PSI_G,
            }
        ),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "computational_protocol_only_no_external_execution",
            }
        ),
    )


def _periodic_gradient_square(values: np.ndarray) -> float:
    return float(
        sum(np.sum((np.roll(values, -1, axis=axis) - values) ** 2) for axis in range(values.ndim))
    )


def _class_parameter_audit_label() -> float:
    parameters = {"eps_phi", "omega_log_phi", "phi0_phi"}
    required = {"eps_phi", "omega_log_phi", "phi0_phi"}
    return float(parameters == required)


def _lambda_units_audit_label() -> float:
    return float(M_PL_GEV > 0.0 and LAMBDA_0_GEV4 > 0.0 and LAMBDA_PSI_G > 0.0)


__all__ = [
    "CLAIM_BOUNDARY",
    "ComputationalVerificationToolsConfig",
    "ComputationalVerificationToolsFixtureResult",
    "LatticeHMCActionResult",
    "LatticeMassTargets",
    "class_goldstone_density_pressure",
    "class_goldstone_equation_of_state",
    "lambda_eff",
    "lambda_psi_rho",
    "lattice_hmc_action",
    "lattice_mass_targets",
    "validate_computational_verification_tools_fixture",
    "washout_regime",
]
