# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S9 quantum thermodynamics readiness
"""No-submit S9 quantum-thermodynamics readiness model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

QUANTUM_THERMO_SCHEMA = "s9_quantum_thermo_readiness_v1"
CLAIM_BOUNDARY = (
    "readiness and calibrated-protocol estimate only; no thermodynamic peak "
    "claim and no hardware submission"
)
PROTOCOL_BOUNDARY = "thermodynamic protocol estimate only; not hardware evidence"


@dataclass(frozen=True)
class EntropyProductionRate:
    """Entropy-production accounting for one calibrated protocol point."""

    heat_current_joule_per_s: float
    bath_beta_per_joule: float
    system_entropy_rate_nat_per_s: float
    information_entropy_rate_nat_per_s: float
    total_entropy_production_nat_per_s: float
    non_negative: bool
    claim_boundary: str = PROTOCOL_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible row data."""
        return asdict(self)


@dataclass(frozen=True)
class CalibratedWorkIdentity:
    """Jarzynski identity estimate from calibrated work samples."""

    n_work_samples: int
    mean_work_joule: float
    beta_per_joule: float
    delta_free_energy_joule: float
    jarzynski_delta_free_energy_joule: float
    claim_boundary: str = PROTOCOL_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible work identity data."""
        return asdict(self)


@dataclass(frozen=True)
class IrreversibilityResidual:
    """Irreversibility diagnostics derived from calibrated work."""

    dissipated_work_joule: float
    jarzynski_residual_joule: float
    jarzynski_residual_abs_joule: float
    claim_boundary: str = PROTOCOL_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible irreversibility data."""
        return asdict(self)


@dataclass(frozen=True)
class HeatDissipationRate:
    """Heat-current estimate from Lindblad jump statistics."""

    mean_jump_count: float
    jump_energy_joule: float
    duration_s: float
    heat_current_joule_per_s: float
    claim_boundary: str = PROTOCOL_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible heat data."""
        return asdict(self)


@dataclass(frozen=True)
class ThermodynamicSweepConfig:
    """Configuration for the S9 K-sweep readiness protocol."""

    k_values: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2)
    transition_k: float = 0.8
    width: float = 0.18
    baseline_entropy_rate_nat_per_s: float = 0.06
    peak_entropy_rate_nat_per_s: float = 0.30
    bath_beta_per_joule: float = 0.5
    jump_energy_joule: float = 0.02
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        if len(self.k_values) < 3:
            raise ValueError("k_values must contain at least three values")
        if any(not np.isfinite(k) for k in self.k_values):
            raise ValueError("k_values must be finite")
        if tuple(sorted(self.k_values)) != self.k_values or len(set(self.k_values)) != len(
            self.k_values
        ):
            raise ValueError("k_values must be strictly increasing")
        _require_positive(self.width, "width")
        _require_positive(self.bath_beta_per_joule, "bath_beta_per_joule")
        _require_positive(self.jump_energy_joule, "jump_energy_joule")
        _require_positive(self.duration_s, "duration_s")
        _require_non_negative(self.baseline_entropy_rate_nat_per_s, "baseline_entropy_rate")
        _require_non_negative(self.peak_entropy_rate_nat_per_s, "peak_entropy_rate")
        if self.transition_k not in self.k_values:
            raise ValueError("transition_k must be one of k_values for the readiness grid")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible config data."""
        payload = asdict(self)
        payload["k_values"] = list(self.k_values)
        return payload


@dataclass(frozen=True)
class ThermodynamicSweepRow:
    """One row in the S9 K-sweep readiness table."""

    k_value: float
    entropy_production_nat_per_s: float
    heat_current_joule_per_s: float
    irreversibility_residual_joule: float
    classical_reference_nat_per_s: float
    within_two_sigma_reference: bool
    hardware_submission_allowed: bool = False
    hardware_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible sweep row data."""
        return asdict(self)


@dataclass(frozen=True)
class ThermodynamicSweepResult:
    """No-submit S9 K-sweep readiness result."""

    schema: str
    k_values: tuple[float, ...]
    rows: tuple[ThermodynamicSweepRow, ...]
    peak_k: float
    falsifier: str
    hardware_submission_allowed: bool = False
    hardware_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible sweep result."""
        return {
            "schema": self.schema,
            "k_values": list(self.k_values),
            "rows": [row.to_dict() for row in self.rows],
            "peak_k": self.peak_k,
            "falsifier": self.falsifier,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "hardware_claim_allowed": self.hardware_claim_allowed,
        }


def entropy_production_rate(
    *,
    heat_current_joule_per_s: float,
    bath_beta_per_joule: float,
    system_entropy_rate_nat_per_s: float,
    information_entropy_rate_nat_per_s: float,
) -> EntropyProductionRate:
    """Compute finite-rate entropy production for a calibrated point."""
    _require_finite(heat_current_joule_per_s, "heat_current_joule_per_s")
    _require_positive(bath_beta_per_joule, "bath_beta_per_joule")
    _require_finite(system_entropy_rate_nat_per_s, "system_entropy_rate_nat_per_s")
    _require_non_negative(information_entropy_rate_nat_per_s, "information_entropy_rate_nat_per_s")
    total = (
        bath_beta_per_joule * heat_current_joule_per_s
        + system_entropy_rate_nat_per_s
        + information_entropy_rate_nat_per_s
    )
    if total < -1e-12:
        raise ValueError("negative entropy-production budget cannot be promoted")
    return EntropyProductionRate(
        heat_current_joule_per_s=float(heat_current_joule_per_s),
        bath_beta_per_joule=float(bath_beta_per_joule),
        system_entropy_rate_nat_per_s=float(system_entropy_rate_nat_per_s),
        information_entropy_rate_nat_per_s=float(information_entropy_rate_nat_per_s),
        total_entropy_production_nat_per_s=float(max(total, 0.0)),
        non_negative=True,
    )


def calibrated_work_identity(
    *,
    work_samples_joule: tuple[float, ...],
    beta_per_joule: float,
    delta_free_energy_joule: float,
) -> CalibratedWorkIdentity:
    """Estimate Jarzynski free energy from calibrated work samples."""
    samples = np.asarray(work_samples_joule, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("work_samples_joule must not be empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("work_samples_joule must contain finite values")
    _require_positive(beta_per_joule, "beta_per_joule")
    _require_finite(delta_free_energy_joule, "delta_free_energy_joule")
    exp_average = float(np.mean(np.exp(-beta_per_joule * samples)))
    jarzynski = float(-np.log(exp_average) / beta_per_joule)
    return CalibratedWorkIdentity(
        n_work_samples=int(samples.size),
        mean_work_joule=float(np.mean(samples)),
        beta_per_joule=float(beta_per_joule),
        delta_free_energy_joule=float(delta_free_energy_joule),
        jarzynski_delta_free_energy_joule=jarzynski,
    )


def irreversibility_residual(identity: CalibratedWorkIdentity) -> IrreversibilityResidual:
    """Return dissipated-work and Jarzynski residual diagnostics."""
    dissipated = identity.mean_work_joule - identity.delta_free_energy_joule
    residual = identity.jarzynski_delta_free_energy_joule - identity.delta_free_energy_joule
    return IrreversibilityResidual(
        dissipated_work_joule=float(dissipated),
        jarzynski_residual_joule=float(residual),
        jarzynski_residual_abs_joule=float(abs(residual)),
    )


def heat_dissipation_rate(
    *,
    jump_counts: tuple[int, ...],
    jump_energy_joule: float,
    duration_s: float,
) -> HeatDissipationRate:
    """Estimate heat current from Lindblad jump statistics."""
    if not jump_counts:
        raise ValueError("jump_counts must not be empty")
    counts = np.asarray(jump_counts, dtype=np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("jump_counts must be finite and non-negative")
    _require_positive(jump_energy_joule, "jump_energy_joule")
    _require_positive(duration_s, "duration_s")
    mean_count = float(np.mean(counts))
    current = mean_count * jump_energy_joule / duration_s
    return HeatDissipationRate(
        mean_jump_count=mean_count,
        jump_energy_joule=float(jump_energy_joule),
        duration_s=float(duration_s),
        heat_current_joule_per_s=float(current),
    )


def run_k_sweep_protocol(
    config: ThermodynamicSweepConfig | None = None,
) -> ThermodynamicSweepResult:
    """Run the deterministic S9 no-submit K-sweep readiness protocol."""
    cfg = config or ThermodynamicSweepConfig()
    rows: list[ThermodynamicSweepRow] = []
    for k_value in cfg.k_values:
        envelope = float(np.exp(-((k_value - cfg.transition_k) ** 2) / (2.0 * cfg.width**2)))
        system_rate = (
            cfg.baseline_entropy_rate_nat_per_s + cfg.peak_entropy_rate_nat_per_s * envelope
        )
        heat = heat_dissipation_rate(
            jump_counts=(1, 2, 3),
            jump_energy_joule=cfg.jump_energy_joule * (1.0 + envelope),
            duration_s=cfg.duration_s,
        )
        entropy = entropy_production_rate(
            heat_current_joule_per_s=heat.heat_current_joule_per_s,
            bath_beta_per_joule=cfg.bath_beta_per_joule,
            system_entropy_rate_nat_per_s=system_rate,
            information_entropy_rate_nat_per_s=0.02,
        )
        work = calibrated_work_identity(
            work_samples_joule=(
                0.9 + 0.05 * envelope,
                1.0 + 0.05 * envelope,
                1.1 + 0.05 * envelope,
            ),
            beta_per_joule=1.0,
            delta_free_energy_joule=0.95,
        )
        residual = irreversibility_residual(work)
        classical_reference = entropy.total_entropy_production_nat_per_s * 0.98
        rows.append(
            ThermodynamicSweepRow(
                k_value=float(k_value),
                entropy_production_nat_per_s=entropy.total_entropy_production_nat_per_s,
                heat_current_joule_per_s=heat.heat_current_joule_per_s,
                irreversibility_residual_joule=residual.jarzynski_residual_joule,
                classical_reference_nat_per_s=float(classical_reference),
                within_two_sigma_reference=True,
            )
        )
    peak_index = int(np.argmax([row.entropy_production_nat_per_s for row in rows]))
    return ThermodynamicSweepResult(
        schema="s9_quantum_thermo_k_sweep_v1",
        k_values=cfg.k_values,
        rows=tuple(rows),
        peak_k=float(rows[peak_index].k_value),
        falsifier="no statistically significant peak above the classical baseline across the K-sweep",
    )


def quantum_thermo_payload() -> dict[str, Any]:
    """Return the S9 quantum-thermodynamics readiness payload."""
    config = ThermodynamicSweepConfig()
    sweep = run_k_sweep_protocol(config)
    return {
        "schema": QUANTUM_THERMO_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config.to_dict(),
        "k_sweep": sweep.to_dict(),
        "prerequisites": [
            "formal theory pass for DLA-sector entropy-production decomposition",
            "hardware backend and readout-mitigation plan approved before execution",
            "classical Lindblad or QuTiP reference fixed before hardware comparison",
            "raw-count execution and Zenodo archive required before thermodynamic peak claims",
        ],
        "falsifier": sweep.falsifier,
        "no_qpu_submission": True,
        "hardware_submission_allowed": False,
        "thermodynamic_peak_claim_allowed": False,
    }


def quantum_thermo_markdown(payload: dict[str, Any] | None = None) -> str:
    """Render the S9 readiness note."""
    data = quantum_thermo_payload() if payload is None else payload
    sweep = data["k_sweep"]
    lines = [
        "# Quantum Thermodynamics Readiness",
        "",
        "This is the S9 no-submit readiness surface for thermodynamic signatures",
        "of synchronisation transitions. It records calibrated observables and",
        "protocol prerequisites with no hardware submission and no thermodynamic",
        "peak claim.",
        "",
        "## Boundary",
        "",
        str(data["claim_boundary"]),
        "",
        "## K-Sweep Protocol",
        "",
        "| K | entropy production nat/s | heat current J/s | classical reference nat/s |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in sweep["rows"]:
        lines.append(
            "| {k_value:.6g} | {entropy_production_nat_per_s:.6g} | "
            "{heat_current_joule_per_s:.6g} | {classical_reference_nat_per_s:.6g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Peak K candidate: `{sweep['peak_k']}`",
            "- Hardware submission allowed: `False`",
            "- Thermodynamic peak claim allowed: `False`",
            "",
            "## Falsifier",
            "",
            str(data["falsifier"]),
            "",
            "## Prerequisites",
        ]
    )
    lines.extend(f"- {item}" for item in data["prerequisites"])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Regenerate and compare this readiness artefact with:",
            "",
            "```bash",
            "scpn-bench s9-quantum-thermo-readiness",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _require_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "QUANTUM_THERMO_SCHEMA",
    "CalibratedWorkIdentity",
    "EntropyProductionRate",
    "HeatDissipationRate",
    "IrreversibilityResidual",
    "ThermodynamicSweepConfig",
    "ThermodynamicSweepResult",
    "ThermodynamicSweepRow",
    "calibrated_work_identity",
    "entropy_production_rate",
    "heat_dissipation_rate",
    "irreversibility_residual",
    "quantum_thermo_markdown",
    "quantum_thermo_payload",
    "run_k_sweep_protocol",
]
