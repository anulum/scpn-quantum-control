# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control evidence
"""Deterministic chimera-control evidence construction, rendering, and byte custody."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    TopologyConstraintLedger,
)

from .objectives import build_chimera_control_objective, propose_phase_control_step
from .observables import measure_multiscale_order_parameters
from .schema import (
    CHIMERA_CONTROL_CLAIM_BOUNDARY,
    ChimeraControlSpecification,
    HierarchyTarget,
    SyntheticRegime,
)
from .synthetic import (
    SyntheticChimeraConfig,
    SyntheticChimeraRun,
    generate_two_population_chimera,
)
from .topology import project_chimera_coupling

CHIMERA_CONTROL_EVIDENCE_SCHEMA = "chimera_multiscale_control_evidence.v2"
CHIMERA_CONTROL_EVIDENCE_DATE = "2026-07-29"
SupportStatus = Literal["supported", "bounded", "descoped"]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite_non_negative(name: str, value: float) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return resolved


@dataclass(frozen=True, slots=True)
class ChimeraSupportRow:
    """One capability row with evidence and an explicit non-claim."""

    capability: str
    status: SupportStatus
    evidence: str
    non_claim: str

    def __post_init__(self) -> None:
        """Validate the support status and descriptive evidence fields."""
        if self.status not in {"supported", "bounded", "descoped"}:
            raise ValueError("status must be supported, bounded, or descoped")
        for name, value in (
            ("capability", self.capability),
            ("evidence", self.evidence),
            ("non_claim", self.non_claim),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready support row."""
        return {
            "capability": self.capability,
            "status": self.status,
            "evidence": self.evidence,
            "non_claim": self.non_claim,
        }


@dataclass(frozen=True, slots=True)
class SyntheticRegimeEvidence:
    """Measured finite-trajectory evidence for one synthetic regime.

    Population statistics contain two values in canonical population order.
    Objective fields describe one unapplied analytic-gradient proposal evaluated
    at the last settled phase vector.
    """

    regime: SyntheticRegime
    trajectory_digest: str
    trajectory_samples: int
    population_mean: tuple[float, float]
    population_min: tuple[float, float]
    population_max: tuple[float, float]
    population_std: tuple[float, float]
    chimera_index: float
    metastability_index: float
    community_metastability: float
    global_order_mean: float
    objective_before: float
    objective_after: float
    proposal_step_size: float
    proposal_accepted: bool

    def __post_init__(self) -> None:
        """Validate trajectory custody, statistics, and proposal evidence."""
        if not _is_sha256(self.trajectory_digest):
            raise ValueError("trajectory_digest must be a SHA-256 hexadecimal digest")
        if (
            isinstance(self.trajectory_samples, bool)
            or not isinstance(self.trajectory_samples, int)
            or self.trajectory_samples < 1
        ):
            raise ValueError("trajectory_samples must be a positive integer")
        for field_name in (
            "population_mean",
            "population_min",
            "population_max",
            "population_std",
        ):
            values = getattr(self, field_name)
            if len(values) != 2 or not all(np.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must contain two finite values")
        for field_name in ("population_mean", "population_min", "population_max"):
            if not all(0.0 <= value <= 1.0 for value in getattr(self, field_name)):
                raise ValueError(f"{field_name} values must lie in [0, 1]")
        if not all(value >= 0.0 for value in self.population_std):
            raise ValueError("population_std values must be non-negative")
        for index in range(2):
            if not (
                self.population_min[index]
                <= self.population_mean[index]
                <= self.population_max[index]
            ):
                raise ValueError("population statistics must satisfy min <= mean <= max")
        for field_name in (
            "chimera_index",
            "metastability_index",
            "community_metastability",
            "global_order_mean",
            "objective_before",
            "objective_after",
            "proposal_step_size",
        ):
            _finite_non_negative(field_name, getattr(self, field_name))
        if self.global_order_mean > 1.0:
            raise ValueError("global_order_mean must lie in [0, 1]")
        if self.proposal_accepted and not self.objective_after < self.objective_before:
            raise ValueError("accepted proposal evidence must show strict objective decrease")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready regime metrics without rounding."""
        return {
            "regime": self.regime.value,
            "trajectory_digest": self.trajectory_digest,
            "trajectory_samples": self.trajectory_samples,
            "population_mean": list(self.population_mean),
            "population_min": list(self.population_min),
            "population_max": list(self.population_max),
            "population_std": list(self.population_std),
            "chimera_index": self.chimera_index,
            "metastability_index": self.metastability_index,
            "community_metastability": self.community_metastability,
            "global_order_mean": self.global_order_mean,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "proposal_step_size": self.proposal_step_size,
            "proposal_accepted": self.proposal_accepted,
        }


@dataclass(frozen=True, slots=True)
class ChimeraMultiscaleEvidence:
    """Complete deterministic chimera-control evidence payload.

    The payload binds the exact synthetic configurations, two measured regime
    rows, finite-difference agreement, topology-ledger before/after violations,
    support matrix, claim boundary, and a SHA-256 content digest.
    """

    schema_version: str
    generated_on: str
    population_size: int
    chimera: SyntheticRegimeEvidence
    synchronised_control: SyntheticRegimeEvidence
    gradient_max_abs_error: float
    topology_violation_before: float
    topology_violation_after: float
    topology_digest: str
    support: tuple[ChimeraSupportRow, ...]
    claim_boundary: str
    content_digest: str

    def __post_init__(self) -> None:
        """Validate the complete evidence payload and its digest fields."""
        if self.schema_version != CHIMERA_CONTROL_EVIDENCE_SCHEMA:
            raise ValueError("schema_version mismatch")
        if not self.generated_on.strip():
            raise ValueError("generated_on must be non-empty")
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size < 2
        ):
            raise ValueError("population_size must be an integer greater than one")
        for name in (
            "gradient_max_abs_error",
            "topology_violation_before",
            "topology_violation_after",
        ):
            _finite_non_negative(name, getattr(self, name))
        if not _is_sha256(self.topology_digest):
            raise ValueError("topology_digest must be a SHA-256 hexadecimal digest")
        if not self.support:
            raise ValueError("support must contain at least one row")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        if not _is_sha256(self.content_digest):
            raise ValueError("content_digest must be a SHA-256 hexadecimal digest")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-ready payload."""
        return {
            "schema_version": self.schema_version,
            "generated_on": self.generated_on,
            "population_size": self.population_size,
            "chimera": self.chimera.to_dict(),
            "synchronised_control": self.synchronised_control.to_dict(),
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "topology_violation_before": self.topology_violation_before,
            "topology_violation_after": self.topology_violation_after,
            "topology_digest": self.topology_digest,
            "support": [row.to_dict() for row in self.support],
            "claim_boundary": self.claim_boundary,
            "content_digest": self.content_digest,
        }


def _specification(run: SyntheticChimeraRun) -> ChimeraControlSpecification:
    if run.config.regime is SyntheticRegime.CHIMERA_TRANSIENT:
        population_target = (1.0, 0.5)
        ensemble_target = 0.7
    else:
        population_target = (1.0, 1.0)
        ensemble_target = 1.0
    return ChimeraControlSpecification(
        run.hierarchy,
        (
            HierarchyTarget("population", population_target),
            HierarchyTarget("ensemble", (ensemble_target,), weight=0.25),
        ),
    )


def _regime_evidence(run: SyntheticChimeraRun) -> SyntheticRegimeEvidence:
    observables = measure_multiscale_order_parameters(run.settled_phases, run.hierarchy)
    population = observables.level("population").community_order_parameters
    objective = build_chimera_control_objective(_specification(run))
    proposal = propose_phase_control_step(objective, run.settled_phases[-1])
    population_mean = np.mean(population, axis=0)
    population_min = np.min(population, axis=0)
    population_max = np.max(population, axis=0)
    population_std = np.std(population, axis=0)
    return SyntheticRegimeEvidence(
        regime=run.config.regime,
        trajectory_digest=run.content_digest,
        trajectory_samples=run.settled_phases.shape[0],
        population_mean=(float(population_mean[0]), float(population_mean[1])),
        population_min=(float(population_min[0]), float(population_min[1])),
        population_max=(float(population_max[0]), float(population_max[1])),
        population_std=(float(population_std[0]), float(population_std[1])),
        chimera_index=run.diagnostics.chimera_index,
        metastability_index=run.diagnostics.metastability_index,
        community_metastability=run.diagnostics.community_metastability,
        global_order_mean=float(np.mean(observables.global_order_parameter)),
        objective_before=proposal.original_value,
        objective_after=proposal.proposed_value,
        proposal_step_size=proposal.step_size,
        proposal_accepted=proposal.accepted,
    )


def _gradient_error(run: SyntheticChimeraRun) -> float:
    objective = build_chimera_control_objective(_specification(run))
    phases = np.array(run.settled_phases[-1], copy=True)
    analytic = objective.evaluate(phases).gradient
    finite = np.empty_like(phases)
    epsilon = 1.0e-6
    for index in range(phases.size):
        plus = phases.copy()
        minus = phases.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    return float(np.max(np.abs(analytic - finite)))


def _support_rows() -> tuple[ChimeraSupportRow, ...]:
    return (
        ChimeraSupportRow(
            "synthetic chimera generators",
            "supported",
            "exact production Sakaguchi force with deterministic RK4 and two frozen regimes",
            "finite trajectories do not prove a thermodynamic-limit attractor",
        ),
        ChimeraSupportRow(
            "differentiable chimera and cluster losses",
            "supported",
            "composed existing analytic cluster-order gradients with finite-difference replay",
            "a local phase objective is not a closed-loop stability certificate",
        ),
        ChimeraSupportRow(
            "multiscale order-parameter suite",
            "supported",
            "nested population and ensemble partitions measured through oscillatools",
            "synthetic hierarchy does not validate a biological or EEG hierarchy",
        ),
        ChimeraSupportRow(
            "optional challenge-registry extension",
            "descoped",
            "the chimera-control package has a direct tested facade; the unrelated registry extension has no consumer",
            "absence from the optional challenge registry does not imply missing production APIs",
        ),
        ChimeraSupportRow(
            "topology-constraint interaction",
            "bounded",
            "existing TopologyConstraintLedger projection with before/after violation custody",
            "projection is not differentiable learning, PH, DLA, hardware, or controllability proof",
        ),
        ChimeraSupportRow(
            "notebook and evidence artefact",
            "supported",
            "executable notebook 50 and deterministic JSON/Markdown byte-check runner",
            "tutorial output is configuration-specific research evidence",
        ),
    )


def _content_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_chimera_multiscale_evidence(
    *,
    population_size: int = 64,
) -> ChimeraMultiscaleEvidence:
    """Build deterministic finite synthetic evidence for both frozen regimes.

    ``population_size=64`` is the committed configuration. The function runs
    no provider, QPU, hardware, network, optimiser service, or external action.
    It integrates two local trajectories, evaluates existing analytic
    objectives, finite-differences one gradient, and projects one local
    coupling candidate through the existing topology ledger.
    """
    chimera_run = generate_two_population_chimera(
        SyntheticChimeraConfig.for_regime(
            SyntheticRegime.CHIMERA_TRANSIENT,
            population_size=population_size,
        )
    )
    synchronised_run = generate_two_population_chimera(
        SyntheticChimeraConfig.for_regime(
            SyntheticRegime.SYNCHRONISED_CONTROL,
            population_size=population_size,
        )
    )
    candidate = np.array(chimera_run.coupling, copy=True) * 1.6
    np.fill_diagonal(candidate, 0.2)
    candidate[0, 1] = candidate[1, 0] = -0.1
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(
            0.0,
            chimera_run.config.intra_coupling / population_size,
        ),
        sign_policy="nonnegative",
        total_weight=(0.0, float(np.sum(chimera_run.coupling))),
    )
    topology = project_chimera_coupling(candidate, chimera_run.hierarchy, ledger)
    chimera_evidence = _regime_evidence(chimera_run)
    synchronised_evidence = _regime_evidence(synchronised_run)
    gradient_error = _gradient_error(chimera_run)
    support = _support_rows()
    base: dict[str, object] = {
        "schema_version": CHIMERA_CONTROL_EVIDENCE_SCHEMA,
        "generated_on": CHIMERA_CONTROL_EVIDENCE_DATE,
        "population_size": population_size,
        "chimera": chimera_evidence.to_dict(),
        "synchronised_control": synchronised_evidence.to_dict(),
        "gradient_max_abs_error": gradient_error,
        "topology_violation_before": topology.violations_before.total,
        "topology_violation_after": topology.violations_after.total,
        "topology_digest": topology.content_digest,
        "support": [row.to_dict() for row in support],
        "claim_boundary": CHIMERA_CONTROL_CLAIM_BOUNDARY,
    }
    return ChimeraMultiscaleEvidence(
        schema_version=CHIMERA_CONTROL_EVIDENCE_SCHEMA,
        generated_on=CHIMERA_CONTROL_EVIDENCE_DATE,
        population_size=population_size,
        chimera=chimera_evidence,
        synchronised_control=synchronised_evidence,
        gradient_max_abs_error=gradient_error,
        topology_violation_before=topology.violations_before.total,
        topology_violation_after=topology.violations_after.total,
        topology_digest=topology.content_digest,
        support=support,
        claim_boundary=CHIMERA_CONTROL_CLAIM_BOUNDARY,
        content_digest=_content_digest(base),
    )


def render_chimera_multiscale_markdown(evidence: ChimeraMultiscaleEvidence) -> str:
    """Render the evidence payload as deterministic reviewer-facing Markdown."""
    chimera = evidence.chimera
    synchronised = evidence.synchronised_control
    lines = [
        "# Chimera and Multiscale Control Evidence",
        "",
        f"- Schema: `{evidence.schema_version}`",
        f"- Generated on: `{evidence.generated_on}`",
        f"- Population size: `{evidence.population_size}` per population",
        f"- Content digest: `{evidence.content_digest}`",
        f"- Claim boundary: {evidence.claim_boundary}",
        "",
        "## Frozen finite synthetic measurements",
        "",
        "| Metric | Chimera transient | Synchronised control |",
        "|---|---:|---:|",
        f"| Chimera index | {chimera.chimera_index:.12g} | {synchronised.chimera_index:.12g} |",
        f"| Global order mean | {chimera.global_order_mean:.12g} | {synchronised.global_order_mean:.12g} |",
        f"| Population 1 mean | {chimera.population_mean[0]:.12g} | {synchronised.population_mean[0]:.12g} |",
        f"| Population 2 mean | {chimera.population_mean[1]:.12g} | {synchronised.population_mean[1]:.12g} |",
        f"| Population 2 minimum | {chimera.population_min[1]:.12g} | {synchronised.population_min[1]:.12g} |",
        f"| Population 2 standard deviation | {chimera.population_std[1]:.12g} | {synchronised.population_std[1]:.12g} |",
        f"| Objective before proposal | {chimera.objective_before:.12g} | {synchronised.objective_before:.12g} |",
        f"| Objective after proposal | {chimera.objective_after:.12g} | {synchronised.objective_after:.12g} |",
        "",
        "The two rows share the exact finite-N equation, integration step, phase lag, seed, and initial-condition construction; only their frozen coupling regime and run length differ. The measurements are regression evidence for this configuration, not an attractor, generalisation, or physical-domain claim.",
        "",
        "## Differentiation and topology custody",
        "",
        f"- Maximum analytic-versus-central-difference gradient error: `{evidence.gradient_max_abs_error:.12g}`.",
        f"- Topology ledger violation before projection: `{evidence.topology_violation_before:.12g}`.",
        f"- Topology ledger violation after projection: `{evidence.topology_violation_after:.12g}`.",
        f"- Topology content digest: `{evidence.topology_digest}`.",
        "",
        "## Scope matrix",
        "",
        "| Capability | Status | Evidence | Non-claim |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {row.capability} | {row.status} | {row.evidence} | {row.non_claim} |"
        for row in evidence.support
    )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py",
            "PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_chimera_multiscale_evidence(
    evidence: ChimeraMultiscaleEvidence,
    *,
    json_path: Path,
    markdown_path: Path,
    check: bool = False,
) -> tuple[Path, Path]:
    """Write or byte-check deterministic JSON and Markdown evidence files.

    In ``check`` mode both files must already exist and match the rendered
    bytes exactly; a mismatch raises ``RuntimeError`` and no file is changed.
    In write mode parent directories are created and both UTF-8 files end with
    one newline.
    """
    json_text = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    markdown_text = render_chimera_multiscale_markdown(evidence)
    if check:
        for path, expected in (
            (json_path, json_text),
            (markdown_path, markdown_text),
        ):
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise RuntimeError(f"evidence drift: {path}")
        return json_path, markdown_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "CHIMERA_CONTROL_EVIDENCE_DATE",
    "CHIMERA_CONTROL_EVIDENCE_SCHEMA",
    "ChimeraMultiscaleEvidence",
    "ChimeraSupportRow",
    "SyntheticRegimeEvidence",
    "build_chimera_multiscale_evidence",
    "render_chimera_multiscale_markdown",
    "write_chimera_multiscale_evidence",
]
