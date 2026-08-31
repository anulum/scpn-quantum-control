# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-control deterministic evidence
"""Deterministic evidence custody for DLA/topology constrained control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.topology_control import (
    CouplingGraphBounds,
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedSPSAOptimizer,
    TopologyConstraintLedger,
)

from .objectives import ParityProtectedQuadraticObjective
from .optimizer import ProjectedGradientConfig, optimise_parity_protected_state
from .parity import ParitySectorProjector
from .projection import (
    topology_projection_jvp,
    topology_projection_support,
    topology_projection_vjp,
)
from .schema import (
    DLA_TOPOLOGY_CLAIM_BOUNDARY,
    ConstraintSupportRow,
    DifferentiabilityKind,
    ParitySector,
)

TOPOLOGY_CONTROL_EVIDENCE_SCHEMA = "topology_control_evidence_v2"
TOPOLOGY_CONTROL_EVIDENCE_DATE = "2026-07-29"

ComplexArray = NDArray[np.complex128]


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True, slots=True)
class DlaTopologyControlEvidence:
    """Frozen synthetic evidence and exact derivative-support rows.

    Parameters
    ----------
    schema_version, generated_on:
        Versioned evidence schema and fixed generation date.
    n_qubits, sector:
        Dense synthetic parity problem size and selected sector.
    initial_objective, final_objective:
        Objective endpoints; construction requires strict decrease.
    initial_leakage_mass, final_leakage_mass:
        Unnormalised outside-sector mass endpoints.
    accepted_steps:
        Positive count of strict-decrease projected steps.
    parity_gradient_max_abs_error, parity_jvp_max_abs_error:
        Maximum analytic-versus-central-difference errors.
    topology_jvp_max_abs_error, topology_adjoint_error:
        Fixed-active-set finite-difference and JVP/VJP identity errors.
    existing_optimizer_final_violation:
        Final production-ledger violation from the composed existing SPSA path.
    topology_differential_digest, trace_digest:
        SHA-256 custody digests for the local topology record and optimisation
        trace.
    unsupported_blockers:
        Ordered unique names of deliberately refused topology branches.
    support:
        Ordered capability decisions, including the optional QGNN wiring row.
    claim_boundary:
        Finite synthetic, no-hardware interpretation boundary.
    content_digest:
        SHA-256 of every preceding canonical evidence field.

    """

    schema_version: str
    generated_on: str
    n_qubits: int
    sector: str
    initial_objective: float
    final_objective: float
    initial_leakage_mass: float
    final_leakage_mass: float
    accepted_steps: int
    parity_gradient_max_abs_error: float
    parity_jvp_max_abs_error: float
    topology_jvp_max_abs_error: float
    topology_adjoint_error: float
    existing_optimizer_final_violation: float
    topology_differential_digest: str
    trace_digest: str
    unsupported_blockers: tuple[str, ...]
    support: tuple[ConstraintSupportRow, ...]
    claim_boundary: str
    content_digest: str

    def __post_init__(self) -> None:
        """Validate and normalize the frozen topology-evidence record."""
        if self.schema_version != TOPOLOGY_CONTROL_EVIDENCE_SCHEMA:
            raise ValueError("schema_version is unsupported")
        if not isinstance(self.generated_on, str) or not self.generated_on.strip():
            raise ValueError("generated_on must be a non-empty string")
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise ValueError("n_qubits must be an integer")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.sector not in {"even", "odd"}:
            raise ValueError("sector must be even or odd")
        metrics = (
            "initial_objective",
            "final_objective",
            "initial_leakage_mass",
            "final_leakage_mass",
            "parity_gradient_max_abs_error",
            "parity_jvp_max_abs_error",
            "topology_jvp_max_abs_error",
            "topology_adjoint_error",
            "existing_optimizer_final_violation",
        )
        for name in metrics:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.final_objective >= self.initial_objective:
            raise ValueError("evidence requires strict objective decrease")
        if self.final_leakage_mass > self.initial_leakage_mass:
            raise ValueError("final leakage must not exceed initial leakage")
        if (
            isinstance(self.accepted_steps, bool)
            or not isinstance(self.accepted_steps, int)
            or self.accepted_steps < 1
        ):
            raise ValueError("accepted_steps must be a positive integer")
        for name in ("topology_differential_digest", "trace_digest", "content_digest"):
            if not _is_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.unsupported_blockers or any(
            not isinstance(item, str) or not item.strip() for item in self.unsupported_blockers
        ):
            raise ValueError("unsupported_blockers must contain non-empty names")
        if not self.support:
            raise ValueError("support must contain at least one row")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence data.

        Parameters
        ----------
        include_digest:
            Include the top-level ``content_digest`` field when true. False is
            useful when independently recomputing the canonical digest.

        Returns
        -------
        dict[str, object]
            Ordered semantic fields whose nested support rows contain only
            JSON-native values.

        """
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "generated_on": self.generated_on,
            "n_qubits": self.n_qubits,
            "sector": self.sector,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "initial_leakage_mass": self.initial_leakage_mass,
            "final_leakage_mass": self.final_leakage_mass,
            "accepted_steps": self.accepted_steps,
            "parity_gradient_max_abs_error": self.parity_gradient_max_abs_error,
            "parity_jvp_max_abs_error": self.parity_jvp_max_abs_error,
            "topology_jvp_max_abs_error": self.topology_jvp_max_abs_error,
            "topology_adjoint_error": self.topology_adjoint_error,
            "existing_optimizer_final_violation": self.existing_optimizer_final_violation,
            "topology_differential_digest": self.topology_differential_digest,
            "trace_digest": self.trace_digest,
            "unsupported_blockers": list(self.unsupported_blockers),
            "support": [row.to_dict() for row in self.support],
            "claim_boundary": self.claim_boundary,
        }
        if include_digest:
            payload["content_digest"] = self.content_digest
        return payload


def _normalised_sector_target(
    projector: ParitySectorProjector,
    rng: np.random.Generator,
) -> ComplexArray:
    raw = np.asarray(
        rng.normal(size=projector.dimension) + 1j * rng.normal(size=projector.dimension),
        dtype=np.complex128,
    )
    target = np.asarray(projector.project(raw), dtype=np.complex128)
    return np.asarray(target / np.linalg.norm(target), dtype=np.complex128)


def _parity_gradient_error(
    objective: ParityProtectedQuadraticObjective,
    state: ComplexArray,
    epsilon: float = 1.0e-6,
) -> float:
    analytic = objective.evaluate(state).gradient
    errors: list[float] = []
    for index in range(state.size):
        real_direction = np.zeros_like(state)
        real_direction[index] = 1.0
        real_fd = (
            objective(state + epsilon * real_direction)
            - objective(state - epsilon * real_direction)
        ) / (2.0 * epsilon)
        imaginary_direction = np.zeros_like(state)
        imaginary_direction[index] = 1.0j
        imaginary_fd = (
            objective(state + epsilon * imaginary_direction)
            - objective(state - epsilon * imaginary_direction)
        ) / (2.0 * epsilon)
        errors.extend(
            (
                abs(float(analytic[index].real) - real_fd),
                abs(float(analytic[index].imag) - imaginary_fd),
            )
        )
    return float(max(errors, default=0.0))


def _parity_jvp_error(
    projector: ParitySectorProjector,
    state: ComplexArray,
    tangent: ComplexArray,
    epsilon: float = 1.0e-6,
) -> float:
    central = (
        projector.project(state + epsilon * tangent) - projector.project(state - epsilon * tangent)
    ) / (2.0 * epsilon)
    return float(np.max(np.abs(projector.jvp(tangent) - central)))


def _support_rows() -> tuple[ConstraintSupportRow, ...]:
    return (
        ConstraintSupportRow(
            "differentiability boundary",
            "supported",
            DifferentiabilityKind.AFFINE,
            "linear/affine branches are separated from non-smooth and discrete branches",
            "classification is exact-local rather than a universal smoothness claim",
        ),
        ConstraintSupportRow(
            "existing contract inventory",
            "supported",
            DifferentiabilityKind.NOT_APPLICABLE,
            "the facade composes DLA parity and topology_control owners",
            "inventory is not new mathematical evidence",
        ),
        ConstraintSupportRow(
            "penalties and projections",
            "supported",
            DifferentiabilityKind.PIECEWISE_SMOOTH,
            "parity JVP/VJP is exact and topology JVP/VJP is fixed-active-set only",
            "PH, connectivity, kinks, and active budget rescaling fail closed",
        ),
        ConstraintSupportRow(
            "constrained optimiser",
            "supported",
            DifferentiabilityKind.AFFINE,
            "synthetic parity projection occurs inside every strict-decrease proposal",
            "no physical Hamiltonian, controller, or hardware is actuated",
        ),
        ConstraintSupportRow(
            "deterministic evidence",
            "supported",
            DifferentiabilityKind.NOT_APPLICABLE,
            "central differences, adjoint identity, custody digests, and blockers are frozen",
            "one finite configuration is not generalisation evidence",
        ),
        ConstraintSupportRow(
            "optional QGNN wiring",
            "descoped",
            DifferentiabilityKind.NOT_APPLICABLE,
            "no current QGNN consumer maps Hilbert-space parity onto graph-message topology",
            "parity and graph topology are not conflated without a typed consumer",
        ),
        ConstraintSupportRow(
            "constraint versus witness documentation",
            "supported",
            DifferentiabilityKind.NOT_APPLICABLE,
            "public docs distinguish projected constraints, diagnostics, and non-claims",
            "documentation does not promote hardware DLA protection",
        ),
    )


def build_dla_topology_control_evidence(
    *,
    n_qubits: int = 4,
    seed: int = 540,
) -> DlaTopologyControlEvidence:
    """Build the deterministic finite synthetic DLA/topology evidence bundle.

    The builder checks parity objective gradients and projector JVPs against
    central differences, a topology-ledger JVP against the production forward
    map, the JVP/VJP adjoint identity, fail-closed blocker discovery, and final
    constraint compliance of the existing projected SPSA optimiser.

    Parameters
    ----------
    n_qubits:
        Dense local parity problem size in ``[2, 8]``.
    seed:
        Integer seed controlling every synthetic array and SPSA perturbation.

    Returns
    -------
    DlaTopologyControlEvidence
        Immutable metrics, support rows, blockers, and custody digests.

    Raises
    ------
    ValueError
        If ``n_qubits`` or ``seed`` violates the bounded public contract.

    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 2 <= n_qubits <= 8:
        raise ValueError("evidence n_qubits must be an integer in [2, 8]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    projector = ParitySectorProjector(n_qubits, ParitySector.EVEN)
    target = _normalised_sector_target(projector, rng)
    initial = np.asarray(
        target
        + 0.35
        * (rng.normal(size=projector.dimension) + 1j * rng.normal(size=projector.dimension)),
        dtype=np.complex128,
    )
    objective = ParityProtectedQuadraticObjective(projector, target, leakage_weight=2.0)
    initial_evaluation = objective.evaluate(initial)
    trace = optimise_parity_protected_state(
        initial,
        objective,
        ProjectedGradientConfig(max_steps=40, initial_step_size=0.5),
    )
    final_evaluation = objective.evaluate(trace.final_state)
    tangent = np.asarray(
        rng.normal(size=projector.dimension) + 1j * rng.normal(size=projector.dimension),
        dtype=np.complex128,
    )

    topology_ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        hardware_edges={(0, 1), (1, 2), (2, 3), (0, 3)},
        frozen_edges={(0, 1): 0.25},
    )
    matrix = rng.uniform(-0.8, 0.8, size=(4, 4))
    np.fill_diagonal(matrix, 0.0)
    matrix_tangent = rng.normal(size=(4, 4))
    cotangent = rng.normal(size=(4, 4))
    topology = topology_projection_jvp(topology_ledger, matrix, matrix_tangent)
    epsilon = 1.0e-6
    topology_central = (
        topology_ledger.project(matrix + epsilon * matrix_tangent)
        - topology_ledger.project(matrix - epsilon * matrix_tangent)
    ) / (2.0 * epsilon)
    topology_jvp_error = float(np.max(np.abs(topology.projected_tangent - topology_central)))
    topology_vjp = topology_projection_vjp(topology_ledger, matrix, cotangent)
    topology_adjoint_error = abs(
        float(np.vdot(topology.projected_tangent, cotangent).real)
        - float(np.vdot(matrix_tangent, topology_vjp).real)
    )

    unsupported_reports = (
        topology_projection_support(TopologyConstraintLedger(), np.zeros((4, 4))),
        topology_projection_support(
            TopologyConstraintLedger(
                bounds=CouplingGraphBounds(-2.0, 2.0),
                sign_policy="signed",
                total_weight=(1.0, 2.0),
            ),
            np.zeros((4, 4)),
        ),
        topology_projection_support(
            TopologyConstraintLedger(
                bounds=CouplingGraphBounds(-2.0, 2.0),
                sign_policy="signed",
                algebraic_connectivity_min=0.2,
            ),
            matrix,
        ),
    )
    blockers = tuple(
        dict.fromkeys(
            blocker for report in unsupported_reports for blocker in report.blocking_capabilities
        )
    )

    existing_ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(0.0, 1.0),
        sign_policy="nonnegative",
        hardware_edges={(0, 1), (1, 2), (2, 3), (0, 3)},
    )
    existing_objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.2),
        ledger=existing_ledger,
        h1_target=1.0,
        allow_approximate_ph_backend=True,
        allow_degenerate=True,
    )
    existing_trace = ProjectedSPSAOptimizer(seed=seed, max_steps=3).optimise(
        rng.uniform(-1.0, 2.0, size=(4, 4)),
        existing_objective,
    )
    existing_violation = existing_ledger.violations(existing_trace.final_matrix).total

    support = _support_rows()
    parity_gradient_error = _parity_gradient_error(objective, initial)
    parity_jvp_error = _parity_jvp_error(projector, initial, tangent)
    canonical: dict[str, object] = {
        "schema_version": TOPOLOGY_CONTROL_EVIDENCE_SCHEMA,
        "generated_on": TOPOLOGY_CONTROL_EVIDENCE_DATE,
        "n_qubits": n_qubits,
        "sector": "even",
        "initial_objective": initial_evaluation.value,
        "final_objective": final_evaluation.value,
        "initial_leakage_mass": initial_evaluation.leakage_mass,
        "final_leakage_mass": final_evaluation.leakage_mass,
        "accepted_steps": trace.accepted_steps,
        "parity_gradient_max_abs_error": parity_gradient_error,
        "parity_jvp_max_abs_error": parity_jvp_error,
        "topology_jvp_max_abs_error": topology_jvp_error,
        "topology_adjoint_error": float(topology_adjoint_error),
        "existing_optimizer_final_violation": float(existing_violation),
        "topology_differential_digest": topology.content_digest,
        "trace_digest": trace.content_digest,
        "unsupported_blockers": list(blockers),
        "support": [row.to_dict() for row in support],
        "claim_boundary": DLA_TOPOLOGY_CLAIM_BOUNDARY,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DlaTopologyControlEvidence(
        schema_version=TOPOLOGY_CONTROL_EVIDENCE_SCHEMA,
        generated_on=TOPOLOGY_CONTROL_EVIDENCE_DATE,
        n_qubits=n_qubits,
        sector="even",
        initial_objective=initial_evaluation.value,
        final_objective=final_evaluation.value,
        initial_leakage_mass=initial_evaluation.leakage_mass,
        final_leakage_mass=final_evaluation.leakage_mass,
        accepted_steps=trace.accepted_steps,
        parity_gradient_max_abs_error=parity_gradient_error,
        parity_jvp_max_abs_error=parity_jvp_error,
        topology_jvp_max_abs_error=topology_jvp_error,
        topology_adjoint_error=float(topology_adjoint_error),
        existing_optimizer_final_violation=float(existing_violation),
        topology_differential_digest=topology.content_digest,
        trace_digest=trace.content_digest,
        unsupported_blockers=blockers,
        support=support,
        claim_boundary=DLA_TOPOLOGY_CLAIM_BOUNDARY,
        content_digest=digest,
    )


def render_dla_topology_control_markdown(evidence: DlaTopologyControlEvidence) -> str:
    """Render a deterministic Markdown evidence report.

    Parameters
    ----------
    evidence:
        Validated immutable evidence object to render.

    Returns
    -------
    str
        Newline-terminated report with endpoint metrics, derivative errors,
        blockers, support rows, digest, and non-claims.

    """
    lines = [
        "# DLA and Topology-Constrained Control Evidence",
        "",
        f"- Schema: `{evidence.schema_version}`",
        f"- Generated: `{evidence.generated_on}`",
        f"- Content digest: `{evidence.content_digest}`",
        f"- Claim boundary: {evidence.claim_boundary}",
        "",
        "## Synthetic parity-protected task",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Qubits | {evidence.n_qubits} |",
        f"| Sector | {evidence.sector} |",
        f"| Initial objective | {evidence.initial_objective:.12g} |",
        f"| Final objective | {evidence.final_objective:.12g} |",
        f"| Initial leakage mass | {evidence.initial_leakage_mass:.12g} |",
        f"| Final leakage mass | {evidence.final_leakage_mass:.12g} |",
        f"| Accepted projected steps | {evidence.accepted_steps} |",
        "",
        "## Derivative checks",
        "",
        "| Check | Maximum/absolute error |",
        "|---|---:|",
        f"| Parity objective gradient | {evidence.parity_gradient_max_abs_error:.12g} |",
        f"| Parity projector JVP | {evidence.parity_jvp_max_abs_error:.12g} |",
        f"| Topology-ledger JVP | {evidence.topology_jvp_max_abs_error:.12g} |",
        f"| Topology JVP/VJP adjoint identity | {evidence.topology_adjoint_error:.12g} |",
        f"| Existing projected optimiser final violation | {evidence.existing_optimizer_final_violation:.12g} |",
        "",
        "Unsupported topology derivative blockers: "
        + ", ".join(f"`{item}`" for item in evidence.unsupported_blockers)
        + ".",
        "",
        "## Slice support",
        "",
        "| Slice | Status | Derivative class | Evidence | Boundary |",
        "|---|---|---|---|---|",
    ]
    for row in evidence.support:
        lines.append(
            f"| {row.capability} | {row.status} | {row.differentiability.value} | "
            f"{row.evidence} | {row.boundary} |"
        )
    lines.extend(
        [
            "",
            "These rows are finite synthetic regression evidence. They are not a full-DLA,",
            "controllability, differentiable-PH, hardware-protection, error-correction,",
            "provider, QPU, advantage, or deployment result.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_dla_topology_control_evidence(
    evidence: DlaTopologyControlEvidence,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    check: bool = False,
) -> tuple[Path, Path]:
    """Write or byte-check canonical JSON and Markdown evidence files.

    Parameters
    ----------
    evidence:
        Validated immutable evidence object.
    json_path, markdown_path:
        Destination paths for sorted UTF-8 JSON and rendered Markdown.
    check:
        If true, perform a read-only byte comparison and refuse any drift. If
        false, create parent directories and replace both exact artefacts.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        JSON and Markdown paths in argument order.

    Raises
    ------
    RuntimeError
        In check mode when either file is absent or differs byte-for-byte.
    OSError
        If filesystem reads, directory creation, or writes fail.

    """
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_bytes = (
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    markdown_bytes = render_dla_topology_control_markdown(evidence).encode()
    targets = ((json_target, json_bytes), (markdown_target, markdown_bytes))
    if check:
        for path, expected in targets:
            if not path.exists() or path.read_bytes() != expected:
                raise RuntimeError(f"DLA/topology evidence drift: {path}")
        return json_target, markdown_target
    for path, content in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return json_target, markdown_target


__all__ = [
    "TOPOLOGY_CONTROL_EVIDENCE_DATE",
    "TOPOLOGY_CONTROL_EVIDENCE_SCHEMA",
    "DlaTopologyControlEvidence",
    "build_dla_topology_control_evidence",
    "render_dla_topology_control_markdown",
    "write_dla_topology_control_evidence",
]
