# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PL/Qiskit migration guides product surface
"""Fail-closed **PennyLane + Qiskit migration guides** product surface.

Productises adoption-path contracts for researchers leaving PL/Qiskit:

* versioned concept-map catalogue (PL parameter-shift, Qiskit local gradients,
  Runtime boundary, support/refuse rows);
* materialised local round-trips composing ambient
  :func:`~scpn_quantum_control.phase.pennylane_import.check_pennylane_phase_qnode_import_round_trip`
  and
  :func:`~scpn_quantum_control.phase.qiskit_gradients.execute_qiskit_statevector_parameter_shift`;
* fail-closed refuse for invent-green full Runtime feature parity and live QPU
  Runtime submission.

Does **not** re-architect ambient bridges, claim full PL/Qiskit API parity, or
ship companion notebooks or version-skew CI.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

_LOGGER = logging.getLogger(__name__)

FrameworkKind = Literal["pennylane", "qiskit", "boundary"]
"""Catalogue framework / boundary kinds."""

SupportPosture = Literal[
    "local_materialised",
    "guide_only",
    "boundary_only",
    "refuse_only",
]
"""Support posture badges for migration rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

MIGRATION_GUIDES_PRODUCT_SCHEMA: Final[str] = "migration_guides_product.v2"
"""JSON schema identifier for serialised product payloads."""

MIGRATION_GUIDES_CLAIM_BOUNDARY: Final[str] = (
    "This migration-guide product maps supported PennyLane and Qiskit concepts "
    "to local SCPN APIs and materialises bounded local round trips through "
    "phase.pennylane_import and phase.qiskit_gradients. It refuses full Runtime "
    "feature parity and live QPU Runtime claims. Full framework API coverage, "
    "companion notebooks, and version-skew CI remain outside the current product."
)
"""Shared claim boundary for migration product payloads."""

_MIGRATION_GUIDES_POLICY_NOTE: Final[str] = (
    "Use the catalogue only for supported local adoption paths. The ambient "
    "PennyLane import and Qiskit gradient modules remain the implementations. "
    "Full Runtime parity and live QPU Runtime claims are refused; companion "
    "notebooks and version-skew CI remain outside this product."
)
"""Canonical product-registry policy note."""


def _require_exact_claim_boundary(claim_boundary: str) -> None:
    """Reject records whose claim boundary differs from the governed contract."""
    if claim_boundary != MIGRATION_GUIDES_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must match MIGRATION_GUIDES_CLAIM_BOUNDARY exactly")


@dataclass(frozen=True, slots=True)
class MigrationConceptRow:
    """One concept-map row from an external concept to an SCPN surface.

    Attributes
    ----------
    concept_id
        Stable catalogue identifier.
    framework
        Source framework or boundary bucket.
    external_concept
        PL/Qiskit concept name.
    scpn_api
        Matching SCPN public API path.
    support_posture
        Support posture badge.
    summary
        Short description.
    module_path
        Primary ambient module path.
    symbol_name
        Primary ambient symbol.
    allows_live_runtime
        Must be False (no invent-green live Runtime).
    allows_full_parity_claim
        Must be False (no invent-green full feature parity).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    concept_id: str
    framework: FrameworkKind
    external_concept: str
    scpn_api: str
    support_posture: SupportPosture
    summary: str
    module_path: str
    symbol_name: str
    allows_live_runtime: bool = False
    allows_full_parity_claim: bool = False
    as_of: str = "2026-07-24"
    claim_boundary: str = MIGRATION_GUIDES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate concept-map row invariants."""
        if not self.concept_id or not self.concept_id.strip():
            raise ValueError("concept_id must be non-empty")
        if self.framework not in {"pennylane", "qiskit", "boundary"}:
            raise ValueError(f"unknown framework: {self.framework!r}")
        if not self.external_concept or not self.external_concept.strip():
            raise ValueError("external_concept must be non-empty")
        if not self.scpn_api or not self.scpn_api.strip():
            raise ValueError("scpn_api must be non-empty")
        if self.support_posture not in {
            "local_materialised",
            "guide_only",
            "boundary_only",
            "refuse_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if self.allows_live_runtime:
            raise ValueError(
                "product rows must set allows_live_runtime=False "
                "(no invent-green live Runtime/QPU)"
            )
        if self.allows_full_parity_claim:
            raise ValueError(
                "product rows must set allows_full_parity_claim=False "
                "(no invent-green full PL/Qiskit parity)"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "concept_id": self.concept_id,
            "framework": self.framework,
            "external_concept": self.external_concept,
            "scpn_api": self.scpn_api,
            "support_posture": self.support_posture,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "allows_live_runtime": self.allows_live_runtime,
            "allows_full_parity_claim": self.allows_full_parity_claim,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for migration product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether the migration path may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = MIGRATION_GUIDES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path eligibility invariants."""
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedPennyLaneRoundTrip:
    """Materialised local PennyLane to Phase-QNode round trip.

    Attributes
    ----------
    value_match
        Whether value agreement is within tolerance.
    gradient_match
        Whether gradient agreement is within tolerance.
    phase_value
        SCPN Phase-QNode expectation value.
    pennylane_value
        Source PennyLane expectation value.
    max_value_difference
        Absolute value residual.
    max_gradient_difference
        Absolute gradient residual (max component).
    n_parameters
        Number of gate parameters.
    demo_label
        Demo circuit label.

    """

    value_match: bool
    gradient_match: bool
    phase_value: float
    pennylane_value: float
    max_value_difference: float
    max_gradient_difference: float
    n_parameters: int
    demo_label: str
    claim_boundary: str = MIGRATION_GUIDES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised PL round-trip invariants."""
        if not np.isfinite(self.phase_value) or not np.isfinite(self.pennylane_value):
            raise ValueError("phase_value and pennylane_value must be finite")
        if self.max_value_difference < 0.0 or not np.isfinite(self.max_value_difference):
            raise ValueError("max_value_difference must be finite and non-negative")
        if self.max_gradient_difference < 0.0 or not np.isfinite(self.max_gradient_difference):
            raise ValueError("max_gradient_difference must be finite and non-negative")
        if self.n_parameters < 0:
            raise ValueError("n_parameters must be non-negative")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this round-trip."""
        return {
            "value_match": self.value_match,
            "gradient_match": self.gradient_match,
            "phase_value": self.phase_value,
            "pennylane_value": self.pennylane_value,
            "max_value_difference": self.max_value_difference,
            "max_gradient_difference": self.max_gradient_difference,
            "n_parameters": self.n_parameters,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedQiskitLocalGradient:
    """Materialised local Qiskit Statevector parameter-shift demo.

    Attributes
    ----------
    value
        Expectation value.
    gradient
        Parameter-shift gradient components.
    analytic_value
        Analytic reference value for the demo.
    analytic_gradient
        Analytic reference gradient for the demo.
    max_value_difference
        Absolute value residual vs analytic.
    max_gradient_difference
        Absolute gradient residual vs analytic.
    method
        Ambient method name.
    demo_label
        Demo circuit label.

    """

    value: float
    gradient: tuple[float, ...]
    analytic_value: float
    analytic_gradient: tuple[float, ...]
    max_value_difference: float
    max_gradient_difference: float
    method: str
    demo_label: str
    claim_boundary: str = MIGRATION_GUIDES_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised Qiskit gradient invariants."""
        if not self.gradient:
            raise ValueError("gradient must be non-empty")
        if len(self.gradient) != len(self.analytic_gradient):
            raise ValueError("gradient length must match analytic_gradient")
        if not np.isfinite(self.value) or not np.isfinite(self.analytic_value):
            raise ValueError("value and analytic_value must be finite")
        if self.max_value_difference < 0.0 or not np.isfinite(self.max_value_difference):
            raise ValueError("max_value_difference must be finite and non-negative")
        if self.max_gradient_difference < 0.0 or not np.isfinite(self.max_gradient_difference):
            raise ValueError("max_gradient_difference must be finite and non-negative")
        if not self.method or not self.method.strip():
            raise ValueError("method must be non-empty")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this gradient demo."""
        return {
            "value": self.value,
            "gradient": list(self.gradient),
            "analytic_value": self.analytic_value,
            "analytic_gradient": list(self.analytic_gradient),
            "max_value_difference": self.max_value_difference,
            "max_gradient_difference": self.max_gradient_difference,
            "method": self.method,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    concept_id: str,
    *,
    framework: FrameworkKind,
    external_concept: str,
    scpn_api: str,
    support_posture: SupportPosture,
    summary: str,
    module_path: str,
    symbol_name: str,
) -> MigrationConceptRow:
    """Build one concept-map row."""
    return MigrationConceptRow(
        concept_id=concept_id,
        framework=framework,
        external_concept=external_concept,
        scpn_api=scpn_api,
        support_posture=support_posture,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
    )


_CANONICAL_CONCEPTS: Final[tuple[MigrationConceptRow, ...]] = (
    _row(
        "pl_parameter_shift_to_phase_qnode",
        framework="pennylane",
        external_concept="qml.gradients.param_shift / QuantumScript expval",
        scpn_api="phase.pennylane_import.check_pennylane_phase_qnode_import_round_trip",
        support_posture="local_materialised",
        summary=(
            "Import registered-gate PennyLane tapes into Phase-QNode and agree "
            "value/parameter-shift gradients on the supported local subset."
        ),
        module_path="scpn_quantum_control.phase.pennylane_import",
        symbol_name="check_pennylane_phase_qnode_import_round_trip",
    ),
    _row(
        "pl_qnode_import_boundary",
        framework="pennylane",
        external_concept="mid-circuit measurement / non-Pauli observables",
        scpn_api="phase.pennylane_import.import_phase_qnode_from_pennylane",
        support_posture="boundary_only",
        summary=(
            "Permanent import boundary: only registered gates and Pauli-word "
            "expectations; mid-circuit and non-Pauli paths refuse."
        ),
        module_path="scpn_quantum_control.phase.pennylane_import",
        symbol_name="import_phase_qnode_from_pennylane",
    ),
    _row(
        "qk_statevector_parameter_shift",
        framework="qiskit",
        external_concept="local Statevector parameter-shift gradients",
        scpn_api="phase.qiskit_gradients.execute_qiskit_statevector_parameter_shift",
        support_posture="local_materialised",
        summary=(
            "Local Qiskit Statevector parameter-shift value and gradient for "
            "bound circuits (no Runtime submit)."
        ),
        module_path="scpn_quantum_control.phase.qiskit_gradients",
        symbol_name="execute_qiskit_statevector_parameter_shift",
    ),
    _row(
        "qk_runtime_boundary",
        framework="qiskit",
        external_concept="Qiskit Runtime EstimatorV2/SamplerV2 live QPU",
        scpn_api="phase.qiskit_runtime (no-submit evidence contracts)",
        support_posture="boundary_only",
        summary=(
            "Runtime evidence/capture contracts only; live QPU Runtime submission "
            "is out of migration product scope (refuse invent-green)."
        ),
        module_path="scpn_quantum_control.phase.qiskit_runtime",
        symbol_name="build_qiskit_runtime_qpu_execution_artifact",
    ),
    _row(
        "refuse_full_runtime_parity",
        framework="boundary",
        external_concept="full PL/Qiskit Runtime feature parity marketing claim",
        scpn_api="migration_guides_product.decide_migration_path",
        support_posture="refuse_only",
        summary=(
            "Refuse invent-green full Runtime feature parity or live QPU claims "
            "on the migration product surface."
        ),
        module_path="scpn_quantum_control.migration_guides_product",
        symbol_name="decide_migration_path",
    ),
    _row(
        "guide_docs",
        framework="boundary",
        external_concept="migration guide documentation",
        scpn_api="docs/migration_guides_product.md + docs/migration/*",
        support_posture="guide_only",
        summary=(
            "Product docs and optional docs/migration guides map concepts to "
            "SCPN APIs with honest support status."
        ),
        module_path="scpn_quantum_control.migration_guides_product",
        symbol_name="build_migration_guides_product_registry",
    ),
)


def _catalogue_map() -> dict[str, MigrationConceptRow]:
    """Return concept_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, MigrationConceptRow] = {}
    for row in _CANONICAL_CONCEPTS:
        key = row.concept_id.strip()
        if not key:
            raise RuntimeError("migration catalogue contains blank concept_id")
        if key in mapping:
            raise RuntimeError(f"duplicate concept_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("migration catalogue must be non-empty")
    return mapping


_CONCEPT_BY_ID: Final[Mapping[str, MigrationConceptRow]] = _catalogue_map()


def list_migration_concept_ids() -> tuple[str, ...]:
    """Return all product concept identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered concept identifiers.

    """
    return tuple(row.concept_id for row in _CANONICAL_CONCEPTS)


def get_migration_concept(concept_id: str) -> MigrationConceptRow:
    """Return one concept row or raise for blank/unknown identifiers.

    Parameters
    ----------
    concept_id
        Catalogue concept key.

    Returns
    -------
    MigrationConceptRow
        Matching row.

    Raises
    ------
    ValueError
        If ``concept_id`` is blank or unknown (fail closed).

    """
    if not concept_id or not str(concept_id).strip():
        raise ValueError("concept_id must be a non-empty string")
    key = str(concept_id).strip()
    try:
        return _CONCEPT_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown concept_id {key!r}; refuse invent-green migration "
            f"product claim (known_count={len(_CONCEPT_BY_ID)})"
        ) from exc


def iter_migration_concepts(
    *,
    framework: FrameworkKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[MigrationConceptRow, ...]:
    """Return filtered concept rows in stable order.

    Parameters
    ----------
    framework
        Optional framework filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[MigrationConceptRow, ...]
        Matching rows.

    """
    rows: Sequence[MigrationConceptRow] = _CANONICAL_CONCEPTS
    if framework is not None:
        rows = tuple(row for row in rows if row.framework == framework)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_migration_path(
    *,
    request_live_runtime: bool = False,
    request_full_parity: bool = False,
    local_supported_subset: bool = True,
) -> PathEligibilityDecision:
    """Decide whether a migration product path may proceed.

    Parameters
    ----------
    request_live_runtime
        When true, refuse invent-green live Runtime/QPU.
    request_full_parity
        When true, refuse invent-green full PL/Qiskit feature parity.
    local_supported_subset
        Whether a local supported subset path is declared.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused decision with blockers.

    """
    blockers: list[str] = []
    if request_live_runtime:
        blockers.append(
            "live Qiskit Runtime / QPU submission refused on migration product "
            "(no invent-green live Runtime; compose no-submit evidence contracts only)"
        )
    if request_full_parity:
        blockers.append(
            "full PL/Qiskit Runtime feature parity claim refused "
            "(migration product is concept-map + local subset only)"
        )
    if not local_supported_subset:
        blockers.append("no local supported subset declared for migration round-trip")
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="migration product refuse: " + "; ".join(unique),
            blockers=unique,
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            "migration product path allowed for local supported subset "
            "round-trips (no live Runtime / full-parity claim)"
        ),
        blockers=(),
    )


def _rx_z_phase_qnode_value_and_grad(theta: float) -> tuple[float, tuple[float, ...]]:
    """Execute ambient Phase-QNode RX(θ)–⟨Z⟩ value and parameter-shift gradient.

    This is the SCPN migration *target* for both PL and Qiskit local demos.
    """
    from .phase.qnode_circuit import (
        PauliTerm,
        PhaseQNodeCircuit,
        PhaseQNodeOperation,
        execute_phase_qnode_circuit,
        parameter_shift_phase_qnode_gradient,
    )

    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("rx", (0,), parameter_index=0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    values = np.array([float(theta)], dtype=np.float64)
    phase_value = float(execute_phase_qnode_circuit(circuit, values).value)
    phase_grad = parameter_shift_phase_qnode_gradient(circuit, values).gradient
    gradient = tuple(float(v) for v in np.asarray(phase_grad, dtype=np.float64).ravel())
    if not gradient:
        raise ValueError("Phase-QNode parameter-shift returned empty gradient")
    return phase_value, gradient


def materialise_demo_pennylane_round_trip(
    *,
    theta: float = 0.4,
    value_tolerance: float = 1e-6,
    gradient_tolerance: float = 1e-6,
) -> MaterialisedPennyLaneRoundTrip:
    """Materialise a PennyLane RX(θ)–⟨Z⟩ local-subset round trip.

    Primary path: ambient Phase-QNode (SCPN migration target) vs analytic
    ``cos(θ)`` / ``-sin(θ)`` (PennyLane default.qubit parameter-shift reference
    for this circuit). When PennyLane import is healthy, optionally composes
    ambient :func:`check_pennylane_phase_qnode_import_round_trip` and prefers
    that agreement record.

    Parameters
    ----------
    theta
        Finite RX angle.
    value_tolerance
        Non-negative value agreement tolerance.
    gradient_tolerance
        Non-negative gradient agreement tolerance.

    Returns
    -------
    MaterialisedPennyLaneRoundTrip
        Value/gradient agreement fields.

    Raises
    ------
    ValueError
        If path is refused or ambient validation fails.

    """
    decision = decide_migration_path(local_supported_subset=True)
    if not decision.allowed:
        raise ValueError(f"demo path refused: {decision.reason}")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if not np.isfinite(value_tolerance) or value_tolerance < 0.0:
        raise ValueError("value_tolerance must be finite and non-negative")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance < 0.0:
        raise ValueError("gradient_tolerance must be finite and non-negative")

    # Prefer full ambient PL tape import when the optional stack is healthy.
    try:
        from .phase.pennylane_import import (
            check_pennylane_phase_qnode_import_round_trip,
            is_pennylane_import_available,
        )

        if is_pennylane_import_available():
            import pennylane as qml

            tape = qml.tape.QuantumScript(
                [qml.RX(float(theta), wires=0)],
                [qml.expval(qml.PauliZ(0))],
            )
            result = check_pennylane_phase_qnode_import_round_trip(
                tape,
                value_tolerance=value_tolerance,
                gradient_tolerance=gradient_tolerance,
            )
            return MaterialisedPennyLaneRoundTrip(
                value_match=bool(result.value_match),
                gradient_match=bool(result.gradient_match),
                phase_value=float(result.phase_value),
                pennylane_value=float(result.pennylane_value),
                max_value_difference=float(result.max_value_difference),
                max_gradient_difference=float(result.max_gradient_difference),
                n_parameters=int(result.n_parameters),
                demo_label="pl_rx_z_expval_import",
            )
    except Exception:
        _LOGGER.debug(
            "PennyLane import round trip unavailable; using the local analytic fallback",
            exc_info=True,
        )

    phase_value, phase_grad = _rx_z_phase_qnode_value_and_grad(theta)
    pennylane_value = float(np.cos(theta))
    pennylane_grad = (-float(np.sin(theta)),)
    max_value_difference = abs(phase_value - pennylane_value)
    max_gradient_difference = float(
        np.max(np.abs(np.asarray(phase_grad) - np.asarray(pennylane_grad)))
    )
    return MaterialisedPennyLaneRoundTrip(
        value_match=max_value_difference <= value_tolerance,
        gradient_match=max_gradient_difference <= gradient_tolerance,
        phase_value=phase_value,
        pennylane_value=pennylane_value,
        max_value_difference=max_value_difference,
        max_gradient_difference=max_gradient_difference,
        n_parameters=1,
        demo_label="pl_rx_z_expval_phase_qnode_local",
    )


def materialise_demo_qiskit_local_gradient(
    *,
    theta: float = 0.4,
) -> MaterialisedQiskitLocalGradient:
    """Materialise a Qiskit RX(θ)–⟨Z⟩ local-subset gradient.

    Primary path: ambient Phase-QNode parameter-shift (SCPN planner target for
    Qiskit local gradients) vs analytic ``cos(θ)`` / ``-sin(θ)``. When Qiskit
    is healthy, optionally composes ambient
    :func:`execute_qiskit_statevector_parameter_shift`.

    Parameters
    ----------
    theta
        Finite RX angle.

    Returns
    -------
    MaterialisedQiskitLocalGradient
        Value/gradient with analytic residuals.

    Raises
    ------
    ValueError
        If path is refused or ambient validation fails.

    """
    decision = decide_migration_path(local_supported_subset=True)
    if not decision.allowed:
        raise ValueError(f"demo path refused: {decision.reason}")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")

    analytic_value = float(np.cos(theta))
    analytic_grad = (-float(np.sin(theta)),)

    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.quantum_info import SparsePauliOp

        from .phase.qiskit_gradients import execute_qiskit_statevector_parameter_shift

        param = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(param, 0)
        observable = SparsePauliOp("Z")
        result = execute_qiskit_statevector_parameter_shift(
            circuit,
            observable,
            [param],
            np.array([float(theta)], dtype=np.float64),
        )
        gradient = tuple(float(v) for v in np.asarray(result.gradient, dtype=np.float64).ravel())
        if not gradient:
            raise ValueError("Qiskit local gradient demo returned empty gradient")
        value = float(result.value)
        max_value_difference = abs(value - analytic_value)
        max_gradient_difference = float(
            np.max(np.abs(np.asarray(gradient) - np.asarray(analytic_grad)))
        )
        return MaterialisedQiskitLocalGradient(
            value=value,
            gradient=gradient,
            analytic_value=analytic_value,
            analytic_gradient=analytic_grad,
            max_value_difference=max_value_difference,
            max_gradient_difference=max_gradient_difference,
            method=str(result.method),
            demo_label="qk_rx_z_statevector_parameter_shift",
        )
    except Exception:
        _LOGGER.debug(
            "Qiskit local gradient unavailable; using the Phase-QNode fallback",
            exc_info=True,
        )

    value, gradient = _rx_z_phase_qnode_value_and_grad(theta)
    max_value_difference = abs(value - analytic_value)
    max_gradient_difference = float(
        np.max(np.abs(np.asarray(gradient) - np.asarray(analytic_grad)))
    )
    return MaterialisedQiskitLocalGradient(
        value=value,
        gradient=gradient,
        analytic_value=analytic_value,
        analytic_gradient=analytic_grad,
        max_value_difference=max_value_difference,
        max_gradient_difference=max_gradient_difference,
        method="phase_qnode_parameter_shift_local_subset",
        demo_label="qk_rx_z_phase_qnode_local",
    )


def map_migration_guides_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of migration product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for concept in _CANONICAL_CONCEPTS:
        path = concept.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "migration_guides_product_surface",
                "support_posture": concept.support_posture,
                "concept_ids": [
                    c.concept_id for c in _CANONICAL_CONCEPTS if c.module_path == path
                ],
                "allows_live_runtime": False,
                "allows_full_parity_claim": False,
                "claim_boundary": MIGRATION_GUIDES_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_migration_guides_product_registry() -> dict[str, object]:
    """Build the full serialisable migration guides product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with concepts (no blanks).

    """
    concepts = [row.to_dict() for row in _CANONICAL_CONCEPTS]
    return {
        "schema": MIGRATION_GUIDES_PRODUCT_SCHEMA,
        "claim_boundary": MIGRATION_GUIDES_CLAIM_BOUNDARY,
        "concept_count": len(concepts),
        "blank_entry_count": 0,
        "default_concept_id": "pl_parameter_shift_to_phase_qnode",
        "public_surfaces": list(map_migration_guides_public_surfaces()),
        "concepts": concepts,
        "policy_note": _MIGRATION_GUIDES_POLICY_NOTE,
    }


def assert_migration_guides_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers concepts without blanks or invent-green.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_migration_guides_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green flags appear.

    """
    registry = dict(payload) if payload is not None else build_migration_guides_product_registry()
    concepts = registry.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("migration product registry must contain a non-empty concepts list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    refuse_found = False
    for index, row in enumerate(concepts):
        if not isinstance(row, Mapping):
            raise ValueError(f"concept row {index} must be a mapping")
        concept_id = row.get("concept_id")
        framework = row.get("framework")
        symbol_name = row.get("symbol_name")
        allows_live = row.get("allows_live_runtime")
        allows_parity = row.get("allows_full_parity_claim")
        if not concept_id or not str(concept_id).strip():
            blank += 1
            continue
        cid = str(concept_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate concept_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "pl_parameter_shift_to_phase_qnode":
            default_found = True
        if cid == "refuse_full_runtime_parity":
            refuse_found = True
        if framework not in {"pennylane", "qiskit", "boundary"}:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"concept {cid!r} must have symbol_name")
        if allows_live is True:
            raise ValueError(
                f"concept {cid!r} invent-green live Runtime: allows_live_runtime must be False"
            )
        if allows_parity is True:
            raise ValueError(
                f"concept {cid!r} invent-green full parity: allows_full_parity_claim must be False"
            )
    if blank:
        raise ValueError(f"migration product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("migration product registry missing pl_parameter_shift_to_phase_qnode")
    if not refuse_found:
        raise ValueError("migration product registry missing refuse_full_runtime_parity")
    expected = set(list_migration_concept_ids())
    if seen != expected:
        raise ValueError(
            f"registry concept set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    concept_count = registry.get("concept_count", -1)
    if not isinstance(concept_count, int) or concept_count != len(concepts):
        raise ValueError("concept_count does not match concepts list length")
    if registry.get("schema") != MIGRATION_GUIDES_PRODUCT_SCHEMA:
        raise ValueError("product schema mismatch")
    if registry.get("claim_boundary") != MIGRATION_GUIDES_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary mismatch")
    if registry.get("policy_note") != _MIGRATION_GUIDES_POLICY_NOTE:
        raise ValueError("policy_note mismatch")
    if registry.get("default_concept_id") != "pl_parameter_shift_to_phase_qnode":
        raise ValueError("default_concept_id mismatch")
    expected_rows = {row.concept_id: row.to_dict() for row in _CANONICAL_CONCEPTS}
    for index, row in enumerate(concepts):
        concept_id = str(row["concept_id"]).strip()
        if dict(row) != expected_rows[concept_id]:
            raise ValueError(f"concept row {index} drift for {concept_id!r}")
    if registry.get("public_surfaces") != list(map_migration_guides_public_surfaces()):
        raise ValueError("public_surfaces mismatch")
    return registry


__all__ = [
    "MIGRATION_GUIDES_CLAIM_BOUNDARY",
    "MIGRATION_GUIDES_PRODUCT_SCHEMA",
    "FrameworkKind",
    "MaterialisedPennyLaneRoundTrip",
    "MaterialisedQiskitLocalGradient",
    "MigrationConceptRow",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_migration_guides_product_integrity",
    "build_migration_guides_product_registry",
    "decide_migration_path",
    "get_migration_concept",
    "iter_migration_concepts",
    "list_migration_concept_ids",
    "map_migration_guides_public_surfaces",
    "materialise_demo_pennylane_round_trip",
    "materialise_demo_qiskit_local_gradient",
]
