# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Identity / robustness control observers
"""Fail-closed control observers over the existing identity metrics.

The product adapts the ambient robustness certificate, coherence budget, and
optional CHSH witness into an immutable co-design/control-stack safety decision. Thresholds
are caller-supplied and explicit. No identity key is replaced or claimed
unbreakable, and the attested-result seal remains an optional pointer only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector

from .identity.coherence_budget import coherence_budget, fidelity_at_depth
from .identity.entanglement_witness import chsh_from_statevector
from .identity.robustness import compute_robustness_certificate

IDENTITY_OBSERVER_SCHEMA: Final[str] = "identity_observer_product.v1"
IDENTITY_OBSERVER_CLAIM_BOUNDARY: Final[str] = (
    "control-loop observer adapter over existing identity metrics; thresholds are "
    "explicit research controls; no unbreakable-identity, cryptographic-strength, "
    "hardware robustness, or universal consciousness claim"
)
ATTESTED_RESULT_SEAL_POINTER: Final[str] = "optional attested-result seal"
ControlAction = Literal["continue", "hold", "abort"]
WitnessStatus = Literal["not_requested", "supported", "unsupported"]


@dataclass(frozen=True, slots=True)
class IdentityMetricInventoryRow:
    """One ambient identity metric and its loop-safety posture."""

    metric_id: str
    module_path: str
    loop_safe: bool
    role: str
    claim_boundary: str = IDENTITY_OBSERVER_CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class IdentityObserverThresholds:
    """Explicit safety thresholds for one observer evaluation."""

    min_energy_gap: float
    max_transition_probability: float
    min_coherence_fidelity: float
    min_chsh_when_observed: float = 2.0

    def __post_init__(self) -> None:
        """Validate finite threshold ranges."""
        values = (
            self.min_energy_gap,
            self.max_transition_probability,
            self.min_coherence_fidelity,
            self.min_chsh_when_observed,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("identity observer thresholds must be finite")
        if self.min_energy_gap <= 0.0:
            raise ValueError("min_energy_gap must be positive")
        if not 0.0 <= self.max_transition_probability <= 1.0:
            raise ValueError("max_transition_probability must be in [0, 1]")
        if not 0.0 < self.min_coherence_fidelity <= 1.0:
            raise ValueError("min_coherence_fidelity must be in (0, 1]")
        if not 0.0 <= self.min_chsh_when_observed <= 2.0 * np.sqrt(2.0):
            raise ValueError("min_chsh_when_observed must be in [0, 2*sqrt(2)]")


@dataclass(frozen=True, slots=True)
class IdentityObserverRecord:
    """Immutable identity metrics mapped into control-loop telemetry."""

    energy_gap: float
    transition_probability: float
    adiabatic_bound: float
    planned_depth: int
    coherence_max_depth: int
    coherence_fidelity: float
    witness_status: WitnessStatus
    chsh_value: float | None
    witness_pair: tuple[int, int] | None
    seal_pointer: str = ATTESTED_RESULT_SEAL_POINTER
    claim_boundary: str = IDENTITY_OBSERVER_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready observer mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentitySafetyDecision:
    """Control decision derived from identity observers."""

    allowed: bool
    action: ControlAction
    reason: str
    blockers: tuple[str, ...]
    observer: IdentityObserverRecord
    thresholds: IdentityObserverThresholds
    schema: str = IDENTITY_OBSERVER_SCHEMA
    claim_boundary: str = IDENTITY_OBSERVER_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready safety decision."""
        return {
            "schema": self.schema,
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "observer": self.observer.to_dict(),
            "thresholds": asdict(self.thresholds),
            "claim_boundary": self.claim_boundary,
        }


_INVENTORY: Final[tuple[IdentityMetricInventoryRow, ...]] = (
    IdentityMetricInventoryRow(
        "robustness_gap",
        "scpn_quantum_control.identity.robustness.compute_robustness_certificate",
        True,
        "energy-gap, transition-probability, and adiabatic-bound observer",
    ),
    IdentityMetricInventoryRow(
        "coherence_budget",
        "scpn_quantum_control.identity.coherence_budget",
        True,
        "planned-depth fidelity observer",
    ),
    IdentityMetricInventoryRow(
        "entanglement_witness",
        "scpn_quantum_control.identity.entanglement_witness.chsh_from_statevector",
        True,
        "optional CHSH observer; unsupported inputs abort",
    ),
    IdentityMetricInventoryRow(
        "identity_key",
        "scpn_quantum_control.identity.identity_key",
        False,
        "existing identity-key product; never a control metric",
    ),
    IdentityMetricInventoryRow(
        "binding_spec",
        "scpn_quantum_control.identity.binding_spec",
        False,
        "topology definition; never interpreted as universal identity proof",
    ),
)

_UNSUITABLE_SCENARIOS: Final[tuple[str, ...]] = (
    "using observer thresholds as proof of unbreakable identity",
    "treating simulated coherence as measured hardware robustness",
    "requiring CHSH from a missing, invalid, or unsupported statevector",
    "interpreting no CHSH violation as absence of all entanglement",
    "using identity observers as consciousness, personhood, or clinical evidence",
)


def identity_metric_inventory() -> tuple[IdentityMetricInventoryRow, ...]:
    """Return the frozen identity-observer ownership inventory."""
    return _INVENTORY


def identity_observer_unsuitable_scenarios() -> tuple[str, ...]:
    """Return explicit over-interpretations that must remain unsupported."""
    return _UNSUITABLE_SCENARIOS


def evaluate_identity_safety(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    thresholds: IdentityObserverThresholds,
    planned_depth: int,
    n_qubits: int,
    noise_strength: float = 0.01,
    sweep_rate: float = 0.1,
    statevector: Statevector | None = None,
    witness_pair: tuple[int, int] | None = None,
    require_witness: bool = False,
) -> IdentitySafetyDecision:
    """Evaluate ambient identity metrics and produce a fail-closed safety trip.

    Robustness and coherence adapters always use the existing production
    functions. A requested witness is evaluated only when both a statevector and
    pair are present; missing or invalid witness inputs abort the control path.
    """
    if planned_depth < 0:
        raise ValueError("planned_depth must be non-negative")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")

    certificate = compute_robustness_certificate(
        np.asarray(K, dtype=np.float64),
        np.asarray(omega, dtype=np.float64),
        noise_strength=noise_strength,
        sweep_rate=sweep_rate,
    )
    budget = coherence_budget(
        n_qubits,
        fidelity_threshold=thresholds.min_coherence_fidelity,
    )
    fidelity = fidelity_at_depth(planned_depth, n_qubits)
    blockers: list[str] = []
    if certificate.energy_gap < thresholds.min_energy_gap:
        blockers.append(
            f"energy_gap {certificate.energy_gap:.12g} below minimum "
            f"{thresholds.min_energy_gap:.12g}"
        )
    if certificate.transition_probability > thresholds.max_transition_probability:
        blockers.append(
            f"transition_probability {certificate.transition_probability:.12g} exceeds maximum "
            f"{thresholds.max_transition_probability:.12g}"
        )
    if fidelity < thresholds.min_coherence_fidelity:
        blockers.append(
            f"coherence_fidelity {fidelity:.12g} below minimum "
            f"{thresholds.min_coherence_fidelity:.12g}"
        )
    if planned_depth > int(budget["max_depth"]):
        blockers.append(
            f"planned_depth {planned_depth} exceeds coherence budget {int(budget['max_depth'])}"
        )

    witness_requested = require_witness or statevector is not None or witness_pair is not None
    witness_status: WitnessStatus = "not_requested"
    chsh_value: float | None = None
    if witness_requested:
        if statevector is None or witness_pair is None:
            witness_status = "unsupported"
            blockers.append("requested CHSH witness requires statevector and witness_pair")
        else:
            try:
                chsh_value = chsh_from_statevector(statevector, *witness_pair)
                witness_status = "supported"
            except ValueError as exc:
                witness_status = "unsupported"
                blockers.append(f"requested CHSH witness unsupported: {exc}")
            if chsh_value is not None and chsh_value < thresholds.min_chsh_when_observed:
                blockers.append(
                    f"CHSH value {chsh_value:.12g} below observed threshold "
                    f"{thresholds.min_chsh_when_observed:.12g}"
                )

    observer = IdentityObserverRecord(
        energy_gap=float(certificate.energy_gap),
        transition_probability=float(certificate.transition_probability),
        adiabatic_bound=float(certificate.adiabatic_bound),
        planned_depth=planned_depth,
        coherence_max_depth=int(budget["max_depth"]),
        coherence_fidelity=float(fidelity),
        witness_status=witness_status,
        chsh_value=chsh_value,
        witness_pair=witness_pair if witness_status == "supported" else None,
    )
    unique_blockers = tuple(dict.fromkeys(blockers))
    if witness_status == "unsupported":
        action: ControlAction = "abort"
    elif unique_blockers:
        action = "hold"
    else:
        action = "continue"
    return IdentitySafetyDecision(
        allowed=not unique_blockers,
        action=action,
        reason=(
            "identity observer thresholds satisfied; control may continue"
            if not unique_blockers
            else f"identity safety trip: {action}"
        ),
        blockers=unique_blockers,
        observer=observer,
        thresholds=thresholds,
    )


__all__ = [
    "ATTESTED_RESULT_SEAL_POINTER",
    "IDENTITY_OBSERVER_CLAIM_BOUNDARY",
    "IDENTITY_OBSERVER_SCHEMA",
    "ControlAction",
    "IdentityMetricInventoryRow",
    "IdentityObserverRecord",
    "IdentityObserverThresholds",
    "IdentitySafetyDecision",
    "WitnessStatus",
    "evaluate_identity_safety",
    "identity_metric_inventory",
    "identity_observer_unsuitable_scenarios",
]
