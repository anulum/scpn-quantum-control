# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA topology-control contracts
"""Immutable support contracts for DLA/topology-constrained control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal

DLA_TOPOLOGY_CLAIM_BOUNDARY = (
    "finite synthetic parity-sector and fixed-active-set topology derivatives only; "
    "no full-DLA, controllability, persistent-homology derivative, hardware-protection, "
    "error-correction, provider, QPU, or deployment claim"
)

SupportStatus = Literal["supported", "unsupported", "descoped"]


class DifferentiabilityKind(Enum):
    """Mathematical derivative class for one constrained operation."""

    LINEAR = "linear"
    AFFINE = "affine"
    PIECEWISE_SMOOTH = "piecewise_smooth"
    NON_SMOOTH = "non_smooth"
    DISCRETE = "discrete"
    NOT_APPLICABLE = "not_applicable"


class ParitySector(Enum):
    """Computational-basis parity sector selected by Hamming weight."""

    EVEN = 0
    ODD = 1


class UnsupportedDifferentiableConstraintError(ValueError):
    """Raised when a caller requests a derivative on an unsupported branch."""


@dataclass(frozen=True, slots=True)
class ConstraintSupportRow:
    """One auditable derivative-support decision.

    Parameters
    ----------
    capability:
        Stable operation or constraint name.
    status:
        ``supported``, ``unsupported``, or explicitly ``descoped``.
    differentiability:
        Mathematical class of the exact operation under review.
    evidence:
        Concrete reason or implemented derivative rule.
    boundary:
        What the row does not establish.

    """

    capability: str
    status: SupportStatus
    differentiability: DifferentiabilityKind
    evidence: str
    boundary: str

    def __post_init__(self) -> None:
        """Validate and normalize support metadata against closed enums."""
        for name in ("capability", "evidence", "boundary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.status not in {"supported", "unsupported", "descoped"}:
            raise ValueError("status must be supported, unsupported, or descoped")
        if not isinstance(self.differentiability, DifferentiabilityKind):
            raise ValueError("differentiability must be a DifferentiabilityKind")

    def to_dict(self) -> dict[str, str]:
        """Return the row as deterministic JSON-compatible strings.

        Returns
        -------
        dict[str, str]
            Stable capability, status, derivative class, evidence, and
            boundary fields.

        """
        return {
            "capability": self.capability,
            "status": self.status,
            "differentiability": self.differentiability.value,
            "evidence": self.evidence,
            "boundary": self.boundary,
        }


@dataclass(frozen=True, slots=True)
class DifferentiabilityReport:
    """Ordered derivative-support decisions for one exact projection point.

    Parameters
    ----------
    rows:
        Non-empty, capability-unique decisions in production projection order.
    claim_boundary:
        Human-readable limit on what the local report establishes.

    Notes
    -----
    ``derivative_supported`` is true only when every row is supported. A
    descoped row therefore blocks JVP/VJP execution just like an unsupported
    row; callers cannot accidentally treat omitted mathematics as identity.

    """

    rows: tuple[ConstraintSupportRow, ...]
    claim_boundary: str = DLA_TOPOLOGY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate row presence, capability uniqueness, and claim metadata."""
        if not self.rows:
            raise ValueError("rows must contain at least one support decision")
        capabilities = tuple(row.capability for row in self.rows)
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("support-row capabilities must be unique")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    @property
    def derivative_supported(self) -> bool:
        """Return whether every in-scope row supports the requested derivative."""
        return all(row.status == "supported" for row in self.rows)

    @property
    def blocking_capabilities(self) -> tuple[str, ...]:
        """Return unsupported or descoped capability names in report order."""
        return tuple(row.capability for row in self.rows if row.status != "supported")

    @property
    def content_digest(self) -> str:
        """Return a SHA-256 digest of ordered support decisions."""
        payload = {
            "rows": [row.to_dict() for row in self.rows],
            "claim_boundary": self.claim_boundary,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require_supported(self) -> None:
        """Require every row to support the requested local derivative.

        Raises
        ------
        UnsupportedDifferentiableConstraintError
            If one or more capabilities are unsupported or descoped. The
            exception message preserves blocker order from ``rows``.

        """
        if self.derivative_supported:
            return
        blockers = ", ".join(self.blocking_capabilities)
        raise UnsupportedDifferentiableConstraintError(
            f"projection derivative is unsupported for: {blockers}"
        )


__all__ = [
    "ConstraintSupportRow",
    "DLA_TOPOLOGY_CLAIM_BOUNDARY",
    "DifferentiabilityKind",
    "DifferentiabilityReport",
    "ParitySector",
    "SupportStatus",
    "UnsupportedDifferentiableConstraintError",
]
