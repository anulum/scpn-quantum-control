# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable operator admission inputs
"""Bound operator policy and dated estimates without loading a provider SDK."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime

from ..canonical_encoding import canonical_digest

MAX_OPERATOR_INTEGER = 2**63 - 1
"""Product transport bound; never an observed provider quota."""


def _text(value: object) -> str:
    """Require bounded Unicode scalar text without control characters."""
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("operator text must contain 1..256 Unicode scalars")
    if any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("operator text contains unsupported characters")
    return value


def _hash(value: object) -> str:
    """Require an exact lowercase SHA-256 identity."""
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("operator identity must be lowercase SHA-256")
    return value


def _integer(value: object) -> int:
    """Require a positive exact integer within the declared transport bound."""
    if (
        type(value) is not int
        or not isinstance(value, int)
        or not 1 <= value <= MAX_OPERATOR_INTEGER
    ):
        raise ValueError("operator counts must be positive uint63 integers")
    return value


def _boolean(value: object) -> bool:
    """Refuse numeric or text imitations of an unattended permission."""
    if type(value) is not bool or not isinstance(value, bool):
        raise ValueError("unattended must be a boolean")
    return value


def _currency(value: object) -> str:
    """Preserve an exact uppercase currency label without exchange inference."""
    if not isinstance(value, str) or not re.fullmatch("[A-Z]{3}", value):
        raise ValueError("currency must contain three uppercase ASCII letters")
    return value


def _money(value: object) -> str:
    """Preserve a bounded nonnegative decimal string without float conversion."""
    if not isinstance(value, str) or not re.fullmatch(
        "(?:0|[1-9][0-9]{0,17})(?:\\.[0-9]{1,9})?", value
    ):
        raise ValueError("cost must be a bounded nonnegative decimal string")
    return value


def _nullable_text(value: object) -> str | None:
    """Retain unknown region or target without an invented default."""
    return None if value is None else _text(value)


def _strings(value: object) -> tuple[str, ...]:
    """Detach one through 256 unique source-owned choices."""
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 256:
        raise ValueError("operator choices require 1..256 entries")
    result = tuple(_text(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("operator choices must be distinct")
    return result


def _fields(
    value: Mapping[str, object],
    cls: type[OperatorPolicy] | type[OperatorRequest] | type[PricingEstimate],
) -> None:
    """Reject missing and extra keys before constructing an input contract."""
    if set(value) != {f.name for f in fields(cls)}:
        raise ValueError("operator contract fields differ")


def utc_second(value: str) -> datetime:
    """Parse exact ASCII UTC seconds without truncation or timezone inference.

    Parameters
    ----------
    value
        Timestamp spelled YYYY-MM-DDTHH:MM:SSZ; fractional seconds are unsupported.

    Returns
    -------
    datetime
        Timezone-aware UTC instant with exact second precision.

    Raises
    ------
    ValueError
        If the spelling or Gregorian date/time is invalid.

    """
    if not isinstance(value, str) or not re.fullmatch(
        "[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ):
        raise ValueError("operator timestamp must be exact ASCII UTC seconds")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@dataclass(frozen=True)
class OperatorPolicy:
    """Immutable plan ceilings supplied by trusted operator configuration.

    Parameters
    ----------
    reference, backend_id
        Opaque policy source and exact HAL route identifiers.
    targets, regions
        Allowed exact names; these declarations do not observe provider availability.
    max_shots, max_concurrency, max_time_limit_ms
        Positive shot, declared concurrent-job and declared millisecond ceilings.
        Concurrency is a plan value, not a global account job ledger; time is a
        requested limit, not a qualified elapsed-time prediction.
    max_cost, currency
        Exact decimal ceiling and currency for the complete request-bound estimate.
    valid_from, expires_at
        Inclusive start and exclusive expiry in ASCII UTC seconds.

    """

    reference: str
    backend_id: str
    targets: tuple[str, ...]
    regions: tuple[str, ...]
    max_shots: int
    max_concurrency: int
    max_time_limit_ms: int
    max_cost: str
    currency: str
    valid_from: str
    expires_at: str

    def __post_init__(self) -> None:
        """Validate all ceilings and detach source choice sequences."""
        _text(self.reference)
        _text(self.backend_id)
        object.__setattr__(self, "targets", _strings(self.targets))
        object.__setattr__(self, "regions", _strings(self.regions))
        for value in (self.max_shots, self.max_concurrency, self.max_time_limit_ms):
            _integer(value)
        _money(self.max_cost)
        _currency(self.currency)
        if utc_second(self.valid_from) >= utc_second(self.expires_at):
            raise ValueError("operator policy validity interval is empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OperatorPolicy:
        """Read exact source fields without accepting imported execution authority.

        Parameters
        ----------
        value
            Complete policy fields from trusted source configuration.

        Returns
        -------
        OperatorPolicy
            Detached immutable policy; construction does not admit a request.

        """
        _fields(value, cls)
        return cls(
            _text(value["reference"]),
            _text(value["backend_id"]),
            _strings(value["targets"]),
            _strings(value["regions"]),
            _integer(value["max_shots"]),
            _integer(value["max_concurrency"]),
            _integer(value["max_time_limit_ms"]),
            _money(value["max_cost"]),
            _currency(value["currency"]),
            _text(value["valid_from"]),
            _text(value["expires_at"]),
        )

    def to_dict(self) -> dict[str, object]:
        """Return fresh exact policy fields for source-owned metadata export."""
        result: dict[str, object] = asdict(self)
        result["targets"], result["regions"] = list(self.targets), list(self.regions)
        return result


@dataclass(frozen=True)
class OperatorRequest:
    """Exact declared operational plan bound to an original HAL workload.

    Parameters
    ----------
    workload_sha256
        Canonical identity of original semantic source and complete native semantics.
    backend_id, target, region
        Requested exact route/device/region; unknown target or region remains None.
    shots, concurrency, time_limit_ms
        Literal positive shots, planned concurrent jobs and declared milliseconds.
    unattended
        Explicit automation permission intent; never inferred from a credential.

    """

    workload_sha256: str
    backend_id: str
    target: str | None
    region: str | None
    shots: int
    concurrency: int
    time_limit_ms: int
    unattended: bool

    def __post_init__(self) -> None:
        """Reject malformed bindings and resources before assessment."""
        _hash(self.workload_sha256)
        _text(self.backend_id)
        _nullable_text(self.target)
        _nullable_text(self.region)
        for value in (self.shots, self.concurrency, self.time_limit_ms):
            _integer(value)
        _boolean(self.unattended)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OperatorRequest:
        """Read a detached request with strict fields and no implicit defaults.

        Parameters
        ----------
        value
            All eight request fields; unknown target/region use explicit None.

        Returns
        -------
        OperatorRequest
            Immutable requested values, without submit approval.

        """
        _fields(value, cls)
        return cls(
            _hash(value["workload_sha256"]),
            _text(value["backend_id"]),
            _nullable_text(value["target"]),
            _nullable_text(value["region"]),
            _integer(value["shots"]),
            _integer(value["concurrency"]),
            _integer(value["time_limit_ms"]),
            _boolean(value["unattended"]),
        )

    def to_dict(self) -> dict[str, object]:
        """Return fresh original plan fields without substituting transport values."""
        return dict(asdict(self))

    @property
    def sha256(self) -> str:
        """Bind every exact request field with the shared typed canonical codec."""
        return canonical_digest("operator_request.v1", self.to_dict())


@dataclass(frozen=True)
class PricingEstimate:
    """Dated source-supplied complete-plan estimate, distinct from actual debit.

    Parameters
    ----------
    request_sha256
        Complete exact OperatorRequest identity, including workload and resources.
    amount
        Nonnegative decimal string, or None for unknown pricing.
    currency, source_ref
        Exact currency label and opaque estimate source reference.
    observed_at, expires_at
        Inclusive observation and exclusive expiry in ASCII UTC seconds.

    """

    request_sha256: str
    amount: str | None
    currency: str
    source_ref: str
    observed_at: str
    expires_at: str

    def __post_init__(self) -> None:
        """Refuse malformed monetary, source and date inputs."""
        _hash(self.request_sha256)
        if self.amount is not None:
            _money(self.amount)
        _currency(self.currency)
        _text(self.source_ref)
        if utc_second(self.observed_at) >= utc_second(self.expires_at):
            raise ValueError("pricing validity interval is empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PricingEstimate:
        """Read exact dated inputs without consulting a provider or price service.

        Parameters
        ----------
        value
            Complete estimate fields; amount is explicitly unknown or a decimal string.

        Returns
        -------
        PricingEstimate
            Immutable supplied estimate, not proof of its producer's authenticity.

        """
        _fields(value, cls)
        amount = value["amount"]
        return cls(
            _hash(value["request_sha256"]),
            None if amount is None else _money(amount),
            _currency(value["currency"]),
            _text(value["source_ref"]),
            _text(value["observed_at"]),
            _text(value["expires_at"]),
        )

    def to_dict(self) -> dict[str, object]:
        """Return fresh supplied estimate inputs, keeping unknown amount explicit."""
        return dict(asdict(self))


@dataclass(frozen=True)
class OperatorPolicyDecision:
    """Source-owned dated verdict; exported decisions never grant fresh submit authority.

    Parameters
    ----------
    request, policy, estimate
        Exact immutable requested plan, configured policy and supplied price inputs.
    assessed_at
        Actual assessment instant in ASCII UTC seconds.
    reasons
        Ordered unique refusal identifiers; empty means plan admission only.
    rejected_substitutions
        Binding differences explicitly refused, never applied to the request.

    """

    request: OperatorRequest
    policy: OperatorPolicy
    estimate: PricingEstimate | None
    assessed_at: str
    reasons: tuple[str, ...]
    rejected_substitutions: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Report the source assessment without implying execution approval."""
        return not self.reasons

    def to_dict(self) -> dict[str, object]:
        """Export exact inputs and refusal provenance without mutating any source."""
        return {
            "request": self.request.to_dict(),
            "policy": self.policy.to_dict(),
            "estimate": None if self.estimate is None else self.estimate.to_dict(),
            "assessed_at": self.assessed_at,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "rejected_substitutions": list(self.rejected_substitutions),
        }


class OperatorPolicyRefused(PermissionError):
    """Retain the exact refusal decision before adapter transport.

    Parameters
    ----------
    decision
        Fresh source-owned refusal; no submission has been invoked by this gate.

    """

    def __init__(self, decision: OperatorPolicyDecision) -> None:
        """Expose immutable diagnostics without replacing requested settings."""
        self.decision = decision
        super().__init__("operator policy refused: " + ", ".join(decision.reasons))
