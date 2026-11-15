# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Owner approvals scoped to one exact quote, with a detached signature checked by a trust store."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from .compiled import CompiledBundle
from .cost import CostQuote, validate_cost_quote
from .target import TargetSnapshot
from .wire import (
    OWNER_APPROVAL_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_MAX_SIGNATURE_BYTES = 8192
_SCHEMES = ("ml_dsa_65",)
_ORIGIN_BY_KEY = {"live": "owner_decision", "test": "synthetic_classical"}
_EGRESS = ("private_derived", "reviewed_synthetic", "value_free")
_COUNTS = ("job_count", "overhead_executions", "resource_units", "price_minor_units")
_RUNNERS = ("live", "offline_fixture")


class SignatureVerifier(Protocol):
    """Trust-store check of one signature; the lane supplies no default."""

    def __call__(
        self,
        *,
        scheme: str,
        key_id: str,
        key_environment: str,
        message: bytes,
        signature: bytes,
    ) -> bool:
        """Return true only when a key trusted for this environment signed the message."""


@dataclass(frozen=True, slots=True)
class ApprovalSignature:
    """Signer identity plus the SHA-256 of the detached signature over the approval."""

    scheme: str
    key_id: str
    key_environment: str
    signature_sha256: str

    def __post_init__(self) -> None:
        """Refuse unknown schemes, unnamed keys and a missing signature digest."""
        if self.scheme not in _SCHEMES:
            raise ValueError("unsupported approval signature scheme")
        _digest(self.key_id, name="approval key id")
        if self.key_environment not in _ORIGIN_BY_KEY:
            raise ValueError("approval key environment must be live or test")
        _digest(self.signature_sha256, name="approval signature digest")

    def signer(self) -> dict[str, object]:
        """Return the signer identity the signature itself covers."""
        return {
            "scheme": self.scheme,
            "key_id": self.key_id,
            "key_environment": self.key_environment,
        }

    def to_wire(self) -> dict[str, object]:
        """Return the exact signature envelope."""
        return {**self.signer(), "signature_sha256": self.signature_sha256}

    @classmethod
    def from_wire(cls, value: object) -> ApprovalSignature:
        """Reject an envelope with missing or extra fields."""
        fields = {"scheme", "key_id", "key_environment", "signature_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ApprovalSignature fields mismatch")
        return cls(**{name: value[name] for name in sorted(fields)})


@dataclass(frozen=True, slots=True)
class OwnerApproval:
    """Design-only owner approval of one compiled bundle at one quoted cost."""

    approval_id: str
    compiled_bundle_digest: str
    bundle_payload_digest: str
    cost_quote_digest: str
    provider: str
    account_route: str
    device_name: str
    max_shots: int
    max_jobs: int
    max_resource_units: int
    max_price_minor_units: int
    currency: str
    valid_from_utc: str
    expires_at_utc: str
    approved_egress: str
    signature: ApprovalSignature
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse string authority, open-ended validity and unbounded caps."""
        for name in ("approval_id", "provider", "account_route", "device_name"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in ("compiled_bundle_digest", "bundle_payload_digest", "cost_quote_digest"):
            _digest(getattr(self, name), name=name)
        for name in ("max_shots", "max_jobs", "max_resource_units", "max_price_minor_units"):
            _positive_int(getattr(self, name), name=name, maximum=10**15)
        if type(self.currency) is not str or _CURRENCY.fullmatch(self.currency) is None:
            raise ValueError("currency must be an ISO 4217 code")
        for name in ("valid_from_utc", "expires_at_utc"):
            value = getattr(self, name)
            if type(value) is not str or _UTC.fullmatch(value) is None:
                raise ValueError(f"{name} must be an ISO-8601 UTC second")
        if self.expires_at_utc <= self.valid_from_utc:
            raise ValueError("an approval must expire after it becomes valid")
        if self.approved_egress not in _EGRESS:
            raise ValueError("unknown approved data-egress scope")
        if type(self.signature) is not ApprovalSignature:
            raise ValueError("an approval needs a structured owner signature, not a string")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "owner_approval":
            raise ValueError("OwnerApproval requires its exact artifact header")
        if self.header.data_origin != _ORIGIN_BY_KEY[self.signature.key_environment]:
            raise ValueError("artifact data origin does not match the signing key environment")
        parents = (self.compiled_bundle_digest, self.cost_quote_digest)
        self.header.validate_content(self._scientific_wire(), parents=parents)

    def _scope_wire(self) -> dict[str, object]:
        """Return the versioned scope, caps and validity window."""
        return {
            "schema": OWNER_APPROVAL_SCHEMA,
            "object_kind": "owner_approval",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in ("signature", "header")
            },
        }

    def signed_content(self) -> dict[str, object]:
        """Return the scope plus the signer identity, everything the signature covers."""
        return {**self._scope_wire(), "signer": self.signature.signer()}

    def signed_message(self) -> bytes:
        """Return the canonical bytes the owner key signs."""
        return canonical_bytes(self.signed_content())

    def _scientific_wire(self) -> dict[str, object]:
        """Return the approval content, signature included, hashed by its header."""
        return {**self._scope_wire(), "signature": self.signature.to_wire()}

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned approval."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> OwnerApproval:
        """Reject unknown fields and restore the frozen approval."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("OwnerApproval fields mismatch")
        if value["schema"] != OWNER_APPROVAL_SCHEMA or value["object_kind"] != "owner_approval":
            raise ValueError("unknown OwnerApproval schema")
        plain = [name for name in cls.__dataclass_fields__ if name not in ("signature", "header")]
        return cls(
            **{name: value[name] for name in plain},
            signature=ApprovalSignature.from_wire(value["signature"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _check_caps(approval: OwnerApproval, quote: CostQuote) -> None:
    """Refuse a quote with any unknown total or above an approved cap."""
    known: dict[str, int] = {}
    for name in _COUNTS:
        quoted = getattr(quote, name)
        if quoted.value is None:
            raise ValueError(f"an approval cannot cover an unknown {name}: {quoted.blocker}")
        known[name] = quoted.value
    for name, quoted_value, cap in (
        ("shots", quote.total_shots, approval.max_shots),
        ("job_count", known["job_count"], approval.max_jobs),
        ("resource_units", known["resource_units"], approval.max_resource_units),
        ("price_minor_units", known["price_minor_units"], approval.max_price_minor_units),
    ):
        if quoted_value > cap:
            raise ValueError(f"quoted {name} above the approved cap")
    if quote.currency != approval.currency:
        raise ValueError("quote and approval currencies differ")


def validate_owner_approval(
    approval: OwnerApproval,
    quote: CostQuote,
    bundle: CompiledBundle,
    target: TargetSnapshot,
    *,
    now_utc: str,
    runner_environment: str,
    signature: bytes,
    verifier: SignatureVerifier,
) -> None:
    """Refuse an approval for another payload, outside its caps or window, or unsigned."""
    if type(approval) is not OwnerApproval:
        raise ValueError("approval validation needs an OwnerApproval")
    validate_cost_quote(quote, bundle, target)
    if (
        approval.compiled_bundle_digest
        != hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    ):
        raise ValueError("approval covers a different compiled bundle")
    if approval.bundle_payload_digest != bundle.payload_digest:
        raise ValueError("approval covers a different compiled payload")
    if approval.cost_quote_digest != hashlib.sha256(canonical_bytes(quote.to_wire())).hexdigest():
        raise ValueError("approval covers a different cost quote")
    scope = (approval.provider, approval.account_route, approval.device_name)
    if scope != (target.provider, target.account_route, target.device_name):
        raise ValueError("approval scope differs from the target route and device")
    _check_caps(approval, quote)
    if type(now_utc) is not str or _UTC.fullmatch(now_utc) is None:
        raise ValueError("approval check time must be an ISO-8601 UTC second")
    if not approval.valid_from_utc <= now_utc < approval.expires_at_utc:
        raise ValueError("approval is outside its validity window")
    if runner_environment not in _RUNNERS:
        raise ValueError("unknown runner environment")
    if runner_environment == "live" and approval.signature.key_environment == "test":
        raise ValueError("a test key cannot approve a live run")
    if type(signature) is not bytes or not 0 < len(signature) <= _MAX_SIGNATURE_BYTES:
        raise ValueError("a detached approval signature must be bounded bytes")
    envelope = approval.signature
    if hashlib.sha256(signature).hexdigest() != envelope.signature_sha256:
        raise ValueError("detached signature differs from the approved signature digest")
    verified = verifier(
        scheme=envelope.scheme,
        key_id=envelope.key_id,
        key_environment=envelope.key_environment,
        message=approval.signed_message(),
        signature=signature,
    )
    if verified is not True:
        raise ValueError("approval signature does not verify against the trust store")


def decode_owner_approval(raw: bytes) -> OwnerApproval:
    """Decode one owner approval from strict canonical wire bytes."""
    return OwnerApproval.from_wire(_strict_json(raw))
