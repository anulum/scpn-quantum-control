# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Frozen cost quotes whose unknown totals and prices never read as zero."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .compiled import CompiledBundle, validate_compiled_bundle
from .target import TargetSnapshot
from .wire import (
    COST_QUOTE_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_DERIVATIONS = ("provider_estimate", "published_rate_card")
_COUNTS = ("job_count", "overhead_executions", "resource_units", "price_minor_units")


@dataclass(frozen=True, slots=True)
class QuotedCount:
    """A known non-negative count, or an unknown that names what it blocks."""

    state: str
    value: int | None
    blocker: str | None

    def __post_init__(self) -> None:
        """Refuse an unknown that carries a number and a known count without one."""
        if self.state == "known":
            if self.blocker is not None:
                raise ValueError("a known count cannot carry a blocker")
            if type(self.value) is not int or not 0 <= self.value <= 10**15:
                raise ValueError("a known count must be a bounded non-negative integer")
        elif self.state == "unknown":
            if self.value is not None:
                raise ValueError("an unknown count cannot carry a value; zero is not unknown")
            object.__setattr__(self, "blocker", _text(self.blocker, name="count blocker"))
        else:
            raise ValueError("count state must be known or unknown")

    def to_wire(self) -> dict[str, object]:
        """Return the exact state, value and blocker."""
        return {"state": self.state, "value": self.value, "blocker": self.blocker}

    @classmethod
    def from_wire(cls, value: object) -> QuotedCount:
        """Reject a count record with missing or extra fields."""
        if type(value) is not dict or set(value) != {"state", "value", "blocker"}:
            raise ValueError("QuotedCount fields mismatch")
        return cls(state=value["state"], value=value["value"], blocker=value["blocker"])


@dataclass(frozen=True, slots=True)
class CostQuote:
    """Design-only resource and price quote for one exact compiled bundle."""

    compiled_bundle_digest: str
    bundle_payload_digest: str
    circuit_count: int
    total_shots: int
    quoted_options: tuple[str, ...]
    job_count: QuotedCount
    overhead_executions: QuotedCount
    resource_unit: str
    resource_derivation: str
    resource_units: QuotedCount
    price_minor_units: QuotedCount
    currency: str
    pricing_source: str
    priced_at_utc: str | None
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse shots sold as credits, untimed prices and unknowns posing as zero."""
        for name in ("compiled_bundle_digest", "bundle_payload_digest"):
            _digest(getattr(self, name), name=name)
        _positive_int(self.circuit_count, name="quoted circuit count", maximum=10_000_000)
        _positive_int(self.total_shots, name="quoted total shots", maximum=10**12)
        if type(self.quoted_options) is not tuple:
            raise ValueError("quoted options must be a tuple")
        names = tuple(_text(name, name="quoted option") for name in self.quoted_options)
        if names != tuple(sorted(set(names))):
            raise ValueError("quoted options must be sorted and unique")
        for name in _COUNTS:
            if type(getattr(self, name)) is not QuotedCount:
                raise ValueError(f"{name} must be a QuotedCount")
        if self.job_count.value == 0:
            raise ValueError("a known job count must be positive")
        self._check_resources_and_price()
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "cost_quote":
            raise ValueError("CostQuote requires its exact artifact header")
        if self.header.data_origin not in ("provider_metadata", "synthetic_classical"):
            raise ValueError("a cost quote derives from provider metadata or a fixture")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.compiled_bundle_digest,)
        )

    def _check_resources_and_price(self) -> None:
        """Require a non-shot resource unit, a named derivation and a sourced, timed price."""
        unit = _text(self.resource_unit, name="resource unit")
        if unit.lower() in ("shot", "shots"):
            raise ValueError("provider credits are not shots")
        if self.resource_derivation == "shots_as_credits":
            raise ValueError("provider credits are not shots")
        if self.resource_derivation not in _DERIVATIONS:
            raise ValueError("unknown resource derivation")
        if type(self.currency) is not str or _CURRENCY.fullmatch(self.currency) is None:
            raise ValueError("currency must be an ISO 4217 code")
        object.__setattr__(
            self, "pricing_source", _text(self.pricing_source, name="pricing source")
        )
        if self.price_minor_units.state == "known":
            if type(self.priced_at_utc) is not str or _UTC.fullmatch(self.priced_at_utc) is None:
                raise ValueError("a known price needs its ISO-8601 UTC quote time")
        elif self.priced_at_utc is not None:
            raise ValueError("an unknown price cannot carry a quote time")

    def blockers(self) -> tuple[tuple[str, str], ...]:
        """Return each unknown total with the blocker it places on approval."""
        return tuple(
            (name, str(getattr(self, name).blocker))
            for name in _COUNTS
            if getattr(self, name).state == "unknown"
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the quote content hashed by its header."""
        return {
            "schema": COST_QUOTE_SCHEMA,
            "object_kind": "cost_quote",
            "compiled_bundle_digest": self.compiled_bundle_digest,
            "bundle_payload_digest": self.bundle_payload_digest,
            "circuit_count": self.circuit_count,
            "total_shots": self.total_shots,
            "quoted_options": list(self.quoted_options),
            **{name: getattr(self, name).to_wire() for name in _COUNTS},
            "resource_unit": self.resource_unit,
            "resource_derivation": self.resource_derivation,
            "currency": self.currency,
            "pricing_source": self.pricing_source,
            "priced_at_utc": self.priced_at_utc,
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned quote."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> CostQuote:
        """Reject unknown fields and restore the frozen quote."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("CostQuote fields mismatch")
        if value["schema"] != COST_QUOTE_SCHEMA or value["object_kind"] != "cost_quote":
            raise ValueError("unknown CostQuote schema")
        if type(value["quoted_options"]) is not list:
            raise ValueError("quoted options must be a list on wire")
        return cls(
            compiled_bundle_digest=value["compiled_bundle_digest"],
            bundle_payload_digest=value["bundle_payload_digest"],
            circuit_count=value["circuit_count"],
            total_shots=value["total_shots"],
            quoted_options=tuple(value["quoted_options"]),
            job_count=QuotedCount.from_wire(value["job_count"]),
            overhead_executions=QuotedCount.from_wire(value["overhead_executions"]),
            resource_unit=value["resource_unit"],
            resource_derivation=value["resource_derivation"],
            resource_units=QuotedCount.from_wire(value["resource_units"]),
            price_minor_units=QuotedCount.from_wire(value["price_minor_units"]),
            currency=value["currency"],
            pricing_source=value["pricing_source"],
            priced_at_utc=value["priced_at_utc"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_cost_quote(quote: CostQuote, bundle: CompiledBundle, target: TargetSnapshot) -> None:
    """Refuse a quote for another bundle, wrong totals, hidden options or invented chunking."""
    if type(quote) is not CostQuote:
        raise ValueError("quote validation needs a CostQuote")
    validate_compiled_bundle(bundle, target)
    if (
        quote.compiled_bundle_digest
        != hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    ):
        raise ValueError("quote prices a different compiled bundle")
    if quote.bundle_payload_digest != bundle.payload_digest:
        raise ValueError("quote payload digest differs from the bundle")
    if quote.circuit_count != len(bundle.cells):
        raise ValueError("quoted circuit count differs from the bundle")
    if quote.total_shots != sum(cell.shots for cell in bundle.cells):
        raise ValueError("quoted total shots differ from the bundle")
    if quote.quoted_options != tuple(name for name, _ in bundle.provider_options):
        raise ValueError("quote does not name every provider option the bundle runs with")
    if quote.header.data_origin != bundle.header.data_origin:
        raise ValueError("quote and bundle data origins differ")
    limit = target.max_circuits_per_job
    if limit.state == "known":
        jobs = math.ceil(len(bundle.cells) / int(limit.require(name="max_circuits_per_job")))
        if quote.job_count.value != jobs:
            raise ValueError("quoted job count differs from the target's circuit batch limit")
    elif quote.job_count.state == "known":
        raise ValueError("a job count cannot be known while the batch limit is unknown")


def decode_cost_quote(raw: bytes) -> CostQuote:
    """Decode one cost quote from strict canonical wire bytes."""
    return CostQuote.from_wire(_strict_json(raw))
