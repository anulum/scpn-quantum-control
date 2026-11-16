# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU offline bundle quote
"""Quote a compiled bundle offline: exact totals and jobs, with every provider-only figure an explicit blocker."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ..contracts.compiled import CompiledBundle
from ..contracts.cost import CostQuote, QuotedCount, validate_cost_quote
from ..contracts.target import TargetSnapshot
from ..contracts.wire import COST_QUOTE_SCHEMA, ArtifactHeader, canonical_bytes

_EXECUTION_OPTIONS = ("optimization_level", "seed_transpiler")
_NO_ESTIMATE = "no provider usage estimate was read offline; a live estimate needs the owner's provider access"
_NO_PRICE = "no price without a provider estimate or a published rate card for the resource units"


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _jobs(bundle: CompiledBundle, target: TargetSnapshot) -> QuotedCount:
    """Return the job count the target's batch limit implies, or its blocker."""
    limit = target.max_circuits_per_job
    if limit.state == "known":
        return QuotedCount(
            "known",
            math.ceil(len(bundle.cells) / int(limit.require(name="max_circuits_per_job"))),
            None,
        )
    return QuotedCount(
        "unknown", None, f"the target does not report a circuit batch limit: {limit.blocker}"
    )


def _overhead(bundle: CompiledBundle) -> QuotedCount:
    """Return zero extra executions when every bundle option is a compile option, else a blocker."""
    extra = sorted({name for name, _ in bundle.provider_options} - set(_EXECUTION_OPTIONS))
    if extra:
        return QuotedCount("unknown", None, f"execution options may add runs: {', '.join(extra)}")
    return QuotedCount("known", 0, None)


def quote_bundle(
    bundle: CompiledBundle,
    target: TargetSnapshot,
    *,
    resource_unit: str,
    currency: str,
    revisions: tuple[str, str],
) -> CostQuote:
    """Return the offline quote of one compiled bundle; resource units and price stay unknown."""
    fields: dict[str, Any] = {
        "compiled_bundle_digest": _sha(bundle.to_wire()),
        "bundle_payload_digest": bundle.payload_digest,
        "circuit_count": len(bundle.cells),
        "total_shots": sum(cell.shots for cell in bundle.cells),
        "quoted_options": tuple(sorted(name for name, _ in bundle.provider_options)),
        "job_count": _jobs(bundle, target),
        "overhead_executions": _overhead(bundle),
        "resource_unit": resource_unit,
        "resource_derivation": "provider_estimate",
        "resource_units": QuotedCount("unknown", None, _NO_ESTIMATE),
        "price_minor_units": QuotedCount("unknown", None, _NO_PRICE),
        "currency": currency,
        "pricing_source": "not_read_offline",
        "priced_at_utc": None,
    }
    content = {
        "schema": COST_QUOTE_SCHEMA,
        "object_kind": "cost_quote",
        **{
            name: value.to_wire() if type(value) is QuotedCount else value
            for name, value in fields.items()
        },
        "quoted_options": list(fields["quoted_options"]),
    }
    header = ArtifactHeader(
        "cost_quote",
        _sha(content),
        (fields["compiled_bundle_digest"],),
        revisions[0],
        revisions[1],
        "offline_design",
        bundle.header.data_origin,
        "design_only",
    )
    quote = CostQuote(**fields, header=header)
    validate_cost_quote(quote, bundle, target)
    return quote
