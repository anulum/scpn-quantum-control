# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — independent semantic capture and refusal operations
"""Capture semantic snapshots and refuse unsupported transforms or aggregation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .stable_core_product import digest_stable_core_payload


@dataclass(frozen=True, slots=True)
class CapturedSemanticRecord:
    """Immutable joint snapshot of a raw record and its companion.

    Attributes
    ----------
    raw_record
        Deep copy of the raw payload as it was at capture time.
    companion
        Deep copy of the companion, or ``None``.
    raw_digest
        Digest of the captured raw payload.
    companion_digest
        Digest of the captured companion, or ``None``.

    """

    raw_record: Mapping[str, Any]
    companion: Mapping[str, Any] | None
    raw_digest: str
    companion_digest: str | None

    def __post_init__(self) -> None:
        """Freeze deep copies so later source mutation cannot reach this snapshot."""
        object.__setattr__(self, "raw_record", copy.deepcopy(dict(self.raw_record)))
        if self.companion is not None:
            object.__setattr__(self, "companion", copy.deepcopy(dict(self.companion)))


def capture_semantic_record(
    raw_record: Mapping[str, Any], companion: Mapping[str, Any] | None = None
) -> CapturedSemanticRecord:
    """Capture raw bytes and companion as one immutable snapshot.

    Parameters
    ----------
    raw_record
        Raw stable-core envelope to snapshot.
    companion
        Companion document to snapshot, or ``None``.

    Returns
    -------
    CapturedSemanticRecord
        Snapshot whose digests are fixed at capture time.

    """
    raw_snapshot = copy.deepcopy(dict(raw_record))
    companion_snapshot = copy.deepcopy(dict(companion)) if companion is not None else None
    return CapturedSemanticRecord(
        raw_record=raw_snapshot,
        companion=companion_snapshot,
        raw_digest=digest_stable_core_payload(raw_snapshot),
        companion_digest=(
            digest_stable_core_payload(companion_snapshot)
            if companion_snapshot is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class TransformDecision:
    """Outcome of a requested semantic transform.

    Attributes
    ----------
    decision
        ``refuse_unsupported_conversion`` until a separate transform owner is
        implemented and verified.
    executed
        Always ``False`` for a refusal; no value is ever converted in place.
    converted_value
        Always ``None``; this metadata reader never converts a value.
    persist_qualified_record
        Whether a qualified record may be persisted after this request.
    detail
        Why the request was accepted or refused.

    """

    decision: Literal["refuse_unsupported_conversion"]
    executed: bool
    converted_value: object
    persist_qualified_record: bool
    detail: str


def apply_semantic_transform(
    companion: Mapping[str, Any], transform_request: Mapping[str, Any]
) -> TransformDecision:
    """Refuse conversions until a separate owner can verify their authority.

    A unit relation is never inferred from labels. A reference listed only in
    the companion is self-asserted metadata, not proof that an actual converter
    accepted the transform. No converter is enrolled in this reader.

    Parameters
    ----------
    companion
        Companion document carrying ``supported_transform_composition``.
    transform_request
        Request naming ``field_path``, ``operation`` and ``accepted_transform_ref``.

    Returns
    -------
    TransformDecision
        Refused transform outcome, with the original value untouched.

    """
    supported = companion.get("supported_transform_composition")
    supported_refs = list(supported) if isinstance(supported, Sequence) else []
    accepted_ref = transform_request.get("accepted_transform_ref")
    field_path = transform_request.get("field_path")
    operation = transform_request.get("operation")

    if accepted_ref is None or accepted_ref not in supported_refs:
        reason = (
            f"{accepted_ref!r}, which is not in the companion's supported "
            f"composition {supported_refs!r}"
        )
    else:
        reason = (
            f"{accepted_ref!r}, listed by the companion but not verified by "
            "an independent transform owner"
        )
    return TransformDecision(
        decision="refuse_unsupported_conversion",
        executed=False,
        converted_value=None,
        persist_qualified_record=False,
        detail=(
            f"{operation!r} on {field_path!r} names accepted transform {reason}; "
            "the source value is unchanged"
        ),
    )


@dataclass(frozen=True, slots=True)
class AggregationDecision:
    """Outcome of a fidelity-component aggregation request.

    Attributes
    ----------
    decision
        ``accept_explicit_fixture_components`` or
        ``refuse_unjustified_error_aggregation``.
    executed
        Always ``False``; this reader computes no aggregate.
    aggregate_value
        Always ``None``; components are preserved, never summed here.
    preserve_components_separately
        Always ``True``; separate estimands remain separate.
    detail
        Why aggregation was refused, or why components stand as supplied.

    """

    decision: Literal["accept_explicit_fixture_components", "refuse_unjustified_error_aggregation"]
    executed: bool
    aggregate_value: None
    preserve_components_separately: bool
    detail: str


def aggregate_fidelity_components(
    fidelity_components: Sequence[Mapping[str, Any]],
    aggregation_request: Mapping[str, Any] | None = None,
) -> AggregationDecision:
    """Refuse to combine uncertainty components without a recorded justification.

    A standard error and a confidence radius derived from the same covariance
    are two descriptions of one uncertainty, not two independent errors, so
    summing them overstates it. Without an explicit justification the request
    is refused and the components are preserved separately.

    Parameters
    ----------
    fidelity_components
        Components as declared on the companion.
    aggregation_request
        Request naming ``components``, ``operation`` and ``justification``,
        or ``None`` when no aggregation is requested.

    Returns
    -------
    AggregationDecision
        Accepted custody, or refusal with the components left separate.

    """
    if aggregation_request is None:
        return AggregationDecision(
            decision="accept_explicit_fixture_components",
            executed=False,
            aggregate_value=None,
            preserve_components_separately=True,
            detail=(
                f"{len(fidelity_components)} component(s) are retained exactly as "
                "declared, with their own estimands, methods and evidence references"
            ),
        )

    justification = aggregation_request.get("justification")
    operation = aggregation_request.get("operation")
    requested = aggregation_request.get("components")
    return AggregationDecision(
        decision="refuse_unjustified_error_aggregation",
        executed=False,
        aggregate_value=None,
        preserve_components_separately=True,
        detail=(
            f"{operation!r} over {list(requested) if requested else []!r} carries "
            f"justification {justification!r}; combining components that describe the "
            "same covariance needs an explicit recorded justification"
        ),
    )


@dataclass(frozen=True, slots=True)
class ModalityQualification:
    """Outcome of qualifying a requested quantity against a native result.

    Attributes
    ----------
    qualification
        ``qualified`` or ``unavailable``.
    executed
        Always ``False``; no backend is invoked to fill a gap.
    padding_or_conversion_performed
        Always ``False``; absent data is never padded or inferred.
    requested_quantity
        The quantity the caller asked to qualify.
    reason
        Why the quantity is available or unavailable.

    """

    qualification: Literal["qualified", "unavailable"]
    executed: bool
    padding_or_conversion_performed: bool
    requested_quantity: str
    reason: str


def qualify_native_modality(
    result: Mapping[str, Any], requested_quantity: str, *, profile: Mapping[str, Any] | None = None
) -> ModalityQualification:
    """Qualify a requested quantity only if the native result actually carries it.

    A backend profile advertising a capability is a declaration about the
    backend, not evidence about this result. When the adapter returned counts
    only, amplitudes cannot be inferred from them, and a declared capability
    does not supply the missing data.

    Parameters
    ----------
    result
        Native adapter result payload.
    requested_quantity
        Quantity to qualify, such as ``statevector_amplitudes``.
    profile
        Backend profile, used only to explain a contradiction, never to supply data.

    Returns
    -------
    ModalityQualification
        Qualified state, or an explicit unavailable reason.

    """
    if requested_quantity in result and result[requested_quantity] is not None:
        return ModalityQualification(
            qualification="qualified",
            executed=False,
            padding_or_conversion_performed=False,
            requested_quantity=requested_quantity,
            reason=f"the native result carries {requested_quantity!r} directly",
        )

    present = sorted(key for key, value in result.items() if value is not None)
    capability = None
    if profile is not None:
        capabilities = profile.get("capabilities")
        if isinstance(capabilities, Mapping):
            capability = capabilities.get(f"supports_{requested_quantity.split('_')[0]}")
    declared = (
        " the profile declares support, but a capability declaration is not this result's data;"
        if capability
        else ""
    )
    return ModalityQualification(
        qualification="unavailable",
        executed=False,
        padding_or_conversion_performed=False,
        requested_quantity=requested_quantity,
        reason=(
            f"the native result carries {present!r} and no {requested_quantity!r};"
            f"{declared} amplitudes cannot be inferred from counts and are not padded"
        ),
    )
