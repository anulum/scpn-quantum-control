# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — WS-1 attestation-verifiable QPU result-pack unit
"""Emit and present the attestation-verifiable ``studio.qpu-result-pack.v1`` unit.

WS-1 grants two verification modes. Compile-path claims are **recompute**-
verifiable (:mod:`scpn_quantum_control.studio.recompute_kernel`): a visitor
replays the digest in the browser. A QPU result cannot be replayed — the shot
statistics are irreproducible — so it is **attestation**-verifiable: the trust
rests on a hardware provider's own signed record, not on a recompute.

This module emits the richer WS-1 unit that carries that axis explicitly. Every
unit declares ``verifiability_mode = attestation`` and binds four things a
verifier checks: the raw-results digest (the returned counts), the calibration
snapshot reference, the bit-exact circuit digest (the link back to a
recompute-verifiable compile), and — when a real device run exists — the
provider attestation.

The absent-signal is loud, never silently downgraded. A unit with no provider
attestation :func:`present_qpu_result_pack` renders ``unverifiable`` and
:func:`seal_qpu_result_pack` refuses to seal it — it is never emitted as
``verified`` on the studio signature alone. The committed hardware packs carry
no live provider attestation yet (that is BL-29 territory), so their units are
honestly ``unverifiable`` today; the shape and the fail-closed boundary are what
this slice lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scpn_studio_platform.seal import HonestyEnvelope, Signer, seal

from .verbs import QPU_RESULT_PACK_SCHEMA

QPU_VERIFIABILITY_MODE = "attestation"
"""A QPU result is attestation-verifiable: its trust rests on a provider signature."""

DEFAULT_EVIDENCE_KIND = "measured"
"""The honesty modality of a raw-count QPU observation: measured on the device."""

DEFAULT_CLAIM_STATUS = "bounded-support"
"""A result pack supports a bounded observation; it never auto reference-validates."""

_PROVENANCE_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "backend",
    "hardware_family",
    "executed_utc",
    "required_job_ids",
    "claim_scope",
    "non_claims",
)

_ATTESTATION_FIELDS: tuple[str, ...] = ("provider", "result_pack_digest", "provider_sig")

QpuPresentationStatus = Literal["attestation-verifiable", "unverifiable"]


@dataclass(frozen=True)
class QpuResultPackPresentation:
    """The honest display verdict for a QPU result-pack unit.

    Parameters
    ----------
    status
        ``"attestation-verifiable"`` only when a well-formed provider
        attestation binds the returned counts; ``"unverifiable"`` otherwise.
    reason
        Human-readable explanation, always present for ``"unverifiable"``.
    """

    status: QpuPresentationStatus
    reason: str

    def __post_init__(self) -> None:
        """Validate the presentation invariants."""
        if self.status not in ("attestation-verifiable", "unverifiable"):
            raise ValueError("qpu presentation status is unknown")
        if not self.reason.strip():
            raise ValueError("qpu presentation reason must be non-empty")


def build_qpu_result_pack_unit(
    pack: Mapping[str, Any],
    *,
    raw_results_digest: str,
    circuit_digest: str | None = None,
    calibration_ref: str | None = None,
    attestation: Mapping[str, str] | None = None,
    evidence_kind: str = DEFAULT_EVIDENCE_KIND,
    claim_status: str = DEFAULT_CLAIM_STATUS,
) -> dict[str, Any]:
    """Build a ``studio.qpu-result-pack.v1`` unit from a hardware pack record.

    Parameters
    ----------
    pack
        A pack record from ``data/hardware_result_packs/manifest.json`` (or an
        equivalent mapping). Its ``id`` is required.
    raw_results_digest
        The ``"sha256:<hex>"`` digest of the returned counts — the value the
        provider attestation signs over.
    circuit_digest
        The ``"sha256:<hex>"`` digest of the compiled circuit, linking the
        result to a recompute-verifiable compile. ``None`` when unavailable.
    calibration_ref
        A reference to the device calibration snapshot for the run. ``None``
        when unavailable.
    attestation
        The provider attestation (``provider``/``result_pack_digest``/
        ``provider_sig``). ``None`` for a committed pack with no live device
        run; the resulting unit is honestly ``unverifiable``.
    evidence_kind
        The honesty modality. Defaults to ``"measured"``.
    claim_status
        The honesty status. Defaults to ``"bounded-support"``.

    Returns
    -------
    dict[str, Any]
        The unit wire form, always carrying ``verifiability_mode = attestation``.

    Raises
    ------
    ValueError
        If the pack has no ``id``, ``raw_results_digest`` is empty, or a
        supplied ``attestation`` is malformed or does not sign the raw results.
    """
    pack_id = str(pack.get("id", "")).strip()
    if not pack_id:
        raise ValueError("pack must carry a non-empty id")
    if not raw_results_digest.strip():
        raise ValueError("raw_results_digest must be a non-empty digest")

    unit: dict[str, Any] = {
        "schema": QPU_RESULT_PACK_SCHEMA,
        "verifiability_mode": QPU_VERIFIABILITY_MODE,
        "evidence_kind": evidence_kind,
        "claim_status": claim_status,
        "raw_results_digest": raw_results_digest,
        "provenance": {field: pack[field] for field in _PROVENANCE_FIELDS if field in pack},
    }
    if circuit_digest is not None:
        unit["circuit_digest"] = circuit_digest
    if calibration_ref is not None:
        unit["calibration_ref"] = calibration_ref
    if attestation is not None:
        unit["attestation"] = _validated_attestation(attestation, raw_results_digest)
    return unit


def _validated_attestation(
    attestation: Mapping[str, str], raw_results_digest: str
) -> dict[str, str]:
    """Return a provider attestation that actually signs the raw results.

    Raises
    ------
    ValueError
        If a field is empty or the attestation signs a different digest than the
        unit's ``raw_results_digest`` — a mismatched attestation is not an
        attestation for this result.
    """
    for field in _ATTESTATION_FIELDS:
        value = attestation.get(field, "")
        if not value.strip():
            raise ValueError(f"provider attestation field {field!r} must be non-empty")
    if attestation["result_pack_digest"] != raw_results_digest:
        raise ValueError("provider attestation signs a different digest than the raw results")
    return {field: attestation[field] for field in _ATTESTATION_FIELDS}


def present_qpu_result_pack(unit: Mapping[str, Any]) -> QpuResultPackPresentation:
    """Return the honest display verdict for a QPU result-pack unit.

    Parameters
    ----------
    unit
        A unit built by :func:`build_qpu_result_pack_unit` (or its wire form).

    Returns
    -------
    QpuResultPackPresentation
        ``"attestation-verifiable"`` only when a well-formed provider
        attestation binds the returned counts; ``"unverifiable"`` otherwise,
        with the reason spelled out.
    """
    if unit.get("schema") != QPU_RESULT_PACK_SCHEMA:
        return QpuResultPackPresentation("unverifiable", "unknown qpu result-pack schema")
    if unit.get("verifiability_mode") != QPU_VERIFIABILITY_MODE:
        return QpuResultPackPresentation("unverifiable", "verifiability mode is not attestation")
    raw_results_digest = str(unit.get("raw_results_digest", "")).strip()
    if not raw_results_digest:
        return QpuResultPackPresentation("unverifiable", "unit carries no raw-results digest")
    attestation = unit.get("attestation")
    if not isinstance(attestation, Mapping) or not attestation:
        return QpuResultPackPresentation(
            "unverifiable",
            "no provider attestation stands behind the returned counts",
        )
    for field in _ATTESTATION_FIELDS:
        if not str(attestation.get(field, "")).strip():
            return QpuResultPackPresentation(
                "unverifiable", f"provider attestation is missing {field!r}"
            )
    if attestation.get("result_pack_digest") != raw_results_digest:
        return QpuResultPackPresentation(
            "unverifiable", "provider attestation signs a different digest than the raw results"
        )
    return QpuResultPackPresentation(
        "attestation-verifiable",
        f"provider {attestation['provider']!r} attests the returned counts",
    )


def seal_qpu_result_pack(
    unit: Mapping[str, Any],
    *,
    signer: Signer,
    grader: Mapping[str, str],
    exactness_class: str | Mapping[str, Any] = "bit-exact",
) -> HonestyEnvelope:
    """Seal a QPU result-pack unit as an attestation-verifiable envelope.

    Parameters
    ----------
    unit
        A ``studio.qpu-result-pack.v1`` unit carrying a provider attestation.
    signer
        QUANTUM's signer (the post-quantum ML-DSA signer in production).
    grader
        ``{"name", "version"}`` of the grading code for this unit.
    exactness_class
        How a recompute would match. Defaults to ``"bit-exact"`` — the
        *circuit* is bit-exact-derivable even though the *counts* are not.

    Returns
    -------
    HonestyEnvelope
        The sealed, attestation-verifiable envelope.

    Raises
    ------
    ValueError
        If the unit renders ``unverifiable`` (no well-formed provider
        attestation) — an unverifiable unit is never sealed as verified.
    """
    presentation = present_qpu_result_pack(unit)
    if presentation.status != "attestation-verifiable":
        raise ValueError(
            "qpu result pack is unverifiable and is never sealed as verified: "
            + presentation.reason
        )
    attestation = dict(unit["attestation"])
    return seal(
        dict(unit),
        signer=signer,
        grader=dict(grader),
        verifiability_mode=QPU_VERIFIABILITY_MODE,
        exactness_class=exactness_class,
        attestation=attestation,
    )


__all__ = [
    "DEFAULT_CLAIM_STATUS",
    "DEFAULT_EVIDENCE_KIND",
    "QPU_VERIFIABILITY_MODE",
    "QpuResultPackPresentation",
    "build_qpu_result_pack_unit",
    "present_qpu_result_pack",
    "seal_qpu_result_pack",
]
