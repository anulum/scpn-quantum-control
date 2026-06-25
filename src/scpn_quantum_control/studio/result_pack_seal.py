# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — WS-1 attestation-mode sealing for hardware result packs
"""Seal a QUANTUM hardware result pack into a verifiable honesty envelope (WS-1).

A QPU result cannot be re-run in a verifier's browser — the shot statistics are
irreproducible — so WS-1 grants it **attestation-verifiable** trust rather than
**recompute-verifiable** trust (the platform's Mode B). The chain of custody is:

1. the studio publishes a ``studio.hardware-result-pack.v1`` claim unit carrying the
   pack's provenance (backend, job IDs, the digest of the returned counts, the
   bit-exact digest of the compiled circuit) and its honesty axes;
2. a **provider attestation** — the hardware provider's own signed record of the job
   result — binds those counts to that device run;
3. QUANTUM seals the unit with its post-quantum key
   (:class:`~scpn_quantum_control.crypto.ml_dsa_seal.MLDSASigner`) through the
   platform :func:`~scpn_studio_platform.seal.seal`, in ``attestation`` mode.

The result is a :class:`~scpn_studio_platform.seal.HonestyEnvelope` any keyring
holder verifies: the studio signature proves the unit is QUANTUM's own and ungraded
upward, and the provider attestation proves the counts came from that device — but it
makes **no** reproducibility claim, which is the honest boundary of hardware
evidence.

This module never fabricates the provider attestation: it is a required input. A
pack with no provider attestation cannot be sealed as attestation-verifiable —
:func:`seal_result_pack` raises rather than emit an envelope that would render as
``verified`` without the provider's own signature behind it (absent-signal is loud,
not silently downgraded).

The module binds to the **existing** ``studio.hardware-result-pack.v1`` evidence
schema (the ``execute``/``replay`` verbs already advertise it); it introduces no new
schema name. It lives on the studio federation surface and so depends on the
platform SDK (the ``studio`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scpn_studio_platform.seal import HonestyEnvelope, Signer, seal

from .verbs import HARDWARE_RESULT_PACK_SCHEMA

DEFAULT_EVIDENCE_KIND = "measured"
"""The honesty modality of a raw-count QPU observation: measured on the device."""

DEFAULT_CLAIM_STATUS = "bounded-support"
"""The default claim status: a result pack supports a bounded observation, it does
not reference-validate one — promotion to a stronger status needs explicit evidence."""

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


def build_result_pack_unit(
    pack: Mapping[str, Any],
    *,
    raw_results_digest: str,
    circuit_digest: str | None = None,
    evidence_kind: str = DEFAULT_EVIDENCE_KIND,
    claim_status: str = DEFAULT_CLAIM_STATUS,
) -> dict[str, Any]:
    """Build a ``studio.hardware-result-pack.v1`` claim unit from a pack record.

    The unit carries the pack's provenance verbatim (so the sealed digest addresses
    the real declared scope and its ``non_claims`` boundary) plus the two digests a
    verifier checks: ``raw_results_digest`` (the returned counts) and, when present,
    ``circuit_digest`` (the bit-exact link back to the recompute-verifiable compiled
    circuit).

    Parameters
    ----------
    pack
        A pack record from ``data/hardware_result_packs/manifest.json`` (or an
        equivalent mapping). Its ``id`` is required.
    raw_results_digest
        The ``"sha256:<hex>"`` digest of the returned counts — the same value the
        provider attestation signs over.
    circuit_digest
        The ``"sha256:<hex>"`` digest of the compiled circuit, linking the hardware
        result to an independently re-derivable circuit. ``None`` when unavailable.
    evidence_kind
        The honesty modality. Defaults to ``"measured"`` (a device observation).
    claim_status
        The honesty status. Defaults to ``"bounded-support"`` — a result pack
        supports a bounded observation; it is never auto reference-validated.

    Returns
    -------
    dict[str, Any]
        The claim-unit wire form, ready to seal.

    Raises
    ------
    ValueError
        If the pack has no ``id`` or ``raw_results_digest`` is empty.
    """
    pack_id = str(pack.get("id", "")).strip()
    if not pack_id:
        raise ValueError("pack must carry a non-empty id")
    if not raw_results_digest.strip():
        raise ValueError("raw_results_digest must be a non-empty digest")

    unit: dict[str, Any] = {
        "schema": HARDWARE_RESULT_PACK_SCHEMA,
        "evidence_kind": evidence_kind,
        "claim_status": claim_status,
        "raw_results_digest": raw_results_digest,
        "provenance": {field: pack[field] for field in _PROVENANCE_FIELDS if field in pack},
    }
    if circuit_digest is not None:
        unit["circuit_digest"] = circuit_digest
    return unit


def build_provider_attestation(
    *, provider: str, result_pack_digest: str, provider_sig: str
) -> dict[str, str]:
    """Build the provider-attestation reference for an attestation-mode envelope.

    This is the hardware provider's own signed record — the load-bearing object of
    attestation-verifiable trust. QUANTUM does not produce it; it carries it.

    Parameters
    ----------
    provider
        The hardware provider identifier (e.g. ``"ibm"``).
    result_pack_digest
        The provider's ``"sha256:<hex>"`` over the raw results; a verifier checks it
        equals the unit's ``raw_results_digest``.
    provider_sig
        The provider's signature over its result record, verifiable against the
        provider's published key.

    Returns
    -------
    dict[str, str]
        The ``{"provider", "result_pack_digest", "provider_sig"}`` mapping the
        envelope's ``attestation`` field expects.

    Raises
    ------
    ValueError
        If any field is empty — an attestation with a missing part is not an
        attestation, and must not masquerade as one.
    """
    for name, value in (
        ("provider", provider),
        ("result_pack_digest", result_pack_digest),
        ("provider_sig", provider_sig),
    ):
        if not value.strip():
            raise ValueError(f"provider attestation field {name!r} must be non-empty")
    return {
        "provider": provider,
        "result_pack_digest": result_pack_digest,
        "provider_sig": provider_sig,
    }


def seal_result_pack(
    unit: Mapping[str, Any],
    *,
    signer: Signer,
    attestation: Mapping[str, str],
    grader: Mapping[str, str],
    exactness_class: str | Mapping[str, Any] = "bit-exact",
) -> HonestyEnvelope:
    """Seal a hardware result-pack unit as an attestation-verifiable envelope.

    Parameters
    ----------
    unit
        The ``studio.hardware-result-pack.v1`` claim unit (see
        :func:`build_result_pack_unit`).
    signer
        QUANTUM's signer — the post-quantum
        :class:`~scpn_quantum_control.crypto.ml_dsa_seal.MLDSASigner` in production.
    attestation
        The provider attestation (see :func:`build_provider_attestation`). Required:
        an attestation-mode seal without a provider attestation would render as
        ``verified`` on the studio signature alone, hiding that no provider stands
        behind the counts. Refused.
    grader
        ``{"name", "version"}`` of the grading code that applies to this unit.
    exactness_class
        How a recompute would match. Defaults to ``"bit-exact"`` — the *circuit* is
        bit-exact-derivable even though the *counts* are not reproducible; the
        attestation, not a recompute, carries the counts' trust.

    Returns
    -------
    HonestyEnvelope
        The sealed, attestation-verifiable envelope.

    Raises
    ------
    ValueError
        If ``attestation`` is empty (no provider attestation to stand behind the
        counts).
    """
    if not attestation:
        raise ValueError(
            "attestation-mode sealing requires a provider attestation; a QPU result "
            "pack with none renders unverifiable, it is never sealed as verified"
        )
    return seal(
        dict(unit),
        signer=signer,
        grader=grader,
        verifiability_mode="attestation",
        exactness_class=exactness_class,
        attestation=dict(attestation),
    )


__all__ = [
    "DEFAULT_CLAIM_STATUS",
    "DEFAULT_EVIDENCE_KIND",
    "build_provider_attestation",
    "build_result_pack_unit",
    "seal_result_pack",
]
