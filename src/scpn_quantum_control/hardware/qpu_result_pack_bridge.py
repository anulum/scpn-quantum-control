# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — provider-neutral QPU result-pack emission bridge
"""Emit a ``studio.qpu-result-pack.v1`` unit from a live provider-neutral result.

The hardware abstraction layer normalises every provider — all sixteen HAL
adapters (IBM, IonQ, IQM, Rigetti, Quantinuum, QuEra, Pasqal, OQC, D-Wave,
Braket, Azure, Cirq, PennyLane, qBraid, Quandela, Strangeworks) — onto a single
:class:`~scpn_quantum_control.hardware.hal.QuantumJobResult`. This module is the
one bridge from that neutral result to the WS-1 attestation-verifiable
``studio.qpu-result-pack.v1`` unit, so wiring it once wires *every* adapter's run
path at the same time — there is no per-provider copy of the emission logic to
drift.

The unit itself, its presentation verdict, and its seal live in
:mod:`scpn_quantum_control.studio.qpu_result_pack`. This bridge only supplies the
two things that module cannot synthesise from a pack record: the deterministic
digest of the *returned counts* (the value a provider attestation signs over) and
the run provenance drawn from the neutral result and its backend profile.

Honesty boundary. A result pack states a bounded observation and never
auto-validates. The claim scope, the human-readable title, and the explicit
non-claims cannot be inferred from a shot histogram, so this bridge *requires*
the caller to state them — it refuses to mint provenance with blank claim
scoping. Without a provider attestation the emitted unit renders ``unverifiable``
downstream, loud and never silently upgraded (see
:func:`~scpn_quantum_control.studio.qpu_result_pack.present_qpu_result_pack`).

The counts digest is byte-for-byte the algorithm the executive deploy template
emits for a provider to sign against
(:mod:`scpn_quantum_control.studio.executive_execute`): canonical JSON with
sorted keys and no whitespace, SHA-256, ``sha256:``-prefixed. A parity test locks
the two together so the on-device signature and the studio-side digest can never
diverge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from scpn_quantum_control.studio.qpu_result_pack import (
    DEFAULT_CLAIM_STATUS,
    DEFAULT_EVIDENCE_KIND,
    build_qpu_result_pack_unit,
)

from .hal import BackendProfile, QuantumJobResult

__all__ = [
    "job_result_provenance",
    "qpu_result_pack_from_job",
    "raw_results_digest",
]


def raw_results_digest(counts: Mapping[str, int]) -> str:
    """Return the ``sha256:`` digest a provider attestation signs over.

    The digest is taken over the canonical JSON encoding of the returned counts:
    keys sorted, no insignificant whitespace. This is byte-for-byte the algorithm
    the executive deploy template hands a provider to sign against, so the
    on-device signature and this studio-side digest agree exactly.

    Parameters
    ----------
    counts
        The returned shot histogram ``{bitstring: count}``. Must be non-empty —
        a result pack over zero counts carries no observation.

    Returns
    -------
    str
        The ``"sha256:<hex>"`` digest of the canonical-JSON counts.

    Raises
    ------
    ValueError
        If ``counts`` is empty.
    """
    if not counts:
        raise ValueError("counts must be non-empty to digest a QPU result")
    payload = json.dumps(dict(counts), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def job_result_provenance(
    result: QuantumJobResult,
    *,
    profile: BackendProfile,
    title: str,
    executed_utc: str,
    claim_scope: str,
    non_claims: Sequence[str],
    pack_id: str | None = None,
    backend: str | None = None,
    hardware_family: str | None = None,
) -> dict[str, Any]:
    """Build the result-pack provenance record from a neutral job result.

    The provenance carries exactly the fields the studio unit reads
    (``id``/``title``/``backend``/``hardware_family``/``executed_utc``/
    ``required_job_ids``/``claim_scope``/``non_claims``). The identifiers are
    drawn from the neutral result where they exist and may be overridden; the
    claim scoping is caller-supplied because it cannot be inferred from counts.

    Parameters
    ----------
    result
        The completed provider-neutral result. Its ``job.job_id`` is recorded as
        the required job id and, by default, as the pack id.
    profile
        The backend route profile. Its ``target_family`` (falling back to
        ``modality``) names the hardware family, and its ``backend_id`` is the
        default concrete backend name.
    title
        A human-readable one-line title for the observation. Must be non-empty.
    executed_utc
        The UTC execution timestamp (e.g. ``"2026-07-11"``). Must be non-empty.
    claim_scope
        What the raw counts support — the bounded claim. Must be non-empty.
    non_claims
        What the pack explicitly does *not* claim. Must list at least one item —
        an honest pack always states its boundary.
    pack_id
        Override for the pack id. Defaults to ``result.job.job_id``.
    backend
        Override for the concrete backend/device name. Defaults to the neutral
        result's ``metadata["backend"]`` when present, else
        ``profile.backend_id``.
    hardware_family
        Override for the hardware family. Defaults to ``profile.target_family``
        or, when unset, ``profile.modality``.

    Returns
    -------
    dict[str, Any]
        The provenance record accepted by
        :func:`~scpn_quantum_control.studio.qpu_result_pack.build_qpu_result_pack_unit`.

    Raises
    ------
    ValueError
        If ``title``, ``executed_utc``, or ``claim_scope`` is blank, or if
        ``non_claims`` is empty.
    """
    if not title.strip():
        raise ValueError("title must be non-empty")
    if not executed_utc.strip():
        raise ValueError("executed_utc must be non-empty")
    if not claim_scope.strip():
        raise ValueError("claim_scope must be non-empty")
    resolved_non_claims = tuple(str(claim) for claim in non_claims)
    if not resolved_non_claims:
        raise ValueError("non_claims must state at least one boundary")

    metadata_backend = result.metadata.get("backend")
    resolved_backend = backend or (
        str(metadata_backend) if isinstance(metadata_backend, str) and metadata_backend else None
    )
    resolved_backend = resolved_backend or profile.backend_id
    resolved_family = hardware_family or profile.target_family or profile.modality
    resolved_id = pack_id or result.job.job_id

    return {
        "id": resolved_id,
        "title": title,
        "backend": resolved_backend,
        "hardware_family": resolved_family,
        "executed_utc": executed_utc,
        "required_job_ids": [result.job.job_id],
        "claim_scope": claim_scope,
        "non_claims": list(resolved_non_claims),
    }


def qpu_result_pack_from_job(
    result: QuantumJobResult,
    *,
    profile: BackendProfile,
    title: str,
    executed_utc: str,
    claim_scope: str,
    non_claims: Sequence[str],
    pack_id: str | None = None,
    backend: str | None = None,
    hardware_family: str | None = None,
    circuit_digest: str | None = None,
    calibration_ref: str | None = None,
    attestation: Mapping[str, str] | None = None,
    evidence_kind: str = DEFAULT_EVIDENCE_KIND,
    claim_status: str = DEFAULT_CLAIM_STATUS,
) -> dict[str, Any]:
    """Emit a ``studio.qpu-result-pack.v1`` unit from a live neutral job result.

    This is the end-to-end bridge: digest the returned counts, assemble the run
    provenance, and hand both to the studio unit builder. It applies to every HAL
    adapter without change because they all return a
    :class:`~scpn_quantum_control.hardware.hal.QuantumJobResult`.

    Parameters
    ----------
    result
        The completed provider-neutral result carrying the returned counts.
    profile
        The backend route profile the result was produced on.
    title, executed_utc, claim_scope, non_claims
        The caller-supplied attestation scope (see :func:`job_result_provenance`).
    pack_id, backend, hardware_family
        Optional identifier overrides (see :func:`job_result_provenance`).
    circuit_digest
        The ``"sha256:<hex>"`` digest of the compiled circuit, linking the result
        to a recompute-verifiable compile. ``None`` when unavailable.
    calibration_ref
        A reference to the device calibration snapshot for the run. ``None`` when
        unavailable.
    attestation
        The provider attestation (``provider``/``result_pack_digest``/
        ``provider_sig``) signing the counts digest. ``None`` for an offline or
        not-yet-attested run; the resulting unit renders ``unverifiable``.
    evidence_kind
        The honesty modality. Defaults to ``"measured"``.
    claim_status
        The honesty status. Defaults to ``"bounded-support"``.

    Returns
    -------
    dict[str, Any]
        The ``studio.qpu-result-pack.v1`` unit wire form.

    Raises
    ------
    ValueError
        If the returned counts are empty, the provenance claim scoping is blank,
        or a supplied ``attestation`` does not sign the computed counts digest.
    """
    digest = raw_results_digest(result.counts)
    provenance = job_result_provenance(
        result,
        profile=profile,
        title=title,
        executed_utc=executed_utc,
        claim_scope=claim_scope,
        non_claims=non_claims,
        pack_id=pack_id,
        backend=backend,
        hardware_family=hardware_family,
    )
    return build_qpu_result_pack_unit(
        provenance,
        raw_results_digest=digest,
        circuit_digest=circuit_digest,
        calibration_ref=calibration_ref,
        attestation=attestation,
        evidence_kind=evidence_kind,
        claim_status=claim_status,
    )
