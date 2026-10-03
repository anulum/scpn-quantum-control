# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dated no-submit backend profile projection
"""Project existing route and HAL metadata without constructing an SDK adapter.

Supplied observations are producer declarations, never live discovery. Only
named fields are exported; arbitrary SDK metadata and credential values are
excluded. Profile-bound references are metadata and grant no execution authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from ..canonical_encoding import canonical_digest
from .hal import BackendProfile, built_in_backend_profiles
from .provider_capability_core import (
    ProviderCapabilitySnapshot,
    ProviderRouteCatalogueEntry,
    build_provider_route_catalogue,
)

BACKEND_PROFILES_SCHEMA = "studio.backend-profiles.v1"
"""Version of the dated offline metadata envelope."""
BACKEND_PROFILE_SCHEMA = "studio.backend-profile.v1"
"""Domain binding one route, physical device and its dated metadata."""
MAX_PROFILE_BYTES = 1024 * 1024
"""UTF-8 import/export ceiling, independent of measured host memory."""
MAX_PROFILES = 256
"""Maximum distinct route profiles in one offline envelope."""


def _digest(value: str) -> str:
    """Admit an opaque digest reference without accepting a credential value.

    Parameters
    ----------
    value
        Candidate immutable metadata reference.

    Returns
    -------
    str
        Unchanged lowercase SHA-256 digest.

    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("profile binding requires a lowercase SHA-256 reference")
    return value


@dataclass(frozen=True, slots=True)
class ProfileBinding:
    """Dependent references attached to one exact profile, without authority.

    Parameters
    ----------
    profile_sha256
        Exact dated profile digest, including route and observed metadata.
    plan_ref, calibration_ref, approval_ref
        Optional immutable metadata digests. Presence never approves execution.
        Selection of another profile must clear all three references.

    """

    profile_sha256: str
    plan_ref: str | None = None
    calibration_ref: str | None = None
    approval_ref: str | None = None

    def __post_init__(self) -> None:
        """Refuse credentials, URLs and malformed metadata reference values."""
        _digest(self.profile_sha256)
        for value in (self.plan_ref, self.calibration_ref, self.approval_ref):
            if value is not None:
                _digest(value)


def _date(value: str) -> str:
    """Admit exact Gregorian day precision without consulting a host clock.

    Parameters
    ----------
    value
        Explicit source observation date.

    Returns
    -------
    str
        Unchanged canonical capture day.

    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("profile observation requires a YYYY-MM-DD date")
    date.fromisoformat(value)
    return value


def _text(value: str) -> str:
    """Keep bounded scalar metadata without controls or whitespace coercion.

    Parameters
    ----------
    value
        Source-owned metadata text.

    Returns
    -------
    str
        Unchanged admissible Unicode text.

    """
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
    ):
        raise ValueError("profile metadata requires bounded nonempty scalar text")
    value.encode("utf-8")
    return value


def _strings(values: Sequence[str]) -> list[str]:
    """Preserve bounded unique source arrays without inferring unsupported values.

    Parameters
    ----------
    values
        Original ordered metadata names.

    Returns
    -------
    list
        Detached exact scalar names; an empty supplied list remains empty.

    """
    result = [_text(value) for value in values]
    if len(result) > 256 or len(set(result)) != len(result):
        raise ValueError("profile metadata array exceeds its ceiling or repeats a name")
    return result


def _count(value: int | None, *, positive: bool = False) -> int | None:
    """Preserve unknown limits and exact uint64 observations without booleans.

    Parameters
    ----------
    value
        Original integer observation or unknown.
    positive
        Whether this original field excludes zero.

    Returns
    -------
    int or None
        Unchanged admissible count or unknown.

    """
    if value is not None and (
        type(value) is not int or value < int(positive) or value > 2**64 - 1
    ):
        raise ValueError("profile observation requires an unsigned bounded integer")
    return value


def _observation(
    row: ProviderRouteCatalogueEntry, snapshot: ProviderCapabilitySnapshot | None
) -> tuple[str, dict[str, object]]:
    """Project whitelisted observation fields for one exact route.

    Parameters
    ----------
    row
        Source-owned dated route.
    snapshot
        Already acquired no-submit observations, or absent.

    Returns
    -------
    tuple
        Exact physical target and detached named observations.

    """
    if snapshot is None:
        return row.device, dict.fromkeys(
            (
                "online",
                "n_qubits",
                "max_shots",
                "max_circuits",
                "queue_depth",
                "calibration_timestamp",
                "calibration_ref",
                "ir_formats",
                "basis_gates",
                "native_features",
            )
        )
    if (
        snapshot.route_id != row.route_id
        or snapshot.aggregator != (row.broker or "direct")
        or snapshot.provider != row.provider
        or snapshot.backend_id != row.device
        or snapshot.no_submit is not True
    ):
        raise ValueError("profile observation belongs to another route")
    if snapshot.online is not None and type(snapshot.online) is not bool:
        raise ValueError("profile online observation requires bool or None")
    timestamp = snapshot.calibration_timestamp
    calibration_ref = None
    if timestamp is not None:
        if (
            not isinstance(timestamp, str)
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z",
                timestamp,
            )
            is None
        ):
            raise ValueError("profile calibration requires an exact UTC timestamp")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        calibration_ref = canonical_digest(
            "studio.calibration-reference.v1",
            {"route_key": row.route_key, "device": snapshot.target_name, "timestamp": timestamp},
        )
    return _text(snapshot.target_name), {
        "online": snapshot.online,
        "ir_formats": _strings(snapshot.supported_ir_formats),
        "basis_gates": _strings(snapshot.basis_gates),
        "native_features": _strings(snapshot.native_features),
        "n_qubits": _count(snapshot.n_qubits, positive=True),
        "max_shots": _count(snapshot.max_shots, positive=True),
        "max_circuits": _count(snapshot.max_circuits, positive=True),
        "queue_depth": _count(snapshot.queue_depth),
        "calibration_timestamp": timestamp,
        "calibration_ref": calibration_ref,
    }


def _profile(
    row: ProviderRouteCatalogueEntry,
    profile: BackendProfile,
    snapshot: ProviderCapabilitySnapshot | None,
) -> dict[str, Any]:
    """Bind original declarations, observations and opaque configuration references.

    Parameters
    ----------
    row
        Exact dated route, including broker/alias.
    profile
        Matching HAL declarations.
    snapshot
        Already supplied observations; no SDK is invoked.

    Returns
    -------
    dict
        Detached profile body and its original canonical digest.

    """
    device, observed = _observation(row, snapshot)
    capabilities = asdict(profile.capabilities)
    for name, value in capabilities.items():
        if name == "max_qubits":
            _count(value, positive=True)
        elif type(value) is not bool:
            raise ValueError("HAL capability declarations require booleans")
    options = {
        name: {
            "supported": capabilities["supports_" + name],
            "reason": (
                "Declared by HAL; runtime support remains unverified"
                if capabilities["supports_" + name]
                else "Unsupported by this HAL profile"
            ),
        }
        for name in ("pulse", "analog")
    }
    body = {
        "route_id": _text(row.route_id),
        "provider": _text(row.provider),
        "broker": _text(row.broker) if row.broker is not None else None,
        "device": _text(device),
        "backend_id": _text(row.device),
        "modality": _text(row.modality),
        "observed_at": _date(row.observed_at),
        "region": _text(profile.region) if profile.region is not None else None,
        "sdk_package": _text(profile.sdk_package),
        "ir_formats": _strings(profile.ir_formats),
        "credential_refs": None
        if row.credential_configuration_refs is None
        else [
            "credential-ref:" + hashlib.sha256(_text(ref).encode("utf-8")).hexdigest()
            for ref in _strings(row.credential_configuration_refs)
        ],
        "declared": capabilities,
        "observed": observed,
        "options": options,
        "verbs": [record.to_dict() for record in row.verbs],
    }
    return {"body": body, "sha256": canonical_digest(BACKEND_PROFILE_SCHEMA, body)}


def build_backend_profiles(
    *,
    observed_at: str,
    catalogue: Sequence[ProviderRouteCatalogueEntry] | None = None,
    profiles: Sequence[BackendProfile] | None = None,
    snapshots: Mapping[str, ProviderCapabilitySnapshot] | None = None,
    binding: ProfileBinding | None = None,
) -> dict[str, Any]:
    """Project a bounded dated route catalogue for the offline Studio consumer.

    Parameters
    ----------
    observed_at
        Explicit Gregorian YYYY-MM-DD capture date; day precision is retained.
    catalogue
        Existing source-owned route rows, or the built-in catalogue. No SDK probes.
    profiles
        Existing HAL declarations matching the catalogue's backend identifiers.
    snapshots
        Optional already acquired observations keyed by exact route identifier.
        Arbitrary metadata is never exported; no callback or adapter is invoked.
    binding
        Optional references for one exact projected profile; a mismatch refuses
        the entire envelope. These references do not certify a plan or approval.

    Returns
    -------
    dict
        Detached digest-bound metadata, separate declarations/observations and
        source-owned disabled-option reasons. Null retains unknown values.

    Raises
    ------
    ValueError
        Date, size, route identity, declarations or binding cannot be admitted.

    """
    _date(observed_at)
    declarations = tuple(built_in_backend_profiles() if profiles is None else profiles)
    by_id = {profile.backend_id: profile for profile in declarations}
    if len(by_id) != len(declarations):
        raise ValueError("profile declarations repeat a backend identity")
    rows = tuple(
        build_provider_route_catalogue(observed_at=observed_at, profiles=declarations)
        if catalogue is None
        else catalogue
    )
    if not rows or len(rows) > MAX_PROFILES:
        raise ValueError("profile catalogue is empty or exceeds its ceiling")
    if len({row.route_id for row in rows}) != len(rows):
        raise ValueError("profile catalogue repeats a route identity")
    observations = dict(snapshots or {})
    if set(observations) - {row.route_id for row in rows}:
        raise ValueError("profile observations name an unknown route")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if row.observed_at != observed_at or row.device not in by_id:
            raise ValueError("profile date or HAL declaration differs")
        projected.append(_profile(row, by_id[row.device], observations.get(row.route_id)))
    if binding is not None and binding.profile_sha256 not in {row["sha256"] for row in projected}:
        raise ValueError("profile binding does not match this dated catalogue")
    envelope = {
        "schema": BACKEND_PROFILES_SCHEMA,
        "body": {
            "observed_at": observed_at,
            "no_submit": True,
            "profiles": projected,
            "binding": asdict(binding) if binding is not None else None,
        },
        "extensions": {},
    }
    result = {**envelope, "sha256": canonical_digest(BACKEND_PROFILES_SCHEMA, envelope)}
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_PROFILE_BYTES:
        raise ValueError("profile JSON exceeds its UTF-8 ceiling")
    return result


__all__ = [
    "BACKEND_PROFILES_SCHEMA",
    "BACKEND_PROFILE_SCHEMA",
    "MAX_PROFILE_BYTES",
    "MAX_PROFILES",
    "ProfileBinding",
    "build_backend_profiles",
]
