# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping static analog platform profiles
"""Load and validate the packaged static analog platform catalogue."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, cast

from .contracts import AnalogPlatformProfile

PLATFORM_CATALOGUE_SCHEMA = "analog_platform_profiles.v1"


@lru_cache(maxsize=1)
def load_platform_profiles() -> tuple[AnalogPlatformProfile, ...]:
    """Return the validated, immutable profile catalogue."""
    resource = files(__package__).joinpath("platform_profiles.v1.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PLATFORM_CATALOGUE_SCHEMA:
        raise ValueError("analog platform catalogue has an unknown schema")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("analog platform catalogue must contain profiles")
    profiles = tuple(_profile_from_dict(item) for item in raw_profiles)
    profile_ids = [profile.profile_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("analog platform catalogue profile ids must be unique")
    return profiles


def platform_profile(profile_id: str) -> AnalogPlatformProfile:
    """Resolve one profile by stable id.

    Raises
    ------
    KeyError
        If no packaged profile has the requested id.

    """
    for profile in load_platform_profiles():
        if profile.profile_id == profile_id:
            return profile
    known = ", ".join(profile.profile_id for profile in load_platform_profiles())
    raise KeyError(f"unknown analog platform profile {profile_id!r}; known: {known}")


def _profile_from_dict(payload: object) -> AnalogPlatformProfile:
    if not isinstance(payload, dict):
        raise ValueError("analog platform profile rows must be objects")
    row = cast(dict[str, Any], payload)
    try:
        return AnalogPlatformProfile(
            profile_id=str(row["profile_id"]),
            display_name=str(row["display_name"]),
            platform_family=str(row["platform_family"]),
            posture=row["posture"],
            supported_topologies=tuple(row["supported_topologies"]),
            max_nodes=row["max_nodes"],
            coupling_abs_min=float(row["coupling_abs_min"]),
            coupling_abs_max=(
                None if row["coupling_abs_max"] is None else float(row["coupling_abs_max"])
            ),
            supports_signed_couplings=bool(row["supports_signed_couplings"]),
            supports_local_detuning=bool(row["supports_local_detuning"]),
            supported_measurements=tuple(row["supported_measurements"]),
            control_model=str(row["control_model"]),
            compiler_platform=(
                None if row["compiler_platform"] is None else str(row["compiler_platform"])
            ),
            arbitrary_pairwise_control_verified=bool(row["arbitrary_pairwise_control_verified"]),
            source_url=str(row["source_url"]),
            verified_at_source_utc=str(row["verified_at_source_utc"]),
            ledger_ref=str(row["ledger_ref"]),
            limitations=tuple(str(item) for item in row["limitations"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("analog platform profile row is malformed") from exc


__all__ = ["PLATFORM_CATALOGUE_SCHEMA", "load_platform_profiles", "platform_profile"]
