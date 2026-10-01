# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — portable settings and reset preview
"""Carry requested settings without importing policy authority or changing saved state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .contracts import ResolvedSettings
from .json_transport import read_json, write_json
from .settings import SettingsPolicy, SettingsRefused, resolve_settings, validate_setting_values

if TYPE_CHECKING:
    from ..hardware.hal import BackendProfile

SETTINGS_BYTE_LIMIT = 65536
"""Maximum portable settings bytes, a product import bound rather than host capacity."""


def export_settings(settings: ResolvedSettings) -> str:
    """Export supported requested values without policy or credential authority.

    Parameters
    ----------
    settings
        Current resolved immutable document.

    Returns
    -------
    str
        Version-one portable JSON; no policy, environment or provider credentials.

    """
    values = settings.body["requested"]
    assert isinstance(values, Mapping)
    validate_setting_values(values)
    wire = write_json({"schema": "quantum_workspace_settings.v1", "values": values})
    if len(wire.encode("utf-8")) > SETTINGS_BYTE_LIMIT:
        raise SettingsRefused("Portable settings byte limit exceeded.")
    return wire


def import_settings(wire: str) -> Mapping[str, object]:
    """Preview imported requested settings without applying them or importing policy.

    Parameters
    ----------
    wire
        Bounded, lossless JSON settings text.

    Returns
    -------
    Mapping[str, object]
        Detached values requiring fresh source-owned resolution before use.

    Raises
    ------
    SettingsRefused
        If the archive, version or non-secret settings shape is unsupported.

    """
    try:
        byte_length = len(wire.encode("utf-8"))
    except UnicodeError as exc:
        raise SettingsRefused("Malformed portable settings Unicode.") from exc
    if byte_length > SETTINGS_BYTE_LIMIT:
        raise SettingsRefused("Portable settings byte limit exceeded.")
    try:
        payload = read_json(wire)
    except ValueError as exc:
        raise SettingsRefused("Malformed portable settings JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "values"}:
        raise SettingsRefused("Portable settings require schema and values only.")
    if payload["schema"] != "quantum_workspace_settings.v1":
        raise SettingsRefused("Unsupported settings version; explicit migration is required.")
    values = payload["values"]
    if not isinstance(values, dict):
        raise SettingsRefused("Portable settings values must be an object.")
    result = cast(dict[str, object], values)
    validate_setting_values(result)
    return result


@dataclass(frozen=True)
class SettingsResetPreview:
    """A reset candidate bound to the exact immutable source it replaces."""

    source_digest: str
    candidate: ResolvedSettings


def preview_settings_reset(
    current: ResolvedSettings,
    defaults: Mapping[str, object],
    *,
    policy: SettingsPolicy,
    environment_ref: Mapping[str, object],
    profile: BackendProfile,
) -> SettingsResetPreview:
    """Resolve defaults into a candidate without mutating the current record.

    Parameters
    ----------
    current
        Current snapshot whose full identity guards confirmation.
    defaults
        Source-owned default settings to preview.
    policy, environment_ref, profile
        Current trusted policy, environment and HAL declaration.

    Returns
    -------
    SettingsResetPreview
        Original digest and independently validated replacement candidate.

    """
    return SettingsResetPreview(
        current.digest,
        resolve_settings(
            defaults,
            {},
            {},
            {},
            policy=policy,
            environment_ref=environment_ref,
            profile=profile,
        ),
    )


def confirm_settings_reset(
    current: ResolvedSettings, preview: SettingsResetPreview
) -> ResolvedSettings:
    """Return the reset candidate only when the current source still matches.

    Parameters
    ----------
    current
        Actual current immutable settings, re-read at confirmation.
    preview
        Previously inspected reset candidate.

    Returns
    -------
    ResolvedSettings
        Candidate to apply explicitly; no storage mutation occurs in this function.

    Raises
    ------
    SettingsRefused
        If the current settings changed after preview.

    """
    if current.digest != preview.source_digest:
        raise SettingsRefused("Settings changed after reset preview; inspect a new preview.")
    return preview.candidate
