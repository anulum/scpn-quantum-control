# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workspace settings resolution
"""Resolve explicit settings layers against an immutable, source-owned policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .canonical import canonical_digest
from .contracts import ResolvedSettings

if TYPE_CHECKING:
    from ..hardware.hal import BackendProfile

SEMANTIC_FIELDS = frozenset(
    {
        "method",
        "backend",
        "device",
        "precision",
        "seed",
        "shots",
        "parameters",
        "units",
        "memory_budget_bytes",
        "n_qubits",
        "concurrency",
        "time_limit_ms",
        "region",
        "unattended",
    }
)
DISPLAY_FIELDS = frozenset({"theme", "notation", "plot_rounding", "layout"})


class SettingsRefused(ValueError):
    """An authored settings refusal suitable for a caller-facing inspector."""


@dataclass(frozen=True)
class SettingsPolicy:
    """Explicit ceilings and permitted values, bound to the governing source.

    Parameters
    ----------
    reference
        Original policy schema and lowercase SHA-256 identity.
    ceilings
        Positive integer caps; absent resource capacity cannot authorize a request.
    choices
        Permitted exact string values for enumerated settings.

    """

    reference: Mapping[str, object]
    ceilings: Mapping[str, int]
    choices: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Validate and snapshot the policy without retaining caller-owned maps."""
        probe = ResolvedSettings(
            {
                "requested": {},
                "effective": {},
                "origins": {},
                "policy_ref": self.reference,
                "environment_ref": self.reference,
                "rejected_fields": [],
            }
        )
        for key, value in self.ceilings.items():
            if key not in {
                "shots",
                "memory_budget_bytes",
                "n_qubits",
                "concurrency",
                "time_limit_ms",
            }:
                raise SettingsRefused("Unsupported policy ceiling field.")
            if type(value) is not int or value <= 0:
                raise SettingsRefused("Policy ceilings must be positive integers.")
        for key, values in self.choices.items():
            if key not in {"method", "backend", "device", "precision", "units", "region"}:
                raise SettingsRefused("Unsupported policy choice field.")
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise SettingsRefused("Policy choices must contain nonempty strings.")
        object.__setattr__(self, "reference", probe.body["policy_ref"])
        object.__setattr__(self, "ceilings", MappingProxyType(dict(self.ceilings)))
        object.__setattr__(
            self, "choices", MappingProxyType({k: tuple(v) for k, v in self.choices.items()})
        )


def validate_setting_values(values: Mapping[str, object]) -> None:
    """Refuse unsupported fields and malformed values before resolving a layer.

    Parameters
    ----------
    values
        One layer containing only supported semantic and display values.

    Raises
    ------
    SettingsRefused
        If a field or value is unsupported.

    """
    for key, value in values.items():
        if key not in SEMANTIC_FIELDS | DISPLAY_FIELDS:
            raise SettingsRefused("Unsupported settings field; credentials cannot be imported.")
        if key in {
            "shots",
            "memory_budget_bytes",
            "n_qubits",
            "seed",
            "plot_rounding",
            "concurrency",
            "time_limit_ms",
        }:
            if type(value) is not int or value < (0 if key in {"seed", "plot_rounding"} else 1):
                raise SettingsRefused(f"Setting {key} requires a valid integer.")
        elif key == "region" and value is None:
            continue
        elif key == "unattended":
            if type(value) is not bool:
                raise SettingsRefused("Setting unattended requires an explicit boolean.")
        elif key == "parameters":
            if not isinstance(value, Mapping) or any(
                not isinstance(name, str)
                or not name
                or type(number) not in {int, float}
                or (isinstance(number, float) and not math.isfinite(number))
                for name, number in value.items()
            ):
                raise SettingsRefused("Parameters require named finite numeric values.")
        elif not isinstance(value, str) or not value:
            raise SettingsRefused(f"Setting {key} requires a nonempty string.")


def _admit_layer(
    requested: Mapping[str, object], policy: SettingsPolicy, profile: BackendProfile
) -> None:
    authority = str(policy.reference["sha256"])
    for key, value in requested.items():
        if key in {"shots", "memory_budget_bytes", "n_qubits", "concurrency", "time_limit_ms"}:
            cap = policy.ceilings.get(key)
            if cap is None or isinstance(value, int) and value > cap:
                raise SettingsRefused(
                    f"Setting {key} exceeds or lacks capacity in policy {authority}."
                )
        if key in policy.choices and value not in policy.choices[key]:
            raise SettingsRefused(f"Setting {key} is forbidden by policy {authority}.")
    if requested.get("backend", profile.backend_id) != profile.backend_id:
        raise SettingsRefused(f"Backend differs from governing route in policy {authority}.")
    if "region" in requested and requested["region"] != profile.region:
        raise SettingsRefused(f"Region differs from governing route in policy {authority}.")
    if "shots" in requested and not profile.capabilities.supports_shots:
        raise SettingsRefused(f"Shots are unsupported by governing route in policy {authority}.")
    qubits = requested.get("n_qubits")
    if (
        profile.capabilities.max_qubits is not None
        and isinstance(qubits, int)
        and qubits > profile.capabilities.max_qubits
    ):
        raise SettingsRefused(f"Qubit request exceeds governing route in policy {authority}.")


def resolve_settings(
    defaults: Mapping[str, object],
    project: Mapping[str, object],
    experiment: Mapping[str, object],
    run: Mapping[str, object],
    *,
    policy: SettingsPolicy,
    environment_ref: Mapping[str, object],
    profile: BackendProfile,
) -> ResolvedSettings:
    """Resolve defaults < project < experiment < run, refusing policy overrides.

    Parameters
    ----------
    defaults, project, experiment, run
        Requested layers; each layer is validated even if subsequently overridden.
    policy
        Immutable operator-supplied restrictions; no ceiling is inferred from hardware.
    environment_ref
        Original execution environment identity.
    profile
        Existing HAL route declaration; resolving settings grants no submit authority.

    Returns
    -------
    ResolvedSettings
        Independent immutable requested/effective values and per-field winning origins.

    Raises
    ------
    SettingsRefused
        If shape, route capability, policy choice or resource admission fails.

    """
    requested: dict[str, object] = {}
    origins: dict[str, object] = {}
    for origin, layer in (
        ("defaults", defaults),
        ("project", project),
        ("experiment", experiment),
        ("run", run),
    ):
        validate_setting_values(layer)
        _admit_layer(layer, policy, profile)
        requested.update(layer)
        origins.update({key: origin for key in layer})
    return ResolvedSettings(
        {
            "requested": requested,
            "effective": requested,
            "origins": origins,
            "policy_ref": policy.reference,
            "environment_ref": environment_ref,
            "rejected_fields": [],
        }
    )


def settings_plan_digest(settings: ResolvedSettings) -> str:
    """Hash effective semantic settings with their original policy and environment.

    Parameters
    ----------
    settings
        An immutable resolved record; its full digest still includes display/provenance.

    Returns
    -------
    str
        Settings contribution to plan identity, not a complete experiment plan hash.

    """
    effective = settings.body["effective"]
    assert isinstance(effective, Mapping)
    validate_setting_values(effective)
    return canonical_digest(
        "quantum_workspace_semantic_settings.v1",
        {
            "effective": {k: v for k, v in effective.items() if k in SEMANTIC_FIELDS},
            "policy_ref": settings.body["policy_ref"],
            "environment_ref": settings.body["environment_ref"],
        },
    )
