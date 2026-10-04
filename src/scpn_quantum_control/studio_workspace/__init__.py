# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio workspace contracts
"""Exact local workspace documents, separate from numerical evidence codecs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .canonical import canonical_bytes, canonical_digest
    from .contracts import (
        ExperimentRevision,
        LocalRunRecord,
        ParameterSpec,
        ResolvedSettings,
        WorkspaceDocument,
        WorkspaceManifest,
        parse_document,
        parse_experiment_revision,
        parse_local_run_record,
        parse_parameter_spec,
        parse_resolved_settings,
        parse_workspace_manifest,
        validate_parameter_binding,
    )
    from .graph import RawArtifact, RawIdentity, WorkspaceAdmission, admit_workspace
    from .json_transport import read_json, write_json
    from .operator_policy import (
        assess_workspace_operator_policy as assess_workspace_operator_policy,
    )
    from .operator_policy import operator_request_from_settings as operator_request_from_settings
    from .settings import SettingsPolicy, SettingsRefused, resolve_settings, settings_plan_digest
    from .settings_portability import (
        SettingsResetPreview,
        confirm_settings_reset,
        export_settings,
        import_settings,
        preview_settings_reset,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "assess_workspace_operator_policy": (
        "scpn_quantum_control.studio_workspace.operator_policy",
        "assess_workspace_operator_policy",
    ),
    "operator_request_from_settings": (
        "scpn_quantum_control.studio_workspace.operator_policy",
        "operator_request_from_settings",
    ),
    "SettingsPolicy": ("scpn_quantum_control.studio_workspace.settings", "SettingsPolicy"),
    "SettingsRefused": ("scpn_quantum_control.studio_workspace.settings", "SettingsRefused"),
    "resolve_settings": ("scpn_quantum_control.studio_workspace.settings", "resolve_settings"),
    "settings_plan_digest": (
        "scpn_quantum_control.studio_workspace.settings",
        "settings_plan_digest",
    ),
    "SettingsResetPreview": (
        "scpn_quantum_control.studio_workspace.settings_portability",
        "SettingsResetPreview",
    ),
    "export_settings": (
        "scpn_quantum_control.studio_workspace.settings_portability",
        "export_settings",
    ),
    "import_settings": (
        "scpn_quantum_control.studio_workspace.settings_portability",
        "import_settings",
    ),
    "preview_settings_reset": (
        "scpn_quantum_control.studio_workspace.settings_portability",
        "preview_settings_reset",
    ),
    "confirm_settings_reset": (
        "scpn_quantum_control.studio_workspace.settings_portability",
        "confirm_settings_reset",
    ),
    "ExperimentRevision": (
        "scpn_quantum_control.studio_workspace.contracts",
        "ExperimentRevision",
    ),
    "LocalRunRecord": ("scpn_quantum_control.studio_workspace.contracts", "LocalRunRecord"),
    "ParameterSpec": ("scpn_quantum_control.studio_workspace.contracts", "ParameterSpec"),
    "RawArtifact": ("scpn_quantum_control.studio_workspace.graph", "RawArtifact"),
    "RawIdentity": ("scpn_quantum_control.studio_workspace.graph", "RawIdentity"),
    "ResolvedSettings": ("scpn_quantum_control.studio_workspace.contracts", "ResolvedSettings"),
    "WorkspaceAdmission": ("scpn_quantum_control.studio_workspace.graph", "WorkspaceAdmission"),
    "WorkspaceDocument": ("scpn_quantum_control.studio_workspace.contracts", "WorkspaceDocument"),
    "WorkspaceManifest": ("scpn_quantum_control.studio_workspace.contracts", "WorkspaceManifest"),
    "admit_workspace": ("scpn_quantum_control.studio_workspace.graph", "admit_workspace"),
    "canonical_bytes": ("scpn_quantum_control.studio_workspace.canonical", "canonical_bytes"),
    "canonical_digest": ("scpn_quantum_control.studio_workspace.canonical", "canonical_digest"),
    "parse_document": ("scpn_quantum_control.studio_workspace.contracts", "parse_document"),
    "parse_experiment_revision": (
        "scpn_quantum_control.studio_workspace.contracts",
        "parse_experiment_revision",
    ),
    "parse_local_run_record": (
        "scpn_quantum_control.studio_workspace.contracts",
        "parse_local_run_record",
    ),
    "parse_parameter_spec": (
        "scpn_quantum_control.studio_workspace.contracts",
        "parse_parameter_spec",
    ),
    "parse_resolved_settings": (
        "scpn_quantum_control.studio_workspace.contracts",
        "parse_resolved_settings",
    ),
    "parse_workspace_manifest": (
        "scpn_quantum_control.studio_workspace.contracts",
        "parse_workspace_manifest",
    ),
    "read_json": ("scpn_quantum_control.studio_workspace.json_transport", "read_json"),
    "validate_parameter_binding": (
        "scpn_quantum_control.studio_workspace.contracts",
        "validate_parameter_binding",
    ),
    "write_json": ("scpn_quantum_control.studio_workspace.json_transport", "write_json"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "SettingsPolicy",
    "SettingsRefused",
    "resolve_settings",
    "settings_plan_digest",
    "SettingsResetPreview",
    "export_settings",
    "import_settings",
    "preview_settings_reset",
    "confirm_settings_reset",
    "ExperimentRevision",
    "LocalRunRecord",
    "ParameterSpec",
    "RawArtifact",
    "RawIdentity",
    "ResolvedSettings",
    "WorkspaceAdmission",
    "WorkspaceDocument",
    "WorkspaceManifest",
    "admit_workspace",
    "canonical_bytes",
    "canonical_digest",
    "parse_document",
    "parse_experiment_revision",
    "parse_local_run_record",
    "parse_parameter_spec",
    "parse_resolved_settings",
    "parse_workspace_manifest",
    "read_json",
    "validate_parameter_binding",
    "write_json",
]
