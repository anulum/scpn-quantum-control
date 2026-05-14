# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 spec loader
"""Load promoted Paper 0 validation specs from repository artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._paths import project_data_path

DEFAULT_UPDE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/paper0_upde_validation_specs_2026-05-13.json"
)
DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_macro_transition_validation_specs_2026-05-13.json"
)
DEFAULT_NEUROVASCULAR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_neurovascular_validation_specs_2026-05-13.json"
)
DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_glial_control_validation_specs_2026-05-13.json"
)
DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_information_thermodynamics_validation_specs_2026-05-13.json"
)
DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_computational_threshold_validation_specs_2026-05-13.json"
)
DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ethical_gauge_validation_specs_2026-05-13.json"
)
DEFAULT_FREE_ENERGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_free_energy_validation_specs_2026-05-13.json"
)


def load_upde_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted UPDE validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_UPDE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"UPDE validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"UPDE validation spec {key!r} not found in {path}")


def load_macro_transition_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted macro-transition validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"macro-transition validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"macro-transition validation spec {key!r} not found in {path}")


def load_neurovascular_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted neurovascular validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_NEUROVASCULAR_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"neurovascular validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"neurovascular validation spec {key!r} not found in {path}")


def load_glial_control_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted glial-control validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"glial-control validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"glial-control validation spec {key!r} not found in {path}")


def load_information_thermodynamics_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted information-thermodynamics validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"information-thermodynamics validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"information-thermodynamics validation spec {key!r} not found in {path}")


def load_computational_threshold_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted computational-threshold validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"computational-threshold validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"computational-threshold validation spec {key!r} not found in {path}")


def load_ethical_gauge_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted ethical-gauge validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ethical-gauge validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"ethical-gauge validation spec {key!r} not found in {path}")


def load_free_energy_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted free-energy validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_FREE_ENERGY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"free-energy validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"free-energy validation spec {key!r} not found in {path}")


__all__ = [
    "DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE",
    "DEFAULT_FREE_ENERGY_SPEC_BUNDLE",
    "DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE",
    "DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE",
    "DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE",
    "DEFAULT_NEUROVASCULAR_SPEC_BUNDLE",
    "DEFAULT_UPDE_SPEC_BUNDLE",
    "load_computational_threshold_validation_spec",
    "load_ethical_gauge_validation_spec",
    "load_free_energy_validation_spec",
    "load_glial_control_validation_spec",
    "load_information_thermodynamics_validation_spec",
    "load_macro_transition_validation_spec",
    "load_neurovascular_validation_spec",
    "load_upde_validation_spec",
]
