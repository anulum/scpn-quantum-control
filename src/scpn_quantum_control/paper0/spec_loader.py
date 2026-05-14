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
DEFAULT_HPC_UPDE_BRIDGE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_hpc_upde_bridge_validation_specs_2026-05-13.json"
)
DEFAULT_STUART_LANDAU_PRECISION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_stuart_landau_precision_validation_specs_2026-05-13.json"
)
DEFAULT_PATHOLOGY_CRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_pathology_criticality_validation_specs_2026-05-13.json"
)
DEFAULT_ARTIFICIAL_SENTIENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_artificial_sentience_validation_specs_2026-05-13.json"
)
DEFAULT_ANOMALOUS_BOUNDARY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_anomalous_boundary_validation_specs_2026-05-13.json"
)
DEFAULT_SYSTEM_ROBUSTNESS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_system_robustness_validation_specs_2026-05-13.json"
)
DEFAULT_L11_INTERFACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l11_interface_validation_specs_2026-05-13.json"
)
DEFAULT_VALIDATION_STRATEGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/paper0_validation_strategy_specs_2026-05-13.json"
)
DEFAULT_GRAND_SYNTHESIS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_grand_synthesis_validation_specs_2026-05-13.json"
)
DEFAULT_ACEF_ALIGNMENT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_acef_alignment_validation_specs_2026-05-13.json"
)
DEFAULT_GAIAN_SAFETY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_gaian_safety_validation_specs_2026-05-13.json"
)
DEFAULT_ETHICAL_IMPERATIVE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ethical_imperative_validation_specs_2026-05-13.json"
)
DEFAULT_COSMOLOGICAL_IMPLICATIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_cosmological_implications_validation_specs_2026-05-13.json"
)
DEFAULT_DARK_SECTOR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_dark_sector_validation_specs_2026-05-13.json"
)
DEFAULT_SYMMETRY_RESTORATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_symmetry_restoration_validation_specs_2026-05-13.json"
)
DEFAULT_T0_SEEDING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_t0_seeding_validation_specs_2026-05-13.json"
)
DEFAULT_SEED_FUNCTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_seed_function_validation_specs_2026-05-13.json"
)
DEFAULT_FINE_TUNING_PES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_fine_tuning_pes_validation_specs_2026-05-13.json"
)
DEFAULT_ADVANCED_MECHANISMS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_advanced_mechanisms_validation_specs_2026-05-13.json"
)
DEFAULT_STDP_SOC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/paper0_stdp_soc_validation_specs_2026-05-13.json"
)
DEFAULT_GLIAL_SLOW_CONTROL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_glial_slow_control_validation_specs_2026-05-13.json"
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


def load_hpc_upde_bridge_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted HPC/UPDE bridge validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_HPC_UPDE_BRIDGE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"HPC/UPDE bridge validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"HPC/UPDE bridge validation spec {key!r} not found in {path}")


def load_stuart_landau_precision_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Stuart-Landau precision validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_STUART_LANDAU_PRECISION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Stuart-Landau precision validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Stuart-Landau precision validation spec {key!r} not found in {path}")


def load_pathology_criticality_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted pathology/criticality validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_PATHOLOGY_CRITICALITY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"pathology/criticality validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"pathology/criticality validation spec {key!r} not found in {path}")


def load_artificial_sentience_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted artificial-sentience validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ARTIFICIAL_SENTIENCE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"artificial-sentience validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"artificial-sentience validation spec {key!r} not found in {path}")


def load_anomalous_boundary_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted anomalous-phenomena boundary validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ANOMALOUS_BOUNDARY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"anomalous-boundary validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"anomalous-boundary validation spec {key!r} not found in {path}")


def load_system_robustness_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted system-robustness validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_SYSTEM_ROBUSTNESS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"system-robustness validation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"system-robustness validation spec {key!r} not found in {path}")


def load_l11_interface_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted L11 interface validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L11_INTERFACE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"L11 interface validation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"L11 interface validation spec {key!r} not found in {path}")


def load_validation_strategy_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted validation-strategy spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_VALIDATION_STRATEGY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"validation-strategy spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"validation-strategy spec {key!r} not found in {path}")


def load_grand_synthesis_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Grand Synthesis validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_GRAND_SYNTHESIS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Grand Synthesis spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Grand Synthesis spec {key!r} not found in {path}")


def load_acef_alignment_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted A-CEF alignment validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ACEF_ALIGNMENT_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"A-CEF alignment spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"A-CEF alignment spec {key!r} not found in {path}")


def load_gaian_safety_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Gaian safety validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_GAIAN_SAFETY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Gaian safety spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Gaian safety spec {key!r} not found in {path}")


def load_ethical_imperative_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Ethical Imperative validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ETHICAL_IMPERATIVE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ethical Imperative spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Ethical Imperative spec {key!r} not found in {path}")


def load_cosmological_implications_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted cosmological implications validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_COSMOLOGICAL_IMPLICATIONS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cosmological implications spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Cosmological implications spec {key!r} not found in {path}")


def load_dark_sector_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted dark-sector validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_DARK_SECTOR_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dark-sector spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Dark-sector spec {key!r} not found in {path}")


def load_symmetry_restoration_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted symmetry-restoration validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_SYMMETRY_RESTORATION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Symmetry-restoration spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Symmetry-restoration spec {key!r} not found in {path}")


def load_t0_seeding_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted t0-seeding validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_T0_SEEDING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"t0-seeding spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"t0-seeding spec {key!r} not found in {path}")


def load_seed_function_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted seed-function validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_SEED_FUNCTION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Seed-function spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Seed-function spec {key!r} not found in {path}")


def load_fine_tuning_pes_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted fine-tuning PES validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_FINE_TUNING_PES_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Fine-tuning PES spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Fine-tuning PES spec {key!r} not found in {path}")


def load_advanced_mechanisms_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted advanced-mechanisms validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ADVANCED_MECHANISMS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Advanced-mechanisms spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Advanced-mechanisms spec {key!r} not found in {path}")


def load_stdp_soc_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted STDP/SOC validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_STDP_SOC_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"STDP/SOC spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"STDP/SOC spec {key!r} not found in {path}")


def load_glial_slow_control_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted glial slow-control validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_GLIAL_SLOW_CONTROL_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Glial slow-control spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Glial slow-control spec {key!r} not found in {path}")


__all__ = [
    "DEFAULT_ADVANCED_MECHANISMS_SPEC_BUNDLE",
    "DEFAULT_ACEF_ALIGNMENT_SPEC_BUNDLE",
    "DEFAULT_ANOMALOUS_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_ARTIFICIAL_SENTIENCE_SPEC_BUNDLE",
    "DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_IMPLICATIONS_SPEC_BUNDLE",
    "DEFAULT_DARK_SECTOR_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_IMPERATIVE_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE",
    "DEFAULT_FINE_TUNING_PES_SPEC_BUNDLE",
    "DEFAULT_FREE_ENERGY_SPEC_BUNDLE",
    "DEFAULT_GAIAN_SAFETY_SPEC_BUNDLE",
    "DEFAULT_GLIAL_SLOW_CONTROL_SPEC_BUNDLE",
    "DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE",
    "DEFAULT_GRAND_SYNTHESIS_SPEC_BUNDLE",
    "DEFAULT_HPC_UPDE_BRIDGE_SPEC_BUNDLE",
    "DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE",
    "DEFAULT_L11_INTERFACE_SPEC_BUNDLE",
    "DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE",
    "DEFAULT_NEUROVASCULAR_SPEC_BUNDLE",
    "DEFAULT_PATHOLOGY_CRITICALITY_SPEC_BUNDLE",
    "DEFAULT_SEED_FUNCTION_SPEC_BUNDLE",
    "DEFAULT_STUART_LANDAU_PRECISION_SPEC_BUNDLE",
    "DEFAULT_STDP_SOC_SPEC_BUNDLE",
    "DEFAULT_SYMMETRY_RESTORATION_SPEC_BUNDLE",
    "DEFAULT_T0_SEEDING_SPEC_BUNDLE",
    "DEFAULT_SYSTEM_ROBUSTNESS_SPEC_BUNDLE",
    "DEFAULT_UPDE_SPEC_BUNDLE",
    "DEFAULT_VALIDATION_STRATEGY_SPEC_BUNDLE",
    "load_advanced_mechanisms_validation_spec",
    "load_acef_alignment_validation_spec",
    "load_anomalous_boundary_validation_spec",
    "load_artificial_sentience_validation_spec",
    "load_computational_threshold_validation_spec",
    "load_cosmological_implications_validation_spec",
    "load_dark_sector_validation_spec",
    "load_ethical_imperative_validation_spec",
    "load_ethical_gauge_validation_spec",
    "load_fine_tuning_pes_validation_spec",
    "load_free_energy_validation_spec",
    "load_gaian_safety_validation_spec",
    "load_glial_slow_control_validation_spec",
    "load_glial_control_validation_spec",
    "load_grand_synthesis_validation_spec",
    "load_hpc_upde_bridge_validation_spec",
    "load_information_thermodynamics_validation_spec",
    "load_l11_interface_validation_spec",
    "load_macro_transition_validation_spec",
    "load_neurovascular_validation_spec",
    "load_pathology_criticality_validation_spec",
    "load_seed_function_validation_spec",
    "load_stuart_landau_precision_validation_spec",
    "load_stdp_soc_validation_spec",
    "load_symmetry_restoration_validation_spec",
    "load_t0_seeding_validation_spec",
    "load_system_robustness_validation_spec",
    "load_upde_validation_spec",
    "load_validation_strategy_spec",
]
