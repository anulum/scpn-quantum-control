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
DEFAULT_L5_ACTIVE_INFERENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l5_active_inference_validation_specs_2026-05-13.json"
)
DEFAULT_L5_ACTIVE_INFERENCE_MATH_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l5_active_inference_math_validation_specs_2026-05-13.json"
)
DEFAULT_L5_TRIPLE_NETWORK_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l5_triple_network_validation_specs_2026-05-13.json"
)
DEFAULT_L5_FOUR_STROKE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l5_four_stroke_validation_specs_2026-05-13.json"
)
DEFAULT_L5_TDA_NEUROPHENOMENOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l5_tda_neurophenomenology_validation_specs_2026-05-13.json"
)
DEFAULT_COLLECTIVE_NICHE_CONSTRUCTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_collective_niche_construction_validation_specs_2026-05-13.json"
)
DEFAULT_CISS_BIOELECTRIC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ciss_bioelectric_validation_specs_2026-05-13.json"
)
DEFAULT_RAG_QEC_STACK_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_rag_qec_stack_validation_specs_2026-05-13.json"
)
DEFAULT_HPC_UPDE_DERIVATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_hpc_upde_derivation_validation_specs_2026-05-13.json"
)
DEFAULT_TWO_TIMESCALE_QUASICRITICAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_two_timescale_quasicritical_validation_specs_2026-05-13.json"
)
DEFAULT_NV_QUANTUM_SENSING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_nv_quantum_sensing_validation_specs_2026-05-13.json"
)
DEFAULT_L11_NTHS_COMPUTATIONAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l11_nths_computational_validation_specs_2026-05-13.json"
)
DEFAULT_CATEGORY_GRAMMAR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_category_grammar_validation_specs_2026-05-13.json"
)
DEFAULT_HAMILTONIAN_INDEX_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_hamiltonian_index_validation_specs_2026-05-13.json"
)
DEFAULT_COSMOLOGICAL_EOS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_cosmological_eos_validation_specs_2026-05-13.json"
)
DEFAULT_COSMOLOGICAL_PREDICTIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_cosmological_predictions_validation_specs_2026-05-13.json"
)
DEFAULT_COMPUTATIONAL_VERIFICATION_TOOLS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_computational_verification_tools_validation_specs_2026-05-13.json"
)
DEFAULT_TERMINAL_BOUNDARY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_terminal_boundary_validation_specs_2026-05-13.json"
)
DEFAULT_THREE_CHANNEL_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_three_channel_coupling_validation_specs_2026-05-13.json"
)
DEFAULT_OPENING_FOUNDATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_opening_foundation_validation_specs_2026-05-13.json"
)
DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_front_matter_context_validation_specs_2026-05-13.json"
)
DEFAULT_CHAPTER_ROADMAP_CONTEXT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_chapter_roadmap_context_validation_specs_2026-05-13.json"
)
DEFAULT_OBJECTIVE_COVER_CONTEXT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_objective_cover_context_validation_specs_2026-05-13.json"
)
DEFAULT_POSITIONING_PREFACE_CONTEXT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_positioning_preface_context_validation_specs_2026-05-13.json"
)
DEFAULT_FOREWORD_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_foreword_coupling_validation_specs_2026-05-13.json"
)
DEFAULT_PREFACE_I_RIGOUR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_preface_i_rigour_validation_specs_2026-05-13.json"
)
DEFAULT_PREFACE_II_VISIONARY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_preface_ii_visionary_validation_specs_2026-05-13.json"
)
DEFAULT_STATUS_METHOD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_status_method_validation_specs_2026-05-13.json"
)
DEFAULT_STATUS_METHOD_CONTINUATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_status_method_continuation_validation_specs_2026-05-13.json"
)
DEFAULT_ANULUM_COLLECTION_MANDATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_anulum_collection_mandate_validation_specs_2026-05-13.json"
)
DEFAULT_LAYER_MONOGRAPH_SUITE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_layer_monograph_suite_validation_specs_2026-05-13.json"
)
DEFAULT_FOUNDATIONAL_VIABILITY_POSTULATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_foundational_viability_postulate_validation_specs_2026-05-13.json"
)
DEFAULT_U1_FIM_MULTISCALE_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_u1_fim_multiscale_dynamics_validation_specs_2026-05-13.json"
)
DEFAULT_LOGOS_RECURSIVE_CLOSURE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_logos_recursive_closure_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOMATIC_NTILDE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiomatic_ntilde_validation_specs_2026-05-13.json"
)
DEFAULT_TERMINOLOGY_BRIDGE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_terminology_bridge_validation_specs_2026-05-13.json"
)
DEFAULT_CORE_OPERATING_ASSUMPTIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_core_operating_assumptions_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_I_PSI_FIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_psi_field_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_I_MODEL_CLASS_OVERVIEW_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_model_class_overview_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_I_META_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_meta_coupling_validation_specs_2026-05-13.json"
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


def load_l5_active_inference_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Layer 5 Active Inference validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L5_ACTIVE_INFERENCE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Layer 5 Active Inference spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Layer 5 Active Inference spec {key!r} not found in {path}")


def load_l5_active_inference_math_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Layer 5 Active Inference math validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L5_ACTIVE_INFERENCE_MATH_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Layer 5 Active Inference math spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Layer 5 Active Inference math spec {key!r} not found in {path}")


def load_l5_triple_network_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Layer 5 Triple Network validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L5_TRIPLE_NETWORK_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Layer 5 Triple Network spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Layer 5 Triple Network spec {key!r} not found in {path}")


def load_l5_four_stroke_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Layer 5 four-stroke validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L5_FOUR_STROKE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Layer 5 four-stroke spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Layer 5 four-stroke spec {key!r} not found in {path}")


def load_l5_tda_neurophenomenology_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Layer 5 TDA/neurophenomenology validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L5_TDA_NEUROPHENOMENOLOGY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Layer 5 TDA/neurophenomenology spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Layer 5 TDA/neurophenomenology spec {key!r} not found in {path}")


def load_collective_niche_construction_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted collective niche construction validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_COLLECTIVE_NICHE_CONSTRUCTION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"collective niche construction spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"collective niche construction spec {key!r} not found in {path}")


def load_ciss_bioelectric_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted CISS-bioelectric validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_CISS_BIOELECTRIC_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CISS-bioelectric spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"CISS-bioelectric spec {key!r} not found in {path}")


def load_rag_qec_stack_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted RAG QEC-stack validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_RAG_QEC_STACK_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RAG QEC-stack spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"RAG QEC-stack spec {key!r} not found in {path}")


def load_hpc_upde_derivation_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted HPC-UPDE derivation validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_HPC_UPDE_DERIVATION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"HPC-UPDE derivation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"HPC-UPDE derivation spec {key!r} not found in {path}")


def load_two_timescale_quasicritical_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted two-timescale quasicritical validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_TWO_TIMESCALE_QUASICRITICAL_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"two-timescale quasicritical spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"two-timescale quasicritical spec {key!r} not found in {path}")


def load_nv_quantum_sensing_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted NV quantum sensing protocol validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_NV_QUANTUM_SENSING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"NV quantum sensing spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"NV quantum sensing spec {key!r} not found in {path}")


def load_l11_nths_computational_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted L11 NTHS computational validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_L11_NTHS_COMPUTATIONAL_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"L11 NTHS computational spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"L11 NTHS computational spec {key!r} not found in {path}")


def load_category_grammar_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted category grammar validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_CATEGORY_GRAMMAR_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"category grammar spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"category grammar spec {key!r} not found in {path}")


def load_hamiltonian_index_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Hamiltonian index validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_HAMILTONIAN_INDEX_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Hamiltonian index spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Hamiltonian index spec {key!r} not found in {path}")


def load_cosmological_eos_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted cosmological equation-of-state validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_COSMOLOGICAL_EOS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cosmological EOS spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"cosmological EOS spec {key!r} not found in {path}")


def load_cosmological_predictions_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted cosmological predictions validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_COSMOLOGICAL_PREDICTIONS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cosmological predictions spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"cosmological predictions spec {key!r} not found in {path}")


def load_computational_verification_tools_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted computational verification tools validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_COMPUTATIONAL_VERIFICATION_TOOLS_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"computational verification tools spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"computational verification tools spec {key!r} not found in {path}")


def load_terminal_boundary_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted terminal boundary validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_TERMINAL_BOUNDARY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"terminal boundary spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"terminal boundary spec {key!r} not found in {path}")


def load_three_channel_coupling_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted three-channel coupling validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_THREE_CHANNEL_COUPLING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"three-channel coupling spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"three-channel coupling spec {key!r} not found in {path}")


def load_opening_foundation_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted opening foundation validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_OPENING_FOUNDATION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"opening foundation spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"opening foundation spec {key!r} not found in {path}")


def load_front_matter_context_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted front matter context validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"front matter context spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"front matter context spec {key!r} not found in {path}")


def load_chapter_roadmap_context_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted chapter roadmap context validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_CHAPTER_ROADMAP_CONTEXT_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"chapter roadmap context spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"chapter roadmap context spec {key!r} not found in {path}")


def load_objective_cover_context_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted objective cover context validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_OBJECTIVE_COVER_CONTEXT_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"objective cover context spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"objective cover context spec {key!r} not found in {path}")


def load_positioning_preface_context_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Positioning Preface context validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_POSITIONING_PREFACE_CONTEXT_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Positioning Preface context spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Positioning Preface context spec {key!r} not found in {path}")


def load_foreword_coupling_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Foreword coupling validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_FOREWORD_COUPLING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Foreword coupling spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Foreword coupling spec {key!r} not found in {path}")


def load_preface_i_rigour_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Preface I rigour validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_PREFACE_I_RIGOUR_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Preface I rigour spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Preface I rigour spec {key!r} not found in {path}")


def load_preface_ii_visionary_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Preface II visionary validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_PREFACE_II_VISIONARY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Preface II visionary spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Preface II visionary spec {key!r} not found in {path}")


def load_status_method_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Status and Method validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_STATUS_METHOD_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Status and Method spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Status and Method spec {key!r} not found in {path}")


def load_status_method_continuation_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Status and Method continuation validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_STATUS_METHOD_CONTINUATION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Status and Method continuation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Status and Method continuation spec {key!r} not found in {path}")


def load_anulum_collection_mandate_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Anulum Collection mandate validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_ANULUM_COLLECTION_MANDATE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Anulum Collection mandate spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Anulum Collection mandate spec {key!r} not found in {path}")


def load_layer_monograph_suite_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted layer monograph suite validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_LAYER_MONOGRAPH_SUITE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"layer monograph suite spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"layer monograph suite spec {key!r} not found in {path}")


def load_foundational_viability_postulate_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted foundational viability postulate validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_FOUNDATIONAL_VIABILITY_POSTULATE_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"foundational viability postulate spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"foundational viability postulate spec {key!r} not found in {path}")


def load_u1_fim_multiscale_dynamics_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted U(1)/FIM multiscale-dynamics validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_U1_FIM_MULTISCALE_DYNAMICS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"U(1)/FIM multiscale-dynamics spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"U(1)/FIM multiscale-dynamics spec {key!r} not found in {path}")


def load_logos_recursive_closure_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Logos recursive-closure validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_LOGOS_RECURSIVE_CLOSURE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Logos recursive-closure spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Logos recursive-closure spec {key!r} not found in {path}")


def load_axiomatic_ntilde_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted formal Logos/Ntilde validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOMATIC_NTILDE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"formal Logos/Ntilde spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"formal Logos/Ntilde spec {key!r} not found in {path}")


def load_terminology_bridge_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted terminology-bridge validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_TERMINOLOGY_BRIDGE_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"terminology-bridge spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"terminology-bridge spec {key!r} not found in {path}")


def load_core_operating_assumptions_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted core-operating-assumptions validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_CORE_OPERATING_ASSUMPTIONS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"core-operating-assumptions spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"core-operating-assumptions spec {key!r} not found in {path}")


def load_axiom_i_psi_field_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I Psi-field validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_PSI_FIELD_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Axiom I Psi-field spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I Psi-field spec {key!r} not found in {path}")


def load_axiom_i_model_class_overview_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I model-class overview validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_MODEL_CLASS_OVERVIEW_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom I model-class overview spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I model-class overview spec {key!r} not found in {path}")


def load_axiom_i_meta_coupling_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I meta-coupling validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_META_COUPLING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Axiom I meta-coupling spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I meta-coupling spec {key!r} not found in {path}")


__all__ = [
    "DEFAULT_ANULUM_COLLECTION_MANDATE_SPEC_BUNDLE",
    "DEFAULT_CATEGORY_GRAMMAR_SPEC_BUNDLE",
    "DEFAULT_CHAPTER_ROADMAP_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_COLLECTIVE_NICHE_CONSTRUCTION_SPEC_BUNDLE",
    "DEFAULT_CISS_BIOELECTRIC_SPEC_BUNDLE",
    "DEFAULT_ADVANCED_MECHANISMS_SPEC_BUNDLE",
    "DEFAULT_ACEF_ALIGNMENT_SPEC_BUNDLE",
    "DEFAULT_ANOMALOUS_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_ARTIFICIAL_SENTIENCE_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_META_COUPLING_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_MODEL_CLASS_OVERVIEW_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_PSI_FIELD_SPEC_BUNDLE",
    "DEFAULT_AXIOMATIC_NTILDE_SPEC_BUNDLE",
    "DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE",
    "DEFAULT_COMPUTATIONAL_VERIFICATION_TOOLS_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_IMPLICATIONS_SPEC_BUNDLE",
    "DEFAULT_CORE_OPERATING_ASSUMPTIONS_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_EOS_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_PREDICTIONS_SPEC_BUNDLE",
    "DEFAULT_DARK_SECTOR_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_IMPERATIVE_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE",
    "DEFAULT_FINE_TUNING_PES_SPEC_BUNDLE",
    "DEFAULT_FOREWORD_COUPLING_SPEC_BUNDLE",
    "DEFAULT_FOUNDATIONAL_VIABILITY_POSTULATE_SPEC_BUNDLE",
    "DEFAULT_FREE_ENERGY_SPEC_BUNDLE",
    "DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_GAIAN_SAFETY_SPEC_BUNDLE",
    "DEFAULT_GLIAL_SLOW_CONTROL_SPEC_BUNDLE",
    "DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE",
    "DEFAULT_GRAND_SYNTHESIS_SPEC_BUNDLE",
    "DEFAULT_HAMILTONIAN_INDEX_SPEC_BUNDLE",
    "DEFAULT_HPC_UPDE_BRIDGE_SPEC_BUNDLE",
    "DEFAULT_HPC_UPDE_DERIVATION_SPEC_BUNDLE",
    "DEFAULT_L11_NTHS_COMPUTATIONAL_SPEC_BUNDLE",
    "DEFAULT_NV_QUANTUM_SENSING_SPEC_BUNDLE",
    "DEFAULT_OBJECTIVE_COVER_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_OPENING_FOUNDATION_SPEC_BUNDLE",
    "DEFAULT_POSITIONING_PREFACE_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_PREFACE_I_RIGOUR_SPEC_BUNDLE",
    "DEFAULT_PREFACE_II_VISIONARY_SPEC_BUNDLE",
    "DEFAULT_TWO_TIMESCALE_QUASICRITICAL_SPEC_BUNDLE",
    "DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE",
    "DEFAULT_L5_ACTIVE_INFERENCE_MATH_SPEC_BUNDLE",
    "DEFAULT_L5_ACTIVE_INFERENCE_SPEC_BUNDLE",
    "DEFAULT_L5_FOUR_STROKE_SPEC_BUNDLE",
    "DEFAULT_L5_TDA_NEUROPHENOMENOLOGY_SPEC_BUNDLE",
    "DEFAULT_L5_TRIPLE_NETWORK_SPEC_BUNDLE",
    "DEFAULT_L11_INTERFACE_SPEC_BUNDLE",
    "DEFAULT_LAYER_MONOGRAPH_SUITE_SPEC_BUNDLE",
    "DEFAULT_LOGOS_RECURSIVE_CLOSURE_SPEC_BUNDLE",
    "DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE",
    "DEFAULT_NEUROVASCULAR_SPEC_BUNDLE",
    "DEFAULT_PATHOLOGY_CRITICALITY_SPEC_BUNDLE",
    "DEFAULT_RAG_QEC_STACK_SPEC_BUNDLE",
    "DEFAULT_SEED_FUNCTION_SPEC_BUNDLE",
    "DEFAULT_STATUS_METHOD_CONTINUATION_SPEC_BUNDLE",
    "DEFAULT_STATUS_METHOD_SPEC_BUNDLE",
    "DEFAULT_STUART_LANDAU_PRECISION_SPEC_BUNDLE",
    "DEFAULT_STDP_SOC_SPEC_BUNDLE",
    "DEFAULT_SYMMETRY_RESTORATION_SPEC_BUNDLE",
    "DEFAULT_T0_SEEDING_SPEC_BUNDLE",
    "DEFAULT_SYSTEM_ROBUSTNESS_SPEC_BUNDLE",
    "DEFAULT_TERMINAL_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_TERMINOLOGY_BRIDGE_SPEC_BUNDLE",
    "DEFAULT_THREE_CHANNEL_COUPLING_SPEC_BUNDLE",
    "DEFAULT_U1_FIM_MULTISCALE_DYNAMICS_SPEC_BUNDLE",
    "DEFAULT_UPDE_SPEC_BUNDLE",
    "DEFAULT_VALIDATION_STRATEGY_SPEC_BUNDLE",
    "load_anulum_collection_mandate_validation_spec",
    "load_collective_niche_construction_validation_spec",
    "load_chapter_roadmap_context_validation_spec",
    "load_ciss_bioelectric_validation_spec",
    "load_advanced_mechanisms_validation_spec",
    "load_acef_alignment_validation_spec",
    "load_anomalous_boundary_validation_spec",
    "load_artificial_sentience_validation_spec",
    "load_axiom_i_meta_coupling_validation_spec",
    "load_axiom_i_model_class_overview_validation_spec",
    "load_axiom_i_psi_field_validation_spec",
    "load_axiomatic_ntilde_validation_spec",
    "load_computational_threshold_validation_spec",
    "load_computational_verification_tools_validation_spec",
    "load_cosmological_implications_validation_spec",
    "load_core_operating_assumptions_validation_spec",
    "load_dark_sector_validation_spec",
    "load_ethical_imperative_validation_spec",
    "load_ethical_gauge_validation_spec",
    "load_fine_tuning_pes_validation_spec",
    "load_foreword_coupling_validation_spec",
    "load_foundational_viability_postulate_validation_spec",
    "load_free_energy_validation_spec",
    "load_front_matter_context_validation_spec",
    "load_gaian_safety_validation_spec",
    "load_glial_slow_control_validation_spec",
    "load_glial_control_validation_spec",
    "load_grand_synthesis_validation_spec",
    "load_hpc_upde_bridge_validation_spec",
    "load_hpc_upde_derivation_validation_spec",
    "load_l11_nths_computational_validation_spec",
    "load_category_grammar_validation_spec",
    "load_hamiltonian_index_validation_spec",
    "load_cosmological_eos_validation_spec",
    "load_cosmological_predictions_validation_spec",
    "load_nv_quantum_sensing_validation_spec",
    "load_objective_cover_context_validation_spec",
    "load_opening_foundation_validation_spec",
    "load_positioning_preface_context_validation_spec",
    "load_preface_i_rigour_validation_spec",
    "load_preface_ii_visionary_validation_spec",
    "load_two_timescale_quasicritical_validation_spec",
    "load_information_thermodynamics_validation_spec",
    "load_l5_active_inference_math_validation_spec",
    "load_l5_active_inference_validation_spec",
    "load_l5_four_stroke_validation_spec",
    "load_l5_tda_neurophenomenology_validation_spec",
    "load_l5_triple_network_validation_spec",
    "load_l11_interface_validation_spec",
    "load_layer_monograph_suite_validation_spec",
    "load_logos_recursive_closure_validation_spec",
    "load_macro_transition_validation_spec",
    "load_neurovascular_validation_spec",
    "load_pathology_criticality_validation_spec",
    "load_rag_qec_stack_validation_spec",
    "load_seed_function_validation_spec",
    "load_status_method_continuation_validation_spec",
    "load_status_method_validation_spec",
    "load_stuart_landau_precision_validation_spec",
    "load_stdp_soc_validation_spec",
    "load_symmetry_restoration_validation_spec",
    "load_t0_seeding_validation_spec",
    "load_system_robustness_validation_spec",
    "load_terminal_boundary_validation_spec",
    "load_terminology_bridge_validation_spec",
    "load_three_channel_coupling_validation_spec",
    "load_u1_fim_multiscale_dynamics_validation_spec",
    "load_upde_validation_spec",
    "load_validation_strategy_spec",
]
