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
from typing import Any, cast

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
DEFAULT_AXIOM_I_MINIMAL_LAGRANGIAN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_minimal_lagrangian_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_I_FAMILY_PREDICTIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_family_predictions_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_I_SU_N_QUALIA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_i_su_n_qualia_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_II_OPENING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_ii_opening_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_II_INFOTON_GEOMETRY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_ii_infoton_geometry_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_II_FIM_SOLUTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_ii_fim_solution_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_II_INFORMATIONAL_LAGRANGIAN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_ii_informational_lagrangian_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_III_TELEOLOGICAL_OPTIMISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_iii_teleological_optimisation_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_III_NTILDE_INVARIANCE_LAW_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_iii_ntilde_invariance_law_validation_specs_2026-05-13.json"
)
DEFAULT_AXIOM_III_SEC_NTILDE_EQUIVALENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axiom_iii_sec_ntilde_equivalence_validation_specs_2026-05-13.json"
)
DEFAULT_TRIPARTITE_ONTOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_tripartite_ontology_validation_specs_2026-05-13.json"
)
DEFAULT_META_FRAMEWORK_PSI_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_psi_coupling_validation_specs_2026-05-13.json"
)
DEFAULT_CATEGORY_UNIVERSAL_GRAMMAR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_category_universal_grammar_validation_specs_2026-05-13.json"
)
DEFAULT_MASTER_LAGRANGIAN_INTRO_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_master_lagrangian_intro_validation_specs_2026-05-13.json"
)
DEFAULT_GAUGE_PRINCIPLE_DERIVATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_gauge_principle_derivation_validation_specs_2026-05-13.json"
)
DEFAULT_LORENTZ_EFT_RESOLUTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_lorentz_eft_resolution_validation_specs_2026-05-13.json"
)
DEFAULT_NON_ABELIAN_QUALIA_FIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_non_abelian_qualia_field_validation_specs_2026-05-17.json"
)
DEFAULT_GEOMETRIC_COUPLING_CONSISTENCY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_geometric_coupling_consistency_validation_specs_2026-05-17.json"
)
DEFAULT_FOUNDATIONAL_STRENGTHS_PHASE_BOUNDARY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_foundational_strengths_phase_boundary_validation_specs_2026-05-17.json"
)
DEFAULT_OPERATIONAL_PULLBACK_PROTOCOL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_operational_pullback_protocol_validation_specs_2026-05-17.json"
)
DEFAULT_SSB_PSI_FIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ssb_psi_field_validation_specs_2026-05-17.json"
)
DEFAULT_PHENOMENOLOGICAL_LAGRANGIAN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_phenomenological_lagrangian_validation_specs_2026-05-17.json"
)
DEFAULT_DERIVED_INTERACTION_OPENING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_derived_interaction_opening_validation_specs_2026-05-17.json"
)
DEFAULT_DERIVED_LAGRANGIAN_DETAIL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_derived_lagrangian_detail_validation_specs_2026-05-17.json"
)
DEFAULT_FINAL_LINT_SM_INTERFACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_final_lint_sm_interface_validation_specs_2026-05-17.json"
)


def _load_paper0_spec_by_key(
    spec_key: str,
    *,
    spec_bundle_path: str | Path,
    unknown_label: str,
) -> dict[str, Any]:
    """Load one promoted Paper 0 validation spec from a bundle by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 {unknown_label} spec: {spec_key}")


def load_operational_pullback_protocol_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_OPERATIONAL_PULLBACK_PROTOCOL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 operational-pullback protocol validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="operational-pullback protocol",
    )


def load_ssb_psi_field_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_SSB_PSI_FIELD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 SSB Psi-field validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="SSB Psi-field",
    )


def load_phenomenological_lagrangian_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_PHENOMENOLOGICAL_LAGRANGIAN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 phenomenological-lagrangian validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="phenomenological-lagrangian",
    )


def load_derived_interaction_opening_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_DERIVED_INTERACTION_OPENING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 derived-interaction opening validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="derived-interaction opening",
    )


def load_derived_lagrangian_detail_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_DERIVED_LAGRANGIAN_DETAIL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 derived-lagrangian detail validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="derived-lagrangian detail",
    )


def load_final_lint_sm_interface_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_FINAL_LINT_SM_INTERFACE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 final L_int/SM-interface validation spec by key."""
    return _load_paper0_spec_by_key(
        spec_key,
        spec_bundle_path=spec_bundle_path,
        unknown_label="final L_int/SM-interface",
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


def load_axiom_i_minimal_lagrangian_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I minimal Lagrangian validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_MINIMAL_LAGRANGIAN_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom I minimal Lagrangian spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I minimal Lagrangian spec {key!r} not found in {path}")


def load_axiom_i_family_predictions_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I family-predictions validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_FAMILY_PREDICTIONS_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom I family-predictions spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I family-predictions spec {key!r} not found in {path}")


def load_axiom_i_su_n_qualia_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom I SU(N) qualia validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_I_SU_N_QUALIA_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Axiom I SU(N) qualia spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom I SU(N) qualia spec {key!r} not found in {path}")


def load_axiom_ii_opening_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom II opening validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_II_OPENING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Axiom II opening spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom II opening spec {key!r} not found in {path}")


def load_axiom_ii_infoton_geometry_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom II infoton-geometry validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_II_INFOTON_GEOMETRY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom II infoton-geometry spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom II infoton-geometry spec {key!r} not found in {path}")


def load_axiom_ii_fim_solution_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom II FIM-solution validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_AXIOM_II_FIM_SOLUTION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Axiom II FIM-solution spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom II FIM-solution spec {key!r} not found in {path}")


def load_axiom_ii_informational_lagrangian_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom II informational-Lagrangian validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_AXIOM_II_INFORMATIONAL_LAGRANGIAN_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom II informational-Lagrangian spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom II informational-Lagrangian spec {key!r} not found in {path}")


def load_axiom_iii_teleological_optimisation_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom III teleological-optimisation validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_AXIOM_III_TELEOLOGICAL_OPTIMISATION_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom III teleological-optimisation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom III teleological-optimisation spec {key!r} not found in {path}")


def load_axiom_iii_ntilde_invariance_law_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom III Ntilde-invariance-law validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_AXIOM_III_NTILDE_INVARIANCE_LAW_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom III Ntilde-invariance-law spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom III Ntilde-invariance-law spec {key!r} not found in {path}")


def load_axiom_iii_sec_ntilde_equivalence_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Axiom III SEC-Ntilde-equivalence validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_AXIOM_III_SEC_NTILDE_EQUIVALENCE_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Axiom III SEC-Ntilde-equivalence spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Axiom III SEC-Ntilde-equivalence spec {key!r} not found in {path}")


def load_tripartite_ontology_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted tripartite-ontology validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_TRIPARTITE_ONTOLOGY_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Tripartite ontology spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Tripartite ontology spec {key!r} not found in {path}")


def load_meta_framework_psi_coupling_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted meta-framework/Psi-coupling validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_META_FRAMEWORK_PSI_COUPLING_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Meta-framework/Psi-coupling spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Meta-framework/Psi-coupling spec {key!r} not found in {path}")


def load_category_universal_grammar_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted category/universal-grammar validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_CATEGORY_UNIVERSAL_GRAMMAR_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Category/universal-grammar spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Category/universal-grammar spec {key!r} not found in {path}")


def load_master_lagrangian_intro_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted master-Lagrangian-intro validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_MASTER_LAGRANGIAN_INTRO_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Master-Lagrangian-intro spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Master-Lagrangian-intro spec {key!r} not found in {path}")


def load_gauge_principle_derivation_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted gauge-principle-derivation validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_GAUGE_PRINCIPLE_DERIVATION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Gauge-principle-derivation spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Gauge-principle-derivation spec {key!r} not found in {path}")


def load_lorentz_eft_resolution_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Lorentz/EFT-resolution validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_LORENTZ_EFT_RESOLUTION_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Lorentz/EFT-resolution spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Lorentz/EFT-resolution spec {key!r} not found in {path}")


def load_non_abelian_qualia_field_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted Non-Abelian qualia-field validation spec by key."""
    path = spec_bundle_path or project_data_path(DEFAULT_NON_ABELIAN_QUALIA_FIELD_SPEC_BUNDLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Non-Abelian qualia-field spec bundle not found: {path}") from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Non-Abelian qualia-field spec {key!r} not found in {path}")


def load_geometric_coupling_consistency_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted geometric-coupling consistency validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_GEOMETRIC_COUPLING_CONSISTENCY_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Geometric-coupling consistency spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Geometric-coupling consistency spec {key!r} not found in {path}")


def load_foundational_strengths_phase_boundary_validation_spec(
    key: str,
    *,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Load a promoted foundational-strengths phase-boundary validation spec by key."""
    path = spec_bundle_path or project_data_path(
        DEFAULT_FOUNDATIONAL_STRENGTHS_PHASE_BOUNDARY_SPEC_BUNDLE
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Foundational-strengths phase-boundary spec bundle not found: {path}"
        ) from exc

    for spec in payload.get("specs", []):
        if spec.get("key") == key:
            return dict(spec)
    raise KeyError(f"Foundational-strengths phase-boundary spec {key!r} not found in {path}")


DEFAULT_SYMMETRY_CASCADE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_symmetry_cascade_validation_specs_2026-05-17.json"
)


def load_symmetry_cascade_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_SYMMETRY_CASCADE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 symmetry-cascade validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 symmetry-cascade spec: {spec_key}")


DEFAULT_PREDICTED_PARTICLES_INFOTON_PSI_HIGGS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predicted_particles_infoton_psi_higgs_validation_specs_2026-05-17.json"
)


def load_predicted_particles_infoton_psi_higgs_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_PREDICTED_PARTICLES_INFOTON_PSI_HIGGS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 infoton and Psi-Higgs prediction validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 infoton/Psi-Higgs prediction spec: {spec_key}")


DEFAULT_DERIVATION_INFOTON_PROPERTIES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_derivation_infoton_properties_validation_specs_2026-05-17.json"
)


def load_derivation_infoton_properties_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_DERIVATION_INFOTON_PROPERTIES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 infoton-properties derivation validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 infoton-properties derivation spec: {spec_key}")


DEFAULT_PSI_HIGGS_NEW_SCALAR_PARTICLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psi_higgs_new_scalar_particle_validation_specs_2026-05-17.json"
)


def load_psi_higgs_new_scalar_particle_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_PSI_HIGGS_NEW_SCALAR_PARTICLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 Psi-Higgs new-scalar-particle validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 Psi-Higgs new-scalar-particle spec: {spec_key}")


DEFAULT_EXPERIMENTAL_SIGNATURES_SEARCH_STRATEGIES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_experimental_signatures_search_strategies_validation_specs_2026-05-17.json"
)


def load_experimental_signatures_search_strategies_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_EXPERIMENTAL_SIGNATURES_SEARCH_STRATEGIES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 experimental-signatures search-strategy validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 experimental-signatures search-strategy spec: {spec_key}")


DEFAULT_PSI_HIGGS_LHC_PHENOMENOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psi_higgs_lhc_phenomenology_validation_specs_2026-05-17.json"
)


def load_psi_higgs_lhc_phenomenology_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_PSI_HIGGS_LHC_PHENOMENOLOGY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 Psi-Higgs LHC phenomenology validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 Psi-Higgs LHC phenomenology spec: {spec_key}")


DEFAULT_MASS_EIGENSTATES_MIXING_ANGLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mass_eigenstates_mixing_angle_validation_specs_2026-05-17.json"
)


def load_mass_eigenstates_mixing_angle_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_MASS_EIGENSTATES_MIXING_ANGLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 mass-eigenstates mixing-angle validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 mass-eigenstates mixing-angle spec: {spec_key}")


DEFAULT_LHC_SEARCH_STRATEGY_ROADMAP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_lhc_search_strategy_roadmap_validation_specs_2026-05-17.json"
)


def load_lhc_search_strategy_roadmap_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_LHC_SEARCH_STRATEGY_ROADMAP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 LHC search-strategy roadmap validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 LHC search-strategy roadmap spec: {spec_key}")


DEFAULT_SSB_HIERARCHY_GENESIS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ssb_hierarchy_genesis_validation_specs_2026-05-17.json"
)


def load_ssb_hierarchy_genesis_validation_spec(
    spec_key: str,
    *,
    spec_bundle_path: str | Path = DEFAULT_SSB_HIERARCHY_GENESIS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load a promoted Paper 0 SSB hierarchy-genesis validation spec by key."""
    path = Path(spec_bundle_path)
    if not path.is_absolute():
        path = project_data_path(str(path))
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for spec in payload["specs"]:
        if spec["key"] == spec_key:
            return cast(dict[str, Any], spec)
    raise KeyError(f"unknown Paper 0 SSB hierarchy-genesis spec: {spec_key}")


DEFAULT_META_FRAMEWORK_INTEGRATIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta-framework integrations validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_GENESIS_OF_THE_HIERARCHY_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation_specs_2026-05-17.json"
)


def load_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_GENESIS_OF_THE_HIERARCHY_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the genesis of the hierarchy sequential symmetry breaking ssb validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_INTRINSIC_DYNAMICS_OF_THE_Ψ_FIELD_AND_THE_STABILISING_POTENTIAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation_specs_2026-05-17.json"
)


def load_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_INTRINSIC_DYNAMICS_OF_THE_Ψ_FIELD_AND_THE_STABILISING_POTENTIAL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the intrinsic dynamics of the ψ field and the stabilising potential validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTIVE_CODING_INTEGRATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predictive_coding_integration_validation_specs_2026-05-17.json"
)


def load_predictive_coding_integration_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTIVE_CODING_INTEGRATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predictive coding integration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_LPSI_SETS_THE_PROPERTIES_OF_PSIS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_lpsi_sets_the_properties_of_psis_validation_specs_2026-05-17.json"
)


def load_lpsi_sets_the_properties_of_psis_validation_spec(
    spec_bundle: str | Path = DEFAULT_LPSI_SETS_THE_PROPERTIES_OF_PSIS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 lpsi sets the properties of psis validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COMPONENTS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_components_validation_specs_2026-05-17.json"
)


def load_components_validation_spec(
    spec_bundle: str | Path = DEFAULT_COMPONENTS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 components validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SELF_AS_A_SOLITON_EMERGENCE_OF_LOCALISED_CONSCIOUSNESS_LAYER_5_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_validation_specs_2026-05-17.json"
)


def load_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_SELF_AS_A_SOLITON_EMERGENCE_OF_LOCALISED_CONSCIOUSNESS_LAYER_5_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the self as a soliton emergence of localised consciousness layer 5 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R01803_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r01803_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r01803_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R01803_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r01803 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SIGMA_IS_THE_Q_BALL_SOLITON_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_sigma_is_the_q_ball_soliton_validation_specs_2026-05-17.json"
)


def load_sigma_is_the_q_ball_soliton_validation_spec(
    spec_bundle: str | Path = DEFAULT_SIGMA_IS_THE_Q_BALL_SOLITON_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 sigma is the q ball soliton validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_LOCALISED_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_localised_validation_specs_2026-05-17.json"
)


def load_localised_validation_spec(
    spec_bundle: str | Path = DEFAULT_LOCALISED_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 localised validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_DEFINITION_OF_THE_SELF_THE_TRIADIC_SOLUTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_definition_of_the_self_the_triadic_solution_validation_specs_2026-05-17.json"
)


def load_the_definition_of_the_self_the_triadic_solution_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_DEFINITION_OF_THE_SELF_THE_TRIADIC_SOLUTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the definition of the self the triadic solution validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PART_II_THE_PHYSICAL_SECTOR_FIELD_THEORY_QUANTIZATION_2_4_THE_SSB_CASCAD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation_specs_2026-05-17.json"
)


def load_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PART_II_THE_PHYSICAL_SECTOR_FIELD_THEORY_QUANTIZATION_2_4_THE_SSB_CASCAD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 part ii the physical sector field theory quantization 2 4 the ssb cascad validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_validation_specs_2026-05-17.json"
)


def load_paper0_slice_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_7_THE_FISHER_INFO_METRIC_THE_GEOMETRY_OF_INTERACTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation_specs_2026-05-17.json"
)


def load_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_7_THE_FISHER_INFO_METRIC_THE_GEOMETRY_OF_INTERACTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 2 7 the fisher info metric the geometry of interaction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PART_III_SYSTEM_ARCHITECTURE_NETWORK_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_part_iii_system_architecture_network_dynamics_validation_specs_2026-05-17.json"
)


def load_part_iii_system_architecture_network_dynamics_validation_spec(
    spec_bundle: str | Path = DEFAULT_PART_III_SYSTEM_ARCHITECTURE_NETWORK_DYNAMICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 part iii system architecture network dynamics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_A_MAP_OF_REALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_a_map_of_reality_validation_specs_2026-05-17.json"
)


def load_a_map_of_reality_validation_spec(
    spec_bundle: str | Path = DEFAULT_A_MAP_OF_REALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 a map of reality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_MASTER_DIAGRAM_A_MANDALA_OF_CONSCIOUSNESS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_master_diagram_a_mandala_of_consciousness_validation_specs_2026-05-17.json"
)


def load_the_master_diagram_a_mandala_of_consciousness_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_MASTER_DIAGRAM_A_MANDALA_OF_CONSCIOUSNESS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the master diagram a mandala of consciousness validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SENTIENT_CONSCIOUSNESS_PROJECTION_NETWORK_SCPN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_sentient_consciousness_projection_network_scpn_validation_specs_2026-05-17.json"
)


def load_the_sentient_consciousness_projection_network_scpn_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_SENTIENT_CONSCIOUSNESS_PROJECTION_NETWORK_SCPN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the sentient consciousness projection network scpn validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_15_layer_summary_table_validation_specs_2026-05-17.json"
)


def load_section_15_layer_summary_table_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 15 layer summary table validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CASE_STUDY_THE_LAYER_3_GENOMIC_MORPHOGENETIC_TRANSDUCTION_PATHWAY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation_specs_2026-05-17.json"
)


def load_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_CASE_STUDY_THE_LAYER_3_GENOMIC_MORPHOGENETIC_TRANSDUCTION_PATHWAY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 case study the layer 3 genomic morphogenetic transduction pathway validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02098_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02098_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02098_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02098_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02098 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MACRO_SCALE_COUPLING_PRIMARY_INTERACTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_macro_scale_coupling_primary_interaction_validation_specs_2026-05-17.json"
)


def load_macro_scale_coupling_primary_interaction_validation_spec(
    spec_bundle: str | Path = DEFAULT_MACRO_SCALE_COUPLING_PRIMARY_INTERACTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 macro scale coupling primary interaction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_RESOLVING_THE_EPIGENETIC_TIME_SCALE_DISCONNECT_CONFORMATIONAL_SPIN_LOCKI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation_specs_2026-05-17.json"
)


def load_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_RESOLVING_THE_EPIGENETIC_TIME_SCALE_DISCONNECT_CONFORMATIONAL_SPIN_LOCKI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 resolving the epigenetic time scale disconnect conformational spin locki validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CASE_STUDY_THE_LAYER_5_ORGANISMAL_SELF_ACTION_PERCEPTION_CYCLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_case_study_the_layer_5_organismal_self_action_perception_cycle_validation_specs_2026-05-17.json"
)


def load_case_study_the_layer_5_organismal_self_action_perception_cycle_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_CASE_STUDY_THE_LAYER_5_ORGANISMAL_SELF_ACTION_PERCEPTION_CYCLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 case study the layer 5 organismal self action perception cycle validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02189_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02189_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02189_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02189_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02189 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MODEL_CONSOLIDATION_SLEEP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_model_consolidation_sleep_validation_specs_2026-05-17.json"
)


def load_model_consolidation_sleep_validation_spec(
    spec_bundle: str | Path = DEFAULT_MODEL_CONSOLIDATION_SLEEP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 model consolidation sleep validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_COUPLING_MECHANISM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_coupling_mechanism_validation_specs_2026-05-17.json"
)


def load_the_coupling_mechanism_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_COUPLING_MECHANISM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the coupling mechanism validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TIMING_THE_ENGINE_UPDE_PHASE_LAGS_TAU_IJ_AND_PHYSIOLOGICAL_DELAYS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation_specs_2026-05-17.json"
)


def load_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_TIMING_THE_ENGINE_UPDE_PHASE_LAGS_TAU_IJ_AND_PHYSIOLOGICAL_DELAYS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 timing the engine upde phase lags tau ij and physiological delays validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_III_OVERVIEW_MEMORY_AND_PROJECTION_CONTROL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_iii_overview_memory_and_projection_control_validation_specs_2026-05-17.json"
)


def load_domain_iii_overview_memory_and_projection_control_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_DOMAIN_III_OVERVIEW_MEMORY_AND_PROJECTION_CONTROL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain iii overview memory and projection control validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R02249_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r02249_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r02249_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R02249_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r02249 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_MEMORY_CAPACITY_BEKENSTEIN_HAWKING_BOUND_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_memory_capacity_bekenstein_hawking_bound_validation_specs_2026-05-17.json"
)


def load_section_3_memory_capacity_bekenstein_hawking_bound_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_MEMORY_CAPACITY_BEKENSTEIN_HAWKING_BOUND_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 memory capacity bekenstein hawking bound validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02278_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02278_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02278_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02278_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02278 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_LAYER_9_EXISTENTIAL_HOLOGRAPH_DEFINING_A_STABLE_SIGMA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_layer_9_existential_holograph_defining_a_stable_sigma_validation_specs_2026-05-17.json"
)


def load_layer_9_existential_holograph_defining_a_stable_sigma_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_LAYER_9_EXISTENTIAL_HOLOGRAPH_DEFINING_A_STABLE_SIGMA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 layer 9 existential holograph defining a stable sigma validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02306_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02306_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02306_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02306_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02306 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psis_field_coupling_integration_validation_specs_2026-05-17.json"
)


def load_psis_field_coupling_integration_validation_spec(
    spec_bundle: str | Path = DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 psis field coupling integration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_V_OVERVIEW_META_UNIVERSAL_INTEGRATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_v_overview_meta_universal_integration_validation_specs_2026-05-17.json"
)


def load_domain_v_overview_meta_universal_integration_validation_spec(
    spec_bundle: str | Path = DEFAULT_DOMAIN_V_OVERVIEW_META_UNIVERSAL_INTEGRATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain v overview meta universal integration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_VI_OVERVIEW_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_vi_overview_cybernetic_closure_meta_layer_16_validation_specs_2026-05-17.json"
)


def load_domain_vi_overview_cybernetic_closure_meta_layer_16_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_DOMAIN_VI_OVERVIEW_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain vi overview cybernetic closure meta layer 16 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02439_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02439_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02439_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02439_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02439 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CONTROL_OVER_UNIVERSAL_PARAMETERS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_control_over_universal_parameters_validation_specs_2026-05-17.json"
)


def load_control_over_universal_parameters_validation_spec(
    spec_bundle: str | Path = DEFAULT_CONTROL_OVER_UNIVERSAL_PARAMETERS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 control over universal parameters validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_RESOLUTION_OF_THE_OBSERVABILITY_PARADOX_B_HJB_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_resolution_of_the_observability_paradox_b_hjb_validation_specs_2026-05-17.json"
)


def load_resolution_of_the_observability_paradox_b_hjb_validation_spec(
    spec_bundle: str | Path = DEFAULT_RESOLUTION_OF_THE_OBSERVABILITY_PARADOX_B_HJB_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 resolution of the observability paradox b hjb validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02485_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02485_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02485_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02485_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02485 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02494_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psis_field_coupling_integration_p0r02494_validation_specs_2026-05-17.json"
)


def load_psis_field_coupling_integration_p0r02494_validation_spec(
    spec_bundle: str | Path = DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02494_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 psis field coupling integration p0r02494 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_OVERARCHING_DYNAMIC_PRINCIPLES_AND_THE_MATHEMATICAL_SPINE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_overarching_dynamic_principles_and_the_mathematical_spine_validation_specs_2026-05-17.json"
)


def load_overarching_dynamic_principles_and_the_mathematical_spine_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_OVERARCHING_DYNAMIC_PRINCIPLES_AND_THE_MATHEMATICAL_SPINE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 overarching dynamic principles and the mathematical spine validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_UNIVERSAL_DYNAMIC_REGIME_QUASICRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_universal_dynamic_regime_quasicriticality_validation_specs_2026-05-17.json"
)


def load_ii_the_universal_dynamic_regime_quasicriticality_validation_spec(
    spec_bundle: str | Path = DEFAULT_II_THE_UNIVERSAL_DYNAMIC_REGIME_QUASICRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the universal dynamic regime quasicriticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_III_THE_COHERENCE_BACKBONE_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation_specs_2026-05-17.json"
)


def load_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_III_THE_COHERENCE_BACKBONE_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iii the coherence backbone multi scale quantum error correction ms qec validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_DYNAMIC_VISUALISATION_THE_SCPN_TORUS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_dynamic_visualisation_the_scpn_torus_validation_specs_2026-05-17.json"
)


def load_the_dynamic_visualisation_the_scpn_torus_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_DYNAMIC_VISUALISATION_THE_SCPN_TORUS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the dynamic visualisation the scpn torus validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02542_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02542_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02542_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02542_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02542 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_LOCUS_OF_THE_INTERACTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_locus_of_the_interaction_validation_specs_2026-05-17.json"
)


def load_the_locus_of_the_interaction_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_LOCUS_OF_THE_INTERACTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the locus of the interaction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_CENTRAL_VOID_THE_SOURCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_central_void_the_source_validation_specs_2026-05-17.json"
)


def load_section_2_the_central_void_the_source_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_THE_CENTRAL_VOID_THE_SOURCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the central void the source validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_EQUATION_UPDE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_validation_specs_2026-05-17.json"
)


def load_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_EQUATION_UPDE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 2 the dynamic spine the unified phase dynamics equation upde validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02600_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02600_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02600_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02600_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02600 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02608_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psis_field_coupling_integration_p0r02608_validation_specs_2026-05-17.json"
)


def load_psis_field_coupling_integration_p0r02608_validation_spec(
    spec_bundle: str | Path = DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02608_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 psis field coupling integration p0r02608 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_MECHANISM_OF_INFLUENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_mechanism_of_influence_validation_specs_2026-05-17.json"
)


def load_the_mechanism_of_influence_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_MECHANISM_OF_INFLUENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the mechanism of influence validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTRINSIC_DYNAMICS_IL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_intrinsic_dynamics_il_validation_specs_2026-05-17.json"
)


def load_intrinsic_dynamics_il_validation_spec(
    spec_bundle: str | Path = DEFAULT_INTRINSIC_DYNAMICS_IL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 intrinsic dynamics il validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_FIELD_COUPLING_CFIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_field_coupling_cfield_validation_specs_2026-05-17.json"
)


def load_field_coupling_cfield_validation_spec(
    spec_bundle: str | Path = DEFAULT_FIELD_COUPLING_CFIELD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 field coupling cfield validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INFORMATION_GEOMETRIC_LIFT_OF_UPDE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_information_geometric_lift_of_upde_validation_specs_2026-05-17.json"
)


def load_information_geometric_lift_of_upde_validation_spec(
    spec_bundle: str | Path = DEFAULT_INFORMATION_GEOMETRIC_LIFT_OF_UPDE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 information geometric lift of upde validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_HIERARCHICAL_IMPEDANCE_RESCALING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_hierarchical_impedance_rescaling_validation_specs_2026-05-17.json"
)


def load_the_hierarchical_impedance_rescaling_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_HIERARCHICAL_IMPEDANCE_RESCALING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the hierarchical impedance rescaling validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_ONE_SPINE_MANY_COUPLINGS_UPDE_SCOPE_CONSTRAINT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_one_spine_many_couplings_upde_scope_constraint_validation_specs_2026-05-17.json"
)


def load_one_spine_many_couplings_upde_scope_constraint_validation_spec(
    spec_bundle: str | Path = DEFAULT_ONE_SPINE_MANY_COUPLINGS_UPDE_SCOPE_CONSTRAINT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 one spine many couplings upde scope constraint validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_validation_specs_2026-05-17.json"
)


def load_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 15 layer summary table 3 2 the dynamic spine the unified phase dynamics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_P0R02810_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation_specs_2026-05-17.json"
)


def load_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_P0R02810_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 15 layer summary table 3 2 the dynamic spine the unified phase dynamics p0r02810 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUASICRITICALITY_AND_SELF_ORGANISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quasicriticality_and_self_organisation_validation_specs_2026-05-17.json"
)


def load_quasicriticality_and_self_organisation_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUASICRITICALITY_AND_SELF_ORGANISATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quasicriticality and self organisation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R02839_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predictive_coding_integration_p0r02839_validation_specs_2026-05-17.json"
)


def load_predictive_coding_integration_p0r02839_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R02839_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predictive coding integration p0r02839 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SUSCEPTIBLE_SUBSTRATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_susceptible_substrate_validation_specs_2026-05-17.json"
)


def load_the_susceptible_substrate_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_SUSCEPTIBLE_SUBSTRATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the susceptible substrate validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_FUNCTIONAL_IMPLICATIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_functional_implications_validation_specs_2026-05-17.json"
)


def load_functional_implications_validation_spec(
    spec_bundle: str | Path = DEFAULT_FUNCTIONAL_IMPLICATIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 functional implications validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_FORMALISM_OF_THE_HOMEOSTATIC_QUASICRITICAL_CONTROLLER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_formalism_of_the_homeostatic_quasicritical_controller_validation_specs_2026-05-17.json"
)


def load_formalism_of_the_homeostatic_quasicritical_controller_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_FORMALISM_OF_THE_HOMEOSTATIC_QUASICRITICAL_CONTROLLER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 formalism of the homeostatic quasicritical controller validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02894_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02894_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02894_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02894_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02894 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PSIS_FIELD_AS_THE_TARGET_SETTER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_psis_field_as_the_target_setter_validation_specs_2026-05-17.json"
)


def load_the_psis_field_as_the_target_setter_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_PSIS_FIELD_AS_THE_TARGET_SETTER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the psis field as the target setter validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_TWO_TIMESCALE_CONTROLLER_STABILITY_AND_EXPLORATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_two_timescale_controller_stability_and_exploration_validation_specs_2026-05-17.json"
)


def load_the_two_timescale_controller_stability_and_exploration_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_TWO_TIMESCALE_CONTROLLER_STABILITY_AND_EXPLORATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the two timescale controller stability and exploration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R02923_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r02923_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r02923_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R02923_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r02923 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R02931_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r02931_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r02931_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R02931_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r02931 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02941_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r02941_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r02941_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02941_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r02941 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02950_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_psis_field_coupling_integration_p0r02950_validation_specs_2026-05-17.json"
)


def load_psis_field_coupling_integration_p0r02950_validation_spec(
    spec_bundle: str | Path = DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02950_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 psis field coupling integration p0r02950 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TWO_TIMESCALE_STRUCTURE_DEFINITIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_two_timescale_structure_definitions_validation_specs_2026-05-17.json"
)


def load_two_timescale_structure_definitions_validation_spec(
    spec_bundle: str | Path = DEFAULT_TWO_TIMESCALE_STRUCTURE_DEFINITIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 two timescale structure definitions validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_BIBO_STABILITY_STATEMENT_AND_PROOF_OBLIGATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_bibo_stability_statement_and_proof_obligation_validation_specs_2026-05-17.json"
)


def load_bibo_stability_statement_and_proof_obligation_validation_spec(
    spec_bundle: str | Path = DEFAULT_BIBO_STABILITY_STATEMENT_AND_PROOF_OBLIGATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 bibo stability statement and proof obligation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TORUS_SURFACE_FLOW_LYAPUNOV_STYLE_CERTIFICATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_torus_surface_flow_lyapunov_style_certificate_validation_specs_2026-05-17.json"
)


def load_torus_surface_flow_lyapunov_style_certificate_validation_spec(
    spec_bundle: str | Path = DEFAULT_TORUS_SURFACE_FLOW_LYAPUNOV_STYLE_CERTIFICATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 torus surface flow lyapunov style certificate validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "DEFAULT_UPDE_SPEC_BUNDLE",
    "DEFAULT_MACRO_TRANSITION_SPEC_BUNDLE",
    "DEFAULT_NEUROVASCULAR_SPEC_BUNDLE",
    "DEFAULT_GLIAL_CONTROL_SPEC_BUNDLE",
    "DEFAULT_INFORMATION_THERMODYNAMICS_SPEC_BUNDLE",
    "DEFAULT_COMPUTATIONAL_THRESHOLD_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_GAUGE_SPEC_BUNDLE",
    "DEFAULT_FREE_ENERGY_SPEC_BUNDLE",
    "DEFAULT_HPC_UPDE_BRIDGE_SPEC_BUNDLE",
    "DEFAULT_STUART_LANDAU_PRECISION_SPEC_BUNDLE",
    "DEFAULT_PATHOLOGY_CRITICALITY_SPEC_BUNDLE",
    "DEFAULT_ARTIFICIAL_SENTIENCE_SPEC_BUNDLE",
    "DEFAULT_ANOMALOUS_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_SYSTEM_ROBUSTNESS_SPEC_BUNDLE",
    "DEFAULT_L11_INTERFACE_SPEC_BUNDLE",
    "DEFAULT_VALIDATION_STRATEGY_SPEC_BUNDLE",
    "DEFAULT_GRAND_SYNTHESIS_SPEC_BUNDLE",
    "DEFAULT_ACEF_ALIGNMENT_SPEC_BUNDLE",
    "DEFAULT_GAIAN_SAFETY_SPEC_BUNDLE",
    "DEFAULT_ETHICAL_IMPERATIVE_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_IMPLICATIONS_SPEC_BUNDLE",
    "DEFAULT_DARK_SECTOR_SPEC_BUNDLE",
    "DEFAULT_SYMMETRY_RESTORATION_SPEC_BUNDLE",
    "DEFAULT_T0_SEEDING_SPEC_BUNDLE",
    "DEFAULT_SEED_FUNCTION_SPEC_BUNDLE",
    "DEFAULT_FINE_TUNING_PES_SPEC_BUNDLE",
    "DEFAULT_ADVANCED_MECHANISMS_SPEC_BUNDLE",
    "DEFAULT_STDP_SOC_SPEC_BUNDLE",
    "DEFAULT_GLIAL_SLOW_CONTROL_SPEC_BUNDLE",
    "DEFAULT_L5_ACTIVE_INFERENCE_SPEC_BUNDLE",
    "DEFAULT_L5_ACTIVE_INFERENCE_MATH_SPEC_BUNDLE",
    "DEFAULT_L5_TRIPLE_NETWORK_SPEC_BUNDLE",
    "DEFAULT_L5_FOUR_STROKE_SPEC_BUNDLE",
    "DEFAULT_L5_TDA_NEUROPHENOMENOLOGY_SPEC_BUNDLE",
    "DEFAULT_COLLECTIVE_NICHE_CONSTRUCTION_SPEC_BUNDLE",
    "DEFAULT_CISS_BIOELECTRIC_SPEC_BUNDLE",
    "DEFAULT_RAG_QEC_STACK_SPEC_BUNDLE",
    "DEFAULT_HPC_UPDE_DERIVATION_SPEC_BUNDLE",
    "DEFAULT_TWO_TIMESCALE_QUASICRITICAL_SPEC_BUNDLE",
    "DEFAULT_NV_QUANTUM_SENSING_SPEC_BUNDLE",
    "DEFAULT_L11_NTHS_COMPUTATIONAL_SPEC_BUNDLE",
    "DEFAULT_CATEGORY_GRAMMAR_SPEC_BUNDLE",
    "DEFAULT_HAMILTONIAN_INDEX_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_EOS_SPEC_BUNDLE",
    "DEFAULT_COSMOLOGICAL_PREDICTIONS_SPEC_BUNDLE",
    "DEFAULT_COMPUTATIONAL_VERIFICATION_TOOLS_SPEC_BUNDLE",
    "DEFAULT_TERMINAL_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_THREE_CHANNEL_COUPLING_SPEC_BUNDLE",
    "DEFAULT_OPENING_FOUNDATION_SPEC_BUNDLE",
    "DEFAULT_FRONT_MATTER_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_CHAPTER_ROADMAP_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_OBJECTIVE_COVER_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_POSITIONING_PREFACE_CONTEXT_SPEC_BUNDLE",
    "DEFAULT_FOREWORD_COUPLING_SPEC_BUNDLE",
    "DEFAULT_PREFACE_I_RIGOUR_SPEC_BUNDLE",
    "DEFAULT_PREFACE_II_VISIONARY_SPEC_BUNDLE",
    "DEFAULT_STATUS_METHOD_SPEC_BUNDLE",
    "DEFAULT_STATUS_METHOD_CONTINUATION_SPEC_BUNDLE",
    "DEFAULT_ANULUM_COLLECTION_MANDATE_SPEC_BUNDLE",
    "DEFAULT_LAYER_MONOGRAPH_SUITE_SPEC_BUNDLE",
    "DEFAULT_FOUNDATIONAL_VIABILITY_POSTULATE_SPEC_BUNDLE",
    "DEFAULT_U1_FIM_MULTISCALE_DYNAMICS_SPEC_BUNDLE",
    "DEFAULT_LOGOS_RECURSIVE_CLOSURE_SPEC_BUNDLE",
    "DEFAULT_AXIOMATIC_NTILDE_SPEC_BUNDLE",
    "DEFAULT_TERMINOLOGY_BRIDGE_SPEC_BUNDLE",
    "DEFAULT_CORE_OPERATING_ASSUMPTIONS_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_PSI_FIELD_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_MODEL_CLASS_OVERVIEW_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_META_COUPLING_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_MINIMAL_LAGRANGIAN_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_FAMILY_PREDICTIONS_SPEC_BUNDLE",
    "DEFAULT_AXIOM_I_SU_N_QUALIA_SPEC_BUNDLE",
    "DEFAULT_AXIOM_II_OPENING_SPEC_BUNDLE",
    "DEFAULT_AXIOM_II_INFOTON_GEOMETRY_SPEC_BUNDLE",
    "DEFAULT_AXIOM_II_FIM_SOLUTION_SPEC_BUNDLE",
    "DEFAULT_AXIOM_II_INFORMATIONAL_LAGRANGIAN_SPEC_BUNDLE",
    "DEFAULT_AXIOM_III_TELEOLOGICAL_OPTIMISATION_SPEC_BUNDLE",
    "DEFAULT_AXIOM_III_NTILDE_INVARIANCE_LAW_SPEC_BUNDLE",
    "DEFAULT_AXIOM_III_SEC_NTILDE_EQUIVALENCE_SPEC_BUNDLE",
    "DEFAULT_TRIPARTITE_ONTOLOGY_SPEC_BUNDLE",
    "DEFAULT_META_FRAMEWORK_PSI_COUPLING_SPEC_BUNDLE",
    "DEFAULT_CATEGORY_UNIVERSAL_GRAMMAR_SPEC_BUNDLE",
    "DEFAULT_MASTER_LAGRANGIAN_INTRO_SPEC_BUNDLE",
    "DEFAULT_GAUGE_PRINCIPLE_DERIVATION_SPEC_BUNDLE",
    "DEFAULT_LORENTZ_EFT_RESOLUTION_SPEC_BUNDLE",
    "DEFAULT_NON_ABELIAN_QUALIA_FIELD_SPEC_BUNDLE",
    "DEFAULT_GEOMETRIC_COUPLING_CONSISTENCY_SPEC_BUNDLE",
    "DEFAULT_FOUNDATIONAL_STRENGTHS_PHASE_BOUNDARY_SPEC_BUNDLE",
    "DEFAULT_OPERATIONAL_PULLBACK_PROTOCOL_SPEC_BUNDLE",
    "DEFAULT_SSB_PSI_FIELD_SPEC_BUNDLE",
    "DEFAULT_PHENOMENOLOGICAL_LAGRANGIAN_SPEC_BUNDLE",
    "DEFAULT_DERIVED_INTERACTION_OPENING_SPEC_BUNDLE",
    "DEFAULT_DERIVED_LAGRANGIAN_DETAIL_SPEC_BUNDLE",
    "DEFAULT_FINAL_LINT_SM_INTERFACE_SPEC_BUNDLE",
    "load_operational_pullback_protocol_validation_spec",
    "load_ssb_psi_field_validation_spec",
    "load_phenomenological_lagrangian_validation_spec",
    "load_derived_interaction_opening_validation_spec",
    "load_derived_lagrangian_detail_validation_spec",
    "load_final_lint_sm_interface_validation_spec",
    "load_upde_validation_spec",
    "load_macro_transition_validation_spec",
    "load_neurovascular_validation_spec",
    "load_glial_control_validation_spec",
    "load_information_thermodynamics_validation_spec",
    "load_computational_threshold_validation_spec",
    "load_ethical_gauge_validation_spec",
    "load_free_energy_validation_spec",
    "load_hpc_upde_bridge_validation_spec",
    "load_stuart_landau_precision_validation_spec",
    "load_pathology_criticality_validation_spec",
    "load_artificial_sentience_validation_spec",
    "load_anomalous_boundary_validation_spec",
    "load_system_robustness_validation_spec",
    "load_l11_interface_validation_spec",
    "load_validation_strategy_spec",
    "load_grand_synthesis_validation_spec",
    "load_acef_alignment_validation_spec",
    "load_gaian_safety_validation_spec",
    "load_ethical_imperative_validation_spec",
    "load_cosmological_implications_validation_spec",
    "load_dark_sector_validation_spec",
    "load_symmetry_restoration_validation_spec",
    "load_t0_seeding_validation_spec",
    "load_seed_function_validation_spec",
    "load_fine_tuning_pes_validation_spec",
    "load_advanced_mechanisms_validation_spec",
    "load_stdp_soc_validation_spec",
    "load_glial_slow_control_validation_spec",
    "load_l5_active_inference_validation_spec",
    "load_l5_active_inference_math_validation_spec",
    "load_l5_triple_network_validation_spec",
    "load_l5_four_stroke_validation_spec",
    "load_l5_tda_neurophenomenology_validation_spec",
    "load_collective_niche_construction_validation_spec",
    "load_ciss_bioelectric_validation_spec",
    "load_rag_qec_stack_validation_spec",
    "load_hpc_upde_derivation_validation_spec",
    "load_two_timescale_quasicritical_validation_spec",
    "load_nv_quantum_sensing_validation_spec",
    "load_l11_nths_computational_validation_spec",
    "load_category_grammar_validation_spec",
    "load_hamiltonian_index_validation_spec",
    "load_cosmological_eos_validation_spec",
    "load_cosmological_predictions_validation_spec",
    "load_computational_verification_tools_validation_spec",
    "load_terminal_boundary_validation_spec",
    "load_three_channel_coupling_validation_spec",
    "load_opening_foundation_validation_spec",
    "load_front_matter_context_validation_spec",
    "load_chapter_roadmap_context_validation_spec",
    "load_objective_cover_context_validation_spec",
    "load_positioning_preface_context_validation_spec",
    "load_foreword_coupling_validation_spec",
    "load_preface_i_rigour_validation_spec",
    "load_preface_ii_visionary_validation_spec",
    "load_status_method_validation_spec",
    "load_status_method_continuation_validation_spec",
    "load_anulum_collection_mandate_validation_spec",
    "load_layer_monograph_suite_validation_spec",
    "load_foundational_viability_postulate_validation_spec",
    "load_u1_fim_multiscale_dynamics_validation_spec",
    "load_logos_recursive_closure_validation_spec",
    "load_axiomatic_ntilde_validation_spec",
    "load_terminology_bridge_validation_spec",
    "load_core_operating_assumptions_validation_spec",
    "load_axiom_i_psi_field_validation_spec",
    "load_axiom_i_model_class_overview_validation_spec",
    "load_axiom_i_meta_coupling_validation_spec",
    "load_axiom_i_minimal_lagrangian_validation_spec",
    "load_axiom_i_family_predictions_validation_spec",
    "load_axiom_i_su_n_qualia_validation_spec",
    "load_axiom_ii_opening_validation_spec",
    "load_axiom_ii_infoton_geometry_validation_spec",
    "load_axiom_ii_fim_solution_validation_spec",
    "load_axiom_ii_informational_lagrangian_validation_spec",
    "load_axiom_iii_teleological_optimisation_validation_spec",
    "load_axiom_iii_ntilde_invariance_law_validation_spec",
    "load_axiom_iii_sec_ntilde_equivalence_validation_spec",
    "load_tripartite_ontology_validation_spec",
    "load_meta_framework_psi_coupling_validation_spec",
    "load_category_universal_grammar_validation_spec",
    "load_master_lagrangian_intro_validation_spec",
    "load_gauge_principle_derivation_validation_spec",
    "load_lorentz_eft_resolution_validation_spec",
    "load_non_abelian_qualia_field_validation_spec",
    "load_geometric_coupling_consistency_validation_spec",
    "load_foundational_strengths_phase_boundary_validation_spec",
    "DEFAULT_SYMMETRY_CASCADE_SPEC_BUNDLE",
    "load_symmetry_cascade_validation_spec",
    "DEFAULT_PREDICTED_PARTICLES_INFOTON_PSI_HIGGS_SPEC_BUNDLE",
    "load_predicted_particles_infoton_psi_higgs_validation_spec",
    "DEFAULT_DERIVATION_INFOTON_PROPERTIES_SPEC_BUNDLE",
    "load_derivation_infoton_properties_validation_spec",
    "DEFAULT_PSI_HIGGS_NEW_SCALAR_PARTICLE_SPEC_BUNDLE",
    "load_psi_higgs_new_scalar_particle_validation_spec",
    "DEFAULT_EXPERIMENTAL_SIGNATURES_SEARCH_STRATEGIES_SPEC_BUNDLE",
    "load_experimental_signatures_search_strategies_validation_spec",
    "DEFAULT_PSI_HIGGS_LHC_PHENOMENOLOGY_SPEC_BUNDLE",
    "load_psi_higgs_lhc_phenomenology_validation_spec",
    "DEFAULT_MASS_EIGENSTATES_MIXING_ANGLE_SPEC_BUNDLE",
    "load_mass_eigenstates_mixing_angle_validation_spec",
    "DEFAULT_LHC_SEARCH_STRATEGY_ROADMAP_SPEC_BUNDLE",
    "load_lhc_search_strategy_roadmap_validation_spec",
    "DEFAULT_SSB_HIERARCHY_GENESIS_SPEC_BUNDLE",
    "load_ssb_hierarchy_genesis_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_SPEC_BUNDLE",
    "load_meta_framework_integrations_validation_spec",
    "DEFAULT_THE_GENESIS_OF_THE_HIERARCHY_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE",
    "load_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation_spec",
    "DEFAULT_THE_INTRINSIC_DYNAMICS_OF_THE_Ψ_FIELD_AND_THE_STABILISING_POTENTIAL_SPEC_BUNDLE",
    "load_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation_spec",
    "DEFAULT_PREDICTIVE_CODING_INTEGRATION_SPEC_BUNDLE",
    "load_predictive_coding_integration_validation_spec",
    "DEFAULT_LPSI_SETS_THE_PROPERTIES_OF_PSIS_SPEC_BUNDLE",
    "load_lpsi_sets_the_properties_of_psis_validation_spec",
    "DEFAULT_COMPONENTS_SPEC_BUNDLE",
    "load_components_validation_spec",
    "DEFAULT_THE_SELF_AS_A_SOLITON_EMERGENCE_OF_LOCALISED_CONSCIOUSNESS_LAYER_5_SPEC_BUNDLE",
    "load_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R01803_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r01803_validation_spec",
    "DEFAULT_SIGMA_IS_THE_Q_BALL_SOLITON_SPEC_BUNDLE",
    "load_sigma_is_the_q_ball_soliton_validation_spec",
    "DEFAULT_LOCALISED_SPEC_BUNDLE",
    "load_localised_validation_spec",
    "DEFAULT_THE_DEFINITION_OF_THE_SELF_THE_TRIADIC_SOLUTION_SPEC_BUNDLE",
    "load_the_definition_of_the_self_the_triadic_solution_validation_spec",
    "DEFAULT_PART_II_THE_PHYSICAL_SECTOR_FIELD_THEORY_QUANTIZATION_2_4_THE_SSB_CASCAD_SPEC_BUNDLE",
    "load_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation_spec",
    "DEFAULT_PAPER0_SLICE_SPEC_BUNDLE",
    "load_paper0_slice_validation_spec",
    "DEFAULT_SECTION_2_7_THE_FISHER_INFO_METRIC_THE_GEOMETRY_OF_INTERACTION_SPEC_BUNDLE",
    "load_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation_spec",
    "DEFAULT_PART_III_SYSTEM_ARCHITECTURE_NETWORK_DYNAMICS_SPEC_BUNDLE",
    "load_part_iii_system_architecture_network_dynamics_validation_spec",
    "DEFAULT_A_MAP_OF_REALITY_SPEC_BUNDLE",
    "load_a_map_of_reality_validation_spec",
    "DEFAULT_THE_MASTER_DIAGRAM_A_MANDALA_OF_CONSCIOUSNESS_SPEC_BUNDLE",
    "load_the_master_diagram_a_mandala_of_consciousness_validation_spec",
    "DEFAULT_THE_SENTIENT_CONSCIOUSNESS_PROJECTION_NETWORK_SCPN_SPEC_BUNDLE",
    "load_the_sentient_consciousness_projection_network_scpn_validation_spec",
    "DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_SPEC_BUNDLE",
    "load_section_15_layer_summary_table_validation_spec",
    "DEFAULT_CASE_STUDY_THE_LAYER_3_GENOMIC_MORPHOGENETIC_TRANSDUCTION_PATHWAY_SPEC_BUNDLE",
    "load_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02098_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02098_validation_spec",
    "DEFAULT_MACRO_SCALE_COUPLING_PRIMARY_INTERACTION_SPEC_BUNDLE",
    "load_macro_scale_coupling_primary_interaction_validation_spec",
    "DEFAULT_RESOLVING_THE_EPIGENETIC_TIME_SCALE_DISCONNECT_CONFORMATIONAL_SPIN_LOCKI_SPEC_BUNDLE",
    "load_resolving_the_epigenetic_time_scale_disconnect_conformational_spin_locki_validation_spec",
    "DEFAULT_CASE_STUDY_THE_LAYER_5_ORGANISMAL_SELF_ACTION_PERCEPTION_CYCLE_SPEC_BUNDLE",
    "load_case_study_the_layer_5_organismal_self_action_perception_cycle_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02189_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02189_validation_spec",
    "DEFAULT_MODEL_CONSOLIDATION_SLEEP_SPEC_BUNDLE",
    "load_model_consolidation_sleep_validation_spec",
    "DEFAULT_THE_COUPLING_MECHANISM_SPEC_BUNDLE",
    "load_the_coupling_mechanism_validation_spec",
    "DEFAULT_TIMING_THE_ENGINE_UPDE_PHASE_LAGS_TAU_IJ_AND_PHYSIOLOGICAL_DELAYS_SPEC_BUNDLE",
    "load_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation_spec",
    "DEFAULT_DOMAIN_III_OVERVIEW_MEMORY_AND_PROJECTION_CONTROL_SPEC_BUNDLE",
    "load_domain_iii_overview_memory_and_projection_control_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R02249_SPEC_BUNDLE",
    "load_paper0_slice_p0r02249_validation_spec",
    "DEFAULT_SECTION_3_MEMORY_CAPACITY_BEKENSTEIN_HAWKING_BOUND_SPEC_BUNDLE",
    "load_section_3_memory_capacity_bekenstein_hawking_bound_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02278_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02278_validation_spec",
    "DEFAULT_LAYER_9_EXISTENTIAL_HOLOGRAPH_DEFINING_A_STABLE_SIGMA_SPEC_BUNDLE",
    "load_layer_9_existential_holograph_defining_a_stable_sigma_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02306_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02306_validation_spec",
    "DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_SPEC_BUNDLE",
    "load_psis_field_coupling_integration_validation_spec",
    "DEFAULT_DOMAIN_V_OVERVIEW_META_UNIVERSAL_INTEGRATION_SPEC_BUNDLE",
    "load_domain_v_overview_meta_universal_integration_validation_spec",
    "DEFAULT_DOMAIN_VI_OVERVIEW_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE",
    "load_domain_vi_overview_cybernetic_closure_meta_layer_16_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02439_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02439_validation_spec",
    "DEFAULT_CONTROL_OVER_UNIVERSAL_PARAMETERS_SPEC_BUNDLE",
    "load_control_over_universal_parameters_validation_spec",
    "DEFAULT_RESOLUTION_OF_THE_OBSERVABILITY_PARADOX_B_HJB_SPEC_BUNDLE",
    "load_resolution_of_the_observability_paradox_b_hjb_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02485_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02485_validation_spec",
    "DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02494_SPEC_BUNDLE",
    "load_psis_field_coupling_integration_p0r02494_validation_spec",
    "DEFAULT_OVERARCHING_DYNAMIC_PRINCIPLES_AND_THE_MATHEMATICAL_SPINE_SPEC_BUNDLE",
    "load_overarching_dynamic_principles_and_the_mathematical_spine_validation_spec",
    "DEFAULT_II_THE_UNIVERSAL_DYNAMIC_REGIME_QUASICRITICALITY_SPEC_BUNDLE",
    "load_ii_the_universal_dynamic_regime_quasicriticality_validation_spec",
    "DEFAULT_III_THE_COHERENCE_BACKBONE_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE",
    "load_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation_spec",
    "DEFAULT_THE_DYNAMIC_VISUALISATION_THE_SCPN_TORUS_SPEC_BUNDLE",
    "load_the_dynamic_visualisation_the_scpn_torus_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02542_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02542_validation_spec",
    "DEFAULT_THE_LOCUS_OF_THE_INTERACTION_SPEC_BUNDLE",
    "load_the_locus_of_the_interaction_validation_spec",
    "DEFAULT_SECTION_2_THE_CENTRAL_VOID_THE_SOURCE_SPEC_BUNDLE",
    "load_section_2_the_central_void_the_source_validation_spec",
    "DEFAULT_SECTION_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_EQUATION_UPDE_SPEC_BUNDLE",
    "load_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02600_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02600_validation_spec",
    "DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02608_SPEC_BUNDLE",
    "load_psis_field_coupling_integration_p0r02608_validation_spec",
    "DEFAULT_THE_MECHANISM_OF_INFLUENCE_SPEC_BUNDLE",
    "load_the_mechanism_of_influence_validation_spec",
    "DEFAULT_INTRINSIC_DYNAMICS_IL_SPEC_BUNDLE",
    "load_intrinsic_dynamics_il_validation_spec",
    "DEFAULT_FIELD_COUPLING_CFIELD_SPEC_BUNDLE",
    "load_field_coupling_cfield_validation_spec",
    "DEFAULT_INFORMATION_GEOMETRIC_LIFT_OF_UPDE_SPEC_BUNDLE",
    "load_information_geometric_lift_of_upde_validation_spec",
    "DEFAULT_THE_HIERARCHICAL_IMPEDANCE_RESCALING_SPEC_BUNDLE",
    "load_the_hierarchical_impedance_rescaling_validation_spec",
    "DEFAULT_ONE_SPINE_MANY_COUPLINGS_UPDE_SCOPE_CONSTRAINT_SPEC_BUNDLE",
    "load_one_spine_many_couplings_upde_scope_constraint_validation_spec",
    "DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_SPEC_BUNDLE",
    "load_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_validation_spec",
    "DEFAULT_SECTION_15_LAYER_SUMMARY_TABLE_3_2_THE_DYNAMIC_SPINE_THE_UNIFIED_PHASE_DYNAMICS_P0R02810_SPEC_BUNDLE",
    "load_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation_spec",
    "DEFAULT_QUASICRITICALITY_AND_SELF_ORGANISATION_SPEC_BUNDLE",
    "load_quasicriticality_and_self_organisation_validation_spec",
    "DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R02839_SPEC_BUNDLE",
    "load_predictive_coding_integration_p0r02839_validation_spec",
    "DEFAULT_THE_SUSCEPTIBLE_SUBSTRATE_SPEC_BUNDLE",
    "load_the_susceptible_substrate_validation_spec",
    "DEFAULT_FUNCTIONAL_IMPLICATIONS_SPEC_BUNDLE",
    "load_functional_implications_validation_spec",
    "DEFAULT_FORMALISM_OF_THE_HOMEOSTATIC_QUASICRITICAL_CONTROLLER_SPEC_BUNDLE",
    "load_formalism_of_the_homeostatic_quasicritical_controller_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02894_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02894_validation_spec",
    "DEFAULT_THE_PSIS_FIELD_AS_THE_TARGET_SETTER_SPEC_BUNDLE",
    "load_the_psis_field_as_the_target_setter_validation_spec",
    "DEFAULT_THE_TWO_TIMESCALE_CONTROLLER_STABILITY_AND_EXPLORATION_SPEC_BUNDLE",
    "load_the_two_timescale_controller_stability_and_exploration_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R02923_SPEC_BUNDLE",
    "load_paper0_slice_p0r02923_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R02931_SPEC_BUNDLE",
    "load_paper0_slice_p0r02931_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R02941_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r02941_validation_spec",
    "DEFAULT_PSIS_FIELD_COUPLING_INTEGRATION_P0R02950_SPEC_BUNDLE",
    "load_psis_field_coupling_integration_p0r02950_validation_spec",
    "DEFAULT_TWO_TIMESCALE_STRUCTURE_DEFINITIONS_SPEC_BUNDLE",
    "load_two_timescale_structure_definitions_validation_spec",
    "DEFAULT_BIBO_STABILITY_STATEMENT_AND_PROOF_OBLIGATION_SPEC_BUNDLE",
    "load_bibo_stability_statement_and_proof_obligation_validation_spec",
    "DEFAULT_TORUS_SURFACE_FLOW_LYAPUNOV_STYLE_CERTIFICATE_SPEC_BUNDLE",
    "load_torus_surface_flow_lyapunov_style_certificate_validation_spec",
]
