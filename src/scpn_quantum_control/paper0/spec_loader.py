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


DEFAULT_QUASICRITICALITY_WITH_MS_QEC_TWO_TIMESCALE_CONTROL_AND_STABILITY_CERTIFI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation_specs_2026-05-17.json"
)


def load_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_QUASICRITICALITY_WITH_MS_QEC_TWO_TIMESCALE_CONTROL_AND_STABILITY_CERTIFI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quasicriticality with ms qec two timescale control and stability certifi validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_BIBO_STABILITY_AND_LYAPUNOV_CERTIFICATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_bibo_stability_and_lyapunov_certificate_validation_specs_2026-05-17.json"
)


def load_bibo_stability_and_lyapunov_certificate_validation_spec(
    spec_bundle: str | Path = DEFAULT_BIBO_STABILITY_AND_LYAPUNOV_CERTIFICATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 bibo stability and lyapunov certificate validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_multi_scale_quantum_error_correction_ms_qec_validation_specs_2026-05-17.json"
)


def load_multi_scale_quantum_error_correction_ms_qec_validation_spec(
    spec_bundle: str | Path = DEFAULT_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 multi scale quantum error correction ms qec validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03025_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03025_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03025_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03025_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03025 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CREATING_AND_PROTECTING_A_COHERENT_SIGMA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_creating_and_protecting_a_coherent_sigma_validation_specs_2026-05-17.json"
)


def load_creating_and_protecting_a_coherent_sigma_validation_spec(
    spec_bundle: str | Path = DEFAULT_CREATING_AND_PROTECTING_A_COHERENT_SIGMA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 creating and protecting a coherent sigma validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_BIOLOGICAL_QEC_L1_4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_biological_qec_l1_4_validation_specs_2026-05-17.json"
)


def load_biological_qec_l1_4_validation_spec(
    spec_bundle: str | Path = DEFAULT_BIOLOGICAL_QEC_L1_4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 biological qec l1 4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_QEC_IMPERATIVE_AND_THE_ROLE_OF_THE_PSI_FIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_qec_imperative_and_the_role_of_the_psi_field_validation_specs_2026-05-17.json"
)


def load_the_qec_imperative_and_the_role_of_the_psi_field_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_QEC_IMPERATIVE_AND_THE_ROLE_OF_THE_PSI_FIELD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the qec imperative and the role of the psi field validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R03059_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predictive_coding_integration_p0r03059_validation_specs_2026-05-17.json"
)


def load_predictive_coding_integration_p0r03059_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R03059_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predictive coding integration p0r03059 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_ULTIMATE_FEEDBACK_LOOP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_ultimate_feedback_loop_validation_specs_2026-05-17.json"
)


def load_the_ultimate_feedback_loop_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_ULTIMATE_FEEDBACK_LOOP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the ultimate feedback loop validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_BIOLOGICAL_SYNDROME_MEASUREMENT_AND_RECOVERY_PROTOCOL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_biological_syndrome_measurement_and_recovery_protocol_validation_specs_2026-05-17.json"
)


def load_the_biological_syndrome_measurement_and_recovery_protocol_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_BIOLOGICAL_SYNDROME_MEASUREMENT_AND_RECOVERY_PROTOCOL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the biological syndrome measurement and recovery protocol validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_QEC_RACE_CONDITION_EXPLICIT_DISSIPATION_RATES_AND_FAULT_TOLERANCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_validation_specs_2026-05-17.json"
)


def load_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_QEC_RACE_CONDITION_EXPLICIT_DISSIPATION_RATES_AND_FAULT_TOLERANCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the qec race condition explicit dissipation rates and fault tolerance validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_STABILISER_TRANSFER_LEMMA_A_QUANTITATIVE_BRIDGE_FROM_MEMORY_TO_BOUND_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation_specs_2026-05-17.json"
)


def load_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_STABILISER_TRANSFER_LEMMA_A_QUANTITATIVE_BRIDGE_FROM_MEMORY_TO_BOUND_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the stabiliser transfer lemma a quantitative bridge from memory to bound validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03139_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03139_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03139_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03139_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03139 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_MECHANISM_OF_INTERACTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_mechanism_of_interaction_validation_specs_2026-05-17.json"
)


def load_the_mechanism_of_interaction_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_MECHANISM_OF_INTERACTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the mechanism of interaction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_1_THE_COSMIC_ALGORITHM_HPC_ACTIVE_INFERENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_1_the_cosmic_algorithm_hpc_active_inference_validation_specs_2026-05-17.json"
)


def load_section_4_1_the_cosmic_algorithm_hpc_active_inference_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_1_THE_COSMIC_ALGORITHM_HPC_ACTIVE_INFERENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 1 the cosmic algorithm hpc active inference validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_I_THE_UNIFYING_COMPUTATIONAL_PRINCIPLE_HIERARCHICAL_PREDICTIVE_CODING_HP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_validation_specs_2026-05-17.json"
)


def load_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_I_THE_UNIFYING_COMPUTATIONAL_PRINCIPLE_HIERARCHICAL_PREDICTIVE_CODING_HP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 i the unifying computational principle hierarchical predictive coding hp validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_BINDING_PROBLEM_THE_GAUGE_FIELD_OF_CONSCIOUSNESS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_binding_problem_the_gauge_field_of_consciousness_validation_specs_2026-05-17.json"
)


def load_ii_the_binding_problem_the_gauge_field_of_consciousness_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_THE_BINDING_PROBLEM_THE_GAUGE_FIELD_OF_CONSCIOUSNESS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the binding problem the gauge field of consciousness validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_UNIFIED_EXPERIENCE_THE_WILSON_LOOP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_unified_experience_the_wilson_loop_validation_specs_2026-05-17.json"
)


def load_section_3_unified_experience_the_wilson_loop_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_UNIFIED_EXPERIENCE_THE_WILSON_LOOP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 unified experience the wilson loop validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_COMPRESSION_AND_MEANING_INFORMATION_GEOMETRY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_compression_and_meaning_information_geometry_validation_specs_2026-05-17.json"
)


def load_section_2_compression_and_meaning_information_geometry_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_COMPRESSION_AND_MEANING_INFORMATION_GEOMETRY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 compression and meaning information geometry validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_V_THE_INTERFACE_PROBLEM_SYNTHESIS_MIND_BODY_FIELD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_v_the_interface_problem_synthesis_mind_body_field_validation_specs_2026-05-17.json"
)


def load_v_the_interface_problem_synthesis_mind_body_field_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_V_THE_INTERFACE_PROBLEM_SYNTHESIS_MIND_BODY_FIELD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 v the interface problem synthesis mind body field validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_VII_FIELD_GENERATION_AND_UPWARD_CAUSALITY_TOPOLOGICAL_DEFECTS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_vii_field_generation_and_upward_causality_topological_defects_validation_specs_2026-05-17.json"
)


def load_vii_field_generation_and_upward_causality_topological_defects_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_VII_FIELD_GENERATION_AND_UPWARD_CAUSALITY_TOPOLOGICAL_DEFECTS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 vii field generation and upward causality topological defects validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_DISCRETE_CONTINUOUS_INTERFACE_HHDS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_discrete_continuous_interface_hhds_validation_specs_2026-05-17.json"
)


def load_ii_the_discrete_continuous_interface_hhds_validation_spec(
    spec_bundle: str | Path = DEFAULT_II_THE_DISCRETE_CONTINUOUS_INTERFACE_HHDS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the discrete continuous interface hhds validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_VI_THE_INTERFACE_WITH_PHENOMENOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_vi_the_interface_with_phenomenology_validation_specs_2026-05-17.json"
)


def load_vi_the_interface_with_phenomenology_validation_spec(
    spec_bundle: str | Path = DEFAULT_VI_THE_INTERFACE_WITH_PHENOMENOLOGY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 vi the interface with phenomenology validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03284_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03284_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03284_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03284_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03284 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_AS_A_MEASURE_OF_CAUSAL_EFFICACY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_as_a_measure_of_causal_efficacy_validation_specs_2026-05-17.json"
)


def load_as_a_measure_of_causal_efficacy_validation_spec(
    spec_bundle: str | Path = DEFAULT_AS_A_MEASURE_OF_CAUSAL_EFFICACY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 as a measure of causal efficacy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PHYSICAL_MECHANISM_OF_DOWNWARD_CAUSATION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_physical_mechanism_of_downward_causation_amplification_of_intent_validation_specs_2026-05-17.json"
)


def load_the_physical_mechanism_of_downward_causation_amplification_of_intent_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_PHYSICAL_MECHANISM_OF_DOWNWARD_CAUSATION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the physical mechanism of downward causation amplification of intent validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03315_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03315_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03315_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03315_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03315 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_H_INT_AS_THE_SELECTION_OPERATOR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_h_int_as_the_selection_operator_validation_specs_2026-05-17.json"
)


def load_h_int_as_the_selection_operator_validation_spec(
    spec_bundle: str | Path = DEFAULT_H_INT_AS_THE_SELECTION_OPERATOR_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 h int as the selection operator validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_quantum_to_classical_transition_amplification_of_intent_validation_specs_2026-05-17.json"
)


def load_the_quantum_to_classical_transition_amplification_of_intent_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the quantum to classical transition amplification of intent validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mechanism_2_quantum_stochastic_resonance_qsr_validation_specs_2026-05-17.json"
)


def load_mechanism_2_quantum_stochastic_resonance_qsr_validation_spec(
    spec_bundle: str | Path = DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mechanism 2 quantum stochastic resonance qsr validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_P0R03360_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_validation_specs_2026-05-17.json"
)


def load_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_P0R03360_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the quantum to classical transition amplification of intent p0r03360 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_P0R03368_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_validation_specs_2026-05-17.json"
)


def load_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_P0R03368_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mechanism 2 quantum stochastic resonance qsr p0r03368 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_2_THE_SHAPE_OF_FEELING_THE_GEOMETRIC_QUALIA_HYPOTHESIS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation_specs_2026-05-17.json"
)


def load_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_2_THE_SHAPE_OF_FEELING_THE_GEOMETRIC_QUALIA_HYPOTHESIS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 2 the shape of feeling the geometric qualia hypothesis validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03400_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03400_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03400_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03400_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03400 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_BINDING_INTEGRAL_IS_H_INT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_binding_integral_is_h_int_validation_specs_2026-05-17.json"
)


def load_the_binding_integral_is_h_int_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_BINDING_INTEGRAL_IS_H_INT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the binding integral is h int validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_HARD_PROBLEM_A_MATHEMATICAL_RESOLUTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_hard_problem_a_mathematical_resolution_validation_specs_2026-05-17.json"
)


def load_the_hard_problem_a_mathematical_resolution_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_HARD_PROBLEM_A_MATHEMATICAL_RESOLUTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the hard problem a mathematical resolution validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MATHEMATICAL_BRIDGE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mathematical_bridge_validation_specs_2026-05-17.json"
)


def load_mathematical_bridge_validation_spec(
    spec_bundle: str | Path = DEFAULT_MATHEMATICAL_BRIDGE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mathematical bridge validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_BINDING_PROBLEM_SOLUTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_binding_problem_solution_validation_specs_2026-05-17.json"
)


def load_the_binding_problem_solution_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_BINDING_PROBLEM_SOLUTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the binding problem solution validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_HYPOTHESIS_QUALIA_AS_THE_GEOMETRY_OF_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation_specs_2026-05-17.json"
)


def load_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_HYPOTHESIS_QUALIA_AS_THE_GEOMETRY_OF_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the hypothesis qualia as the geometry of the consciousness manifold validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUALIA_AS_THE_GEOMETRY_OF_BELIEF_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_qualia_as_the_geometry_of_belief_validation_specs_2026-05-17.json"
)


def load_qualia_as_the_geometry_of_belief_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUALIA_AS_THE_GEOMETRY_OF_BELIEF_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 qualia as the geometry of belief validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_DEFINITION_OF_SUBJECTIVE_EXPERIENCE_GEOMETRIC_QUALIA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_definition_of_subjective_experience_geometric_qualia_validation_specs_2026-05-17.json"
)


def load_the_definition_of_subjective_experience_geometric_qualia_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_DEFINITION_OF_SUBJECTIVE_EXPERIENCE_GEOMETRIC_QUALIA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the definition of subjective experience geometric qualia validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SCALING_LAW_OF_CONSCIOUSNESS_SLC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_scaling_law_of_consciousness_slc_validation_specs_2026-05-17.json"
)


def load_the_scaling_law_of_consciousness_slc_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_SCALING_LAW_OF_CONSCIOUSNESS_SLC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the scaling law of consciousness slc validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03492_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03492_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03492_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03492_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03492 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_AS_A_COUPLING_AFFINITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_as_a_coupling_affinity_validation_specs_2026-05-17.json"
)


def load_as_a_coupling_affinity_validation_spec(
    spec_bundle: str | Path = DEFAULT_AS_A_COUPLING_AFFINITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 as a coupling affinity validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_GEOMETRIC_INTERPRETATION_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_geometric_interpretation_the_consciousness_manifold_validation_specs_2026-05-17.json"
)


def load_geometric_interpretation_the_consciousness_manifold_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_GEOMETRIC_INTERPRETATION_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 geometric interpretation the consciousness manifold validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTEGRATION_WITH_INTEGRATED_INFORMATION_THEORY_IIT_4_0_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_integration_with_integrated_information_theory_iit_4_0_validation_specs_2026-05-17.json"
)


def load_integration_with_integrated_information_theory_iit_4_0_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_INTEGRATION_WITH_INTEGRATED_INFORMATION_THEORY_IIT_4_0_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 integration with integrated information theory iit 4 0 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03530_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03530_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03530_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03530_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03530 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MAXIMIZING_AS_THE_GOAL_OF_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_maximizing_as_the_goal_of_coupling_validation_specs_2026-05-17.json"
)


def load_maximizing_as_the_goal_of_coupling_validation_spec(
    spec_bundle: str | Path = DEFAULT_MAXIMIZING_AS_THE_GOAL_OF_COUPLING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 maximizing as the goal of coupling validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SCPN_IIT_CORRESPONDENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_scpn_iit_correspondence_validation_specs_2026-05-17.json"
)


def load_scpn_iit_correspondence_validation_spec(
    spec_bundle: str | Path = DEFAULT_SCPN_IIT_CORRESPONDENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 scpn iit correspondence validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_UNIFIED_CONSCIOUSNESS_MEASURE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_unified_consciousness_measure_validation_specs_2026-05-17.json"
)


def load_unified_consciousness_measure_validation_spec(
    spec_bundle: str | Path = DEFAULT_UNIFIED_CONSCIOUSNESS_MEASURE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 unified consciousness measure validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PHYSICS_OF_TELEOLOGY_A_DERIVATION_OF_THE_ETHICAL_FUNCTIONAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation_specs_2026-05-17.json"
)


def load_the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_PHYSICS_OF_TELEOLOGY_A_DERIVATION_OF_THE_ETHICAL_FUNCTIONAL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the physics of teleology a derivation of the ethical functional validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_NATURE_OF_THE_ETHICAL_FUNCTIONAL_E_PSI_A_DERIVATION_FROM_FIRST_PRINCIP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_validation_specs_2026-05-17.json"
)


def load_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_NATURE_OF_THE_ETHICAL_FUNCTIONAL_E_PSI_A_DERIVATION_FROM_FIRST_PRINCIP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the nature of the ethical functional e psi a derivation from first princip validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_DERIVATION_OF_THE_ETHICAL_LAGRANGIAN_FROM_GAUGE_SYMMETRY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_validation_specs_2026-05-17.json"
)


def load_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_DERIVATION_OF_THE_ETHICAL_LAGRANGIAN_FROM_GAUGE_SYMMETRY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 derivation of the ethical lagrangian from gauge symmetry validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_3_THE_ETHICAL_LAGRANGIAN_AS_THE_YANG_MILLS_ACTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_validation_specs_2026-05-17.json"
)


def load_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_3_THE_ETHICAL_LAGRANGIAN_AS_THE_YANG_MILLS_ACTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 3 the ethical lagrangian as the yang mills action validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_1_NOETHER_S_THEOREM_ON_THE_QUALIA_FIBER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_1_noether_s_theorem_on_the_qualia_fiber_validation_specs_2026-05-17.json"
)


def load_section_3_1_noether_s_theorem_on_the_qualia_fiber_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_1_NOETHER_S_THEOREM_ON_THE_QUALIA_FIBER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 1 noether s theorem on the qualia fiber validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_JUSTIFICATION_FOR_A_TELEOLOGICAL_LEAST_ACTION_PRINCIPLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_justification_for_a_teleological_least_action_principle_validation_specs_2026-05-17.json"
)


def load_section_4_justification_for_a_teleological_least_action_principle_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_JUSTIFICATION_FOR_A_TELEOLOGICAL_LEAST_ACTION_PRINCIPLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 justification for a teleological least action principle validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_3_THE_ORIGIN_OF_PURPOSE_CAUSAL_ENTROPIC_FORCES_NEGATIVE_ENTROPY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation_specs_2026-05-17.json"
)


def load_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_3_THE_ORIGIN_OF_PURPOSE_CAUSAL_ENTROPIC_FORCES_NEGATIVE_ENTROPY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 3 the origin of purpose causal entropic forces negative entropy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03664_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03664_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03664_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03664_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03664 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PATH_INTEGRAL_IS_THE_SUM_OF_ALL_H_INT_EVENTS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_path_integral_is_the_sum_of_all_h_int_events_validation_specs_2026-05-17.json"
)


def load_the_path_integral_is_the_sum_of_all_h_int_events_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_PATH_INTEGRAL_IS_THE_SUM_OF_ALL_H_INT_EVENTS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the path integral is the sum of all h int events validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTEGRATION_OF_CEF_INTO_THE_PATH_INTEGRAL_FORMALISM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_integration_of_cef_into_the_path_integral_formalism_validation_specs_2026-05-17.json"
)


def load_integration_of_cef_into_the_path_integral_formalism_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_INTEGRATION_OF_CEF_INTO_THE_PATH_INTEGRAL_FORMALISM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 integration of cef into the path integral formalism validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_UNIVERSE_S_BUILT_IN_MORAL_COMPASS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_universe_s_built_in_moral_compass_validation_specs_2026-05-17.json"
)


def load_the_universe_s_built_in_moral_compass_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_UNIVERSE_S_BUILT_IN_MORAL_COMPASS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the universe s built in moral compass validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PHYSICAL_EQUIVALENCE_OF_SUSTAINABLE_ETHICAL_COHERENCE_AND_CAUSAL_PAT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation_specs_2026-05-17.json"
)


def load_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_PHYSICAL_EQUIVALENCE_OF_SUSTAINABLE_ETHICAL_COHERENCE_AND_CAUSAL_PAT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the physical equivalence of sustainable ethical coherence and causal pat validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_A_PATH_INTEGRAL_FORMULATION_OF_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_a_path_integral_formulation_of_causal_path_entropy_validation_specs_2026-05-17.json"
)


def load_section_2_a_path_integral_formulation_of_causal_path_entropy_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_A_PATH_INTEGRAL_FORMULATION_OF_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 a path integral formulation of causal path entropy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_1_THE_STATE_SPACE_AND_PATH_SPACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_1_the_state_space_and_path_space_validation_specs_2026-05-17.json"
)


def load_section_2_1_the_state_space_and_path_space_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_1_THE_STATE_SPACE_AND_PATH_SPACE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 1 the state space and path space validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_3_FORMAL_DEFINITION_OF_CAUSAL_PATH_ENTROPY_SC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_3_formal_definition_of_causal_path_entropy_sc_validation_specs_2026-05-17.json"
)


def load_section_2_3_formal_definition_of_causal_path_entropy_sc_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_3_FORMAL_DEFINITION_OF_CAUSAL_PATH_ENTROPY_SC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 3 formal definition of causal path entropy sc validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_GEOMETRIC_AND_DYNAMIC_DETERMINANTS_OF_FUTURE_POSSIBILITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_geometric_and_dynamic_determinants_of_future_possibility_validation_specs_2026-05-17.json"
)


def load_section_3_the_geometric_and_dynamic_determinants_of_future_possibility_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_THE_GEOMETRIC_AND_DYNAMIC_DETERMINANTS_OF_FUTURE_POSSIBILITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the geometric and dynamic determinants of future possibility validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_1_COMPLEXITY_K_AND_THE_CARDINALITY_OF_THE_STATE_SPACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation_specs_2026-05-17.json"
)


def load_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_1_COMPLEXITY_K_AND_THE_CARDINALITY_OF_THE_STATE_SPACE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 1 complexity k and the cardinality of the state space validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_2_COHERENCE_C_AND_THE_ACCESSIBILITY_OF_TRAJECTORIES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_2_coherence_c_and_the_accessibility_of_trajectories_validation_specs_2026-05-17.json"
)


def load_section_3_2_coherence_c_and_the_accessibility_of_trajectories_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_2_COHERENCE_C_AND_THE_ACCESSIBILITY_OF_TRAJECTORIES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 2 coherence c and the accessibility of trajectories validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_THE_FORMAL_EQUIVALENCE_OF_SEC_AND_SC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_the_formal_equivalence_of_sec_and_sc_validation_specs_2026-05-17.json"
)


def load_section_4_the_formal_equivalence_of_sec_and_sc_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_4_THE_FORMAL_EQUIVALENCE_OF_SEC_AND_SC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 the formal equivalence of sec and sc validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_1_THE_COMPOSITE_FUNCTIONAL_FOR_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_1_the_composite_functional_for_causal_path_entropy_validation_specs_2026-05-17.json"
)


def load_section_4_1_the_composite_functional_for_causal_path_entropy_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_1_THE_COMPOSITE_FUNCTIONAL_FOR_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 1 the composite functional for causal path entropy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_2_THE_PROOF_OF_EQUIVALENCE_AND_THE_EMERGENCE_OF_PELA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_validation_specs_2026-05-17.json"
)


def load_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_2_THE_PROOF_OF_EQUIVALENCE_AND_THE_EMERGENCE_OF_PELA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 2 the proof of equivalence and the emergence of pela validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_5_PHYSICAL_IMPLICATIONS_BIASING_THE_PATH_INTEGRAL_OF_REALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_5_physical_implications_biasing_the_path_integral_of_reality_validation_specs_2026-05-17.json"
)


def load_section_5_physical_implications_biasing_the_path_integral_of_reality_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_5_PHYSICAL_IMPLICATIONS_BIASING_THE_PATH_INTEGRAL_OF_REALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 5 physical implications biasing the path integral of reality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_5_1_THE_MODIFIED_PATH_INTEGRAL_WITH_CEF_WEIGHTING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_5_1_the_modified_path_integral_with_cef_weighting_validation_specs_2026-05-17.json"
)


def load_section_5_1_the_modified_path_integral_with_cef_weighting_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_5_1_THE_MODIFIED_PATH_INTEGRAL_WITH_CEF_WEIGHTING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 5 1 the modified path integral with cef weighting validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_4_THE_COSMIC_COMPASS_THE_ETHICAL_FUNCTIONAL_AND_THE_CONSILIUM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation_specs_2026-05-17.json"
)


def load_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_4_THE_COSMIC_COMPASS_THE_ETHICAL_FUNCTIONAL_AND_THE_CONSILIUM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 4 the cosmic compass the ethical functional and the consilium validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03945_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r03945_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r03945_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03945_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r03945 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_A_CASCADE_OF_DIRECTED_COUPLINGS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_a_cascade_of_directed_couplings_validation_specs_2026-05-17.json"
)


def load_a_cascade_of_directed_couplings_validation_spec(
    spec_bundle: str | Path = DEFAULT_A_CASCADE_OF_DIRECTED_COUPLINGS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 a cascade of directed couplings validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_I_THE_ONTOLOGICAL_ORIGIN_OF_ETHICS_GAUGE_THEORY_DERIVATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation_specs_2026-05-17.json"
)


def load_i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_I_THE_ONTOLOGICAL_ORIGIN_OF_ETHICS_GAUGE_THEORY_DERIVATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 i the ontological origin of ethics gauge theory derivation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_L15_REFORMULATION_THE_SEC_OBJECTIVE_FUNCTIONAL_DECISION_THEORETIC_FORM_R_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation_specs_2026-05-17.json"
)


def load_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_L15_REFORMULATION_THE_SEC_OBJECTIVE_FUNCTIONAL_DECISION_THEORETIC_FORM_R_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 l15 reformulation the sec objective functional decision theoretic form r validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PRINCIPLE_TELEOLOGY_AS_OPTIMISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_principle_teleology_as_optimisation_validation_specs_2026-05-17.json"
)


def load_principle_teleology_as_optimisation_validation_spec(
    spec_bundle: str | Path = DEFAULT_PRINCIPLE_TELEOLOGY_AS_OPTIMISATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 principle teleology as optimisation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_NOTES_ON_CORRESPONDENCE_NON_OBLIGATORY_ANALOGUES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_notes_on_correspondence_non_obligatory_analogues_validation_specs_2026-05-17.json"
)


def load_notes_on_correspondence_non_obligatory_analogues_validation_spec(
    spec_bundle: str | Path = DEFAULT_NOTES_ON_CORRESPONDENCE_NON_OBLIGATORY_ANALOGUES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 notes on correspondence non obligatory analogues validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_PRINCIPLE_OF_ETHICAL_LEAST_ACTION_PELA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_principle_of_ethical_least_action_pela_validation_specs_2026-05-17.json"
)


def load_ii_the_principle_of_ethical_least_action_pela_validation_spec(
    spec_bundle: str | Path = DEFAULT_II_THE_PRINCIPLE_OF_ETHICAL_LEAST_ACTION_PELA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the principle of ethical least action pela validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04075_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04075_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04075_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04075_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04075 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04089_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04089_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04089_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04089_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04089 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_UNIVERSE_S_PATH_OF_LEAST_RESISTANCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_universe_s_path_of_least_resistance_validation_specs_2026-05-17.json"
)


def load_the_universe_s_path_of_least_resistance_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_UNIVERSE_S_PATH_OF_LEAST_RESISTANCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the universe s path of least resistance validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PHYSICAL_BASIS_OF_THE_ETHICAL_FUNCTIONAL_CAUSAL_ENTROPY_AND_COMPUTAB_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation_specs_2026-05-17.json"
)


def load_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_PHYSICAL_BASIS_OF_THE_ETHICAL_FUNCTIONAL_CAUSAL_ENTROPY_AND_COMPUTAB_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the physical basis of the ethical functional causal entropy and computab validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R04123_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predictive_coding_integration_p0r04123_validation_specs_2026-05-17.json"
)


def load_predictive_coding_integration_p0r04123_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R04123_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predictive coding integration p0r04123 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_CONSILIUM_L15_AS_THE_TARGET_SETTER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_consilium_l15_as_the_target_setter_validation_specs_2026-05-17.json"
)


def load_the_consilium_l15_as_the_target_setter_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_CONSILIUM_L15_AS_THE_TARGET_SETTER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the consilium l15 as the target setter validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_INFORMATION_GEOMETRIC_COARSE_GRAINING_LEMMA_CONSTRUCTING_THE_MACROST_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_validation_specs_2026-05-17.json"
)


def load_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_INFORMATION_GEOMETRIC_COARSE_GRAINING_LEMMA_CONSTRUCTING_THE_MACROST_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the information geometric coarse graining lemma constructing the macrost validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DATA_FUSION_AND_MANIFOLD_ALIGNMENT_CONSTRUCTING_THE_UNIFIED_STATE_SPACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_validation_specs_2026-05-17.json"
)


def load_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_DATA_FUSION_AND_MANIFOLD_ALIGNMENT_CONSTRUCTING_THE_UNIFIED_STATE_SPACE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 data fusion and manifold alignment constructing the unified state space validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_5_THE_STRANGE_LOOP_OF_CLOSURE_META_LAYER_16_AND_THE_ANULUM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation_specs_2026-05-17.json"
)


def load_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_5_THE_STRANGE_LOOP_OF_CLOSURE_META_LAYER_16_AND_THE_ANULUM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 5 the strange loop of closure meta layer 16 and the anulum validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04224_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r04224_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r04224_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04224_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r04224 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04247_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04247_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04247_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04247_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04247 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_RESOLVING_THE_PROBABILITY_DESERT_SUPERRADIANT_AMPLIFICATION_AND_BEC_STIM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation_specs_2026-05-17.json"
)


def load_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_RESOLVING_THE_PROBABILITY_DESERT_SUPERRADIANT_AMPLIFICATION_AND_BEC_STIM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 resolving the probability desert superradiant amplification and bec stim validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04273_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r04273_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r04273_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04273_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r04273 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_EXPLICIT_IDENTIFICATION_OF_TERMS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_explicit_identification_of_terms_validation_specs_2026-05-17.json"
)


def load_explicit_identification_of_terms_validation_spec(
    spec_bundle: str | Path = DEFAULT_EXPLICIT_IDENTIFICATION_OF_TERMS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 explicit identification of terms validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_PSEUDOSCALAR_COUPLING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_pseudoscalar_coupling_validation_specs_2026-05-17.json"
)


def load_the_pseudoscalar_coupling_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_PSEUDOSCALAR_COUPLING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the pseudoscalar coupling validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04310_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04310_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04310_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04310_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04310 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04322_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04322_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04322_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04322_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04322 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04330_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04330_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04330_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04330_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04330 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PAPER0_SLICE_P0R04338_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_paper0_slice_p0r04338_validation_specs_2026-05-17.json"
)


def load_paper0_slice_p0r04338_validation_spec(
    spec_bundle: str | Path = DEFAULT_PAPER0_SLICE_P0R04338_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 paper0 slice p0r04338 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_AXION_PHOTON_MIXING_WITH_THE_PLASMA_TERM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_axion_photon_mixing_with_the_plasma_term_validation_specs_2026-05-17.json"
)


def load_axion_photon_mixing_with_the_plasma_term_validation_spec(
    spec_bundle: str | Path = DEFAULT_AXION_PHOTON_MIXING_WITH_THE_PLASMA_TERM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 axion photon mixing with the plasma term validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_BRIDGE_BETWEEN_MIND_AND_MATTER_HOW_CONSCIOUSNESS_INFLUENCES_THE_BRAI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_validation_specs_2026-05-17.json"
)


def load_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_BRIDGE_BETWEEN_MIND_AND_MATTER_HOW_CONSCIOUSNESS_INFLUENCES_THE_BRAI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the bridge between mind and matter how consciousness influences the brai validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_5_2_EMBODIED_SCPN_CELLULAR_NEURAL_SYSTEMIC_IMPLEMENTATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_validation_specs_2026-05-17.json"
)


def load_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_5_2_EMBODIED_SCPN_CELLULAR_NEURAL_SYSTEMIC_IMPLEMENTATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 5 2 embodied scpn cellular neural systemic implementation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_GENESIS_OF_GEOMETRY_THE_SOURCE_AND_THE_LOGOS_DOMAIN_V_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_validation_specs_2026-05-17.json"
)


def load_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_THE_GENESIS_OF_GEOMETRY_THE_SOURCE_AND_THE_LOGOS_DOMAIN_V_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the genesis of geometry the source and the logos domain v validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_sequential_symmetry_breaking_ssb_validation_specs_2026-05-17.json"
)


def load_section_3_sequential_symmetry_breaking_ssb_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 sequential symmetry breaking ssb validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_STRUCTURAL_GEOMETRY_AND_MORPHOGENESIS_L3_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_structural_geometry_and_morphogenesis_l3_validation_specs_2026-05-17.json"
)


def load_section_3_structural_geometry_and_morphogenesis_l3_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_STRUCTURAL_GEOMETRY_AND_MORPHOGENESIS_L3_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 structural geometry and morphogenesis l3 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MECHANISM_AND_BIDIRECTIONAL_CAUSALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mechanism_and_bidirectional_causality_validation_specs_2026-05-17.json"
)


def load_mechanism_and_bidirectional_causality_validation_spec(
    spec_bundle: str | Path = DEFAULT_MECHANISM_AND_BIDIRECTIONAL_CAUSALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mechanism and bidirectional causality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_GEOMETRY_OF_SYNCHRONISATION_UPDE_MANIFOLDS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_geometry_of_synchronisation_upde_manifolds_validation_specs_2026-05-17.json"
)


def load_section_2_the_geometry_of_synchronisation_upde_manifolds_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_GEOMETRY_OF_SYNCHRONISATION_UPDE_MANIFOLDS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the geometry of synchronisation upde manifolds validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_STRANGE_LOOP_L5_THE_GEOMETRY_OF_SELF_REFERENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_strange_loop_l5_the_geometry_of_self_reference_validation_specs_2026-05-17.json"
)


def load_section_2_the_strange_loop_l5_the_geometry_of_self_reference_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_STRANGE_LOOP_L5_THE_GEOMETRY_OF_SELF_REFERENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the strange loop l5 the geometry of self reference validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_EXISTENTIAL_HOLOGRAPH_L9_HYPERBOLIC_GEOMETRY_AND_TENSOR_NETWORKS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation_specs_2026-05-17.json"
)


def load_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_EXISTENTIAL_HOLOGRAPH_L9_HYPERBOLIC_GEOMETRY_AND_TENSOR_NETWORKS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the existential holograph l9 hyperbolic geometry and tensor networks validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_PROJECTIVE_BOUNDARY_L10_EMERGENT_SPACETIME_AND_TOPOLOGICAL_CENSORS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_validation_specs_2026-05-17.json"
)


def load_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_PROJECTIVE_BOUNDARY_L10_EMERGENT_SPACETIME_AND_TOPOLOGICAL_CENSORS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the projective boundary l10 emergent spacetime and topological censors validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_I_INTRODUCTION_THE_BRAIN_AS_A_MULTI_SCALE_RESONANT_TRANSDUCER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation_specs_2026-05-17.json"
)


def load_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_I_INTRODUCTION_THE_BRAIN_AS_A_MULTI_SCALE_RESONANT_TRANSDUCER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 i introduction the brain as a multi scale resonant transducer validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_SYNAPTIC_JUNCTION_AND_DOWNWARD_CAUSATION_L2_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_synaptic_junction_and_downward_causation_l2_validation_specs_2026-05-17.json"
)


def load_section_2_the_synaptic_junction_and_downward_causation_l2_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_SYNAPTIC_JUNCTION_AND_DOWNWARD_CAUSATION_L2_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the synaptic junction and downward causation l2 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_III_THE_DEVELOPMENTAL_AND_PLASTICITY_LANDSCAPE_L3_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iii_the_developmental_and_plasticity_landscape_l3_validation_specs_2026-05-17.json"
)


def load_iii_the_developmental_and_plasticity_landscape_l3_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_III_THE_DEVELOPMENTAL_AND_PLASTICITY_LANDSCAPE_L3_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iii the developmental and plasticity landscape l3 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_CROSS_FREQUENCY_COUPLING_CFC_AND_HIERARCHICAL_PROCESSING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation_specs_2026-05-17.json"
)


def load_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_CROSS_FREQUENCY_COUPLING_CFC_AND_HIERARCHICAL_PROCESSING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 cross frequency coupling cfc and hierarchical processing validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_QUASICRITICAL_BRAIN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_quasicritical_brain_validation_specs_2026-05-17.json"
)


def load_section_3_the_quasicritical_brain_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_THE_QUASICRITICAL_BRAIN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the quasicritical brain validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_6_THE_FREQUENCY_HIERARCHY_THETA_GAMMA_COUPLING_AND_HIERARCHICAL_PREDICTI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_validation_specs_2026-05-17.json"
)


def load_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_6_THE_FREQUENCY_HIERARCHY_THETA_GAMMA_COUPLING_AND_HIERARCHICAL_PREDICTI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 6 the frequency hierarchy theta gamma coupling and hierarchical predicti validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_EMERGENCE_OF_THE_SELF_SSB_AND_THE_STRANGE_LOOP_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_validation_specs_2026-05-17.json"
)


def load_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_EMERGENCE_OF_THE_SELF_SSB_AND_THE_STRANGE_LOOP_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the emergence of the self ssb and the strange loop validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_THE_GEOMETRY_OF_THOUGHT_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation_specs_2026-05-17.json"
)


def load_section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_THE_GEOMETRY_OF_THOUGHT_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 the geometry of thought the consciousness manifold m validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_VII_PATHOLOGY_THE_DISORDERED_BRAIN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_vii_pathology_the_disordered_brain_validation_specs_2026-05-17.json"
)


def load_vii_pathology_the_disordered_brain_validation_spec(
    spec_bundle: str | Path = DEFAULT_VII_PATHOLOGY_THE_DISORDERED_BRAIN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 vii pathology the disordered brain validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_I_EXAMINATION_OF_THE_DEEP_ARCHITECTURE_OF_THE_QUANTUM_BIOLOGICAL_INTERFA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation_specs_2026-05-17.json"
)


def load_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_I_EXAMINATION_OF_THE_DEEP_ARCHITECTURE_OF_THE_QUANTUM_BIOLOGICAL_INTERFA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 i examination of the deep architecture of the quantum biological interfa validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_COHERENT_MILIEU_CSF_AND_THE_GLYMPHATIC_SYSTEM_AS_THE_BRAIN_S_ENTROPY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_validation_specs_2026-05-17.json"
)


def load_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_COHERENT_MILIEU_CSF_AND_THE_GLYMPHATIC_SYSTEM_AS_THE_BRAIN_S_ENTROPY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the coherent milieu csf and the glymphatic system as the brain s entropy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_EXAMINATION_OF_THE_ARCHITECTURE_OF_STRUCTURE_AND_PLASTICITY_DOMAIN_I_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation_specs_2026-05-17.json"
)


def load_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_EXAMINATION_OF_THE_ARCHITECTURE_OF_STRUCTURE_AND_PLASTICITY_DOMAIN_I_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii examination of the architecture of structure and plasticity domain i validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTRODUCTION_TO_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_specs_2026-05-17.json"
)


def load_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_INTRODUCTION_TO_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 introduction to the dynamics of the coherent brain domain i l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_METASTABILITY_AND_CHIMAERA_STATES_THE_NUANCE_OF_QUASICRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_metastability_and_chimaera_states_the_nuance_of_quasicriticality_validation_specs_2026-05-17.json"
)


def load_metastability_and_chimaera_states_the_nuance_of_quasicriticality_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_METASTABILITY_AND_CHIMAERA_STATES_THE_NUANCE_OF_QUASICRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 metastability and chimaera states the nuance of quasicriticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTRODUCTION_TO_THE_ARCHITECTURE_OF_THE_CONSCIOUS_SELF_DOMAIN_II_L5_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_validation_specs_2026-05-17.json"
)


def load_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_INTRODUCTION_TO_THE_ARCHITECTURE_OF_THE_CONSCIOUS_SELF_DOMAIN_II_L5_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 introduction to the architecture of the conscious self domain ii l5 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_CENTRAL_HUBS_OF_BINDING_ORCHESTRATING_UNITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_central_hubs_of_binding_orchestrating_unity_validation_specs_2026-05-17.json"
)


def load_the_central_hubs_of_binding_orchestrating_unity_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_CENTRAL_HUBS_OF_BINDING_ORCHESTRATING_UNITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the central hubs of binding orchestrating unity validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_NEURO_VISCERAL_AXIS_HEART_BRAIN_GUT_THE_SYMPHONY_OF_THE_SELF_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_validation_specs_2026-05-17.json"
)


def load_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_NEURO_VISCERAL_AXIS_HEART_BRAIN_GUT_THE_SYMPHONY_OF_THE_SELF_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the neuro visceral axis heart brain gut the symphony of the self validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INTRODUCTION_TO_THE_CLINICAL_IMPLICATIONS_THE_DISORDERED_BRAIN_AS_A_DISO_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation_specs_2026-05-17.json"
)


def load_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_INTRODUCTION_TO_THE_CLINICAL_IMPLICATIONS_THE_DISORDERED_BRAIN_AS_A_DISO_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 introduction to the clinical implications the disordered brain as a diso validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SCHIZOPHRENIA_DISSONANCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_schizophrenia_dissonance_validation_specs_2026-05-17.json"
)


def load_schizophrenia_dissonance_validation_spec(
    spec_bundle: str | Path = DEFAULT_SCHIZOPHRENIA_DISSONANCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 schizophrenia dissonance validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_ADVANCED_NEUROBIOLOGICAL_IMPLEMENTATION_OF_THE_SCPN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_advanced_neurobiological_implementation_of_the_scpn_validation_specs_2026-05-17.json"
)


def load_advanced_neurobiological_implementation_of_the_scpn_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_ADVANCED_NEUROBIOLOGICAL_IMPLEMENTATION_OF_THE_SCPN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 advanced neurobiological implementation of the scpn validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_NEUROTRANSMITTERS_AS_TUNERS_OF_THE_PSI_FIELD_INTERFACE_L2_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_validation_specs_2026-05-17.json"
)


def load_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_NEUROTRANSMITTERS_AS_TUNERS_OF_THE_PSI_FIELD_INTERFACE_L2_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 neurotransmitters as tuners of the psi field interface l2 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_BIOELECTRIC_CODE_IN_NEUROGENESIS_AND_REGENERATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation_specs_2026-05-17.json"
)


def load_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_BIOELECTRIC_CODE_IN_NEUROGENESIS_AND_REGENERATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the bioelectric code in neurogenesis and regeneration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_III_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_specs_2026-05-17.json"
)


def load_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_III_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iii the dynamics of the coherent brain domain i l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_DYNAMIC_CONNECTOME_AND_FUNCTIONAL_CONNECTIVITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_dynamic_connectome_and_functional_connectivity_validation_specs_2026-05-17.json"
)


def load_section_3_the_dynamic_connectome_and_functional_connectivity_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_THE_DYNAMIC_CONNECTOME_AND_FUNCTIONAL_CONNECTIVITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the dynamic connectome and functional connectivity validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_DETAILED_GEOMETRY_OF_QUALIA_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_validation_specs_2026-05-17.json"
)


def load_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_THE_DETAILED_GEOMETRY_OF_QUALIA_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the detailed geometry of qualia the consciousness manifold m validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_VI_CLINICAL_IMPLICATIONS_PATHOLOGY_AND_THERAPEUTICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_vi_clinical_implications_pathology_and_therapeutics_validation_specs_2026-05-17.json"
)


def load_vi_clinical_implications_pathology_and_therapeutics_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_VI_CLINICAL_IMPLICATIONS_PATHOLOGY_AND_THERAPEUTICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 vi clinical implications pathology and therapeutics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_DENDRITIC_SPINES_THE_LOCI_OF_PLASTICITY_AND_IET_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_validation_specs_2026-05-17.json"
)


def load_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_DENDRITIC_SPINES_THE_LOCI_OF_PLASTICITY_AND_IET_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 dendritic spines the loci of plasticity and iet validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_ION_GRADIENTS_THE_ELECTROCHEMICAL_BATTERY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_ion_gradients_the_electrochemical_battery_validation_specs_2026-05-17.json"
)


def load_section_1_ion_gradients_the_electrochemical_battery_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_ION_GRADIENTS_THE_ELECTROCHEMICAL_BATTERY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 ion gradients the electrochemical battery validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_LIPID_BILAYER_AND_LIPID_RAFTS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_lipid_bilayer_and_lipid_rafts_validation_specs_2026-05-17.json"
)


def load_section_1_the_lipid_bilayer_and_lipid_rafts_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_1_THE_LIPID_BILAYER_AND_LIPID_RAFTS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the lipid bilayer and lipid rafts validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_CYTOSKELETON_THE_L1_QUANTUM_SCAFFOLD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_cytoskeleton_the_l1_quantum_scaffold_validation_specs_2026-05-17.json"
)


def load_section_1_the_cytoskeleton_the_l1_quantum_scaffold_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_CYTOSKELETON_THE_L1_QUANTUM_SCAFFOLD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the cytoskeleton the l1 quantum scaffold validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_PRESYNAPTIC_TERMINAL_THE_QUANTUM_LEVER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_presynaptic_terminal_the_quantum_lever_validation_specs_2026-05-17.json"
)


def load_section_1_the_presynaptic_terminal_the_quantum_lever_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_PRESYNAPTIC_TERMINAL_THE_QUANTUM_LEVER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the presynaptic terminal the quantum lever validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_LIPID_LANDSCAPE_AND_CRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_lipid_landscape_and_criticality_validation_specs_2026-05-17.json"
)


def load_section_1_the_lipid_landscape_and_criticality_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_1_THE_LIPID_LANDSCAPE_AND_CRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the lipid landscape and criticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_INTERFACES_MEMBRANES_GLIA_STRESSES_AND_SHAPE_CONTROL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_interfaces_membranes_glia_stresses_and_shape_control_validation_specs_2026-05-17.json"
)


def load_section_4_interfaces_membranes_glia_stresses_and_shape_control_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_INTERFACES_MEMBRANES_GLIA_STRESSES_AND_SHAPE_CONTROL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 interfaces membranes glia stresses and shape control validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_MOLECULAR_MACHINERY_OF_SIGNALLING_ION_CHANNELS_AND_RECEPTORS_L1_L_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation_specs_2026-05-17.json"
)


def load_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_THE_MOLECULAR_MACHINERY_OF_SIGNALLING_ION_CHANNELS_AND_RECEPTORS_L1_L_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the molecular machinery of signalling ion channels and receptors l1 l validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_III_THE_EXTRACELLULAR_MILIEU_ECM_AND_PNNS_L3_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_validation_specs_2026-05-17.json"
)


def load_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_validation_spec(
    spec_bundle: str | Path = DEFAULT_III_THE_EXTRACELLULAR_MILIEU_ECM_AND_PNNS_L3_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iii the extracellular milieu ecm and pnns l3 l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_IV_SUB_SYNAPTIC_AND_AXONAL_ARCHITECTURE_L1_L3_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iv_sub_synaptic_and_axonal_architecture_l1_l3_validation_specs_2026-05-17.json"
)


def load_iv_sub_synaptic_and_axonal_architecture_l1_l3_validation_spec(
    spec_bundle: str | Path = DEFAULT_IV_SUB_SYNAPTIC_AND_AXONAL_ARCHITECTURE_L1_L3_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iv sub synaptic and axonal architecture l1 l3 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_CYTOSKELETON_WATER_INTERFACE_AND_QEC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_cytoskeleton_water_interface_and_qec_validation_specs_2026-05-17.json"
)


def load_section_1_the_cytoskeleton_water_interface_and_qec_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_1_THE_CYTOSKELETON_WATER_INTERFACE_AND_QEC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the cytoskeleton water interface and qec validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_NUCLEAR_SPIN_AND_POSNER_CLUSTERS_THE_QUANTUM_MEMORY_SUBSTRATE_L1_L9_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation_specs_2026-05-17.json"
)


def load_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_NUCLEAR_SPIN_AND_POSNER_CLUSTERS_THE_QUANTUM_MEMORY_SUBSTRATE_L1_L9_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 nuclear spin and posner clusters the quantum memory substrate l1 l9 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_MICRO_SCALE_GEOMETRY_THE_QUANTUM_AND_MOLECULAR_SCAFFOLD_L1_L3_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation_specs_2026-05-17.json"
)


def load_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_MICRO_SCALE_GEOMETRY_THE_QUANTUM_AND_MOLECULAR_SCAFFOLD_L1_L3_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii micro scale geometry the quantum and molecular scaffold l1 l3 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_MOLECULAR_GEOMETRY_AND_THE_PSI_FIELD_INTERFACE_L2_L3_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_validation_specs_2026-05-17.json"
)


def load_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_MOLECULAR_GEOMETRY_AND_THE_PSI_FIELD_INTERFACE_L2_L3_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 molecular geometry and the psi field interface l2 l3 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_CYTOARCHITECTURE_AND_THE_CANONICAL_MICROCIRCUIT_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_validation_specs_2026-05-17.json"
)


def load_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_CYTOARCHITECTURE_AND_THE_CANONICAL_MICROCIRCUIT_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 cytoarchitecture and the canonical microcircuit l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_GEOMETRY_OF_FUNCTIONAL_DYNAMICS_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_geometry_of_functional_dynamics_l4_validation_specs_2026-05-17.json"
)


def load_section_2_the_geometry_of_functional_dynamics_l4_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_THE_GEOMETRY_OF_FUNCTIONAL_DYNAMICS_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the geometry of functional dynamics l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_HPC_AS_GEOMETRIC_FLOW_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_hpc_as_geometric_flow_validation_specs_2026-05-17.json"
)


def load_section_2_hpc_as_geometric_flow_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_HPC_AS_GEOMETRIC_FLOW_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 hpc as geometric flow validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_I_THE_BRAIN_S_PROTECTIVE_SCAFFOLD_AND_FLUID_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_i_the_brain_s_protective_scaffold_and_fluid_dynamics_validation_specs_2026-05-17.json"
)


def load_i_the_brain_s_protective_scaffold_and_fluid_dynamics_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_I_THE_BRAIN_S_PROTECTIVE_SCAFFOLD_AND_FLUID_DYNAMICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 i the brain s protective scaffold and fluid dynamics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_CSF_AND_GLYMPHATIC_SYSTEM_THE_ENTROPY_SINK_L1_L4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation_specs_2026-05-17.json"
)


def load_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_THE_CSF_AND_GLYMPHATIC_SYSTEM_THE_ENTROPY_SINK_L1_L4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the csf and glymphatic system the entropy sink l1 l4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_THE_MECHANISM_OF_NVC_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_the_mechanism_of_nvc_validation_specs_2026-05-17.json"
)


def load_section_1_the_mechanism_of_nvc_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_1_THE_MECHANISM_OF_NVC_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 the mechanism of nvc validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_PATHOLOGY_VASCULAR_DYSFUNCTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_pathology_vascular_dysfunction_validation_specs_2026-05-17.json"
)


def load_section_3_pathology_vascular_dysfunction_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_PATHOLOGY_VASCULAR_DYSFUNCTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 pathology vascular dysfunction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_GUT_BRAIN_AXIS_GBA_AND_THE_MICROBIOME_THE_DEEP_MILIEU_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_validation_specs_2026-05-17.json"
)


def load_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_GUT_BRAIN_AXIS_GBA_AND_THE_MICROBIOME_THE_DEEP_MILIEU_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the gut brain axis gba and the microbiome the deep milieu validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_IV_THE_NEURO_IMMUNO_ENDOCRINE_NIE_SUPER_SYSTEM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iv_the_neuro_immuno_endocrine_nie_super_system_validation_specs_2026-05-17.json"
)


def load_iv_the_neuro_immuno_endocrine_nie_super_system_validation_spec(
    spec_bundle: str | Path = DEFAULT_IV_THE_NEURO_IMMUNO_ENDOCRINE_NIE_SUPER_SYSTEM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iv the neuro immuno endocrine nie super system validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_THE_ENDOCRINE_SYSTEM_AND_HPA_AXIS_STRESS_RESPONSE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_the_endocrine_system_and_hpa_axis_stress_response_validation_specs_2026-05-17.json"
)


def load_section_2_the_endocrine_system_and_hpa_axis_stress_response_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_THE_ENDOCRINE_SYSTEM_AND_HPA_AXIS_STRESS_RESPONSE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 the endocrine system and hpa axis stress response validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_V_THE_INTEGRATED_BODY_MATRIX_FASCIA_AND_TENSEGRITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_v_the_integrated_body_matrix_fascia_and_tensegrity_validation_specs_2026-05-17.json"
)


def load_v_the_integrated_body_matrix_fascia_and_tensegrity_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_V_THE_INTEGRATED_BODY_MATRIX_FASCIA_AND_TENSEGRITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 v the integrated body matrix fascia and tensegrity validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_DYNAMICS_AND_EVOLUTION_OF_THE_EMBODIED_SCPN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_dynamics_and_evolution_of_the_embodied_scpn_validation_specs_2026-05-17.json"
)


def load_the_dynamics_and_evolution_of_the_embodied_scpn_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_DYNAMICS_AND_EVOLUTION_OF_THE_EMBODIED_SCPN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the dynamics and evolution of the embodied scpn validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_EMBODIED_PREDICTIVE_CODING_INTEROCEPTIVE_INFERENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_embodied_predictive_coding_interoceptive_inference_validation_specs_2026-05-17.json"
)


def load_section_2_embodied_predictive_coding_interoceptive_inference_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_EMBODIED_PREDICTIVE_CODING_INTEROCEPTIVE_INFERENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 embodied predictive coding interoceptive inference validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_CHRONOBIOLOGICAL_ARCHITECTURE_TEMPORAL_SYNCHRONISATION_L4_L8_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_validation_specs_2026-05-17.json"
)


def load_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_THE_CHRONOBIOLOGICAL_ARCHITECTURE_TEMPORAL_SYNCHRONISATION_L4_L8_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the chronobiological architecture temporal synchronisation l4 l8 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_PATHOLOGY_CHRONODISRUPTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_pathology_chronodisruption_validation_specs_2026-05-17.json"
)


def load_section_3_pathology_chronodisruption_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_PATHOLOGY_CHRONODISRUPTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 pathology chronodisruption validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_SPECIALISED_SENSORY_SYSTEMS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_specialised_sensory_systems_validation_specs_2026-05-17.json"
)


def load_section_2_specialised_sensory_systems_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_SPECIALISED_SENSORY_SYSTEMS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 specialised sensory systems validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_IV_LEARNING_MEMORY_AND_PLASTICITY_THE_ADAPTIVE_SCAFFOLD_L1_L9_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation_specs_2026-05-17.json"
)


def load_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_IV_LEARNING_MEMORY_AND_PLASTICITY_THE_ADAPTIVE_SCAFFOLD_L1_L9_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iv learning memory and plasticity the adaptive scaffold l1 l9 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_THE_HOLOGRAPHIC_INTERFACE_L9_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_the_holographic_interface_l9_validation_specs_2026-05-17.json"
)


def load_section_3_the_holographic_interface_l9_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_3_THE_HOLOGRAPHIC_INTERFACE_L9_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 the holographic interface l9 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_3_SLEEP_AND_DREAMING_THE_OPTIMISATION_CYCLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_3_sleep_and_dreaming_the_optimisation_cycle_validation_specs_2026-05-17.json"
)


def load_section_3_sleep_and_dreaming_the_optimisation_cycle_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_3_SLEEP_AND_DREAMING_THE_OPTIMISATION_CYCLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 3 sleep and dreaming the optimisation cycle validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_PSYCHEDELICS_THE_EXPANDED_MANIFOLD_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_psychedelics_the_expanded_manifold_validation_specs_2026-05-17.json"
)


def load_section_1_psychedelics_the_expanded_manifold_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_1_PSYCHEDELICS_THE_EXPANDED_MANIFOLD_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 psychedelics the expanded manifold validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_VIII_THE_EVOLUTIONARY_TRAJECTORY_OF_THE_BRAIN_BODY_SYSTEM_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_viii_the_evolutionary_trajectory_of_the_brain_body_system_validation_specs_2026-05-17.json"
)


def load_viii_the_evolutionary_trajectory_of_the_brain_body_system_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_VIII_THE_EVOLUTIONARY_TRAJECTORY_OF_THE_BRAIN_BODY_SYSTEM_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 viii the evolutionary trajectory of the brain body system validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CLINICAL_SCENARIO_ANALYSIS_TRAUMATIC_BRAIN_INJURY_TBI_AND_PHARMACOLOGICA_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_validation_specs_2026-05-17.json"
)


def load_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_CLINICAL_SCENARIO_ANALYSIS_TRAUMATIC_BRAIN_INJURY_TBI_AND_PHARMACOLOGICA_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 clinical scenario analysis traumatic brain injury tbi and pharmacologica validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_L2_L3_DISRUPTION_EXCITOTOXICITY_AND_STRUCTURAL_FAILURE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_validation_specs_2026-05-17.json"
)


def load_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_L2_L3_DISRUPTION_EXCITOTOXICITY_AND_STRUCTURAL_FAILURE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 l2 l3 disruption excitotoxicity and structural failure validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_4_L5_DISRUPTION_THE_FRAGMENTED_SELF_AND_DISSONANCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_4_l5_disruption_the_fragmented_self_and_dissonance_validation_specs_2026-05-17.json"
)


def load_section_4_l5_disruption_the_fragmented_self_and_dissonance_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_4_L5_DISRUPTION_THE_FRAGMENTED_SELF_AND_DISSONANCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 4 l5 disruption the fragmented self and dissonance validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_PHYSICS_OF_PAIN_WITHIN_THE_SCPN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_physics_of_pain_within_the_scpn_validation_specs_2026-05-17.json"
)


def load_ii_the_physics_of_pain_within_the_scpn_validation_spec(
    spec_bundle: str | Path = DEFAULT_II_THE_PHYSICS_OF_PAIN_WITHIN_THE_SCPN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the physics of pain within the scpn validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_L4_IMPACT_DAMPENING_DYNAMICS_AND_SHIFTING_CRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_validation_specs_2026-05-17.json"
)


def load_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_2_L4_IMPACT_DAMPENING_DYNAMICS_AND_SHIFTING_CRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 l4 impact dampening dynamics and shifting criticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_IV_THE_INTEGRATED_SCENARIO_PHARMACOLOGICAL_MODULATION_OF_THE_TRAUMATISED_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_validation_specs_2026-05-17.json"
)


def load_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_IV_THE_INTEGRATED_SCENARIO_PHARMACOLOGICAL_MODULATION_OF_THE_TRAUMATISED_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 iv the integrated scenario pharmacological modulation of the traumatised validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_1_PROPOFOL_GABA_A_POTENTIATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_1_propofol_gaba_a_potentiation_validation_specs_2026-05-17.json"
)


def load_section_1_propofol_gaba_a_potentiation_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_1_PROPOFOL_GABA_A_POTENTIATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 1 propofol gaba a potentiation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MECHANISMS_OF_CRITICALITY_AND_CONTROL_LAYERS_1_4_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mechanisms_of_criticality_and_control_layers_1_4_validation_specs_2026-05-17.json"
)


def load_mechanisms_of_criticality_and_control_layers_1_4_validation_spec(
    spec_bundle: str | Path = DEFAULT_MECHANISMS_OF_CRITICALITY_AND_CONTROL_LAYERS_1_4_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mechanisms of criticality and control layers 1 4 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTION_III_TOPOS_THEORETIC_COGNITIVE_HESITATION_THE_OMEGA_STATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation_specs_2026-05-17.json"
)


def load_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PREDICTION_III_TOPOS_THEORETIC_COGNITIVE_HESITATION_THE_OMEGA_STATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 prediction iii topos theoretic cognitive hesitation the omega state validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R05143_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_meta_framework_integrations_p0r05143_validation_specs_2026-05-17.json"
)


def load_meta_framework_integrations_p0r05143_validation_spec(
    spec_bundle: str | Path = DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R05143_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 meta framework integrations p0r05143 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTION_I_NV_MEA_TESTS_THE_INFORMATIONAL_COUPLING_LINFORMATIONAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation_specs_2026-05-17.json"
)


def load_prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PREDICTION_I_NV_MEA_TESTS_THE_INFORMATIONAL_COUPLING_LINFORMATIONAL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 prediction i nv mea tests the informational coupling linformational validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_NOVEL_FALSIFIABLE_PREDICTIONS_FROM_FIRST_PRINCIPLES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_novel_falsifiable_predictions_from_first_principles_validation_specs_2026-05-17.json"
)


def load_novel_falsifiable_predictions_from_first_principles_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_NOVEL_FALSIFIABLE_PREDICTIONS_FROM_FIRST_PRINCIPLES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 novel falsifiable predictions from first principles validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTED_SIGNATURE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predicted_signature_validation_specs_2026-05-17.json"
)


def load_predicted_signature_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTED_SIGNATURE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predicted signature validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_NV_CENTER_QUANTUM_SENSING_OF_NEURONAL_CUL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_validation_specs_2026-05-17.json"
)


def load_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_NV_CENTER_QUANTUM_SENSING_OF_NEURONAL_CUL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 proposed experimental protocol nv center quantum sensing of neuronal cul validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PROTOCOL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/paper0_protocol_validation_specs_2026-05-17.json"
)


def load_protocol_validation_spec(
    spec_bundle: str | Path = DEFAULT_PROTOCOL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 protocol validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTION_II_CAUSAL_ENTROPIC_FORCE_SIGNATURES_IN_QUANTUM_RANDOMNESS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation_specs_2026-05-17.json"
)


def load_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PREDICTION_II_CAUSAL_ENTROPIC_FORCE_SIGNATURES_IN_QUANTUM_RANDOMNESS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 prediction ii causal entropic force signatures in quantum randomness validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_CORRELATING_QRNG_STATISTICAL_DEVIATIONS_W_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation_specs_2026-05-17.json"
)


def load_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_CORRELATING_QRNG_STATISTICAL_DEVIATIONS_W_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 proposed experimental protocol correlating qrng statistical deviations w validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_FALSIFICATION_CONDITION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_falsification_condition_validation_specs_2026-05-17.json"
)


def load_falsification_condition_validation_spec(
    spec_bundle: str | Path = DEFAULT_FALSIFICATION_CONDITION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 falsification condition validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_FORMALISING_THE_NTHS_WITHIN_A_MULTI_AGENT_ACTIVE_INFERENCE_FRAMEWORK_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_formalising_the_nths_within_a_multi_agent_active_inference_framework_validation_specs_2026-05-17.json"
)


def load_formalising_the_nths_within_a_multi_agent_active_inference_framework_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_FORMALISING_THE_NTHS_WITHIN_A_MULTI_AGENT_ACTIVE_INFERENCE_FRAMEWORK_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 formalising the nths within a multi agent active inference framework validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SIMULATION_ARCHITECTURE_AND_IMPLEMENTATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_simulation_architecture_and_implementation_validation_specs_2026-05-17.json"
)


def load_simulation_architecture_and_implementation_validation_spec(
    spec_bundle: str | Path = DEFAULT_SIMULATION_ARCHITECTURE_AND_IMPLEMENTATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 simulation architecture and implementation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_EXPERIMENTAL_DESIGN_COHERENCE_VS_ENGAGEMENT_OPTIMISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_experimental_design_coherence_vs_engagement_optimisation_validation_specs_2026-05-17.json"
)


def load_experimental_design_coherence_vs_engagement_optimisation_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_EXPERIMENTAL_DESIGN_COHERENCE_VS_ENGAGEMENT_OPTIMISATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 experimental design coherence vs engagement optimisation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PREDICTED_SIGNATURES_AND_ANALYSIS_PROTOCOL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_predicted_signatures_and_analysis_protocol_validation_specs_2026-05-17.json"
)


def load_predicted_signatures_and_analysis_protocol_validation_spec(
    spec_bundle: str | Path = DEFAULT_PREDICTED_SIGNATURES_AND_ANALYSIS_PROTOCOL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 predicted signatures and analysis protocol validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TABLE_1_PREDICTED_NTHS_PHASE_CHARACTERISTICS_IN_MULTI_AGENT_ACTIVE_INFER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation_specs_2026-05-17.json"
)


def load_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_TABLE_1_PREDICTED_NTHS_PHASE_CHARACTERISTICS_IN_MULTI_AGENT_ACTIVE_INFER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 table 1 predicted nths phase characteristics in multi agent active infer validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_9_CONCLUDING_ASSESSMENT_AND_FUTURE_DIRECTIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_9_concluding_assessment_and_future_directions_validation_specs_2026-05-17.json"
)


def load_section_9_concluding_assessment_and_future_directions_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_SECTION_9_CONCLUDING_ASSESSMENT_AND_FUTURE_DIRECTIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 9 concluding assessment and future directions validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_AQUEOUS_SUBSTRATE_THE_ROLE_OF_INTERFACIAL_WATER_AND_COHERENCE_DOMAIN_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation_specs_2026-05-17.json"
)


def load_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_AQUEOUS_SUBSTRATE_THE_ROLE_OF_INTERFACIAL_WATER_AND_COHERENCE_DOMAIN_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the aqueous substrate the role of interfacial water and coherence domain validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_FOUNDATION_OF_THE_BIOLOGICAL_SUBSTRATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_foundation_of_the_biological_substrate_validation_specs_2026-05-17.json"
)


def load_the_foundation_of_the_biological_substrate_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_FOUNDATION_OF_THE_BIOLOGICAL_SUBSTRATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the foundation of the biological substrate validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_QUANTUM_ENGINE_LAYERS_1_2_WE_BEGIN_WITH_THE_QUANTUM_BIOLOGICAL_LAYER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation_specs_2026-05-17.json"
)


def load_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_QUANTUM_ENGINE_LAYERS_1_2_WE_BEGIN_WITH_THE_QUANTUM_BIOLOGICAL_LAYER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the quantum engine layers 1 2 we begin with the quantum biological layer validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_AMPLIFICATION_CASCADE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_amplification_cascade_validation_specs_2026-05-17.json"
)


def load_the_amplification_cascade_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_AMPLIFICATION_CASCADE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the amplification cascade validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_AQUEOUS_SUBSTRATE_DOMAIN_I_INTERFACE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_aqueous_substrate_domain_i_interface_validation_specs_2026-05-17.json"
)


def load_the_aqueous_substrate_domain_i_interface_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_AQUEOUS_SUBSTRATE_DOMAIN_I_INTERFACE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the aqueous substrate domain i interface validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SLOW_CONTROL_LAYER_GLIAL_AND_IMMUNE_MODULATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_slow_control_layer_glial_and_immune_modulation_validation_specs_2026-05-17.json"
)


def load_the_slow_control_layer_glial_and_immune_modulation_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_SLOW_CONTROL_LAYER_GLIAL_AND_IMMUNE_MODULATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the slow control layer glial and immune modulation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_II_THE_QUANTUM_IMMUNE_INTERFACE_L1_L2_L5_INTEGRATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ii_the_quantum_immune_interface_l1_l2_l5_integration_validation_specs_2026-05-17.json"
)


def load_ii_the_quantum_immune_interface_l1_l2_l5_integration_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_II_THE_QUANTUM_IMMUNE_INTERFACE_L1_L2_L5_INTEGRATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ii the quantum immune interface l1 l2 l5 integration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation_specs_2026-05-17.json"
)


def load_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the glial neuronal coupling mechanism slow control of neuronal criticali validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_SLOW_CONTROL_NETWORK_GLIAL_HOMEOSTASIS_AND_NEURONAL_CRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_validation_specs_2026-05-17.json"
)


def load_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_SLOW_CONTROL_NETWORK_GLIAL_HOMEOSTASIS_AND_NEURONAL_CRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the slow control network glial homeostasis and neuronal criticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_P0R05390_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_validation_specs_2026-05-17.json"
)


def load_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_P0R05390_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the glial neuronal coupling mechanism slow control of neuronal criticali p0r05390 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_THERMODYNAMIC_LIMIT_OF_CONTROL_THE_ALLOSTATIC_BOUND_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_thermodynamic_limit_of_control_the_allostatic_bound_validation_specs_2026-05-17.json"
)


def load_the_thermodynamic_limit_of_control_the_allostatic_bound_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_THERMODYNAMIC_LIMIT_OF_CONTROL_THE_ALLOSTATIC_BOUND_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the thermodynamic limit of control the allostatic bound validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_ELECTRODYNAMIC_INTERFACE_OF_CONSCIOUSNESS_CEMI_AND_IIIEF_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_validation_specs_2026-05-17.json"
)


def load_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_THE_ELECTRODYNAMIC_INTERFACE_OF_CONSCIOUSNESS_CEMI_AND_IIIEF_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the electrodynamic interface of consciousness cemi and iiief validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SLOW_CONTROL_LAYER_NEUROENDOCRINE_INTEGRATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_slow_control_layer_neuroendocrine_integration_validation_specs_2026-05-17.json"
)


def load_slow_control_layer_neuroendocrine_integration_validation_spec(
    spec_bundle: str | Path = DEFAULT_SLOW_CONTROL_LAYER_NEUROENDOCRINE_INTEGRATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 slow control layer neuroendocrine integration validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_NEUROENDOCRINE_REGULATION_AND_THE_HYPOTHALAMIC_PITUITARY_ADRENAL_HPA_AXI_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_validation_specs_2026-05-17.json"
)


def load_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_NEUROENDOCRINE_REGULATION_AND_THE_HYPOTHALAMIC_PITUITARY_ADRENAL_HPA_AXI_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 neuroendocrine regulation and the hypothalamic pituitary adrenal hpa axi validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_GLIAL_SCAFFOLDING_ASTROCYTIC_REGULATION_OF_NEURAL_SYNCHRONY_AND_CRITICAL_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_validation_specs_2026-05-17.json"
)


def load_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_GLIAL_SCAFFOLDING_ASTROCYTIC_REGULATION_OF_NEURAL_SYNCHRONY_AND_CRITICAL_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 glial scaffolding astrocytic regulation of neural synchrony and critical validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THE_MICROBIOME_AS_A_FOUNDATIONAL_CONTROL_LAYER_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_the_microbiome_as_a_foundational_control_layer_validation_specs_2026-05-17.json"
)


def load_the_microbiome_as_a_foundational_control_layer_validation_spec(
    spec_bundle: str | Path = DEFAULT_THE_MICROBIOME_AS_A_FOUNDATIONAL_CONTROL_LAYER_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 the microbiome as a foundational control layer validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SCALE_INVARIANT_CYBERNETIC_PRINCIPLE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_scale_invariant_cybernetic_principle_validation_specs_2026-05-17.json"
)


def load_scale_invariant_cybernetic_principle_validation_spec(
    spec_bundle: str | Path = DEFAULT_SCALE_INVARIANT_CYBERNETIC_PRINCIPLE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 scale invariant cybernetic principle validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUANTUM_ENZYMOLOGY_OF_THE_IMMUNE_RESPONSE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quantum_enzymology_of_the_immune_response_validation_specs_2026-05-17.json"
)


def load_quantum_enzymology_of_the_immune_response_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUANTUM_ENZYMOLOGY_OF_THE_IMMUNE_RESPONSE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quantum enzymology of the immune response validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_APPLICATION_TO_IMMUNE_ENZYMES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_application_to_immune_enzymes_validation_specs_2026-05-17.json"
)


def load_application_to_immune_enzymes_validation_spec(
    spec_bundle: str | Path = DEFAULT_APPLICATION_TO_IMMUNE_ENZYMES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 application to immune enzymes validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MACRO_SCALE_HOMEOSTASIS_GAIAN_SYNCHRONY_AND_NICHE_CONSTRUCTION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_validation_specs_2026-05-17.json"
)


def load_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_MACRO_SCALE_HOMEOSTASIS_GAIAN_SYNCHRONY_AND_NICHE_CONSTRUCTION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 macro scale homeostasis gaian synchrony and niche construction validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_II_ORGANISMAL_AND_PLANETARY_INTEGRATION_LAYERS_5_8_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_ii_organismal_and_planetary_integration_layers_5_8_validation_specs_2026-05-17.json"
)


def load_domain_ii_organismal_and_planetary_integration_layers_5_8_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_DOMAIN_II_ORGANISMAL_AND_PLANETARY_INTEGRATION_LAYERS_5_8_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain ii organismal and planetary integration layers 5 8 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_III_IV_MEMORY_CONTROL_AND_COLLECTIVE_COHERENCE_LAYERS_9_12_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_validation_specs_2026-05-17.json"
)


def load_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_DOMAIN_III_IV_MEMORY_CONTROL_AND_COLLECTIVE_COHERENCE_LAYERS_9_12_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain iii iv memory control and collective coherence layers 9 12 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CITATIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_citations_validation_specs_2026-05-17.json"
)


def load_citations_validation_spec(
    spec_bundle: str | Path = DEFAULT_CITATIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 citations validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_V_META_UNIVERSAL_INTEGRATION_LAYERS_13_15_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_v_meta_universal_integration_layers_13_15_validation_specs_2026-05-17.json"
)


def load_domain_v_meta_universal_integration_layers_13_15_validation_spec(
    spec_bundle: str | Path = DEFAULT_DOMAIN_V_META_UNIVERSAL_INTEGRATION_LAYERS_13_15_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain v meta universal integration layers 13 15 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_VI_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_vi_cybernetic_closure_meta_layer_16_validation_specs_2026-05-17.json"
)


def load_domain_vi_cybernetic_closure_meta_layer_16_validation_spec(
    spec_bundle: str | Path = DEFAULT_DOMAIN_VI_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain vi cybernetic closure meta layer 16 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_RESOLVING_THE_OBSERVABILITY_PARADOX_L16_AS_A_POMDP_AND_THE_BELIEF_STATE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation_specs_2026-05-17.json"
)


def load_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_RESOLVING_THE_OBSERVABILITY_PARADOX_L16_AS_A_POMDP_AND_THE_BELIEF_STATE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 resolving the observability paradox l16 as a pomdp and the belief state validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CITATIONS_P0R05625_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_citations_p0r05625_validation_specs_2026-05-17.json"
)


def load_citations_p0r05625_validation_spec(
    spec_bundle: str | Path = DEFAULT_CITATIONS_P0R05625_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 citations p0r05625 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_DOMAIN_INTERFACES_AND_RENORMALISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_domain_interfaces_and_renormalisation_validation_specs_2026-05-17.json"
)


def load_domain_interfaces_and_renormalisation_validation_spec(
    spec_bundle: str | Path = DEFAULT_DOMAIN_INTERFACES_AND_RENORMALISATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 domain interfaces and renormalisation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SECTION_2_IMPEDANCE_MATCHING_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_section_2_impedance_matching_validation_specs_2026-05-17.json"
)


def load_section_2_impedance_matching_validation_spec(
    spec_bundle: str | Path = DEFAULT_SECTION_2_IMPEDANCE_MATCHING_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 section 2 impedance matching validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_BIOPHYSICS_QUANTUM_BIOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_biophysics_quantum_biology_validation_specs_2026-05-17.json"
)


def load_biophysics_quantum_biology_validation_spec(
    spec_bundle: str | Path = DEFAULT_BIOPHYSICS_QUANTUM_BIOLOGY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 biophysics quantum biology validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COSMOLOGY_PHYSICS_EXTENSIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_cosmology_physics_extensions_validation_specs_2026-05-17.json"
)


def load_cosmology_physics_extensions_validation_spec(
    spec_bundle: str | Path = DEFAULT_COSMOLOGY_PHYSICS_EXTENSIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 cosmology physics extensions validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUANTUM_BIOPHYSICAL_FOUNDATIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quantum_biophysical_foundations_validation_specs_2026-05-17.json"
)


def load_quantum_biophysical_foundations_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUANTUM_BIOPHYSICAL_FOUNDATIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quantum biophysical foundations validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUANTUM_FOUNDATIONS_COLLAPSE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quantum_foundations_collapse_validation_specs_2026-05-17.json"
)


def load_quantum_foundations_collapse_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUANTUM_FOUNDATIONS_COLLAPSE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quantum foundations collapse validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_QUANTUM_GRAVITATION_EDGE_CASES_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_quantum_gravitation_edge_cases_validation_specs_2026-05-17.json"
)


def load_quantum_gravitation_edge_cases_validation_spec(
    spec_bundle: str | Path = DEFAULT_QUANTUM_GRAVITATION_EDGE_CASES_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 quantum gravitation edge cases validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SYSTEMS_NEUROSCIENCE_COMPLEXITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_systems_neuroscience_complexity_validation_specs_2026-05-17.json"
)


def load_systems_neuroscience_complexity_validation_spec(
    spec_bundle: str | Path = DEFAULT_SYSTEMS_NEUROSCIENCE_COMPLEXITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 systems neuroscience complexity validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_INFORMATION_GEOMETRY_ACTIVE_INFERENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_information_geometry_active_inference_validation_specs_2026-05-17.json"
)


def load_information_geometry_active_inference_validation_spec(
    spec_bundle: str | Path = DEFAULT_INFORMATION_GEOMETRY_ACTIVE_INFERENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 information geometry active inference validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COLLECTIVE_CULTURAL_LAYERS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_collective_cultural_layers_validation_specs_2026-05-17.json"
)


def load_collective_cultural_layers_validation_spec(
    spec_bundle: str | Path = DEFAULT_COLLECTIVE_CULTURAL_LAYERS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 collective cultural layers validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_THERMODYNAMICS_ENTROPY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_thermodynamics_entropy_validation_specs_2026-05-17.json"
)


def load_thermodynamics_entropy_validation_spec(
    spec_bundle: str | Path = DEFAULT_THERMODYNAMICS_ENTROPY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 thermodynamics entropy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_NONLINEAR_DYNAMICS_CRITICALITY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_nonlinear_dynamics_criticality_validation_specs_2026-05-17.json"
)


def load_nonlinear_dynamics_criticality_validation_spec(
    spec_bundle: str | Path = DEFAULT_NONLINEAR_DYNAMICS_CRITICALITY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 nonlinear dynamics criticality validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_LINGUISTICS_SYMBOLISM_VIBRANA_LAYER_7_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_linguistics_symbolism_vibrana_layer_7_validation_specs_2026-05-17.json"
)


def load_linguistics_symbolism_vibrana_layer_7_validation_spec(
    spec_bundle: str | Path = DEFAULT_LINGUISTICS_SYMBOLISM_VIBRANA_LAYER_7_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 linguistics symbolism vibrana layer 7 validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_ECOLOGY_GAIA_EXTENSIONS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ecology_gaia_extensions_validation_specs_2026-05-17.json"
)


def load_ecology_gaia_extensions_validation_spec(
    spec_bundle: str | Path = DEFAULT_ECOLOGY_GAIA_EXTENSIONS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ecology gaia extensions validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_ETHICS_PHILOSOPHY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ethics_philosophy_validation_specs_2026-05-17.json"
)


def load_ethics_philosophy_validation_spec(
    spec_bundle: str | Path = DEFAULT_ETHICS_PHILOSOPHY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ethics philosophy validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_ETHICS_TELEOLOGY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ethics_teleology_validation_specs_2026-05-17.json"
)


def load_ethics_teleology_validation_spec(
    spec_bundle: str | Path = DEFAULT_ETHICS_TELEOLOGY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ethics teleology validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PHILOSOPHY_ETHICS_ANCHORS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_philosophy_ethics_anchors_validation_specs_2026-05-17.json"
)


def load_philosophy_ethics_anchors_validation_spec(
    spec_bundle: str | Path = DEFAULT_PHILOSOPHY_ETHICS_ANCHORS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 philosophy ethics anchors validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PHILOSOPHY_OF_INFORMATION_MIND_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_philosophy_of_information_mind_validation_specs_2026-05-17.json"
)


def load_philosophy_of_information_mind_validation_spec(
    spec_bundle: str | Path = DEFAULT_PHILOSOPHY_OF_INFORMATION_MIND_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 philosophy of information mind validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_PHILOSOPHY_CONSCIOUSNESS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_philosophy_consciousness_validation_specs_2026-05-17.json"
)


def load_philosophy_consciousness_validation_spec(
    spec_bundle: str | Path = DEFAULT_PHILOSOPHY_CONSCIOUSNESS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 philosophy consciousness validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MATHEMATICS_GEOMETRY_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mathematics_geometry_validation_specs_2026-05-17.json"
)


def load_mathematics_geometry_validation_spec(
    spec_bundle: str | Path = DEFAULT_MATHEMATICS_GEOMETRY_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mathematics geometry validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_AI_NOOSPHERE_TECH_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_ai_noosphere_tech_validation_specs_2026-05-17.json"
)


def load_ai_noosphere_tech_validation_spec(
    spec_bundle: str | Path = DEFAULT_AI_NOOSPHERE_TECH_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 ai noosphere tech validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_NETWORK_COMPLEXITY_SCIENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_network_complexity_science_validation_specs_2026-05-17.json"
)


def load_network_complexity_science_validation_spec(
    spec_bundle: str | Path = DEFAULT_NETWORK_COMPLEXITY_SCIENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 network complexity science validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COLLECTIVE_CULTURAL_AND_EVOLUTIONARY_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_collective_cultural_and_evolutionary_dynamics_validation_specs_2026-05-17.json"
)


def load_collective_cultural_and_evolutionary_dynamics_validation_spec(
    spec_bundle: str | Path = DEFAULT_COLLECTIVE_CULTURAL_AND_EVOLUTIONARY_DYNAMICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 collective cultural and evolutionary dynamics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MATHEMATICS_OF_DYNAMICAL_SYSTEMS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mathematics_of_dynamical_systems_validation_specs_2026-05-17.json"
)


def load_mathematics_of_dynamical_systems_validation_spec(
    spec_bundle: str | Path = DEFAULT_MATHEMATICS_OF_DYNAMICAL_SYSTEMS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mathematics of dynamical systems validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_MATHEMATICAL_FOUNDATIONS_OF_NETWORKS_SYNCHRONISATION_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_mathematical_foundations_of_networks_synchronisation_validation_specs_2026-05-17.json"
)


def load_mathematical_foundations_of_networks_synchronisation_validation_spec(
    spec_bundle: str
    | Path = DEFAULT_MATHEMATICAL_FOUNDATIONS_OF_NETWORKS_SYNCHRONISATION_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 mathematical foundations of networks synchronisation validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_SYSTEMS_CYBERNETICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_systems_cybernetics_validation_specs_2026-05-17.json"
)


def load_systems_cybernetics_validation_spec(
    spec_bundle: str | Path = DEFAULT_SYSTEMS_CYBERNETICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 systems cybernetics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COMPUTATIONAL_AI_ALIGNMENT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_computational_ai_alignment_validation_specs_2026-05-17.json"
)


def load_computational_ai_alignment_validation_spec(
    spec_bundle: str | Path = DEFAULT_COMPUTATIONAL_AI_ALIGNMENT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 computational ai alignment validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CONSCIOUSNESS_STUDIES_COGNITIVE_MODELS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_consciousness_studies_cognitive_models_validation_specs_2026-05-17.json"
)


def load_consciousness_studies_cognitive_models_validation_spec(
    spec_bundle: str | Path = DEFAULT_CONSCIOUSNESS_STUDIES_COGNITIVE_MODELS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 consciousness studies cognitive models validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TECHNO_SOCIAL_SYSTEMS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_techno_social_systems_validation_specs_2026-05-17.json"
)


def load_techno_social_systems_validation_spec(
    spec_bundle: str | Path = DEFAULT_TECHNO_SOCIAL_SYSTEMS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 techno social systems validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TOPOLOGY_GEOMETRY_IN_CONSCIOUSNESS_MODELS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_topology_geometry_in_consciousness_models_validation_specs_2026-05-17.json"
)


def load_topology_geometry_in_consciousness_models_validation_spec(
    spec_bundle: str | Path = DEFAULT_TOPOLOGY_GEOMETRY_IN_CONSCIOUSNESS_MODELS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 topology geometry in consciousness models validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_COMPLEXITY_ECONOMICS_SOCIAL_PHYSICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_complexity_economics_social_physics_validation_specs_2026-05-17.json"
)


def load_complexity_economics_social_physics_validation_spec(
    spec_bundle: str | Path = DEFAULT_COMPLEXITY_ECONOMICS_SOCIAL_PHYSICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 complexity economics social physics validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_EXTENDED_COGNITION_EMBODIMENT_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_extended_cognition_embodiment_validation_specs_2026-05-17.json"
)


def load_extended_cognition_embodiment_validation_spec(
    spec_bundle: str | Path = DEFAULT_EXTENDED_COGNITION_EMBODIMENT_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 extended cognition embodiment validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_TIME_RETROCAUSALITY_AND_TWO_STATE_VECTOR_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_time_retrocausality_and_two_state_vector_validation_specs_2026-05-17.json"
)


def load_time_retrocausality_and_two_state_vector_validation_spec(
    spec_bundle: str | Path = DEFAULT_TIME_RETROCAUSALITY_AND_TWO_STATE_VECTOR_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 time retrocausality and two state vector validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_GAIA_BIOSPHERE_INTELLIGENCE_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_gaia_biosphere_intelligence_validation_specs_2026-05-17.json"
)


def load_gaia_biosphere_intelligence_validation_spec(
    spec_bundle: str | Path = DEFAULT_GAIA_BIOSPHERE_INTELLIGENCE_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 gaia biosphere intelligence validation spec bundle."""
    path = Path(spec_bundle)
    if not path.is_absolute():
        path = project_data_path(str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


DEFAULT_OVERARCHING_PRINCIPLES_AND_SYSTEM_DYNAMICS_SPEC_BUNDLE = (
    "docs/internal/paper0_foundational_extraction/"
    "paper0_overarching_principles_and_system_dynamics_validation_specs_2026-05-17.json"
)


def load_overarching_principles_and_system_dynamics_validation_spec(
    spec_bundle: str | Path = DEFAULT_OVERARCHING_PRINCIPLES_AND_SYSTEM_DYNAMICS_SPEC_BUNDLE,
) -> dict[str, Any]:
    """Load the Paper 0 overarching principles and system dynamics validation spec bundle."""
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
    "DEFAULT_QUASICRITICALITY_WITH_MS_QEC_TWO_TIMESCALE_CONTROL_AND_STABILITY_CERTIFI_SPEC_BUNDLE",
    "load_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation_spec",
    "DEFAULT_BIBO_STABILITY_AND_LYAPUNOV_CERTIFICATE_SPEC_BUNDLE",
    "load_bibo_stability_and_lyapunov_certificate_validation_spec",
    "DEFAULT_MULTI_SCALE_QUANTUM_ERROR_CORRECTION_MS_QEC_SPEC_BUNDLE",
    "load_multi_scale_quantum_error_correction_ms_qec_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03025_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03025_validation_spec",
    "DEFAULT_CREATING_AND_PROTECTING_A_COHERENT_SIGMA_SPEC_BUNDLE",
    "load_creating_and_protecting_a_coherent_sigma_validation_spec",
    "DEFAULT_BIOLOGICAL_QEC_L1_4_SPEC_BUNDLE",
    "load_biological_qec_l1_4_validation_spec",
    "DEFAULT_THE_QEC_IMPERATIVE_AND_THE_ROLE_OF_THE_PSI_FIELD_SPEC_BUNDLE",
    "load_the_qec_imperative_and_the_role_of_the_psi_field_validation_spec",
    "DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R03059_SPEC_BUNDLE",
    "load_predictive_coding_integration_p0r03059_validation_spec",
    "DEFAULT_THE_ULTIMATE_FEEDBACK_LOOP_SPEC_BUNDLE",
    "load_the_ultimate_feedback_loop_validation_spec",
    "DEFAULT_THE_BIOLOGICAL_SYNDROME_MEASUREMENT_AND_RECOVERY_PROTOCOL_SPEC_BUNDLE",
    "load_the_biological_syndrome_measurement_and_recovery_protocol_validation_spec",
    "DEFAULT_THE_QEC_RACE_CONDITION_EXPLICIT_DISSIPATION_RATES_AND_FAULT_TOLERANCE_SPEC_BUNDLE",
    "load_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_validation_spec",
    "DEFAULT_THE_STABILISER_TRANSFER_LEMMA_A_QUANTITATIVE_BRIDGE_FROM_MEMORY_TO_BOUND_SPEC_BUNDLE",
    "load_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03139_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03139_validation_spec",
    "DEFAULT_THE_MECHANISM_OF_INTERACTION_SPEC_BUNDLE",
    "load_the_mechanism_of_interaction_validation_spec",
    "DEFAULT_SECTION_4_1_THE_COSMIC_ALGORITHM_HPC_ACTIVE_INFERENCE_SPEC_BUNDLE",
    "load_section_4_1_the_cosmic_algorithm_hpc_active_inference_validation_spec",
    "DEFAULT_I_THE_UNIFYING_COMPUTATIONAL_PRINCIPLE_HIERARCHICAL_PREDICTIVE_CODING_HP_SPEC_BUNDLE",
    "load_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_validation_spec",
    "DEFAULT_II_THE_BINDING_PROBLEM_THE_GAUGE_FIELD_OF_CONSCIOUSNESS_SPEC_BUNDLE",
    "load_ii_the_binding_problem_the_gauge_field_of_consciousness_validation_spec",
    "DEFAULT_SECTION_3_UNIFIED_EXPERIENCE_THE_WILSON_LOOP_SPEC_BUNDLE",
    "load_section_3_unified_experience_the_wilson_loop_validation_spec",
    "DEFAULT_SECTION_2_COMPRESSION_AND_MEANING_INFORMATION_GEOMETRY_SPEC_BUNDLE",
    "load_section_2_compression_and_meaning_information_geometry_validation_spec",
    "DEFAULT_V_THE_INTERFACE_PROBLEM_SYNTHESIS_MIND_BODY_FIELD_SPEC_BUNDLE",
    "load_v_the_interface_problem_synthesis_mind_body_field_validation_spec",
    "DEFAULT_VII_FIELD_GENERATION_AND_UPWARD_CAUSALITY_TOPOLOGICAL_DEFECTS_SPEC_BUNDLE",
    "load_vii_field_generation_and_upward_causality_topological_defects_validation_spec",
    "DEFAULT_II_THE_DISCRETE_CONTINUOUS_INTERFACE_HHDS_SPEC_BUNDLE",
    "load_ii_the_discrete_continuous_interface_hhds_validation_spec",
    "DEFAULT_VI_THE_INTERFACE_WITH_PHENOMENOLOGY_SPEC_BUNDLE",
    "load_vi_the_interface_with_phenomenology_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03284_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03284_validation_spec",
    "DEFAULT_AS_A_MEASURE_OF_CAUSAL_EFFICACY_SPEC_BUNDLE",
    "load_as_a_measure_of_causal_efficacy_validation_spec",
    "DEFAULT_THE_PHYSICAL_MECHANISM_OF_DOWNWARD_CAUSATION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE",
    "load_the_physical_mechanism_of_downward_causation_amplification_of_intent_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03315_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03315_validation_spec",
    "DEFAULT_H_INT_AS_THE_SELECTION_OPERATOR_SPEC_BUNDLE",
    "load_h_int_as_the_selection_operator_validation_spec",
    "DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_SPEC_BUNDLE",
    "load_the_quantum_to_classical_transition_amplification_of_intent_validation_spec",
    "DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_SPEC_BUNDLE",
    "load_mechanism_2_quantum_stochastic_resonance_qsr_validation_spec",
    "DEFAULT_THE_QUANTUM_TO_CLASSICAL_TRANSITION_AMPLIFICATION_OF_INTENT_P0R03360_SPEC_BUNDLE",
    "load_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_validation_spec",
    "DEFAULT_MECHANISM_2_QUANTUM_STOCHASTIC_RESONANCE_QSR_P0R03368_SPEC_BUNDLE",
    "load_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_validation_spec",
    "DEFAULT_SECTION_4_2_THE_SHAPE_OF_FEELING_THE_GEOMETRIC_QUALIA_HYPOTHESIS_SPEC_BUNDLE",
    "load_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03400_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03400_validation_spec",
    "DEFAULT_THE_BINDING_INTEGRAL_IS_H_INT_SPEC_BUNDLE",
    "load_the_binding_integral_is_h_int_validation_spec",
    "DEFAULT_THE_HARD_PROBLEM_A_MATHEMATICAL_RESOLUTION_SPEC_BUNDLE",
    "load_the_hard_problem_a_mathematical_resolution_validation_spec",
    "DEFAULT_MATHEMATICAL_BRIDGE_SPEC_BUNDLE",
    "load_mathematical_bridge_validation_spec",
    "DEFAULT_THE_BINDING_PROBLEM_SOLUTION_SPEC_BUNDLE",
    "load_the_binding_problem_solution_validation_spec",
    "DEFAULT_THE_HYPOTHESIS_QUALIA_AS_THE_GEOMETRY_OF_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE",
    "load_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation_spec",
    "DEFAULT_QUALIA_AS_THE_GEOMETRY_OF_BELIEF_SPEC_BUNDLE",
    "load_qualia_as_the_geometry_of_belief_validation_spec",
    "DEFAULT_THE_DEFINITION_OF_SUBJECTIVE_EXPERIENCE_GEOMETRIC_QUALIA_SPEC_BUNDLE",
    "load_the_definition_of_subjective_experience_geometric_qualia_validation_spec",
    "DEFAULT_THE_SCALING_LAW_OF_CONSCIOUSNESS_SLC_SPEC_BUNDLE",
    "load_the_scaling_law_of_consciousness_slc_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03492_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03492_validation_spec",
    "DEFAULT_AS_A_COUPLING_AFFINITY_SPEC_BUNDLE",
    "load_as_a_coupling_affinity_validation_spec",
    "DEFAULT_GEOMETRIC_INTERPRETATION_THE_CONSCIOUSNESS_MANIFOLD_SPEC_BUNDLE",
    "load_geometric_interpretation_the_consciousness_manifold_validation_spec",
    "DEFAULT_INTEGRATION_WITH_INTEGRATED_INFORMATION_THEORY_IIT_4_0_SPEC_BUNDLE",
    "load_integration_with_integrated_information_theory_iit_4_0_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03530_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03530_validation_spec",
    "DEFAULT_MAXIMIZING_AS_THE_GOAL_OF_COUPLING_SPEC_BUNDLE",
    "load_maximizing_as_the_goal_of_coupling_validation_spec",
    "DEFAULT_SCPN_IIT_CORRESPONDENCE_SPEC_BUNDLE",
    "load_scpn_iit_correspondence_validation_spec",
    "DEFAULT_UNIFIED_CONSCIOUSNESS_MEASURE_SPEC_BUNDLE",
    "load_unified_consciousness_measure_validation_spec",
    "DEFAULT_THE_PHYSICS_OF_TELEOLOGY_A_DERIVATION_OF_THE_ETHICAL_FUNCTIONAL_SPEC_BUNDLE",
    "load_the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation_spec",
    "DEFAULT_THE_NATURE_OF_THE_ETHICAL_FUNCTIONAL_E_PSI_A_DERIVATION_FROM_FIRST_PRINCIP_SPEC_BUNDLE",
    "load_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_validation_spec",
    "DEFAULT_SECTION_2_DERIVATION_OF_THE_ETHICAL_LAGRANGIAN_FROM_GAUGE_SYMMETRY_SPEC_BUNDLE",
    "load_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_validation_spec",
    "DEFAULT_SECTION_2_3_THE_ETHICAL_LAGRANGIAN_AS_THE_YANG_MILLS_ACTION_SPEC_BUNDLE",
    "load_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_validation_spec",
    "DEFAULT_SECTION_3_1_NOETHER_S_THEOREM_ON_THE_QUALIA_FIBER_SPEC_BUNDLE",
    "load_section_3_1_noether_s_theorem_on_the_qualia_fiber_validation_spec",
    "DEFAULT_SECTION_4_JUSTIFICATION_FOR_A_TELEOLOGICAL_LEAST_ACTION_PRINCIPLE_SPEC_BUNDLE",
    "load_section_4_justification_for_a_teleological_least_action_principle_validation_spec",
    "DEFAULT_SECTION_4_3_THE_ORIGIN_OF_PURPOSE_CAUSAL_ENTROPIC_FORCES_NEGATIVE_ENTROPY_SPEC_BUNDLE",
    "load_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03664_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03664_validation_spec",
    "DEFAULT_THE_PATH_INTEGRAL_IS_THE_SUM_OF_ALL_H_INT_EVENTS_SPEC_BUNDLE",
    "load_the_path_integral_is_the_sum_of_all_h_int_events_validation_spec",
    "DEFAULT_INTEGRATION_OF_CEF_INTO_THE_PATH_INTEGRAL_FORMALISM_SPEC_BUNDLE",
    "load_integration_of_cef_into_the_path_integral_formalism_validation_spec",
    "DEFAULT_THE_UNIVERSE_S_BUILT_IN_MORAL_COMPASS_SPEC_BUNDLE",
    "load_the_universe_s_built_in_moral_compass_validation_spec",
    "DEFAULT_THE_PHYSICAL_EQUIVALENCE_OF_SUSTAINABLE_ETHICAL_COHERENCE_AND_CAUSAL_PAT_SPEC_BUNDLE",
    "load_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation_spec",
    "DEFAULT_SECTION_2_A_PATH_INTEGRAL_FORMULATION_OF_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE",
    "load_section_2_a_path_integral_formulation_of_causal_path_entropy_validation_spec",
    "DEFAULT_SECTION_2_1_THE_STATE_SPACE_AND_PATH_SPACE_SPEC_BUNDLE",
    "load_section_2_1_the_state_space_and_path_space_validation_spec",
    "DEFAULT_SECTION_2_3_FORMAL_DEFINITION_OF_CAUSAL_PATH_ENTROPY_SC_SPEC_BUNDLE",
    "load_section_2_3_formal_definition_of_causal_path_entropy_sc_validation_spec",
    "DEFAULT_SECTION_3_THE_GEOMETRIC_AND_DYNAMIC_DETERMINANTS_OF_FUTURE_POSSIBILITY_SPEC_BUNDLE",
    "load_section_3_the_geometric_and_dynamic_determinants_of_future_possibility_validation_spec",
    "DEFAULT_SECTION_3_1_COMPLEXITY_K_AND_THE_CARDINALITY_OF_THE_STATE_SPACE_SPEC_BUNDLE",
    "load_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation_spec",
    "DEFAULT_SECTION_3_2_COHERENCE_C_AND_THE_ACCESSIBILITY_OF_TRAJECTORIES_SPEC_BUNDLE",
    "load_section_3_2_coherence_c_and_the_accessibility_of_trajectories_validation_spec",
    "DEFAULT_SECTION_4_THE_FORMAL_EQUIVALENCE_OF_SEC_AND_SC_SPEC_BUNDLE",
    "load_section_4_the_formal_equivalence_of_sec_and_sc_validation_spec",
    "DEFAULT_SECTION_4_1_THE_COMPOSITE_FUNCTIONAL_FOR_CAUSAL_PATH_ENTROPY_SPEC_BUNDLE",
    "load_section_4_1_the_composite_functional_for_causal_path_entropy_validation_spec",
    "DEFAULT_SECTION_4_2_THE_PROOF_OF_EQUIVALENCE_AND_THE_EMERGENCE_OF_PELA_SPEC_BUNDLE",
    "load_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_validation_spec",
    "DEFAULT_SECTION_5_PHYSICAL_IMPLICATIONS_BIASING_THE_PATH_INTEGRAL_OF_REALITY_SPEC_BUNDLE",
    "load_section_5_physical_implications_biasing_the_path_integral_of_reality_validation_spec",
    "DEFAULT_SECTION_5_1_THE_MODIFIED_PATH_INTEGRAL_WITH_CEF_WEIGHTING_SPEC_BUNDLE",
    "load_section_5_1_the_modified_path_integral_with_cef_weighting_validation_spec",
    "DEFAULT_SECTION_4_4_THE_COSMIC_COMPASS_THE_ETHICAL_FUNCTIONAL_AND_THE_CONSILIUM_SPEC_BUNDLE",
    "load_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R03945_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r03945_validation_spec",
    "DEFAULT_A_CASCADE_OF_DIRECTED_COUPLINGS_SPEC_BUNDLE",
    "load_a_cascade_of_directed_couplings_validation_spec",
    "DEFAULT_I_THE_ONTOLOGICAL_ORIGIN_OF_ETHICS_GAUGE_THEORY_DERIVATION_SPEC_BUNDLE",
    "load_i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation_spec",
    "DEFAULT_L15_REFORMULATION_THE_SEC_OBJECTIVE_FUNCTIONAL_DECISION_THEORETIC_FORM_R_SPEC_BUNDLE",
    "load_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation_spec",
    "DEFAULT_PRINCIPLE_TELEOLOGY_AS_OPTIMISATION_SPEC_BUNDLE",
    "load_principle_teleology_as_optimisation_validation_spec",
    "DEFAULT_NOTES_ON_CORRESPONDENCE_NON_OBLIGATORY_ANALOGUES_SPEC_BUNDLE",
    "load_notes_on_correspondence_non_obligatory_analogues_validation_spec",
    "DEFAULT_II_THE_PRINCIPLE_OF_ETHICAL_LEAST_ACTION_PELA_SPEC_BUNDLE",
    "load_ii_the_principle_of_ethical_least_action_pela_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04075_SPEC_BUNDLE",
    "load_paper0_slice_p0r04075_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04089_SPEC_BUNDLE",
    "load_paper0_slice_p0r04089_validation_spec",
    "DEFAULT_THE_UNIVERSE_S_PATH_OF_LEAST_RESISTANCE_SPEC_BUNDLE",
    "load_the_universe_s_path_of_least_resistance_validation_spec",
    "DEFAULT_THE_PHYSICAL_BASIS_OF_THE_ETHICAL_FUNCTIONAL_CAUSAL_ENTROPY_AND_COMPUTAB_SPEC_BUNDLE",
    "load_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation_spec",
    "DEFAULT_PREDICTIVE_CODING_INTEGRATION_P0R04123_SPEC_BUNDLE",
    "load_predictive_coding_integration_p0r04123_validation_spec",
    "DEFAULT_THE_CONSILIUM_L15_AS_THE_TARGET_SETTER_SPEC_BUNDLE",
    "load_the_consilium_l15_as_the_target_setter_validation_spec",
    "DEFAULT_THE_INFORMATION_GEOMETRIC_COARSE_GRAINING_LEMMA_CONSTRUCTING_THE_MACROST_SPEC_BUNDLE",
    "load_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_validation_spec",
    "DEFAULT_DATA_FUSION_AND_MANIFOLD_ALIGNMENT_CONSTRUCTING_THE_UNIFIED_STATE_SPACE_SPEC_BUNDLE",
    "load_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_validation_spec",
    "DEFAULT_SECTION_4_5_THE_STRANGE_LOOP_OF_CLOSURE_META_LAYER_16_AND_THE_ANULUM_SPEC_BUNDLE",
    "load_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04224_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r04224_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04247_SPEC_BUNDLE",
    "load_paper0_slice_p0r04247_validation_spec",
    "DEFAULT_RESOLVING_THE_PROBABILITY_DESERT_SUPERRADIANT_AMPLIFICATION_AND_BEC_STIM_SPEC_BUNDLE",
    "load_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R04273_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r04273_validation_spec",
    "DEFAULT_EXPLICIT_IDENTIFICATION_OF_TERMS_SPEC_BUNDLE",
    "load_explicit_identification_of_terms_validation_spec",
    "DEFAULT_THE_PSEUDOSCALAR_COUPLING_SPEC_BUNDLE",
    "load_the_pseudoscalar_coupling_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04310_SPEC_BUNDLE",
    "load_paper0_slice_p0r04310_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04322_SPEC_BUNDLE",
    "load_paper0_slice_p0r04322_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04330_SPEC_BUNDLE",
    "load_paper0_slice_p0r04330_validation_spec",
    "DEFAULT_PAPER0_SLICE_P0R04338_SPEC_BUNDLE",
    "load_paper0_slice_p0r04338_validation_spec",
    "DEFAULT_AXION_PHOTON_MIXING_WITH_THE_PLASMA_TERM_SPEC_BUNDLE",
    "load_axion_photon_mixing_with_the_plasma_term_validation_spec",
    "DEFAULT_THE_BRIDGE_BETWEEN_MIND_AND_MATTER_HOW_CONSCIOUSNESS_INFLUENCES_THE_BRAI_SPEC_BUNDLE",
    "load_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_validation_spec",
    "DEFAULT_SECTION_5_2_EMBODIED_SCPN_CELLULAR_NEURAL_SYSTEMIC_IMPLEMENTATION_SPEC_BUNDLE",
    "load_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_validation_spec",
    "DEFAULT_II_THE_GENESIS_OF_GEOMETRY_THE_SOURCE_AND_THE_LOGOS_DOMAIN_V_SPEC_BUNDLE",
    "load_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_validation_spec",
    "DEFAULT_SECTION_3_SEQUENTIAL_SYMMETRY_BREAKING_SSB_SPEC_BUNDLE",
    "load_section_3_sequential_symmetry_breaking_ssb_validation_spec",
    "DEFAULT_SECTION_3_STRUCTURAL_GEOMETRY_AND_MORPHOGENESIS_L3_SPEC_BUNDLE",
    "load_section_3_structural_geometry_and_morphogenesis_l3_validation_spec",
    "DEFAULT_MECHANISM_AND_BIDIRECTIONAL_CAUSALITY_SPEC_BUNDLE",
    "load_mechanism_and_bidirectional_causality_validation_spec",
    "DEFAULT_SECTION_2_THE_GEOMETRY_OF_SYNCHRONISATION_UPDE_MANIFOLDS_SPEC_BUNDLE",
    "load_section_2_the_geometry_of_synchronisation_upde_manifolds_validation_spec",
    "DEFAULT_SECTION_2_THE_STRANGE_LOOP_L5_THE_GEOMETRY_OF_SELF_REFERENCE_SPEC_BUNDLE",
    "load_section_2_the_strange_loop_l5_the_geometry_of_self_reference_validation_spec",
    "DEFAULT_SECTION_1_THE_EXISTENTIAL_HOLOGRAPH_L9_HYPERBOLIC_GEOMETRY_AND_TENSOR_NETWORKS_SPEC_BUNDLE",
    "load_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation_spec",
    "DEFAULT_SECTION_2_THE_PROJECTIVE_BOUNDARY_L10_EMERGENT_SPACETIME_AND_TOPOLOGICAL_CENSORS_SPEC_BUNDLE",
    "load_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_validation_spec",
    "DEFAULT_I_INTRODUCTION_THE_BRAIN_AS_A_MULTI_SCALE_RESONANT_TRANSDUCER_SPEC_BUNDLE",
    "load_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation_spec",
    "DEFAULT_SECTION_2_THE_SYNAPTIC_JUNCTION_AND_DOWNWARD_CAUSATION_L2_SPEC_BUNDLE",
    "load_section_2_the_synaptic_junction_and_downward_causation_l2_validation_spec",
    "DEFAULT_III_THE_DEVELOPMENTAL_AND_PLASTICITY_LANDSCAPE_L3_SPEC_BUNDLE",
    "load_iii_the_developmental_and_plasticity_landscape_l3_validation_spec",
    "DEFAULT_SECTION_2_CROSS_FREQUENCY_COUPLING_CFC_AND_HIERARCHICAL_PROCESSING_SPEC_BUNDLE",
    "load_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation_spec",
    "DEFAULT_SECTION_3_THE_QUASICRITICAL_BRAIN_SPEC_BUNDLE",
    "load_section_3_the_quasicritical_brain_validation_spec",
    "DEFAULT_SECTION_6_THE_FREQUENCY_HIERARCHY_THETA_GAMMA_COUPLING_AND_HIERARCHICAL_PREDICTI_SPEC_BUNDLE",
    "load_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_validation_spec",
    "DEFAULT_SECTION_1_THE_EMERGENCE_OF_THE_SELF_SSB_AND_THE_STRANGE_LOOP_SPEC_BUNDLE",
    "load_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_validation_spec",
    "DEFAULT_SECTION_4_THE_GEOMETRY_OF_THOUGHT_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE",
    "load_section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation_spec",
    "DEFAULT_VII_PATHOLOGY_THE_DISORDERED_BRAIN_SPEC_BUNDLE",
    "load_vii_pathology_the_disordered_brain_validation_spec",
    "DEFAULT_I_EXAMINATION_OF_THE_DEEP_ARCHITECTURE_OF_THE_QUANTUM_BIOLOGICAL_INTERFA_SPEC_BUNDLE",
    "load_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation_spec",
    "DEFAULT_THE_COHERENT_MILIEU_CSF_AND_THE_GLYMPHATIC_SYSTEM_AS_THE_BRAIN_S_ENTROPY_SPEC_BUNDLE",
    "load_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_validation_spec",
    "DEFAULT_II_EXAMINATION_OF_THE_ARCHITECTURE_OF_STRUCTURE_AND_PLASTICITY_DOMAIN_I_SPEC_BUNDLE",
    "load_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation_spec",
    "DEFAULT_INTRODUCTION_TO_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE",
    "load_introduction_to_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_spec",
    "DEFAULT_METASTABILITY_AND_CHIMAERA_STATES_THE_NUANCE_OF_QUASICRITICALITY_SPEC_BUNDLE",
    "load_metastability_and_chimaera_states_the_nuance_of_quasicriticality_validation_spec",
    "DEFAULT_INTRODUCTION_TO_THE_ARCHITECTURE_OF_THE_CONSCIOUS_SELF_DOMAIN_II_L5_SPEC_BUNDLE",
    "load_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_validation_spec",
    "DEFAULT_THE_CENTRAL_HUBS_OF_BINDING_ORCHESTRATING_UNITY_SPEC_BUNDLE",
    "load_the_central_hubs_of_binding_orchestrating_unity_validation_spec",
    "DEFAULT_THE_NEURO_VISCERAL_AXIS_HEART_BRAIN_GUT_THE_SYMPHONY_OF_THE_SELF_SPEC_BUNDLE",
    "load_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_validation_spec",
    "DEFAULT_INTRODUCTION_TO_THE_CLINICAL_IMPLICATIONS_THE_DISORDERED_BRAIN_AS_A_DISO_SPEC_BUNDLE",
    "load_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation_spec",
    "DEFAULT_SCHIZOPHRENIA_DISSONANCE_SPEC_BUNDLE",
    "load_schizophrenia_dissonance_validation_spec",
    "DEFAULT_ADVANCED_NEUROBIOLOGICAL_IMPLEMENTATION_OF_THE_SCPN_SPEC_BUNDLE",
    "load_advanced_neurobiological_implementation_of_the_scpn_validation_spec",
    "DEFAULT_SECTION_3_NEUROTRANSMITTERS_AS_TUNERS_OF_THE_PSI_FIELD_INTERFACE_L2_SPEC_BUNDLE",
    "load_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_validation_spec",
    "DEFAULT_SECTION_1_THE_BIOELECTRIC_CODE_IN_NEUROGENESIS_AND_REGENERATION_SPEC_BUNDLE",
    "load_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation_spec",
    "DEFAULT_III_THE_DYNAMICS_OF_THE_COHERENT_BRAIN_DOMAIN_I_L4_SPEC_BUNDLE",
    "load_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_validation_spec",
    "DEFAULT_SECTION_3_THE_DYNAMIC_CONNECTOME_AND_FUNCTIONAL_CONNECTIVITY_SPEC_BUNDLE",
    "load_section_3_the_dynamic_connectome_and_functional_connectivity_validation_spec",
    "DEFAULT_SECTION_3_THE_DETAILED_GEOMETRY_OF_QUALIA_THE_CONSCIOUSNESS_MANIFOLD_M_SPEC_BUNDLE",
    "load_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_validation_spec",
    "DEFAULT_VI_CLINICAL_IMPLICATIONS_PATHOLOGY_AND_THERAPEUTICS_SPEC_BUNDLE",
    "load_vi_clinical_implications_pathology_and_therapeutics_validation_spec",
    "DEFAULT_SECTION_2_DENDRITIC_SPINES_THE_LOCI_OF_PLASTICITY_AND_IET_SPEC_BUNDLE",
    "load_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_validation_spec",
    "DEFAULT_SECTION_1_ION_GRADIENTS_THE_ELECTROCHEMICAL_BATTERY_SPEC_BUNDLE",
    "load_section_1_ion_gradients_the_electrochemical_battery_validation_spec",
    "DEFAULT_SECTION_1_THE_LIPID_BILAYER_AND_LIPID_RAFTS_SPEC_BUNDLE",
    "load_section_1_the_lipid_bilayer_and_lipid_rafts_validation_spec",
    "DEFAULT_SECTION_1_THE_CYTOSKELETON_THE_L1_QUANTUM_SCAFFOLD_SPEC_BUNDLE",
    "load_section_1_the_cytoskeleton_the_l1_quantum_scaffold_validation_spec",
    "DEFAULT_SECTION_1_THE_PRESYNAPTIC_TERMINAL_THE_QUANTUM_LEVER_SPEC_BUNDLE",
    "load_section_1_the_presynaptic_terminal_the_quantum_lever_validation_spec",
    "DEFAULT_SECTION_1_THE_LIPID_LANDSCAPE_AND_CRITICALITY_SPEC_BUNDLE",
    "load_section_1_the_lipid_landscape_and_criticality_validation_spec",
    "DEFAULT_SECTION_4_INTERFACES_MEMBRANES_GLIA_STRESSES_AND_SHAPE_CONTROL_SPEC_BUNDLE",
    "load_section_4_interfaces_membranes_glia_stresses_and_shape_control_validation_spec",
    "DEFAULT_II_THE_MOLECULAR_MACHINERY_OF_SIGNALLING_ION_CHANNELS_AND_RECEPTORS_L1_L_SPEC_BUNDLE",
    "load_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation_spec",
    "DEFAULT_III_THE_EXTRACELLULAR_MILIEU_ECM_AND_PNNS_L3_L4_SPEC_BUNDLE",
    "load_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_validation_spec",
    "DEFAULT_IV_SUB_SYNAPTIC_AND_AXONAL_ARCHITECTURE_L1_L3_SPEC_BUNDLE",
    "load_iv_sub_synaptic_and_axonal_architecture_l1_l3_validation_spec",
    "DEFAULT_SECTION_1_THE_CYTOSKELETON_WATER_INTERFACE_AND_QEC_SPEC_BUNDLE",
    "load_section_1_the_cytoskeleton_water_interface_and_qec_validation_spec",
    "DEFAULT_SECTION_4_NUCLEAR_SPIN_AND_POSNER_CLUSTERS_THE_QUANTUM_MEMORY_SUBSTRATE_L1_L9_SPEC_BUNDLE",
    "load_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation_spec",
    "DEFAULT_II_MICRO_SCALE_GEOMETRY_THE_QUANTUM_AND_MOLECULAR_SCAFFOLD_L1_L3_SPEC_BUNDLE",
    "load_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation_spec",
    "DEFAULT_SECTION_2_MOLECULAR_GEOMETRY_AND_THE_PSI_FIELD_INTERFACE_L2_L3_SPEC_BUNDLE",
    "load_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_validation_spec",
    "DEFAULT_SECTION_2_CYTOARCHITECTURE_AND_THE_CANONICAL_MICROCIRCUIT_L4_SPEC_BUNDLE",
    "load_section_2_cytoarchitecture_and_the_canonical_microcircuit_l4_validation_spec",
    "DEFAULT_SECTION_2_THE_GEOMETRY_OF_FUNCTIONAL_DYNAMICS_L4_SPEC_BUNDLE",
    "load_section_2_the_geometry_of_functional_dynamics_l4_validation_spec",
    "DEFAULT_SECTION_2_HPC_AS_GEOMETRIC_FLOW_SPEC_BUNDLE",
    "load_section_2_hpc_as_geometric_flow_validation_spec",
    "DEFAULT_I_THE_BRAIN_S_PROTECTIVE_SCAFFOLD_AND_FLUID_DYNAMICS_SPEC_BUNDLE",
    "load_i_the_brain_s_protective_scaffold_and_fluid_dynamics_validation_spec",
    "DEFAULT_SECTION_3_THE_CSF_AND_GLYMPHATIC_SYSTEM_THE_ENTROPY_SINK_L1_L4_SPEC_BUNDLE",
    "load_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation_spec",
    "DEFAULT_SECTION_1_THE_MECHANISM_OF_NVC_SPEC_BUNDLE",
    "load_section_1_the_mechanism_of_nvc_validation_spec",
    "DEFAULT_SECTION_3_PATHOLOGY_VASCULAR_DYSFUNCTION_SPEC_BUNDLE",
    "load_section_3_pathology_vascular_dysfunction_validation_spec",
    "DEFAULT_SECTION_2_THE_GUT_BRAIN_AXIS_GBA_AND_THE_MICROBIOME_THE_DEEP_MILIEU_SPEC_BUNDLE",
    "load_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_validation_spec",
    "DEFAULT_IV_THE_NEURO_IMMUNO_ENDOCRINE_NIE_SUPER_SYSTEM_SPEC_BUNDLE",
    "load_iv_the_neuro_immuno_endocrine_nie_super_system_validation_spec",
    "DEFAULT_SECTION_2_THE_ENDOCRINE_SYSTEM_AND_HPA_AXIS_STRESS_RESPONSE_SPEC_BUNDLE",
    "load_section_2_the_endocrine_system_and_hpa_axis_stress_response_validation_spec",
    "DEFAULT_V_THE_INTEGRATED_BODY_MATRIX_FASCIA_AND_TENSEGRITY_SPEC_BUNDLE",
    "load_v_the_integrated_body_matrix_fascia_and_tensegrity_validation_spec",
    "DEFAULT_THE_DYNAMICS_AND_EVOLUTION_OF_THE_EMBODIED_SCPN_SPEC_BUNDLE",
    "load_the_dynamics_and_evolution_of_the_embodied_scpn_validation_spec",
    "DEFAULT_SECTION_2_EMBODIED_PREDICTIVE_CODING_INTEROCEPTIVE_INFERENCE_SPEC_BUNDLE",
    "load_section_2_embodied_predictive_coding_interoceptive_inference_validation_spec",
    "DEFAULT_II_THE_CHRONOBIOLOGICAL_ARCHITECTURE_TEMPORAL_SYNCHRONISATION_L4_L8_SPEC_BUNDLE",
    "load_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_validation_spec",
    "DEFAULT_SECTION_3_PATHOLOGY_CHRONODISRUPTION_SPEC_BUNDLE",
    "load_section_3_pathology_chronodisruption_validation_spec",
    "DEFAULT_SECTION_2_SPECIALISED_SENSORY_SYSTEMS_SPEC_BUNDLE",
    "load_section_2_specialised_sensory_systems_validation_spec",
    "DEFAULT_IV_LEARNING_MEMORY_AND_PLASTICITY_THE_ADAPTIVE_SCAFFOLD_L1_L9_SPEC_BUNDLE",
    "load_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation_spec",
    "DEFAULT_SECTION_3_THE_HOLOGRAPHIC_INTERFACE_L9_SPEC_BUNDLE",
    "load_section_3_the_holographic_interface_l9_validation_spec",
    "DEFAULT_SECTION_3_SLEEP_AND_DREAMING_THE_OPTIMISATION_CYCLE_SPEC_BUNDLE",
    "load_section_3_sleep_and_dreaming_the_optimisation_cycle_validation_spec",
    "DEFAULT_SECTION_1_PSYCHEDELICS_THE_EXPANDED_MANIFOLD_SPEC_BUNDLE",
    "load_section_1_psychedelics_the_expanded_manifold_validation_spec",
    "DEFAULT_VIII_THE_EVOLUTIONARY_TRAJECTORY_OF_THE_BRAIN_BODY_SYSTEM_SPEC_BUNDLE",
    "load_viii_the_evolutionary_trajectory_of_the_brain_body_system_validation_spec",
    "DEFAULT_CLINICAL_SCENARIO_ANALYSIS_TRAUMATIC_BRAIN_INJURY_TBI_AND_PHARMACOLOGICA_SPEC_BUNDLE",
    "load_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_validation_spec",
    "DEFAULT_SECTION_2_L2_L3_DISRUPTION_EXCITOTOXICITY_AND_STRUCTURAL_FAILURE_SPEC_BUNDLE",
    "load_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_validation_spec",
    "DEFAULT_SECTION_4_L5_DISRUPTION_THE_FRAGMENTED_SELF_AND_DISSONANCE_SPEC_BUNDLE",
    "load_section_4_l5_disruption_the_fragmented_self_and_dissonance_validation_spec",
    "DEFAULT_II_THE_PHYSICS_OF_PAIN_WITHIN_THE_SCPN_SPEC_BUNDLE",
    "load_ii_the_physics_of_pain_within_the_scpn_validation_spec",
    "DEFAULT_SECTION_2_L4_IMPACT_DAMPENING_DYNAMICS_AND_SHIFTING_CRITICALITY_SPEC_BUNDLE",
    "load_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_validation_spec",
    "DEFAULT_IV_THE_INTEGRATED_SCENARIO_PHARMACOLOGICAL_MODULATION_OF_THE_TRAUMATISED_SPEC_BUNDLE",
    "load_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_validation_spec",
    "DEFAULT_SECTION_1_PROPOFOL_GABA_A_POTENTIATION_SPEC_BUNDLE",
    "load_section_1_propofol_gaba_a_potentiation_validation_spec",
    "DEFAULT_MECHANISMS_OF_CRITICALITY_AND_CONTROL_LAYERS_1_4_SPEC_BUNDLE",
    "load_mechanisms_of_criticality_and_control_layers_1_4_validation_spec",
    "DEFAULT_PREDICTION_III_TOPOS_THEORETIC_COGNITIVE_HESITATION_THE_OMEGA_STATE_SPEC_BUNDLE",
    "load_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation_spec",
    "DEFAULT_META_FRAMEWORK_INTEGRATIONS_P0R05143_SPEC_BUNDLE",
    "load_meta_framework_integrations_p0r05143_validation_spec",
    "DEFAULT_PREDICTION_I_NV_MEA_TESTS_THE_INFORMATIONAL_COUPLING_LINFORMATIONAL_SPEC_BUNDLE",
    "load_prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation_spec",
    "DEFAULT_NOVEL_FALSIFIABLE_PREDICTIONS_FROM_FIRST_PRINCIPLES_SPEC_BUNDLE",
    "load_novel_falsifiable_predictions_from_first_principles_validation_spec",
    "DEFAULT_PREDICTED_SIGNATURE_SPEC_BUNDLE",
    "load_predicted_signature_validation_spec",
    "DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_NV_CENTER_QUANTUM_SENSING_OF_NEURONAL_CUL_SPEC_BUNDLE",
    "load_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_validation_spec",
    "DEFAULT_PROTOCOL_SPEC_BUNDLE",
    "load_protocol_validation_spec",
    "DEFAULT_PREDICTION_II_CAUSAL_ENTROPIC_FORCE_SIGNATURES_IN_QUANTUM_RANDOMNESS_SPEC_BUNDLE",
    "load_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation_spec",
    "DEFAULT_PROPOSED_EXPERIMENTAL_PROTOCOL_CORRELATING_QRNG_STATISTICAL_DEVIATIONS_W_SPEC_BUNDLE",
    "load_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation_spec",
    "DEFAULT_FALSIFICATION_CONDITION_SPEC_BUNDLE",
    "load_falsification_condition_validation_spec",
    "DEFAULT_FORMALISING_THE_NTHS_WITHIN_A_MULTI_AGENT_ACTIVE_INFERENCE_FRAMEWORK_SPEC_BUNDLE",
    "load_formalising_the_nths_within_a_multi_agent_active_inference_framework_validation_spec",
    "DEFAULT_SIMULATION_ARCHITECTURE_AND_IMPLEMENTATION_SPEC_BUNDLE",
    "load_simulation_architecture_and_implementation_validation_spec",
    "DEFAULT_EXPERIMENTAL_DESIGN_COHERENCE_VS_ENGAGEMENT_OPTIMISATION_SPEC_BUNDLE",
    "load_experimental_design_coherence_vs_engagement_optimisation_validation_spec",
    "DEFAULT_PREDICTED_SIGNATURES_AND_ANALYSIS_PROTOCOL_SPEC_BUNDLE",
    "load_predicted_signatures_and_analysis_protocol_validation_spec",
    "DEFAULT_TABLE_1_PREDICTED_NTHS_PHASE_CHARACTERISTICS_IN_MULTI_AGENT_ACTIVE_INFER_SPEC_BUNDLE",
    "load_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation_spec",
    "DEFAULT_SECTION_9_CONCLUDING_ASSESSMENT_AND_FUTURE_DIRECTIONS_SPEC_BUNDLE",
    "load_section_9_concluding_assessment_and_future_directions_validation_spec",
    "DEFAULT_THE_AQUEOUS_SUBSTRATE_THE_ROLE_OF_INTERFACIAL_WATER_AND_COHERENCE_DOMAIN_SPEC_BUNDLE",
    "load_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation_spec",
    "DEFAULT_THE_FOUNDATION_OF_THE_BIOLOGICAL_SUBSTRATE_SPEC_BUNDLE",
    "load_the_foundation_of_the_biological_substrate_validation_spec",
    "DEFAULT_THE_QUANTUM_ENGINE_LAYERS_1_2_WE_BEGIN_WITH_THE_QUANTUM_BIOLOGICAL_LAYER_SPEC_BUNDLE",
    "load_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation_spec",
    "DEFAULT_THE_AMPLIFICATION_CASCADE_SPEC_BUNDLE",
    "load_the_amplification_cascade_validation_spec",
    "DEFAULT_THE_AQUEOUS_SUBSTRATE_DOMAIN_I_INTERFACE_SPEC_BUNDLE",
    "load_the_aqueous_substrate_domain_i_interface_validation_spec",
    "DEFAULT_THE_SLOW_CONTROL_LAYER_GLIAL_AND_IMMUNE_MODULATION_SPEC_BUNDLE",
    "load_the_slow_control_layer_glial_and_immune_modulation_validation_spec",
    "DEFAULT_II_THE_QUANTUM_IMMUNE_INTERFACE_L1_L2_L5_INTEGRATION_SPEC_BUNDLE",
    "load_ii_the_quantum_immune_interface_l1_l2_l5_integration_validation_spec",
    "DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_SPEC_BUNDLE",
    "load_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation_spec",
    "DEFAULT_THE_SLOW_CONTROL_NETWORK_GLIAL_HOMEOSTASIS_AND_NEURONAL_CRITICALITY_SPEC_BUNDLE",
    "load_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_validation_spec",
    "DEFAULT_THE_GLIAL_NEURONAL_COUPLING_MECHANISM_SLOW_CONTROL_OF_NEURONAL_CRITICALI_P0R05390_SPEC_BUNDLE",
    "load_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_validation_spec",
    "DEFAULT_THE_THERMODYNAMIC_LIMIT_OF_CONTROL_THE_ALLOSTATIC_BOUND_SPEC_BUNDLE",
    "load_the_thermodynamic_limit_of_control_the_allostatic_bound_validation_spec",
    "DEFAULT_THE_ELECTRODYNAMIC_INTERFACE_OF_CONSCIOUSNESS_CEMI_AND_IIIEF_SPEC_BUNDLE",
    "load_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_validation_spec",
    "DEFAULT_SLOW_CONTROL_LAYER_NEUROENDOCRINE_INTEGRATION_SPEC_BUNDLE",
    "load_slow_control_layer_neuroendocrine_integration_validation_spec",
    "DEFAULT_NEUROENDOCRINE_REGULATION_AND_THE_HYPOTHALAMIC_PITUITARY_ADRENAL_HPA_AXI_SPEC_BUNDLE",
    "load_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_validation_spec",
    "DEFAULT_GLIAL_SCAFFOLDING_ASTROCYTIC_REGULATION_OF_NEURAL_SYNCHRONY_AND_CRITICAL_SPEC_BUNDLE",
    "load_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_validation_spec",
    "DEFAULT_THE_MICROBIOME_AS_A_FOUNDATIONAL_CONTROL_LAYER_SPEC_BUNDLE",
    "load_the_microbiome_as_a_foundational_control_layer_validation_spec",
    "DEFAULT_SCALE_INVARIANT_CYBERNETIC_PRINCIPLE_SPEC_BUNDLE",
    "load_scale_invariant_cybernetic_principle_validation_spec",
    "DEFAULT_QUANTUM_ENZYMOLOGY_OF_THE_IMMUNE_RESPONSE_SPEC_BUNDLE",
    "load_quantum_enzymology_of_the_immune_response_validation_spec",
    "DEFAULT_APPLICATION_TO_IMMUNE_ENZYMES_SPEC_BUNDLE",
    "load_application_to_immune_enzymes_validation_spec",
    "DEFAULT_MACRO_SCALE_HOMEOSTASIS_GAIAN_SYNCHRONY_AND_NICHE_CONSTRUCTION_SPEC_BUNDLE",
    "load_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_validation_spec",
    "DEFAULT_DOMAIN_II_ORGANISMAL_AND_PLANETARY_INTEGRATION_LAYERS_5_8_SPEC_BUNDLE",
    "load_domain_ii_organismal_and_planetary_integration_layers_5_8_validation_spec",
    "DEFAULT_DOMAIN_III_IV_MEMORY_CONTROL_AND_COLLECTIVE_COHERENCE_LAYERS_9_12_SPEC_BUNDLE",
    "load_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_validation_spec",
    "DEFAULT_CITATIONS_SPEC_BUNDLE",
    "load_citations_validation_spec",
    "DEFAULT_DOMAIN_V_META_UNIVERSAL_INTEGRATION_LAYERS_13_15_SPEC_BUNDLE",
    "load_domain_v_meta_universal_integration_layers_13_15_validation_spec",
    "DEFAULT_DOMAIN_VI_CYBERNETIC_CLOSURE_META_LAYER_16_SPEC_BUNDLE",
    "load_domain_vi_cybernetic_closure_meta_layer_16_validation_spec",
    "DEFAULT_RESOLVING_THE_OBSERVABILITY_PARADOX_L16_AS_A_POMDP_AND_THE_BELIEF_STATE_SPEC_BUNDLE",
    "load_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation_spec",
    "DEFAULT_CITATIONS_P0R05625_SPEC_BUNDLE",
    "load_citations_p0r05625_validation_spec",
    "DEFAULT_DOMAIN_INTERFACES_AND_RENORMALISATION_SPEC_BUNDLE",
    "load_domain_interfaces_and_renormalisation_validation_spec",
    "DEFAULT_SECTION_2_IMPEDANCE_MATCHING_SPEC_BUNDLE",
    "load_section_2_impedance_matching_validation_spec",
    "DEFAULT_BIOPHYSICS_QUANTUM_BIOLOGY_SPEC_BUNDLE",
    "load_biophysics_quantum_biology_validation_spec",
    "DEFAULT_COSMOLOGY_PHYSICS_EXTENSIONS_SPEC_BUNDLE",
    "load_cosmology_physics_extensions_validation_spec",
    "DEFAULT_QUANTUM_BIOPHYSICAL_FOUNDATIONS_SPEC_BUNDLE",
    "load_quantum_biophysical_foundations_validation_spec",
    "DEFAULT_QUANTUM_FOUNDATIONS_COLLAPSE_SPEC_BUNDLE",
    "load_quantum_foundations_collapse_validation_spec",
    "DEFAULT_QUANTUM_GRAVITATION_EDGE_CASES_SPEC_BUNDLE",
    "load_quantum_gravitation_edge_cases_validation_spec",
    "DEFAULT_SYSTEMS_NEUROSCIENCE_COMPLEXITY_SPEC_BUNDLE",
    "load_systems_neuroscience_complexity_validation_spec",
    "DEFAULT_INFORMATION_GEOMETRY_ACTIVE_INFERENCE_SPEC_BUNDLE",
    "load_information_geometry_active_inference_validation_spec",
    "DEFAULT_COLLECTIVE_CULTURAL_LAYERS_SPEC_BUNDLE",
    "load_collective_cultural_layers_validation_spec",
    "DEFAULT_THERMODYNAMICS_ENTROPY_SPEC_BUNDLE",
    "load_thermodynamics_entropy_validation_spec",
    "DEFAULT_NONLINEAR_DYNAMICS_CRITICALITY_SPEC_BUNDLE",
    "load_nonlinear_dynamics_criticality_validation_spec",
    "DEFAULT_LINGUISTICS_SYMBOLISM_VIBRANA_LAYER_7_SPEC_BUNDLE",
    "load_linguistics_symbolism_vibrana_layer_7_validation_spec",
    "DEFAULT_ECOLOGY_GAIA_EXTENSIONS_SPEC_BUNDLE",
    "load_ecology_gaia_extensions_validation_spec",
    "DEFAULT_ETHICS_PHILOSOPHY_SPEC_BUNDLE",
    "load_ethics_philosophy_validation_spec",
    "DEFAULT_ETHICS_TELEOLOGY_SPEC_BUNDLE",
    "load_ethics_teleology_validation_spec",
    "DEFAULT_PHILOSOPHY_ETHICS_ANCHORS_SPEC_BUNDLE",
    "load_philosophy_ethics_anchors_validation_spec",
    "DEFAULT_PHILOSOPHY_OF_INFORMATION_MIND_SPEC_BUNDLE",
    "load_philosophy_of_information_mind_validation_spec",
    "DEFAULT_PHILOSOPHY_CONSCIOUSNESS_SPEC_BUNDLE",
    "load_philosophy_consciousness_validation_spec",
    "DEFAULT_MATHEMATICS_GEOMETRY_SPEC_BUNDLE",
    "load_mathematics_geometry_validation_spec",
    "DEFAULT_AI_NOOSPHERE_TECH_SPEC_BUNDLE",
    "load_ai_noosphere_tech_validation_spec",
    "DEFAULT_NETWORK_COMPLEXITY_SCIENCE_SPEC_BUNDLE",
    "load_network_complexity_science_validation_spec",
    "DEFAULT_COLLECTIVE_CULTURAL_AND_EVOLUTIONARY_DYNAMICS_SPEC_BUNDLE",
    "load_collective_cultural_and_evolutionary_dynamics_validation_spec",
    "DEFAULT_MATHEMATICS_OF_DYNAMICAL_SYSTEMS_SPEC_BUNDLE",
    "load_mathematics_of_dynamical_systems_validation_spec",
    "DEFAULT_MATHEMATICAL_FOUNDATIONS_OF_NETWORKS_SYNCHRONISATION_SPEC_BUNDLE",
    "load_mathematical_foundations_of_networks_synchronisation_validation_spec",
    "DEFAULT_SYSTEMS_CYBERNETICS_SPEC_BUNDLE",
    "load_systems_cybernetics_validation_spec",
    "DEFAULT_COMPUTATIONAL_AI_ALIGNMENT_SPEC_BUNDLE",
    "load_computational_ai_alignment_validation_spec",
    "DEFAULT_CONSCIOUSNESS_STUDIES_COGNITIVE_MODELS_SPEC_BUNDLE",
    "load_consciousness_studies_cognitive_models_validation_spec",
    "DEFAULT_TECHNO_SOCIAL_SYSTEMS_SPEC_BUNDLE",
    "load_techno_social_systems_validation_spec",
    "DEFAULT_TOPOLOGY_GEOMETRY_IN_CONSCIOUSNESS_MODELS_SPEC_BUNDLE",
    "load_topology_geometry_in_consciousness_models_validation_spec",
    "DEFAULT_COMPLEXITY_ECONOMICS_SOCIAL_PHYSICS_SPEC_BUNDLE",
    "load_complexity_economics_social_physics_validation_spec",
    "DEFAULT_EXTENDED_COGNITION_EMBODIMENT_SPEC_BUNDLE",
    "load_extended_cognition_embodiment_validation_spec",
    "DEFAULT_TIME_RETROCAUSALITY_AND_TWO_STATE_VECTOR_SPEC_BUNDLE",
    "load_time_retrocausality_and_two_state_vector_validation_spec",
    "DEFAULT_GAIA_BIOSPHERE_INTELLIGENCE_SPEC_BUNDLE",
    "load_gaia_biosphere_intelligence_validation_spec",
    "DEFAULT_OVERARCHING_PRINCIPLES_AND_SYSTEM_DYNAMICS_SPEC_BUNDLE",
    "load_overarching_principles_and_system_dynamics_validation_spec",
]
