# Compiler Alias-Activity Evidence

- artifact_id: `compiler-alias-activity-evidence-20260706`
- source_commit: `534010da`
- classification: `functional_non_isolated`
- promotion_ready: `False`
- alias_activity_verified: `True`
- complete lattice cases: `5`
- blocked lattice cases: `3`

Observed alias-edge kinds:

- `alias_analysis`
- `control_path_alias`
- `expression_rebinding_alias`
- `list_alias`
- `local_rebinding_alias`
- `loop_carried_state`
- `mutation_version`
- `object_attribute_alias`
- `view_alias`

| Case | Status | Alias kinds | Blockers |
|---|---|---|---|
| `blocked_branch_attribute_alias` | `blocked_lattice` | `alias_analysis`, `control_path_alias`, `expression_rebinding_alias`, `object_attribute_alias` | `control_path_aliases_require_branch_semantics`, `non_executed_phi_inputs_require_branch_semantics` |
| `blocked_non_executed_branch_alias` | `blocked_lattice` | `alias_analysis`, `control_path_alias`, `expression_rebinding_alias` | `control_path_aliases_require_branch_semantics`, `non_executed_phi_inputs_require_branch_semantics` |
| `blocked_static_slice_mutation_alias` | `blocked_lattice` | `alias_analysis`, `expression_rebinding_alias`, `mutation_version`, `view_alias` | `mutation_effects_require_versioned_alias_semantics` |
| `complete_list_alias` | `complete_lattice` | `alias_analysis`, `expression_rebinding_alias`, `list_alias`, `local_rebinding_alias` | none |
| `complete_loop_carried_state_alias` | `complete_lattice` | `alias_analysis`, `expression_rebinding_alias`, `loop_carried_state` | none |
| `complete_object_attribute_alias` | `complete_lattice` | `alias_analysis`, `expression_rebinding_alias`, `object_attribute_alias` | none |
| `complete_scalar_rebinding_alias` | `complete_lattice` | `alias_analysis`, `expression_rebinding_alias`, `local_rebinding_alias` | none |
| `complete_view_alias` | `complete_lattice` | `alias_analysis`, `expression_rebinding_alias`, `object_attribute_alias`, `view_alias` | none |

Focused test evidence:

- `tests/test_program_ad_alias_effects.py::test_program_ad_alias_effect_analysis_tracks_static_slice_mutation`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_blocks_non_executed_attribute_paths`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_records_non_executed_phi_blockers`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_complete_emitted_ir`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_list_alias_provenance`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_loop_carried_state_provenance`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_reports_rebinding_provenance`
- `tests/test_program_ad_alias_effects.py::test_program_ad_static_alias_lattice_tracks_local_object_attribute_aliases`

Claim boundary: Compiler alias-activity evidence only: Program AD static alias-lattice reports are executed locally for bounded view, list, scalar-rebinding, object-attribute, loop-carried, branch-control, and mutation-version cases; this does not promote general compiler AD, isolated benchmark, provider, hardware, GPU, or performance claim.
