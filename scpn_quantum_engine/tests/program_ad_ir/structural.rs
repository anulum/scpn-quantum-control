// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR structural replay tests

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_structural_array_ops() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STRUCTURAL_ARRAY_PROGRAM_AD_IR,
        &[2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    let expected_gradient = [
        15.0,
        20.0,
        2.0 / 6.0,
        5.0 / 6.0,
        2.0 / 6.0,
        5.0 / 6.0,
        2.0 / 6.0,
        5.0 / 6.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(130.0));
    assert_eq!(
        result.parameter_targets,
        vec!["%0[0]", "%0[1]", "%1[0]", "%1[1]", "%1[2]", "%1[3]", "%1[4]", "%1[5]"]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_structural_assembly_ops() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR,
        &[
            2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
        ],
    )
    .unwrap();

    let expected_gradient = [
        11.0, 32.0, 23.0, 44.0, 2.0, 5.0, 7.0, 11.0, 2.0, 7.0, 5.0, 11.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(827.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%1[0]", "%1[1]", "%2[0]", "%2[1]", "%2[2]", "%2[3]", "%3[0]",
            "%3[1]", "%3[2]", "%3[3]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 11);
    assert_eq!(result.supported_effect_count, 11);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_assembly_without_axis_metadata() {
    let missing_axis = STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR.replace(
        "\"operation\": \"concatenate:axis:0\"",
        "\"operation\": \"concatenate\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_axis,
        &[
            2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
        ],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static axis metadata")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_axis_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_AXIS_REDUCTION_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
    )
    .unwrap();

    let expected_gradient = [
        10.0 + 7.0 / 3.0,
        20.0 + 7.0 / 3.0,
        30.0 + 7.0 / 3.0,
        10.0 + 11.0 / 3.0,
        20.0 + 11.0 / 3.0,
        30.0 + 11.0 / 3.0,
        5.0,
        7.0,
        9.0,
        2.0,
        5.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(529.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_shaped_reduction_without_axis_metadata() {
    let missing_axis = STATIC_AXIS_REDUCTION_PROGRAM_AD_IR
        .replace("\"operation\": \"sum:axis:0\"", "\"operation\": \"sum\"");
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_axis,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static axis metadata")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_source_map_indexing() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    let expected_gradient = [20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(400.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%1[0]", "%1[1]", "%1[2]", "%1[3]", "%1[4]",
            "%1[5]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 5);
    assert_eq!(result.supported_effect_count, 5);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_source_map_inert_source_alias_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"assignment_binding\", \"target\": \"source:2\", \"kind\": \"alias_analysis\", \"version\": 0}, {\"source\": \"expr:2:np.take(source,_[2,_0,_2])\", \"target\": \"name:gathered\", \"kind\": \"expression_rebinding_alias\", \"version\": 1}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(400.0));
    assert_eq!(result.effect_count, 5);
    assert_eq!(result.supported_effect_count, 5);
    assert_eq!(
        result.gradient,
        vec![20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0]
    );
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_malformed_alias_analysis_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"dynamic_binding\", \"target\": \"source:2\", \"kind\": \"alias_analysis\", \"version\": 0}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_malformed_expression_alias_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"expr:2:np.take(source,_[2,_0,_2])\", \"target\": \"slot:gathered\", \"kind\": \"expression_rebinding_alias\", \"version\": 0}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_source_map_without_metadata() {
    let missing_map = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"operation\": \"index_map:s2,s0,s2,c-1.5,s3,s1\"",
        "\"operation\": \"index_map\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_map,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static source-map metadata")));
}
