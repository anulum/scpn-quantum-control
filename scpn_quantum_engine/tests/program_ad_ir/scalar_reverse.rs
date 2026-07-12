// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR scalar reverse tests

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_scalar_reverse_subset() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        EXECUTABLE_SCALAR_PROGRAM_AD_IR,
        &[0.4, -0.2],
    )
    .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 7);
    assert_eq!(result.supported_effect_count, 7);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - (2.0_f64 * 0.4_f64 + 0.4_f64.cos())).abs() <= 1.0e-12);
    assert!((result.gradient[1] - 2.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.parameter_targets, vec!["%0", "%1"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_executed_branch_metadata() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        EXECUTED_BRANCH_PROGRAM_AD_IR,
        &[0.4, -0.2],
    )
    .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - (2.0_f64 * 0.4_f64 + 0.4_f64.cos())).abs() <= 1.0e-12);
    assert!((result.gradient[1] - 2.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.parameter_targets, vec!["%0", "%1"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_scalar_primitive_family() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        SCALAR_PRIMITIVE_FAMILY_PROGRAM_AD_IR,
        &[0.4, -0.2, 0.25, 0.1],
    )
    .unwrap();

    let x: f64 = 0.4;
    let y: f64 = -0.2;
    let z: f64 = 0.25;
    let w: f64 = 0.1;
    let expected = (x + 2.0).sqrt()
        + y.tanh()
        + z.ln_1p()
        + w.exp_m1()
        + 1.0 / (x + 3.0)
        + (0.2 * y).asin()
        + (0.1 * z).acos()
        + (w + 1.0).abs();
    let expected_gradient = [
        0.5 / (x + 2.0).sqrt() - 1.0 / ((x + 3.0) * (x + 3.0)),
        1.0 - y.tanh() * y.tanh() + 0.2 / (1.0 - (0.2 * y) * (0.2 * y)).sqrt(),
        1.0 / (1.0 + z) - 0.1 / (1.0 - (0.1 * z) * (0.1 * z)).sqrt(),
        w.exp() + 1.0,
    ];

    assert!(result.supported);
    assert_eq!(result.effect_count, 24);
    assert_eq!(result.supported_effect_count, 24);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 4);
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.parameter_targets, vec!["%0", "%1", "%2", "%3"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_array_elementwise_broadcast_sum() {
    let inputs = [0.2_f64, -0.3, 0.5, 1.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR,
        &inputs,
    )
    .unwrap();
    let x = [inputs[0], inputs[1], inputs[2]];
    let bias = inputs[3];
    let expected_value: f64 = x.iter().map(|value| value.sin() * (value + bias)).sum();
    let expected_gradient = [
        x[0].cos() * (x[0] + bias) + x[0].sin(),
        x[1].cos() * (x[1] + bias) + x[1].sin(),
        x[2].cos() * (x[2] + bias) + x[2].sin(),
        x.iter().map(|value| value.sin()).sum(),
    ];

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.effect_count, 6);
    assert_eq!(result.supported_effect_count, 6);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec!["%0[0]", "%0[1]", "%0[2]", "%1"]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_vector_objective() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR,
        &[0.2_f64, -0.3],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.value.is_none());
    assert!(result.gradient.is_empty());
    assert_eq!(result.effect_count, 2);
    assert_eq!(result.supported_effect_count, 2);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires a scalar objective")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_abs_cusp() {
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(ABS_CUSP_PROGRAM_AD_IR, &[0.0]).unwrap();

    assert!(!result.supported);
    assert_eq!(result.value, None);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("abs gradient is undefined at zero"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_inert_view_alias() {
    // A reshape/transpose/slice view leaves an inert view_alias edge while the op-effects
    // keep referencing canonical scalar SSA, so the bounded Rust replay stays exact.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        INERT_VIEW_ALIAS_PROGRAM_AD_IR,
        &[3.0, -2.0],
    )
    .unwrap();

    assert!(result.supported);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - 13.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - 6.0_f64).abs() <= 1.0e-12);
    assert!((result.gradient[1] - (-4.0_f64)).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_mutation_alias() {
    // Non-view alias kinds (here a mutation_version edge) can change a value's content and
    // stay outside the bounded scalar replay.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        MUTATION_ALIAS_PROGRAM_AD_IR,
        &[2.0, 3.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}
