// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR scalar forward tests

#[test]
fn program_ad_effect_ir_rust_interpreter_executes_opcode_bearing_scalar_subset() {
    let result =
        interpret_program_ad_effect_ir_forward(EXECUTABLE_SCALAR_PROGRAM_AD_IR, &[0.4, -0.2])
            .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 7);
    assert_eq!(result.supported_effect_count, 7);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_interpreter_replays_executed_branch_metadata() {
    let result =
        interpret_program_ad_effect_ir_forward(EXECUTED_BRANCH_PROGRAM_AD_IR, &[0.4, -0.2])
            .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_interpreter_fails_closed_without_operation_metadata() {
    let legacy_ir = EXECUTABLE_SCALAR_PROGRAM_AD_IR
        .replace(", \"operation\": \"parameter\"", "")
        .replace(", \"operation\": \"mul\"", "")
        .replace(", \"operation\": \"add\"", "")
        .replace(", \"operation\": \"sin\"", "");
    let result = interpret_program_ad_effect_ir_forward(&legacy_ir, &[0.4, -0.2]).unwrap();

    assert!(!result.supported);
    assert_eq!(result.value, None);
    assert_eq!(result.supported_effect_count, 0);
    assert!(result.blocked_reasons[0].contains("operation metadata"));
}
