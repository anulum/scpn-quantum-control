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

#[test]
fn public_forward_admits_ordering_and_owned_symbols_with_capacity_recovery() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_metadata_admission;
    use std::{cell::RefCell, rc::Rc};
    let symbol = format!("%{}", "a".repeat(65_536));
    let source = EXECUTABLE_SCALAR_PROGRAM_AD_IR.replace("%0", &symbol);
    let parsed_requests = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&parsed_requests);
    let ir = with_replay_metadata_admission(
        move |bytes| { captured.borrow_mut().push(bytes); Ok(()) },
        || parse_program_ad_effect_ir(&source),
    ).unwrap();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&requests);
    let actual = with_replay_metadata_admission(
        move |bytes| { captured.borrow_mut().push(bytes); Ok(()) },
        || interpret_program_ad_effect_ir_forward(&source, &[0.4, -0.2]),
    ).unwrap();
    let expected = 0.4_f64.powi(2) - 0.4 + 0.4_f64.sin();
    assert!(actual.supported);
    assert!((actual.value.unwrap() - expected).abs() < 1e-12);
    let requests = requests.borrow();
    let parser_count = parsed_requests.borrow().len();
    assert_eq!(&requests[..parser_count], parsed_requests.borrow().as_slice());
    let ordering_bytes = ir.effects.len() * (
        std::mem::size_of::<(usize, &scpn_quantum_program_ad_replay::program_ad_ir::ProgramADEffect)>()
        + std::mem::size_of::<&scpn_quantum_program_ad_replay::program_ad_ir::ProgramADEffect>()
    );
    assert_eq!(requests[parser_count], ordering_bytes);
    assert!(requests[parser_count + 1..].contains(&symbol.len()));
    let required: usize = requests.iter().sum();
    for limit in [required, required - 1] {
        let charged = std::cell::Cell::new(0usize);
        let result = with_replay_metadata_admission(
            move |bytes| {
                let total = charged.get().checked_add(bytes).unwrap();
                if total > limit { return Err("forward metadata limit".to_owned()); }
                charged.set(total); Ok(())
            },
            || interpret_program_ad_effect_ir_forward(&source, &[0.4, -0.2]),
        );
        if limit == required { assert_eq!(result.unwrap(), actual); }
        else { assert_eq!(result.unwrap_err(), "forward metadata limit"); }
    }
    assert_eq!(interpret_program_ad_effect_ir_forward(&source, &[0.4, -0.2]).unwrap(), actual);
}
