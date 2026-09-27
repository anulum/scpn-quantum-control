// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public diagonal admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

fn diagonal_ir(operation: &str, count: usize) -> String {
    let mut values = Vec::new();
    let mut effects = Vec::new();
    let mut inputs = Vec::new();
    for index in 0..count {
        let name = format!("%{index}");
        inputs.push(name.clone());
        values.push(serde_json::json!({"name":name,"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
        effects.push(serde_json::json!({"index":index,"kind":"parameter","target":name,"inputs":[format!("x{index}")],"version":0,"ordering":index,"operation":"parameter"}));
    }
    let target = format!("%{count}");
    values.push(serde_json::json!({"name":target,"producer":count,"version":0,"shape":[],"dtype":"float64","effect":count}));
    effects.push(serde_json::json!({"index":count,"kind":"primitive","target":target,"inputs":inputs,"version":0,"ordering":count,"operation":operation}));
    serde_json::json!({"format":"program_ad_effect_ir.v1","ssa_values":values,"effects":effects,"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}).to_string()
}

#[test]
fn public_diagonal_identity_offsets_and_metadata_cancellation_recover() {
    for operation in [
        "linalg:diag:3:offset:0:construct:2",
        "linalg:diag:3:offset:2:construct:1",
        "linalg:diag:3:offset:-2:construct:1",
        "linalg:diag:3x4:offset:1:extract:2",
        "linalg:diag:3x4:offset:-1:extract:1",
        "linalg:diag:5x2:offset:-3:extract:1",
        "linalg:diag:100000000x1:offset:0:extract:0",
        "linalg:diagflat:2x3:offset:0:construct:5",
        "linalg:diagflat:2x3:offset:2:construct:4",
        "linalg:diagflat:2x3:offset:-2:construct:4",
    ] {
        // The compact primitive receives the selected scalar, not a dense matrix.
        let source=diagonal_ir(operation,1);
        let calls=Rc::new(Cell::new(0usize));
        let recorded=Rc::clone(&calls);
        let baseline=with_replay_checkpoint(
            move || { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&source,&[7.0]),
        ).unwrap();
        assert!(baseline.supported,"{:?}",baseline.blocked_reasons);
        assert_eq!(baseline.value,Some(7.0));
        assert_eq!(baseline.gradient,vec![1.0]);
        assert!(calls.get()>1);
        for boundary in 1..=calls.get() {
            let observed=Rc::new(Cell::new(0usize));
            let recorded=Rc::clone(&observed);
            let refused=with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get()+1);
                    if recorded.get()>=boundary { Err("diagonal owner cancelled".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source,&[7.0]),
            );
            match refused {
                Err(reason)=>assert!(reason.contains("diagonal owner cancelled"),"{reason}"),
                Ok(result)=>{
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|r|r.contains("diagonal owner cancelled")),"{:?}",result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry=interpret_program_ad_effect_ir_value_and_gradient(&source,&[7.0]).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value,baseline.value);
            assert_eq!(retry.gradient,baseline.gradient);
        }
    }
}

#[test]
fn public_diagonal_absurd_dense_shapes_offsets_and_empty_selections_refuse_then_retry() {
    for operation in [
        format!("linalg:diag:{}:offset:1:construct:0",usize::MAX),
        format!("linalg:diag:{}x2:offset:0:extract:0",usize::MAX),
        format!("linalg:diag:2:offset:{}:construct:0",i64::MIN),
        format!("linalg:diag:2:offset:{}:construct:0",i64::MAX),
        "linalg:diag:3x4:offset:4:extract:0".to_owned(),
        "linalg:diag:3x4:offset:-3:extract:0".to_owned(),
        format!("linalg:diag:3x4:offset:{}:extract:0",i64::MIN),
        "linalg:diag:3x4:offset:-1:extract:2".to_owned(),
        "linalg:diag:3:offset:0:construct:3".to_owned(),
        "linalg:diag:1x2x3:offset:0:extract:0".to_owned(),
        "linalg:diag:0:offset:0:construct:0".to_owned(),
        "linalg:diag:3:offset:0:unknown:0".to_owned(),
        "linalg:diag:3:offset:0:construct:0:extra".to_owned(),
        format!("linalg:diagflat:{}x2:offset:0:construct:0",usize::MAX),
        format!("linalg:diagflat:2:offset:{}:construct:0",i64::MAX),
        format!("linalg:diagflat:2:offset:{}:construct:0",i64::MIN),
        "linalg:diagflat:2x3:offset:0:construct:6".to_owned(),
        "linalg:diagflat:0:offset:0:construct:0".to_owned(),
        "linalg:diagflat:2x3:offset:0:construct:0:extra".to_owned(),
    ] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir(&operation,1),&[7.0]).unwrap();
        assert!(!refused.supported,"{operation}");
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir("linalg:diag:3x4:offset:-1:extract:1",1),&[7.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(7.0));
        assert_eq!(retry.gradient,vec![1.0]);
    }
    for operation in ["linalg:diag:3:offset:0:construct:0","linalg:diagflat:3:offset:0:construct:0"] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir(operation,2),&[7.0,8.0]).unwrap();
        assert!(!refused.supported);
        assert!(refused.blocked_reasons.iter().any(|r|r.contains("exactly one source operand")));
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir(operation,1),&[7.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(7.0));
    }
}

#[test]
fn public_diagonal_rejects_dense_output_even_when_source_length_is_addressable() {
    let side=(isize::MAX as usize/std::mem::size_of::<f64>())/2;
    for family in ["diag","diagflat"] {
        let source=diagonal_ir(&format!("linalg:{family}:{side}:offset:0:construct:0"),1);
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&source,&[7.0]).unwrap();
        assert!(!refused.supported);
        assert!(refused.blocked_reasons.iter().any(|r|r.contains("output bytes exceed native addressability")),"{:?}",refused.blocked_reasons);
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir(&format!("linalg:{family}:3:offset:0:construct:0"),1),&[7.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(7.0));
        assert_eq!(retry.gradient,vec![1.0]);
    }
}

#[test]
fn public_compact_diagonal_workspace_budgets_are_inclusive_and_recover() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    for operation in [
        "linalg:diag:3:offset:0:construct:2",
        "linalg:diag:3:offset:2:construct:1",
        "linalg:diag:3:offset:-2:construct:1",
        "linalg:diag:3x4:offset:1:extract:2",
        "linalg:diag:3x4:offset:-1:extract:1",
        "linalg:diag:100000000x1:offset:0:extract:0",
        "linalg:diagflat:2x3:offset:0:construct:5",
        "linalg:diagflat:2x3:offset:2:construct:4",
        "linalg:diagflat:2x3:offset:-2:construct:4",
    ] {
        let source=diagonal_ir(operation,1);
        for gradient_surface in [false,true] {
            let expected=ReplayMemoryRequest { forward_bytes:16, adjoint_bytes:if gradient_surface { 24 } else { 0 }, intermediate_bytes:if gradient_surface { 16 } else { 8 } };
            let total=expected.total_bytes().unwrap();
            let replay=|| {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&source,&[7.0]).map(|r|(r.supported,r.value,r.gradient,r.blocked_reasons))
                } else { interpret_program_ad_effect_ir_forward(&source,&[7.0]).map(|r|(r.supported,r.value,Vec::new(),r.blocked_reasons)) }
            };
            for budget in [total-1,total,total+1] {
                let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
                let result=with_replay_memory_admission(
                    move |request| {
                        recorded.set(recorded.get()+1);assert_eq!(request,expected);
                        if request.total_bytes()? > budget { Err("compact diagonal workspace budget refused".to_owned()) } else { Ok(()) }
                    },
                    replay,
                );
                assert_eq!(calls.get(),1);
                match result {
                    Err(reason)=>{assert!(budget<total);assert!(reason.contains("compact diagonal workspace budget refused"));}
                    Ok(result)=>{
                        assert_eq!(result.0,budget>=total);
                        if result.0 {assert_eq!(result.1,Some(7.0));if gradient_surface {assert_eq!(result.2,[1.0]);}}
                        else {assert!(result.3.iter().any(|reason|reason.contains("compact diagonal workspace budget refused")));}
                    }
                }
                let retry=replay().unwrap();assert!(retry.0,"{:?}",retry.3);assert_eq!(retry.1,Some(7.0));
                if gradient_surface {assert_eq!(retry.2,[1.0]);}
            }
        }
    }
}

#[test]
fn public_diagonal_metadata_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let side=(isize::MAX as usize/8)/2;
    for (operation,count) in [
        (format!("linalg:diag:{side}:offset:0:construct:0"),1),
        (format!("linalg:diagflat:{side}:offset:0:construct:0"),1),
        (format!("linalg:diag:2:offset:{}:construct:0",i64::MIN),1),
        (format!("linalg:diagflat:2:offset:{}:construct:0",i64::MIN),1),
        ("linalg:diag:3x4:offset:4:extract:0".to_owned(),1),
        ("linalg:diagflat:2x3:offset:0:construct:6".to_owned(),1),
        ("linalg:diag:3:offset:0:construct:0".to_owned(),2),
        ("linalg:diagflat:3:offset:0:construct:0".to_owned(),2),
    ] {
        for gradient_surface in [false,true] {
            let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
            let source=diagonal_ir(&operation,count);let inputs=vec![7.0;count];
            let result=with_replay_memory_admission(
                move |_| {recorded.set(recorded.get()+1);Ok(())},
                || {
                    if gradient_surface {interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs).map(|r|r.supported)}
                    else {interpret_program_ad_effect_ir_forward(&source,&inputs).map(|r|r.supported)}
                },
            ).unwrap();
            assert!(!result,"{operation}");assert_eq!(calls.get(),0);
            let retry=interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir("linalg:diag:3x4:offset:-1:extract:1",1),&[7.0]).unwrap();
            assert!(retry.supported);assert_eq!(retry.value,Some(7.0));assert_eq!(retry.gradient,[1.0]);
        }
    }
}

#[test]
fn public_diagonal_admitted_identity_preserves_negative_zero() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest,with_replay_memory_admission};
    for operation in ["linalg:diag:3:offset:0:construct:0","linalg:diagflat:3:offset:0:construct:0"] {
        let result=with_replay_memory_admission(
            |request| {assert_eq!(request,ReplayMemoryRequest {forward_bytes:16,adjoint_bytes:24,intermediate_bytes:16});Ok(())},
            || interpret_program_ad_effect_ir_value_and_gradient(&diagonal_ir(operation,1),&[-0.0]),
        ).unwrap();
        assert!(result.supported);assert_eq!(result.value.unwrap().to_bits(),(-0.0_f64).to_bits());assert_eq!(result.gradient,[1.0]);
    }
}
