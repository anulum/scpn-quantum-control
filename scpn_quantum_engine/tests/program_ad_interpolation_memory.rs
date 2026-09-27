// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public interpolation admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

fn interpolation_ir(operation: &str) -> String {
    let mut values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..4 {
        let name = format!("%{index}");
        values.push(serde_json::json!({"name":name,"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
        effects.push(serde_json::json!({"index":index,"kind":"parameter","target":name,"inputs":[format!("x{index}")],"version":0,"ordering":index,"operation":"parameter"}));
    }
    values.push(serde_json::json!({"name":"%4","producer":4,"version":0,"shape":[],"dtype":"float64","effect":4}));
    effects.push(serde_json::json!({"index":4,"kind":"primitive","target":"%4","inputs":["%0","%1","%2","%3"],"version":0,"ordering":4,"operation":operation}));
    serde_json::json!({"format":"program_ad_effect_ir.v1","ssa_values":values,"effects":effects,"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}).to_string()
}

#[test]
fn public_interpolation_checks_each_observed_boundary_and_recovers() {
    for (sample, left, right, expected, gradient) in [
        (1.0,"none","none",4.0,vec![2.0,0.5,0.5,0.0]),
        (3.0,"none","none",8.0,vec![2.0,0.0,0.5,0.5]),
        (-1.0,"none","none",2.0,vec![0.0,1.0,0.0,0.0]),
        (5.0,"none","none",10.0,vec![0.0,0.0,0.0,1.0]),
        (-1.0,"-3","12",-3.0,vec![0.0;4]),
        (5.0,"-3","12",12.0,vec![0.0;4]),
    ] {
        let source = interpolation_ir(&format!("interpolation:interp:samples:1:grid:0,2,4:left:{left}:right:{right}:out:0"));
        let inputs = [sample,2.0,6.0,10.0];
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
        ).unwrap();
        assert!(baseline.supported,"{:?}",baseline.blocked_reasons);
        assert_eq!(baseline.value,Some(expected));
        assert_eq!(baseline.gradient,gradient);
        assert!(calls.get()>1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get()+1);
                    if recorded.get()>=boundary { Err("interpolation owner cancelled".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
            );
            match refused {
                Err(reason)=>assert!(reason.contains("interpolation owner cancelled"),"{reason}"),
                Ok(result)=>{
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|r|r.contains("interpolation owner cancelled")),"{:?}",result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value,baseline.value);
            assert_eq!(retry.gradient,baseline.gradient);
        }
    }
}

#[test]
fn public_interpolation_refuses_invalid_sizes_knots_and_metadata_before_retry() {
    let valid = "interpolation:interp:samples:1:grid:0,2,4:left:none:right:none:out:0";
    let inputs = [1.0,2.0,6.0,10.0];
    for operation in [
        valid.replace("samples:1",&format!("samples:{}",usize::MAX)),
        valid.replace("samples:1","samples:0"),
        valid.replace("grid:0,2,4","grid:0,0,4"),
        valid.replace("grid:0,2,4","grid:0,NaN,4"),
        valid.replace("grid:0,2,4","grid:0"),
        valid.replace("out:0","out:1"),
        format!("{valid}:extra"),
        valid.replace(":out:0",""),
    ] {
        let refused = interpret_program_ad_effect_ir_value_and_gradient(&interpolation_ir(&operation),&inputs).unwrap();
        assert!(!refused.supported,"{operation}");
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&interpolation_ir(valid),&inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(4.0));
        assert_eq!(retry.gradient,vec![2.0,0.5,0.5,0.0]);
    }
    for knot in [0.0,2.0,4.0] {
        let refused = interpret_program_ad_effect_ir_value_and_gradient(&interpolation_ir(valid),&[knot,2.0,6.0,10.0]).unwrap();
        assert!(!refused.supported);
        assert!(refused.blocked_reasons.iter().any(|r|r.contains("avoid grid knots")));
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&interpolation_ir(valid),&inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(4.0));
    }
}

#[test]
fn public_interpolation_refuses_nonfinite_derivative_then_releases_owner() {
    let source = interpolation_ir("interpolation:interp:samples:1:grid:0,2,4:left:none:right:none:out:0");
    // The interpolated value is finite, but the fp difference overflows.
    let refused = interpret_program_ad_effect_ir_value_and_gradient(&source,&[1.0,-1.0e308,1.0e308,0.0]).unwrap();
    assert!(!refused.supported);
    assert!(refused.blocked_reasons.iter().any(|r|r.contains("cotangent entries must be finite")),"{:?}",refused.blocked_reasons);
    replay_checkpoint().unwrap();
    let retry = interpret_program_ad_effect_ir_value_and_gradient(&source,&[1.0,2.0,6.0,10.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value,Some(4.0));
    assert_eq!(retry.gradient,vec![2.0,0.5,0.5,0.0]);
}

#[test]
fn public_interpolation_workspace_budgets_cover_grid_and_boundary_storage() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    for (sample,left,right,value,gradient) in [
        (1.0,"none","none",4.0,[2.0,0.5,0.5,0.0]),
        (3.0,"none","none",8.0,[2.0,0.0,0.5,0.5]),
        (-1.0,"none","none",2.0,[0.0,1.0,0.0,0.0]),
        (5.0,"none","none",10.0,[0.0,0.0,0.0,1.0]),
        (-1.0,"-3","12",-3.0,[0.0;4]),
        (5.0,"-3","12",12.0,[0.0;4]),
    ] {
        let ir=interpolation_ir(&format!("interpolation:interp:samples:1:grid:0,2,4:left:{left}:right:{right}:out:0"));
        let inputs=[sample,2.0,6.0,10.0];
        for gradient_surface in [false,true] {
            let expected=ReplayMemoryRequest { forward_bytes:40, adjoint_bytes:if gradient_surface { 72 } else { 0 }, intermediate_bytes:if gradient_surface { 88 } else { 56 } };
            let total=expected.total_bytes().unwrap();
            let replay=|| {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir,&inputs).map(|r|(r.supported,r.value,r.gradient,r.blocked_reasons))
                } else { interpret_program_ad_effect_ir_forward(&ir,&inputs).map(|r|(r.supported,r.value,Vec::new(),r.blocked_reasons)) }
            };
            for budget in [total-1,total,total+1] {
                let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
                let result=with_replay_memory_admission(
                    move |request| {
                        recorded.set(recorded.get()+1);assert_eq!(request,expected);
                        if request.total_bytes()? > budget { Err("interpolation workspace budget refused".to_owned()) } else { Ok(()) }
                    },
                    replay,
                );
                assert_eq!(calls.get(),1);
                match result {
                    Err(reason)=>{ assert!(budget<total);assert!(reason.contains("interpolation workspace budget refused")); }
                    Ok(result)=>{
                        assert_eq!(result.0,budget>=total);
                        if result.0 { assert_eq!(result.1,Some(value));if gradient_surface { assert_eq!(result.2,gradient); } }
                        else { assert!(result.3.iter().any(|reason|reason.contains("interpolation workspace budget refused"))); }
                    }
                }
                let retry=replay().unwrap();assert!(retry.0,"{:?}",retry.3);assert_eq!(retry.1,Some(value));
                if gradient_surface { assert_eq!(retry.2,gradient); }
            }
        }
    }
}

#[test]
fn public_interpolation_bad_metadata_refuses_before_numeric_admission() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for operation in [
        format!("interpolation:interp:samples:{}:grid:0,2,4:left:none:right:none:out:0",usize::MAX),
        "interpolation:interp:samples:1:grid:0,4,2:left:none:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,0,4:left:none:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,NaN,4:left:none:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,Inf,4:left:none:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,bad,4:left:none:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,2,4:left:NaN:right:none:out:0".to_owned(),
        "interpolation:interp:samples:1:grid:0,2,4:left:none:right:none:out:1".to_owned(),
        "interpolation:interp:samples:1:grid:0,2:left:none:right:none:out:0".to_owned(),
    ] {
        for gradient_surface in [false,true] {
            let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);let ir=interpolation_ir(&operation);
            let result=with_replay_memory_admission(
                move |_| { recorded.set(recorded.get()+1);Ok(()) },
                || {
                    if gradient_surface { interpret_program_ad_effect_ir_value_and_gradient(&ir,&[1.0,2.0,6.0,10.0]).map(|r|r.supported) }
                    else { interpret_program_ad_effect_ir_forward(&ir,&[1.0,2.0,6.0,10.0]).map(|r|r.supported) }
                },
            ).unwrap();
            assert!(!result,"{operation}");assert_eq!(calls.get(),0);
            let retry=interpret_program_ad_effect_ir_value_and_gradient(
                &interpolation_ir("interpolation:interp:samples:1:grid:0,2,4:left:none:right:none:out:0"),&[1.0,2.0,6.0,10.0],
            ).unwrap();
            assert!(retry.supported);assert_eq!(retry.value,Some(4.0));assert_eq!(retry.gradient,[2.0,0.5,0.5,0.0]);
        }
    }
}
