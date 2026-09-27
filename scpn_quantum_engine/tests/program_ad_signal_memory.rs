// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public signal admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

fn signal_ir(operation: &str, count: usize) -> String {
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
fn public_signal_modes_stream_values_gradients_and_owned_cancellation() {
    // Independent polynomial coefficient and reversed-kernel fixtures.
    for (kind, expected, gradients) in [
        ("convolve",[4.0,13.0,22.0,15.0],[
            [4.0,0.0,0.0,1.0,0.0], [5.0,4.0,0.0,2.0,1.0],
            [0.0,5.0,4.0,3.0,2.0], [0.0,0.0,5.0,0.0,3.0],
        ]),
        ("correlate",[5.0,14.0,23.0,12.0],[
            [5.0,0.0,0.0,0.0,1.0], [4.0,5.0,0.0,1.0,2.0],
            [0.0,4.0,5.0,2.0,3.0], [0.0,0.0,4.0,3.0,0.0],
        ]),
    ] {
        for (mode,start,size) in [("full",0,4),("same",0,3),("valid",1,2)] {
            for output in 0..size {
                let source = signal_ir(&format!("signal:{kind}:left:3:right:2:mode:{mode}:out:{output}"),5);
                let inputs = [1.0,2.0,3.0,4.0,5.0];
                let calls = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&calls);
                let baseline = with_replay_checkpoint(
                    move || { recorded.set(recorded.get()+1); Ok(()) },
                    || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
                ).unwrap();
                assert!(baseline.supported,"{:?}",baseline.blocked_reasons);
                assert_eq!(baseline.value,Some(expected[start+output]));
                assert_eq!(baseline.gradient,gradients[start+output]);
                assert!(calls.get()>1);
                for boundary in 1..=calls.get() {
                    let observed = Rc::new(Cell::new(0usize));
                    let recorded = Rc::clone(&observed);
                    let refused = with_replay_checkpoint(
                        move || {
                            recorded.set(recorded.get()+1);
                            if recorded.get()>=boundary { Err("signal owner cancelled".to_owned()) } else { Ok(()) }
                        },
                        || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
                    );
                    match refused {
                        Err(reason)=>assert!(reason.contains("signal owner cancelled"),"{reason}"),
                        Ok(result)=>{
                            assert!(!result.supported);
                            assert!(result.blocked_reasons.iter().any(|r|r.contains("signal owner cancelled")),"{:?}",result.blocked_reasons);
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
    }
}

#[test]
fn public_signal_right_longer_windows_and_singleton_signed_zero_are_preserved() {
    for (kind,mode,output,expected,gradient) in [
        ("convolve","same",2,22.0,[3.0,2.0,0.0,5.0,4.0]),
        ("correlate","valid",0,23.0,[2.0,3.0,0.0,4.0,5.0]),
    ] {
        let source=signal_ir(&format!("signal:{kind}:left:2:right:3:mode:{mode}:out:{output}"),5);
        let result=interpret_program_ad_effect_ir_value_and_gradient(&source,&[4.0,5.0,1.0,2.0,3.0]).unwrap();
        assert!(result.supported,"{:?}",result.blocked_reasons);
        assert_eq!(result.value,Some(expected));
        assert_eq!(result.gradient,gradient);
    }
    for kind in ["convolve","correlate"] {
        for mode in ["full","same","valid"] {
            let source=signal_ir(&format!("signal:{kind}:left:1:right:1:mode:{mode}:out:0"),2);
            let result=interpret_program_ad_effect_ir_value_and_gradient(&source,&[-0.0,2.0]).unwrap();
            assert!(result.supported);
            assert_eq!(result.value.unwrap().to_bits(),(-0.0_f64).to_bits());
            assert_eq!(result.gradient,vec![2.0,0.0]);
        }
    }
}

#[test]
fn public_signal_refuses_overflow_malformed_windows_and_nonfinite_inputs_then_recovers() {
    let valid="signal:convolve:left:3:right:2:mode:full:out:1";
    let inputs=[1.0,2.0,3.0,4.0,5.0];
    for operation in [
        valid.replace("left:3",&format!("left:{}",usize::MAX)),
        valid.replace("right:2",&format!("right:{}",usize::MAX)),
        valid.replace("left:3","left:0"),
        valid.replace("right:2","right:0"),
        valid.replace("mode:full","mode:unknown"),
        valid.replace("out:1","out:4"),
        valid.replace("out:1","out:-1"),
        format!("{valid}:extra"),
        valid.replace(":out:1",""),
    ] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&signal_ir(&operation,5),&inputs).unwrap();
        assert!(!refused.supported,"{operation}");
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&signal_ir(valid,5),&inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(13.0));
        assert_eq!(retry.gradient,vec![5.0,4.0,0.0,2.0,1.0]);
    }
    for bad in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&signal_ir(valid,5),&[bad,2.0,3.0,4.0,5.0]).unwrap();
        assert!(!refused.supported);
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&signal_ir(valid,5),&inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(13.0));
    }
}

#[test]
fn public_signal_refuses_overflowing_reverse_scatter_with_finite_forward_value() {
    let mut ir: serde_json::Value = serde_json::from_str(&signal_ir("signal:convolve:left:3:right:2:mode:full:out:1",5)).unwrap();
    let values=ir["ssa_values"].as_array_mut().unwrap();
    for index in [6,7] {
        values.push(serde_json::json!({"name":format!("%{index}"),"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
    }
    let effects=ir["effects"].as_array_mut().unwrap();
    effects.push(serde_json::json!({"index":6,"kind":"parameter","target":"%6","inputs":["weight"],"version":0,"ordering":6,"operation":"parameter"}));
    effects.push(serde_json::json!({"index":7,"kind":"pure","target":"%7","inputs":["%5","%6"],"version":0,"ordering":7,"operation":"mul"}));
    let source=ir.to_string();
    let refused=interpret_program_ad_effect_ir_value_and_gradient(&source,&[0.0,0.0,0.0,1.0e308,1.0e308,2.0]).unwrap();
    assert!(!refused.supported);
    assert!(refused.blocked_reasons.iter().any(|r|r.contains("cotangent entries must be finite")),"{:?}",refused.blocked_reasons);
    replay_checkpoint().unwrap();
    let retry=interpret_program_ad_effect_ir_value_and_gradient(&source,&[1.0,2.0,3.0,4.0,5.0,2.0]).unwrap();
    assert!(retry.supported,"{:?}",retry.blocked_reasons);
    assert_eq!(retry.value,Some(26.0));
    assert_eq!(retry.gradient,vec![10.0,8.0,0.0,4.0,2.0,13.0]);
}

#[test]
fn public_signal_workspace_budget_boundaries_preserve_modes_and_recover() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    for (operation, inputs, expected_value, expected_gradient) in [
        ("signal:convolve:left:3:right:2:mode:full:out:1", [1.0,2.0,3.0,4.0,5.0], 13.0, [5.0,4.0,0.0,2.0,1.0]),
        ("signal:correlate:left:3:right:2:mode:full:out:1", [1.0,2.0,3.0,4.0,5.0], 14.0, [4.0,5.0,0.0,1.0,2.0]),
        ("signal:convolve:left:3:right:2:mode:same:out:2", [1.0,2.0,3.0,4.0,5.0], 22.0, [0.0,5.0,4.0,3.0,2.0]),
        ("signal:correlate:left:3:right:2:mode:valid:out:1", [1.0,2.0,3.0,4.0,5.0], 23.0, [0.0,4.0,5.0,2.0,3.0]),
        ("signal:convolve:left:2:right:3:mode:same:out:2", [4.0,5.0,1.0,2.0,3.0], 22.0, [3.0,2.0,0.0,5.0,4.0]),
    ] {
        let source=signal_ir(operation,5);
        for gradient_surface in [false,true] {
            let expected=ReplayMemoryRequest { forward_bytes:48, adjoint_bytes:if gradient_surface { 88 } else { 0 }, intermediate_bytes:if gradient_surface { 80 } else { 40 } };
            let total=expected.total_bytes().unwrap();
            let replay=|| {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs).map(|r|(r.supported,r.value,r.gradient,r.blocked_reasons))
                } else { interpret_program_ad_effect_ir_forward(&source,&inputs).map(|r|(r.supported,r.value,Vec::new(),r.blocked_reasons)) }
            };
            for budget in [total-1,total,total+1] {
                let calls=Rc::new(Cell::new(0usize));
                let recorded=Rc::clone(&calls);
                let result=with_replay_memory_admission(
                    move |request| {
                        recorded.set(recorded.get()+1);
                        assert_eq!(request,expected);
                        if request.total_bytes()? > budget { Err("signal workspace budget refused".to_owned()) } else { Ok(()) }
                    },
                    replay,
                );
                assert_eq!(calls.get(),1);
                match result {
                    Err(reason)=>{ assert!(budget<total); assert!(reason.contains("signal workspace budget refused")); }
                    Ok(result)=>{
                        assert_eq!(result.0,budget>=total);
                        if result.0 {
                            assert_eq!(result.1,Some(expected_value));
                            if gradient_surface { assert_eq!(result.2,expected_gradient); }
                        } else { assert!(result.3.iter().any(|reason|reason.contains("signal workspace budget refused"))); }
                    }
                }
                let retry=replay().unwrap();
                assert!(retry.0,"{:?}",retry.3);
                assert_eq!(retry.1,Some(expected_value));
                if gradient_surface { assert_eq!(retry.2,expected_gradient); }
            }
        }
    }
}

#[test]
fn public_signal_metadata_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (operation, count) in [
        (format!("signal:convolve:left:{}:right:2:mode:full:out:0",usize::MAX),5),
        ("signal:convolve:left:0:right:2:mode:full:out:0".to_owned(),5),
        ("signal:correlate:left:3:right:2:mode:unknown:out:0".to_owned(),5),
        ("signal:convolve:left:3:right:2:mode:valid:out:2".to_owned(),5),
        ("signal:convolve:left:3:right:2:mode:full:out:0:extra".to_owned(),5),
        ("signal:correlate:left:3:right:2:mode:full:out:0".to_owned(),1),
    ] {
        for gradient_surface in [false,true] {
            let callbacks=Rc::new(Cell::new(0usize));
            let recorded=Rc::clone(&callbacks);
            let source=signal_ir(&operation,count);
            let inputs=vec![2.0;count];
            let result=with_replay_memory_admission(
                move |_| { recorded.set(recorded.get()+1); Ok(()) },
                || {
                    if gradient_surface {
                        interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs).map(|r|r.supported)
                    } else { interpret_program_ad_effect_ir_forward(&source,&inputs).map(|r|r.supported) }
                },
            ).unwrap();
            assert!(!result,"{operation}");
            assert_eq!(callbacks.get(),0);
            let retry=interpret_program_ad_effect_ir_value_and_gradient(
                &signal_ir("signal:convolve:left:3:right:2:mode:full:out:1",5),&[1.0,2.0,3.0,4.0,5.0],
            ).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value,Some(13.0));
            assert_eq!(retry.gradient,[5.0,4.0,0.0,2.0,1.0]);
        }
    }
}
