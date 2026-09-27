// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public pseudoinverse admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

fn pinv_ir(operation: &str, count: usize) -> String {
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
    // Indexed linalg outputs must be consumed by a scalar objective, not returned raw.
    let objective_index = count + 1;
    let objective = format!("%{objective_index}");
    values.push(serde_json::json!({"name":objective,"producer":objective_index,"version":0,"shape":[],"dtype":"float64","effect":objective_index}));
    effects.push(serde_json::json!({"index":objective_index,"kind":"pure","target":objective,"inputs":[target,"0.0"],"version":0,"ordering":objective_index,"operation":"add"}));
    serde_json::json!({"format":"program_ad_effect_ir.v1","ssa_values":values,"effects":effects,"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}).to_string()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual-expected).abs()<=1.0e-12,"expected {expected}, got {actual}");
}

#[test]
fn public_pinv_rank_one_tall_wide_and_square_values_gradients_and_owned_cancellation() {
    for (operation,inputs,expected,gradient) in [
        ("linalg:pinv:1x1:0:0:0",vec![2.0],0.5,vec![-0.25]),
        ("linalg:pinv:1x2:0:1:0",vec![3.0,4.0],4.0/25.0,vec![-24.0/625.0,-7.0/625.0]),
        ("linalg:pinv:3x1:0:0:1",vec![1.0,2.0,2.0],2.0/9.0,vec![-4.0/81.0,1.0/81.0,-8.0/81.0]),
        ("linalg:pinv:1x3:0:1:0",vec![1.0,2.0,2.0],2.0/9.0,vec![-4.0/81.0,1.0/81.0,-8.0/81.0]),
        ("linalg:pinv:2x2:0:0:0",vec![2.0,0.0,0.0,4.0],0.5,vec![-0.25,0.0,0.0,0.0]),
        ("linalg:pinv:2x2:0:0:1",vec![2.0,0.0,0.0,4.0],0.0,vec![0.0,-0.125,0.0,0.0]),
        ("linalg:pinv:3x2:0:0:2",vec![2.0,0.0,0.0,4.0,0.0,0.0],0.0,vec![0.0,0.0,0.0,0.0,0.25,0.0]),
        ("linalg:pinv:2x3:0:2:0",vec![2.0,0.0,0.0,0.0,4.0,0.0],0.0,vec![0.0,0.0,0.25,0.0,0.0,0.0]),
        ("linalg:pinv:2x3:0:1:1",vec![2.0,0.0,0.0,0.0,4.0,0.0],0.25,vec![0.0,0.0,0.0,0.0,-0.0625,0.0]),
    ] {
        let source=pinv_ir(operation,inputs.len());
        let calls=Rc::new(Cell::new(0usize));
        let recorded=Rc::clone(&calls);
        let baseline=with_replay_checkpoint(
            move || { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
        ).unwrap();
        assert!(baseline.supported,"{:?}",baseline.blocked_reasons);
        assert_close(baseline.value.unwrap(),expected);
        assert_eq!(baseline.gradient.len(),gradient.len());
        for (actual,expected) in baseline.gradient.iter().zip(&gradient) { assert_close(*actual,*expected); }
        assert!(calls.get()>1);
        for boundary in 1..=calls.get() {
            let observed=Rc::new(Cell::new(0usize));
            let recorded=Rc::clone(&observed);
            let refused=with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get()+1);
                    if recorded.get()>=boundary { Err("pseudoinverse owner cancelled".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
            );
            match refused {
                Err(reason)=>assert!(reason.contains("pseudoinverse owner cancelled"),"{reason}"),
                Ok(result)=>{
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|r|r.contains("pseudoinverse owner cancelled")),"{:?}",result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry=interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value,baseline.value);
            assert_eq!(retry.gradient,baseline.gradient);
        }
    }
}

#[test]
fn public_pinv_overflow_rank_and_malformed_cutoff_refusals_recover() {
    for operation in [
        format!("linalg:pinv:{}x2:0:0:0",usize::MAX),
        format!("linalg:pinv:2x{}:0:0:0",usize::MAX),
        "linalg:pinv:0x2:0:0:0".to_owned(),
        "linalg:pinv:2:0:0:0".to_owned(),
        "linalg:pinv:2x2x2:0:0:0".to_owned(),
        "linalg:pinv:2x2:-1:0:0".to_owned(),
        "linalg:pinv:2x2:NaN:0:0".to_owned(),
        "linalg:pinv:2x2:0:2:0".to_owned(),
        "linalg:pinv:2x2:0:0:2".to_owned(),
        "linalg:pinv:2x2:1:0:0".to_owned(),
        "linalg:pinv:2x2:0:0:0:extra".to_owned(),
    ] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir(&operation,4),&[2.0,0.0,0.0,4.0]).unwrap();
        assert!(!refused.supported,"{operation}");
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir("linalg:pinv:2x2:0:0:0",4),&[2.0,0.0,0.0,4.0]).unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(),0.5);
        assert_eq!(retry.gradient,vec![-0.25,0.0,0.0,0.0]);
    }
    for (operation,inputs) in [
        ("linalg:pinv:1x2:0:0:0",vec![0.0,0.0]),
        ("linalg:pinv:2x2:0:0:0",vec![1.0,2.0,2.0,4.0]),
        ("linalg:pinv:3x3:0:0:0",vec![1.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,3.0]),
    ] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir(operation,inputs.len()),&inputs).unwrap();
        assert!(!refused.supported);
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir("linalg:pinv:1x1:0:0:0",1),&[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(0.5));
        assert_eq!(retry.gradient,vec![-0.25]);
    }
}

#[test]
fn public_pinv_workspace_budget_refusal_and_retry_cover_rank_and_orientation() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    for (operation, inputs, workspace, forward_workspace, value, gradient) in [
        ("linalg:pinv:1x1:0:0:0", vec![2.0], 152usize, 24usize, 0.5, vec![-0.25]),
        ("linalg:pinv:1x2:0:1:0", vec![3.0, 4.0], 296, 48, 4.0/25.0, vec![-24.0/625.0, -7.0/625.0]),
        ("linalg:pinv:3x1:0:0:1", vec![1.0, 2.0, 2.0], 664, 72, 2.0/9.0, vec![-4.0/81.0, 1.0/81.0, -8.0/81.0]),
        ("linalg:pinv:3x2:0:0:2", vec![2.0, 0.0, 0.0, 4.0, 0.0, 0.0], 1000, 192, 0.0, vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.0]),
        ("linalg:pinv:2x3:0:2:0", vec![2.0, 0.0, 0.0, 0.0, 4.0, 0.0], 880, 192, 0.0, vec![0.0, 0.0, 0.25, 0.0, 0.0, 0.0]),
    ] {
        let source = pinv_ir(operation, inputs.len());
        let forward_bytes = (inputs.len() + 2) * 8;
        let adjoint_bytes = (2 * inputs.len() + 2) * 8;
        let expected = ReplayMemoryRequest { forward_bytes, adjoint_bytes, intermediate_bytes:workspace };
        let total = forward_bytes + adjoint_bytes + workspace;
        for budget in [total - 1, total, total + 1] {
            let result = with_replay_memory_admission(
                move |request| {
                    assert_eq!(request, expected);
                    if request.total_bytes()? > budget { Err("pinv workspace budget refused".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            ).unwrap();
            assert_eq!(result.supported, budget >= total);
            if result.supported {
                assert_close(result.value.unwrap(), value);
                for (actual, expected) in result.gradient.iter().zip(&gradient) { assert_close(*actual, *expected); }
                assert_eq!(result.gradient.len(), gradient.len());
            } else { assert!(result.blocked_reasons.iter().any(|reason| reason.contains("pinv workspace budget refused"))); }
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
            assert!(retry.supported);
            assert_close(retry.value.unwrap(), value);
            assert_eq!(retry.gradient.len(), gradient.len());
            for (actual, expected) in retry.gradient.iter().zip(&gradient) { assert_close(*actual, *expected); }
        }
        let forward_total = forward_bytes + forward_workspace;
        for budget in [forward_total - 1, forward_total, forward_total + 1] {
            let result = with_replay_memory_admission(
                move |request| {
                    assert_eq!(request, ReplayMemoryRequest { forward_bytes, adjoint_bytes:0, intermediate_bytes:forward_workspace });
                    if request.total_bytes()? > budget { Err("pinv forward budget refused".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_forward(&source, &inputs),
            );
            if budget < forward_total { assert!(result.unwrap_err().contains("pinv forward budget refused")); }
            else { let result = result.unwrap(); assert!(result.supported); assert_close(result.value.unwrap(), value); }
        }
    }
}

#[test]
fn public_wide_pinv_accounts_for_projector_construction_peak() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    let source = pinv_ir("linalg:pinv:1x32:0:0:0", 32);
    let mut inputs = vec![0.0; 32];
    inputs[0] = 2.0;
    // During left-projector subtraction, three 32x32 arrays coexist with four sources.
    // This exceeds the later retained VJP inventory for this orientation.
    let expected = ReplayMemoryRequest { forward_bytes:272, adjoint_bytes:528, intermediate_bytes:25600 };
    for budget in [26399usize, 26400, 26401] {
        let result = with_replay_memory_admission(
            move |request| {
                assert_eq!(request, expected);
                if request.total_bytes()? > budget { Err("wide projector budget refused".to_owned()) } else { Ok(()) }
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        ).unwrap();
        assert_eq!(result.supported, budget >= 26400);
        if result.supported {
            assert_close(result.value.unwrap(), 0.5);
            assert_eq!(result.gradient.len(), 32);
            assert_close(result.gradient[0], -0.25);
            assert!(result.gradient[1..].iter().all(|value| value.abs() <= 1.0e-12));
        } else { assert!(result.blocked_reasons.iter().any(|reason| reason.contains("wide projector budget refused"))); }
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 0.5);
        assert_close(retry.gradient[0], -0.25);
        assert!(retry.gradient[1..].iter().all(|value| value.abs() <= 1.0e-12));
    }
}

#[test]
fn public_absurd_pinv_projector_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let side = isize::MAX as usize / std::mem::size_of::<f64>();
    for operation in [format!("linalg:pinv:1x{side}:0:0:0"), format!("linalg:pinv:{side}x1:0:0:0")] {
        let callbacks = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&callbacks);
        let result = with_replay_memory_admission(
            move |_| { recorded.set(recorded.get() + 1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir(&operation, 1), &[2.0]),
        ).unwrap();
        assert!(!result.supported);
        assert_eq!(callbacks.get(), 0);
        assert!(result.blocked_reasons.iter().any(|reason| reason.contains("pinv matrix bytes exceed native addressability")), "{:?}", result.blocked_reasons);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&pinv_ir("linalg:pinv:1x1:0:0:0", 1), &[2.0]).unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 0.5);
        assert_close(retry.gradient[0], -0.25);
    }
}

#[test]
fn public_indexed_pinv_output_still_requires_scalar_objective() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    let scalar_source = pinv_ir("linalg:pinv:1x1:0:0:0", 1);
    let mut raw: serde_json::Value = serde_json::from_str(&scalar_source).unwrap();
    raw["ssa_values"].as_array_mut().unwrap().pop();
    raw["effects"].as_array_mut().unwrap().pop();
    let forward = interpret_program_ad_effect_ir_forward(&raw.to_string(), &[2.0]).unwrap();
    assert!(!forward.supported);
    assert!(forward.blocked_reasons.iter().any(|reason| reason.contains("indexed multi-output linalg result")));
    let gradient = interpret_program_ad_effect_ir_value_and_gradient(&raw.to_string(), &[2.0]).unwrap();
    assert!(!gradient.supported);
    assert!(gradient.blocked_reasons.iter().any(|reason| reason.contains("indexed multi-output linalg result")));
    let retry = interpret_program_ad_effect_ir_value_and_gradient(&scalar_source, &[2.0]).unwrap();
    assert!(retry.supported);
    assert_close(retry.value.unwrap(), 0.5);
    assert_close(retry.gradient[0], -0.25);
}
