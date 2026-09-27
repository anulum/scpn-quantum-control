// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Public product/moment memory and recovery tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
use std::cell::Cell;
use std::rc::Rc;

fn reduction_ir(operation: &str, source_shape: &[usize], target_shape: &[usize]) -> String {
    serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":source_shape,"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":target_shape,"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":operation},
            {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual-expected).abs() <= 1e-12,"expected {expected}, got {actual}");
}

fn assert_numeric(result: &scpn_quantum_program_ad_replay::program_ad_ir::ProgramADRustValueAndGradientResult, value: f64, gradient: &[f64]) {
    assert!(result.supported,"{:?}",result.blocked_reasons);
    assert_close(result.value.unwrap(),value); assert_eq!(result.gradient.len(),gradient.len());
    for (actual,expected) in result.gradient.iter().zip(gradient) { assert_close(*actual,*expected); }
}

#[test]
fn public_product_and_moment_budgets_preserve_independent_oracles_and_recover() {
    let standard_deviation = (35.0_f64/12.0).sqrt();
    for (operation, target, axis_size, inputs, value, gradient) in [
        ("prod", vec![], None, [1.0,3.0,5.0,2.0,4.0,6.0], 720.0, [720.0,240.0,144.0,360.0,180.0,120.0]),
        ("prod:axis:1", vec![2], Some(3), [1.0,3.0,5.0,2.0,4.0,6.0], 63.0, [15.0,5.0,3.0,24.0,12.0,8.0]),
        ("prod:axis:-2", vec![3], Some(2), [1.0,3.0,5.0,2.0,4.0,6.0], 44.0, [2.0,4.0,6.0,1.0,3.0,5.0]),
        ("prod:axis:1", vec![2], Some(3), [0.0,3.0,5.0,2.0,4.0,6.0], 48.0, [15.0,0.0,0.0,24.0,12.0,8.0]),
        ("var:axis:1:ddof:1", vec![2], Some(3), [1.0,3.0,5.0,2.0,4.0,6.0], 8.0, [-2.0,0.0,2.0,-2.0,0.0,2.0]),
        ("std:correction:1:axis:-1", vec![2], Some(3), [1.0,3.0,5.0,2.0,4.0,6.0], 4.0, [-0.5,0.0,0.5,-0.5,0.0,0.5]),
        ("var:axis:0", vec![3], Some(2), [1.0,3.0,5.0,2.0,4.0,6.0], 0.75, [-0.5,-0.5,-0.5,0.5,0.5,0.5]),
        ("std:axis:0", vec![3], Some(2), [1.0,3.0,5.0,2.0,4.0,6.0], 1.5, [-0.5,-0.5,-0.5,0.5,0.5,0.5]),
        ("var", vec![], None, [1.0,3.0,5.0,2.0,4.0,6.0], 35.0/12.0, [-5.0/6.0,-1.0/6.0,0.5,-0.5,1.0/6.0,5.0/6.0]),
        ("var:correction:0.5", vec![], None, [1.0,3.0,5.0,2.0,4.0,6.0], 35.0/11.0, [-10.0/11.0,-2.0/11.0,6.0/11.0,-6.0/11.0,2.0/11.0,10.0/11.0]),
        ("std", vec![], None, [1.0,3.0,5.0,2.0,4.0,6.0], standard_deviation, [-2.5/(6.0*standard_deviation),-0.5/(6.0*standard_deviation),1.5/(6.0*standard_deviation),-1.5/(6.0*standard_deviation),0.5/(6.0*standard_deviation),2.5/(6.0*standard_deviation)]),
    ] {
        let ir = reduction_ir(operation,&[2,3],&target);
        let output: usize = target.iter().product(); let q = target.len();
        // Independent role inventory: source/contribution/reduced copies plus
        // target cotangent, five conservative source rank buffers and its shape.
        let word = std::mem::size_of::<usize>();
        let accumulation = (18+output)*8+(10+q)*word;
        let workspace = if let Some(axis_size) = axis_size {
            let pair_groups = 6*std::mem::size_of::<(usize,f64)>()
                + output*std::mem::size_of::<Vec<(usize,f64)>>();
            let group_peak = (12+output+2*axis_size)*8+(4+3*q)*word+pair_groups;
            accumulation.max(group_peak)
        } else { accumulation };
        let retained = 6+output+1;
        let expected = ReplayMemoryRequest { forward_bytes:retained*8,
            adjoint_bytes:(retained+6)*8, intermediate_bytes:workspace };
        let total = expected.total_bytes().unwrap();
        for budget in [total-1,total,total+1] {
            let calls=Rc::new(Cell::new(0usize)); let recorded=Rc::clone(&calls);
            let result=with_replay_memory_admission(
                move |request| {
                    recorded.set(recorded.get()+1); assert_eq!(request,expected);
                    if request.total_bytes()? > budget {Err("reduction workspace budget refused".to_owned())} else {Ok(())}
                }, || interpret_program_ad_effect_ir_value_and_gradient(&ir,&inputs),
            );
            assert_eq!(calls.get(),1);
            match result {
                Err(reason)=>{assert!(budget<total);assert!(reason.contains("reduction workspace budget refused"));}
                Ok(result)=>{
                    assert_eq!(result.supported,budget>=total);
                    if result.supported {assert_numeric(&result,value,&gradient);}
                    else {assert!(result.blocked_reasons.iter().any(|r|r.contains("reduction workspace budget refused")));}
                }
            }
            assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&ir,&inputs).unwrap(),value,&gradient);
        }
    }
}

#[test]
fn public_reduction_bad_metadata_target_and_denominator_refuse_before_admission() {
    for (operation,target) in [
        ("prod",vec![2]),("prod:axis:2",vec![2]),("prod:axis:-3",vec![2]),
        ("prod:axis:1:extra",vec![2]),("prod:axis",vec![2]),("prod:axis:1",vec![3]),
        ("var:ddof:6",vec![]),("std:axis:1:correction:3",vec![2]),
        ("var:correction:-1",vec![]),("std:ddof:NaN",vec![]),
        ("var:ddof:1:correction:0",vec![]),("std:axis:1:axis:0",vec![2]),
        ("var:axis:2",vec![2]),("std:axis:1",vec![2,1]),("var",vec![2]),
        ("std:unknown:1",vec![]),("var:axis",vec![]),
    ] {
        let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
        let result=with_replay_memory_admission(
            move |_| {recorded.set(recorded.get()+1);Ok(())},
            || interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir(operation,&[2,3],&target),&[1.0,3.0,5.0,2.0,4.0,6.0]),
        ).unwrap();
        assert!(!result.supported,"{operation}");assert_eq!(calls.get(),0);
        assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir("prod:axis:1",&[2,3],&[2]),&[1.0,3.0,5.0,2.0,4.0,6.0]).unwrap(),63.0,&[15.0,5.0,3.0,24.0,12.0,8.0]);
    }
}

#[test]
fn public_reduction_broadcast_workspace_overflow_refuses_before_materialization() {
    for operation in ["prod:axis:1","var:axis:1","std:axis:1"] {
        // Four numeric entries and one source-index/value pair per source entry
        // are retained during the single wide reduction group. Exercise both
        // ceiling-sized and larger overflow-scale requests portably. Earlier
        // addressability guards may refuse before final aggregate arithmetic.
        let per_entry = 4*std::mem::size_of::<f64>()+std::mem::size_of::<(usize,f64)>();
        for huge in [isize::MAX as usize/per_entry+1,isize::MAX as usize/(per_entry/2)] {
        let ir = serde_json::json!({
            "format":"program_ad_effect_ir.v1",
            "ssa_values":[
                {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
                {"name":"%1","producer":1,"version":0,"shape":[1,huge],"dtype":"float64","effect":1},
                {"name":"%2","producer":2,"version":0,"shape":[1],"dtype":"float64","effect":2},
                {"name":"%3","producer":3,"version":0,"shape":[],"dtype":"float64","effect":3}
            ],
            "effects":[
                {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":"broadcast_to"},
                {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":operation},
                {"index":3,"kind":"primitive","target":"%3","inputs":["%2"],"version":0,"ordering":3,"operation":"sum"}
            ],"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
        }).to_string();
        let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
        let result=with_replay_memory_admission(
            move |_| {recorded.set(recorded.get()+1);Ok(())},
            || interpret_program_ad_effect_ir_value_and_gradient(&ir,&[1.0]),
        ).unwrap();
        assert!(!result.supported);assert_eq!(calls.get(),0);
        assert!(result.blocked_reasons.iter().any(|r|r.contains("reduction workspace exceeds native addressable memory")),"{:?}",result.blocked_reasons);
        assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir("prod",&[2],&[]),&[2.0,3.0]).unwrap(),6.0,&[3.0,2.0]);
        }
    }
}

#[test]
fn public_reduction_numeric_singularities_still_refuse_after_admission() {
    for (operation,inputs,reason) in [
        ("prod:axis:1",[0.0,0.0,5.0,2.0,4.0,6.0],"at most one zero"),
        ("std:axis:1",[1.0,1.0,1.0,2.0,4.0,6.0],"positive variance"),
    ] {
        let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
        let result=with_replay_memory_admission(
            move |_| {recorded.set(recorded.get()+1);Ok(())},
            || interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir(operation,&[2,3],&[2]),&inputs),
        ).unwrap();
        assert!(!result.supported);assert_eq!(calls.get(),1);
        assert!(result.blocked_reasons.iter().any(|r|r.contains(reason)),"{:?}",result.blocked_reasons);
        assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir("prod",&[2],&[]),&[2.0,3.0]).unwrap(),6.0,&[3.0,2.0]);
    }
}


#[test]
fn public_reduction_scalar_source_budget_and_scalar_forward_boundary() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    for (operation,value,gradient) in [("prod",2.0,1.0),("var",0.0,0.0),("var:correction:0.5",0.0,0.0)] {
        let ir=reduction_ir(operation,&[],&[]);
        let expected=ReplayMemoryRequest {forward_bytes:24,adjoint_bytes:32,intermediate_bytes:32};
        for budget in [87,88,89] {
            let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
            let result=with_replay_memory_admission(
                move |request| {
                    recorded.set(recorded.get()+1);assert_eq!(request,expected);
                    if request.total_bytes()? > budget {Err("scalar reduction budget refused".to_owned())} else {Ok(())}
                }, || interpret_program_ad_effect_ir_value_and_gradient(&ir,&[2.0]),
            );
            assert_eq!(calls.get(),1);
            match result {
                Err(reason)=>{assert!(budget<88);assert!(reason.contains("scalar reduction budget refused"));}
                Ok(result)=>{
                    assert_eq!(result.supported,budget>=88);
                    if result.supported {assert_numeric(&result,value,&[gradient]);}
                    else {assert!(result.blocked_reasons.iter().any(|r|r.contains("scalar reduction budget refused")));}
                }
            }
            assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&ir,&[2.0]).unwrap(),value,&[gradient]);
        }
        let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
        let unsupported=with_replay_memory_admission(
            move |_| {recorded.set(recorded.get()+1);Ok(())},
            || interpret_program_ad_effect_ir_forward(&ir,&[2.0]),
        ).unwrap();
        assert!(!unsupported.supported);assert_eq!(calls.get(),0);
        assert!(unsupported.blocked_reasons.iter().any(|r|r.contains("product/moment reductions require numeric replay")));
    }
}

#[test]
fn public_reduction_wrong_arity_refuses_before_admission_and_recovers() {
    for operation in ["prod","var","std"] {
        for arity in [0,2] {
            let mut ir:serde_json::Value=serde_json::from_str(&reduction_ir(operation,&[2],&[])).unwrap();
            ir["effects"][1]["inputs"]=serde_json::json!((0..arity).map(|_|"%0").collect::<Vec<_>>());
            let calls=Rc::new(Cell::new(0usize));let recorded=Rc::clone(&calls);
            let result=with_replay_memory_admission(
                move |_| {recorded.set(recorded.get()+1);Ok(())},
                || interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(),&[2.0,3.0]),
            ).unwrap();
            assert!(!result.supported);assert_eq!(calls.get(),0);
            assert_numeric(&interpret_program_ad_effect_ir_value_and_gradient(&reduction_ir("prod",&[2],&[]),&[2.0,3.0]).unwrap(),6.0,&[3.0,2.0]);
        }
    }
}
