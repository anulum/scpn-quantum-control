// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public trapezoid admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

fn trapezoid_ir(operation: &str, target_shape: &[usize]) -> String {
    serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[2,3],"dtype":"float64","effect":0},
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

#[test]
fn public_trapezoid_grid_families_axes_values_and_gradients_observe_owned_cancellation() {
    // Independent segment areas and endpoint/interior weights for two rows.
    for (operation,shape,expected,gradient) in [
        ("trapezoid",vec![2],15.5,vec![0.5,1.0,0.5,0.5,1.0,0.5]),
        ("trapezoid:axis:-1:dx:2",vec![2],31.0,vec![1.0,2.0,1.0,1.0,2.0,1.0]),
        ("trapezoid:axis:0",vec![3],12.0,vec![0.5;6]),
        ("trapezoid:x:0,1:axis:-2",vec![3],12.0,vec![0.5;6]),
        ("trapezoid:axis:1:x:0,1,3",vec![2],25.5,vec![0.5,1.5,1.0,0.5,1.5,1.0]),
        ("trapezoid:x:3,1,0:axis:1",vec![2],-21.0,vec![-1.0,-1.5,-0.5,-1.0,-1.5,-0.5]),
        ("trapezoid:axis:1:x:0,2,1",vec![2],1.0,vec![1.0,0.5,-0.5,1.0,0.5,-0.5]),
        ("trapezoid:axis:1:xfull:0,1,3,0,2,5",vec![2],36.5,vec![0.5,1.5,1.0,1.0,2.5,1.5]),
        ("trapezoid:axis:0:xfull:0,1,2,2,4,6",vec![3],40.5,vec![1.0,1.5,2.0,1.0,1.5,2.0]),
        ("trapezoid:dx:0",vec![2],0.0,vec![0.0;6]),
    ] {
        let source=trapezoid_ir(operation,&shape);
        let inputs=[1.0,2.0,4.0,3.0,5.0,9.0];
        let calls=Rc::new(Cell::new(0usize));
        let recorded=Rc::clone(&calls);
        let baseline=with_replay_checkpoint(
            move || { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
        ).unwrap();
        assert!(baseline.supported,"{:?}",baseline.blocked_reasons);
        assert_eq!(baseline.value,Some(expected));
        assert_eq!(baseline.gradient,gradient);
        assert!(calls.get()>1);
        for boundary in 1..=calls.get() {
            let observed=Rc::new(Cell::new(0usize));
            let recorded=Rc::clone(&observed);
            let refused=with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get()+1);
                    if recorded.get()>=boundary { Err("trapezoid owner cancelled".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source,&inputs),
            );
            match refused {
                Err(reason)=>assert!(reason.contains("trapezoid owner cancelled"),"{reason}"),
                Ok(result)=>{
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|r|r.contains("trapezoid owner cancelled")),"{:?}",result.blocked_reasons);
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
fn public_trapezoid_invalid_grids_shapes_and_widths_refuse_before_retry() {
    let inputs=[1.0,2.0,4.0,3.0,5.0,9.0];
    for operation in [
        "trapezoid:axis:2", "trapezoid:axis:-3", "trapezoid:axis:0:axis:1",
        "trapezoid:dx:1:dx:2", "trapezoid:dx:1:x:0,1,2", "trapezoid:dx:NaN",
        "trapezoid:x:", "trapezoid:x:0,1", "trapezoid:x:0,1,2,3",
        "trapezoid:x:0,NaN,2", "trapezoid:xfull:0,1,2", "trapezoid:xfull:0,1,2,3,4,5,6",
        "trapezoid:x:-1e308,1e308,0", "trapezoid:unknown:1", "trapezoid:axis",
    ] {
        let refused=interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir(operation,&[2]),&inputs).unwrap();
        assert!(!refused.supported,"{operation}");
        let retry=interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir("trapezoid",&[2]),&inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value,Some(15.5));
        assert_eq!(retry.gradient,vec![0.5,1.0,0.5,0.5,1.0,0.5]);
    }
    let refused=interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir("trapezoid",&[3]),&inputs).unwrap();
    assert!(!refused.supported);
    assert!(refused.blocked_reasons.iter().any(|r|r.contains("target shape")));
    let mut impossible: serde_json::Value=serde_json::from_str(&trapezoid_ir("trapezoid",&[2])).unwrap();
    impossible["ssa_values"][0]["shape"]=serde_json::json!([usize::MAX,2]);
    let refused=interpret_program_ad_effect_ir_value_and_gradient(&impossible.to_string(),&inputs).unwrap();
    assert!(!refused.supported);
    let retry=interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir("trapezoid",&[2]),&inputs).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value,Some(15.5));
}


#[test]
fn public_trapezoid_workspace_budget_preserves_grid_contract_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    for (operation, source_shape, target_shape, inputs, value, gradient) in [
        ("trapezoid", vec![2,3], vec![2], vec![1.0,2.0,4.0,3.0,5.0,9.0], 15.5, vec![0.5,1.0,0.5,0.5,1.0,0.5]),
        ("trapezoid:axis:-1:dx:2", vec![2,3], vec![2], vec![1.0,2.0,4.0,3.0,5.0,9.0], 31.0, vec![1.0,2.0,1.0,1.0,2.0,1.0]),
        ("trapezoid:x:3,1,0:axis:1", vec![2,3], vec![2], vec![1.0,2.0,4.0,3.0,5.0,9.0], -21.0, vec![-1.0,-1.5,-0.5,-1.0,-1.5,-0.5]),
        ("trapezoid:axis:1:xfull:0,1,3,0,2,5", vec![2,3], vec![2], vec![1.0,2.0,4.0,3.0,5.0,9.0], 36.5, vec![0.5,1.5,1.0,1.0,2.5,1.5]),
        ("trapezoid:axis:0:xfull:0,1,2,2,4,6", vec![2,3], vec![3], vec![1.0,2.0,4.0,3.0,5.0,9.0], 40.5, vec![1.0,1.5,2.0,1.0,1.5,2.0]),
        ("trapezoid:x:0,2", vec![2], vec![], vec![3.0,5.0], 8.0, vec![1.0,1.0]),
        ("trapezoid:dx:0", vec![2], vec![], vec![3.0,5.0], 0.0, vec![0.0,0.0]),
    ] {
        let mut ir: serde_json::Value = serde_json::from_str(&trapezoid_ir(operation, &target_shape)).unwrap();
        ir["ssa_values"][0]["shape"] = serde_json::json!(source_shape);
        let ir = ir.to_string();
        let source_count = inputs.len();
        let output_count: usize = target_shape.iter().product();
        // Independent buffer inventory: the adapter accumulation phase retains
        // three source-sized numeric copies plus the cloned target cotangent.
        // Source/contribution/reduced/broadcast/index rank buffers and cotangent
        // shape cover its conservative coordinate peak, separate from SSA tape.
        let workspace = (3*source_count+output_count)*8
            + (5*source_shape.len()+target_shape.len())*std::mem::size_of::<usize>();
        let retained = source_count+output_count+1;
        let expected = ReplayMemoryRequest { forward_bytes: retained*8,
            adjoint_bytes: (retained+source_count)*8, intermediate_bytes: workspace };
        let total = expected.total_bytes().unwrap();
        for budget in [total-1,total,total+1] {
            let calls = Rc::new(Cell::new(0usize)); let recorded = Rc::clone(&calls);
            let result = with_replay_memory_admission(
                move |request| {
                    recorded.set(recorded.get()+1); assert_eq!(request,expected);
                    if request.total_bytes()? > budget { Err("trapezoid workspace budget refused".to_owned()) } else { Ok(()) }
                }, || interpret_program_ad_effect_ir_value_and_gradient(&ir,&inputs),
            );
            assert_eq!(calls.get(),1);
            match result {
                Err(reason) => { assert!(budget<total); assert!(reason.contains("trapezoid workspace budget refused")); }
                Ok(result) => {
                    assert_eq!(result.supported,budget>=total);
                    if result.supported { assert_eq!(result.value,Some(value)); assert_eq!(result.gradient,gradient); }
                    else { assert!(result.blocked_reasons.iter().any(|r|r.contains("trapezoid workspace budget refused"))); }
                }
            }
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir,&inputs).unwrap();
            assert!(retry.supported,"{:?}",retry.blocked_reasons); assert_eq!(retry.value,Some(value)); assert_eq!(retry.gradient,gradient);
        }
    }
}

#[test]
fn public_trapezoid_bad_metadata_and_shapes_refuse_before_numeric_admission() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (operation, target) in [
        ("trapezoid:axis:2",vec![2]), ("trapezoid:axis:-3",vec![2]),
        ("trapezoid:axis:0:axis:1",vec![2]), ("trapezoid:dx:NaN",vec![2]),
        ("trapezoid:dx:1:x:0,1,2",vec![2]), ("trapezoid:x:",vec![2]),
        ("trapezoid:x:0,NaN,2",vec![2]), ("trapezoid:x:0,1",vec![2]),
        ("trapezoid:xfull:0,1,2",vec![2]), ("trapezoid:unknown:1",vec![2]),
        ("trapezoid:axis",vec![2]), ("trapezoid",vec![3]), ("trapezoid",vec![2,1]),
    ] {
        let calls = Rc::new(Cell::new(0usize)); let recorded = Rc::clone(&calls);
        let ir = trapezoid_ir(operation,&target);
        let result = with_replay_memory_admission(
            move |_| { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir,&[1.0,2.0,4.0,3.0,5.0,9.0]),
        ).unwrap();
        assert!(!result.supported,"{operation}"); assert_eq!(calls.get(),0);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir("trapezoid",&[2]),&[1.0,2.0,4.0,3.0,5.0,9.0]).unwrap();
        assert!(retry.supported); assert_eq!(retry.value,Some(15.5)); assert_eq!(retry.gradient,vec![0.5,1.0,0.5,0.5,1.0,0.5]);
    }
}

#[test]
fn public_scalar_forward_trapezoid_remains_unsupported_before_numeric_admission() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let calls = Rc::new(Cell::new(0usize)); let recorded = Rc::clone(&calls);
    let result = with_replay_memory_admission(
        move |_| { recorded.set(recorded.get()+1); Ok(()) },
        || interpret_program_ad_effect_ir_forward(&trapezoid_ir("trapezoid",&[2]),&[1.0]),
    ).unwrap();
    assert!(!result.supported); assert_eq!(calls.get(),0);
    assert!(result.blocked_reasons.iter().any(|r|r.contains("trapezoid requires ranked numeric replay")));
}


#[test]
fn public_trapezoid_unranked_short_axis_and_wrong_arity_refuse_before_admission() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (source_shape,inputs,arity) in [(vec![],vec![1.0],1),(vec![1],vec![1.0],1),(vec![2,3],vec![1.0;6],0),(vec![2,3],vec![1.0;6],2)] {
        let mut ir: serde_json::Value = serde_json::from_str(&trapezoid_ir("trapezoid",&[])).unwrap();
        ir["ssa_values"][0]["shape"] = serde_json::json!(source_shape);
        ir["effects"][1]["inputs"] = serde_json::json!((0..arity).map(|_|"%0").collect::<Vec<_>>());
        let calls = Rc::new(Cell::new(0usize)); let recorded = Rc::clone(&calls);
        let result = with_replay_memory_admission(
            move |_| { recorded.set(recorded.get()+1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(),&inputs),
        ).unwrap();
        assert!(!result.supported); assert_eq!(calls.get(),0);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&trapezoid_ir("trapezoid",&[2]),&[1.0,2.0,4.0,3.0,5.0,9.0]).unwrap();
        assert!(retry.supported); assert_eq!(retry.value,Some(15.5)); assert_eq!(retry.gradient,vec![0.5,1.0,0.5,0.5,1.0,0.5]);
    }
}
