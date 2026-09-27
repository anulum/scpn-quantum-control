// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Public linalg metadata and workspace admission

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use serde_json::json;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn public_linalg_replay_refuses_overflow_and_malformed_metadata_then_recovers() {
    let maximum = usize::MAX;
    let addressable_floats = isize::MAX as usize / std::mem::size_of::<f64>();
    for operation in [
        format!("linalg:det:{maximum}x{maximum}"),
        format!("linalg:inv:{maximum}x{maximum}:0:0"),
        format!("linalg:solve:{maximum}x{maximum}:rhs:{maximum}:0"),
        format!("linalg:solve:1x1:rhs:1x{maximum}:0:0"),
        format!("linalg:solve:1x1:rhs:1x{addressable_floats}:0:0"),
        "linalg:det:0x0".to_owned(),
        "linalg:inv:2x2:2:0".to_owned(),
        "linalg:solve:2x2:rhs:2x0:0:0".to_owned(),
        "linalg:det:4x4:extra".to_owned(),
        "linalg:inv:2x2:0:0:extra".to_owned(),
        "linalg:solve:2x2:rhs:2x2:0:0:extra".to_owned(),
        "linalg:det:4x4".to_owned(),
        "linalg:inv:4x4:0:0".to_owned(),
        "linalg:solve:2x2:rhs:2:0".to_owned(),
    ] {
        let serialization = linalg_objective_ir(&operation, 1);
        let forward = interpret_program_ad_effect_ir_forward(&serialization, &[2.0]).unwrap();
        assert!(!forward.supported, "{operation}");
        assert!(forward.value.is_none());
        assert!(!forward.blocked_reasons.is_empty());
        let reverse =
            interpret_program_ad_effect_ir_value_and_gradient(&serialization, &[2.0]).unwrap();
        assert!(!reverse.supported, "{operation}");
        assert!(reverse.value.is_none());
        assert!(reverse.gradient.is_empty());
        assert!(!reverse.blocked_reasons.is_empty());
        assert_linalg_objective("linalg:inv:2x2:0:0", &[2.0, 0.0, 0.0, 4.0], 0.5,
            &[-0.25, 0.0, 0.0, 0.0]);
    }
}

#[test]
fn public_linalg_replay_preserves_inverse_and_solve_values_and_gradients() {
    assert_linalg_objective("linalg:inv:2x2:0:0", &[2.0, 0.0, 0.0, 4.0], 0.5,
        &[-0.25, 0.0, 0.0, 0.0]);
    assert_linalg_objective("linalg:inv:3x3:1:1",
        &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], 1.0 / 3.0,
        &[0.0, 0.0, 0.0, 0.0, -1.0 / 9.0, 0.0, 0.0, 0.0, 0.0]);
    assert_linalg_objective("linalg:solve:2x2:rhs:2:0",
        &[2.0, 0.0, 0.0, 4.0, 6.0, 8.0], 3.0,
        &[-1.5, -2.0, 0.0, 0.0, 0.5, 0.0]);
    assert_linalg_objective("linalg:solve:2x2:rhs:2x2:0:1",
        &[2.0, 0.0, 0.0, 4.0, 6.0, 10.0, 8.0, 12.0], 5.0,
        &[-2.5, -3.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]);
    let diagonal = [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0];
    assert_linalg_objective("linalg:det:4x4", &diagonal, 120.0,
        &[60.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0,
            0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 24.0]);
    assert_linalg_objective("linalg:inv:4x4:0:0", &diagonal, 0.5,
        &[-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

fn assert_linalg_objective(operation: &str, inputs: &[f64], value: f64, gradient: &[f64]) {
    let serialization = linalg_objective_ir(operation, inputs.len());
    let forward = interpret_program_ad_effect_ir_forward(&serialization, inputs).unwrap();
    assert!(forward.supported, "{operation}: {:?}", forward.blocked_reasons);
    assert!((forward.value.unwrap() - value).abs() <= 1.0e-12);
    let reverse = interpret_program_ad_effect_ir_value_and_gradient(&serialization, inputs).unwrap();
    assert!(reverse.supported, "{operation}: {:?}", reverse.blocked_reasons);
    assert!((reverse.value.unwrap() - value).abs() <= 1.0e-12);
    assert_eq!(reverse.gradient.len(), gradient.len());
    for (actual, expected) in reverse.gradient.iter().zip(gradient) {
        assert!((actual - expected).abs() <= 1.0e-12, "{operation}: {actual} != {expected}");
    }
}

fn linalg_objective_ir(operation: &str, parameters: usize) -> String {
    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..parameters + 2 {
        let target = format!("%{index}");
        ssa_values.push(json!({
            "name": target, "producer": index, "version": 0,
            "shape": [], "dtype": "float64", "effect": index,
        }));
        let (kind, opcode, inputs) = if index < parameters {
            ("parameter", "parameter", vec![format!("input{index}")])
        } else if index == parameters {
            ("primitive", operation, (0..parameters).map(|p| format!("%{p}")).collect())
        } else {
            ("pure", "mul", vec![format!("%{parameters}"), "1.0".to_owned()])
        };
        effects.push(json!({
            "index": index, "kind": kind, "target": target, "inputs": inputs,
            "version": 0, "ordering": index, "operation": opcode,
        }));
    }
    json!({
        "format": "program_ad_effect_ir.v1", "ssa_values": ssa_values,
        "effects": effects, "alias_edges": [], "control_regions": [],
        "phi_nodes": [], "bytecode_offsets": [0],
    }).to_string()
}

#[test]
fn public_general_linalg_replay_observes_owned_boundaries_and_recovers() {
    let diagonal = vec![
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0,
    ];
    let mut solve_parameters = diagonal.clone();
    solve_parameters.extend_from_slice(&[6.0, 9.0, 12.0, 15.0]);
    let mut matrix_rhs_parameters = diagonal.clone();
    matrix_rhs_parameters.extend_from_slice(&[6.0, 10.0, 9.0, 15.0, 12.0, 20.0, 15.0, 25.0]);
    for (operation, parameters, expected_value, expected_gradient) in [
        ("linalg:det:4x4", diagonal.clone(), 120.0,
            vec![60.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 24.0]),
        ("linalg:inv:4x4:0:0", diagonal, 0.5,
            vec![-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("linalg:solve:4x4:rhs:4:0", solve_parameters, 3.0,
            vec![-1.5, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        ("linalg:solve:4x4:rhs:4x2:0:1", matrix_rhs_parameters, 5.0,
            vec![-2.5, -2.5, -2.5, -2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("linalg:det:4x4", vec![0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0], -120.0,
            vec![0.0, -60.0, 0.0, 0.0, -40.0, 0.0, 0.0, 0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0, -24.0]),
    ] {
        let ir = linalg_objective_ir(operation, parameters.len());
        for gradient_surface in [false, true] {
            let replay = || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &parameters)
                        .map(|result| (result.supported, result.value, result.gradient, result.blocked_reasons))
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &parameters)
                        .map(|result| (result.supported, result.value, Vec::new(), result.blocked_reasons))
                }
            };
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let baseline = with_replay_checkpoint(
                move || { recorded.set(recorded.get() + 1); Ok(()) },
                replay,
            ).unwrap();
            assert!(baseline.0, "{operation}: {:?}", baseline.3);
            assert_eq!(baseline.1, Some(expected_value));
            if gradient_surface {
                assert_eq!(baseline.2, expected_gradient);
            }
            assert!(calls.get() > 1);
            for boundary in 1..=calls.get() {
                let observed = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&observed);
                let refused = with_replay_checkpoint(
                    move || {
                        recorded.set(recorded.get() + 1);
                        if recorded.get() >= boundary {
                            Err("general linalg owner cancelled".to_owned())
                        } else {
                            Ok(())
                        }
                    },
                    replay,
                );
                match refused {
                    Err(reason) => assert!(reason.contains("general linalg owner cancelled"), "{operation}: {reason}"),
                    Ok(result) => {
                        assert!(!result.0, "{operation}");
                        assert!(result.3.iter().any(|reason| reason.contains("general linalg owner cancelled")), "{operation}: {:?}", result.3);
                    }
                }
                replay_checkpoint().unwrap();
                let retry = replay().unwrap();
                assert!(retry.0, "{operation}: {:?}", retry.3);
                assert_eq!(retry.1, baseline.1);
                assert_eq!(retry.2, baseline.2);
            }
        }
    }
}

#[test]
fn public_gaussian_lu_solve_workspace_budgets_are_inclusive_and_recover() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    let diagonal = vec![2.0,0.0,0.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,4.0,0.0,0.0,0.0,0.0,5.0];
    let mut solve_many = vec![2.0];
    solve_many.extend([2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0]);
    for (operation, inputs, forward_workspace, reverse_workspace, value, gradient) in [
        ("linalg:inv:1x1:0:0",vec![2.0],32usize,32usize,0.5,vec![-0.25]),
        ("linalg:inv:2x2:0:0",vec![2.0,0.0,0.0,4.0],64,64,0.5,vec![-0.25,0.0,0.0,0.0]),
        ("linalg:det:2x2",vec![2.0,0.0,0.0,4.0],0,0,8.0,vec![4.0,0.0,0.0,2.0]),
        ("linalg:det:02x02",vec![2.0,0.0,0.0,4.0],64,64,8.0,vec![4.0,0.0,0.0,2.0]),
        ("linalg:det:4x4",diagonal,256,512,120.0,vec![60.0,0.0,0.0,0.0,0.0,40.0,0.0,0.0,0.0,0.0,30.0,0.0,0.0,0.0,0.0,24.0]),
        ("linalg:solve:2x2:rhs:2:0",vec![2.0,0.0,0.0,4.0,6.0,8.0],80,96,3.0,vec![-1.5,-2.0,0.0,0.0,0.5,0.0]),
        ("linalg:solve:1x1:rhs:1x8:0:7",solve_many,96,144,8.0,vec![-4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5]),
    ] {
        let ir = linalg_objective_ir(operation, inputs.len());
        for gradient_surface in [false, true] {
            let expected = ReplayMemoryRequest {
                forward_bytes:(inputs.len()+2)*8,
                adjoint_bytes:if gradient_surface { (2*inputs.len()+2)*8 } else { 0 },
                intermediate_bytes:if gradient_surface { reverse_workspace } else { forward_workspace },
            };
            let total = expected.total_bytes().unwrap();
            let replay = || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
                        .map(|r| (r.supported,r.value,r.gradient,r.blocked_reasons))
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &inputs)
                        .map(|r| (r.supported,r.value,Vec::new(),r.blocked_reasons))
                }
            };
            for budget in [total-1,total,total+1] {
                let result = with_replay_memory_admission(
                    move |request| {
                        assert_eq!(request,expected);
                        if request.total_bytes()? > budget { Err("Gaussian/LU/solve workspace budget refused".to_owned()) } else { Ok(()) }
                    },
                    replay,
                );
                match result {
                    Err(reason) => { assert!(budget<total); assert!(reason.contains("Gaussian/LU/solve workspace budget refused")); }
                    Ok(result) => {
                        assert_eq!(result.0,budget>=total);
                        if result.0 {
                            assert!((result.1.unwrap()-value).abs()<=1.0e-12);
                            if gradient_surface { assert_eq!(result.2,gradient); }
                        } else { assert!(result.3.iter().any(|reason|reason.contains("Gaussian/LU/solve workspace budget refused"))); }
                    }
                }
                let retry = replay().unwrap();
                assert!(retry.0,"{:?}",retry.3);
                assert!((retry.1.unwrap()-value).abs()<=1.0e-12);
                if gradient_surface { assert_eq!(retry.2,gradient); }
            }
        }
    }
}

#[test]
fn public_general_linalg_metadata_refuses_before_numeric_memory_callback() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for operation in [
        format!("linalg:inv:{0}x{0}:0:0",usize::MAX),
        format!("linalg:det:{0}x{0}",usize::MAX),
        format!("linalg:solve:1x1:rhs:1x{}:0:0",isize::MAX as usize/8),
        "linalg:inv:2x2:2:0".to_owned(),
        "linalg:solve:2x2:rhs:2:0".to_owned(),
    ] {
        for gradient_surface in [false,true] {
            let callbacks=Rc::new(Cell::new(0usize));
            let recorded=Rc::clone(&callbacks);
            let ir=linalg_objective_ir(&operation,1);
            let result=with_replay_memory_admission(
                move |_| { recorded.set(recorded.get()+1); Ok(()) },
                || {
                    if gradient_surface {
                        interpret_program_ad_effect_ir_value_and_gradient(&ir,&[2.0]).map(|r|r.supported)
                    } else { interpret_program_ad_effect_ir_forward(&ir,&[2.0]).map(|r|r.supported) }
                },
            ).unwrap();
            assert!(!result,"{operation}");
            assert_eq!(callbacks.get(),0);
            assert_linalg_objective("linalg:inv:2x2:0:0",&[2.0,0.0,0.0,4.0],0.5,&[-0.25,0.0,0.0,0.0]);
        }
    }
}
