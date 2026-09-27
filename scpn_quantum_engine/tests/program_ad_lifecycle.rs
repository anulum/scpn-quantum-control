// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public program-AD lifecycle ownership tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use std::cell::Cell;
use std::rc::Rc;

const IR: &str = r#"{
    "format": "program_ad_effect_ir.v1",
    "ssa_values": [
        {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
        {"name":"%1","producer":1,"version":0,"shape":[],"dtype":"float64","effect":1}
    ],
    "effects": [
        {"index":0,"kind":"parameter","target":"%0","inputs":["p"],"version":0,"ordering":0,"operation":"parameter"},
        {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":"cumsum:shape:1:axis:flat:out:0"}
    ],
    "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
}"#;

#[test]
fn public_replay_observes_each_owned_boundary_and_recovers() {
    let observed = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&observed);
    let result = with_replay_checkpoint(
        move || { recorded.set(recorded.get() + 1); Ok(()) },
        || interpret_program_ad_effect_ir_value_and_gradient(IR, &[2.0]),
    ).unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(2.0));
    assert_eq!(result.gradient, [1.0]);
    assert!(observed.get() > 1);
    for boundary in 1..=observed.get() {
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let result = with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                if recorded.get() >= boundary { Err("owned replay interrupted".to_owned()) } else { Ok(()) }
            },
            || interpret_program_ad_effect_ir_value_and_gradient(IR, &[2.0]),
        );
        match result {
            Err(reason) => assert!(reason.contains("owned replay interrupted"), "{reason}"),
            Ok(result) => {
                assert!(!result.supported);
                assert!(result.blocked_reasons.iter().any(|reason| reason.contains("owned replay interrupted")), "{:?}", result.blocked_reasons);
            }
        }
        replay_checkpoint().unwrap();
        let retry = interpret_program_ad_effect_ir_value_and_gradient(IR, &[2.0]).unwrap();
        assert!(retry.supported, "{:?}", retry.blocked_reasons);
        assert_eq!(retry.value, Some(2.0));
        assert_eq!(retry.gradient, [1.0]);
    }
}

#[test]
fn public_nested_policy_cannot_clear_parent_refusal() {
    let result = with_replay_checkpoint(
        || Err("parent cancelled".to_owned()),
        || with_replay_checkpoint(|| Ok(()), replay_checkpoint),
    );
    assert_eq!(result.unwrap_err(), "parent cancelled");
    replay_checkpoint().unwrap();
}

#[test]
fn public_policy_restores_parent_after_return_and_panic() {
    with_replay_checkpoint(|| Err("retained parent".to_owned()), || {
        with_replay_checkpoint(|| Ok(()), || {});
        assert_eq!(replay_checkpoint().unwrap_err(), "retained parent");
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_replay_checkpoint(|| Ok(()), || panic!("owned child unwind"));
        }));
        assert!(panic.is_err());
        assert_eq!(replay_checkpoint().unwrap_err(), "retained parent");
    });
    replay_checkpoint().unwrap();
}

#[test]
fn public_policy_callback_allows_owned_reentrant_replay() {
    let entered = Rc::new(Cell::new(false));
    let observed = Rc::clone(&entered);
    with_replay_checkpoint(
        move || {
            if !observed.replace(true) {
                with_replay_checkpoint(|| Ok(()), replay_checkpoint)?;
            }
            Ok(())
        },
        replay_checkpoint,
    ).unwrap();
    assert!(entered.get());
    replay_checkpoint().unwrap();
}


#[test]
fn public_checkpointed_cumulative_sum_preserves_signed_zero() {
    let result = with_replay_checkpoint(
        || Ok(()),
        || interpret_program_ad_effect_ir_value_and_gradient(IR, &[-0.0]),
    ).unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(result.gradient, [1.0]);
}

#[test]
fn public_shaped_unary_reductions_observe_owned_boundaries_and_recover() {
    for (unary, expected_item, expected_derivative) in [
        ("sin", 0.25_f64.sin(), 0.25_f64.cos()),
        ("exp", 0.25_f64.exp(), 0.25_f64.exp()),
        ("log1p", 0.25_f64.ln_1p(), 0.8),
        ("sqrt", 0.5, 1.0),
    ] {
        for reduction in ["sum", "mean"] {
            let ir = serde_json::json!({
                "format": "program_ad_effect_ir.v1",
                "ssa_values": [
                    {"name":"%0","producer":0,"version":0,"shape":[257],"dtype":"float64","effect":0},
                    {"name":"%1","producer":1,"version":0,"shape":[257],"dtype":"float64","effect":1},
                    {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
                ],
                "effects": [
                    {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                    {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":unary},
                    {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":reduction}
                ],
                "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
            }).to_string();
            let inputs = [0.25; 257];
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let baseline = with_replay_checkpoint(
                move || { recorded.set(recorded.get() + 1); Ok(()) },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs),
            ).unwrap();
            assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
            let scale = if reduction == "mean" { 1.0 / inputs.len() as f64 } else { 1.0 };
            let expected = expected_item * inputs.len() as f64 * scale;
            assert!((baseline.value.unwrap() - expected).abs() < 1.0e-10);
            assert_eq!(baseline.gradient.len(), inputs.len());
            for derivative in &baseline.gradient {
                assert!((*derivative - expected_derivative * scale).abs() < 1.0e-12);
            }
            assert!(calls.get() > 1);
            for boundary in 1..=calls.get() {
                let observed = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&observed);
                let refused = with_replay_checkpoint(
                    move || {
                        recorded.set(recorded.get() + 1);
                        if recorded.get() >= boundary {
                            Err("shaped replay cancelled".to_owned())
                        } else {
                            Ok(())
                        }
                    },
                    || interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs),
                );
                match refused {
                    Err(reason) => assert!(reason.contains("shaped replay cancelled"), "{reason}"),
                    Ok(result) => {
                        assert!(!result.supported);
                        assert!(result.blocked_reasons.iter().any(|reason| reason.contains("shaped replay cancelled")), "{:?}", result.blocked_reasons);
                    }
                }
                replay_checkpoint().unwrap();
                let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs).unwrap();
                assert!(retry.supported, "{:?}", retry.blocked_reasons);
                assert_eq!(retry.value, baseline.value);
                assert_eq!(retry.gradient, baseline.gradient);
            }
        }
    }
}

#[test]
fn public_sum_and_mean_lifecycle_preserve_signed_zero() {
    for operation in ["sum", "mean"] {
        let ir = IR.replace("cumsum:shape:1:axis:flat:out:0", operation);
        let result = with_replay_checkpoint(
            || Ok(()),
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &[-0.0]),
        ).unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(result.value.unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(result.gradient, [1.0]);
    }
}

#[test]
fn public_structural_replay_observes_owned_buffers_and_recovers() {
    for (operation, source_shape, output_shape, duplicated, expected_value, derivative) in [
        ("reshape", vec![3, 2], vec![2, 3], false, 1.5, 1.0),
        ("ravel", vec![3, 2], vec![6], false, 1.5, 1.0),
        ("transpose", vec![3, 2], vec![2, 3], false, 1.5, 1.0),
        ("broadcast_to", vec![1, 2], vec![3, 2], false, 1.5, 3.0),
        ("concatenate:axis:0", vec![3, 2], vec![6, 2], true, 3.0, 2.0),
        ("stack:axis:0", vec![3, 2], vec![2, 3, 2], true, 3.0, 2.0),
        ("sum:axis:0", vec![3, 2], vec![2], false, 1.5, 1.0),
        ("mean:axis:1", vec![3, 2], vec![3], false, 0.75, 0.5),
    ] {
        let count = source_shape.iter().product::<usize>();
        let effect_inputs = if duplicated { vec!["%0", "%0"] } else { vec!["%0"] };
        let ir = serde_json::json!({
            "format": "program_ad_effect_ir.v1",
            "ssa_values": [
                {"name":"%0","producer":0,"version":0,"shape":source_shape,"dtype":"float64","effect":0},
                {"name":"%1","producer":1,"version":0,"shape":output_shape,"dtype":"float64","effect":1},
                {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
            ],
            "effects": [
                {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                {"index":1,"kind":"primitive","target":"%1","inputs":effect_inputs,"version":0,"ordering":1,"operation":operation},
                {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
            ],
            "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
        }).to_string();
        let inputs = vec![0.25; count];
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || { recorded.set(recorded.get() + 1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs),
        ).unwrap();
        assert!(baseline.supported, "{operation}: {:?}", baseline.blocked_reasons);
        assert_eq!(baseline.value, Some(expected_value));
        assert_eq!(baseline.gradient, vec![derivative; count]);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("structural replay cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("structural replay cancelled"), "{operation}: {reason}"),
                Ok(result) => {
                    assert!(!result.supported, "{operation}");
                    assert!(result.blocked_reasons.iter().any(|reason| reason.contains("structural replay cancelled")), "{operation}: {:?}", result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs).unwrap();
            assert!(retry.supported, "{operation}: {:?}", retry.blocked_reasons);
            assert_eq!(retry.value, baseline.value, "{operation}");
            assert_eq!(retry.gradient, baseline.gradient, "{operation}");
        }
    }
}

#[test]
fn public_product_groups_observe_owned_workspaces_and_recover() {
    for (operation, output_shape, parameters, expected_value, expected_gradient) in [
        ("prod", vec![], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 5040.0,
            vec![2520.0, 1680.0, 1260.0, 1008.0, 840.0, 720.0]),
        ("prod:axis:0", vec![3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 56.0,
            vec![5.0, 6.0, 7.0, 2.0, 3.0, 4.0]),
        ("prod:axis:1", vec![2], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 234.0,
            vec![12.0, 8.0, 6.0, 42.0, 35.0, 30.0]),
        ("prod", vec![], vec![0.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0.0,
            vec![2520.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("prod:axis:0", vec![3], vec![0.0, 3.0, 4.0, 5.0, 0.0, 7.0], 28.0,
            vec![5.0, 0.0, 7.0, 0.0, 3.0, 4.0]),
    ] {
        let ir = serde_json::json!({
            "format":"program_ad_effect_ir.v1",
            "ssa_values":[
                {"name":"%0","producer":0,"version":0,"shape":[2,3],"dtype":"float64","effect":0},
                {"name":"%1","producer":1,"version":0,"shape":output_shape,"dtype":"float64","effect":1},
                {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
            ],
            "effects":[
                {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":operation},
                {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
            ],
            "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
        }).to_string();
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || { recorded.set(recorded.get() + 1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &parameters),
        ).unwrap();
        assert!(baseline.supported, "{operation}: {:?}", baseline.blocked_reasons);
        assert_eq!(baseline.value, Some(expected_value));
        assert_eq!(baseline.gradient, expected_gradient);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("product workspace owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &parameters),
            );
            match refused {
                Err(reason) => assert!(reason.contains("product workspace owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|reason| reason.contains("product workspace owner cancelled")), "{:?}", result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &parameters).unwrap();
            assert!(retry.supported, "{:?}", retry.blocked_reasons);
            assert_eq!(retry.value, baseline.value);
            assert_eq!(retry.gradient, baseline.gradient);
        }
        let multiple_zeros = [0.0, 0.0, 4.0, 0.0, 6.0, 7.0];
        let refused = interpret_program_ad_effect_ir_value_and_gradient(&ir, &multiple_zeros).unwrap();
        assert!(!refused.supported);
        assert!(refused.blocked_reasons.iter().any(|reason| reason.contains("at most one zero")), "{:?}", refused.blocked_reasons);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &parameters).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, baseline.value);
        assert_eq!(retry.gradient, baseline.gradient);
    }
}

#[test]
fn public_moment_replay_observes_owned_workspaces_and_recovers() {
    for (operation, shape, expected, gradient) in [
        ("var", vec![], 5.0, vec![-1.5, -0.5, 0.5, 1.5]),
        ("var:ddof:2", vec![], 10.0, vec![-3.0, -1.0, 1.0, 3.0]),
        ("var:axis:0", vec![2], 8.0, vec![-2.0, -2.0, 2.0, 2.0]),
        ("std:axis:1", vec![2], 2.0, vec![-0.5, 0.5, -0.5, 0.5]),
    ] {
        let ir = serde_json::json!({
            "format":"program_ad_effect_ir.v1",
            "ssa_values":[
                {"name":"%0","producer":0,"version":0,"shape":[2,2],"dtype":"float64","effect":0},
                {"name":"%1","producer":1,"version":0,"shape":shape,"dtype":"float64","effect":1},
                {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
            ],
            "effects":[
                {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":operation},
                {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
            ],
            "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
        });
        let source = ir.to_string();
        let inputs = [2.0, 4.0, 6.0, 8.0];
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || { recorded.set(recorded.get() + 1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        ).unwrap();
        assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
        assert_eq!(baseline.value, Some(expected));
        assert_eq!(baseline.gradient, gradient);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary { Err("moment owner cancelled".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("moment owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|reason| reason.contains("moment owner cancelled")), "{:?}", result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, baseline.value);
            assert_eq!(retry.gradient, baseline.gradient);
        }
        let mut invalid = ir;
        invalid["effects"][1]["operation"] = serde_json::json!("std:axis:1:ddof:2");
        invalid["ssa_values"][1]["shape"] = serde_json::json!([2]);
        let refused = interpret_program_ad_effect_ir_value_and_gradient(&invalid.to_string(), &inputs).unwrap();
        assert!(!refused.supported);
        assert!(refused.blocked_reasons.iter().any(|reason| reason.contains("correction must be less")));
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, baseline.value);
        assert_eq!(retry.gradient, baseline.gradient);
    }
}
