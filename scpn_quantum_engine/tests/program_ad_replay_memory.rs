// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public retained replay memory admission tests

use scpn_quantum_program_ad_replay::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    with_replay_memory_admission, ReplayMemoryRequest,
};
use std::cell::Cell;
use std::rc::Rc;

fn scalar_ir() -> String {
    serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[],"dtype":"float64","effect":1}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"pure","target":"%1","inputs":["%0","%0"],"version":0,"ordering":1,"operation":"mul"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string()
}

#[test]
fn public_retained_replay_admission_is_inclusive_and_recovers_after_refusal() {
    for budget in [39usize, 40, 41] {
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let result = with_replay_memory_admission(
            move |request| {
                recorded.set(recorded.get() + 1);
                // Two retained float64 primals, two adjoints and one gradient.
                assert_eq!(
                    request,
                    ReplayMemoryRequest {
                        forward_bytes: 16,
                        adjoint_bytes: 24,
                        intermediate_bytes: 0
                    }
                );
                if request.total_bytes()? > budget {
                    Err("retained replay exceeds owned budget".to_owned())
                } else {
                    Ok(())
                }
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0]),
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(result.supported, budget >= 40);
        if result.supported {
            assert_eq!(result.value, Some(4.0));
            assert_eq!(result.gradient, [4.0]);
        } else {
            assert!(result
                .blocked_reasons
                .iter()
                .any(|r| r.contains("retained replay exceeds owned budget")));
        }
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, [4.0]);
    }
    let result = with_replay_memory_admission(
        |request| {
            assert_eq!(
                request,
                ReplayMemoryRequest {
                    forward_bytes: 16,
                    adjoint_bytes: 0,
                    intermediate_bytes: 0
                }
            );
            Ok(())
        },
        || interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]),
    )
    .unwrap();
    assert!(result.supported);
    assert_eq!(result.value, Some(4.0));
}

#[test]
fn public_child_replay_cannot_clear_parent_retained_memory_refusal() {
    let child_calls = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&child_calls);
    let refused = with_replay_memory_admission(
        |_| Err("parent retained memory refused".to_owned()),
        || {
            with_replay_memory_admission(
                move |_| {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]),
            )
        },
    );
    assert!(refused
        .unwrap_err()
        .contains("parent retained memory refused"));
    assert_eq!(child_calls.get(), 0);
    let retry = interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(4.0));
}

#[test]
fn public_retained_sum_overflow_refuses_before_admission_callback_or_values() {
    for divisor in [8usize, 32] {
        let side = isize::MAX as usize / divisor;
        let source = serde_json::json!({
            "format":"program_ad_effect_ir.v1",
            "ssa_values":[
                {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
                {"name":"%1","producer":1,"version":0,"shape":[side],"dtype":"float64","effect":1},
                {"name":"%2","producer":2,"version":0,"shape":[side],"dtype":"float64","effect":2},
                {"name":"%3","producer":3,"version":0,"shape":[],"dtype":"float64","effect":3}
            ],
            "effects":[
                {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
                {"index":1,"kind":"pure","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":"broadcast_to"},
                {"index":2,"kind":"pure","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"broadcast_to"},
                {"index":3,"kind":"primitive","target":"%3","inputs":["%2"],"version":0,"ordering":3,"operation":"sum"}
            ],
            "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
        }).to_string();
        let called = Rc::new(Cell::new(false));
        let recorded = Rc::clone(&called);
        let result = with_replay_memory_admission(
            move |_| {
                recorded.set(true);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]),
        )
        .unwrap();
        assert!(!result.supported);
        assert!(
            result
                .blocked_reasons
                .iter()
                .any(|r| r.contains("retained") && r.contains("addressability")),
            "{:?}",
            result.blocked_reasons
        );
        assert!(!called.get());
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, [4.0]);
    }
}

#[test]
fn public_elementwise_shape_metadata_refuses_before_retained_admission_and_recovers() {
    for operation in ["mul", "exp"] {
        let mut source: serde_json::Value = serde_json::from_str(&scalar_ir()).unwrap();
        source["effects"][1]["operation"] = serde_json::json!(operation);
        if operation == "exp" {
            source["effects"][1]["inputs"] = serde_json::json!(["%0"]);
        }
        source["ssa_values"][1]["shape"] = serde_json::json!([3]);
        let called = Rc::new(Cell::new(false));
        let recorded = Rc::clone(&called);
        let refused = with_replay_memory_admission(
            move |_| {
                recorded.set(true);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source.to_string(), &[0.0]),
        )
        .unwrap();
        assert!(!refused.supported);
        assert!(
            refused
                .blocked_reasons
                .iter()
                .any(|r| r.contains("target shape metadata")),
            "{:?}",
            refused.blocked_reasons
        );
        assert!(!called.get());
        source["ssa_values"][1]["shape"] = serde_json::json!([]);
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&source.to_string(), &[0.0]).unwrap();
        assert!(retry.supported);
        if operation == "mul" {
            assert_eq!(retry.value, Some(0.0));
            assert_eq!(retry.gradient, [0.0]);
        } else {
            assert_eq!(retry.value, Some(1.0));
            assert_eq!(retry.gradient, [1.0]);
        }
    }
}

#[test]
fn public_memory_policy_restores_after_unwind_and_allows_owned_reentrant_replay() {
    let unwound = std::panic::catch_unwind(|| {
        with_replay_memory_admission(
            |_| panic!("owned memory admission unwind"),
            || interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]),
        )
    });
    assert!(unwound.is_err());
    let retry = interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(4.0));
    let entered = Rc::new(Cell::new(false));
    let observed = Rc::clone(&entered);
    let total = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&total);
    let result = with_replay_memory_admission(
        move |request| {
            recorded.set(recorded.get() + request.total_bytes()?);
            if !observed.replace(true) {
                let nested = interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0])?;
                assert!(nested.supported);
                assert_eq!(nested.value, Some(4.0));
            }
            Ok(())
        },
        || interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0]),
    )
    .unwrap();
    assert!(result.supported);
    assert_eq!(result.value, Some(4.0));
    assert_eq!(result.gradient, [4.0]);
    assert!(entered.get());
    assert_eq!(total.get(), 56);
}

fn ranked_binary_ir(left: &[usize], right: &[usize], target: &[usize]) -> String {
    serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":left,"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":right,"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":target,"dtype":"float64","effect":2},
            {"name":"%3","producer":3,"version":0,"shape":[],"dtype":"float64","effect":3}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["left"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"parameter","target":"%1","inputs":["right"],"version":0,"ordering":1,"operation":"parameter"},
            {"index":2,"kind":"pure","target":"%2","inputs":["%0","%1"],"version":0,"ordering":2,"operation":"mul"},
            {"index":3,"kind":"primitive","target":"%3","inputs":["%2"],"version":0,"ordering":3,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string()
}

#[test]
fn public_ranked_binary_admission_preserves_broadcast_oracles_and_recovery() {
    for (left, right, target, inputs, value, gradient) in [
        (
            vec![2, 1],
            vec![1, 3],
            vec![2, 3],
            vec![2.0, 3.0, 5.0, 7.0, 11.0],
            115.0,
            vec![23.0, 23.0, 5.0, 5.0, 5.0],
        ),
        (
            vec![2, 3],
            vec![3],
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0],
            67.0,
            vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0],
        ),
        (
            vec![],
            vec![2, 1],
            vec![2, 1],
            vec![2.0, 5.0, 7.0],
            24.0,
            vec![12.0, 2.0, 2.0],
        ),
        (
            vec![1; 300],
            vec![1],
            vec![1; 300],
            vec![2.0, 3.0],
            6.0,
            vec![3.0, 2.0],
        ),
    ] {
        let source = ranked_binary_ir(&left, &right, &target);
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let result = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        )
        .unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(calls.get(), 1);
        assert_eq!(result.value, Some(value));
        assert_eq!(result.gradient, gradient);
        for wrong in [vec![], vec![1], vec![1; target.len() + 1]] {
            let source = ranked_binary_ir(&left, &right, &wrong);
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let refused = with_replay_memory_admission(
                move |_| {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            )
            .unwrap();
            assert!(!refused.supported);
            assert_eq!(calls.get(), 0);
            assert!(refused
                .blocked_reasons
                .iter()
                .any(|r| r.contains("target shape metadata")));
            let retry = interpret_program_ad_effect_ir_value_and_gradient(
                &ranked_binary_ir(&left, &right, &target),
                &inputs,
            )
            .unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(value));
            assert_eq!(retry.gradient, gradient);
        }
    }
}

#[test]
fn public_incompatible_broadcast_refuses_before_memory_callback_and_recovers() {
    for target in [vec![], vec![2, 1], vec![3, 1]] {
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let source = ranked_binary_ir(&[2, 1], &[3, 1], &target);
        let refused = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || {
                interpret_program_ad_effect_ir_value_and_gradient(
                    &source,
                    &[1.0, 2.0, 3.0, 4.0, 5.0],
                )
            },
        )
        .unwrap();
        assert!(!refused.supported);
        assert_eq!(calls.get(), 0);
        assert!(refused
            .blocked_reasons
            .iter()
            .any(|r| r.contains("cannot broadcast")));
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, [4.0]);
    }
}

#[test]
fn public_elementwise_arity_refuses_before_memory_admission_and_recovers() {
    for (operations, expected) in [
        (&["add", "sub", "mul", "div", "pow"][..], 2usize),
        (
            &[
                "sin",
                "cos",
                "exp",
                "expm1",
                "log",
                "log1p",
                "sqrt",
                "tan",
                "tanh",
                "arcsin",
                "arccos",
                "reciprocal",
                "abs",
            ][..],
            1usize,
        ),
    ] {
        for operation in operations {
            for count in [0usize, 1, 2, 3] {
                if count == expected {
                    continue;
                }
                let mut source: serde_json::Value = serde_json::from_str(&scalar_ir()).unwrap();
                source["effects"][1]["operation"] = serde_json::json!(operation);
                source["effects"][1]["inputs"] = serde_json::json!(vec!["%0"; count]);
                for adjoint in [false, true] {
                    let calls = Rc::new(Cell::new(0usize));
                    let recorded = Rc::clone(&calls);
                    let refused = with_replay_memory_admission(
                        move |_| {
                            recorded.set(recorded.get() + 1);
                            Ok(())
                        },
                        || {
                            if adjoint {
                                let result = interpret_program_ad_effect_ir_value_and_gradient(
                                    &source.to_string(),
                                    &[2.0],
                                )
                                .unwrap();
                                (result.supported, result.blocked_reasons)
                            } else {
                                let result = interpret_program_ad_effect_ir_forward(
                                    &source.to_string(),
                                    &[2.0],
                                )
                                .unwrap();
                                (result.supported, result.blocked_reasons)
                            }
                        },
                    );
                    assert!(!refused.0, "{operation} accepted {count} inputs");
                    assert_eq!(calls.get(), 0, "{operation} admitted {count} inputs");
                    let reason = if expected == 2 {
                        "requires two inputs"
                    } else {
                        "requires one input"
                    };
                    assert!(
                        refused.1.iter().any(|r| r.contains(reason)),
                        "{:?}",
                        refused.1
                    );
                    let retry =
                        interpret_program_ad_effect_ir_value_and_gradient(&scalar_ir(), &[2.0])
                            .unwrap();
                    assert!(retry.supported);
                    assert_eq!(retry.value, Some(4.0));
                    assert_eq!(retry.gradient, [4.0]);
                    let forward =
                        interpret_program_ad_effect_ir_forward(&scalar_ir(), &[2.0]).unwrap();
                    assert!(forward.supported);
                    assert_eq!(forward.value, Some(4.0));
                }
            }
        }
    }
}
