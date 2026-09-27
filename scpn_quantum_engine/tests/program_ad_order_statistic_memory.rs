// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public order-statistic admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn order_statistic_ir(operation: &str, source_shape: &[usize], target_shape: &[usize]) -> String {
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

#[test]
fn public_order_statistics_sort_and_scatter_with_owned_cancellation() {
    for (operation, shape, expected, gradient) in [
        ("min", vec![], 1.0, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ("max", vec![], 8.0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("median", vec![], 3.5, vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
        (
            "quantile:q:0.25",
            vec![],
            2.25,
            vec![0.0, 0.0, 0.0, 0.75, 0.0, 0.25],
        ),
        (
            "percentile:q:75",
            vec![],
            5.5,
            vec![0.75, 0.0, 0.25, 0.0, 0.0, 0.0],
        ),
        ("median:axis:0", vec![3], 12.0, vec![0.5; 6]),
        (
            "max:axis:1",
            vec![2],
            14.0,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    ] {
        let source = order_statistic_ir(operation, &[2, 3], &shape);
        let inputs = [6.0, 1.0, 4.0, 2.0, 8.0, 3.0];
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        )
        .unwrap();
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
                    if recorded.get() >= boundary {
                        Err("order-statistic owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(
                    reason.contains("order-statistic owner cancelled"),
                    "{reason}"
                ),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(
                        result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("order-statistic owner cancelled")),
                        "{:?}",
                        result.blocked_reasons
                    );
                }
            }
            replay_checkpoint().unwrap();
            let retry =
                interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, baseline.value);
            assert_eq!(retry.gradient, baseline.gradient);
        }
        let tied = interpret_program_ad_effect_ir_value_and_gradient(&source, &[1.0; 6]).unwrap();
        assert!(!tied.supported);
        assert!(tied
            .blocked_reasons
            .iter()
            .any(|r| r.contains("strictly ordered")));
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, baseline.value);
        assert_eq!(retry.gradient, baseline.gradient);
    }
}

#[test]
fn public_order_statistic_budgets_preserve_selected_oracles_and_recover() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    for (operation, target, axis_size, value, gradient) in [
        ("min", vec![], None, 1.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ("max", vec![], None, 8.0, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("median", vec![], None, 3.5, [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
        (
            "quantile:q:0.25",
            vec![],
            None,
            2.25,
            [0.0, 0.0, 0.0, 0.75, 0.0, 0.25],
        ),
        (
            "percentile:q:75",
            vec![],
            None,
            5.5,
            [0.75, 0.0, 0.25, 0.0, 0.0, 0.0],
        ),
        ("median:axis:0", vec![3], Some(2), 12.0, [0.5; 6]),
        (
            "max:axis:1",
            vec![2],
            Some(3),
            14.0,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "min:axis:-1",
            vec![2],
            Some(3),
            3.0,
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ),
        (
            "quantile:q:0",
            vec![],
            None,
            1.0,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "quantile:q:1",
            vec![],
            None,
            8.0,
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "percentile:q:0",
            vec![],
            None,
            1.0,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "percentile:q:100",
            vec![],
            None,
            8.0,
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "quantile:q:0.25:axis:1",
            vec![2],
            Some(3),
            5.0,
            [0.0, 0.5, 0.5, 0.5, 0.0, 0.5],
        ),
        (
            "percentile:axis:1:q:75",
            vec![2],
            Some(3),
            10.5,
            [0.5, 0.0, 0.5, 0.0, 0.5, 0.5],
        ),
    ] {
        let ir = order_statistic_ir(operation, &[2, 3], &target);
        let inputs = [6.0, 1.0, 4.0, 2.0, 8.0, 3.0];
        let output: usize = target.iter().product();
        let q = target.len();
        let word = std::mem::size_of::<usize>();
        let pair = std::mem::size_of::<(usize, f64)>();
        // Frozen role inventory: indexed source groups, optional outer headers,
        // numeric source/contribution/cotangent and separate validation/order/VJP
        // scratch; accumulation has source/contribution/reduced numeric copies.
        let group_size = axis_size.unwrap_or(6usize);
        let scratch = (group_size * 8usize.max(word)).max(2 * pair);
        let coordinate_words = if axis_size.is_some() { 4 + 3 * q } else { 2 };
        let group_headers = if axis_size.is_some() {
            output * std::mem::size_of::<Vec<(usize, f64)>>()
        } else {
            0
        };
        let kernel =
            (12 + output) * 8 + coordinate_words * word + 6 * pair + group_headers + scratch;
        let accumulation = (18 + output) * 8 + (10 + q) * word;
        let retained = 6 + output + 1;
        let expected = ReplayMemoryRequest {
            forward_bytes: retained * 8,
            adjoint_bytes: (retained + 6) * 8,
            intermediate_bytes: kernel.max(accumulation),
        };
        let total = expected.total_bytes().unwrap();
        for budget in [total - 1, total, total + 1] {
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let result = with_replay_memory_admission(
                move |request| {
                    recorded.set(recorded.get() + 1);
                    assert_eq!(request, expected);
                    if request.total_bytes()? > budget {
                        Err("order-statistic budget refused".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs),
            );
            assert_eq!(calls.get(), 1);
            match result {
                Err(reason) => {
                    assert!(budget < total);
                    assert!(reason.contains("order-statistic budget refused"));
                }
                Ok(result) => {
                    assert_eq!(result.supported, budget >= total);
                    if result.supported {
                        assert_eq!(result.value, Some(value));
                        assert_eq!(result.gradient, gradient);
                    } else {
                        assert!(result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("order-statistic budget refused")));
                    }
                }
            }
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(value));
            assert_eq!(retry.gradient, gradient);
        }
    }
}

#[test]
fn public_order_statistic_invalid_q_axis_shape_and_arity_refuse_before_admission() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (operation, target, arity) in [
        ("median:q:0.5", vec![], 1),
        ("quantile", vec![], 1),
        ("quantile:q:NaN", vec![], 1),
        ("quantile:q:-0.1", vec![], 1),
        ("quantile:q:1.1", vec![], 1),
        ("percentile:q:-1", vec![], 1),
        ("percentile:q:101", vec![], 1),
        ("quantile:q:0.5:q:0.7", vec![], 1),
        ("median:axis:1:axis:0", vec![2], 1),
        ("max:axis:2", vec![2], 1),
        ("min:axis:-3", vec![2], 1),
        ("max:axis", vec![], 1),
        ("median:unknown:0", vec![], 1),
        ("median", vec![2], 1),
        ("max:axis:1", vec![3], 1),
        ("quantile:q:0.5:axis:1", vec![2, 1], 1),
        ("max", vec![], 0),
        ("min", vec![], 2),
    ] {
        let mut ir: serde_json::Value =
            serde_json::from_str(&order_statistic_ir(operation, &[2, 3], &target)).unwrap();
        ir["effects"][1]["inputs"] =
            serde_json::json!((0..arity).map(|_| "%0").collect::<Vec<_>>());
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let result = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || {
                interpret_program_ad_effect_ir_value_and_gradient(
                    &ir.to_string(),
                    &[6.0, 1.0, 4.0, 2.0, 8.0, 3.0],
                )
            },
        )
        .unwrap();
        assert!(!result.supported, "{operation}");
        assert_eq!(calls.get(), 0);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &order_statistic_ir("median", &[2, 3], &[]),
            &[6.0, 1.0, 4.0, 2.0, 8.0, 3.0],
        )
        .unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(3.5));
        assert_eq!(retry.gradient, [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]);
    }
}

#[test]
fn public_order_statistic_singleton_budget_and_ties_preserve_boundaries() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    for operation in ["max", "min", "median", "quantile:q:0.3", "percentile:q:90"] {
        let ir = order_statistic_ir(operation, &[], &[]);
        let expected = ReplayMemoryRequest {
            forward_bytes: 24,
            adjoint_bytes: 32,
            intermediate_bytes: 24 + 3 * std::mem::size_of::<(usize, f64)>(),
        };
        let total = expected.total_bytes().unwrap();
        for budget in [total - 1, total, total + 1] {
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let result = with_replay_memory_admission(
                move |request| {
                    recorded.set(recorded.get() + 1);
                    assert_eq!(request, expected);
                    if request.total_bytes()? > budget {
                        Err("singleton order-statistic budget refused".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0]),
            );
            assert_eq!(calls.get(), 1);
            match result {
                Err(reason) => {
                    assert!(budget < total);
                    assert!(reason.contains("singleton order-statistic budget refused"));
                }
                Ok(result) => {
                    assert_eq!(result.supported, budget >= total);
                    if result.supported {
                        assert_eq!(result.value, Some(2.0));
                        assert_eq!(result.gradient, [1.0]);
                    } else {
                        assert!(result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("singleton order-statistic budget refused")));
                    }
                }
            }
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0]).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(2.0));
            assert_eq!(retry.gradient, [1.0]);
        }
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let unsupported = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_forward(&ir, &[2.0]),
        )
        .unwrap();
        assert!(!unsupported.supported);
        assert_eq!(calls.get(), 0);
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let tied = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || {
                interpret_program_ad_effect_ir_value_and_gradient(
                    &order_statistic_ir(operation, &[2], &[]),
                    &[1.0, 1.0],
                )
            },
        )
        .unwrap();
        assert!(!tied.supported);
        assert_eq!(calls.get(), 1);
        assert!(tied
            .blocked_reasons
            .iter()
            .any(|r| r.contains("strictly ordered")));
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(2.0));
        assert_eq!(retry.gradient, [1.0]);
    }
}

#[test]
fn public_order_statistic_wide_broadcast_refuses_before_materialization() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let per_entry = 2 * std::mem::size_of::<f64>()
        + std::mem::size_of::<(usize, f64)>()
        + std::mem::size_of::<f64>().max(std::mem::size_of::<usize>());
    let huge = isize::MAX as usize / per_entry + 1;
    for operation in [
        "max:axis:1",
        "min:axis:1",
        "median:axis:1",
        "quantile:axis:1:q:0.5",
        "percentile:q:25:axis:1",
    ] {
        let ir=serde_json::json!({
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
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let result = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0]),
        )
        .unwrap();
        assert!(!result.supported);
        assert_eq!(calls.get(), 0);
        assert!(
            result
                .blocked_reasons
                .iter()
                .any(|r| r.contains("reduction workspace exceeds native addressable memory")),
            "{:?}",
            result.blocked_reasons
        );
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &order_statistic_ir("median", &[2, 3], &[]),
            &[6.0, 1.0, 4.0, 2.0, 8.0, 3.0],
        )
        .unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(3.5));
        assert_eq!(retry.gradient, [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]);
    }
}
