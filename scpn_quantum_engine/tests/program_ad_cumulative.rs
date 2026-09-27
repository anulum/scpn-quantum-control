// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use serde_json::json;

const CUMULATIVE_WEIGHTED_OBJECTIVE_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10},
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11},
    {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12},
    {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13},
    {"name": "%14", "producer": 14, "version": 0, "shape": [], "dtype": "float64", "effect": 14},
    {"name": "%15", "producer": 15, "version": 0, "shape": [], "dtype": "float64", "effect": 15},
    {"name": "%16", "producer": 16, "version": 0, "shape": [], "dtype": "float64", "effect": 16},
    {"name": "%17", "producer": 17, "version": 0, "shape": [], "dtype": "float64", "effect": 17},
    {"name": "%18", "producer": 18, "version": 0, "shape": [], "dtype": "float64", "effect": 18},
    {"name": "%19", "producer": 19, "version": 0, "shape": [], "dtype": "float64", "effect": 19},
    {"name": "%20", "producer": 20, "version": 0, "shape": [], "dtype": "float64", "effect": 20}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x0"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["x1"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["x2"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["x3"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["x4"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["x5"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["w0"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["w1"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["w2"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["w3"], "version": 0, "ordering": 9, "operation": "parameter"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 10, "operation": "cumsum:shape:2x3:axis:1:out:4"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 11, "operation": "cumprod:shape:2x3:axis:1:out:5"},
    {"index": 12, "kind": "primitive", "target": "%12", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 12, "operation": "diff:shape:2x3:n:2:axis:1:out:1"},
    {"index": 13, "kind": "primitive", "target": "%13", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 13, "operation": "cumsum:shape:2x3:axis:flat:out:3"},
    {"index": 14, "kind": "pure", "target": "%14", "inputs": ["%10", "%6"], "version": 0, "ordering": 14, "operation": "mul"},
    {"index": 15, "kind": "pure", "target": "%15", "inputs": ["%11", "%7"], "version": 0, "ordering": 15, "operation": "mul"},
    {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%12", "%8"], "version": 0, "ordering": 16, "operation": "mul"},
    {"index": 17, "kind": "pure", "target": "%17", "inputs": ["%13", "%9"], "version": 0, "ordering": 17, "operation": "mul"},
    {"index": 18, "kind": "pure", "target": "%18", "inputs": ["%14", "%15"], "version": 0, "ordering": 18, "operation": "add"},
    {"index": 19, "kind": "pure", "target": "%19", "inputs": ["%18", "%16"], "version": 0, "ordering": 19, "operation": "add"},
    {"index": 20, "kind": "pure", "target": "%20", "inputs": ["%19", "%17"], "version": 0, "ordering": 20, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

#[test]
fn value_and_gradient_replays_static_cumulative_nodes() {
    let inputs = [1.25, -0.75, 2.0, 0.5, 1.5, -1.25, 0.2, -0.4, 0.6, -0.8];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        CUMULATIVE_WEIGHTED_OBJECTIVE_IR,
        &inputs,
    )
    .expect("static cumulative replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("static_cumulative_primitives"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        -3.875,
    );
    assert_eq!(result.gradient.len(), 10);
    for (actual, expected) in result
        .gradient
        .iter()
        .zip([-0.8, -0.8, -0.8, 0.75, -0.75, 0.3, 2.0, -0.9375, -3.75, 3.0])
    {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_rejects_static_cumulative_diff_order_outside_axis() {
    let malformed = CUMULATIVE_WEIGHTED_OBJECTIVE_IR.replace(
        "diff:shape:2x3:n:2:axis:1:out:1",
        "diff:shape:2x3:n:4:axis:1:out:1",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &malformed,
        &[1.25, -0.75, 2.0, 0.5, 1.5, -1.25, 0.2, -0.4, 0.6, -0.8],
    )
    .expect("malformed static cumulative replay should serialize");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("exceeds axis length")));
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}",
    );
}

fn one_cumulative_objective_ir(operation: &str, count: usize) -> String {
    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..count {
        let target = format!("%{index}");
        ssa_values.push(json!({
            "name": target, "producer": index, "version": 0,
            "shape": [], "dtype": "float64", "effect": index
        }));
        effects.push(json!({
            "index": index, "kind": "parameter", "target": target,
            "inputs": [format!("x{index}")], "version": 0,
            "ordering": index, "operation": "parameter"
        }));
    }
    let target = format!("%{count}");
    ssa_values.push(json!({
        "name": target, "producer": count, "version": 0,
        "shape": [], "dtype": "float64", "effect": count
    }));
    effects.push(json!({
        "index": count, "kind": "primitive", "target": target,
        "inputs": (0..count).map(|index| format!("%{index}")).collect::<Vec<_>>(),
        "version": 0, "ordering": count, "operation": operation
    }));
    json!({
        "format": "program_ad_effect_ir.v1", "ssa_values": ssa_values,
        "effects": effects, "alias_edges": [], "control_regions": [],
        "phi_nodes": [], "bytecode_offsets": []
    })
    .to_string()
}

#[test]
fn public_cumulative_replay_preserves_zero_safe_independent_differentials() {
    for (operation, value, gradient) in [
        ("cumsum:shape:3:axis:flat:out:2", 5.0, [1.0, 1.0, 1.0]),
        ("cumprod:shape:3:axis:flat:out:2", 0.0, [0.0, 6.0, 0.0]),
        ("diff:shape:3:n:1:axis:0:out:1", 3.0, [0.0, -1.0, 1.0]),
    ] {
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &one_cumulative_objective_ir(operation, 3),
            &[2.0, 0.0, 3.0],
        )
        .unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(result.value, Some(value));
        assert_eq!(result.gradient, gradient);
    }
}

#[test]
fn public_cumulative_replay_checked_binomial_preserves_representable_coefficient() {
    let mut inputs = vec![0.0; 33];
    inputs[16] = 1.0;
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &one_cumulative_objective_ir("diff:shape:33:n:32:axis:0:out:0", 33),
        &inputs,
    )
    .unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(601080390.0));
    assert_eq!(result.gradient.len(), 33);
    assert_eq!(result.gradient[16], 601080390.0);
    assert_eq!(result.gradient[0], 1.0);
    assert_eq!(result.gradient[32], 1.0);
}

#[test]
fn public_cumulative_replay_refuses_unrepresentable_metadata_and_recovers() {
    for (operation, count, expected_reason) in [
        (
            format!("cumsum:shape:{}x2:axis:flat:out:0", usize::MAX),
            1,
            "size overflowed",
        ),
        (
            "diff:shape:101:n:100:axis:0:out:0".to_owned(),
            101,
            "binomial coefficient overflowed",
        ),
    ] {
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &one_cumulative_objective_ir(&operation, count),
            &vec![1.0; count],
        )
        .unwrap();
        assert!(!result.supported);
        assert!(
            result
                .blocked_reasons
                .iter()
                .any(|reason| reason.contains(expected_reason)),
            "{:?}",
            result.blocked_reasons
        );
    }
    let valid = interpret_program_ad_effect_ir_value_and_gradient(
        &one_cumulative_objective_ir("cumsum:shape:3:axis:flat:out:2", 3),
        &[2.0, 0.0, 3.0],
    )
    .unwrap();
    assert!(valid.supported, "{:?}", valid.blocked_reasons);
    assert_eq!(valid.value, Some(5.0));
    assert_eq!(valid.gradient, [1.0, 1.0, 1.0]);
}

#[test]
fn public_cumulative_workspace_budgets_cover_coordinates_and_terms() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    use std::cell::Cell;
    use std::rc::Rc;
    let word = std::mem::size_of::<usize>();
    let term = std::mem::size_of::<(usize, f64)>();
    for (operation, inputs, value, gradient, index_bytes) in [
        (
            "cumsum:shape:2x3:axis:1:out:4",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            9.0,
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            (6 + 2) * word,
        ),
        (
            "cumprod:shape:2x3:axis:1:out:5",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            120.0,
            [0.0, 0.0, 0.0, 30.0, 24.0, 20.0],
            (6 + 3) * word,
        ),
        (
            "cumsum:shape:2x3:axis:flat:out:3",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            10.0,
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            (2 + 4) * word,
        ),
        (
            "diff:shape:2x3:n:2:axis:1:out:1",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            0.0,
            [0.0, 0.0, 0.0, 1.0, -2.0, 1.0],
            8 * word + 3 * term,
        ),
        (
            "diff:out:4:axis:1:n:0:shape:2x3",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            5.0,
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            8 * word + term,
        ),
        (
            "cumprod:shape:2x3:axis:1:out:5",
            [1.0, 2.0, 3.0, 0.0, 5.0, 6.0],
            0.0,
            [0.0, 0.0, 0.0, 30.0, 0.0, 0.0],
            (6 + 3) * word,
        ),
        (
            "cumsum:shape:2x3:axis:0:out:4",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            7.0,
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            (6 + 2) * word,
        ),
    ] {
        let ir = one_cumulative_objective_ir(operation, 6);
        for gradient_surface in [false, true] {
            let expected = ReplayMemoryRequest {
                forward_bytes: 56,
                adjoint_bytes: if gradient_surface { 104 } else { 0 },
                intermediate_bytes: if gradient_surface { 96 } else { 48 } + index_bytes,
            };
            let total = expected.total_bytes().unwrap();
            let replay = || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
                        .map(|r| (r.supported, r.value, r.gradient, r.blocked_reasons))
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &inputs)
                        .map(|r| (r.supported, r.value, Vec::new(), r.blocked_reasons))
                }
            };
            for budget in [total - 1, total, total + 1] {
                let calls = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&calls);
                let result = with_replay_memory_admission(
                    move |request| {
                        recorded.set(recorded.get() + 1);
                        assert_eq!(request, expected);
                        if request.total_bytes()? > budget {
                            Err("cumulative workspace budget refused".to_owned())
                        } else {
                            Ok(())
                        }
                    },
                    replay,
                );
                assert_eq!(calls.get(), 1);
                match result {
                    Err(reason) => {
                        assert!(budget < total);
                        assert!(reason.contains("cumulative workspace budget refused"));
                    }
                    Ok(result) => {
                        assert_eq!(result.0, budget >= total);
                        if result.0 {
                            assert_close(result.1.unwrap(), value);
                            if gradient_surface {
                                assert_eq!(result.2, gradient);
                            }
                        } else {
                            assert!(result
                                .3
                                .iter()
                                .any(|r| r.contains("cumulative workspace budget refused")));
                        }
                    }
                }
                let retry = replay().unwrap();
                assert!(retry.0, "{:?}", retry.3);
                assert_close(retry.1.unwrap(), value);
                if gradient_surface {
                    assert_eq!(retry.2, gradient);
                }
            }
        }
    }
}

#[test]
fn public_cumulative_bad_metadata_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    use std::cell::Cell;
    use std::rc::Rc;
    for operation in [
        "cumsum:shape:2x3:axis:1:out:6".to_owned(),
        "cumsum:shape:2x3:axis:2:out:0".to_owned(),
        "cumsum:shape:2x3:axis:1:out:0:shape:2x3".to_owned(),
        "cumsum:shape:2x0:axis:1:out:0".to_owned(),
        "cumprod:shape:2x4:axis:1:out:0".to_owned(),
        "cumprod:shape:2x3:n:1:axis:1:out:0".to_owned(),
        "diff:shape:2x3:n:4:axis:1:out:0".to_owned(),
        "diff:shape:2x3:n:3:axis:1:out:0".to_owned(),
        "diff:shape:2x3:n:1:axis:flat:out:0".to_owned(),
        "diff:shape:2x3:axis:1:out:0".to_owned(),
        "diff:shape:2x3:n:1:axis:1:out".to_owned(),
        format!("cumsum:shape:{}x2:axis:flat:out:0", usize::MAX),
        format!("cumsum:shape:{}:axis:flat:out:0", isize::MAX),
    ] {
        let ir = one_cumulative_objective_ir(&operation, 6);
        for gradient_surface in [false, true] {
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let supported = with_replay_memory_admission(
                move |_| {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || {
                    if gradient_surface {
                        interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0; 6])
                            .map(|r| r.supported)
                    } else {
                        interpret_program_ad_effect_ir_forward(&ir, &[1.0; 6]).map(|r| r.supported)
                    }
                },
            )
            .unwrap();
            assert!(!supported, "{operation}");
            assert_eq!(calls.get(), 0);
            let retry = interpret_program_ad_effect_ir_value_and_gradient(
                &one_cumulative_objective_ir("cumsum:shape:3:axis:flat:out:2", 3),
                &[2.0, 0.0, 3.0],
            )
            .unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(5.0));
            assert_eq!(retry.gradient, [1.0, 1.0, 1.0]);
        }
    }
}
