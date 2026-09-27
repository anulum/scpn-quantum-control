// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Public matrix-power storage and recovery contracts

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{replay_checkpoint, with_replay_checkpoint};
use serde_json::json;
use std::cell::Cell;
use std::rc::Rc;


#[test]
fn public_matrix_power_replay_interrupts_owned_workspaces_and_recovers() {
    for exponent in [0, 3, -3] {
        let ir = matrix_power_weighted_objective_ir(exponent, [1.0; 4]);
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.0, 0.0, 4.0]),
        ).unwrap();
        assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("matrix-power owned cancellation".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.0, 0.0, 4.0]),
            );
            match refused {
                Err(reason) => assert!(reason.contains("matrix-power owned cancellation"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(result.blocked_reasons.iter().any(|reason| reason.contains("matrix-power owned cancellation")), "{:?}", result.blocked_reasons);
                }
            }
            replay_checkpoint().unwrap();
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.0, 0.0, 4.0]).unwrap();
            assert!(retry.supported, "{:?}", retry.blocked_reasons);
            assert_eq!(retry.value, baseline.value);
            assert_eq!(retry.gradient, baseline.gradient);
        }
    }
}

#[test]
fn public_replay_preserves_diagonal_power_and_inverse_differentials() {
    for (power, expected_value, expected_gradient) in [
        (0, 2.0, [0.0, 0.0, 0.0, 0.0]),
        (1, 6.0, [1.0, 1.0, 1.0, 1.0]),
        (2, 20.0, [4.0, 6.0, 6.0, 8.0]),
        (-1, 0.75, [-0.25, -0.125, -0.125, -0.0625]),
        (-2, 0.3125, [-0.25, -0.09375, -0.09375, -0.03125]),
    ] {
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &matrix_power_weighted_objective_ir(power, [1.0; 4]),
            &[2.0, 0.0, 0.0, 4.0],
        )
        .unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(result.value, Some(expected_value));
        assert_eq!(result.gradient, expected_gradient);
    }
}

#[test]
fn public_replay_rejects_invalid_matrix_metadata_and_recovers() {
    let invalid = [
        ("linalg:matrix_power:2x2:power", "operation metadata"),
        ("linalg:matrix_power:2x2:power:2:0:0:extra", "operation metadata"),
        ("linalg:matrix_power:2x2x2:power:2:0:0", "shape metadata"),
        ("linalg:matrix_power:2:power:2:0:0", "shape metadata"),
        ("linalg:matrix_power:badx2:power:2:0:0", "row metadata"),
        ("linalg:matrix_power:2xbad:power:2:0:0", "column metadata"),
        ("linalg:matrix_power:0x0:power:2:0:0", "non-empty square"),
        ("linalg:matrix_power:1x2:power:2:0:0", "non-empty square"),
        ("linalg:matrix_power:3x3:power:2:0:0", "flattened matrix operands"),
        ("linalg:matrix_power:2x2:power:bad:0:0", "exponent metadata"),
        ("linalg:matrix_power:2x2:power:2:bad:0", "output-row metadata"),
        ("linalg:matrix_power:2x2:power:2:0:bad", "output-column metadata"),
        ("linalg:matrix_power:2x2:power:2:2:0", "outside matrix shape"),
        ("linalg:matrix_power:2x2:power:-9223372036854775808:0:0", "outside replay range"),
    ];
    for (operation, expected_reason) in invalid {
        assert_refusal_and_retry(operation, expected_reason);
    }
    let huge_shape = format!("linalg:matrix_power:{0}x{0}:power:2:0:0", usize::MAX);
    assert_refusal_and_retry(&huge_shape, "shape size overflows");
}

fn assert_refusal_and_retry(operation: &str, expected_reason: &str) {
    let mut ir: serde_json::Value =
        serde_json::from_str(&matrix_power_weighted_objective_ir(2, [1.0; 4])).unwrap();
    ir["effects"][4]["operation"] = json!(operation);
    let refused = interpret_program_ad_effect_ir_value_and_gradient(
        &ir.to_string(),
        &[2.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    assert!(!refused.supported, "{operation}");
    assert!(
        refused.blocked_reasons.iter().any(|reason| {
            reason.contains("matrix_power") && reason.contains(expected_reason)
        }),
        "{operation}: {:?}",
        refused.blocked_reasons
    );
    let retry = interpret_program_ad_effect_ir_value_and_gradient(
        &matrix_power_weighted_objective_ir(2, [1.0; 4]),
        &[2.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    assert!(retry.supported, "{:?}", retry.blocked_reasons);
    assert_eq!(retry.value, Some(20.0));
    assert_eq!(retry.gradient, [4.0, 6.0, 6.0, 8.0]);
}

fn matrix_power_weighted_objective_ir(exponent: i64, weights: [f64; 4]) -> String {
    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..15 {
        ssa_values.push(json!({
            "name": format!("%{index}"),
            "producer": index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": index,
        }));
    }
    for index in 0..4 {
        effects.push(json!({
            "index": index,
            "kind": "parameter",
            "target": format!("%{index}"),
            "inputs": [format!("p{index}")],
            "version": 0,
            "ordering": index,
            "operation": "parameter",
        }));
    }
    for (offset, (row, column)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
        let index = 4 + offset;
        effects.push(json!({
            "index": index,
            "kind": "primitive",
            "target": format!("%{index}"),
            "inputs": ["%0", "%1", "%2", "%3"],
            "version": 0,
            "ordering": index,
            "operation": format!("linalg:matrix_power:2x2:power:{exponent}:{row}:{column}"),
        }));
    }
    for (offset, weight) in weights.into_iter().enumerate() {
        let index = 8 + offset;
        effects.push(json!({
            "index": index,
            "kind": "pure",
            "target": format!("%{index}"),
            "inputs": [format!("%{}", 4 + offset), weight.to_string()],
            "version": 0,
            "ordering": index,
            "operation": "mul",
        }));
    }
    effects.push(json!({
        "index": 12,
        "kind": "pure",
        "target": "%12",
        "inputs": ["%8", "%9"],
        "version": 0,
        "ordering": 12,
        "operation": "add",
    }));
    effects.push(json!({
        "index": 13,
        "kind": "pure",
        "target": "%13",
        "inputs": ["%10", "%11"],
        "version": 0,
        "ordering": 13,
        "operation": "add",
    }));
    effects.push(json!({
        "index": 14,
        "kind": "pure",
        "target": "%14",
        "inputs": ["%12", "%13"],
        "version": 0,
        "ordering": 14,
        "operation": "add",
    }));
    json!({
        "format": "program_ad_effect_ir.v1",
        "ssa_values": ssa_values,
        "effects": effects,
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [0],
    })
    .to_string()
}


fn scalar_matrix_power_ir(exponent: i64) -> String {
    json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[],"dtype":"float64","effect":1}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":format!("linalg:matrix_power:1x1:power:{exponent}:0:0")}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string()
}

#[test]
fn public_matrix_power_workspace_and_prefix_budget_boundaries_recover() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{ReplayMemoryRequest, with_replay_memory_admission};
    let headers = 2 * std::mem::size_of::<Vec<f64>>();
    for (exponent, workspace, forward_workspace, value, gradient) in [
        (0, 24usize, 16usize, 1.0, 0.0),
        (2, 80 + headers, 24, 4.0, 4.0),
        (-2, 88 + headers, 32, 0.25, -0.25),
    ] {
        let source = scalar_matrix_power_ir(exponent);
        let expected = ReplayMemoryRequest { forward_bytes:16, adjoint_bytes:24, intermediate_bytes:workspace };
        let total = 40 + workspace;
        for budget in [total - 1, total, total + 1] {
            let result = with_replay_memory_admission(
                move |request| {
                    assert_eq!(request, expected);
                    if request.total_bytes()? > budget { Err("matrix-power workspace budget refused".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]),
            ).unwrap();
            assert_eq!(result.supported, budget >= total);
            if result.supported { assert_eq!(result.value, Some(value)); assert_eq!(result.gradient, [gradient]); }
            else { assert!(result.blocked_reasons.iter().any(|reason| reason.contains("matrix-power workspace budget refused"))); }
            let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(value));
            assert_eq!(retry.gradient, [gradient]);
        }
        let forward_total = 16 + forward_workspace;
        for budget in [forward_total - 1, forward_total, forward_total + 1] {
            let result = with_replay_memory_admission(
                move |request| {
                    assert_eq!(request, ReplayMemoryRequest { forward_bytes:16, adjoint_bytes:0, intermediate_bytes:forward_workspace });
                    if request.total_bytes()? > budget { Err("forward matrix-power budget refused".to_owned()) } else { Ok(()) }
                },
                || interpret_program_ad_effect_ir_forward(&source, &[2.0]),
            );
            if budget < forward_total { assert!(result.unwrap_err().contains("forward matrix-power budget refused")); }
            else { let result = result.unwrap(); assert!(result.supported); assert_eq!(result.value, Some(value)); }
        }
    }
}

#[test]
fn public_extreme_power_gradient_refuses_before_numeric_work_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for exponent in [i64::MAX, i64::MIN] {
        let callbacks = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&callbacks);
        let result = with_replay_memory_admission(
            move |_| { recorded.set(recorded.get() + 1); Ok(()) },
            || interpret_program_ad_effect_ir_value_and_gradient(&scalar_matrix_power_ir(exponent), &[2.0]),
        ).unwrap();
        assert!(!result.supported);
        assert_eq!(callbacks.get(), 0);
        assert!(result.blocked_reasons.iter().any(|reason| reason.contains("matrix_power")
            && (reason.contains("addressable memory") || reason.contains("outside replay range"))), "{:?}", result.blocked_reasons);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&scalar_matrix_power_ir(2), &[2.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, [4.0]);
    }
}
