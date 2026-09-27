// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Public shaped parameter addressability contracts

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};
use serde_json::json;

#[test]
fn public_replay_rejects_unaddressable_shapes_before_parameter_materialisation() {
    for (shape, reason) in [
        (
            vec![usize::MAX],
            "shaped value bytes exceed native addressability",
        ),
        (vec![usize::MAX, 2], "shaped value size overflowed"),
        (vec![0], "non-zero dimensions"),
    ] {
        let result =
            interpret_program_ad_effect_ir_value_and_gradient(&two_parameter_ir(&shape, &[1]), &[])
                .unwrap();
        assert!(!result.supported);
        assert!(result
            .blocked_reasons
            .iter()
            .any(|entry| entry.contains(reason)));
        assert_small_replay_recovers();
    }
}

#[test]
fn public_replay_checks_aggregate_parameter_bytes_before_copying_inputs() {
    let individually_addressable = (isize::MAX as usize / std::mem::size_of::<f64>()) / 2 + 1;
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &two_parameter_ir(&[individually_addressable], &[individually_addressable]),
        &[],
    )
    .unwrap();
    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|entry| { entry.contains("parameter bytes exceed native addressability") }));
    assert_small_replay_recovers();
}

#[test]
fn public_replay_preserves_scalar_shapes_and_parameter_mismatch_refusal() {
    let scalar =
        interpret_program_ad_effect_ir_value_and_gradient(&two_parameter_ir(&[], &[]), &[2.0, 3.0])
            .unwrap();
    assert!(scalar.supported, "{:?}", scalar.blocked_reasons);
    assert_eq!(scalar.value, Some(6.0));
    assert_eq!(scalar.gradient, [3.0, 2.0]);
    let mismatch = interpret_program_ad_effect_ir_value_and_gradient(
        &two_parameter_ir(&[2], &[2]),
        &[1.0, 2.0, 3.0],
    )
    .unwrap();
    assert!(!mismatch.supported);
    assert!(mismatch
        .blocked_reasons
        .iter()
        .any(|entry| { entry.contains("parameter count 4 does not match input count 3") }));
    assert_small_replay_recovers();
}

#[test]
fn public_replay_borrows_last_duplicate_shape_without_changing_parameter_order() {
    let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[2], &[2])).unwrap();
    let duplicate = json!({
        "name": "%0", "producer": 0, "version": 0,
        "shape": [], "dtype": "float64", "effect": 0,
    });
    ir["ssa_values"].as_array_mut().unwrap().push(duplicate);
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(), &[2.0, 3.0, 4.0])
            .unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(14.0));
    assert_eq!(result.gradient, [7.0, 2.0, 2.0]);
    assert_eq!(result.parameter_targets, ["%0", "%1[0]", "%1[1]"]);
}

#[test]
fn public_replay_missing_shape_refuses_and_keeps_next_parameter_copy_usable() {
    let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[2], &[2])).unwrap();
    ir["ssa_values"].as_array_mut().unwrap().remove(0);
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| { reason.contains("target %0 is missing SSA shape metadata") }));
    assert_small_replay_recovers();
}

#[test]
fn public_replay_preserves_unicode_parameter_labels_and_sources() {
    let ir = two_parameter_ir(&[2], &[2]).replace("%0", "%theta_θ");
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(21.0));
    assert_eq!(result.gradient, [7.0, 7.0, 3.0, 3.0]);
    assert_eq!(
        result.parameter_targets,
        ["%theta_θ[0]", "%theta_θ[1]", "%1[0]", "%1[1]"]
    );
}

#[test]
fn public_replay_preserves_owned_operand_and_broadcast_cotangent_copies() {
    let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[1], &[2])).unwrap();
    ir["ssa_values"].as_array_mut().unwrap().push(json!({
        "name": "%5", "producer": 5, "version": 0,
        "shape": [2], "dtype": "float64", "effect": 5,
    }));
    ir["effects"][2]["ordering"] = json!(3);
    ir["effects"][2]["inputs"] = json!(["%5"]);
    ir["effects"][3]["ordering"] = json!(4);
    ir["effects"][4]["ordering"] = json!(5);
    ir["effects"].as_array_mut().unwrap().push(json!({
        "index": 5, "kind": "pure", "target": "%5", "inputs": ["%0"],
        "version": 0, "ordering": 2, "operation": "broadcast_to",
    }));
    let inputs = [2.0, 3.0, 4.0];
    for _ in 0..2 {
        let result =
            interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(), &inputs).unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(result.value, Some(28.0));
        assert_eq!(result.gradient, [14.0, 4.0, 4.0]);
        assert_eq!(result.parameter_targets, ["%0[0]", "%1[0]", "%1[1]"]);
    }
    assert_eq!(inputs, [2.0, 3.0, 4.0]);
    assert_small_replay_recovers();
}

#[test]
fn public_replay_preserves_stack_concatenate_transpose_and_reverse_buffers() {
    for (operation, shape) in [
        ("stack:axis:0", vec![2, 2]),
        ("concatenate:axis:0", vec![4]),
    ] {
        let mut ir: serde_json::Value =
            serde_json::from_str(&two_parameter_ir(&[2], &[2])).unwrap();
        for (index, name, shape) in [(5, "%5", &shape), (6, "%6", &shape)] {
            ir["ssa_values"].as_array_mut().unwrap().push(json!({
                "name": name, "producer": index, "version": 0,
                "shape": shape, "dtype": "float64", "effect": index,
            }));
        }
        ir["effects"][2]["inputs"] = json!(["%6"]);
        ir["effects"][2]["ordering"] = json!(4);
        ir["effects"][3]["ordering"] = json!(5);
        ir["effects"][4]["ordering"] = json!(6);
        ir["effects"].as_array_mut().unwrap().push(json!({
            "index": 5, "kind": "pure", "target": "%5", "inputs": ["%0", "%1"],
            "version": 0, "ordering": 2, "operation": operation,
        }));
        ir["effects"].as_array_mut().unwrap().push(json!({
            "index": 6, "kind": "pure", "target": "%6", "inputs": ["%5"],
            "version": 0, "ordering": 3, "operation": "transpose",
        }));
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &ir.to_string(),
            &[1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert!(
            result.supported,
            "{operation}: {:?}",
            result.blocked_reasons
        );
        assert_eq!(result.value, Some(70.0));
        assert_eq!(result.gradient, [7.0, 7.0, 17.0, 17.0]);
        assert_small_replay_recovers();
    }
}

#[test]
fn public_replay_preserves_signed_and_quotient_elementwise_reverse_outputs() {
    for (operation, value, gradient) in [
        ("sub", -4.0, [1.0, 1.0, -1.0, -1.0]),
        (
            "div",
            3.0 / 7.0,
            [1.0 / 7.0, 1.0 / 7.0, -3.0 / 49.0, -3.0 / 49.0],
        ),
    ] {
        let mut ir: serde_json::Value =
            serde_json::from_str(&two_parameter_ir(&[2], &[2])).unwrap();
        ir["effects"][4]["operation"] = json!(operation);
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &ir.to_string(),
            &[1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert!(
            result.supported,
            "{operation}: {:?}",
            result.blocked_reasons
        );
        assert_eq!(result.value, Some(value));
        for (actual, expected) in result.gradient.iter().zip(gradient) {
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        assert_small_replay_recovers();
    }
}

#[test]
fn public_replay_preserves_rectangular_axis_reduction_coordinates_and_gradients() {
    for (operation, shape, value, derivative, right_derivative) in [
        ("sum:axis:0", vec![3], 42.0, 2.0, 21.0),
        ("sum:axis:-1", vec![2], 42.0, 2.0, 21.0),
        ("mean:axis:0", vec![3], 21.0, 1.0, 10.5),
        ("mean:axis:-1", vec![2], 14.0, 2.0 / 3.0, 7.0),
    ] {
        let mut ir: serde_json::Value =
            serde_json::from_str(&two_parameter_ir(&[2, 3], &[])).unwrap();
        ir["ssa_values"][2]["shape"] = json!(shape);
        ir["effects"][2]["operation"] = json!(operation);
        ir["effects"][4]["inputs"] = json!(["%5", "%3"]);
        ir["effects"][4]["ordering"] = json!(5);
        ir["ssa_values"].as_array_mut().unwrap().push(json!({
            "name": "%5", "producer": 5, "version": 0,
            "shape": [], "dtype": "float64", "effect": 5,
        }));
        ir["effects"].as_array_mut().unwrap().push(json!({
            "index": 5, "kind": "primitive", "target": "%5", "inputs": ["%2"],
            "version": 0, "ordering": 4, "operation": "sum",
        }));
        let result = interpret_program_ad_effect_ir_value_and_gradient(
            &ir.to_string(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0],
        )
        .unwrap();
        assert!(
            result.supported,
            "{operation}: {:?}",
            result.blocked_reasons
        );
        assert!((result.value.unwrap() - value).abs() <= 1.0e-12);
        assert_eq!(result.gradient.len(), 7);
        for actual in &result.gradient[..6] {
            assert!((actual - derivative).abs() <= 1.0e-12);
        }
        assert!((result.gradient[6] - right_derivative).abs() <= 1.0e-12);
        assert_small_replay_recovers();
    }
}

#[test]
fn public_replay_preserves_literal_and_unused_parameter_scalar_storage() {
    let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[2], &[2])).unwrap();
    ir["effects"][4]["inputs"] = json!(["%2", "2.0"]);
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(6.0));
    assert_eq!(result.gradient, [2.0, 2.0, 0.0, 0.0]);
    assert_eq!(
        result.parameter_targets,
        ["%0[0]", "%0[1]", "%1[0]", "%1[1]"]
    );
    assert_small_replay_recovers();
}

#[test]
fn public_replay_preserves_unary_scalar_output_and_reverse_seed() {
    let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[], &[])).unwrap();
    ir["effects"][4]["operation"] = json!("sqrt");
    ir["effects"][4]["inputs"] = json!(["%2"]);
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&ir.to_string(), &[9.0, 4.0]).unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(3.0));
    assert!((result.gradient[0] - 1.0 / 6.0).abs() <= 1.0e-12);
    assert_eq!(result.gradient[1], 0.0);
    assert_small_replay_recovers();
}

#[test]
fn public_replay_preserves_stable_ordering_for_ties_and_shuffled_rows() {
    for equal_ordering in [false, true] {
        let mut ir: serde_json::Value = serde_json::from_str(&two_parameter_ir(&[], &[])).unwrap();
        let effects = ir["effects"].as_array_mut().unwrap();
        // Identity products keep this ordering fixture within both scalar interpreters.
        for (index, source) in [(2, "%0"), (3, "%1")] {
            effects[index]["kind"] = json!("pure");
            effects[index]["operation"] = json!("mul");
            effects[index]["inputs"] = json!([source, "1.0"]);
        }
        if equal_ordering {
            for effect in effects.iter_mut() {
                effect["ordering"] = json!(0);
            }
        } else {
            effects.swap(0, 4);
            effects.swap(1, 3);
        }
        let serialization = ir.to_string();
        let forward = interpret_program_ad_effect_ir_forward(&serialization, &[2.0, 3.0]).unwrap();
        assert!(forward.supported, "{:?}", forward.blocked_reasons);
        assert_eq!(forward.value, Some(6.0));
        let reverse =
            interpret_program_ad_effect_ir_value_and_gradient(&serialization, &[2.0, 3.0]).unwrap();
        assert!(reverse.supported, "{:?}", reverse.blocked_reasons);
        assert_eq!(reverse.value, Some(6.0));
        assert_eq!(reverse.gradient, [3.0, 2.0]);
        assert_eq!(reverse.parameter_targets, ["%0", "%1"]);
        assert_small_replay_recovers();
    }
}

fn assert_small_replay_recovers() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &two_parameter_ir(&[2], &[2]),
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(21.0));
    assert_eq!(result.gradient, [7.0, 7.0, 3.0, 3.0]);
}

fn two_parameter_ir(left_shape: &[usize], right_shape: &[usize]) -> String {
    let shapes = [left_shape, right_shape, &[], &[], &[]];
    let ssa_values: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            json!({
                "name": format!("%{index}"), "producer": index, "version": 0,
                "shape": shape, "dtype": "float64", "effect": index,
            })
        })
        .collect();
    let rows = [
        ("parameter", "parameter", vec!["left"]),
        ("parameter", "parameter", vec!["right"]),
        ("primitive", "sum", vec!["%0"]),
        ("primitive", "sum", vec!["%1"]),
        ("pure", "mul", vec!["%2", "%3"]),
    ];
    let effects: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(index, (kind, operation, inputs))| {
            json!({
                "index": index, "kind": kind, "target": format!("%{index}"),
                "inputs": inputs, "version": 0, "ordering": index, "operation": operation,
            })
        })
        .collect();
    json!({
        "format": "program_ad_effect_ir.v1", "ssa_values": ssa_values,
        "effects": effects, "alias_edges": [], "control_regions": [],
        "phi_nodes": [], "bytecode_offsets": [0],
    })
    .to_string()
}
