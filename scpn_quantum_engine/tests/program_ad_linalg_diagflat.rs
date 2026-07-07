// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD diagflat linalg replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

fn weighted_diagflat_ir(
    shape_label: &str,
    offset: &str,
    construct_indices: &[String],
    weights: &[f64],
) -> String {
    let input_count = construct_indices.len();
    assert_eq!(weights.len(), input_count);

    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..input_count {
        ssa_values.push(format!(
            r#"{{"name": "%{index}", "producer": {index}, "version": 0, "shape": [], "dtype": "float64", "effect": {index}}}"#
        ));
        effects.push(format!(
            r#"{{"index": {index}, "kind": "parameter", "target": "%{index}", "inputs": [], "version": 0, "ordering": {index}, "operation": "parameter"}}"#
        ));
    }

    let mut next_index = input_count;
    let mut output_targets = Vec::new();
    for (source, construct) in construct_indices.iter().enumerate() {
        ssa_values.push(format!(
            r#"{{"name": "%{next_index}", "producer": {next_index}, "version": 0, "shape": [], "dtype": "float64", "effect": {next_index}}}"#
        ));
        effects.push(format!(
            r#"{{"index": {next_index}, "kind": "op", "target": "%{next_index}", "inputs": ["%{source}"], "version": 0, "ordering": {next_index}, "operation": "linalg:diagflat:{shape_label}:offset:{offset}:construct:{construct}"}}"#
        ));
        output_targets.push(next_index);
        next_index += 1;
    }

    let mut accumulated_target: Option<usize> = None;
    for (offset_index, weight) in weights.iter().enumerate() {
        let output_target = output_targets[offset_index];
        let mul_index = next_index;
        ssa_values.push(format!(
            r#"{{"name": "%{mul_index}", "producer": {mul_index}, "version": 0, "shape": [], "dtype": "float64", "effect": {mul_index}}}"#
        ));
        effects.push(format!(
            r#"{{"index": {mul_index}, "kind": "op", "target": "%{mul_index}", "inputs": ["%{output_target}", "{weight}"], "version": 0, "ordering": {mul_index}, "operation": "mul"}}"#
        ));
        next_index += 1;
        accumulated_target = match accumulated_target {
            None => Some(mul_index),
            Some(left_target) => {
                let add_index = next_index;
                ssa_values.push(format!(
                    r#"{{"name": "%{add_index}", "producer": {add_index}, "version": 0, "shape": [], "dtype": "float64", "effect": {add_index}}}"#
                ));
                effects.push(format!(
                    r#"{{"index": {add_index}, "kind": "op", "target": "%{add_index}", "inputs": ["%{left_target}", "%{mul_index}"], "version": 0, "ordering": {add_index}, "operation": "add"}}"#
                ));
                next_index += 1;
                Some(add_index)
            }
        };
    }

    format!(
        r#"{{
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [{}],
        "effects": [{}],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": []
    }}"#,
        ssa_values.join(",\n            "),
        effects.join(",\n            ")
    )
}

fn ascending_indices(count: usize) -> Vec<String> {
    (0..count).map(|index| index.to_string()).collect()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} differs from expected {expected}"
    );
}

#[test]
fn rust_program_ad_replays_vector_diagflat_value_and_gradient() {
    let inputs = vec![1.5, -2.0, 0.75];
    let weights = vec![0.4, -0.2, 0.3];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diagflat_ir("3", "0", &ascending_indices(3), &weights),
        &inputs,
    )
    .expect("vector diagflat replay should be supported");

    let expected_value = (0.4 * 1.5 + -0.2 * -2.0) + 0.3 * 0.75;

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        expected_value,
        0.0,
    );
    assert_eq!(result.gradient.len(), weights.len());
    for (actual, expected) in result.gradient.iter().zip(weights.iter()) {
        assert_close(*actual, *expected, 0.0);
    }
}

#[test]
fn rust_program_ad_replays_matrix_source_diagflat_with_negative_offset() {
    let inputs = vec![0.5, -1.25, 2.0, 3.5];
    let weights = vec![1.0, -0.5, 0.25, 2.0];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diagflat_ir("2x2", "-1", &ascending_indices(4), &weights),
        &inputs,
    )
    .expect("matrix-source diagflat replay should be supported");

    let expected_value = ((1.0 * 0.5 + -0.5 * -1.25) + 0.25 * 2.0) + 2.0 * 3.5;

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        expected_value,
        0.0,
    );
    assert_eq!(result.gradient.len(), weights.len());
    for (actual, expected) in result.gradient.iter().zip(weights.iter()) {
        assert_close(*actual, *expected, 0.0);
    }
}

#[test]
fn rust_program_ad_diagflat_fails_closed_on_malformed_metadata() {
    let ir = weighted_diagflat_ir("3", "0", &ascending_indices(3), &[0.4, -0.2, 0.3])
        .replace(":offset:", ":shift:");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0, 3.0])
        .expect("malformed diagflat metadata should gate, not crash");

    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diagflat operation metadata is malformed"),
        "{error}"
    );
}

#[test]
fn rust_program_ad_diagflat_fails_closed_on_out_of_range_source_index() {
    let indices = vec![
        "0".to_owned(),
        "1".to_owned(),
        "2".to_owned(),
        "4".to_owned(),
    ];
    let ir = weighted_diagflat_ir("2x2", "0", &indices, &[1.0, 1.0, 1.0, 1.0]);
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0, 3.0, 4.0])
        .expect("out-of-range diagflat index should gate, not crash");

    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diagflat source index is outside the flattened source shape"),
        "{error}"
    );
}

#[test]
fn rust_program_ad_diagflat_fails_closed_on_malformed_shape_metadata() {
    let ir = weighted_diagflat_ir("2xa", "0", &ascending_indices(2), &[1.0, 1.0]);
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0])
        .expect("malformed diagflat shape should gate, not crash");

    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diagflat shape metadata is malformed"),
        "{error}"
    );
}

#[test]
fn rust_program_ad_diagflat_fails_closed_on_extra_source_operands() {
    let ir = weighted_diagflat_ir("2", "0", &ascending_indices(2), &[1.0, 1.0])
        .replace(r#""inputs": ["%0"]"#, r#""inputs": ["%0", "%1"]"#);
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0])
        .expect("extra diagflat operands should gate, not crash");

    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diagflat replay requires exactly one source operand"),
        "{error}"
    );
}
