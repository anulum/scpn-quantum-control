// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD pseudoinverse linalg replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

fn weighted_pinv_ir(rows: usize, cols: usize, rcond: &str, weights: &[f64]) -> String {
    let input_count = rows * cols;
    let output_count = rows * cols;
    assert_eq!(weights.len(), output_count);

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

    let input_names = (0..input_count)
        .map(|index| format!(r#""%{index}""#))
        .collect::<Vec<_>>()
        .join(", ");
    let mut next_index = input_count;
    let mut output_targets = Vec::new();
    for output_row in 0..cols {
        for output_col in 0..rows {
            ssa_values.push(format!(
                r#"{{"name": "%{next_index}", "producer": {next_index}, "version": 0, "shape": [], "dtype": "float64", "effect": {next_index}}}"#
            ));
            effects.push(format!(
                r#"{{"index": {next_index}, "kind": "op", "target": "%{next_index}", "inputs": [{input_names}], "version": 0, "ordering": {next_index}, "operation": "linalg:pinv:{rows}x{cols}:{rcond}:{output_row}:{output_col}"}}"#
            ));
            output_targets.push(next_index);
            next_index += 1;
        }
    }

    let mut accumulated_target: Option<usize> = None;
    for (offset, weight) in weights.iter().enumerate() {
        let output_target = output_targets[offset];
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

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} differs from expected {expected}"
    );
}

#[test]
fn rust_program_ad_replays_full_rank_3x2_pinv_value_and_gradient() {
    let matrix = vec![2.0, 0.2, 0.3, 1.4, 0.5, -0.7];
    let weights = vec![0.4, -0.2, 0.3, 0.1, -0.5, 0.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_pinv_ir(3, 2, "0.0", &weights),
        &matrix,
    )
    .expect("full-rank 3x2 pinv replay should be supported");

    let expected_value = -0.130_756_586_716_041_4_f64;
    let expected_gradient = [
        -0.108_282_554_258_182_73_f64,
        0.099_489_977_187_421_77_f64,
        -0.040_028_087_662_086_04_f64,
        0.192_578_732_093_240_28_f64,
        0.002_042_541_823_601_450_6_f64,
        -0.098_286_614_644_587_32_f64,
    ];

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        expected_value,
        1.0e-12,
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient.iter()) {
        assert_close(*actual, *expected, 1.0e-12);
    }
}

#[test]
fn rust_program_ad_rejects_rank_deficient_pinv_gradient() {
    let rank_deficient = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
    let weights = vec![0.4, -0.2, 0.3, 0.1, -0.5, 0.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_pinv_ir(3, 2, "0.0", &weights),
        &rank_deficient,
    )
    .expect("unsupported rank-deficient pinv replay should return a structured result");
    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("pinv") && error.contains("full-rank"),
        "unexpected error: {error}"
    );
}

#[test]
fn rust_program_ad_rejects_unsupported_pinv_shape() {
    let matrix = vec![1.0, 0.0, 0.0];
    let weights = vec![1.0, -0.5, 0.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_pinv_ir(3, 1, "0.0", &weights),
        &matrix,
    )
    .expect("unsupported rank-1 pinv replay should return a structured result");
    assert!(!result.supported);
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("pinv") && error.contains("supports"),
        "unexpected error: {error}"
    );
}
