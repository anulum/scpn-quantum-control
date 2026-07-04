// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral linalg replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

fn spectral_ir() -> String {
    r#"{
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
            {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8}
        ],
        "effects": [
            {"index": 0, "kind": "parameter", "target": "%0", "inputs": [], "version": 0, "ordering": 0, "operation": "parameter"},
            {"index": 1, "kind": "parameter", "target": "%1", "inputs": [], "version": 0, "ordering": 1, "operation": "parameter"},
            {"index": 2, "kind": "parameter", "target": "%2", "inputs": [], "version": 0, "ordering": 2, "operation": "parameter"},
            {"index": 3, "kind": "parameter", "target": "%3", "inputs": [], "version": 0, "ordering": 3, "operation": "parameter"},
            {"index": 4, "kind": "op", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:eigvalsh:0"},
            {"index": 5, "kind": "op", "target": "%5", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 5, "operation": "linalg:eigvalsh:1"},
            {"index": 6, "kind": "op", "target": "%6", "inputs": ["%4", "0.75"], "version": 0, "ordering": 6, "operation": "mul"},
            {"index": 7, "kind": "op", "target": "%7", "inputs": ["%5", "-1.25"], "version": 0, "ordering": 7, "operation": "mul"},
            {"index": 8, "kind": "op", "target": "%8", "inputs": ["%6", "%7"], "version": 0, "ordering": 8, "operation": "add"}
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": []
    }"#
    .to_owned()
}

fn eigvals_spectral_ir() -> String {
    r#"{
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
            {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8}
        ],
        "effects": [
            {"index": 0, "kind": "parameter", "target": "%0", "inputs": [], "version": 0, "ordering": 0, "operation": "parameter"},
            {"index": 1, "kind": "parameter", "target": "%1", "inputs": [], "version": 0, "ordering": 1, "operation": "parameter"},
            {"index": 2, "kind": "parameter", "target": "%2", "inputs": [], "version": 0, "ordering": 2, "operation": "parameter"},
            {"index": 3, "kind": "parameter", "target": "%3", "inputs": [], "version": 0, "ordering": 3, "operation": "parameter"},
            {"index": 4, "kind": "op", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:eigvals:2x2:0"},
            {"index": 5, "kind": "op", "target": "%5", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 5, "operation": "linalg:eigvals:2x2:1"},
            {"index": 6, "kind": "op", "target": "%6", "inputs": ["%4", "0.75"], "version": 0, "ordering": 6, "operation": "mul"},
            {"index": 7, "kind": "op", "target": "%7", "inputs": ["%5", "-1.25"], "version": 0, "ordering": 7, "operation": "mul"},
            {"index": 8, "kind": "op", "target": "%8", "inputs": ["%6", "%7"], "version": 0, "ordering": 8, "operation": "add"}
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": []
    }"#
    .to_owned()
}

fn single_eigenvalue_ir(operation: &str) -> String {
    format!(
        r#"{{
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [
            {{"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}},
            {{"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}},
            {{"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}},
            {{"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3}},
            {{"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}}
        ],
        "effects": [
            {{"index": 0, "kind": "parameter", "target": "%0", "inputs": [], "version": 0, "ordering": 0, "operation": "parameter"}},
            {{"index": 1, "kind": "parameter", "target": "%1", "inputs": [], "version": 0, "ordering": 1, "operation": "parameter"}},
            {{"index": 2, "kind": "parameter", "target": "%2", "inputs": [], "version": 0, "ordering": 2, "operation": "parameter"}},
            {{"index": 3, "kind": "parameter", "target": "%3", "inputs": [], "version": 0, "ordering": 3, "operation": "parameter"}},
            {{"index": 4, "kind": "op", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "{operation}"}}
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": []
    }}"#
    )
}

fn symmetric_2x2_eigenvalue(a: f64, b: f64, d: f64, index: usize) -> f64 {
    let center = 0.5 * (a + d);
    let radius = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    if index == 0 {
        center - radius
    } else {
        center + radius
    }
}

fn eigenvector_outer(a: f64, b: f64, d: f64, index: usize) -> [f64; 4] {
    if b.abs() <= 1.0e-14 {
        let lower_is_first_axis = a <= d;
        return if (index == 0 && lower_is_first_axis) || (index == 1 && !lower_is_first_axis) {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
    }
    let lambda = symmetric_2x2_eigenvalue(a, b, d, index);
    let raw_x = b;
    let raw_y = lambda - a;
    let norm = (raw_x * raw_x + raw_y * raw_y).sqrt();
    let x = raw_x / norm;
    let y = raw_y / norm;
    [x * x, x * y, x * y, y * y]
}

fn real_2x2_eigenvalue(a: f64, b: f64, c: f64, d: f64, index: usize) -> f64 {
    let center = 0.5 * (a + d);
    let radius = 0.5 * ((a - d) * (a - d) + 4.0 * b * c).sqrt();
    if index == 0 {
        center - radius
    } else {
        center + radius
    }
}

fn real_2x2_eigvals_adjoint(a: f64, b: f64, c: f64, d: f64, weights: [f64; 2]) -> [f64; 4] {
    let gap = ((a - d) * (a - d) + 4.0 * b * c).sqrt();
    let diagonal_delta = a - d;
    let lower = [
        0.5 - diagonal_delta / (2.0 * gap),
        -c / gap,
        -b / gap,
        0.5 + diagonal_delta / (2.0 * gap),
    ];
    let upper = [
        0.5 + diagonal_delta / (2.0 * gap),
        c / gap,
        b / gap,
        0.5 - diagonal_delta / (2.0 * gap),
    ];
    [
        weights[0] * lower[0] + weights[1] * upper[0],
        weights[0] * lower[1] + weights[1] * upper[1],
        weights[0] * lower[2] + weights[1] * upper[2],
        weights[0] * lower[3] + weights[1] * upper[3],
    ]
}

#[test]
fn rust_program_ad_replays_distinct_2x2_eigvalsh_value_and_gradient() {
    let inputs = [2.0, 0.35, 0.35, 3.0];
    let result = interpret_program_ad_effect_ir_value_and_gradient(&spectral_ir(), &inputs)
        .expect("valid spectral IR should parse");

    assert!(result.supported, "{:?}", result.blocked_reasons);
    let lambda0 = symmetric_2x2_eigenvalue(inputs[0], inputs[1], inputs[3], 0);
    let lambda1 = symmetric_2x2_eigenvalue(inputs[0], inputs[1], inputs[3], 1);
    let expected_value = 0.75 * lambda0 - 1.25 * lambda1;
    let outer0 = eigenvector_outer(inputs[0], inputs[1], inputs[3], 0);
    let outer1 = eigenvector_outer(inputs[0], inputs[1], inputs[3], 1);
    let expected_gradient = outer0
        .iter()
        .zip(outer1.iter())
        .map(|(lower, upper)| 0.75 * lower - 1.25 * upper)
        .collect::<Vec<f64>>();

    assert_eq!(result.supported_effect_count, 9);
    assert_eq!(result.parameter_targets, ["%0", "%1", "%2", "%3"]);
    assert!(
        (result.value.expect("supported result must have value") - expected_value).abs() < 1.0e-12
    );
    for (observed, expected) in result.gradient.iter().zip(expected_gradient.iter()) {
        assert!((observed - expected).abs() < 1.0e-12);
    }
}

#[test]
fn rust_program_ad_replays_real_distinct_2x2_eigvals_value_and_gradient() {
    let inputs = [2.0, 0.4, 0.15, 3.0];
    let result = interpret_program_ad_effect_ir_value_and_gradient(&eigvals_spectral_ir(), &inputs)
        .expect("valid eigvals spectral IR should parse");

    assert!(result.supported, "{:?}", result.blocked_reasons);
    let lambda0 = real_2x2_eigenvalue(inputs[0], inputs[1], inputs[2], inputs[3], 0);
    let lambda1 = real_2x2_eigenvalue(inputs[0], inputs[1], inputs[2], inputs[3], 1);
    let expected_value = 0.75 * lambda0 - 1.25 * lambda1;
    let expected_gradient =
        real_2x2_eigvals_adjoint(inputs[0], inputs[1], inputs[2], inputs[3], [0.75, -1.25]);

    assert_eq!(result.supported_effect_count, 9);
    assert_eq!(result.parameter_targets, ["%0", "%1", "%2", "%3"]);
    assert!(
        (result.value.expect("supported result must have value") - expected_value).abs() < 1.0e-12
    );
    for (observed, expected) in result.gradient.iter().zip(expected_gradient.iter()) {
        assert!((observed - expected).abs() < 1.0e-12);
    }
}

#[test]
fn rust_program_ad_rejects_degenerate_2x2_eigvalsh_gradient() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_eigenvalue_ir("linalg:eigvalsh:0"),
        &[1.0, 0.0, 0.0, 1.0],
    )
    .expect("valid spectral IR should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("distinct")));
}

#[test]
fn rust_program_ad_rejects_degenerate_2x2_eigvals_gradient() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_eigenvalue_ir("linalg:eigvals:2x2:0"),
        &[1.0, 0.0, 0.0, 1.0],
    )
    .expect("valid eigvals spectral IR should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("distinct")));
}

#[test]
fn rust_program_ad_rejects_complex_2x2_eigvals_spectrum() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_eigenvalue_ir("linalg:eigvals:2x2:0"),
        &[0.0, -1.0, 1.0, 0.0],
    )
    .expect("valid eigvals spectral IR should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("real distinct eigenvalues")));
}

#[test]
fn rust_program_ad_rejects_nonsymmetric_2x2_eigvalsh_input() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_eigenvalue_ir("linalg:eigvalsh:0"),
        &[2.0, 0.5, 0.25, 3.0],
    )
    .expect("valid spectral IR should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("symmetric")));
}
