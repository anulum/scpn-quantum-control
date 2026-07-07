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

fn eigh_spectral_ir() -> String {
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
            {"index": 0, "kind": "parameter", "target": "%0", "inputs": [], "version": 0, "ordering": 0, "operation": "parameter"},
            {"index": 1, "kind": "parameter", "target": "%1", "inputs": [], "version": 0, "ordering": 1, "operation": "parameter"},
            {"index": 2, "kind": "parameter", "target": "%2", "inputs": [], "version": 0, "ordering": 2, "operation": "parameter"},
            {"index": 3, "kind": "parameter", "target": "%3", "inputs": [], "version": 0, "ordering": 3, "operation": "parameter"},
            {"index": 4, "kind": "op", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:eigh:eigenvalue:2x2:L:0"},
            {"index": 5, "kind": "op", "target": "%5", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 5, "operation": "linalg:eigh:eigenvalue:2x2:L:1"},
            {"index": 6, "kind": "op", "target": "%6", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 6, "operation": "linalg:eigh:eigenvector:2x2:L:0:0"},
            {"index": 7, "kind": "op", "target": "%7", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 7, "operation": "linalg:eigh:eigenvector:2x2:L:1:0"},
            {"index": 8, "kind": "op", "target": "%8", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 8, "operation": "linalg:eigh:eigenvector:2x2:L:0:1"},
            {"index": 9, "kind": "op", "target": "%9", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 9, "operation": "linalg:eigh:eigenvector:2x2:L:1:1"},
            {"index": 10, "kind": "op", "target": "%10", "inputs": ["%4", "0.75"], "version": 0, "ordering": 10, "operation": "mul"},
            {"index": 11, "kind": "op", "target": "%11", "inputs": ["%5", "-1.25"], "version": 0, "ordering": 11, "operation": "mul"},
            {"index": 12, "kind": "op", "target": "%12", "inputs": ["%6", "0.2"], "version": 0, "ordering": 12, "operation": "mul"},
            {"index": 13, "kind": "op", "target": "%13", "inputs": ["%7", "-0.4"], "version": 0, "ordering": 13, "operation": "mul"},
            {"index": 14, "kind": "op", "target": "%14", "inputs": ["%8", "0.6"], "version": 0, "ordering": 14, "operation": "mul"},
            {"index": 15, "kind": "op", "target": "%15", "inputs": ["%9", "0.1"], "version": 0, "ordering": 15, "operation": "mul"},
            {"index": 16, "kind": "op", "target": "%16", "inputs": ["%10", "%11"], "version": 0, "ordering": 16, "operation": "add"},
            {"index": 17, "kind": "op", "target": "%17", "inputs": ["%16", "%12"], "version": 0, "ordering": 17, "operation": "add"},
            {"index": 18, "kind": "op", "target": "%18", "inputs": ["%17", "%13"], "version": 0, "ordering": 18, "operation": "add"},
            {"index": 19, "kind": "op", "target": "%19", "inputs": ["%18", "%14"], "version": 0, "ordering": 19, "operation": "add"},
            {"index": 20, "kind": "op", "target": "%20", "inputs": ["%19", "%15"], "version": 0, "ordering": 20, "operation": "add"}
        ],
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": []
    }"#
    .to_owned()
}

fn svdvals_spectral_ir() -> String {
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
            {"index": 4, "kind": "op", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:svdvals:2x2:0"},
            {"index": 5, "kind": "op", "target": "%5", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 5, "operation": "linalg:svdvals:2x2:1"},
            {"index": 6, "kind": "op", "target": "%6", "inputs": ["%4", "0.5"], "version": 0, "ordering": 6, "operation": "mul"},
            {"index": 7, "kind": "op", "target": "%7", "inputs": ["%5", "-1.3"], "version": 0, "ordering": 7, "operation": "mul"},
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

fn symmetric_2x2_eigenvectors(a: f64, b: f64, d: f64) -> [[f64; 2]; 2] {
    let lambda0 = symmetric_2x2_eigenvalue(a, b, d, 0);
    let lambda1 = symmetric_2x2_eigenvalue(a, b, d, 1);
    let raw0 = if b > 0.0 && a <= d {
        [-b, a - lambda0]
    } else {
        [b, lambda0 - a]
    };
    let raw1 = if b > 0.0 && a > d {
        [-b, a - lambda1]
    } else {
        [b, lambda1 - a]
    };
    let norm0 = (raw0[0] * raw0[0] + raw0[1] * raw0[1]).sqrt();
    let norm1 = (raw1[0] * raw1[0] + raw1[1] * raw1[1]).sqrt();
    [
        [raw0[0] / norm0, raw1[0] / norm1],
        [raw0[1] / norm0, raw1[1] / norm1],
    ]
}

fn symmetric_2x2_eigh_adjoint(
    a: f64,
    b: f64,
    d: f64,
    eigenvalue_weights: [f64; 2],
    eigenvector_weights: [[f64; 2]; 2],
) -> [f64; 4] {
    let eigenvalues = [
        symmetric_2x2_eigenvalue(a, b, d, 0),
        symmetric_2x2_eigenvalue(a, b, d, 1),
    ];
    let eigenvectors = symmetric_2x2_eigenvectors(a, b, d);
    let mut adjoint = [0.0_f64; 4];
    for column in 0..2 {
        let vector = [eigenvectors[0][column], eigenvectors[1][column]];
        adjoint[0] += eigenvalue_weights[column] * vector[0] * vector[0];
        adjoint[1] += eigenvalue_weights[column] * vector[0] * vector[1];
        adjoint[2] += eigenvalue_weights[column] * vector[1] * vector[0];
        adjoint[3] += eigenvalue_weights[column] * vector[1] * vector[1];
    }
    for column in 0..2 {
        let other = 1 - column;
        let cotangent_column = [
            eigenvector_weights[0][column],
            eigenvector_weights[1][column],
        ];
        let other_vector = [eigenvectors[0][other], eigenvectors[1][other]];
        let column_vector = [eigenvectors[0][column], eigenvectors[1][column]];
        let dot = other_vector[0] * cotangent_column[0] + other_vector[1] * cotangent_column[1];
        let scale = dot / (eigenvalues[column] - eigenvalues[other]);
        let raw = [
            scale * other_vector[0] * column_vector[0],
            scale * other_vector[0] * column_vector[1],
            scale * other_vector[1] * column_vector[0],
            scale * other_vector[1] * column_vector[1],
        ];
        adjoint[0] += raw[0];
        adjoint[1] += 0.5 * (raw[1] + raw[2]);
        adjoint[2] += 0.5 * (raw[2] + raw[1]);
        adjoint[3] += raw[3];
    }
    adjoint
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

fn symmetric_2x2_unit_eigenvector(a: f64, b: f64, d: f64, upper: bool) -> [f64; 2] {
    let sign_index = if upper { 1 } else { 0 };
    let eigenvalue = symmetric_2x2_eigenvalue(a, b, d, sign_index);
    if b.abs() <= 1.0e-12 {
        return if (upper && a >= d) || (!upper && a < d) {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
    }
    let raw = [b, eigenvalue - a];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1]).sqrt();
    [raw[0] / norm, raw[1] / norm]
}

fn svdvals_2x2_values_and_adjoint(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    weights: [f64; 2],
) -> ([f64; 2], [f64; 4]) {
    let gram00 = a * a + c * c;
    let gram01 = a * b + c * d;
    let gram11 = b * b + d * d;
    let upper_vector = symmetric_2x2_unit_eigenvector(gram00, gram01, gram11, true);
    let lower_vector = symmetric_2x2_unit_eigenvector(gram00, gram01, gram11, false);
    let upper_sigma = symmetric_2x2_eigenvalue(gram00, gram01, gram11, 1).sqrt();
    let lower_sigma = symmetric_2x2_eigenvalue(gram00, gram01, gram11, 0).sqrt();
    let mut adjoint = [0.0_f64; 4];
    for (sigma, vector, weight) in [
        (upper_sigma, upper_vector, weights[0]),
        (lower_sigma, lower_vector, weights[1]),
    ] {
        let av = [a * vector[0] + b * vector[1], c * vector[0] + d * vector[1]];
        adjoint[0] += weight * av[0] * vector[0] / sigma;
        adjoint[1] += weight * av[0] * vector[1] / sigma;
        adjoint[2] += weight * av[1] * vector[0] / sigma;
        adjoint[3] += weight * av[1] * vector[1] / sigma;
    }
    ([upper_sigma, lower_sigma], adjoint)
}

#[test]
fn rust_program_ad_replays_distinct_2x2_svdvals_value_and_gradient() {
    let inputs = [2.0, 0.3, -0.2, 1.1];
    let result = interpret_program_ad_effect_ir_value_and_gradient(&svdvals_spectral_ir(), &inputs)
        .expect("valid svdvals spectral IR should parse");
    let (singular_values, expected_gradient) =
        svdvals_2x2_values_and_adjoint(inputs[0], inputs[1], inputs[2], inputs[3], [0.5, -1.3]);
    let expected_value = 0.5 * singular_values[0] - 1.3 * singular_values[1];

    assert!(result.supported, "{:?}", result.blocked_reasons);
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
fn rust_program_ad_rejects_rank_deficient_2x2_svdvals_gradient() {
    let ir = single_eigenvalue_ir("linalg:svdvals:2x2:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 2.0, 2.0, 4.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("svdvals gradient requires positive singular values")));
}

#[test]
fn rust_program_ad_rejects_repeated_2x2_svdvals_gradient() {
    let ir = single_eigenvalue_ir("linalg:svdvals:2x2:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[1.0, 0.0, 0.0, 1.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("svdvals gradient requires distinct singular values")));
}

#[test]
fn rust_program_ad_rejects_malformed_svdvals_metadata() {
    let ir = single_eigenvalue_ir("linalg:svdvals:3x3");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.3, -0.2, 1.1])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("svdvals operation metadata is malformed")));
}

#[test]
fn rust_program_ad_replays_distinct_2x2_eigh_value_and_gradient() {
    let inputs = [2.0, 0.35, 0.35, 3.0];
    let result = interpret_program_ad_effect_ir_value_and_gradient(&eigh_spectral_ir(), &inputs)
        .expect("valid eigh spectral IR should parse");

    assert!(result.supported, "{:?}", result.blocked_reasons);
    let eigenvalues = [
        symmetric_2x2_eigenvalue(inputs[0], inputs[1], inputs[3], 0),
        symmetric_2x2_eigenvalue(inputs[0], inputs[1], inputs[3], 1),
    ];
    let eigenvectors = symmetric_2x2_eigenvectors(inputs[0], inputs[1], inputs[3]);
    let expected_value = 0.75 * eigenvalues[0] - 1.25 * eigenvalues[1] + 0.2 * eigenvectors[0][0]
        - 0.4 * eigenvectors[0][1]
        + 0.6 * eigenvectors[1][0]
        + 0.1 * eigenvectors[1][1];
    let expected_gradient = symmetric_2x2_eigh_adjoint(
        inputs[0],
        inputs[1],
        inputs[3],
        [0.75, -1.25],
        [[0.2, -0.4], [0.6, 0.1]],
    );

    assert_eq!(result.supported_effect_count, 21);
    assert_eq!(result.parameter_targets, ["%0", "%1", "%2", "%3"]);
    assert!(
        (result.value.expect("supported result must have value") - expected_value).abs() < 1.0e-12
    );
    for (observed, expected) in result.gradient.iter().zip(expected_gradient.iter()) {
        assert!((observed - expected).abs() < 1.0e-12);
    }
}

#[test]
fn rust_program_ad_rejects_nonsymmetric_2x2_eigh_input() {
    let ir = single_eigenvalue_ir("linalg:eigh:eigenvalue:2x2:L:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.4, -0.2, 3.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("eigh requires a symmetric 2x2 matrix")));
}

#[test]
fn rust_program_ad_rejects_degenerate_2x2_eigh_gradient() {
    let ir = single_eigenvalue_ir("linalg:eigh:eigenvalue:2x2:L:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.0, 0.0, 2.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("eigh gradient requires distinct eigenvalues")));
}

#[test]
fn rust_program_ad_rejects_diagonal_2x2_eigh_eigenvector_gradient() {
    let ir = single_eigenvalue_ir("linalg:eigh:eigenvector:2x2:L:0:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.0, 0.0, 3.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result.blocked_reasons.iter().any(|reason| {
        reason.contains("eigh eigenvector gradient requires nonzero off-diagonal entries")
    }));
}

#[test]
fn rust_program_ad_rejects_malformed_2x2_eigh_metadata() {
    let ir = single_eigenvalue_ir("linalg:eigh:eigenvector:2x2:L:2:0");
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[2.0, 0.35, 0.35, 3.0])
        .expect("valid IR shape should parse");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| { reason.contains("eigh eigenvector column and row must be 0 or 1") }));
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
