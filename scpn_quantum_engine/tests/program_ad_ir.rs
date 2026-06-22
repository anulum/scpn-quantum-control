// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR parity tests

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
    parse_program_ad_effect_ir,
};

const VALID_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [2], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "x", "inputs": [], "version": 0, "ordering": 0},
    {"index": 1, "kind": "control_branch", "target": "branch", "inputs": ["%0"], "version": 0, "ordering": 1}
  ],
  "alias_edges": [
    {"source": "view", "target": "base", "kind": "view_alias", "version": 0}
  ],
  "control_regions": [
    {"index": 0, "kind": "runtime_branch", "predicate": "%0 > 0", "entered": true, "source_line": null}
  ],
  "phi_nodes": [
    {"index": 0, "target": "phi:runtime_branch:0", "incoming": ["executed_true", "executed_false"], "control_region": 0, "selected": "executed_true", "source_line": null}
  ],
  "bytecode_offsets": [0, 2, 4]
}"#;

const EXECUTABLE_SCALAR_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0", "%0"], "version": 0, "ordering": 2, "operation": "mul"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%1", "2.0"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%2", "%3"], "version": 0, "ordering": 4, "operation": "add"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%0"], "version": 0, "ordering": 5, "operation": "sin"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%4", "%5"], "version": 0, "ordering": 6, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const ABS_CUSP_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "abs"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const EXECUTED_BRANCH_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "control_branch", "target": "%2", "inputs": [], "version": 0, "ordering": 2, "operation": "branch:%0:gt:%1:True"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%0", "%0"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%1", "2.0"], "version": 0, "ordering": 4, "operation": "mul"},
    {"index": 5, "kind": "pure", "target": "%5", "inputs": ["%3", "%4"], "version": 0, "ordering": 5, "operation": "add"},
    {"index": 6, "kind": "primitive", "target": "%6", "inputs": ["%0"], "version": 0, "ordering": 6, "operation": "sin"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%5", "%6"], "version": 0, "ordering": 7, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [
    {"index": 0, "kind": "runtime_branch", "predicate": "branch:%0:gt:%1:True", "entered": true, "source_line": null},
    {"index": 1, "kind": "source_control_flow", "predicate": "if_expression", "entered": true, "source_line": 3}
  ],
  "phi_nodes": [
    {"index": 0, "target": "phi:runtime_branch:0", "incoming": ["executed_true", "executed_false"], "control_region": 0, "selected": "executed_true", "source_line": null}
  ],
  "bytecode_offsets": [0, 2, 4]
}"#;

const SCALAR_PRIMITIVE_FAMILY_PROGRAM_AD_IR: &str = r#"{
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
    {"name": "%20", "producer": 20, "version": 0, "shape": [], "dtype": "float64", "effect": 20},
    {"name": "%21", "producer": 21, "version": 0, "shape": [], "dtype": "float64", "effect": 21},
    {"name": "%22", "producer": 22, "version": 0, "shape": [], "dtype": "float64", "effect": 22},
    {"name": "%23", "producer": 23, "version": 0, "shape": [], "dtype": "float64", "effect": 23}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["z"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["w"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%0", "2.0"], "version": 0, "ordering": 4, "operation": "add"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%4"], "version": 0, "ordering": 5, "operation": "sqrt"},
    {"index": 6, "kind": "primitive", "target": "%6", "inputs": ["%1"], "version": 0, "ordering": 6, "operation": "tanh"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%5", "%6"], "version": 0, "ordering": 7, "operation": "add"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%2"], "version": 0, "ordering": 8, "operation": "log1p"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%7", "%8"], "version": 0, "ordering": 9, "operation": "add"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%3"], "version": 0, "ordering": 10, "operation": "expm1"},
    {"index": 11, "kind": "pure", "target": "%11", "inputs": ["%9", "%10"], "version": 0, "ordering": 11, "operation": "add"},
    {"index": 12, "kind": "pure", "target": "%12", "inputs": ["%0", "3.0"], "version": 0, "ordering": 12, "operation": "add"},
    {"index": 13, "kind": "primitive", "target": "%13", "inputs": ["%12"], "version": 0, "ordering": 13, "operation": "reciprocal"},
    {"index": 14, "kind": "pure", "target": "%14", "inputs": ["%11", "%13"], "version": 0, "ordering": 14, "operation": "add"},
    {"index": 15, "kind": "pure", "target": "%15", "inputs": ["%1", "0.2"], "version": 0, "ordering": 15, "operation": "mul"},
    {"index": 16, "kind": "primitive", "target": "%16", "inputs": ["%15"], "version": 0, "ordering": 16, "operation": "arcsin"},
    {"index": 17, "kind": "pure", "target": "%17", "inputs": ["%14", "%16"], "version": 0, "ordering": 17, "operation": "add"},
    {"index": 18, "kind": "pure", "target": "%18", "inputs": ["%2", "0.1"], "version": 0, "ordering": 18, "operation": "mul"},
    {"index": 19, "kind": "primitive", "target": "%19", "inputs": ["%18"], "version": 0, "ordering": 19, "operation": "arccos"},
    {"index": 20, "kind": "pure", "target": "%20", "inputs": ["%17", "%19"], "version": 0, "ordering": 20, "operation": "add"},
    {"index": 21, "kind": "pure", "target": "%21", "inputs": ["%3", "1.0"], "version": 0, "ordering": 21, "operation": "add"},
    {"index": 22, "kind": "primitive", "target": "%22", "inputs": ["%21"], "version": 0, "ordering": 22, "operation": "abs"},
    {"index": 23, "kind": "pure", "target": "%23", "inputs": ["%20", "%22"], "version": 0, "ordering": 23, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

#[test]
fn program_ad_effect_ir_parser_round_trips_python_payload_shape() {
    let ir = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();

    assert_eq!(ir.format, "program_ad_effect_ir.v1");
    assert_eq!(ir.ssa_values.len(), 2);
    assert_eq!(ir.ssa_values[1].shape, vec![2]);
    assert_eq!(ir.effects[1].kind, "control_branch");
    assert_eq!(ir.alias_edges[0].kind, "view_alias");
    assert_eq!(ir.control_regions[0].kind, "runtime_branch");
    assert!(ir.control_regions[0].entered);
    assert_eq!(
        ir.phi_nodes[0].incoming,
        vec!["executed_true", "executed_false"]
    );
    assert_eq!(ir.bytecode_offsets, vec![0, 2, 4]);

    let summary = ir.metadata_summary();
    assert_eq!(summary.format, "program_ad_effect_ir.v1");
    assert_eq!(summary.ssa_value_count, 2);
    assert_eq!(summary.effect_count, 2);
    assert_eq!(summary.alias_edge_count, 1);
    assert_eq!(summary.control_region_count, 1);
    assert_eq!(summary.phi_node_count, 1);
    assert_eq!(summary.claim_boundary, "metadata_only_no_program_execution");
}

#[test]
fn program_ad_effect_ir_parser_fails_closed_on_malformed_payloads() {
    let wrong_format =
        VALID_PROGRAM_AD_IR.replace("program_ad_effect_ir.v1", "program_ad_effect_ir.v2");
    assert!(parse_program_ad_effect_ir(&wrong_format)
        .unwrap_err()
        .contains("format must be program_ad_effect_ir.v1"));

    let wrong_effect_shape = r#"{
      "format": "program_ad_effect_ir.v1",
      "ssa_values": [],
      "effects": {},
      "alias_edges": [],
      "control_regions": [],
      "phi_nodes": [],
      "bytecode_offsets": []
    }"#;
    assert!(parse_program_ad_effect_ir(wrong_effect_shape)
        .unwrap_err()
        .contains("effects"));

    let bad_phi = VALID_PROGRAM_AD_IR.replace(
        "\"incoming\": [\"executed_true\", \"executed_false\"]",
        "\"incoming\": [\"executed_true\"]",
    );
    assert!(parse_program_ad_effect_ir(&bad_phi)
        .unwrap_err()
        .contains("phi_nodes incoming"));

    let bad_source_line = VALID_PROGRAM_AD_IR.replace(
        "\"kind\": \"runtime_branch\", \"predicate\": \"%0 > 0\", \"entered\": true, \"source_line\": null",
        "\"kind\": \"runtime_branch\", \"predicate\": \"%0 > 0\", \"entered\": true, \"source_line\": 0",
    );
    assert!(parse_program_ad_effect_ir(&bad_source_line)
        .unwrap_err()
        .contains("source_line"));
}

#[test]
fn program_ad_effect_ir_rust_interpreter_executes_opcode_bearing_scalar_subset() {
    let result =
        interpret_program_ad_effect_ir_forward(EXECUTABLE_SCALAR_PROGRAM_AD_IR, &[0.4, -0.2])
            .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 7);
    assert_eq!(result.supported_effect_count, 7);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_interpreter_replays_executed_branch_metadata() {
    let result =
        interpret_program_ad_effect_ir_forward(EXECUTED_BRANCH_PROGRAM_AD_IR, &[0.4, -0.2])
            .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_scalar_reverse_subset() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        EXECUTABLE_SCALAR_PROGRAM_AD_IR,
        &[0.4, -0.2],
    )
    .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 7);
    assert_eq!(result.supported_effect_count, 7);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - (2.0_f64 * 0.4_f64 + 0.4_f64.cos())).abs() <= 1.0e-12);
    assert!((result.gradient[1] - 2.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.parameter_targets, vec!["%0", "%1"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_executed_branch_metadata() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        EXECUTED_BRANCH_PROGRAM_AD_IR,
        &[0.4, -0.2],
    )
    .unwrap();

    let expected = 0.4_f64 * 0.4_f64 + 2.0_f64 * -0.2_f64 + 0.4_f64.sin();
    assert!(result.supported);
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - (2.0_f64 * 0.4_f64 + 0.4_f64.cos())).abs() <= 1.0e-12);
    assert!((result.gradient[1] - 2.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.parameter_targets, vec!["%0", "%1"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_scalar_primitive_family() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        SCALAR_PRIMITIVE_FAMILY_PROGRAM_AD_IR,
        &[0.4, -0.2, 0.25, 0.1],
    )
    .unwrap();

    let x: f64 = 0.4;
    let y: f64 = -0.2;
    let z: f64 = 0.25;
    let w: f64 = 0.1;
    let expected = (x + 2.0).sqrt()
        + y.tanh()
        + z.ln_1p()
        + w.exp_m1()
        + 1.0 / (x + 3.0)
        + (0.2 * y).asin()
        + (0.1 * z).acos()
        + (w + 1.0).abs();
    let expected_gradient = [
        0.5 / (x + 2.0).sqrt() - 1.0 / ((x + 3.0) * (x + 3.0)),
        1.0 - y.tanh() * y.tanh() + 0.2 / (1.0 - (0.2 * y) * (0.2 * y)).sqrt(),
        1.0 / (1.0 + z) - 0.1 / (1.0 - (0.1 * z) * (0.1 * z)).sqrt(),
        w.exp() + 1.0,
    ];

    assert!(result.supported);
    assert_eq!(result.effect_count, 24);
    assert_eq!(result.supported_effect_count, 24);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 4);
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.parameter_targets, vec!["%0", "%1", "%2", "%3"]);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_interpreter_fails_closed_without_operation_metadata() {
    let legacy_ir = EXECUTABLE_SCALAR_PROGRAM_AD_IR
        .replace(", \"operation\": \"parameter\"", "")
        .replace(", \"operation\": \"mul\"", "")
        .replace(", \"operation\": \"add\"", "")
        .replace(", \"operation\": \"sin\"", "");
    let result = interpret_program_ad_effect_ir_forward(&legacy_ir, &[0.4, -0.2]).unwrap();

    assert!(!result.supported);
    assert_eq!(result.value, None);
    assert_eq!(result.supported_effect_count, 0);
    assert!(result.blocked_reasons[0].contains("operation metadata"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_abs_cusp() {
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(ABS_CUSP_PROGRAM_AD_IR, &[0.0]).unwrap();

    assert!(!result.supported);
    assert_eq!(result.value, None);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("abs gradient is undefined at zero"));
}

const INERT_VIEW_ALIAS_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0", "%0"], "version": 0, "ordering": 2, "operation": "mul"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%1", "%1"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%2", "%3"], "version": 0, "ordering": 4, "operation": "add"}
  ],
  "alias_edges": [
    {"source": "%array[0]", "target": "view:reshape:0[0]", "kind": "view_alias", "version": 0}
  ],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const MUTATION_ALIAS_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0", "%1"], "version": 0, "ordering": 2, "operation": "mul"}
  ],
  "alias_edges": [
    {"source": "%0", "target": "%0", "kind": "mutation_version", "version": 1}
  ],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_inert_view_alias() {
    // A reshape/transpose/slice view leaves an inert view_alias edge while the op-effects
    // keep referencing canonical scalar SSA, so the bounded Rust replay stays exact.
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(INERT_VIEW_ALIAS_PROGRAM_AD_IR, &[3.0, -2.0])
            .unwrap();

    assert!(result.supported);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - 13.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - 6.0_f64).abs() <= 1.0e-12);
    assert!((result.gradient[1] - (-4.0_f64)).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_mutation_alias() {
    // Non-view alias kinds (here a mutation_version edge) can change a value's content and
    // stay outside the bounded scalar replay.
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(MUTATION_ALIAS_PROGRAM_AD_IR, &[2.0, 3.0])
            .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}

const LINALG_TRACE_2X2_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%0", "%3"], "version": 0, "ordering": 4, "operation": "linalg:trace:2x2:offset:0"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

const LINALG_DET_2X2_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:det:2x2"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_trace() {
    // trace([[a,b],[c,d]]) = a + d; gradient is 1 on the diagonal, 0 off it.
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(LINALG_TRACE_2X2_PROGRAM_AD_IR, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_det_2x2() {
    // det([[a,b],[c,d]]) = a*d - b*c; cofactor gradient [d, -c, -b, a].
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(LINALG_DET_2X2_PROGRAM_AD_IR, &[2.0, 1.0, 1.0, 3.0])
            .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 4);
    assert!((result.gradient[0] - 3.0_f64).abs() <= 1.0e-12);
    assert!((result.gradient[1] - (-1.0_f64)).abs() <= 1.0e-12);
    assert!((result.gradient[2] - (-1.0_f64)).abs() <= 1.0e-12);
    assert!((result.gradient[3] - 2.0_f64).abs() <= 1.0e-12);
}

const LINALG_INV_2X2_ELEMENT_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%0", "%1", "%2", "%3"], "version": 0, "ordering": 4, "operation": "linalg:inv:2x2:0:0"},
    {"index": 5, "kind": "pure", "target": "%5", "inputs": ["%4", "1.0"], "version": 0, "ordering": 5, "operation": "mul"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

const LINALG_SOLVE_2X2_FINAL_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["r0"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["r1"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 6, "operation": "linalg:solve:2x2:rhs:2:0"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_inverse_element() {
    // inv([[a,b],[c,d]])[0,0] = d/det; reduced by *1.0 so it is the program value.
    // For [2,1,1,3]: det=5, M=[0.6,-0.2,-0.2,0.4]; d(M00)/dA = [-0.36, 0.12, 0.12, -0.04].
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(LINALG_INV_2X2_ELEMENT_PROGRAM_AD_IR, &[2.0, 1.0, 1.0, 3.0])
            .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 0.6_f64).abs() <= 1.0e-12);
    let expected = [-0.36_f64, 0.12, 0.12, -0.04];
    assert_eq!(result.gradient.len(), 4);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-12, "{got} vs {want}");
    }
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_indexed_multi_output_linalg() {
    // A bare solve element as the final effect: the IR does not record which component the
    // program returned, so the replay fails closed rather than replaying the wrong element.
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(LINALG_SOLVE_2X2_FINAL_PROGRAM_AD_IR, &[3.0, 1.0, 2.0, 4.0, 5.0, 6.0])
            .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("indexed multi-output linalg"));
}

const LINALG_DET_3X3_PROGRAM_AD_IR: &str = r#"{
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
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["e"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["f"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["g"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["h"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["i"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8"], "version": 0, "ordering": 9, "operation": "linalg:det:3x3"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_det_3x3() {
    // det of [[2,0,1],[1,3,2],[0,1,4]] = 21; gradient is the cofactor matrix.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_3X3_PROGRAM_AD_IR,
        &[2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 21.0_f64).abs() <= 1.0e-12);
    let expected = [10.0_f64, -4.0, 1.0, 1.0, 8.0, -2.0, -3.0, -3.0, 6.0];
    assert_eq!(result.gradient.len(), 9);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-12, "{got} vs {want}");
    }
}

const LINALG_INV_3X3_ELEMENT_PROGRAM_AD_IR: &str = r#"{
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
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["e"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["f"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["g"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["h"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["i"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8"], "version": 0, "ordering": 9, "operation": "linalg:inv:3x3:0:0"},
    {"index": 10, "kind": "pure", "target": "%10", "inputs": ["%9", "1.0"], "version": 0, "ordering": 10, "operation": "mul"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_inverse_3x3_element() {
    // inv([[2,0,1],[1,3,2],[0,1,4]])[0,0] = 10/21; reduced by *1.0 so it is the program value.
    // d(M00)/dA00 = -M00^2 = -(10/21)^2.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_INV_3X3_ELEMENT_PROGRAM_AD_IR,
        &[2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    let m00 = 10.0_f64 / 21.0;
    assert!((result.value.unwrap() - m00).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 9);
    assert!((result.gradient[0] - (-m00 * m00)).abs() <= 1.0e-12);
}

const LINALG_DET_4X4_DIAGONAL_PROGRAM_AD_IR: &str = r#"{"format": "program_ad_effect_ir.v1", "ssa_values": [{"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}, {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}, {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}, {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3}, {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}, {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5}, {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6}, {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7}, {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8}, {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}, {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10}, {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11}, {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12}, {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13}, {"name": "%14", "producer": 14, "version": 0, "shape": [], "dtype": "float64", "effect": 14}, {"name": "%15", "producer": 15, "version": 0, "shape": [], "dtype": "float64", "effect": 15}, {"name": "%16", "producer": 16, "version": 0, "shape": [], "dtype": "float64", "effect": 16}], "effects": [{"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"}, {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"}, {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"}, {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"}, {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["e"], "version": 0, "ordering": 4, "operation": "parameter"}, {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["f"], "version": 0, "ordering": 5, "operation": "parameter"}, {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["g"], "version": 0, "ordering": 6, "operation": "parameter"}, {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["h"], "version": 0, "ordering": 7, "operation": "parameter"}, {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["i"], "version": 0, "ordering": 8, "operation": "parameter"}, {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["j"], "version": 0, "ordering": 9, "operation": "parameter"}, {"index": 10, "kind": "parameter", "target": "%10", "inputs": ["k"], "version": 0, "ordering": 10, "operation": "parameter"}, {"index": 11, "kind": "parameter", "target": "%11", "inputs": ["l"], "version": 0, "ordering": 11, "operation": "parameter"}, {"index": 12, "kind": "parameter", "target": "%12", "inputs": ["m"], "version": 0, "ordering": 12, "operation": "parameter"}, {"index": 13, "kind": "parameter", "target": "%13", "inputs": ["n"], "version": 0, "ordering": 13, "operation": "parameter"}, {"index": 14, "kind": "parameter", "target": "%14", "inputs": ["o"], "version": 0, "ordering": 14, "operation": "parameter"}, {"index": 15, "kind": "parameter", "target": "%15", "inputs": ["p"], "version": 0, "ordering": 15, "operation": "parameter"}, {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11", "%12", "%13", "%14", "%15"], "version": 0, "ordering": 16, "operation": "linalg:det:4x4"}], "alias_edges": [], "control_regions": [], "phi_nodes": [], "bytecode_offsets": [0]}
"#;

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_general_linalg_det_4x4() {
    // det(diag(2,3,4,5)) = 120 via the LU general path; gradient is the adjugate,
    // diag(60, 40, 30, 24), zero off the diagonal.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_4X4_DIAGONAL_PROGRAM_AD_IR,
        &[
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 120.0_f64).abs() <= 1.0e-9);
    let expected = [
        60.0_f64, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 24.0,
    ];
    assert_eq!(result.gradient.len(), 16);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-9, "{got} vs {want}");
    }
}
