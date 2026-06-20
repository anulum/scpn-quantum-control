// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR parity tests

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, parse_program_ad_effect_ir,
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
        "bounded_rust_program_ad_ir_scalar_forward_interpreter_no_reverse_ad_no_llvm_jit"
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
