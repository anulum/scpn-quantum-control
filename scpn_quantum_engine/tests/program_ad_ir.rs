// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR parity tests

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
    mirror_program_ad_registry_metadata, parse_program_ad_effect_ir,
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

const VALID_REGISTRY_COVERAGE_SNAPSHOT: &str = r#"{
  "supported": true,
  "covered_primitives": 3,
  "total_primitives": 3,
  "blocked_identities": [],
  "family_counts": {
    "elementwise": 2,
    "linalg": 1
  },
  "rows": [
    {
      "family": "elementwise",
      "primitive": "sin",
      "identity": "scpn.program_ad.elementwise:sin@1",
      "derivative_rule": "program_ad_elementwise_sin_derivative_rule",
      "has_batching_rule": true,
      "has_lowering_rule": false,
      "has_lowering_metadata": true,
      "has_shape_rule": true,
      "has_dtype_rule": true,
      "has_static_argument_rule": true,
      "nondifferentiable_policy": "program_ad_trace_exact_fail_closed",
      "effect": "pure",
      "lowering_metadata_keys": [
        "mlir_op",
        "nondifferentiable_boundary",
        "nondifferentiable_boundary_policy"
      ],
      "complete": true,
      "blocked_reasons": [],
      "claim_boundary": "registry-dispatched Program AD primitive coverage over declared derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts only; not executable Rust, LLVM, JIT, provider, hardware, or performance evidence"
    },
    {
      "family": "elementwise",
      "primitive": "sqrt",
      "identity": "scpn.program_ad.elementwise:sqrt@1",
      "derivative_rule": "program_ad_elementwise_sqrt_derivative_rule",
      "has_batching_rule": true,
      "has_lowering_rule": false,
      "has_lowering_metadata": true,
      "has_shape_rule": true,
      "has_dtype_rule": true,
      "has_static_argument_rule": true,
      "nondifferentiable_policy": "program_ad_trace_exact_fail_closed",
      "effect": "pure",
      "lowering_metadata_keys": [
        "mlir_op",
        "nondifferentiable_boundary",
        "nondifferentiable_boundary_policy"
      ],
      "complete": true,
      "blocked_reasons": [],
      "claim_boundary": "registry-dispatched Program AD primitive coverage over declared derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts only; not executable Rust, LLVM, JIT, provider, hardware, or performance evidence"
    },
    {
      "family": "linalg",
      "primitive": "det",
      "identity": "scpn.program_ad.linalg:det@1",
      "derivative_rule": "program_ad_linalg_det_derivative_rule",
      "has_batching_rule": true,
      "has_lowering_rule": false,
      "has_lowering_metadata": true,
      "has_shape_rule": true,
      "has_dtype_rule": true,
      "has_static_argument_rule": true,
      "nondifferentiable_policy": "program_ad_trace_exact_fail_closed",
      "effect": "pure",
      "lowering_metadata_keys": [
        "mlir_op",
        "nondifferentiable_boundary",
        "nondifferentiable_boundary_policy"
      ],
      "complete": true,
      "blocked_reasons": [],
      "claim_boundary": "registry-dispatched Program AD primitive coverage over declared derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts only; not executable Rust, LLVM, JIT, provider, hardware, or performance evidence"
    }
  ],
  "claim_boundary": "registry-dispatched Program AD primitive coverage over declared derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts only; not executable Rust, LLVM, JIT, provider, hardware, or performance evidence"
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

const ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [3], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [3], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [3], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["bias"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "primitive", "target": "%2", "inputs": ["%0"], "version": 0, "ordering": 2, "operation": "sin"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%0", "%1"], "version": 0, "ordering": 3, "operation": "add"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%2", "%3"], "version": 0, "ordering": 4, "operation": "mul"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%4"], "version": 0, "ordering": 5, "operation": "sum"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [2], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "sin"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const STRUCTURAL_ARRAY_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [6], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [2, 1], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [3, 2], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [6], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [6], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["theta"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0"], "version": 0, "ordering": 2, "operation": "reshape"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%2"], "version": 0, "ordering": 3, "operation": "broadcast_to"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%3"], "version": 0, "ordering": 4, "operation": "transpose"},
    {"index": 5, "kind": "pure", "target": "%5", "inputs": ["%4"], "version": 0, "ordering": 5, "operation": "ravel"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%5", "%1"], "version": 0, "ordering": 6, "operation": "mul"},
    {"index": 7, "kind": "primitive", "target": "%7", "inputs": ["%6"], "version": 0, "ordering": 7, "operation": "mean"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [2], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [4], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [4], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [4], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [2, 2], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [4], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [4], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [4], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [4], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["left"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["right"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["concat_weights"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["stack_weights"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%0", "%1"], "version": 0, "ordering": 4, "operation": "concatenate:axis:0"},
    {"index": 5, "kind": "pure", "target": "%5", "inputs": ["%0", "%1"], "version": 0, "ordering": 5, "operation": "stack:axis:1"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%5"], "version": 0, "ordering": 6, "operation": "ravel"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%4", "%2"], "version": 0, "ordering": 7, "operation": "mul"},
    {"index": 8, "kind": "pure", "target": "%8", "inputs": ["%6", "%3"], "version": 0, "ordering": 8, "operation": "mul"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%7", "%8"], "version": 0, "ordering": 9, "operation": "add"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%9"], "version": 0, "ordering": 10, "operation": "sum"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_AXIS_REDUCTION_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [3], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [2], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [3], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [2], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [3], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [2], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["column_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["row_weights"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "primitive", "target": "%3", "inputs": ["%0"], "version": 0, "ordering": 3, "operation": "sum:axis:0"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%0"], "version": 0, "ordering": 4, "operation": "mean:axis:-1"},
    {"index": 5, "kind": "pure", "target": "%5", "inputs": ["%3", "%1"], "version": 0, "ordering": 5, "operation": "mul"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%4", "%2"], "version": 0, "ordering": 6, "operation": "mul"},
    {"index": 7, "kind": "primitive", "target": "%7", "inputs": ["%5"], "version": 0, "ordering": 7, "operation": "sum"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%6"], "version": 0, "ordering": 8, "operation": "sum"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%7", "%8"], "version": 0, "ordering": 9, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [4], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [6], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [6], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [6], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["source"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0"], "version": 0, "ordering": 2, "operation": "index_map:s2,s0,s2,c-1.5,s3,s1"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%2", "%1"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%3"], "version": 0, "ordering": 4, "operation": "sum"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [3], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [2], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [3], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [2], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [3], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [2], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10},
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11},
    {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12},
    {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["column_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["row_weights"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["all_weight"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%0"], "version": 0, "ordering": 4, "operation": "prod:axis:0"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%0"], "version": 0, "ordering": 5, "operation": "prod:axis:-1"},
    {"index": 6, "kind": "primitive", "target": "%6", "inputs": ["%0"], "version": 0, "ordering": 6, "operation": "prod"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%4", "%1"], "version": 0, "ordering": 7, "operation": "mul"},
    {"index": 8, "kind": "pure", "target": "%8", "inputs": ["%5", "%2"], "version": 0, "ordering": 8, "operation": "mul"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%6", "%3"], "version": 0, "ordering": 9, "operation": "mul"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%7"], "version": 0, "ordering": 10, "operation": "sum"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%8"], "version": 0, "ordering": 11, "operation": "sum"},
    {"index": 12, "kind": "pure", "target": "%12", "inputs": ["%10", "%11"], "version": 0, "ordering": 12, "operation": "add"},
    {"index": 13, "kind": "pure", "target": "%13", "inputs": ["%12", "%9"], "version": 0, "ordering": 13, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [4], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["source"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "prod"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [3], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [2], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [3], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [2], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [3], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [2], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10},
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11},
    {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12},
    {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["column_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["row_weights"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["all_weight"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%0"], "version": 0, "ordering": 4, "operation": "var:axis:0"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%0"], "version": 0, "ordering": 5, "operation": "std:axis:-1"},
    {"index": 6, "kind": "primitive", "target": "%6", "inputs": ["%0"], "version": 0, "ordering": 6, "operation": "var"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%4", "%1"], "version": 0, "ordering": 7, "operation": "mul"},
    {"index": 8, "kind": "pure", "target": "%8", "inputs": ["%5", "%2"], "version": 0, "ordering": 8, "operation": "mul"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%6", "%3"], "version": 0, "ordering": 9, "operation": "mul"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%7"], "version": 0, "ordering": 10, "operation": "sum"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%8"], "version": 0, "ordering": 11, "operation": "sum"},
    {"index": 12, "kind": "pure", "target": "%12", "inputs": ["%10", "%11"], "version": 0, "ordering": 12, "operation": "add"},
    {"index": 13, "kind": "pure", "target": "%13", "inputs": ["%12", "%9"], "version": 0, "ordering": 13, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STD_ZERO_VARIANCE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["source"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "std"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [3], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [2], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [2], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [3], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [3], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [2], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10},
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11},
    {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12},
    {"name": "%13", "producer": 13, "version": 0, "shape": [2], "dtype": "float64", "effect": 13},
    {"name": "%14", "producer": 14, "version": 0, "shape": [3], "dtype": "float64", "effect": 14},
    {"name": "%15", "producer": 15, "version": 0, "shape": [3], "dtype": "float64", "effect": 15},
    {"name": "%16", "producer": 16, "version": 0, "shape": [2], "dtype": "float64", "effect": 16},
    {"name": "%17", "producer": 17, "version": 0, "shape": [], "dtype": "float64", "effect": 17},
    {"name": "%18", "producer": 18, "version": 0, "shape": [], "dtype": "float64", "effect": 18},
    {"name": "%19", "producer": 19, "version": 0, "shape": [], "dtype": "float64", "effect": 19},
    {"name": "%20", "producer": 20, "version": 0, "shape": [2], "dtype": "float64", "effect": 20},
    {"name": "%21", "producer": 21, "version": 0, "shape": [3], "dtype": "float64", "effect": 21},
    {"name": "%22", "producer": 22, "version": 0, "shape": [], "dtype": "float64", "effect": 22},
    {"name": "%23", "producer": 23, "version": 0, "shape": [], "dtype": "float64", "effect": 23},
    {"name": "%24", "producer": 24, "version": 0, "shape": [], "dtype": "float64", "effect": 24},
    {"name": "%25", "producer": 25, "version": 0, "shape": [], "dtype": "float64", "effect": 25},
    {"name": "%26", "producer": 26, "version": 0, "shape": [], "dtype": "float64", "effect": 26},
    {"name": "%27", "producer": 27, "version": 0, "shape": [], "dtype": "float64", "effect": 27},
    {"name": "%28", "producer": 28, "version": 0, "shape": [], "dtype": "float64", "effect": 28},
    {"name": "%29", "producer": 29, "version": 0, "shape": [], "dtype": "float64", "effect": 29},
    {"name": "%30", "producer": 30, "version": 0, "shape": [], "dtype": "float64", "effect": 30},
    {"name": "%31", "producer": 31, "version": 0, "shape": [], "dtype": "float64", "effect": 31}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["max_column_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["min_row_weights"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["max_all_weight"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["min_all_weight"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["median_weight"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["quantile_row_weights"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["percentile_column_weights"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%0"], "version": 0, "ordering": 8, "operation": "max:axis:0"},
    {"index": 9, "kind": "primitive", "target": "%9", "inputs": ["%0"], "version": 0, "ordering": 9, "operation": "min:axis:-1"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%0"], "version": 0, "ordering": 10, "operation": "max"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%0"], "version": 0, "ordering": 11, "operation": "min"},
    {"index": 12, "kind": "primitive", "target": "%12", "inputs": ["%0"], "version": 0, "ordering": 12, "operation": "median"},
    {"index": 13, "kind": "primitive", "target": "%13", "inputs": ["%0"], "version": 0, "ordering": 13, "operation": "quantile:axis:1:q:0.25"},
    {"index": 14, "kind": "primitive", "target": "%14", "inputs": ["%0"], "version": 0, "ordering": 14, "operation": "percentile:axis:0:q:75.0"},
    {"index": 15, "kind": "pure", "target": "%15", "inputs": ["%8", "%1"], "version": 0, "ordering": 15, "operation": "mul"},
    {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%9", "%2"], "version": 0, "ordering": 16, "operation": "mul"},
    {"index": 17, "kind": "pure", "target": "%17", "inputs": ["%10", "%3"], "version": 0, "ordering": 17, "operation": "mul"},
    {"index": 18, "kind": "pure", "target": "%18", "inputs": ["%11", "%4"], "version": 0, "ordering": 18, "operation": "mul"},
    {"index": 19, "kind": "pure", "target": "%19", "inputs": ["%12", "%5"], "version": 0, "ordering": 19, "operation": "mul"},
    {"index": 20, "kind": "pure", "target": "%20", "inputs": ["%13", "%6"], "version": 0, "ordering": 20, "operation": "mul"},
    {"index": 21, "kind": "pure", "target": "%21", "inputs": ["%14", "%7"], "version": 0, "ordering": 21, "operation": "mul"},
    {"index": 22, "kind": "primitive", "target": "%22", "inputs": ["%15"], "version": 0, "ordering": 22, "operation": "sum"},
    {"index": 23, "kind": "primitive", "target": "%23", "inputs": ["%16"], "version": 0, "ordering": 23, "operation": "sum"},
    {"index": 24, "kind": "primitive", "target": "%24", "inputs": ["%20"], "version": 0, "ordering": 24, "operation": "sum"},
    {"index": 25, "kind": "primitive", "target": "%25", "inputs": ["%21"], "version": 0, "ordering": 25, "operation": "sum"},
    {"index": 26, "kind": "pure", "target": "%26", "inputs": ["%22", "%23"], "version": 0, "ordering": 26, "operation": "add"},
    {"index": 27, "kind": "pure", "target": "%27", "inputs": ["%26", "%17"], "version": 0, "ordering": 27, "operation": "add"},
    {"index": 28, "kind": "pure", "target": "%28", "inputs": ["%27", "%18"], "version": 0, "ordering": 28, "operation": "add"},
    {"index": 29, "kind": "pure", "target": "%29", "inputs": ["%28", "%19"], "version": 0, "ordering": 29, "operation": "add"},
    {"index": 30, "kind": "pure", "target": "%30", "inputs": ["%29", "%24"], "version": 0, "ordering": 30, "operation": "add"},
    {"index": 31, "kind": "pure", "target": "%31", "inputs": ["%30", "%25"], "version": 0, "ordering": 31, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const ORDER_STATISTIC_TIE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["source"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "max"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [2], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [2], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [6], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [2], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["row_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["all_weight"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "primitive", "target": "%3", "inputs": ["%0"], "version": 0, "ordering": 3, "operation": "trapezoid:axis:1:x:0,0.25,1.0"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%0"], "version": 0, "ordering": 4, "operation": "ravel"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%4"], "version": 0, "ordering": 5, "operation": "trapezoid:axis:0:dx:0.25"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%3", "%1"], "version": 0, "ordering": 6, "operation": "mul"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%5", "%2"], "version": 0, "ordering": 7, "operation": "mul"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%6"], "version": 0, "ordering": 8, "operation": "sum"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%8", "%7"], "version": 0, "ordering": 9, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [2, 3], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [2], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [2], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [6], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [2], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["matrix"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["row_weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["all_weight"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "primitive", "target": "%3", "inputs": ["%0"], "version": 0, "ordering": 3, "operation": "trapezoid:axis:1:xfull:0,0.25,1.0,0,0.5,1.5"},
    {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%0"], "version": 0, "ordering": 4, "operation": "ravel"},
    {"index": 5, "kind": "primitive", "target": "%5", "inputs": ["%4"], "version": 0, "ordering": 5, "operation": "trapezoid:axis:0:dx:0.25"},
    {"index": 6, "kind": "pure", "target": "%6", "inputs": ["%3", "%1"], "version": 0, "ordering": 6, "operation": "mul"},
    {"index": 7, "kind": "pure", "target": "%7", "inputs": ["%5", "%2"], "version": 0, "ordering": 7, "operation": "mul"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%6"], "version": 0, "ordering": 8, "operation": "sum"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%8", "%7"], "version": 0, "ordering": 9, "operation": "add"}
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
fn program_ad_registry_metadata_mirror_validates_coverage_snapshot() {
    let result = mirror_program_ad_registry_metadata(VALID_REGISTRY_COVERAGE_SNAPSHOT).unwrap();

    assert!(result.supported);
    assert_eq!(result.primitive_count, 3);
    assert_eq!(result.covered_primitives, 3);
    assert_eq!(result.family_counts.get("elementwise"), Some(&2));
    assert_eq!(result.family_counts.get("linalg"), Some(&1));
    assert_eq!(result.facet_counts.get("derivative_rule"), Some(&3));
    assert_eq!(result.facet_counts.get("lowering_metadata"), Some(&3));
    assert_eq!(result.executable_operations, vec!["det", "sin", "sqrt"]);
    assert_eq!(result.executable_operation_count, 3);
    assert_eq!(
        result.claim_boundary,
        "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion"
    );
}

#[test]
fn program_ad_registry_metadata_mirror_fails_closed_on_snapshot_drift() {
    let drifted =
        VALID_REGISTRY_COVERAGE_SNAPSHOT.replace("\"elementwise\": 2", "\"elementwise\": 3");

    assert!(mirror_program_ad_registry_metadata(&drifted)
        .unwrap_err()
        .contains("family_counts"));
    assert!(mirror_program_ad_registry_metadata("")
        .unwrap_err()
        .contains("non-empty JSON"));
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
        "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
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
        "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
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
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
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
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
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
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_array_elementwise_broadcast_sum() {
    let inputs = [0.2_f64, -0.3, 0.5, 1.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR,
        &inputs,
    )
    .unwrap();
    let x = [inputs[0], inputs[1], inputs[2]];
    let bias = inputs[3];
    let expected_value: f64 = x.iter().map(|value| value.sin() * (value + bias)).sum();
    let expected_gradient = [
        x[0].cos() * (x[0] + bias) + x[0].sin(),
        x[1].cos() * (x[1] + bias) + x[1].sin(),
        x[2].cos() * (x[2] + bias) + x[2].sin(),
        x.iter().map(|value| value.sin()).sum(),
    ];

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.effect_count, 6);
    assert_eq!(result.supported_effect_count, 6);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec!["%0[0]", "%0[1]", "%0[2]", "%1"]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_structural_array_ops() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STRUCTURAL_ARRAY_PROGRAM_AD_IR,
        &[2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    let expected_gradient = [
        15.0,
        20.0,
        2.0 / 6.0,
        5.0 / 6.0,
        2.0 / 6.0,
        5.0 / 6.0,
        2.0 / 6.0,
        5.0 / 6.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(130.0));
    assert_eq!(
        result.parameter_targets,
        vec!["%0[0]", "%0[1]", "%1[0]", "%1[1]", "%1[2]", "%1[3]", "%1[4]", "%1[5]"]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 8);
    assert_eq!(result.supported_effect_count, 8);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_structural_assembly_ops() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR,
        &[
            2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
        ],
    )
    .unwrap();

    let expected_gradient = [
        11.0, 32.0, 23.0, 44.0, 2.0, 5.0, 7.0, 11.0, 2.0, 7.0, 5.0, 11.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(827.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%1[0]", "%1[1]", "%2[0]", "%2[1]", "%2[2]", "%2[3]", "%3[0]",
            "%3[1]", "%3[2]", "%3[3]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 11);
    assert_eq!(result.supported_effect_count, 11);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_assembly_without_axis_metadata() {
    let missing_axis = STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR.replace(
        "\"operation\": \"concatenate:axis:0\"",
        "\"operation\": \"concatenate\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_axis,
        &[
            2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
        ],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static axis metadata")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_axis_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_AXIS_REDUCTION_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
    )
    .unwrap();

    let expected_gradient = [
        10.0 + 7.0 / 3.0,
        20.0 + 7.0 / 3.0,
        30.0 + 7.0 / 3.0,
        10.0 + 11.0 / 3.0,
        20.0 + 11.0 / 3.0,
        30.0 + 11.0 / 3.0,
        5.0,
        7.0,
        9.0,
        2.0,
        5.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(529.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_shaped_reduction_without_axis_metadata() {
    let missing_axis = STATIC_AXIS_REDUCTION_PROGRAM_AD_IR
        .replace("\"operation\": \"sum:axis:0\"", "\"operation\": \"sum\"");
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_axis,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static axis metadata")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_source_map_indexing() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    let expected_gradient = [20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(400.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%1[0]", "%1[1]", "%1[2]", "%1[3]", "%1[4]",
            "%1[5]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 5);
    assert_eq!(result.supported_effect_count, 5);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_source_map_inert_source_alias_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"assignment_binding\", \"target\": \"source:2\", \"kind\": \"alias_analysis\", \"version\": 0}, {\"source\": \"expr:2:np.take(source,_[2,_0,_2])\", \"target\": \"name:gathered\", \"kind\": \"expression_rebinding_alias\", \"version\": 1}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(400.0));
    assert_eq!(result.effect_count, 5);
    assert_eq!(result.supported_effect_count, 5);
    assert_eq!(
        result.gradient,
        vec![20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0]
    );
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_malformed_alias_analysis_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"dynamic_binding\", \"target\": \"source:2\", \"kind\": \"alias_analysis\", \"version\": 0}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_malformed_expression_alias_metadata() {
    let alias_ir = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"alias_edges\": []",
        "\"alias_edges\": [{\"source\": \"expr:2:np.take(source,_[2,_0,_2])\", \"target\": \"slot:gathered\", \"kind\": \"expression_rebinding_alias\", \"version\": 0}]",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &alias_ir,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("non-view alias-bearing"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_source_map_without_metadata() {
    let missing_map = STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        "\"operation\": \"index_map:s2,s0,s2,c-1.5,s3,s1\"",
        "\"operation\": \"index_map\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &missing_map,
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires static source-map metadata")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_product_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.0, 2.0, 3.0, -0.25, 0.1,
        ],
    )
    .unwrap();

    let expected_gradient = [
        92.0, 40.0, 42.0, 11.0, 6.4, 13.0, 4.0, 10.0, 18.0, 6.0, 120.0, 720.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(88.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 14);
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_single_zero_product() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        &[0.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(0.0));
    assert_eq!(result.gradient, vec![24.0, 0.0, 0.0, 0.0]);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_multi_zero_product() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        &[0.0, 2.0, 0.0, 4.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("prod gradient supports at most one zero input")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_variance_std_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25,
        ],
    )
    .unwrap();

    let sqrt_14 = 14.0_f64.sqrt();
    let sqrt_6 = 6.0_f64.sqrt();
    let expected_value = 455.0_f64 / 144.0 + 0.5 * (sqrt_14 - sqrt_6);
    let expected_gradient = [
        -0.5 - 2.0 / sqrt_14 - 2.0 / 9.0,
        1.875 - 0.5 / sqrt_14 - 5.0 / 36.0,
        -3.0 + 2.5 / sqrt_14 + 1.0 / 36.0,
        0.5 + 0.75 / sqrt_6 - 1.0 / 18.0,
        -1.875 + 1.0 / 9.0,
        3.0 - 0.75 / sqrt_6 + 5.0 / 18.0,
        1.0,
        2.25,
        2.25,
        sqrt_14 / 3.0,
        2.0 * sqrt_6 / 3.0,
        35.0 / 9.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 14);
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_corrected_variance_std_reductions() {
    let ir = STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR
        .replace(
            "\"operation\": \"var:axis:0\"",
            "\"operation\": \"var:axis:0:ddof:1\"",
        )
        .replace(
            "\"operation\": \"std:axis:-1\"",
            "\"operation\": \"std:axis:-1:correction:1\"",
        )
        .replace("\"operation\": \"var\"", "\"operation\": \"var:ddof:2\"");
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &ir,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.0, 2.0, 1.25, -0.75, 0.6,
        ],
    )
    .unwrap();

    let row_std = (7.0_f64 / 3.0).sqrt();
    let expected_value = 7.5 + 1.25 * row_std;
    let expected_gradient = [
        -1.8 - 5.0 / (6.0 * row_std),
        2.5 - 5.0 / (24.0 * row_std),
        -5.9 + 25.0 / (24.0 * row_std),
        1.175,
        -2.6,
        6.625,
        2.0,
        4.5,
        4.5,
        row_std,
        2.0,
        35.0 / 6.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_degenerate_moment_correction() {
    let ir = STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
        "\"operation\": \"std:axis:-1\"",
        "\"operation\": \"std:axis:-1:ddof:3\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &ir,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25,
        ],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("correction must be less than reduction group size")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_zero_variance_std() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STD_ZERO_VARIANCE_PROGRAM_AD_IR,
        &[2.0, 2.0, 2.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("std gradient requires positive variance")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_order_statistic_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR,
        &[
            3.0, -2.0, 0.5, 1.0, -1.5, 2.0, 0.7, -1.3, 0.25, 1.1, -0.4, 0.8, -0.6, 0.9, 1.2, -0.4,
            0.75, -1.1, 0.5,
        ],
    )
    .unwrap();

    let expected_gradient = [
        2.0625, 0.825, 1.175, 0.4375, -2.725, 0.625, 3.0, -1.5, 2.0, -2.0, -1.5, 3.0, -2.0, 0.75,
        -0.75, -0.25, 2.5, -1.625, 1.625,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 10.9_f64).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3", "%4", "%5", "%6[0]", "%6[1]", "%7[0]", "%7[1]", "%7[2]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 32);
    assert_eq!(result.supported_effect_count, 32);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_order_statistic_ties() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ORDER_STATISTIC_TIE_PROGRAM_AD_IR,
        &[2.0, 2.0, 1.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("strictly ordered values")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_trapezoid_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    let expected_gradient = [
        0.3125, 1.125, 0.875, -0.0625, -0.625, -0.5, 2.625, 0.4375, 1.75,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.46875).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_full_grid_trapezoid_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    let expected_gradient = [0.3125, 1.125, 0.875, -0.25, -1.0, -0.6875, 2.625, 0.5, 1.75];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.375).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_invalid_trapezoid_metadata() {
    let invalid_ir = STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR.replace(
        "\"operation\": \"trapezoid:axis:1:x:0,0.25,1.0\"",
        "\"operation\": \"trapezoid:axis:1:x:0,1\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &invalid_ir,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("x metadata length must match integration axis size")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_vector_objective() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR,
        &[0.2_f64, -0.3],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.value.is_none());
    assert!(result.gradient.is_empty());
    assert_eq!(result.effect_count, 2);
    assert_eq!(result.supported_effect_count, 2);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("requires a scalar objective")));
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
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        INERT_VIEW_ALIAS_PROGRAM_AD_IR,
        &[3.0, -2.0],
    )
    .unwrap();

    assert!(result.supported);
    assert!(result.blocked_reasons.is_empty());
    assert!((result.value.unwrap() - 13.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 2);
    assert!((result.gradient[0] - 6.0_f64).abs() <= 1.0e-12);
    assert!((result.gradient[1] - (-4.0_f64)).abs() <= 1.0e-12);
    assert_eq!(
        result.claim_boundary,
        "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    );
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_mutation_alias() {
    // Non-view alias kinds (here a mutation_version edge) can change a value's content and
    // stay outside the bounded scalar replay.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        MUTATION_ALIAS_PROGRAM_AD_IR,
        &[2.0, 3.0],
    )
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
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_TRACE_2X2_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_det_2x2() {
    // det([[a,b],[c,d]]) = a*d - b*c; cofactor gradient [d, -c, -b, a].
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_2X2_PROGRAM_AD_IR,
        &[2.0, 1.0, 1.0, 3.0],
    )
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
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_INV_2X2_ELEMENT_PROGRAM_AD_IR,
        &[2.0, 1.0, 1.0, 3.0],
    )
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
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_SOLVE_2X2_FINAL_PROGRAM_AD_IR,
        &[3.0, 1.0, 2.0, 4.0, 5.0, 6.0],
    )
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
