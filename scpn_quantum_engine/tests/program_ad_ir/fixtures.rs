// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR shared test fixtures

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

const LINALG_DET_4X4_DIAGONAL_PROGRAM_AD_IR: &str = r#"{"format": "program_ad_effect_ir.v1", "ssa_values": [{"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}, {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}, {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}, {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3}, {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}, {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5}, {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6}, {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7}, {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8}, {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9}, {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10}, {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11}, {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12}, {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13}, {"name": "%14", "producer": 14, "version": 0, "shape": [], "dtype": "float64", "effect": 14}, {"name": "%15", "producer": 15, "version": 0, "shape": [], "dtype": "float64", "effect": 15}, {"name": "%16", "producer": 16, "version": 0, "shape": [], "dtype": "float64", "effect": 16}], "effects": [{"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"}, {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"}, {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["c"], "version": 0, "ordering": 2, "operation": "parameter"}, {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["d"], "version": 0, "ordering": 3, "operation": "parameter"}, {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["e"], "version": 0, "ordering": 4, "operation": "parameter"}, {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["f"], "version": 0, "ordering": 5, "operation": "parameter"}, {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["g"], "version": 0, "ordering": 6, "operation": "parameter"}, {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["h"], "version": 0, "ordering": 7, "operation": "parameter"}, {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["i"], "version": 0, "ordering": 8, "operation": "parameter"}, {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["j"], "version": 0, "ordering": 9, "operation": "parameter"}, {"index": 10, "kind": "parameter", "target": "%10", "inputs": ["k"], "version": 0, "ordering": 10, "operation": "parameter"}, {"index": 11, "kind": "parameter", "target": "%11", "inputs": ["l"], "version": 0, "ordering": 11, "operation": "parameter"}, {"index": 12, "kind": "parameter", "target": "%12", "inputs": ["m"], "version": 0, "ordering": 12, "operation": "parameter"}, {"index": 13, "kind": "parameter", "target": "%13", "inputs": ["n"], "version": 0, "ordering": 13, "operation": "parameter"}, {"index": 14, "kind": "parameter", "target": "%14", "inputs": ["o"], "version": 0, "ordering": 14, "operation": "parameter"}, {"index": 15, "kind": "parameter", "target": "%15", "inputs": ["p"], "version": 0, "ordering": 15, "operation": "parameter"}, {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11", "%12", "%13", "%14", "%15"], "version": 0, "ordering": 16, "operation": "linalg:det:4x4"}], "alias_edges": [], "control_regions": [], "phi_nodes": [], "bytecode_offsets": [0]}
"#;
