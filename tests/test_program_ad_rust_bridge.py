# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for Program AD Rust bridge wrappers
"""Tests for the extracted Program AD Rust bridge wrappers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad
from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADInterpreterResult,
    RustProgramADRegistryMetadataMirrorResult,
    RustProgramADValueAndGradientResult,
    interpret_program_ad_effect_ir_with_rust,
    mirror_program_ad_registry_metadata_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)


@dataclass(frozen=True)
class _IR:
    serialization: str


_ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR = """{
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
}"""

_ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR = """{
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
}"""

_STRUCTURAL_ARRAY_PROGRAM_AD_IR = """{
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
}"""

_STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR = """{
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
}"""

_STATIC_AXIS_REDUCTION_PROGRAM_AD_IR = """{
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
}"""

_STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR = """{
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
}"""

_STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR = """{
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
}"""

_PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR = """{
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
}"""

_STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR = """{
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
}"""

_STD_ZERO_VARIANCE_PROGRAM_AD_IR = """{
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
}"""

_STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR = """{
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
}"""

_ORDER_STATISTIC_TIE_PROGRAM_AD_IR = """{
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
}"""

_STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR = """{
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
}"""

_STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR = """{
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
}"""


def _install_fake_engine(monkeypatch: pytest.MonkeyPatch, engine: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", engine)


def test_value_and_gradient_bridge_is_shared_by_module_and_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and facade imports should call the same extracted Rust bridge."""

    calls: list[tuple[str, list[float]]] = []
    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        calls.append((serialization, inputs))
        return json.dumps(
            {
                "supported": True,
                "value": 3.5,
                "gradient": [1.0, -2.0],
                "parameter_targets": ["%0", "%1"],
                "effect_count": 4,
                "supported_effect_count": 4,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = replay
    _install_fake_engine(monkeypatch, fake_engine)
    program_ir = _IR(serialization="program-ir-json")

    direct = value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        np.array([0.25, -0.5], dtype=np.float64),
    )
    facade = differentiable_facade.value_and_grad_program_ad_effect_ir_with_rust(
        program_ir,
        [0.25, -0.5],
    )

    assert isinstance(direct, RustProgramADValueAndGradientResult)
    assert isinstance(facade, RustProgramADValueAndGradientResult)
    assert calls == [
        ("program-ir-json", [0.25, -0.5]),
        ("program-ir-json", [0.25, -0.5]),
    ]
    assert direct.supported == facade.supported
    assert direct.value == facade.value
    assert direct.parameter_targets == facade.parameter_targets
    assert direct.effect_count == facade.effect_count
    assert direct.supported_effect_count == facade.supported_effect_count
    assert direct.blocked_reasons == facade.blocked_reasons
    assert direct.claim_boundary == facade.claim_boundary
    np.testing.assert_allclose(direct.gradient, np.array([1.0, -2.0], dtype=np.float64))
    np.testing.assert_allclose(direct.gradient, facade.gradient)


def test_forward_interpreter_bridge_normalises_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward interpreter wrapper should parse Rust JSON into a typed result."""

    fake_engine = ModuleType("scpn_quantum_engine")

    def replay(serialization: str, inputs: list[float]) -> str:
        assert serialization == "program-ir-json"
        assert inputs == [2.0]
        return json.dumps(
            {
                "supported": True,
                "value": 9.0,
                "effect_count": 3,
                "supported_effect_count": 3,
                "blocked_reasons": [],
                "claim_boundary": "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_forward = replay
    _install_fake_engine(monkeypatch, fake_engine)

    result = interpret_program_ad_effect_ir_with_rust("program-ir-json", [2])

    assert isinstance(result, RustProgramADInterpreterResult)
    assert result.supported is True
    assert result.value == pytest.approx(9.0)
    assert result.supported_effect_count == 3
    assert "no_llvm_jit" in result.claim_boundary


def test_registry_metadata_mirror_is_shared_by_module_and_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct and facade imports should share the Rust registry metadata mirror."""

    calls: list[dict[str, object]] = []
    fake_engine = ModuleType("scpn_quantum_engine")

    def mirror(snapshot: str) -> str:
        payload = json.loads(snapshot)
        assert isinstance(payload, dict)
        assert payload["supported"] is True
        assert payload["covered_primitives"] == payload["total_primitives"] == 118
        assert payload["family_counts"] == {
            "array": 6,
            "shape": 17,
            "reduction": 11,
            "stencil": 1,
            "interpolation": 1,
            "assembly": 21,
            "signal": 2,
            "elementwise": 24,
            "selection": 11,
            "product": 7,
            "cumulative": 3,
            "linalg": 14,
        }
        assert len(cast(list[object], payload["rows"])) == 118
        calls.append(payload)
        return json.dumps(
            {
                "supported": True,
                "primitive_count": 118,
                "covered_primitives": 118,
                "family_counts": payload["family_counts"],
                "facet_counts": {
                    "derivative_rule": 118,
                    "batching_rule": 118,
                    "lowering_metadata": 118,
                    "shape_rule": 118,
                    "dtype_rule": 118,
                    "static_argument_rule": 118,
                    "nondifferentiable_policy": 118,
                    "effect": 118,
                },
                "executable_operation_count": 4,
                "executable_operations": ["det", "sin", "sqrt", "tanh"],
                "blocked_reasons": [],
                "claim_boundary": "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion",
            }
        )

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_registry_metadata_mirror = mirror
    _install_fake_engine(monkeypatch, fake_engine)

    direct = mirror_program_ad_registry_metadata_with_rust()
    facade = differentiable_facade.mirror_program_ad_registry_metadata_with_rust()

    assert isinstance(direct, RustProgramADRegistryMetadataMirrorResult)
    assert isinstance(facade, RustProgramADRegistryMetadataMirrorResult)
    assert len(calls) == 2
    assert direct.supported is True
    assert direct.primitive_count == 118
    assert direct.family_counts["elementwise"] == 24
    assert direct.facet_counts["lowering_metadata"] == 118
    assert direct.executable_operations == ("det", "sin", "sqrt", "tanh")
    assert direct.as_dict() == facade.as_dict()


def test_rust_program_ad_value_and_gradient_replay_matches_python_trace() -> None:
    """Rust Program AD scalar replay should match the emitted Python trace."""

    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return x * x + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert isinstance(rust_result, RustProgramADValueAndGradientResult)
    assert rust_result.supported is True
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert len(rust_result.parameter_targets) == values.size
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_registry_metadata_mirror_validates_python_registry() -> None:
    """Rust registry metadata mirror should validate Python registry coverage."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_registry_metadata_mirror", None))

    mirror = mirror_program_ad_registry_metadata_with_rust()

    assert mirror.supported is True, mirror.blocked_reasons
    assert mirror.primitive_count == 118
    assert mirror.covered_primitives == 118
    assert mirror.family_counts["elementwise"] == 24
    assert mirror.family_counts["linalg"] == 14
    assert mirror.facet_counts["derivative_rule"] == 118
    assert mirror.facet_counts["lowering_metadata"] == 118
    assert {"sin", "sqrt", "tanh", "det", "multi_dot", "eigh", "eigvals", "eigvalsh"} <= set(
        mirror.executable_operations
    )
    assert mirror.executable_operation_count == len(mirror.executable_operations)
    assert (
        mirror.claim_boundary
        == "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion"
    )


def test_rust_program_ad_value_and_gradient_replays_executed_branch_trace() -> None:
    """Rust Program AD replay should preserve executed scalar branch semantics."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return (x * x if x > y else y * y) + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None
    assert result.program_ir.control_regions
    assert result.program_ir.phi_nodes
    assert not result.program_ir.alias_edges
    assert "runtime_branch" in {region.kind for region in result.program_ir.control_regions}, (
        result.program_ir.control_regions
    )
    assert '"kind":"runtime_branch"' in result.program_ir.serialization
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_scalar_primitive_family_trace() -> None:
    """Rust Program AD replay should preserve emitted scalar primitive-family traces."""

    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2, 0.25, 0.1], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z, w = trace_values
        return (
            np.sqrt(x + 2.0)
            + np.tanh(y)
            + np.log1p(z)
            + np.expm1(w)
            + np.reciprocal(x + 3.0)
            + np.arcsin(0.2 * y)
            + np.arccos(0.1 * z)
            + abs(w + 1.0)
        )

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(
            Parameter("x"),
            Parameter("y"),
            Parameter("z"),
            Parameter("w"),
        ),
    )
    assert result.program_ir is not None
    assert not result.program_ir.alias_edges
    assert not result.program_ir.control_regions
    operations = {effect.operation for effect in result.program_ir.effects}
    assert {
        "sqrt",
        "tanh",
        "log1p",
        "expm1",
        "reciprocal",
        "arcsin",
        "arccos",
        "abs",
    } <= operations

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_static_take_trace() -> None:
    """Rust Program AD replay should preserve Python-emitted static gather adjoints."""

    pytest.importorskip("scpn_quantum_engine")
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    weights = np.array([1.5, -2.0, 0.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        gathered = np.take(trace_values, [2, 0, 2])
        return np.sum(gathered * weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    assert result.program_ir is not None
    assert any(
        edge.kind == "alias_analysis"
        and edge.source == "assignment_binding"
        and edge.target.startswith("source:")
        for edge in result.program_ir.alias_edges
    )
    assert any(
        edge.kind == "expression_rebinding_alias"
        and edge.source.startswith("expr:")
        and "np.take" in edge.source
        and edge.target.startswith("name:")
        for edge in result.program_ir.alias_edges
    )
    assert any(
        edge.kind == "view_alias" and edge.target.startswith("view:take:")
        for edge in result.program_ir.alias_edges
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, [-2.0, 0.0, 1.75, 0.0], atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_array_elementwise_broadcast_sum() -> None:
    """Rust Program AD replay should handle shaped elementwise array adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([0.2, -0.3, 0.5, 1.25], dtype=np.float64)
    x = values[:3]
    bias = float(values[3])
    expected_value = float(np.sum(np.sin(x) * (x + bias)))
    expected_gradient = np.concatenate(
        (
            np.cos(x) * (x + bias) + np.sin(x),
            np.array([np.sum(np.sin(x))], dtype=np.float64),
        )
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(expected_value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == ("%0[0]", "%0[1]", "%0[2]", "%1")
    assert rust_result.supported_effect_count == 6
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_structural_array_ops() -> None:
    """Rust Program AD replay should handle structural array adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
    expected_gradient = np.array(
        [
            15.0,
            20.0,
            2.0 / 6.0,
            5.0 / 6.0,
            2.0 / 6.0,
            5.0 / 6.0,
            2.0 / 6.0,
            5.0 / 6.0,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STRUCTURAL_ARRAY_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(130.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%1[3]",
        "%1[4]",
        "%1[5]",
    )
    assert rust_result.supported_effect_count == 8
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_structural_assembly_ops() -> None:
    """Rust Program AD replay should handle static concatenate/stack adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [11.0, 32.0, 23.0, 44.0, 2.0, 5.0, 7.0, 11.0, 2.0, 7.0, 5.0, 11.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(827.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%1[0]",
        "%1[1]",
        "%2[0]",
        "%2[1]",
        "%2[2]",
        "%2[3]",
        "%3[0]",
        "%3[1]",
        "%3[2]",
        "%3[3]",
    )
    assert rust_result.supported_effect_count == 11
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_static_axis_reductions() -> None:
    """Rust Program AD replay should handle static-axis sum/mean adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_AXIS_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(529.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
    )
    assert rust_result.supported_effect_count == 10
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_rejects_reduction_without_axis_metadata() -> None:
    """Shaped reduction replay should require static-axis metadata."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    missing_axis = _STATIC_AXIS_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "sum:axis:0"',
        '"operation": "sum"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        missing_axis,
        np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any("requires static axis metadata" in reason for reason in rust_result.blocked_reasons)


def test_rust_program_ad_value_and_gradient_replays_static_source_map_indexing() -> None:
    """Rust Program AD replay should scatter static source-map adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(400.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%1[3]",
        "%1[4]",
        "%1[5]",
    )
    assert rust_result.supported_effect_count == 5
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_rejects_source_map_without_metadata() -> None:
    """Static source-map replay should require explicit map metadata."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    missing_map = _STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        '"operation": "index_map:s2,s0,s2,c-1.5,s3,s1"',
        '"operation": "index_map"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        missing_map,
        np.array(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any(
        "requires static source-map metadata" in reason for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_static_product_reductions() -> None:
    """Rust Program AD replay should handle static product reductions."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.0, 2.0, 3.0, -0.25, 0.1],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [92.0, 40.0, 42.0, 11.0, 6.4, 13.0, 4.0, 10.0, 18.0, 6.0, 120.0, 720.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(88.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
    )
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_replays_single_zero_product() -> None:
    """Rust Program AD product replay should differentiate one zero input."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, [24.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_rust_program_ad_value_and_gradient_rejects_multi_zero_product() -> None:
    """Rust Program AD product replay should fail closed on multi-zero products."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        np.array([0.0, 2.0, 0.0, 4.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "prod gradient supports at most one zero input" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_static_variance_std_reductions() -> None:
    """Rust Program AD replay should handle population variance/std reductions."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25],
        dtype=np.float64,
    )
    sqrt_14 = np.sqrt(14.0)
    sqrt_6 = np.sqrt(6.0)
    expected_gradient = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(
        455.0 / 144.0 + 0.5 * (sqrt_14 - sqrt_6),
        abs=1.0e-12,
    )
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
    )
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_replays_corrected_variance_std_reductions() -> None:
    """Rust Program AD replay should support static ddof/correction metadata."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.0, 2.0, 1.25, -0.75, 0.6],
        dtype=np.float64,
    )
    row_std = np.sqrt(7.0 / 3.0)
    corrected_ir = (
        _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
            '"operation": "var:axis:0"',
            '"operation": "var:axis:0:ddof:1"',
        )
        .replace(
            '"operation": "std:axis:-1"',
            '"operation": "std:axis:-1:correction:1"',
        )
        .replace('"operation": "var"', '"operation": "var:ddof:2"')
    )
    expected_gradient = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(corrected_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(7.5 + 1.25 * row_std, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_rejects_degenerate_moment_correction() -> None:
    """Rust Program AD replay should reject non-positive variance denominators."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    invalid_ir = _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "std:axis:-1"',
        '"operation": "std:axis:-1:ddof:3"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        invalid_ir,
        np.array(
            [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any(
        "correction must be less than reduction group size" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_rejects_zero_variance_std() -> None:
    """Rust Program AD std replay should fail closed at zero variance."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STD_ZERO_VARIANCE_PROGRAM_AD_IR,
        np.array([2.0, 2.0, 2.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "std gradient requires positive variance" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_order_statistic_reductions() -> None:
    """Rust Program AD replay should route strict order-statistic adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [
            3.0,
            -2.0,
            0.5,
            1.0,
            -1.5,
            2.0,
            0.7,
            -1.3,
            0.25,
            1.1,
            -0.4,
            0.8,
            -0.6,
            0.9,
            1.2,
            -0.4,
            0.75,
            -1.1,
            0.5,
        ],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [
            2.0625,
            0.825,
            1.175,
            0.4375,
            -2.725,
            0.625,
            3.0,
            -1.5,
            2.0,
            -2.0,
            -1.5,
            3.0,
            -2.0,
            0.75,
            -0.75,
            -0.25,
            2.5,
            -1.625,
            1.625,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(10.9, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
        "%4",
        "%5",
        "%6[0]",
        "%6[1]",
        "%7[0]",
        "%7[1]",
        "%7[2]",
    )
    assert rust_result.supported_effect_count == 32


def test_rust_program_ad_value_and_gradient_rejects_order_statistic_ties() -> None:
    """Rust Program AD order-statistic replay should fail closed on ties."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ORDER_STATISTIC_TIE_PROGRAM_AD_IR,
        np.array([2.0, 2.0, 1.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any("strictly ordered values" in reason for reason in rust_result.blocked_reasons)


def test_rust_program_ad_value_and_gradient_replays_static_trapezoid_reductions() -> None:
    """Rust Program AD replay should route static-grid trapezoid adjoints."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [0.3125, 1.125, 0.875, -0.0625, -0.625, -0.5, 2.625, 0.4375, 1.75],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(5.46875, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%2",
    )
    assert rust_result.supported_effect_count == 10


def test_rust_program_ad_value_and_gradient_replays_full_grid_trapezoid_reductions() -> None:
    """Rust Program AD replay should support full-shape static trapezoid grids."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [0.3125, 1.125, 0.875, -0.25, -1.0, -0.6875, 2.625, 0.5, 1.75],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(5.375, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == 10


def test_rust_program_ad_value_and_gradient_rejects_invalid_trapezoid_metadata() -> None:
    """Rust Program AD trapezoid replay should validate static grid metadata."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    invalid_ir = _STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "trapezoid:axis:1:x:0,0.25,1.0"',
        '"operation": "trapezoid:axis:1:x:0,1"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        invalid_ir,
        np.array([1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "x metadata length must match integration axis size" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_rejects_vector_objective() -> None:
    """Rust Program AD replay should fail closed on non-scalar objectives."""

    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR,
        np.array([0.2, -0.3], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert rust_result.value is None
    assert rust_result.gradient.size == 0
    assert any("requires a scalar objective" in reason for reason in rust_result.blocked_reasons)


def test_bridge_fails_closed_when_native_extension_or_export_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional native extension failures should stay structured and fail-closed."""

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_extension = value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_extension.supported is False
    assert missing_extension.gradient.size == 0
    assert missing_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )

    fake_engine = ModuleType("scpn_quantum_engine")
    _install_fake_engine(monkeypatch, fake_engine)
    missing_export = value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_export.supported is False
    assert missing_export.blocked_reasons == (
        "scpn_quantum_engine native extension lacks Program AD value+gradient replay",
    )

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_forward_extension = interpret_program_ad_effect_ir_with_rust("{}", [0.0])
    assert missing_forward_extension.supported is False
    assert missing_forward_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    missing_mirror_extension = mirror_program_ad_registry_metadata_with_rust()
    assert missing_mirror_extension.supported is False
    assert missing_mirror_extension.primitive_count == 118
    assert missing_mirror_extension.blocked_reasons == (
        "scpn_quantum_engine native extension is not built",
    )

    fake_engine = ModuleType("scpn_quantum_engine")
    _install_fake_engine(monkeypatch, fake_engine)
    missing_mirror_export = mirror_program_ad_registry_metadata_with_rust()
    assert missing_mirror_export.supported is False
    assert missing_mirror_export.blocked_reasons == (
        "scpn_quantum_engine native extension lacks Program AD registry metadata mirror",
    )


def test_bridge_rejects_malformed_inputs_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bridge validation should reject unsafe coercions and malformed Rust payloads."""

    with pytest.raises(ValueError, match="must contain real numeric scalars"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [True])
    with pytest.raises(ValueError, match="must be a non-empty string"):
        interpret_program_ad_effect_ir_with_rust("", [0.0])

    fake_engine = ModuleType("scpn_quantum_engine")

    def invalid_json(_serialization: str, _inputs: list[float]) -> str:
        return "{"

    engine_exports = cast(Any, fake_engine)
    engine_exports.program_ad_effect_ir_interpret_forward = invalid_json
    _install_fake_engine(monkeypatch, fake_engine)
    with pytest.raises(ValueError, match="returned invalid JSON"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_json_object(_serialization: str, _inputs: list[float]) -> str:
        return "[]"

    engine_exports.program_ad_effect_ir_interpret_forward = non_json_object
    with pytest.raises(ValueError, match="payload must be a JSON object"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_text_forward(_serialization: str, _inputs: list[float]) -> int:
        return 1

    engine_exports.program_ad_effect_ir_interpret_forward = non_text_forward
    with pytest.raises(ValueError, match="must return JSON text"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def malformed_value_and_gradient(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": True,
                "value": 1.0,
                "gradient": "bad",
                "parameter_targets": [],
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": [],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = malformed_value_and_gradient
    with pytest.raises(ValueError, match="gradient must be a JSON list"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])

    def non_text_value_and_gradient(_serialization: str, _inputs: list[float]) -> int:
        return 1

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = non_text_value_and_gradient
    with pytest.raises(ValueError, match="must return JSON text"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])

    with pytest.raises(ValueError, match="must be one-dimensional"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", cast(Any, [[0.0]]))
    with pytest.raises(ValueError, match="must contain finite values"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [float("nan")])
    with pytest.raises(ValueError, match="must be a rectangular numeric array"):
        value_and_grad_program_ad_effect_ir_with_rust(
            "{}",
            cast(Any, [[1.0], [1.0, 2.0]]),
        )

    def bad_forward_count(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "effect_count": True,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_forward = bad_forward_count
    with pytest.raises(ValueError, match="effect_count must be an integer"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def bad_forward_boundary(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": 3,
            }
        )

    engine_exports.program_ad_effect_ir_interpret_forward = bad_forward_boundary
    with pytest.raises(ValueError, match="claim_boundary must be a string"):
        interpret_program_ad_effect_ir_with_rust("{}", [0.0])

    def bad_value_and_gradient_targets(_serialization: str, _inputs: list[float]) -> str:
        return json.dumps(
            {
                "supported": False,
                "value": None,
                "gradient": [],
                "parameter_targets": "bad",
                "effect_count": 1,
                "supported_effect_count": 0,
                "blocked_reasons": ["blocked"],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_effect_ir_interpret_value_and_gradient = (
        bad_value_and_gradient_targets
    )
    with pytest.raises(ValueError, match="parameter target must be a list"):
        value_and_grad_program_ad_effect_ir_with_rust("{}", [0.0])

    def malformed_registry_mirror(_snapshot: str) -> str:
        return json.dumps(
            {
                "supported": True,
                "primitive_count": 118,
                "covered_primitives": 118,
                "family_counts": {"elementwise": 24},
                "facet_counts": [],
                "executable_operation_count": 1,
                "executable_operations": ["sin"],
                "blocked_reasons": [],
                "claim_boundary": "bounded",
            }
        )

    engine_exports.program_ad_registry_metadata_mirror = malformed_registry_mirror
    with pytest.raises(ValueError, match="facet_counts must be a JSON object"):
        mirror_program_ad_registry_metadata_with_rust()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "supported": "yes",
                "value": None,
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported flag must be boolean",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": -1,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "counts must be non-negative",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 2,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported count exceeds effect count",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("",),
                "claim_boundary": "bounded",
            },
            "blocked reasons must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported state is inconsistent",
        ),
        (
            {
                "supported": False,
                "value": None,
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "",
            },
            "claim boundary must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": float("inf"),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be finite",
        ),
    ],
)
def test_interpreter_result_rejects_invalid_states(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Interpreter result container should reject inconsistent states."""

    with pytest.raises(ValueError, match=match):
        RustProgramADInterpreterResult(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "supported": "yes",
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported flag must be boolean",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([[0.0]], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "gradient must be one-dimensional",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 0,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "target count must match gradient",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": -1,
                "supported_effect_count": 0,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "counts must be non-negative",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 2,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "supported count exceeds effect count",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "bounded",
            },
            "targets must be non-empty",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("",),
                "claim_boundary": "bounded",
            },
            "blocked reasons must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": None,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "supported state is inconsistent",
        ),
        (
            {
                "supported": False,
                "value": None,
                "gradient": np.array([], dtype=np.float64),
                "parameter_targets": (),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": ("blocked",),
                "claim_boundary": "",
            },
            "claim boundary must be non-empty",
        ),
        (
            {
                "supported": True,
                "value": np.array([1.0], dtype=np.float64),
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be a real numeric scalar",
        ),
        (
            {
                "supported": True,
                "value": True,
                "gradient": np.array([0.0], dtype=np.float64),
                "parameter_targets": ("%0",),
                "effect_count": 1,
                "supported_effect_count": 1,
                "blocked_reasons": (),
                "claim_boundary": "bounded",
            },
            "value must be a real numeric scalar",
        ),
    ],
)
def test_value_and_gradient_result_rejects_invalid_states(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Value+gradient result container should reject inconsistent states."""

    with pytest.raises(ValueError, match=match):
        RustProgramADValueAndGradientResult(**kwargs)  # type: ignore[arg-type]


def _objective_reshape_sumsq(values: Any) -> Any:
    return np.sum(np.reshape(values, (2, 2)) ** 2)


def _objective_mutation(values: Any) -> Any:
    work = values * 1.0
    work[0] = work[0] * work[1]
    return np.sum(work)


def test_bridge_replays_inert_view_alias_program_with_real_engine() -> None:
    """With the real engine, a reshape view-alias program replays bit-exact."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_reshape_sumsq, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine does not yet replay view aliases: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_reshape_sumsq, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)
    assert rust.claim_boundary.endswith(
        "view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_bridge_fails_closed_on_mutation_alias_with_real_engine() -> None:
    """A mutation-aliasing program stays outside the bounded Rust replay."""

    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([2.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_mutation, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("non-view alias" in reason for reason in rust.blocked_reasons)


def _objective_trace_2x2(values: Any) -> Any:
    return np.trace(np.reshape(values, (2, 2)))


def _objective_det_2x2(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (2, 2)))


def test_bridge_replays_linalg_trace_with_real_engine() -> None:
    """With the real engine, a 2x2 trace program replays bit-exact."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_trace_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:trace replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_trace_2x2, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)


def test_bridge_replays_linalg_det_2x2_with_real_engine() -> None:
    """With the real engine, a 2x2 determinant program replays within float64 tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_2x2, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)
    assert rust.claim_boundary.endswith(
        "elementwise_structural_array_static_source_map_static_reductions_and_static_linalg_primitives_value_and_gradient_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def _objective_inv_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.inv(np.reshape(values, (2, 2))))


def _objective_solve_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:]))


def _objective_solve_2x2_indexed(values: Any) -> Any:
    return np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:])[0]


def test_bridge_replays_linalg_inverse_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 inverse program replays within tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_inv_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:inv replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_replays_linalg_solve_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 linear solve replays within tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:solve replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_solve_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_fails_closed_on_indexed_multi_output_linalg_with_real_engine() -> None:
    """A bare indexed solve component stays outside the bounded Rust replay."""

    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 6.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_indexed, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("indexed multi-output linalg" in reason for reason in rust.blocked_reasons)


def _objective_det_3x3(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (3, 3)))


def test_bridge_replays_linalg_det_3x3_with_real_engine() -> None:
    """With the real engine, a 3x3 determinant program replays within float64 tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_3x3, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_3x3, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def _objective_inv_3x3_sum(values: Any) -> Any:
    return np.sum(np.linalg.inv(np.reshape(values, (3, 3))))


def _objective_solve_3x3_sum(values: Any) -> Any:
    return np.sum(np.linalg.solve(np.reshape(values[:9], (3, 3)), values[9:]))


def test_bridge_replays_linalg_inverse_3x3_with_real_engine() -> None:
    """With the real engine, a reduced 3x3 inverse program replays bit-exact."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_inv_3x3_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:inv:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_3x3_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_replays_linalg_solve_3x3_with_real_engine() -> None:
    """With the real engine, a reduced 3x3 linear solve replays within tolerance."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array(
        [2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0, 1.0, 2.0, 3.0], dtype=np.float64
    )
    result = whole_program_value_and_grad(_objective_solve_3x3_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:solve:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_solve_3x3_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def _diagonally_dominant(n: int, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    return (np.eye(n) * (n + 3.0) + rng.random((n, n))).ravel()


def _objective_det_nxn(values: Any) -> Any:
    n = int(round(float(np.sqrt(values.size))))
    return np.linalg.det(np.reshape(values, (n, n)))


def _objective_inv_nxn_sum(values: Any) -> Any:
    n = int(round(float(np.sqrt(values.size))))
    return np.sum(np.linalg.inv(np.reshape(values, (n, n))))


def test_bridge_replays_general_linalg_det_4x4_with_real_engine() -> None:
    """With the real engine, a 4x4 determinant replays via the general LU path."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = _diagonally_dominant(4, seed=11)
    result = whole_program_value_and_grad(_objective_det_nxn, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks general linalg:det: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_nxn, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-9, atol=1.0e-9)


def test_bridge_replays_general_linalg_inverse_5x5_with_real_engine() -> None:
    """With the real engine, a reduced 5x5 inverse replays via the general LU path."""

    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = _diagonally_dominant(5, seed=23)
    result = whole_program_value_and_grad(_objective_inv_nxn_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks general linalg:inv: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_nxn_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-9, atol=1.0e-9)
