// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR metadata parity

//! Rust metadata parser for Python-emitted `program_ad_effect_ir.v1` payloads.
//!
//! This module mirrors the bounded Python Program AD IR schema so Rust-side
//! tooling can inspect evidence metadata, execute a narrow scalar forward
//! interpreter, and replay bounded scalar, elementwise-array, static structural,
//! static source-map indexing, static product, corrected moment,
//! order-statistic, static-grid trapezoid reductions, compact interpolation,
//! signal, stencil, and cumulative primitives, and static-linalg value+gradient
//! traces when opcode-bearing rows are present.
//! Dynamic indexing, dynamic axis, dynamic q/method, dynamic moment-correction,
//! dynamic trapezoid-grid, and zero-variance `std` boundaries are audited as
//! explicit fail-closed outcomes.
//! It does not promote LLVM lowering, JIT execution, reverse-mode compiler AD,
//! hardware execution, or performance claims.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::program_ad_cumulative_reduction::{
    cumulative_output_cotangent, cumulative_output_value, is_cumulative_operation,
};
use crate::program_ad_interpolation_reduction::{
    interpolation_output_cotangent, interpolation_output_value, is_interpolation_operation,
};
use crate::program_ad_linalg_array::{
    is_multi_dot_operation, multi_dot_output_cotangent, multi_dot_output_value,
};
use crate::program_ad_linalg_diag::{diag_output_cotangent, diag_output_value, is_diag_operation};
use crate::program_ad_linalg_diagflat::{
    diagflat_output_cotangent, diagflat_output_value, is_diagflat_operation,
};
use crate::program_ad_linalg_matrix_power::{
    is_matrix_power_operation, matrix_power_output_cotangent, matrix_power_output_value,
};
use crate::program_ad_linalg_pinv::{is_pinv_operation, pinv_output_cotangent, pinv_output_value};
use crate::program_ad_linalg_spectral::{
    eig_output_cotangent, eig_output_value, eigh_output_cotangent, eigh_output_value,
    eigvals_output_cotangent, eigvals_output_value, eigvalsh_output_cotangent,
    eigvalsh_output_value, is_eig_operation, is_eigh_operation, is_eigvals_operation,
    is_eigvalsh_operation,
};
use crate::program_ad_linalg_svd::{
    is_svdvals_operation, svdvals_output_cotangent, svdvals_output_value,
};
use crate::program_ad_order_statistic_reduction::{
    is_order_statistic_operation, order_statistic_cotangent, order_statistic_values,
};
use crate::program_ad_product_reduction::{
    product_all_cotangent, product_all_value, product_axis_cotangent, product_axis_values,
};
pub use crate::program_ad_registry_mirror::{
    mirror_program_ad_registry_metadata, ProgramADRegistryMetadataMirrorSummary,
};
use crate::program_ad_signal_reduction::{
    is_signal_operation, signal_output_cotangent, signal_output_value,
};
use crate::program_ad_static_source_map::{
    apply_static_source_map, scatter_static_source_map_cotangent,
};
use crate::program_ad_stencil_reduction::{
    is_stencil_operation, stencil_output_cotangent, stencil_output_value,
};
use crate::program_ad_trapezoid_reduction::{
    is_trapezoid_operation, trapezoid_cotangent, trapezoid_values,
};
use crate::program_ad_variance_reduction::{
    parse_moment_reduction_metadata, std_all_cotangent, std_all_value, std_axis_cotangent,
    std_axis_values, variance_all_cotangent, variance_all_value, variance_axis_cotangent,
    variance_axis_values,
};

include!("program_ad_ir/schema_parser.rs");
include!("program_ad_ir/scalar_forward.rs");
include!("program_ad_ir/value_state.rs");
include!("program_ad_ir/reverse_dispatch.rs");
include!("program_ad_ir/reverse_linalg.rs");
include!("program_ad_ir/reverse_reductions.rs");
include!("program_ad_ir/reverse_structural.rs");
include!("program_ad_ir/numeric_dispatch.rs");
include!("program_ad_ir/numeric_linalg.rs");
include!("program_ad_ir/numeric_reductions.rs");
include!("program_ad_ir/numeric_structural.rs");
include!("program_ad_ir/scalar_forward_ops.rs");
include!("program_ad_ir/bindings.rs");
