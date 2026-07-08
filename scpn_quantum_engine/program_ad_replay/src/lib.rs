// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded program-AD effect-IR replay crate root

//! Bounded, bit-exact program-AD effect-IR value+gradient replay.
//!
//! This crate holds the pure replay the engine and the Studio WASM kernel share.
//! Its dependencies (serde, serde_json, nalgebra) all compile to
//! `wasm32-unknown-unknown`; the only Python coupling is a handful of thin
//! `#[pyfunction]` wrappers gated behind the optional `pyo3` feature, which the
//! engine crate enables and the WASM build omits. Splitting the replay out this
//! way keeps a single source of truth: the browser recompute runs the same code
//! the engine does, not a fork.

pub mod program_ad_cumulative_reduction;
pub mod program_ad_interpolation_reduction;
pub mod program_ad_ir;
pub mod program_ad_linalg_array;
pub mod program_ad_linalg_diag;
pub mod program_ad_linalg_diagflat;
pub mod program_ad_linalg_matrix_power;
pub mod program_ad_linalg_pinv;
pub mod program_ad_linalg_spectral;
pub mod program_ad_linalg_svd;
pub mod program_ad_order_statistic_reduction;
pub mod program_ad_product_reduction;
pub mod program_ad_registry_mirror;
pub mod program_ad_signal_reduction;
pub mod program_ad_static_source_map;
pub mod program_ad_stencil_reduction;
pub mod program_ad_trapezoid_reduction;
pub mod program_ad_variance_reduction;
