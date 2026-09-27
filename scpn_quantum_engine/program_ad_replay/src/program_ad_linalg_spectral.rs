// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral linalg replay helpers

//! Bounded spectral linear-algebra replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:eigvalsh:<index>`,
//! `linalg:eigvals:2x2:<index>`,
//! `linalg:eig:eigenvalue:2x2:<index>`,
//! `linalg:eig:eigenvector:2x2:<column>:<row>`,
//! `linalg:eigh:eigenvalue:2x2:<UPLO>:<index>`, or
//! `linalg:eigh:eigenvector:2x2:<UPLO>:<column>:<row>` SSA node per scalar
//! spectral output. This module owns the Rust-side 2x2 spectral contracts so
//! the main Program AD IR evaluator stays a dispatcher. The helper deliberately
//! fails closed for non-2x2 matrices, non-symmetric Hermitian inputs,
//! zero-offdiagonal `eigh` eigenvector outputs, complex or repeated spectra,
//! ill-conditioned `eig` eigenbases, and malformed output metadata because
//! those cases need broader spectral policy before they can be promoted.

use crate::program_ad_ir::reserve_replay_buffer;
use crate::program_ad_lifecycle::replay_checkpoint;

const DISTINCT_EIGENVALUE_TOLERANCE: f64 = 1.0e-10;
const REAL_SPECTRUM_TOLERANCE: f64 = 1.0e-12;
const SYMMETRY_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct Eigvalsh2x2 {
    output_index: usize,
    diagonal: [f64; 2],
    off_diagonal: f64,
    eigenvalues: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
struct Eigvals2x2 {
    output_index: usize,
    sign: f64,
    values: [f64; 4],
    eigenvalues: [f64; 2],
    gap: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EigOutput {
    Eigenvalue { index: usize },
    Eigenvector { column: usize, row: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct Eig2x2 {
    output: EigOutput,
    eigenvalues: [f64; 2],
    right_eigenvectors: [[f64; 2]; 2],
    left_eigenvector_rows: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EighOutput {
    Eigenvalue { index: usize },
    Eigenvector { column: usize, row: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct Eigh2x2 {
    output: EighOutput,
    eigenvalues: [f64; 2],
    eigenvectors: [[f64; 2]; 2],
}

/// Return whether an operation label belongs to bounded `np.linalg.eigvalsh` replay.
pub(crate) fn is_eigvalsh_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigvalsh:")
}

/// Return whether an operation label belongs to bounded `np.linalg.eigvals` replay.
pub(crate) fn is_eigvals_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigvals:")
}

/// Return whether an operation label belongs to bounded `np.linalg.eig` replay.
pub(crate) fn is_eig_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eig:")
}

/// Return whether an operation label belongs to bounded `np.linalg.eigh` replay.
pub(crate) fn is_eigh_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigh:")
}

/// Evaluate one scalar eigenvalue from a row-major 2x2 symmetric Program AD node.
pub(crate) fn eigvalsh_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigvalsh_2x2(effect_index, operation, input_values)?;
    Ok(metadata.eigenvalues[metadata.output_index])
}

/// Evaluate one scalar eigenvalue from a row-major 2x2 real-simple Program AD node.
pub(crate) fn eigvals_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigvals_2x2(effect_index, operation, input_values)?;
    Ok(metadata.eigenvalues[metadata.output_index])
}

/// Evaluate one scalar output from a row-major 2x2 real-simple `eig` node.
pub(crate) fn eig_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eig_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EigOutput::Eigenvalue { index } => Ok(metadata.eigenvalues[index]),
        EigOutput::Eigenvector { column, row } => Ok(metadata.right_eigenvectors[row][column]),
    }
}

/// Evaluate one scalar output from a row-major 2x2 symmetric `eigh` Program AD node.
pub(crate) fn eigh_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigh_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EighOutput::Eigenvalue { index } => Ok(metadata.eigenvalues[index]),
        EighOutput::Eigenvector { column, row } => Ok(metadata.eigenvectors[row][column]),
    }
}

/// Return local reverse contributions for one scalar 2x2 `eigvalsh` output node.
pub(crate) fn eigvalsh_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigvalsh cotangent must be finite"
        ));
    }
    let metadata = parse_eigvalsh_2x2(effect_index, operation, input_values)?;
    let outer = eigenvector_outer(&metadata)?;
    spectral_contributions(
        effect_index,
        outer.map(|component| output_cotangent * component),
    )
}

/// Return local reverse contributions for one scalar 2x2 `eigvals` output node.
pub(crate) fn eigvals_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigvals cotangent must be finite"
        ));
    }
    let metadata = parse_eigvals_2x2(effect_index, operation, input_values)?;
    let [a, b, c, d] = metadata.values;
    let diagonal_delta = a - d;
    let sign = metadata.sign;
    spectral_contributions(
        effect_index,
        [
            output_cotangent * (0.5 + sign * diagonal_delta / (2.0 * metadata.gap)),
            output_cotangent * sign * c / metadata.gap,
            output_cotangent * sign * b / metadata.gap,
            output_cotangent * (0.5 - sign * diagonal_delta / (2.0 * metadata.gap)),
        ],
    )
}

/// Return local reverse contributions for one scalar 2x2 `eig` output node.
pub(crate) fn eig_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eig cotangent must be finite"
        ));
    }
    let metadata = parse_eig_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EigOutput::Eigenvalue { index } => {
            let left = metadata.left_eigenvector_rows[index];
            let right = [
                metadata.right_eigenvectors[0][index],
                metadata.right_eigenvectors[1][index],
            ];
            spectral_contributions(
                effect_index,
                [
                    output_cotangent * left[0] * right[0],
                    output_cotangent * left[0] * right[1],
                    output_cotangent * left[1] * right[0],
                    output_cotangent * left[1] * right[1],
                ],
            )
        }
        EigOutput::Eigenvector { column, row } => {
            let mut contributions = reserve_replay_buffer(4)?;
            for basis_row in 0..2 {
                for basis_column in 0..2 {
                    replay_checkpoint()?;
                    contributions.push(
                        output_cotangent
                            * eig_eigenvector_jvp_entry(
                                &metadata,
                                column,
                                row,
                                basis_row,
                                basis_column,
                            ),
                    );
                }
            }
            validate_spectral_contributions(effect_index, &contributions)?;
            Ok(contributions)
        }
    }
}

/// Return local reverse contributions for one scalar 2x2 `eigh` output node.
pub(crate) fn eigh_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigh cotangent must be finite"
        ));
    }
    let metadata = parse_eigh_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EighOutput::Eigenvalue { index } => {
            let vector = [
                metadata.eigenvectors[0][index],
                metadata.eigenvectors[1][index],
            ];
            spectral_contributions(
                effect_index,
                vector_outer(vector).map(|component| output_cotangent * component),
            )
        }
        EighOutput::Eigenvector { column, row } => {
            let other = 1 - column;
            let lambda_delta = metadata.eigenvalues[column] - metadata.eigenvalues[other];
            let other_vector = [
                metadata.eigenvectors[0][other],
                metadata.eigenvectors[1][other],
            ];
            let column_vector = [
                metadata.eigenvectors[0][column],
                metadata.eigenvectors[1][column],
            ];
            let scale = output_cotangent * other_vector[row] / lambda_delta;
            let raw = [
                scale * other_vector[0] * column_vector[0],
                scale * other_vector[0] * column_vector[1],
                scale * other_vector[1] * column_vector[0],
                scale * other_vector[1] * column_vector[1],
            ];
            spectral_contributions(
                effect_index,
                [
                    raw[0],
                    0.5 * (raw[1] + raw[2]),
                    0.5 * (raw[2] + raw[1]),
                    raw[3],
                ],
            )
        }
    }
}

fn spectral_contributions(effect_index: usize, values: [f64; 4]) -> Result<Vec<f64>, String> {
    validate_spectral_contributions(effect_index, &values)?;
    let mut contributions = reserve_replay_buffer(4)?;
    for value in values {
        replay_checkpoint()?;
        contributions.push(value);
    }
    replay_checkpoint()?;
    Ok(contributions)
}

fn validate_spectral_contributions(effect_index: usize, values: &[f64]) -> Result<(), String> {
    replay_checkpoint()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} spectral cotangent entries must be finite"
        ));
    }
    Ok(())
}

include!("program_ad_linalg_spectral/matrices.rs");
include!("program_ad_linalg_spectral/metadata.rs");
include!("program_ad_linalg_spectral/algebra.rs");

include!("program_ad_linalg_spectral/workspace.rs");
