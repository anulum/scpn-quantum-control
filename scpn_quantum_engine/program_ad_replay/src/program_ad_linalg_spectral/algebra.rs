// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral algebra helpers

fn eigvals_2x2_order_signs(a: f64, b: f64, c: f64, d: f64, scale: f64) -> [f64; 2] {
    let off_diagonal_tolerance = SYMMETRY_TOLERANCE * scale;
    let b_is_zero = b.abs() <= off_diagonal_tolerance;
    let c_is_zero = c.abs() <= off_diagonal_tolerance;
    if b_is_zero && !c_is_zero {
        if d <= a {
            [-1.0, 1.0]
        } else {
            [1.0, -1.0]
        }
    } else if a < d {
        [-1.0, 1.0]
    } else {
        [1.0, -1.0]
    }
}

fn eig_2x2_right_eigenvectors(
    effect_index: usize,
    values: [f64; 4],
    eigenvalues: [f64; 2],
    scale: f64,
) -> Result<[[f64; 2]; 2], String> {
    let [a, b, c, d] = values;
    let mut vectors = [[0.0; 2]; 2];
    for (column, eigenvalue) in eigenvalues.iter().enumerate() {
        replay_checkpoint()?;
        let mut raw = if b.abs() > SYMMETRY_TOLERANCE * scale {
            [-b, a - eigenvalue]
        } else {
            [d - eigenvalue, -c]
        };
        if squared_norm(raw) <= SYMMETRY_TOLERANCE * scale {
            raw = [d - eigenvalue, -c];
        }
        let norm = squared_norm(raw).sqrt();
        if !norm.is_finite() || norm <= SYMMETRY_TOLERANCE * scale {
            return Err(format!(
                "effect {effect_index} eig requires a well-conditioned eigenbasis"
            ));
        }
        vectors[0][column] = raw[0] / norm;
        vectors[1][column] = raw[1] / norm;
    }
    Ok(vectors)
}

fn invert_eig_2x2_basis(
    effect_index: usize,
    right_eigenvectors: [[f64; 2]; 2],
) -> Result<[[f64; 2]; 2], String> {
    let determinant = right_eigenvectors[0][0] * right_eigenvectors[1][1]
        - right_eigenvectors[0][1] * right_eigenvectors[1][0];
    if !determinant.is_finite() || determinant.abs() <= DISTINCT_EIGENVALUE_TOLERANCE {
        return Err(format!(
            "effect {effect_index} eig requires a well-conditioned eigenbasis"
        ));
    }
    Ok([
        [
            right_eigenvectors[1][1] / determinant,
            -right_eigenvectors[0][1] / determinant,
        ],
        [
            -right_eigenvectors[1][0] / determinant,
            right_eigenvectors[0][0] / determinant,
        ],
    ])
}

fn eig_eigenvector_jvp_entry(
    metadata: &Eig2x2,
    column: usize,
    row: usize,
    basis_row: usize,
    basis_column: usize,
) -> f64 {
    let source = [
        metadata.right_eigenvectors[0][column],
        metadata.right_eigenvectors[1][column],
    ];
    let mut raw = [0.0, 0.0];
    for other in 0..2 {
        if other == column {
            continue;
        }
        let other_vector = [
            metadata.right_eigenvectors[0][other],
            metadata.right_eigenvectors[1][other],
        ];
        let left = metadata.left_eigenvector_rows[other];
        let numerator = left[basis_row] * source[basis_column];
        let scale = numerator / (metadata.eigenvalues[column] - metadata.eigenvalues[other]);
        raw[0] += scale * other_vector[0];
        raw[1] += scale * other_vector[1];
    }
    let gauge_projection = source[0] * raw[0] + source[1] * raw[1];
    raw[row] - source[row] * gauge_projection
}

fn eigenvector_outer(metadata: &Eigvalsh2x2) -> Result<[f64; 4], String> {
    let [a, d] = metadata.diagonal;
    if metadata.off_diagonal.abs() <= SYMMETRY_TOLERANCE {
        return Ok(diagonal_eigenvector_outer(a, d, metadata.output_index));
    }
    let lambda = metadata.eigenvalues[metadata.output_index];
    let primary = [metadata.off_diagonal, lambda - a];
    let secondary = [lambda - d, metadata.off_diagonal];
    let raw = if squared_norm(primary) >= squared_norm(secondary) {
        primary
    } else {
        secondary
    };
    let norm = squared_norm(raw).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return Err("eigvalsh eigenvector normalization must be finite".to_owned());
    }
    let x = raw[0] / norm;
    let y = raw[1] / norm;
    Ok([x * x, x * y, y * x, y * y])
}

fn diagonal_eigenvector_outer(a: f64, d: f64, output_index: usize) -> [f64; 4] {
    let lower_is_first_axis = a <= d;
    if (output_index == 0 && lower_is_first_axis) || (output_index == 1 && !lower_is_first_axis) {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn eigh_eigenvectors_2x2(
    a: f64,
    b: f64,
    d: f64,
    eigenvalues: [f64; 2],
) -> Result<[[f64; 2]; 2], String> {
    let scale = 1.0_f64.max(a.abs()).max(b.abs()).max(d.abs());
    if b.abs() <= SYMMETRY_TOLERANCE * scale {
        return Ok(diagonal_eigenvectors(a, d));
    }
    let raw0 = if b > 0.0 && a <= d {
        [-b, a - eigenvalues[0]]
    } else {
        [b, eigenvalues[0] - a]
    };
    let raw1 = if b > 0.0 && a > d {
        [-b, a - eigenvalues[1]]
    } else {
        [b, eigenvalues[1] - a]
    };
    let column0 = normalise_eigh_vector(raw0)?;
    let column1 = normalise_eigh_vector(raw1)?;
    Ok([[column0[0], column1[0]], [column0[1], column1[1]]])
}

fn diagonal_eigenvectors(a: f64, d: f64) -> [[f64; 2]; 2] {
    if a <= d {
        [[1.0, 0.0], [0.0, 1.0]]
    } else {
        [[0.0, 1.0], [1.0, 0.0]]
    }
}

fn normalise_eigh_vector(raw: [f64; 2]) -> Result<[f64; 2], String> {
    let norm = squared_norm(raw).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return Err("eigh eigenvector normalization must be finite".to_owned());
    }
    Ok([raw[0] / norm, raw[1] / norm])
}

fn vector_outer(vector: [f64; 2]) -> [f64; 4] {
    [
        vector[0] * vector[0],
        vector[0] * vector[1],
        vector[1] * vector[0],
        vector[1] * vector[1],
    ]
}

fn squared_norm(vector: [f64; 2]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1]
}
