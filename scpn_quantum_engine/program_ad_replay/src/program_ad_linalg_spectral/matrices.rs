// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral matrices helpers

fn parse_eigvalsh_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigvalsh2x2, String> {
    replay_checkpoint()?;
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigvalsh Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvalsh inputs must be finite"
        ));
    }
    let output_index = parse_eigvalsh_index(effect_index, operation)?;
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigvalsh Rust replay supports only 2x2 matrices"
        ));
    };
    let scale = input_values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if (b - c).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(format!(
            "effect {effect_index} eigvalsh requires a symmetric 2x2 matrix"
        ));
    }
    let off_diagonal = 0.5 * (b + c);
    let diagonal_delta = a - d;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * off_diagonal * off_diagonal).sqrt();
    let center = 0.5 * (a + d);
    let radius = 0.5 * gap;
    let eigenvalues = [center - radius, center + radius];
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvalsh output must be finite"
        ));
    }
    let eigen_scale = eigenvalues
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigvalsh gradient requires distinct eigenvalues"
        ));
    }
    Ok(Eigvalsh2x2 {
        output_index,
        diagonal: [*a, *d],
        off_diagonal,
        eigenvalues,
    })
}

fn parse_eigvals_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigvals2x2, String> {
    replay_checkpoint()?;
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigvals Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvals inputs must be finite"
        ));
    }
    let output_index = parse_eigvals_index(effect_index, operation)?;
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigvals Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let scale = values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let diagonal_delta = a - d;
    let discriminant = diagonal_delta * diagonal_delta + 4.0 * b * c;
    let discriminant_tolerance = REAL_SPECTRUM_TOLERANCE * scale * scale;
    if discriminant < -discriminant_tolerance {
        return Err(format!(
            "effect {effect_index} eigvals requires real distinct eigenvalues"
        ));
    }
    let gap = discriminant.max(0.0).sqrt();
    let center = 0.5 * (a + d);
    let lower = center - 0.5 * gap;
    let upper = center + 0.5 * gap;
    let eigen_scale = 1.0_f64.max(lower.abs()).max(upper.abs());
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigvals requires real distinct eigenvalues"
        ));
    }
    let ordered_signs = eigvals_2x2_order_signs(*a, *b, *c, *d, scale);
    let sign = ordered_signs[output_index];
    let eigenvalues = ordered_signs.map(
        |ordered_sign| {
            if ordered_sign < 0.0 {
                lower
            } else {
                upper
            }
        },
    );
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvals output must be finite"
        ));
    }
    Ok(Eigvals2x2 {
        output_index,
        sign,
        values,
        eigenvalues,
        gap,
    })
}

fn parse_eig_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eig2x2, String> {
    replay_checkpoint()?;
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eig Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eig inputs must be finite"));
    }
    let output = parse_eig_output(effect_index, operation)?;
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eig Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let scale = values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let diagonal_delta = a - d;
    let discriminant = diagonal_delta * diagonal_delta + 4.0 * b * c;
    let discriminant_tolerance = REAL_SPECTRUM_TOLERANCE * scale * scale;
    if discriminant < -discriminant_tolerance {
        return Err(format!(
            "effect {effect_index} eig requires real eigenvalues"
        ));
    }
    let gap = discriminant.max(0.0).sqrt();
    let center = 0.5 * (a + d);
    let lower = center - 0.5 * gap;
    let upper = center + 0.5 * gap;
    let eigen_scale = 1.0_f64.max(lower.abs()).max(upper.abs());
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eig requires distinct eigenvalues"
        ));
    }
    let ordered_signs = eigvals_2x2_order_signs(*a, *b, *c, *d, scale);
    let eigenvalues = ordered_signs.map(
        |ordered_sign| {
            if ordered_sign < 0.0 {
                lower
            } else {
                upper
            }
        },
    );
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eig output must be finite"));
    }
    let right_eigenvectors = eig_2x2_right_eigenvectors(effect_index, values, eigenvalues, scale)?;
    let left_eigenvector_rows = invert_eig_2x2_basis(effect_index, right_eigenvectors)?;
    Ok(Eig2x2 {
        output,
        eigenvalues,
        right_eigenvectors,
        left_eigenvector_rows,
    })
}

fn parse_eigh_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigh2x2, String> {
    replay_checkpoint()?;
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eigh inputs must be finite"));
    }
    let output = parse_eigh_output(effect_index, operation)?;
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let scale = values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if (b - c).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(format!(
            "effect {effect_index} eigh requires a symmetric 2x2 matrix"
        ));
    }
    let off_diagonal = 0.5 * (b + c);
    if matches!(output, EighOutput::Eigenvector { .. })
        && off_diagonal.abs() <= SYMMETRY_TOLERANCE * scale
    {
        return Err(format!(
            "effect {effect_index} eigh eigenvector gradient requires nonzero off-diagonal entries"
        ));
    }
    let diagonal_delta = a - d;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * off_diagonal * off_diagonal).sqrt();
    let center = 0.5 * (a + d);
    let radius = 0.5 * gap;
    let eigenvalues = [center - radius, center + radius];
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eigh output must be finite"));
    }
    let eigen_scale = eigenvalues
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigh gradient requires distinct eigenvalues"
        ));
    }
    let eigenvectors = eigh_eigenvectors_2x2(*a, off_diagonal, *d, eigenvalues)?;
    Ok(Eigh2x2 {
        output,
        eigenvalues,
        eigenvectors,
    })
}
