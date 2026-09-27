// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral metadata helpers

fn parse_eigvalsh_index(effect_index: usize, operation: &str) -> Result<usize, String> {
    let (parts, count) = spectral_fields(effect_index, operation)?;
    if count != 3 || parts[0] != "linalg" || parts[1] != "eigvalsh" {
        return Err(format!(
            "effect {effect_index} eigvalsh operation metadata is malformed"
        ));
    }
    let index = parts[2]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} eigvalsh output index metadata is malformed"))?;
    if index >= 2 {
        return Err(format!("effect {effect_index} eigvalsh 2x2 output index must be 0 or 1"));
    }
    Ok(index)
}

fn parse_eigvals_index(effect_index: usize, operation: &str) -> Result<usize, String> {
    let (parts, count) = spectral_fields(effect_index, operation)?;
    if count != 4 || parts[0] != "linalg" || parts[1] != "eigvals" || parts[2] != "2x2" {
        return Err(format!(
            "effect {effect_index} eigvals operation metadata is malformed"
        ));
    }
    let index = parts[3]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} eigvals output index metadata is malformed"))?;
    if index >= 2 {
        return Err(format!("effect {effect_index} eigvals 2x2 output index must be 0 or 1"));
    }
    Ok(index)
}

fn parse_eig_output(effect_index: usize, operation: &str) -> Result<EigOutput, String> {
    let (parts, count) = spectral_fields(effect_index, operation)?;
    if count < 5 || parts[0] != "linalg" || parts[1] != "eig" {
        return Err(format!(
            "effect {effect_index} eig operation metadata is malformed"
        ));
    }
    if parts[3] != "2x2" {
        return Err(format!(
            "effect {effect_index} eig Rust replay supports only 2x2 matrices"
        ));
    }
    match parts[2] {
        "eigenvalue" if count == 5 => {
            let index = parts[4].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eig eigenvalue index metadata is malformed")
            })?;
            if index >= 2 {
                return Err(format!(
                    "effect {effect_index} eig eigenvalue index must be 0 or 1"
                ));
            }
            Ok(EigOutput::Eigenvalue { index })
        }
        "eigenvector" if count == 6 => {
            let column = parts[4].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eig eigenvector column metadata is malformed")
            })?;
            let row = parts[5].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eig eigenvector row metadata is malformed")
            })?;
            if column >= 2 || row >= 2 {
                return Err(format!(
                    "effect {effect_index} eig eigenvector column and row must be 0 or 1"
                ));
            }
            Ok(EigOutput::Eigenvector { column, row })
        }
        _ => Err(format!(
            "effect {effect_index} eig operation metadata is malformed"
        )),
    }
}

fn parse_eigh_output(effect_index: usize, operation: &str) -> Result<EighOutput, String> {
    let (parts, count) = spectral_fields(effect_index, operation)?;
    if count < 6 || parts[0] != "linalg" || parts[1] != "eigh" {
        return Err(format!(
            "effect {effect_index} eigh operation metadata is malformed"
        ));
    }
    if parts[3] != "2x2" {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    }
    if parts[4] != "L" && parts[4] != "U" {
        return Err(format!(
            "effect {effect_index} eigh UPLO metadata must be L or U"
        ));
    }
    match parts[2] {
        "eigenvalue" if count == 6 => {
            let index = parts[5].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvalue index metadata is malformed")
            })?;
            if index >= 2 {
                return Err(format!(
                    "effect {effect_index} eigh eigenvalue index must be 0 or 1"
                ));
            }
            Ok(EighOutput::Eigenvalue { index })
        }
        "eigenvector" if count == 7 => {
            let column = parts[5].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvector column metadata is malformed")
            })?;
            let row = parts[6].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvector row metadata is malformed")
            })?;
            if column >= 2 || row >= 2 {
                return Err(format!(
                    "effect {effect_index} eigh eigenvector column and row must be 0 or 1"
                ));
            }
            Ok(EighOutput::Eigenvector { column, row })
        }
        _ => Err(format!(
            "effect {effect_index} eigh operation metadata is malformed"
        )),
    }
}


fn spectral_fields(effect_index: usize, operation: &str) -> Result<([&str; 7], usize), String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut parts = [""; 7];
    let mut count = 0usize;
    for field in operation.split(':') {
        replay_checkpoint()?;
        let slot = parts.get_mut(count).ok_or_else(|| {
            format!("effect {effect_index} spectral operation metadata is malformed")
        })?;
        *slot = field;
        count += 1;
    }
    Ok((parts, count))
}
