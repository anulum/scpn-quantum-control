// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD signal replay

//! Compact convolution and correlation replay for bounded Program AD IR.
//!
//! The replay accepts Python-emitted scalar output opcodes for rank-1 static
//! `convolve` and `correlate` operations. Reverse replay returns the flattened
//! left/right operand cotangent contribution for one compact output element and
//! treats mode/shape metadata as nondifferentiable static metadata.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalKind {
    Convolve,
    Correlate,
}

impl SignalKind {
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "convolve" => Some(Self::Convolve),
            "correlate" => Some(Self::Correlate),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Convolve => "convolve",
            Self::Correlate => "correlate",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalMode {
    Full,
    Same,
    Valid,
}

impl SignalMode {
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "full" => Some(Self::Full),
            "same" => Some(Self::Same),
            "valid" => Some(Self::Valid),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Same => "same",
            Self::Valid => "valid",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SignalSpec {
    kind: SignalKind,
    left_size: usize,
    right_size: usize,
    mode: SignalMode,
    output_index: usize,
}

/// Return whether an operation string names a compact signal primitive.
pub(crate) fn is_signal_operation(operation: &str) -> bool {
    let mut parts = operation.split(':');
    matches!(parts.next(), Some("signal"))
        && parts.next().and_then(SignalKind::from_label).is_some()
}

/// Evaluate one compact signal output element.
pub(crate) fn signal_output_value(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
) -> Result<f64, String> {
    let spec = parse_signal_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let full_index = full_output_index(effect_index, &spec)?;
    let value = signal_terms(&spec, full_index)?
        .iter()
        .map(|(left_index, right_index)| {
            source_values[*left_index] * source_values[spec.left_size + *right_index]
        })
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} signal {} compact value must be finite",
            spec.kind.label()
        ))
    }
}

/// Build flattened left/right cotangent contribution for one compact signal output.
pub(crate) fn signal_output_cotangent(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} signal cotangent must be finite"
        ));
    }
    let spec = parse_signal_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let full_index = full_output_index(effect_index, &spec)?;
    let mut contribution = vec![0.0_f64; source_values.len()];
    for (left_index, right_index) in signal_terms(&spec, full_index)? {
        contribution[left_index] += cotangent * source_values[spec.left_size + right_index];
        contribution[spec.left_size + right_index] += cotangent * source_values[left_index];
    }
    Ok(contribution)
}

fn parse_signal_operation(effect_index: usize, operation: &str) -> Result<SignalSpec, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 10
        || parts[0] != "signal"
        || parts[2] != "left"
        || parts[4] != "right"
        || parts[6] != "mode"
        || parts[8] != "out"
    {
        return Err(format!(
            "effect {effect_index} signal operation metadata is malformed"
        ));
    }
    let kind = SignalKind::from_label(parts[1]).ok_or_else(|| {
        format!(
            "effect {effect_index} signal operation kind {} is unsupported",
            parts[1]
        )
    })?;
    let left_size = parse_positive_size(effect_index, kind, "left", parts[3])?;
    let right_size = parse_positive_size(effect_index, kind, "right", parts[5])?;
    let mode = SignalMode::from_label(parts[7]).ok_or_else(|| {
        format!(
            "effect {effect_index} signal {} mode {} is unsupported",
            kind.label(),
            parts[7]
        )
    })?;
    let output_index = parts[9].parse::<usize>().map_err(|_| {
        format!(
            "effect {effect_index} signal {} output index must be non-negative",
            kind.label()
        )
    })?;
    Ok(SignalSpec {
        kind,
        left_size,
        right_size,
        mode,
        output_index,
    })
}

fn parse_positive_size(
    effect_index: usize,
    kind: SignalKind,
    role: &str,
    label: &str,
) -> Result<usize, String> {
    let size = label.parse::<usize>().map_err(|_| {
        format!(
            "effect {effect_index} signal {} {role} size must be positive",
            kind.label()
        )
    })?;
    if size == 0 {
        return Err(format!(
            "effect {effect_index} signal {} {role} size must be positive",
            kind.label()
        ));
    }
    Ok(size)
}

fn validate_source(
    effect_index: usize,
    spec: &SignalSpec,
    source_values: &[f64],
) -> Result<(), String> {
    let expected_size = spec.left_size + spec.right_size;
    if source_values.len() != expected_size {
        return Err(format!(
            "effect {effect_index} signal {} expects {expected_size} inputs, got {}",
            spec.kind.label(),
            source_values.len()
        ));
    }
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} signal {} inputs must be finite",
            spec.kind.label()
        ));
    }
    Ok(())
}

fn output_window(left_size: usize, right_size: usize, mode: SignalMode) -> (usize, usize) {
    match mode {
        SignalMode::Full => (0, left_size + right_size - 1),
        SignalMode::Same => {
            let output_size = left_size.max(right_size);
            let start = (left_size.min(right_size) - 1) / 2;
            (start, start + output_size)
        }
        SignalMode::Valid => {
            let output_size = left_size.max(right_size) - left_size.min(right_size) + 1;
            let start = left_size.min(right_size) - 1;
            (start, start + output_size)
        }
    }
}

fn full_output_index(effect_index: usize, spec: &SignalSpec) -> Result<usize, String> {
    let (start, stop) = output_window(spec.left_size, spec.right_size, spec.mode);
    let output_size = stop - start;
    if spec.output_index >= output_size {
        return Err(format!(
            "effect {effect_index} signal {} mode {} output index {} is outside output size {output_size}",
            spec.kind.label(),
            spec.mode.label(),
            spec.output_index
        ));
    }
    Ok(start + spec.output_index)
}

fn signal_terms(spec: &SignalSpec, full_index: usize) -> Result<Vec<(usize, usize)>, String> {
    let left_start = full_index.saturating_add(1).saturating_sub(spec.right_size);
    let left_stop = spec.left_size.min(full_index + 1);
    let mut terms = Vec::with_capacity(left_stop.saturating_sub(left_start));
    for left_index in left_start..left_stop {
        let convolve_right_index = full_index - left_index;
        let right_index = match spec.kind {
            SignalKind::Convolve => convolve_right_index,
            SignalKind::Correlate => spec
                .right_size
                .checked_sub(1 + convolve_right_index)
                .ok_or_else(|| "signal correlate right index underflowed".to_owned())?,
        };
        if right_index >= spec.right_size {
            return Err("signal right index is outside operand size".to_owned());
        }
        terms.push((left_index, right_index));
    }
    Ok(terms)
}
