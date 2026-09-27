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

use crate::program_ad_ir::filled_replay_buffer;
use crate::program_ad_lifecycle::replay_checkpoint;

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
    let mut value = -0.0_f64;
    visit_signal_terms(&spec, full_index, |left_index, right_index| {
        value += source_values[left_index] * source_values[spec.left_size + right_index];
        Ok(())
    })?;
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
    let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
    visit_signal_terms(&spec, full_index, |left_index, right_index| {
        contribution[left_index] += cotangent * source_values[spec.left_size + right_index];
        contribution[spec.left_size + right_index] += cotangent * source_values[left_index];
        Ok(())
    })?;
    for (index, value) in contribution.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} signal cotangent entries must be finite"
            ));
        }
    }
    replay_checkpoint()?;
    Ok(contribution)
}

fn parse_signal_operation(effect_index: usize, operation: &str) -> Result<SignalSpec, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) {
        replay_checkpoint()?;
    }
    let mut fields = operation.split(':');
    let mut parts = [""; 10];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| {
            format!("effect {effect_index} signal operation metadata is malformed")
        })?;
    }
    if fields.next().is_some()
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
    replay_checkpoint()?;
    validate_signal_source_count(effect_index, spec, source_values.len())?;
    for (index, value) in source_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} signal {} inputs must be finite",
                spec.kind.label()
            ));
        }
    }
    replay_checkpoint()?;
    Ok(())
}

fn output_window(
    left_size: usize,
    right_size: usize,
    mode: SignalMode,
) -> Result<(usize, usize), String> {
    let minimum = left_size.min(right_size);
    let maximum = left_size.max(right_size);
    let minimum_last = minimum
        .checked_sub(1)
        .ok_or_else(|| "signal operand sizes must be positive".to_owned())?;
    let (start, output_size) = match mode {
        SignalMode::Full => (
            0,
            left_size
                .checked_add(right_size)
                .and_then(|size| size.checked_sub(1))
                .ok_or_else(|| "signal full output size overflows".to_owned())?,
        ),
        SignalMode::Same => (minimum_last / 2, maximum),
        SignalMode::Valid => (
            minimum_last,
            maximum
                .checked_sub(minimum_last)
                .ok_or_else(|| "signal valid output size underflows".to_owned())?,
        ),
    };
    let stop = start
        .checked_add(output_size)
        .ok_or_else(|| "signal output window overflows".to_owned())?;
    Ok((start, stop))
}

fn full_output_index(effect_index: usize, spec: &SignalSpec) -> Result<usize, String> {
    let (start, stop) = output_window(spec.left_size, spec.right_size, spec.mode)?;
    let output_size = stop - start;
    if spec.output_index >= output_size {
        return Err(format!(
            "effect {effect_index} signal {} mode {} output index {} is outside output size {output_size}",
            spec.kind.label(),
            spec.mode.label(),
            spec.output_index
        ));
    }
    start
        .checked_add(spec.output_index)
        .ok_or_else(|| "signal output index overflows".to_owned())
}

fn visit_signal_terms(
    spec: &SignalSpec,
    full_index: usize,
    mut visit: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    replay_checkpoint()?;
    let next = full_index
        .checked_add(1)
        .ok_or_else(|| "signal term index overflows".to_owned())?;
    let left_start = next.saturating_sub(spec.right_size);
    let left_stop = spec.left_size.min(next);
    for left_index in left_start..left_stop {
        replay_checkpoint()?;
        let convolve_right_index = full_index
            .checked_sub(left_index)
            .ok_or_else(|| "signal convolve right index underflowed".to_owned())?;
        let right_index = match spec.kind {
            SignalKind::Convolve => convolve_right_index,
            SignalKind::Correlate => convolve_right_index
                .checked_add(1)
                .and_then(|offset| spec.right_size.checked_sub(offset))
                .ok_or_else(|| "signal correlate right index underflowed".to_owned())?,
        };
        if right_index >= spec.right_size {
            return Err("signal right index is outside operand size".to_owned());
        }
        visit(left_index, right_index)?;
    }
    replay_checkpoint()?;
    Ok(())
}

fn validate_signal_source_count(
    effect_index: usize,
    spec: &SignalSpec,
    input_count: usize,
) -> Result<usize, String> {
    replay_checkpoint()?;
    let expected_size = spec
        .left_size
        .checked_add(spec.right_size)
        .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| {
            format!("effect {effect_index} signal input bytes exceed native addressability")
        })?;
    if input_count != expected_size {
        return Err(format!(
            "effect {effect_index} signal {} expects {expected_size} inputs, got {}",
            spec.kind.label(),
            input_count
        ));
    }
    Ok(expected_size)
}

include!("program_ad_signal_reduction/workspace.rs");
