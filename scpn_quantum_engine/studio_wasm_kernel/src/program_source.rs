// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded source-owned program compilation

//! An explicit OpenQASM2 subset emitted as immutable source-bound IR.
//!
//! Import preserves classical conditions and readout, including mid-circuit
//! measurement. Emission does not execute gates or claim static-unitary
//! qualification for effectful programs. The existing static qualifier retains
//! its separate, narrower admission contract.

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::program_source_lexer::{tokenize, Token};

/// Source transport bound, checked before token allocation.
pub const MAX_SOURCE_BYTES: usize = 1_048_576;
/// Largest admitted ordered operation count.
pub const MAX_OPERATIONS: usize = 4096;

/// Half-open Unicode scalar offsets and one-based original source coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SourceSpan {
    /// First scalar offset included in this token or statement.
    pub start: usize,
    /// First scalar offset excluded; equal to start for an EOF diagnostic.
    pub end: usize,
    /// One-based line of the start.
    pub line: usize,
    /// One-based Unicode scalar column of the start.
    pub column: usize,
}

/// Located, authored refusal with no native interpreter exception text.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramDiagnostic {
    /// Stable refusal category.
    pub code: String,
    /// Caller-safe description of the refused construct.
    pub message: String,
    /// Exact offending token or zero-width missing-token location.
    pub source_span: SourceSpan,
}

impl ProgramDiagnostic {
    /// Bind a stable authored category and message to an original location.
    pub(super) fn new(code: &str, message: &str, source_span: SourceSpan) -> Self {
        Self {
            code: code.to_owned(),
            message: message.to_owned(),
            source_span,
        }
    }
}

/// Exact whole-register condition; decimal strings retain all64 classical bits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClassicalCondition {
    /// Declared classical register, always c in this supported subset.
    pub register: String,
    /// Canonical unsigned decimal value with no lossy JSON number conversion.
    pub value: String,
}

/// An ordered gate/effect with exact operand and phase parameter identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramOperation {
    /// Native gate name, or measure/reset/barrier.
    pub name: String,
    /// Sixteen-digit IEEE754 float64 hex parameters in native argument order.
    pub parameters: Vec<String>,
    /// Global qubit indices in native operand order.
    pub qubits: Vec<usize>,
    /// Global classical bit indices for measurement destinations.
    pub clbits: Vec<usize>,
    /// Whole-register condition, present only for conditional gates.
    pub condition: Option<ClassicalCondition>,
    /// Entire original statement including any conditional prefix.
    pub source_span: SourceSpan,
}

/// Immutable, emitted-only compilation snapshot; no numerical runtime result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledProgram {
    /// Versioned schema of the source-bound record.
    pub schema: String,
    /// Original exact source, including inert comments and whitespace.
    pub source: String,
    /// SHA256 of exact original UTF8 source bytes.
    pub source_sha256: String,
    /// Declared q register width, bounded to1..8.
    pub num_qubits: usize,
    /// Declared c register width, bounded to0..64.
    pub num_clbits: usize,
    /// Original ordered operations, including effects and classical controls.
    pub operations: Vec<ProgramOperation>,
    /// Ordered qubit/classical-bit readout pairs, including repeated readout.
    pub measurements: Vec<(usize, usize)>,
    /// Explicit emitted_not_executed claim boundary.
    pub execution_status: String,
}

/// Compile the supported source subset without executing any imported source.
///
/// The grammar admits one q register and optional c register, indexed operands,
/// finite decimal rotation parameters, listed native gates, measure/reset/barrier
/// and if(c==integer) gate conditions. Arbitrary includes, expressions, custom
/// definitions, other source languages and larger inputs receive located refusal.
pub fn compile_program_source(source: &str) -> Result<CompiledProgram, ProgramDiagnostic> {
    let origin = SourceSpan {
        start: 0,
        end: usize::from(!source.is_empty()),
        line: 1,
        column: 1,
    };
    if source.len() > MAX_SOURCE_BYTES {
        return Err(ProgramDiagnostic::new(
            "source_budget",
            "Source exceeds the1MiB import budget.",
            origin,
        ));
    }
    if source.trim().is_empty() {
        return Err(ProgramDiagnostic::new(
            "invalid_source",
            "Source must contain a supported program.",
            origin,
        ));
    }
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        cursor: 0,
        source,
        num_qubits: 0,
        num_clbits: 0,
    };
    parser.expect("OPENQASM")?;
    parser.expect("2.0")?;
    parser.expect(";")?;
    parser.expect("include")?;
    let include = parser.take()?;
    if include.text != "\"qelib1.inc\"" {
        return Err(ProgramDiagnostic::new(
            "unsupported_include",
            "Only the native qelib1.inc include is supported.",
            include.span,
        ));
    }
    parser.expect(";")?;
    parser.expect("qreg")?;
    parser.expect("q")?;
    parser.expect("[")?;
    parser.num_qubits = parser.width(8)?;
    parser.expect("]")?;
    parser.expect(";")?;
    if parser.peek("creg") {
        parser.cursor += 1;
        parser.expect("c")?;
        parser.expect("[")?;
        parser.num_clbits = parser.width(64)?;
        parser.expect("]")?;
        parser.expect(";")?;
    }
    let mut operations = Vec::new();
    while parser.cursor < parser.tokens.len() {
        if operations.len() == MAX_OPERATIONS {
            return Err(ProgramDiagnostic::new(
                "circuit_budget",
                "Program exceeds4096 operations.",
                parser.tokens[parser.cursor].span,
            ));
        }
        operations.push(parser.operation()?);
    }
    let measurements = operations
        .iter()
        .filter(|op| op.name == "measure")
        .map(|op| (op.qubits[0], op.clbits[0]))
        .collect();
    Ok(CompiledProgram {
        schema: "studio.program-source.v1".to_owned(),
        source: source.to_owned(),
        source_sha256: format!("{:x}", Sha256::digest(source.as_bytes())),
        num_qubits: parser.num_qubits,
        num_clbits: parser.num_clbits,
        operations,
        measurements,
        execution_status: "emitted_not_executed".to_owned(),
    })
}

/// Cursor over an already bounded token stream; no expression evaluation.
struct Parser<'a> {
    tokens: Vec<Token>,
    cursor: usize,
    source: &'a str,
    num_qubits: usize,
    num_clbits: usize,
}

impl Parser<'_> {
    /// Inspect one literal without consuming or changing source.
    fn peek(&self, value: &str) -> bool {
        self.tokens
            .get(self.cursor)
            .is_some_and(|token| token.text == value)
    }

    /// Consume one token, reporting the original EOF position if absent.
    fn take(&mut self) -> Result<Token, ProgramDiagnostic> {
        let token = self.tokens.get(self.cursor).cloned().ok_or_else(|| {
            let start = self.source.chars().count();
            let line = self.source.bytes().filter(|byte| *byte == b'\n').count() + 1;
            let column = self
                .source
                .rsplit('\n')
                .next()
                .unwrap_or("")
                .chars()
                .count()
                + 1;
            ProgramDiagnostic::new(
                "invalid_source",
                "Source is missing a required token.",
                SourceSpan {
                    start,
                    end: start,
                    line,
                    column,
                },
            )
        })?;
        self.cursor += 1;
        Ok(token)
    }

    /// Require an explicit grammar token with its original refusal location.
    fn expect(&mut self, value: &str) -> Result<Token, ProgramDiagnostic> {
        let token = self.take()?;
        if token.text != value {
            return Err(ProgramDiagnostic::new(
                "invalid_source",
                "Token does not match the supported source grammar.",
                token.span,
            ));
        }
        Ok(token)
    }

    /// Parse bounded integer register width before allocating any bit state.
    fn width(&mut self, maximum: usize) -> Result<usize, ProgramDiagnostic> {
        let token = self.take()?;
        let value = token.text.parse::<usize>().ok();
        value
            .filter(|value| {
                *value > 0
                    && *value <= maximum
                    && token.text.bytes().all(|byte| byte.is_ascii_digit())
                    && (token.text.len() == 1 || !token.text.starts_with('0'))
            })
            .ok_or_else(|| {
                ProgramDiagnostic::new(
                    "circuit_budget",
                    "Register width is outside the supported budget.",
                    token.span,
                )
            })
    }

    /// Preserve an indexed native operand and refuse undeclared/out-of-range bits.
    fn operand(&mut self, register: &str) -> Result<usize, ProgramDiagnostic> {
        self.expect(register)?;
        self.expect("[")?;
        let token = self.take()?;
        let width = if register == "q" {
            self.num_qubits
        } else {
            self.num_clbits
        };
        let value = token
            .text
            .parse::<usize>()
            .ok()
            .filter(|value| {
                *value < width
                    && token.text.bytes().all(|byte| byte.is_ascii_digit())
                    && (token.text.len() == 1 || !token.text.starts_with('0'))
            })
            .ok_or_else(|| {
                ProgramDiagnostic::new(
                    "invalid_operand",
                    "Operand is outside the declared register.",
                    token.span,
                )
            })?;
        self.expect("]")?;
        Ok(value)
    }

    /// Parse an exact c register comparison without converting it to a JSON float.
    fn condition(&mut self) -> Result<Option<ClassicalCondition>, ProgramDiagnostic> {
        if !self.peek("if") {
            return Ok(None);
        }
        self.cursor += 1;
        self.expect("(")?;
        self.expect("c")?;
        self.expect("==")?;
        let token = self.take()?;
        let value = token.text.parse::<u64>().ok();
        if self.num_clbits == 0
            || !token.text.bytes().all(|byte| byte.is_ascii_digit())
            || (token.text.len() > 1 && token.text.starts_with('0'))
            || value.is_none()
            || (self.num_clbits < 64
                && value.is_some_and(|value| value >= (1_u64 << self.num_clbits)))
        {
            return Err(ProgramDiagnostic::new(
                "invalid_condition",
                "Condition value must fit the declared classical register.",
                token.span,
            ));
        }
        self.expect(")")?;
        Ok(Some(ClassicalCondition {
            register: "c".to_owned(),
            value: token.text,
        }))
    }

    /// Admit listed gates/effects with exact arity, phase bits and original span.
    fn operation(&mut self) -> Result<ProgramOperation, ProgramDiagnostic> {
        let start = self.tokens[self.cursor].span;
        let condition = self.condition()?;
        let gate = self.take()?;
        let (parameter_count, qubit_count) = match gate.text.as_str() {
            "h" | "x" | "y" | "z" | "s" | "sdg" | "t" | "tdg" | "id" | "sx" | "sxdg" => (0, 1),
            "rx" | "ry" | "rz" | "p" | "u1" => (1, 1),
            "u2" => (2, 1),
            "u" | "u3" => (3, 1),
            "cx" | "cz" | "swap" => (0, 2),
            "rxx" | "ryy" | "rzz" => (1, 2),
            "measure" | "reset" => (0, 1),
            "barrier" => (0, 0),
            _ => {
                return Err(ProgramDiagnostic::new(
                    "unsupported_operation",
                    "Operation is outside the supported program subset.",
                    gate.span,
                ))
            }
        };
        if condition.is_some() && matches!(gate.text.as_str(), "measure" | "reset" | "barrier") {
            return Err(ProgramDiagnostic::new(
                "unsupported_operation",
                "Classical conditions are supported only on gates.",
                gate.span,
            ));
        }
        let mut parameters = Vec::new();
        if parameter_count > 0 {
            self.expect("(")?;
            for index in 0..parameter_count {
                if index > 0 {
                    self.expect(",")?;
                }
                let token = self.take()?;
                let value = token
                    .text
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        ProgramDiagnostic::new(
                            "invalid_parameter",
                            "Gate parameters must be finite decimal numbers in radians.",
                            token.span,
                        )
                    })?;
                parameters.push(format!("{:016x}", value.to_bits()));
            }
            self.expect(")")?;
        }
        let mut qubits = Vec::new();
        if gate.text == "barrier"
            && self.peek("q")
            && self
                .tokens
                .get(self.cursor + 1)
                .is_some_and(|token| token.text == ";")
        {
            self.cursor += 1;
            qubits.extend(0..self.num_qubits);
        } else {
            qubits.push(self.operand("q")?);
            let count = if qubit_count == 0 {
                usize::MAX
            } else {
                qubit_count
            };
            while qubits.len() < count && self.peek(",") {
                self.cursor += 1;
                qubits.push(self.operand("q")?);
            }
            if qubit_count > 0 && qubits.len() != qubit_count {
                return Err(ProgramDiagnostic::new(
                    "invalid_operand",
                    "Gate operand count does not match its declared arity.",
                    gate.span,
                ));
            }
        }
        let mut unique = qubits.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != qubits.len() {
            return Err(ProgramDiagnostic::new(
                "invalid_operand",
                "An operation cannot repeat the same qubit operand.",
                gate.span,
            ));
        }
        let mut clbits = Vec::new();
        if gate.text == "measure" {
            self.expect("->")?;
            clbits.push(self.operand("c")?);
        }
        let end = self.expect(";")?.span.end;
        Ok(ProgramOperation {
            name: gate.text,
            parameters,
            qubits,
            clbits,
            condition,
            source_span: SourceSpan { end, ..start },
        })
    }
}
