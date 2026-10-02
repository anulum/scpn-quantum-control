// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded source tokens with original scalar spans

//! Tokenisation only: comments remain inert and no source is evaluated.

use crate::program_source::{ProgramDiagnostic, SourceSpan};

/// One admitted lexical token and its original Unicode scalar coordinates.
#[derive(Clone, Debug)]
pub(super) struct Token {
    pub text: String,
    pub span: SourceSpan,
}

/// Scan bounded source without expressions, filesystem lookup or recursion.
pub(super) fn tokenize(source: &str) -> Result<Vec<Token>, ProgramDiagnostic> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let (mut cursor, mut line, mut column) = (0, 1, 1);
    while cursor < chars.len() {
        if chars[cursor].is_ascii_whitespace() {
            advance(&chars, &mut cursor, &mut line, &mut column);
            continue;
        }
        if chars[cursor] == '/' && chars.get(cursor + 1) == Some(&'/') {
            while cursor < chars.len() && chars[cursor] != '\n' {
                advance(&chars, &mut cursor, &mut line, &mut column);
            }
            continue;
        }
        let mut span = SourceSpan {
            start: cursor,
            end: cursor + 1,
            line,
            column,
        };
        let ch = chars[cursor];
        if ch == '"' {
            advance(&chars, &mut cursor, &mut line, &mut column);
            while cursor < chars.len() && !matches!(chars[cursor], '"' | '\n' | '\r') {
                advance(&chars, &mut cursor, &mut line, &mut column);
            }
            if chars.get(cursor) != Some(&'"') {
                return Err(ProgramDiagnostic::new(
                    "invalid_token",
                    "Token is outside the supported source subset.",
                    span,
                ));
            }
            advance(&chars, &mut cursor, &mut line, &mut column);
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            while cursor < chars.len()
                && (chars[cursor].is_ascii_alphanumeric() || chars[cursor] == '_')
            {
                advance(&chars, &mut cursor, &mut line, &mut column);
            }
        } else if ch.is_ascii_digit()
            || matches!(ch, '.' | '+' | '-') && chars.get(cursor + 1) != Some(&'>')
        {
            while cursor < chars.len()
                && (chars[cursor].is_ascii_alphanumeric()
                    || matches!(chars[cursor], '.' | '+' | '-'))
            {
                advance(&chars, &mut cursor, &mut line, &mut column);
            }
        } else if matches!(
            (ch, chars.get(cursor + 1)),
            ('-', Some('>')) | ('=', Some('='))
        ) {
            advance(&chars, &mut cursor, &mut line, &mut column);
            advance(&chars, &mut cursor, &mut line, &mut column);
        } else if matches!(ch, '[' | ']' | '(' | ')' | ',' | ';') {
            advance(&chars, &mut cursor, &mut line, &mut column);
        } else {
            return Err(ProgramDiagnostic::new(
                "invalid_token",
                "Token is outside the supported source subset.",
                span,
            ));
        }
        span.end = cursor;
        tokens.push(Token {
            text: chars[span.start..cursor].iter().collect(),
            span,
        });
        if tokens.len() > 65_536 {
            return Err(ProgramDiagnostic::new(
                "source_budget",
                "Source exceeds the bounded token budget.",
                span,
            ));
        }
    }
    Ok(tokens)
}

/// Advance once while retaining one-based original line and scalar column.
fn advance(chars: &[char], cursor: &mut usize, line: &mut usize, column: &mut usize) {
    if chars[*cursor] == '\n' {
        *line += 1;
        *column = 1;
    } else {
        *column += 1;
    }
    *cursor += 1;
}
