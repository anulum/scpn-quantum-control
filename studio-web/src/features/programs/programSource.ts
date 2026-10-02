// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact emitted source record projection

/** Versioned record shared by native Python and the actual Rust/WASM compiler. */
export const PROGRAM_SOURCE_SCHEMA = "studio.program-source.v1";

/** Exact Unicode scalar offsets and one-based original source coordinates. */
export interface SourceSpan {
  /** First scalar included. */
  readonly start: number;
  /** First scalar excluded; equal to start for a missing token. */
  readonly end: number;
  /** One-based original line. */
  readonly line: number;
  /** One-based original scalar column. */
  readonly column: number;
}

/** Whole-register comparison with exact 64-bit integer text. */
export interface ProgramCondition {
  /** Admitted original c register. */
  readonly register: "c";
  /** Canonical unsigned decimal value. */
  readonly value: string;
}

/** Original ordered gate or effect without a numerical runtime claim. */
export interface ProgramOperation {
  /** Native gate or explicit measure/reset/barrier name. */
  readonly name: string;
  /** Ordered exact IEEE float64 parameter hex strings, in radians. */
  readonly parameters: readonly string[];
  /** Ordered global native qubit operands. */
  readonly qubits: readonly number[];
  /** Original global classical destinations. */
  readonly clbits: readonly number[];
  /** Retained classical condition, or null for unconditional operation. */
  readonly condition: ProgramCondition | null;
  /** Entire original statement, including any conditional prefix. */
  readonly source_span: SourceSpan;
}

/** Immutable source emission snapshot; compilation does not execute the program. */
export interface CompiledProgram {
  /** Supported versioned schema. */
  readonly schema: typeof PROGRAM_SOURCE_SCHEMA;
  /** Exact original source text. */
  readonly source: string;
  /** Digest of original UTF8 bytes. */
  readonly source_sha256: string;
  /** Original q register width. */
  readonly num_qubits: number;
  /** Original c register width, or zero if absent. */
  readonly num_clbits: number;
  /** Ordered original operation records. */
  readonly operations: readonly ProgramOperation[];
  /** Ordered qubit/classical-bit readout pairs. */
  readonly measurements: ReadonlyArray<readonly [number, number]>;
  /** Emission boundary, independent of runtime execution. */
  readonly execution_status: "emitted_not_executed";
}

/** Located authored compiler refusal. */
export interface ProgramDiagnostic {
  /** Stable refusal category. */
  readonly code: string;
  /** Authored caller-safe explanation. */
  readonly message: string;
  /** Exact offending token, or missing-token EOF location. */
  readonly source_span: SourceSpan;
}

/** A real source-bound record or a located refusal; neither claims execution. */
export type ProgramCompileResult =
  | {
      /** Successful emission. */
      readonly ok: true;
      /** Immutable original record. */
      readonly value: CompiledProgram;
    }
  | {
      /** Refused emission. */
      readonly ok: false;
      /** Located authored reason. */
      readonly diagnostic: ProgramDiagnostic;
    };

/** Listed native arities for form construction and wire validation only. */
export const PROGRAM_GATE_SHAPES: Readonly<Record<string, readonly [number, number]>> = Object.freeze({
  h: [0, 1], x: [0, 1], y: [0, 1], z: [0, 1], s: [0, 1], sdg: [0, 1], t: [0, 1], tdg: [0, 1],
  id: [0, 1], sx: [0, 1], sxdg: [0, 1], rx: [1, 1], ry: [1, 1], rz: [1, 1], p: [1, 1], u1: [1, 1],
  u2: [2, 1], u: [3, 1], u3: [3, 1], cx: [0, 2], cz: [0, 2], swap: [0, 2], rxx: [1, 2], ryy: [1, 2],
  rzz: [1, 2], measure: [0, 1], reset: [0, 1], barrier: [0, 0],
});

/** Test one untrusted JSON object without coercing arrays or scalar types. */
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Check safe finite register/offset integers without rounding. */
function integer(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum && value <= maximum;
}

/** Admit an exact scalar span within its original source. */
function span(value: unknown, positions: readonly number[], size: number): value is SourceSpan {
  if (!object(value)) return false;
  if (!integer(value["start"], 0, size) || !integer(value["end"], value["start"], size)) return false;
  let low = 0, high = positions.length;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (positions[middle]! <= value["start"]) low = middle;
    else high = middle;
  }
  return value["line"] === low + 1 && value["column"] === value["start"] - positions[low]! + 1;
}

/** Decode exact IEEE parameter bits for display without changing stored identity. */
export function parameterValue(hex: string): number | null {
  if (!/^[0-9a-f]{16}$/.test(hex)) return null;
  const view = new DataView(new ArrayBuffer(8));
  view.setBigUint64(0, BigInt("0x" + hex), false);
  const value = view.getFloat64(0, false);
  return Number.isFinite(value) ? value : null;
}

/** Validate an emitted operation without parsing or executing source text. */
function operation(value: unknown, positions: readonly number[], size: number, n: number, c: number): value is ProgramOperation {
  if (!object(value) || typeof value["name"] !== "string") return false;
  if (!Object.hasOwn(PROGRAM_GATE_SHAPES, value["name"])) return false;
  const shape = PROGRAM_GATE_SHAPES[value["name"]];
  if (shape === undefined || !span(value["source_span"], positions, size)) return false;
  const parameters = value["parameters"], qubits = value["qubits"], clbits = value["clbits"];
  if (!Array.isArray(parameters) || parameters.length !== shape[0] || !parameters.every(p => typeof p === "string" && parameterValue(p) !== null)) return false;
  if (!Array.isArray(qubits) || qubits.length === 0 || !qubits.every(q => integer(q, 0, n - 1)) || new Set(qubits).size !== qubits.length) return false;
  if (shape[1] !== 0 && qubits.length !== shape[1]) return false;
  if (!Array.isArray(clbits) || clbits.length !== (value["name"] === "measure" ? 1 : 0) || !clbits.every(b => integer(b, 0, c - 1))) return false;
  const condition = value["condition"];
  if (condition === null) return true;
  return !["measure", "reset", "barrier"].includes(value["name"]) && object(condition) && condition["register"] === "c" && typeof condition["value"] === "string" &&
    /^(0|[1-9][0-9]{0,19})$/.test(condition["value"]) && c > 0 && BigInt(condition["value"]) < (1n << BigInt(c));
}

/** Validate and deep-freeze the actual compiler response before a view can use it.
 * @param value Untrusted JSON from the real compiler binding.
 * @param source Exact draft sent to the compiler.
 * @returns Source-bound immutable response, or null for a malformed wire record.
 */
export function parseProgramCompileResult(value: unknown, source: string): ProgramCompileResult | null {
  if (!object(value)) return null;
  const positions = [0];
  let size = 0;
  for (const scalar of source) { size += 1; if (scalar === "\n") positions.push(size); }
  if (value["ok"] === false) {
    const diagnostic = value["diagnostic"];
    if (!object(diagnostic) || typeof diagnostic["code"] !== "string" || !diagnostic["code"] ||
      typeof diagnostic["message"] !== "string" || !diagnostic["message"] || !span(diagnostic["source_span"], positions, size)) return null;
    return Object.freeze({ ok: false, diagnostic: Object.freeze({
      code: diagnostic["code"], message: diagnostic["message"], source_span: Object.freeze({ ...diagnostic["source_span"] }),
    }) });
  }
  if (value["ok"] !== true || !object(value["value"])) return null;
  const record = value["value"];
  const n = record["num_qubits"], c = record["num_clbits"];
  if (record["schema"] !== PROGRAM_SOURCE_SCHEMA || record["source"] !== source || typeof record["source_sha256"] !== "string" ||
    !/^[0-9a-f]{64}$/.test(record["source_sha256"]) || record["execution_status"] !== "emitted_not_executed" ||
    !integer(n, 1, 8) || !integer(c, 0, 64)) return null;
  const operations = record["operations"], measurements = record["measurements"];
  if (!Array.isArray(operations) || operations.length > 4096 || !operations.every(op => operation(op, positions, size, n, c))) return null;
  const expected = operations.filter(op => op.name === "measure").map(op => [op.qubits[0], op.clbits[0]]);
  if (!Array.isArray(measurements) || JSON.stringify(measurements) !== JSON.stringify(expected)) return null;
  const frozen = operations.map(op => Object.freeze({
    name: op.name, parameters: Object.freeze([...op.parameters]), qubits: Object.freeze([...op.qubits]),
    clbits: Object.freeze([...op.clbits]), source_span: Object.freeze({ ...op.source_span }),
    condition: op.condition === null ? null : Object.freeze({ ...op.condition }),
  }));
  return Object.freeze({ ok: true, value: Object.freeze({
    schema: PROGRAM_SOURCE_SCHEMA, source, source_sha256: record["source_sha256"], num_qubits: n, num_clbits: c,
    operations: Object.freeze(frozen),
    measurements: Object.freeze(measurements.map(pair => Object.freeze([pair[0], pair[1]] as const))),
    execution_status: "emitted_not_executed",
  }) });
}

/** Convert original scalar offsets into DOM UTF16 selection positions.
 * @param source Original source, possibly containing Unicode comments.
 * @param location Original scalar span.
 * @returns UTF16 start/end offsets for the source textarea.
 */
export function sourceSelection(source: string, location: SourceSpan): readonly [number, number] {
  const scalars = Array.from(source);
  return [scalars.slice(0, location.start).join("").length, scalars.slice(0, location.end).join("").length];
}
