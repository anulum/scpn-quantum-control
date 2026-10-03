// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact native compiler trace admission

import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";
import type { SourceSpan } from "../programs/programSource";

/** Versioned metadata binding; legacy native pass digests keep their own codec. */
export const COMPILER_TRACE_SCHEMA = "studio.compiler-trace.v1";
/** Bounded UTF8 imported trace, independent of measured host memory. */
export const MAX_TRACE_BYTES = 16 * 1024 * 1024;

/** Display an original native instruction without executing it. */
export interface TraceOperation {
  /** Original native operation name. */ readonly name: string;
  /** Exact binary64 parameters, in radians. */ readonly parameters: readonly number[];
  /** Ordered global operands. */ readonly qubits: readonly number[];
  /** Original classical destinations. */ readonly clbits: readonly number[];
  /** Scalar coordinates within this representation's own source. */ readonly source_span: SourceSpan;
}
/** Original source representation and its two distinct digest domains. */
export interface TraceRepresentation {
  /** Workspace-domain binding of the entire native IR. */ readonly sha256: string;
  /** Original UTF8 source identity. */ readonly sourceSha256: string;
  /** Exact source text. */ readonly source: string;
  /** Native register widths. */ readonly numQubits: number;
  /** Native classical width. */ readonly numClbits: number;
  /** Phase retained separately from QASM2. */ readonly globalPhase: number;
  /** Original ordered instructions. */ readonly operations: readonly TraceOperation[];
}
/** Source-owned dependency depth and declared dense numeric payloads. */
export interface TraceMetrics {
  /** Native operation count including readout and barriers. */ readonly operationCount: number;
  /** Gate counts excluding measurement/reset/barrier effects. */ readonly gates: Readonly<Record<string, number>>;
  /** Operand dependency depth; barriers synchronise without increasing depth. */ readonly depth: number;
  /** Declared complex128 statevector payload only. */ readonly statevectorBytes: number;
  /** Declared complex128 dense reference operator payload only. */ readonly operatorBytes: number;
}
/** A native qualification declaration, never browser numerical verification. */
export interface QualifiedTracePass {
  /** Original native row state. */ readonly state: "qualified";
  /** Recorded pass identity. */ readonly name: string;
  /** Legacy native pass digest, retained without replacing its codec. */ readonly nativeDigest: string;
  /** Logical-to-physical input mapping. */ readonly inputLayout: readonly number[];
  /** Logical-to-physical output mapping. */ readonly outputLayout: readonly number[];
  /** Logical-to-physical output classical mapping. */ readonly classicalLayout: readonly number[];
  /** Input qubit/clbit to output qubit/clbit correspondence. */ readonly observableMap: ReadonlyArray<readonly [number, number, number, number]>;
  /** Source-owned input representation. */ readonly input: TraceRepresentation;
  /** Source-owned output representation. */ readonly output: TraceRepresentation;
  /** Exact supplied pass parameters; not a dispatch request. */ readonly parameters: Readonly<Record<string, unknown>>;
  /** Declared input resource and count metadata. */ readonly before: TraceMetrics;
  /** Declared output resource and count metadata. */ readonly after: TraceMetrics;
  /** Native reference error under its recorded policy. */ readonly operatorError: number;
  /** Original native tolerance, never widened. */ readonly tolerance: number;
  /** Native global phase difference, in radians. */ readonly phaseDelta: number;
  /** Whether native admission allowed a global phase. */ readonly allowGlobalPhase: boolean;
  /** Ordered source-bound input readout effects. */ readonly inputMeasurements: ReadonlyArray<readonly [number, number]>;
  /** Ordered source-bound output readout effects. */ readonly outputMeasurements: ReadonlyArray<readonly [number, number]>;
}
/** Missing evidence cannot acquire an empty successful representation. */
export interface MissingTracePass {
  /** Explicit absent artifact. */ readonly state: "missing";
  /** Producer's original explanation. */ readonly reason: string;
}
/** One ordered qualified declaration or a visibly missing artifact. */
export type TracePass = QualifiedTracePass | MissingTracePass;
/** Immutable admitted metadata snapshot, separate from current editor text. */
export interface CompilerTraceSnapshot {
  /** Original exact portable text used for export. */ readonly text: string;
  /** New envelope digest. */ readonly sha256: string;
  /** Pinned original source. */ readonly source: string;
  /** Pinned source UTF8 identity. */ readonly sourceSha256: string;
  /** First source-bearing pass followed by ordered native declarations. */ readonly passes: readonly [QualifiedTracePass, ...TracePass[]];
  /** False when any native pass artifact is absent. */ readonly complete: boolean;
  /** Actual supplied compiler/backend snapshot. */ readonly backend: Readonly<Record<string, unknown>>;
  /** Local compiler version declared by the original producer. */ readonly compilerVersion: string;
  /** Textual interchange artifact, or explicit absence. */ readonly emittedText: string | null;
  /** Metadata import does not rerun the native numerical qualifier. */ readonly nativeVerification: "producer_declaration";
}
/** Caller-safe admission outcome; rejection carries no replacement snapshot. */
export type CompilerTraceResult = {
  /** Exact metadata binding succeeded. */ readonly ok: true;
  /** Detached immutable admitted snapshot. */ readonly value: CompilerTraceSnapshot;
} | {
  /** Refusal carries no replacement snapshot. */ readonly ok: false;
  /** Stable authored refusal category. */ readonly code: string;
  /** Location of the refused field in the imported envelope. */ readonly path: string;
  /** Caller-safe authored explanation. */ readonly message: string;
};

class TraceRefused extends Error {
  constructor(readonly code: string, readonly path: string, message: string) { super(message); }
}
function refuse(path: string, message: string, code = "invalid_trace"): never {
  throw new TraceRefused(code, path, message);
}
function object(value: unknown, path: string, keys?: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) refuse(path, "Object required.");
  const row = value as Record<string, unknown>;
  if (keys && (Object.keys(row).length !== keys.length || keys.some(key => !Object.hasOwn(row, key)))) refuse(path, "Required trace fields differ.");
  return row;
}
function array(value: unknown, path: string, maximum: number): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) refuse(path, "Bounded array required.");
  return value;
}
function text(value: unknown, path: string, maximum = 1024): string {
  if (typeof value !== "string" || !value || value.length > maximum) refuse(path, "Bounded nonempty text required.");
  return value;
}
function integer(value: unknown, path: string, minimum: number, maximum: number): number {
  if (typeof value !== "bigint" || value < BigInt(minimum) || value > BigInt(maximum)) refuse(path, "Exact bounded integer required.");
  return Number(value);
}
function finite(value: unknown, path: string): number {
  // The lossless JSON reader rejects every non-finite numeric token first.
  if (typeof value !== "number") refuse(path, "Finite binary64 field required.");
  return value;
}
function digest(value: unknown, path: string): string {
  const result = text(value, path, 64);
  if (!/^[0-9a-f]{64}$/.test(result)) refuse(path, "SHA-256 hex required.");
  return result;
}
function indices(value: unknown, path: string, width: number, bijection = false): number[] {
  const result = array(value, path, 64).map((item, i) => integer(item, path + "[" + i + "]", 0, width - 1));
  if (bijection && (result.length !== width || new Set(result).size !== width)) refuse(path, "Layout must be a complete bijection.");
  return result;
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) {
    for (const member of Object.values(value)) freeze(member);
    Object.freeze(value);
  }
  return value;
}
async function sourceDigest(source: string): Promise<string> {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
  return Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, "0")).join("");
}
function span(value: unknown, path: string, starts: readonly number[], size: number): SourceSpan {
  const row = object(value, path, ["start", "end", "line", "column"]);
  const start = integer(row["start"], path + ".start", 0, size);
  const end = integer(row["end"], path + ".end", start + 1, size);
  let low = 0, high = starts.length;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (starts[middle]! <= start) low = middle; else high = middle;
  }
  const line = integer(row["line"], path + ".line", 1, size + 1);
  const column = integer(row["column"], path + ".column", 1, size + 1);
  if (line !== low + 1 || column !== start - starts[low]! + 1) refuse(path, "Source coordinates differ from original scalars.");
  return { start, end, line, column };
}
async function representation(value: unknown, path: string): Promise<TraceRepresentation> {
  const snapshot = object(value, path, ["sha256", "source_sha256", "ir"]);
  const ir = object(snapshot["ir"], path + ".ir", ["source", "num_qubits", "num_clbits", "global_phase", "operations", "quantum_registers", "classical_registers"]);
  const source = text(ir["source"], path + ".ir.source", 1024 * 1024);
  if (new TextEncoder().encode(source).length > 1024 * 1024) refuse(path, "Source exceeds one MiB.");
  const n = integer(ir["num_qubits"], path + ".num_qubits", 1, 8);
  const c = integer(ir["num_clbits"], path + ".num_clbits", 0, 64);
  const phase = finite(ir["global_phase"], path + ".global_phase");
  const starts = [0];
  let size = 0;
  for (const scalar of source) { size++; if (scalar === "\n") starts.push(size); }
  for (const [key, width] of [["quantum_registers", n], ["classical_registers", c]] as const) {
    for (const register of array(ir[key], path + "." + key, 64)) {
      const pair = array(register, path, 2);
      if (pair.length !== 2) refuse(path, "Register name/indices required.");
      text(pair[0], path);
      indices(pair[1], path, width);
    }
  }
  const operations = array(ir["operations"], path + ".operations", 4096).map((item, i): TraceOperation => {
    const at = path + ".operations[" + i + "]";
    const op = object(item, at, ["name", "parameters", "qubits", "clbits", "source_span"]);
    const name = text(op["name"], at + ".name");
    const parameters = array(op["parameters"], at, 64).map(item => finite(item, at + ".parameters"));
    const qubits = indices(op["qubits"], at + ".qubits", n), clbits = indices(op["clbits"], at + ".clbits", c);
    if (name === "measure" && (qubits.length !== 1 || clbits.length !== 1 || parameters.length !== 0)) refuse(at, "Measurement needs one original operand pair.");
    return { name, parameters, qubits, clbits, source_span: span(op["source_span"], at + ".source_span", starts, size) };
  });
  const sourceSha256 = digest(snapshot["source_sha256"], path + ".source_sha256");
  const sha256 = digest(snapshot["sha256"], path + ".sha256");
  if (await sourceDigest(source) !== sourceSha256) refuse(path, "Original source digest differs.", "source_mismatch");
  if (await canonicalDigest("studio.circuit-snapshot.v1", ir) !== sha256) refuse(path, "Native IR snapshot binding differs.", "digest_mismatch");
  return { source, sourceSha256, sha256, numQubits: n, numClbits: c, globalPhase: phase, operations };
}
function metrics(value: unknown, ir: TraceRepresentation, path: string): TraceMetrics {
  const row = object(value, path, ["operation_count", "gate_counts", "depth", "num_qubits", "num_clbits", "statevector_bytes", "operator_bytes"]);
  const operationCount = integer(row["operation_count"], path, 0, 4096);
  const gates = Object.create(null) as Record<string, number>;
  for (const [name, count] of Object.entries(object(row["gate_counts"], path))) gates[name] = integer(count, path, 1, 4096);
  const depth = integer(row["depth"], path, 0, 4096);
  if (operationCount !== ir.operations.length || integer(row["num_qubits"], path, 1, 8) !== ir.numQubits || integer(row["num_clbits"], path, 0, 64) !== ir.numClbits) refuse(path, "Resource metadata does not bind its native representation.");
  const statevectorBytes = integer(row["statevector_bytes"], path, 0, 16 * 256);
  const operatorBytes = integer(row["operator_bytes"], path, 0, 16 * 65536);
  if (statevectorBytes !== 16 * 2 ** ir.numQubits || operatorBytes !== 16 * 4 ** ir.numQubits) refuse(path, "Declared dense payload differs from width.");
  return { operationCount, gates, depth, statevectorBytes, operatorBytes };
}
async function pass(value: unknown, path: string): Promise<TracePass> {
  const row = object(value, path);
  if (row["state"] === "missing") {
    object(row, path, ["state", "reason"]);
    return { state: "missing", reason: text(row["reason"], path + ".reason") };
  }
  object(row, path, ["state", "record", "native_record_sha256", "parameters", "input", "output", "metrics", "effects"]);
  if (row["state"] !== "qualified") refuse(path, "Unsupported native pass state.");
  const record = object(row["record"], path + ".record", ["pass_name", "input_layout", "output_layout", "output_classical_layout", "observable_map", "global_phase_delta", "operator_error", "tolerance", "allow_global_phase", "reference_backend", "basis_convention", "schema"]);
  if (record["schema"] !== "circuit_pass.v1" || record["reference_backend"] !== "qiskit.quantum_info.Operator" || record["basis_convention"] !== "qiskit_little_endian") refuse(path, "Unsupported native reference policy.");
  const [input, output] = await Promise.all([representation(row["input"], path + ".input"), representation(row["output"], path + ".output")]);
  if (input.numQubits !== output.numQubits || input.numClbits !== output.numClbits) refuse(path, "Native pass widths differ.");
  const inputLayout = indices(record["input_layout"], path, input.numQubits, true);
  const outputLayout = indices(record["output_layout"], path, output.numQubits, true);
  const classicalLayout = indices(record["output_classical_layout"], path, input.numClbits, true);
  const observableMap = array(record["observable_map"], path, 64).map((item): readonly [number, number, number, number] => {
    const pairs = array(item, path, 4);
    if (pairs.length !== 4) refuse(path, "Four readout correspondence indices required.");
    return [integer(pairs[0], path, 0, input.numQubits - 1), integer(pairs[1], path, 0, input.numClbits - 1), integer(pairs[2], path, 0, output.numQubits - 1), integer(pairs[3], path, 0, output.numClbits - 1)];
  });
  const policy = object(row["metrics"], path, ["input", "output", "delta"]);
  const before = metrics(policy["input"], input, path + ".metrics.input"), after = metrics(policy["output"], output, path + ".metrics.output");
  const delta = object(policy["delta"], path, ["operation_count", "depth", "statevector_bytes", "operator_bytes"]);
  for (const [key, old, next] of [["operation_count", before.operationCount, after.operationCount], ["depth", before.depth, after.depth], ["statevector_bytes", before.statevectorBytes, after.statevectorBytes], ["operator_bytes", before.operatorBytes, after.operatorBytes]] as const) {
    if (integer(delta[key], path, -1048576, 1048576) !== next - old) refuse(path, "Pass resource/count delta differs.");
  }
  const effects = object(row["effects"], path, ["input_measurements", "output_measurements", "source_mapping"]);
  const measurements: Array<Array<readonly [number, number]>> = [];
  for (const [key, ir] of [["input_measurements", input], ["output_measurements", output]] as const) {
    const pairs = array(effects[key], path + ".effects." + key, 4096).map((value): readonly [number, number] => {
      const pair = array(value, path, 2);
      if (pair.length !== 2) refuse(path, "Readout effect needs one operand pair.");
      return [integer(pair[0], path, 0, ir.numQubits - 1), integer(pair[1], path, 0, ir.numClbits - 1)];
    });
    const native = ir.operations.filter(op => op.name === "measure").map(op => [op.qubits[0], op.clbits[0]]);
    if (String(pairs) !== String(native)) refuse(path, "Readout effects differ from original native operations.");
    measurements.push(pairs);
  }
  if (effects["source_mapping"] !== "each representation retains its own source spans; cross-pass operation correspondence is unavailable") refuse(path, "Unsupported cross-pass source correspondence.");
  const operatorError = finite(record["operator_error"], path), tolerance = finite(record["tolerance"], path), phaseDelta = finite(record["global_phase_delta"], path);
  if (!(0 < tolerance && tolerance <= 1e-12 && 0 <= operatorError && operatorError <= tolerance) || typeof record["allow_global_phase"] !== "boolean") refuse(path, "Invalid original qualification policy.");
  return { state: "qualified", name: text(record["pass_name"], path), nativeDigest: digest(row["native_record_sha256"], path), inputLayout, outputLayout, classicalLayout, observableMap, input, output, before, after, parameters: object(row["parameters"], path), operatorError, tolerance, phaseDelta, allowGlobalPhase: record["allow_global_phase"], inputMeasurements: measurements[0]!, outputMeasurements: measurements[1]! };
}
/** Admit source-bound native metadata; no compiler, numerical qualifier or device is invoked.
 * @param original Exact original producer JSON text.
 * @returns Frozen admitted snapshot or an authored refusal with no replacement state.
 */
export async function parseCompilerTrace(original: string): Promise<CompilerTraceResult> {
  try {
    if (original.length > MAX_TRACE_BYTES || new TextEncoder().encode(original).length > MAX_TRACE_BYTES) refuse("$", "Compiler trace exceeds sixteen MiB.", "resource_limit");
    const wire = object(readJson(original), "$");
    if (wire["schema"] !== COMPILER_TRACE_SCHEMA) refuse("$.schema", "Compiler trace version is unsupported.", "unsupported_version");
    object(wire, "$", ["schema", "body", "extensions", "sha256"]);
    const body = object(wire["body"], "$.body", ["source", "source_sha256", "backend_snapshot", "passes", "complete", "execution_status", "emitted_ir"]);
    object(wire["extensions"], "$.extensions");
    if (body["execution_status"] !== "emitted_not_executed") refuse("$.body.execution_status", "Native trace must retain emitted-not-executed status.");
    const backend = object(body["backend_snapshot"], "$.body.backend_snapshot", ["compiler", "compiler_version", "target", "reference_backend", "basis_convention", "settings"]);
    if (backend["compiler"] !== "qiskit" || backend["target"] !== null || backend["reference_backend"] !== "qiskit.quantum_info.Operator" || backend["basis_convention"] !== "qiskit_little_endian") refuse("$.body.backend_snapshot", "Unsupported compiler snapshot; no device substitution is admitted.");
    const compilerVersion = text(backend["compiler_version"], "$.body.backend_snapshot.compiler_version", 256);
    object(backend["settings"], "$.body.backend_snapshot.settings");
    const rows = array(body["passes"], "$.body.passes", 32);
    if (rows.length === 0) refuse("$.body.passes", "At least one source-bearing native pass is required.");
    const passes: TracePass[] = [];
    for (let index = 0; index < rows.length; index++) passes.push(await pass(rows[index], "$.body.passes[" + index + "]"));
    const first = passes[0]!;
    if (first.state !== "qualified") refuse("$.body.passes[0]", "Original source-bearing pass is missing.");
    const source = text(body["source"], "$.body.source", 1024 * 1024), sourceSha256 = digest(body["source_sha256"], "$.body.source_sha256");
    if (source !== first.input.source || sourceSha256 !== first.input.sourceSha256) refuse("$.body.source", "Original pinned source differs from first native input.", "source_mismatch");
    for (let index = 1; index < passes.length; index++) {
      const before = passes[index - 1]!, next = passes[index]!;
      if (before.state === "qualified" && next.state === "qualified" && (before.output.sha256 !== next.input.sha256 || String(before.outputLayout) !== String(next.inputLayout))) refuse("$.body.passes", "Native pass source/IR/layout continuity differs.");
    }
    const complete = passes.every(row => row.state === "qualified");
    if (body["complete"] !== complete) refuse("$.body.complete", "Missing native artifacts cannot declare a complete trace.");
    let emittedText: string | null = null;
    if (body["emitted_ir"] !== null) {
      const ir = object(body["emitted_ir"], "$.body.emitted_ir", ["text", "sha256", "dialect", "resource_counts", "metadata", "execution_status"]);
      if (ir["execution_status"] !== "textual_ir") refuse("$.body.emitted_ir", "Textual MLIR cannot declare native execution.");
      emittedText = text(ir["text"], "$.body.emitted_ir.text", MAX_TRACE_BYTES);
      if (await sourceDigest(emittedText) !== digest(ir["sha256"], "$.body.emitted_ir.sha256")) refuse("$.body.emitted_ir", "Emitted MLIR byte identity differs.", "digest_mismatch");
      text(ir["dialect"], "$.body.emitted_ir.dialect"); object(ir["resource_counts"], "$.body.emitted_ir.resource_counts"); object(ir["metadata"], "$.body.emitted_ir.metadata");
    }
    const sha256 = digest(wire["sha256"], "$.sha256");
    if (await canonicalDigest(COMPILER_TRACE_SCHEMA, { schema: wire["schema"], body, extensions: wire["extensions"] }) !== sha256) refuse("$", "Compiler trace envelope digest differs.", "digest_mismatch");
    const ordered: [QualifiedTracePass, ...TracePass[]] = [first, ...passes.slice(1)];
    return { ok: true, value: freeze({ text: original, sha256, source, sourceSha256, passes: ordered, complete, backend, compilerVersion, emittedText, nativeVerification: "producer_declaration" as const }) };
  } catch (error) {
    return error instanceof TraceRefused ? { ok: false, code: error.code, path: error.path, message: error.message }
      : { ok: false, code: "trace_unavailable", path: "$", message: "Compiler trace JSON or digest could not be admitted." };
  }
}
