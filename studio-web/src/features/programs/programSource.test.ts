// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable compiler wire validation

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, expect, it } from "vitest";
import { instantiateProgramCompiler } from "./programCompiler";
import { parameterValue, parseProgramCompileResult, sourceSelection } from "./programSource";
import type { ProgramCompileResult } from "./programSource";

const source = '// λ😀\nOPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; if(c==2) rz(-0.0) q[1]; measure q[1] -> c[0];';
let emitted: ProgramCompileResult, refused: ProgramCompileResult;
beforeAll(async () => {
  const bytes = new Uint8Array(readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"))).buffer;
  const compile = await instantiateProgramCompiler(bytes);
  emitted = await compile(source); refused = await compile(source + " unknown q[0];");
  expect(emitted.ok).toBe(true); expect(refused.ok).toBe(false);
});

function record(): Record<string, unknown> { return JSON.parse(JSON.stringify(emitted)).value; }
function first(record: Record<string, unknown>): Record<string, unknown> { return (record["operations"] as Record<string, unknown>[])[0]!; }

it("freezes exact original native values at every nested mutable boundary", () => {
  const result = parseProgramCompileResult(emitted, source);
  if (!result?.ok) throw new Error("Actual supported native source was refused");
  const value = result.value;
  for (const object of [value, value.operations, value.operations[0], value.operations[0]!.parameters, value.operations[0]!.qubits,
    value.operations[0]!.clbits, value.operations[0]!.condition, value.operations[0]!.source_span, value.measurements, value.measurements[0]]) expect(Object.isFrozen(object)).toBe(true);
  expect(Object.is(parameterValue(value.operations[0]!.parameters[0]!), -0)).toBe(true);
  expect(parameterValue("7ff0000000000000")).toBeNull(); expect(parameterValue("nothex")).toBeNull();
});

it.each([
  ["schema", "studio.program-source.v2"], ["source", source + "changed"], ["source_sha256", "bad"],
  ["execution_status", "executed"], ["num_qubits", 9], ["num_qubits", true], ["num_clbits", 65],
  ["operations", null], ["measurements", [[0, 0]]],
])("refuses corrupted record field %s", (key, value) => {
  const wire = record(); wire[key as string] = value;
  expect(parseProgramCompileResult({ ok: true, value: wire }, source)).toBeNull();
});

it.each([
  ["name", "unknown"], ["name", "toString"], ["name", 1], ["parameters", []], ["parameters", ["7ff8000000000000"]],
  ["parameters", [7]], ["qubits", []], ["qubits", [2]], ["qubits", [0, 0]], ["qubits", [0, 1]],
  ["clbits", [0]], ["condition", { register: "d", value: "2" }], ["condition", { register: "c", value: "4" }],
  ["condition", { register: "c", value: "02" }], ["condition", { register: "c", value: 2 }],
  ["source_span", { start: 0, end: 1, line: 9, column: 1 }],
  ["source_span", { start: -1, end: 1, line: 1, column: 1 }], ["source_span", { start: 0, end: 9999, line: 1, column: 1 }],
])("refuses corrupted original operation field %s", (key, value) => {
  const wire = record(); first(wire)[key as string] = value;
  expect(parseProgramCompileResult({ ok: true, value: wire }, source)).toBeNull();
});

it("refuses operations and source spans that are not objects", () => {
  const wire = record(); wire["operations"] = [null];
  expect(parseProgramCompileResult({ ok: true, value: wire }, source)).toBeNull();
  const other = record(); first(other)["source_span"] = null;
  expect(parseProgramCompileResult({ ok: true, value: other }, source)).toBeNull();
});

it.each([null, [], 1, { ok: "true" }, { ok: true, value: null }, { ok: false, diagnostic: null }])("refuses non-record response %s", value => {
  expect(parseProgramCompileResult(value, source)).toBeNull();
});

it("retains native Unicode scalar coordinates and converts them to DOM positions", () => {
  const text = "😀λ@";
  const diagnostic = { code: "invalid_token", message: "Unsupported token", source_span: { start: 2, end: 3, line: 1, column: 3 } };
  expect(parseProgramCompileResult({ ok: false, diagnostic }, text)).not.toBeNull();
  expect(sourceSelection(text, diagnostic.source_span)).toEqual([3, 4]);
  expect(parseProgramCompileResult(refused, source + " unknown q[0];")).toEqual(refused);
});

it.each(["code", "message", "source_span"])("refuses malformed diagnostic %s", key => {
  const wire = JSON.parse(JSON.stringify(refused)); wire.diagnostic[key] = null;
  expect(parseProgramCompileResult(wire, source + " unknown q[0];")).toBeNull();
});
