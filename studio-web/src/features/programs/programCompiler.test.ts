// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual source compiler and fail-closed ABI boundaries

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, expect, it, vi } from "vitest";
import corpus from "../../../../tests/data/program_authoring/corpus.json";
import { bindProgramCompiler, compileProgramSource, instantiateProgramCompiler, MAX_PROGRAM_SOURCE_BYTES } from "./programCompiler";
import type { ProgramCompiler, ProgramCompilerExports } from "./programCompiler";
import { INITIAL_PROGRAM_SOURCE } from "./ProgramEditor";

const bytes = new Uint8Array(readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"))).buffer;
let compiler: ProgramCompiler;
beforeAll(async () => { compiler = await instantiateProgramCompiler(bytes); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it.each(corpus.cases)("shared native source: $name", async testCase => {
  const result = await compiler(testCase.source);
  expect(result.ok).toBe(testCase.ok);
  if (result.ok) {
    expect(result.value.operations.map(op => op.name)).toEqual(testCase.operations);
    expect(result.value.measurements).toEqual(testCase.measurements);
    if (testCase.parameters !== undefined) expect(result.value.operations.map(op => op.parameters)).toEqual(testCase.parameters);
    if (testCase.conditions !== undefined) expect(result.value.operations.map(op => op.condition)).toEqual(testCase.conditions);
    expect(result.value.source).toBe(testCase.source);
    expect(Object.isFrozen(result.value)).toBe(true);
  } else {
    expect(result.diagnostic.code).toBe(testCase.code);
    const start = Array.from(testCase.source.slice(0, testCase.source.indexOf(testCase.token!, testCase.token_after))).length;
    expect(result.diagnostic.source_span.start).toBe(start);
    expect(result.diagnostic.source_span.end).toBe(start + Array.from(testCase.token!).length);
  }
});

async function exports(): Promise<ProgramCompilerExports> {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as ProgramCompilerExports;
}

it("refuses oversized source before calling the real allocator", async () => {
  const native = await exports(); const allocate = vi.fn(native.scpn_alloc);
  const result = await bindProgramCompiler({ ...native, scpn_alloc: allocate })("a".repeat(MAX_PROGRAM_SOURCE_BYTES + 1));
  expect(result.ok).toBe(false); expect(allocate).not.toHaveBeenCalled();
});

it.each([1, 2])("refuses failed allocation %s and releases earlier real allocations", async failed => {
  const native = await exports(); let count = 0; const free = vi.fn(native.scpn_free);
  const result = await bindProgramCompiler({ ...native, scpn_free: free, scpn_alloc: size => ++count === failed ? 0 : native.scpn_alloc(size) })(INITIAL_PROGRAM_SOURCE);
  expect(result).toMatchObject({ ok: false, diagnostic: { code: "allocation_failed" } });
  expect(free).toHaveBeenCalledTimes(failed - 1);
});

it("contains a native cleanup trap while attempting both real releases", async () => {
  const native = await exports(); const released: number[] = [];
  const result = await bindProgramCompiler({ ...native, scpn_free: (pointer, length) => { native.scpn_free(pointer, length); released.push(pointer); if (released.length === 1) throw new Error("native private cleanup"); } })(INITIAL_PROGRAM_SOURCE);
  expect(result).toMatchObject({ ok: false, diagnostic: { code: "compiler_failed" } });
  expect(released).toHaveLength(2);
  expect(JSON.stringify(result)).not.toContain("native private cleanup");
});

it.each([-1, 0, 8 * 1_048_576 + 1, 0.5])("refuses a corrupt ABI response length %s", async length => {
  const native = await exports(); const free = vi.fn(native.scpn_free);
  const result = await bindProgramCompiler({ ...native, scpn_free: free, scpn_program_source_compile: () => length })(INITIAL_PROGRAM_SOURCE);
  expect(result).toMatchObject({ ok: false, diagnostic: { code: "kernel_refused" } }); expect(free).toHaveBeenCalledTimes(2);
});

it.each(["{", "{}", '"not a record"']) ("refuses malformed external ABI JSON %s", async wire => {
  const native = await exports();
  const result = await bindProgramCompiler({ ...native, scpn_program_source_compile: (_input, _length, output) => {
    const encoded = new TextEncoder().encode(wire); new Uint8Array(native.memory.buffer, output, encoded.length).set(encoded); return encoded.length;
  } })(INITIAL_PROGRAM_SOURCE);
  expect(result.ok).toBe(false);
});

it("refuses a compiler record bound to a forged source digest", async () => {
  const native = await exports();
  const result = await bindProgramCompiler({ ...native, scpn_program_source_compile: (input, length, output, size) => {
    const written = native.scpn_program_source_compile(input, length, output, size);
    const record = JSON.parse(new TextDecoder().decode(new Uint8Array(native.memory.buffer, output, written)));
    record.value.source_sha256 = "0".repeat(64);
    const changed = new TextEncoder().encode(JSON.stringify(record)); new Uint8Array(native.memory.buffer, output, changed.length).set(changed); return changed.length;
  } })(INITIAL_PROGRAM_SOURCE);
  expect(result).toMatchObject({ ok: false, diagnostic: { code: "source_mismatch" } });
});

it("refuses a WASM module missing the required compiler exports", async () => {
  await expect(instantiateProgramCompiler(new Uint8Array([0,97,115,109,1,0,0,0]))).rejects.toThrow("Source compiler exports are unavailable");
});

it("uses actual WASM bytes returned by the transport", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(bytes)));
  expect((await compileProgramSource(INITIAL_PROGRAM_SOURCE)).ok).toBe(true);
});

it.each(["http", "network", "module"])("contains unavailable compiler transport: %s", async fault => {
  vi.stubGlobal("fetch", vi.fn(async () => {
    if (fault === "network") throw new Error("private network detail");
    return fault === "http" ? new Response(null, { status: 404 }) : new Response("not wasm");
  }));
  expect(await compileProgramSource(INITIAL_PROGRAM_SOURCE)).toMatchObject({ ok: false, diagnostic: { code: "compiler_unavailable" } });
});
