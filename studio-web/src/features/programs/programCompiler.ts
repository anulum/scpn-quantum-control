// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual browser source compiler binding

import { parseProgramCompileResult } from "./programSource";
import type { ProgramCompileResult } from "./programSource";

/** Original shipped compiler kernel; relative URL remains valid under federation. */
export const PROGRAM_COMPILER_WASM_URL = new URL("../wasm/scpn_quantum_studio_wasm_kernel.wasm", import.meta.url).href;
/** Bound input transport before allocating guest memory. */
export const MAX_PROGRAM_SOURCE_BYTES = 1_048_576;
/** Bound response buffer matching the source compiler's exported ABI. */
export const MAX_PROGRAM_RESPONSE_BYTES = 8 * 1_048_576;

/** The original allocator plus additive emitted-source ABI. */
export interface ProgramCompilerExports {
  /** Guest memory containing owned host buffers. */
  readonly memory: WebAssembly.Memory;
  /** Allocate the exact number of bytes, returning zero on refusal. */
  scpn_alloc(length: number): number;
  /** Release an allocation using its original byte length. */
  scpn_free(pointer: number, length: number): void;
  /** Emit JSON record/diagnostic bytes, returning their length or negative status. */
  scpn_program_source_compile(input: number, inputLength: number, output: number, outputLength: number): number;
}

/** One real draft compilation, without gate simulation or provider calls. */
export type ProgramCompiler = (source: string) => Promise<ProgramCompileResult>;

/** Surface an authored runtime refusal rather than native exception text. */
function refusal(code: string, message: string): ProgramCompileResult {
  return { ok: false, diagnostic: { code, message, source_span: { start: 0, end: 0, line: 1, column: 1 } } };
}

/** Bind the real additive compiler ABI and release both allocations on every path.
 * @param exports Actual instantiated guest exports.
 * @returns Source compiler that validates response identity before exposing a plan.
 */
export function bindProgramCompiler(exports: ProgramCompilerExports): ProgramCompiler {
  return async source => {
    const input = new TextEncoder().encode(source);
    if (input.byteLength > MAX_PROGRAM_SOURCE_BYTES) return refusal("source_budget", "Source exceeds the 1 MiB import budget.");
    const allocationLength = Math.max(input.byteLength, 1);
    let inputPointer = 0, outputPointer = 0;
    try {
      try {
      inputPointer = exports.scpn_alloc(allocationLength);
      if (inputPointer === 0) return refusal("allocation_failed", "Source compiler could not allocate its input.");
      outputPointer = exports.scpn_alloc(MAX_PROGRAM_RESPONSE_BYTES);
      if (outputPointer === 0) return refusal("allocation_failed", "Source compiler could not allocate its response.");
      new Uint8Array(exports.memory.buffer, inputPointer, input.byteLength).set(input);
      const length = exports.scpn_program_source_compile(inputPointer, input.byteLength, outputPointer, MAX_PROGRAM_RESPONSE_BYTES);
      if (!Number.isSafeInteger(length) || length <= 0 || length > MAX_PROGRAM_RESPONSE_BYTES) return refusal("kernel_refused", "Source compiler refused the request.");
      const text = new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(exports.memory.buffer, outputPointer, length));
      const result = parseProgramCompileResult(JSON.parse(text), source);
      if (result === null) return refusal("invalid_compiler_record", "Source compiler returned an invalid source record.");
      if (result.ok) {
        const digest = await crypto.subtle.digest("SHA-256", input);
        const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
        if (hex !== result.value.source_sha256) return refusal("source_mismatch", "Source compiler response does not match the original draft.");
      }
        return result;
      } finally {
        try { if (inputPointer !== 0) exports.scpn_free(inputPointer, allocationLength); }
        finally { if (outputPointer !== 0) exports.scpn_free(outputPointer, MAX_PROGRAM_RESPONSE_BYTES); }
      }
    } catch {
      return refusal("compiler_failed", "Source compiler could not complete this request.");
    }
  };
}

/** Instantiate source-owned compiler bytes already in hand.
 * @param bytes Actual locked Rust/WASM binary bytes.
 * @returns Closure bound to the original kernel's source compilation ABI.
 */
export async function instantiateProgramCompiler(bytes: BufferSource): Promise<ProgramCompiler> {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports;
  if (!(exports["memory"] instanceof WebAssembly.Memory) || typeof exports["scpn_alloc"] !== "function" ||
    typeof exports["scpn_free"] !== "function" || typeof exports["scpn_program_source_compile"] !== "function") {
    throw new Error("Source compiler exports are unavailable.");
  }
  return bindProgramCompiler(exports as unknown as ProgramCompilerExports);
}

/** Fetch only the source-owned shipped compiler and emit the actual draft.
 * @param source Original user draft, retained unchanged on refusal.
 * @returns Real emitted record or authored unavailable/compilation diagnostic.
 */
export async function compileProgramSource(source: string): Promise<ProgramCompileResult> {
  try {
    const response = await fetch(PROGRAM_COMPILER_WASM_URL);
    if (!response.ok) return refusal("compiler_unavailable", "Source compiler could not be loaded.");
    const compiler = await instantiateProgramCompiler(await response.arrayBuffer());
    return await compiler(source);
  } catch {
    return refusal("compiler_unavailable", "Source compiler could not be loaded.");
  }
}
