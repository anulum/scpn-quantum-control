// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web program-AD gradient replay (ST-12)

/**
 * Browser-side program-AD gradient replay for the committed rational unit.
 *
 * The category moat, made tangible: the panel loads the SAME bounded replay the
 * engine ships (compiled to a standalone WASM kernel) and recomputes a
 * displayed gradient in the visitor's browser, then compares it to the committed
 * value+gradient. Nothing reimplements the AD in JavaScript — the WASM kernel is
 * the single source of truth.
 *
 * Fail-closed (compliance rule 7): a forged gradient renders `mismatch`; a wrong
 * schema, malformed input, or a kernel-level rejection renders `unverifiable`.
 * A tampered unit never renders `match`.
 */

import committedUnitJson from "../../../data/studio/program_ad_replay_rational_20260708.json";

export const PROGRAM_AD_SCHEMA = "scpn_qc_studio_program_ad_replay_v1";
export const KERNEL_EXPORT = "scpn_program_ad_replay";
export const KERNEL_WASM_URL = "wasm/scpn_quantum_studio_program_ad_wasm.wasm";
const KERNEL_OK = 0;
const ALLOC_FAILED = -1;

export interface ProgramAdUnit {
  readonly schema: string;
  readonly artifactId: string;
  readonly claimBoundary: string;
  readonly inputHex: string;
  readonly expectedValue: number;
  readonly expectedGradient: readonly number[];
  readonly parameterTargets: readonly string[];
}

export type Loaded<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

export type ReplayResult =
  | { readonly ok: true; readonly value: number; readonly gradient: number[] }
  | { readonly ok: false; readonly code: number };

/** The kernel replay closure: input bytes + gradient arity to a value+gradient. */
export type KernelReplay = (input: Uint8Array, gradientLength: number) => ReplayResult;

export type ReplayDisplay = "match" | "mismatch" | "unverifiable";

export interface ReplayVerdict {
  readonly display: ReplayDisplay;
  readonly reason?: string;
  readonly recomputed?: { readonly value: number; readonly gradient: number[] };
}

export interface KernelExports {
  readonly memory: WebAssembly.Memory;
  readonly scpn_alloc: (len: number) => number;
  readonly scpn_free: (ptr: number, len: number) => void;
  readonly scpn_program_ad_replay: (
    inputPtr: number,
    inputLen: number,
    outputPtr: number,
    outputLen: number,
  ) => number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberArray(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number")) {
    return null;
  }
  return value as number[];
}

/** Parse the committed artefact into a guarded unit (exported for tests). */
export function parseProgramAdUnit(raw: unknown): Loaded<ProgramAdUnit> {
  if (!isRecord(raw) || !isRecord(raw["program"]) || !isRecord(raw["expected"])) {
    return { ok: false, reason: "program-AD artefact is missing its blocks" };
  }
  const expected = raw["expected"];
  const program = raw["program"];
  const gradient = numberArray(expected["gradient"]);
  const targets = program["parameter_targets"];
  if (
    typeof raw["schema"] !== "string" ||
    typeof raw["artifact_id"] !== "string" ||
    typeof raw["claim_boundary"] !== "string" ||
    typeof raw["input_hex"] !== "string" ||
    typeof expected["value"] !== "number" ||
    gradient === null ||
    !Array.isArray(targets) ||
    targets.some((entry) => typeof entry !== "string")
  ) {
    return { ok: false, reason: "program-AD unit fields are malformed" };
  }
  return {
    ok: true,
    value: {
      schema: raw["schema"],
      artifactId: raw["artifact_id"],
      claimBoundary: raw["claim_boundary"],
      inputHex: raw["input_hex"],
      expectedValue: expected["value"],
      expectedGradient: gradient,
      parameterTargets: targets as string[],
    },
  };
}

/** Decode a lowercase hex string to bytes, or null when malformed. */
export function hexToBytes(hex: string): Uint8Array | null {
  if (hex.length === 0 || hex.length % 2 !== 0 || !/^[0-9a-f]+$/.test(hex)) {
    return null;
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

/**
 * Bind a replay closure over a kernel's exports.
 *
 * Allocates guest memory through the kernel's own allocator, copies the input,
 * runs the replay, reads the `[value ; gradient]` block, and always frees every
 * buffer. An allocation failure or negative status surfaces as a fail-closed
 * result instead of a fabricated gradient.
 */
export function bindProgramAd(exports: KernelExports): KernelReplay {
  return (input: Uint8Array, gradientLength: number): ReplayResult => {
    const outputLen = (1 + gradientLength) * 8;
    const inputPtr = exports.scpn_alloc(input.length);
    if (inputPtr === 0) {
      return { ok: false, code: ALLOC_FAILED };
    }
    const outputPtr = exports.scpn_alloc(outputLen);
    if (outputPtr === 0) {
      exports.scpn_free(inputPtr, input.length);
      return { ok: false, code: ALLOC_FAILED };
    }
    try {
      new Uint8Array(exports.memory.buffer, inputPtr, input.length).set(input);
      const status = exports.scpn_program_ad_replay(inputPtr, input.length, outputPtr, outputLen);
      if (status !== KERNEL_OK) {
        return { ok: false, code: status };
      }
      const view = new DataView(exports.memory.buffer.slice(outputPtr, outputPtr + outputLen));
      const value = view.getFloat64(0, true);
      const gradient: number[] = [];
      for (let index = 0; index < gradientLength; index += 1) {
        gradient.push(view.getFloat64((index + 1) * 8, true));
      }
      return { ok: true, value, gradient };
    } finally {
      exports.scpn_free(inputPtr, input.length);
      exports.scpn_free(outputPtr, outputLen);
    }
  };
}

/** Instantiate the WASM kernel and return a replay closure. */
export async function instantiateProgramAd(wasmBytes: BufferSource): Promise<KernelReplay> {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  return bindProgramAd(instance.exports as unknown as KernelExports);
}

/** Fetch and instantiate the deployed kernel (browser runtime path). */
export async function fetchProgramAd(url: string = KERNEL_WASM_URL): Promise<KernelReplay> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`kernel fetch failed: ${response.status}`);
  }
  return instantiateProgramAd(await response.arrayBuffer());
}

/**
 * Verify a program-AD unit by replaying its frozen input through the kernel.
 *
 * Fail-closed order: a wrong schema or malformed input short-circuits to
 * `unverifiable` before the kernel runs. A faithful unit whose recomputed
 * value+gradient are bit-identical to the claim renders `match`; a forged
 * claim renders `mismatch`.
 */
export function verifyProgramAdUnit(unit: ProgramAdUnit, replay: KernelReplay): ReplayVerdict {
  if (unit.schema !== PROGRAM_AD_SCHEMA) {
    return { display: "unverifiable", reason: "unknown program-AD schema" };
  }
  const input = hexToBytes(unit.inputHex);
  if (input === null) {
    return { display: "unverifiable", reason: "input payload is not valid hex" };
  }
  const result = replay(input, unit.expectedGradient.length);
  if (!result.ok) {
    return { display: "unverifiable", reason: `kernel rejected the input (code ${result.code})` };
  }
  const gradientMatches =
    result.gradient.length === unit.expectedGradient.length &&
    result.gradient.every((component, index) => component === unit.expectedGradient[index]);
  if (result.value !== unit.expectedValue || !gradientMatches) {
    return {
      display: "mismatch",
      reason: "recomputed value/gradient does not match the claim",
      recomputed: { value: result.value, gradient: result.gradient },
    };
  }
  return { display: "match", recomputed: { value: result.value, gradient: result.gradient } };
}

/** The committed rational program-AD unit, guarded. */
export const programAdUnit: Loaded<ProgramAdUnit> = parseProgramAdUnit(committedUnitJson);
