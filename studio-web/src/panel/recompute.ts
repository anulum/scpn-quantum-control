// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web browser recompute path (WS-1)

/**
 * Browser-side recompute verifier for the committed XY-compile unit.
 *
 * The category moat, made tangible: the panel loads the SAME Rust kernel the
 * repository ships (compiled to WASM) and recomputes the structural compile
 * digest in the visitor's browser, then compares it to the committed signed
 * digest. Nothing here reimplements the digest in JavaScript — a JS
 * reimplementation could drift; the WASM kernel is the single source of truth.
 *
 * Every failure mode is loud and fail-closed (compliance rule 7): a forged
 * digest renders `mismatch`, and a stripped grade, wrong schema, malformed
 * input, or a kernel-level rejection all render `unverifiable`. A tampered
 * unit can never render `match`.
 */

import committedArtifactJson from "../../../data/studio/xy_compile_recompute_unit_20260708.json";

export const RECOMPUTE_SCHEMA = "studio.xy-compile-recompute.v1";
export const RECOMPUTE_EXACTNESS = "bit-exact";
export const RECOMPUTE_MODE = "recompute";
export const KERNEL_EXPORT = "scpn_xy_compile_digest";
// Module-relative for federated loading (see kuramoto.ts KERNEL_WASM_URL).
export const KERNEL_WASM_URL = new URL(
  "../wasm/scpn_quantum_studio_wasm_kernel.wasm",
  import.meta.url,
).href;
const DIGEST_LEN = 32;
const KERNEL_OK = 0;

export interface RecomputeUnit {
  readonly schema: string;
  readonly verifiabilityMode: string;
  readonly exactnessClass: string;
  readonly claimedDigest: string;
  readonly inputHex: string;
}

export type Loaded<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

/** The kernel recompute closure: input bytes to a digest or a status code. */
export type KernelRecompute = (input: Uint8Array) => RecomputeResult;

export type RecomputeResult =
  | { readonly ok: true; readonly digest: string }
  | { readonly ok: false; readonly code: number };

export type RecomputeDisplay = "match" | "mismatch" | "unverifiable";

export interface RecomputeVerdict {
  readonly display: RecomputeDisplay;
  readonly reason?: string;
  readonly recomputed?: string;
}

export interface KernelExports {
  readonly memory: WebAssembly.Memory;
  readonly scpn_alloc: (len: number) => number;
  readonly scpn_free: (ptr: number, len: number) => void;
  readonly scpn_xy_compile_digest: (ptr: number, len: number, out: number) => number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parse an arbitrary payload into a guarded recompute unit (exported for tests). */
export function parseRecomputeUnit(raw: unknown): Loaded<RecomputeUnit> {
  if (!isRecord(raw) || !isRecord(raw["unit"])) {
    return { ok: false, reason: "recompute artefact is missing its unit block" };
  }
  const unit = raw["unit"];
  if (
    typeof unit["schema"] !== "string" ||
    typeof unit["verifiability_mode"] !== "string" ||
    typeof unit["exactness_class"] !== "string" ||
    typeof unit["claimed_digest"] !== "string" ||
    typeof unit["input_hex"] !== "string"
  ) {
    return { ok: false, reason: "recompute unit fields are malformed" };
  }
  return {
    ok: true,
    value: {
      schema: unit["schema"],
      verifiabilityMode: unit["verifiability_mode"],
      exactnessClass: unit["exactness_class"],
      claimedDigest: unit["claimed_digest"],
      inputHex: unit["input_hex"],
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

function bytesToHex(bytes: Uint8Array): string {
  let hex = "";
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

const ALLOC_FAILED = -1;

/**
 * Bind a recompute closure over a kernel's exports.
 *
 * The closure allocates guest memory through the kernel's own allocator,
 * copies the input in, calls the digest export, reads the 32-byte digest, and
 * always frees every buffer it allocated. An allocation failure or a negative
 * status code surfaces as a fail-closed result instead of a fabricated digest.
 */
export function bindKernel(exports: KernelExports): KernelRecompute {
  return (input: Uint8Array): RecomputeResult => {
    const inputPtr = exports.scpn_alloc(input.length);
    if (inputPtr === 0) {
      return { ok: false, code: ALLOC_FAILED };
    }
    const outputPtr = exports.scpn_alloc(DIGEST_LEN);
    if (outputPtr === 0) {
      exports.scpn_free(inputPtr, input.length);
      return { ok: false, code: ALLOC_FAILED };
    }
    try {
      new Uint8Array(exports.memory.buffer, inputPtr, input.length).set(input);
      const status = exports.scpn_xy_compile_digest(inputPtr, input.length, outputPtr);
      if (status !== KERNEL_OK) {
        return { ok: false, code: status };
      }
      const digest = new Uint8Array(
        exports.memory.buffer.slice(outputPtr, outputPtr + DIGEST_LEN),
      );
      return { ok: true, digest: `sha256:${bytesToHex(digest)}` };
    } finally {
      exports.scpn_free(inputPtr, input.length);
      exports.scpn_free(outputPtr, DIGEST_LEN);
    }
  };
}

/** Instantiate the WASM kernel and return a recompute closure. */
export async function instantiateKernel(wasmBytes: BufferSource): Promise<KernelRecompute> {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  return bindKernel(instance.exports as unknown as KernelExports);
}

/** Fetch and instantiate the deployed kernel (browser runtime path). */
export async function fetchKernel(url: string = KERNEL_WASM_URL): Promise<KernelRecompute> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`kernel fetch failed: ${response.status}`);
  }
  return instantiateKernel(await response.arrayBuffer());
}

/**
 * Verify a recompute unit by replaying its input through the kernel.
 *
 * Fail-closed order: a stripped or mismatched grade, wrong schema, or
 * malformed input all short-circuit to `unverifiable` before the kernel runs,
 * so a tampered unit never reaches the digest comparison. A faithful unit
 * whose recomputed digest matches the claim renders `match`; a forged claim
 * renders `mismatch`.
 */
export function verifyRecomputeUnit(
  unit: RecomputeUnit,
  recompute: KernelRecompute,
): RecomputeVerdict {
  if (unit.schema !== RECOMPUTE_SCHEMA) {
    return { display: "unverifiable", reason: "unknown recompute schema" };
  }
  if (unit.verifiabilityMode !== RECOMPUTE_MODE) {
    return { display: "unverifiable", reason: "verifiability mode is not recompute" };
  }
  if (unit.exactnessClass !== RECOMPUTE_EXACTNESS) {
    return { display: "unverifiable", reason: "exactness class is not bit-exact" };
  }
  const input = hexToBytes(unit.inputHex);
  if (input === null) {
    return { display: "unverifiable", reason: "input payload is not valid hex" };
  }
  const result = recompute(input);
  if (!result.ok) {
    return { display: "unverifiable", reason: `kernel rejected the input (code ${result.code})` };
  }
  if (result.digest !== unit.claimedDigest) {
    return { display: "mismatch", reason: "recomputed digest does not match the claim", recomputed: result.digest };
  }
  return { display: "match", recomputed: result.digest };
}

/** The committed Paper-27 recompute unit, guarded. */
export const recomputeUnit: Loaded<RecomputeUnit> = parseRecomputeUnit(committedArtifactJson);
