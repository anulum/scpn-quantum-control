// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web program-AD gradient replay

/**
 * Browser-side program-AD gradient replay for the committed rational unit.
 *
 * The category moat, made tangible: the panel loads the SAME bounded replay the
 * engine ships (compiled to a standalone WASM kernel) and recomputes a
 * displayed gradient in the visitor's browser, then compares it to the committed
 * value+gradient. Nothing reimplements the AD in JavaScript — the WASM kernel is
 * the single source of truth.
 *
 * Fail-closed (compliance rule 7): a disagreeing gradient renders `mismatch`; a wrong
 * schema, malformed input, or a kernel-level rejection renders `unverifiable`.
 * A tampered unit never renders `match`.
 */

import committedUnitJson from "../../../data/studio/program_ad_replay_rational_20260714.json";

/** Schema identifier the committed replay unit must declare. */
export const PROGRAM_AD_SCHEMA = "scpn_qc_studio_program_ad_replay_v2";
/** Identifier of the committed replay artefact this panel renders. */
export const PROGRAM_AD_ARTIFACT_ID = "studio-program-ad-replay-rational-20260714";
/** What a successful in-browser recompute proves, stated so the card cannot overclaim. */
export const PROGRAM_AD_CLAIM_BOUNDARY =
  "bit-exact reverse-mode value+gradient of a bounded rational scalar program f(x, y) = x*x + 2*y; recompute proves the browser reproduces the engine's bounded replay and is not a claim about transcendental, linalg, or unbounded programs";
/** Digest the decoded input must match before the kernel is given it. */
export const PROGRAM_AD_INPUT_SHA256 =
  "sha256:8a5055efa19e51da321ede4245702eadf42aed1fbbb3d3ea0061e2ec9a25285f";
/** Largest effect-IR payload the kernel accepts, shared with the Python packer. */
export const MAX_PROGRAM_AD_REPLAY_IR_BYTES = 1_048_576;
/** Largest scalar-input arity the kernel accepts, shared with the Python packer. */
export const MAX_PROGRAM_AD_REPLAY_INPUTS = 4_096;
/** Largest whole payload the two bounds above allow, headers included. */
export const MAX_PROGRAM_AD_REPLAY_INPUT_BYTES =
  4 + MAX_PROGRAM_AD_REPLAY_IR_BYTES + 4 + MAX_PROGRAM_AD_REPLAY_INPUTS * 8;
/** Name of the kernel export the panel binds to. */
export const KERNEL_EXPORT = "scpn_program_ad_replay";
// Module-relative for federated loading (see kuramoto.ts KERNEL_WASM_URL).
/** Resolved URL of the shipped replay kernel module. */
export const KERNEL_WASM_URL = new URL(
  "../wasm/scpn_quantum_studio_program_ad_wasm.wasm",
  import.meta.url,
).href;
const KERNEL_OK = 0;
const ALLOC_FAILED = -1;
const INVALID_LENGTH = -2;
const CANONICAL_PARAMETER_TARGETS = ["%0", "%1"] as const;

/**
 * One committed replay unit: an input, and the value and gradient it must reproduce.
 */
export interface ProgramAdUnit {
  /** Schema identifier the artefact declares. */
  readonly schema: string;
  /** Identifier of the artefact this unit came from. */
  readonly artifactId: string;
  /** What a successful recompute does and does not prove. */
  readonly claimBoundary: string;
  /** The effect-IR payload, hex-encoded as committed. */
  readonly inputHex: string;
  /** Digest the decoded payload must hash to before it is replayed. */
  readonly inputSha256: string;
  /** Scalar the bounded replay must produce. */
  readonly expectedValue: number;
  /** Reverse-mode gradient the replay must produce, in `parameterTargets` order. */
  readonly expectedGradient: readonly number[];
  /** SSA names the gradient is taken with respect to. */
  readonly parameterTargets: readonly string[];
}

/** A load result that cannot throw: a parsed `value`, or the `reason` it was refused. */
export type Loaded<T> =
  | {
      /** Discriminant: the artefact parsed. */
      readonly ok: true;
      /** The parsed value. */
      readonly value: T;
    }
  | {
      /** Discriminant: the artefact was refused. */
      readonly ok: false;
      /** Why it was refused. */
      readonly reason: string;
    };

/** A replay outcome: the recomputed pair, or the kernel's fail-closed status code. */
export type ReplayResult =
  | {
      /** Discriminant: the kernel replayed the program. */
      readonly ok: true;
      /** The recomputed scalar. */
      readonly value: number;
      /** The recomputed reverse-mode gradient. */
      readonly gradient: number[];
    }
  | {
      /** Discriminant: the kernel refused. */
      readonly ok: false;
      /** The kernel's status code, which names the reason. */
      readonly code: number;
    };

/** The kernel replay closure: input bytes + gradient arity to a value+gradient. */
export type KernelReplay = (input: Uint8Array, gradientLength: number) => ReplayResult;

/** The three states a recompute card can show; there is no silent fourth. */
export type ReplayDisplay = "match" | "mismatch" | "unverifiable";

/**
 * The outcome shown to a reader after a recompute attempt.
 */
export interface ReplayVerdict {
  /** Which of the three states the card renders. */
  readonly display: ReplayDisplay;
  /** Why the recompute could not be trusted; present unless it matched. */
  readonly reason?: string;
  /** What the browser actually produced, when it produced anything. */
  readonly recomputed?: {
    /** The scalar the browser produced. */
    readonly value: number;
    /** The gradient the browser produced, in the unit's target order. */
    readonly gradient: number[];
  };
}

/**
 * The replay kernel's export surface, as the panel binds it.
 */
export interface KernelExports {
  /** The kernel's linear memory, which the panel reads results out of. */
  readonly memory: WebAssembly.Memory;
  /** Allocate `len` bytes inside the kernel and return the pointer. */
  readonly scpn_alloc: (len: number) => number;
  /** Release a pointer previously obtained from `scpn_alloc`. */
  readonly scpn_free: (ptr: number, len: number) => void;
  /** Run one replay; returns a status code, not a value. */
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
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))
  ) {
    return null;
  }
  return value as number[];
}

function stringArray(value: unknown): string[] | null {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((entry) => typeof entry !== "string" || entry.length === 0)
  ) {
    return null;
  }
  const strings = value as string[];
  return new Set(strings).size === strings.length ? strings : null;
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function unitContractError(unit: ProgramAdUnit): string | null {
  if (unit.schema !== PROGRAM_AD_SCHEMA) return "unknown program-AD schema";
  if (unit.artifactId !== PROGRAM_AD_ARTIFACT_ID) return "unknown program-AD artifact id";
  if (unit.claimBoundary !== PROGRAM_AD_CLAIM_BOUNDARY) {
    return "program-AD claim boundary is not canonical";
  }
  if (unit.inputSha256 !== PROGRAM_AD_INPUT_SHA256) {
    return "program-AD input digest is not canonical";
  }
  if (!Number.isFinite(unit.expectedValue)) return "expected value is not finite";
  if (
    unit.expectedGradient.length === 0 ||
    unit.expectedGradient.some((value) => !Number.isFinite(value))
  ) {
    return "expected gradient is not a finite vector";
  }
  if (!sameStrings(unit.parameterTargets, CANONICAL_PARAMETER_TARGETS)) {
    return "program-AD parameter targets are not canonical";
  }
  if (unit.expectedGradient.length !== unit.parameterTargets.length) {
    return "gradient arity does not match the parameter targets";
  }
  return null;
}

/** Parse the committed artefact into a guarded unit (exported for tests). */
/** Parse the committed replay unit into a guarded value. */
export function parseProgramAdUnit(raw: unknown): Loaded<ProgramAdUnit> {
  if (!isRecord(raw) || !isRecord(raw["program"]) || !isRecord(raw["expected"])) {
    return { ok: false, reason: "program-AD artefact is missing its blocks" };
  }
  const expected = raw["expected"];
  const program = raw["program"];
  const gradient = numberArray(expected["gradient"]);
  const targets = stringArray(program["parameter_targets"]);
  if (
    typeof raw["schema"] !== "string" ||
    typeof raw["artifact_id"] !== "string" ||
    typeof raw["claim_boundary"] !== "string" ||
    typeof raw["input_hex"] !== "string" ||
    typeof raw["input_sha256"] !== "string" ||
    typeof expected["value"] !== "number" ||
    !Number.isFinite(expected["value"]) ||
    gradient === null ||
    targets === null
  ) {
    return { ok: false, reason: "program-AD unit fields are malformed" };
  }
  const value: ProgramAdUnit = {
    schema: raw["schema"],
    artifactId: raw["artifact_id"],
    claimBoundary: raw["claim_boundary"],
    inputHex: raw["input_hex"],
    inputSha256: raw["input_sha256"],
    expectedValue: expected["value"],
    expectedGradient: gradient,
    parameterTargets: targets,
  };
  const contractError = unitContractError(value);
  if (contractError !== null) return { ok: false, reason: contractError };
  if (hexToBytes(value.inputHex) === null) {
    return { ok: false, reason: "input payload is not valid hex" };
  }
  return {
    ok: true,
    value,
  };
}

/** Decode a bounded lowercase hex string to bytes, or null when malformed. */
/** Decode a hex payload, or return `null` rather than a partial buffer. */
export function hexToBytes(hex: string): Uint8Array | null {
  if (
    hex.length === 0 ||
    hex.length > MAX_PROGRAM_AD_REPLAY_INPUT_BYTES * 2 ||
    hex.length % 2 !== 0 ||
    !/^[0-9a-f]+$/.test(hex)
  ) {
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
  for (const byte of bytes) hex += byte.toString(16).padStart(2, "0");
  return hex;
}

/** Return the Web-Crypto SHA-256 binding for exact replay input bytes. */
/** SHA-256 of the decoded input, formatted as the artefact records it. */
export async function digestProgramAdInput(input: Uint8Array): Promise<string> {
  if (globalThis.crypto?.subtle === undefined) {
    throw new Error("Web Crypto SHA-256 is unavailable");
  }
  const stableInput = Uint8Array.from(input);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", stableInput);
  return `sha256:${bytesToHex(new Uint8Array(digest))}`;
}

/**
 * Bind a replay closure over a kernel's exports.
 *
 * Allocates guest memory through the kernel's own allocator, copies the input,
 * runs the replay, reads the `[value ; gradient]` block, and always frees every
 * buffer. An allocation failure or negative status surfaces as a fail-closed
 * result instead of a fabricated gradient.
 */
/**
 * Bind the kernel exports into a replay closure that owns its allocations.
 *
 * Every buffer taken from `scpn_alloc` is released on both the success and the
 * failure path, so a refused replay does not leak kernel memory.
 */
export function bindProgramAd(exports: KernelExports): KernelReplay {
  return (input: Uint8Array, gradientLength: number): ReplayResult => {
    if (
      input.length === 0 ||
      input.length > MAX_PROGRAM_AD_REPLAY_INPUT_BYTES ||
      !Number.isSafeInteger(gradientLength) ||
      gradientLength <= 0 ||
      gradientLength > MAX_PROGRAM_AD_REPLAY_INPUTS
    ) {
      return { ok: false, code: INVALID_LENGTH };
    }
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
/** Instantiate the replay kernel from bytes already in hand. */
export async function instantiateProgramAd(wasmBytes: BufferSource): Promise<KernelReplay> {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  return bindProgramAd(instance.exports as unknown as KernelExports);
}

/** Fetch and instantiate the deployed kernel (browser runtime path). */
/** Fetch and instantiate the deployed replay kernel (browser runtime path). */
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
 * Fail-closed order: a wrong schema, malformed input, or SHA-256 mismatch
 * short-circuits to `unverifiable` before the kernel runs. A faithful unit
 * whose recomputed value+gradient are bit-identical to the claim renders
 * `match`; a claim the recomputation disagrees with renders `mismatch`.
 */
/**
 * Recompute a committed unit in the browser and judge the result.
 *
 * The digest is checked before the kernel runs, so a payload that does not
 * match the artefact is refused rather than replayed and compared.
 */
export async function verifyProgramAdUnit(
  unit: ProgramAdUnit,
  replay: KernelReplay,
): Promise<ReplayVerdict> {
  const contractError = unitContractError(unit);
  if (contractError !== null) return { display: "unverifiable", reason: contractError };
  const input = hexToBytes(unit.inputHex);
  if (input === null) return { display: "unverifiable", reason: "input payload is not valid hex" };
  let digest: string;
  try {
    digest = await digestProgramAdInput(input);
  } catch {
    return { display: "unverifiable", reason: "replay input digest could not be verified" };
  }
  if (digest !== PROGRAM_AD_INPUT_SHA256 || digest !== unit.inputSha256) {
    return { display: "unverifiable", reason: "replay input does not match its SHA-256 binding" };
  }
  let result: ReplayResult;
  try {
    result = replay(input, unit.expectedGradient.length);
  } catch {
    return { display: "unverifiable", reason: "kernel replay failed" };
  }
  if (!result.ok) {
    return { display: "unverifiable", reason: `kernel rejected the input (code ${result.code})` };
  }
  if (!Number.isFinite(result.value) || result.gradient.some((value) => !Number.isFinite(value))) {
    return { display: "unverifiable", reason: "kernel returned a non-finite result" };
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
/** The committed replay unit, parsed at module load. */
export const programAdUnit: Loaded<ProgramAdUnit> = parseProgramAdUnit(committedUnitJson);
