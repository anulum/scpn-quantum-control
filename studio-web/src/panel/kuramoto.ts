// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Kuramoto Play kernel binding (ST-11)

/**
 * Browser-side driver for the bounded Kuramoto live simulator.
 *
 * The Play panel loads the SAME Rust kernel the repository ships (compiled to
 * WASM) and integrates the Kuramoto order parameter R(t) in the visitor's
 * browser. Nothing here reimplements the integrator in JavaScript — the WASM
 * kernel is the single source of truth; this module only packs the canonical
 * input, drives the allocator, and decodes the output.
 *
 * Every boundary is fail-closed: a request past the kernel's declared N or step
 * limit, or any kernel-level rejection, surfaces as an explicit error rather
 * than a fabricated trajectory.
 */

import committedScenarioJson from "../../../data/studio/kuramoto_scenario_meanfield_20260708.json";

export const KURAMOTO_INPUT_VERSION = 1;
export const KURAMOTO_SIMULATE_EXPORT = "scpn_kuramoto_simulate";
// Resolved module-relative: built chunks live under assets/, the shipped
// kernels one level up under wasm/ — page-relative paths 404 when the panel
// is federated under a different origin path (e.g. the Hub's /platform/).
export const KERNEL_WASM_URL = new URL(
  "../wasm/scpn_quantum_studio_wasm_kernel.wasm",
  import.meta.url,
).href;
const HEADER_LEN = 32;
const KERNEL_OK = 0;
const ALLOC_FAILED = -1;

export type KuramotoMode = "mean-field" | "networked";
const MODE_CODES: Readonly<Record<KuramotoMode, number>> = { "mean-field": 0, networked: 1 };

export interface KuramotoRequest {
  readonly mode: KuramotoMode;
  readonly omega: readonly number[];
  readonly theta0: readonly number[];
  readonly steps: number;
  readonly dt: number;
  readonly coupling: number;
  /** Row-major n×n matrix; required for the networked kernel, omitted otherwise. */
  readonly kNm?: readonly number[];
}

export interface KuramotoRun {
  /** `steps + 1` order-parameter samples, index 0 is the initial state. */
  readonly orderParameter: Float64Array;
  /** The `n` final phases. */
  readonly thetaFinal: Float64Array;
}

export type SimulateResult =
  | { readonly ok: true; readonly run: KuramotoRun }
  | { readonly ok: false; readonly reason: string };

/** The kernel simulate closure: a validated request to an R(t) trajectory. */
export type KernelSimulate = (request: KuramotoRequest) => SimulateResult;

export interface KuramotoExports {
  readonly memory: WebAssembly.Memory;
  readonly scpn_alloc: (len: number) => number;
  readonly scpn_free: (ptr: number, len: number) => void;
  readonly scpn_kuramoto_simulate: (
    inputPtr: number,
    inputLen: number,
    outputPtr: number,
    outputLen: number,
  ) => number;
  readonly scpn_kuramoto_max_oscillators: () => number;
  readonly scpn_kuramoto_max_steps: () => number;
}

/** The kernel's declared fail-closed bounds, read from the WASM itself. */
export interface KuramotoBounds {
  readonly maxOscillators: number;
  readonly maxSteps: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Pack a request into the kernel's canonical little-endian input, or fail closed. */
export function encodeKuramotoInput(request: KuramotoRequest): Uint8Array | null {
  const n = request.omega.length;
  if (n < 1 || request.theta0.length !== n) {
    return null;
  }
  if (!Number.isInteger(request.steps) || request.steps < 1) {
    return null;
  }
  if (!Number.isFinite(request.dt) || request.dt <= 0) {
    return null;
  }
  if (!Number.isFinite(request.coupling)) {
    return null;
  }
  const modeCode = MODE_CODES[request.mode];
  const kLen = request.mode === "networked" ? n * n : 0;
  if (request.mode === "networked" && (request.kNm?.length ?? -1) !== kLen) {
    return null;
  }
  const values = [...request.omega, ...request.theta0, ...(request.kNm ?? [])];
  if (kLen === 0 && request.kNm !== undefined && request.kNm.length !== 0) {
    return null;
  }
  const buffer = new ArrayBuffer(HEADER_LEN + values.length * 8);
  const view = new DataView(buffer);
  view.setUint32(0, KURAMOTO_INPUT_VERSION, true);
  view.setUint32(4, modeCode, true);
  view.setUint32(8, n, true);
  view.setUint32(12, request.steps, true);
  view.setFloat64(16, request.dt, true);
  view.setFloat64(24, request.coupling, true);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      return null;
    }
    view.setFloat64(HEADER_LEN + index * 8, value as number, true);
  }
  return new Uint8Array(buffer);
}

/** Read the kernel's declared bounds from its exports. */
export function readBounds(exports: KuramotoExports): KuramotoBounds {
  return {
    maxOscillators: exports.scpn_kuramoto_max_oscillators(),
    maxSteps: exports.scpn_kuramoto_max_steps(),
  };
}

/**
 * Bind a simulate closure over a kernel's exports.
 *
 * The closure packs the request, allocates guest memory through the kernel's
 * own allocator, runs the integrator, decodes the `[R(t) ; θ_final]` output,
 * and always frees every buffer. A malformed request, allocation failure, or a
 * negative status code surfaces as a fail-closed result, never a fabricated run.
 */
export function bindKuramoto(exports: KuramotoExports): KernelSimulate {
  return (request: KuramotoRequest): SimulateResult => {
    const n = request.omega.length;
    const input = encodeKuramotoInput(request);
    if (input === null) {
      return { ok: false, reason: "request is malformed" };
    }
    const outputLen = (request.steps + 1 + n) * 8;
    const inputPtr = exports.scpn_alloc(input.length);
    if (inputPtr === 0) {
      return { ok: false, reason: "input allocation failed" };
    }
    const outputPtr = exports.scpn_alloc(outputLen);
    if (outputPtr === 0) {
      exports.scpn_free(inputPtr, input.length);
      return { ok: false, reason: "output allocation failed" };
    }
    try {
      new Uint8Array(exports.memory.buffer, inputPtr, input.length).set(input);
      const status = exports.scpn_kuramoto_simulate(inputPtr, input.length, outputPtr, outputLen);
      if (status !== KERNEL_OK) {
        return { ok: false, reason: `kernel rejected the request (code ${status})` };
      }
      const raw = exports.memory.buffer.slice(outputPtr, outputPtr + outputLen);
      const values = new Float64Array(raw);
      return {
        ok: true,
        run: {
          orderParameter: values.slice(0, request.steps + 1),
          thetaFinal: values.slice(request.steps + 1),
        },
      };
    } finally {
      exports.scpn_free(inputPtr, input.length);
      exports.scpn_free(outputPtr, outputLen);
    }
  };
}

/** Instantiate the WASM kernel and return its simulate closure plus bounds. */
export async function instantiateKuramoto(
  wasmBytes: BufferSource,
): Promise<{ simulate: KernelSimulate; bounds: KuramotoBounds }> {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exports = instance.exports as unknown as KuramotoExports;
  return { simulate: bindKuramoto(exports), bounds: readBounds(exports) };
}

/** Fetch and instantiate the deployed kernel (browser runtime path). */
export async function fetchKuramoto(
  url: string = KERNEL_WASM_URL,
): Promise<{ simulate: KernelSimulate; bounds: KuramotoBounds }> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`kernel fetch failed: ${response.status}`);
  }
  return instantiateKuramoto(await response.arrayBuffer());
}

export interface KuramotoScenario {
  readonly artifactId: string;
  readonly boundaries: KuramotoBounds;
  readonly mode: KuramotoMode;
  readonly n: number;
  readonly steps: number;
  readonly dt: number;
  readonly coupling: number;
  readonly omega: readonly number[];
  readonly theta0: readonly number[];
  readonly expectedOrderParameter: readonly number[];
}

export type Loaded<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

function numberArray(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number")) {
    return null;
  }
  return value as number[];
}

/** Parse the committed scenario artefact into a guarded value (exported for tests). */
export function parseScenario(raw: unknown): Loaded<KuramotoScenario> {
  if (!isRecord(raw) || !isRecord(raw["scenario"]) || !isRecord(raw["expected"])) {
    return { ok: false, reason: "scenario artefact is missing its blocks" };
  }
  const scenario = raw["scenario"];
  const boundaries = raw["boundaries"];
  const omega = numberArray(scenario["omega"]);
  const theta0 = numberArray(scenario["theta0"]);
  const expected = numberArray(raw["expected"]["order_parameter"]);
  const mode = scenario["mode"];
  if (
    omega === null ||
    theta0 === null ||
    expected === null ||
    (mode !== "mean-field" && mode !== "networked") ||
    !isRecord(boundaries) ||
    typeof boundaries["max_oscillators"] !== "number" ||
    typeof boundaries["max_steps"] !== "number" ||
    typeof scenario["n"] !== "number" ||
    typeof scenario["steps"] !== "number" ||
    typeof scenario["dt"] !== "number" ||
    typeof scenario["coupling"] !== "number" ||
    typeof raw["artifact_id"] !== "string"
  ) {
    return { ok: false, reason: "scenario artefact fields are malformed" };
  }
  return {
    ok: true,
    value: {
      artifactId: raw["artifact_id"],
      boundaries: {
        maxOscillators: boundaries["max_oscillators"],
        maxSteps: boundaries["max_steps"],
      },
      mode,
      n: scenario["n"],
      steps: scenario["steps"],
      dt: scenario["dt"],
      coupling: scenario["coupling"],
      omega,
      theta0,
      expectedOrderParameter: expected,
    },
  };
}

/** The committed mean-field Play scenario, guarded. */
export const committedScenario: Loaded<KuramotoScenario> = parseScenario(committedScenarioJson);

/** The largest absolute R(t) deviation between a run and the committed expectation. */
export function maxOrderParameterDeviation(
  run: KuramotoRun,
  expected: readonly number[],
): number {
  if (run.orderParameter.length !== expected.length) {
    return Number.POSITIVE_INFINITY;
  }
  let worst = 0;
  for (let index = 0; index < expected.length; index += 1) {
    worst = Math.max(worst, Math.abs(run.orderParameter[index]! - expected[index]!));
  }
  return worst;
}
