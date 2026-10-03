// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound owned kernel transport

import { dataEntries } from "../shared/contracts/canonical";
import type { KuramotoBounds, KuramotoRequest, KuramotoRun } from "../panel/kuramoto";
import type { ResourcePolicy } from "../shared/resources/admission";

/** Maximum binary transfer, independent of oscillator and trajectory admission. */
export const MAX_KERNEL_BINARY_BYTES = 2 * 1024 * 1024;
/** Operational disposal deadline ceiling; no hard real-time guarantee is implied. */
export const MAX_KERNEL_DEADLINE_MS = 60_000;

function ownedView(value: unknown, brand: string): boolean {
  if (!ArrayBuffer.isView(value)) return false;
  const prototype: object = Object.getPrototypeOf(Uint8Array.prototype) as object;
  const tag = Object.getOwnPropertyDescriptor(prototype, Symbol.toStringTag)!.get!;
  if (Reflect.apply(tag, value, []) !== brand) return false;
  const buffer = Reflect.apply(Object.getOwnPropertyDescriptor(prototype, "buffer")!.get!, value, []) as unknown;
  try {
    Reflect.apply(Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, "byteLength")!.get!, buffer, []);
    return true;
  } catch { return false; }
}

/** Require an actual float64 view backed by a transferable ArrayBuffer, across native realms. */
export function ownedWorkerVector(value: unknown): value is Float64Array<ArrayBuffer> {
  return ownedView(value, "Float64Array");
}

/** Require actual uint8 binary bytes with an owned transferable backing buffer. */
export function ownedWorkerBinary(value: unknown): value is Uint8Array<ArrayBuffer> {
  return ownedView(value, "Uint8Array");
}

/** Read the actual typed binary byte count without invoking a caller's shadow accessor. */
export function kernelBinarySize(bytes: Uint8Array): number {
  const prototype: object = Object.getPrototypeOf(Uint8Array.prototype) as object;
  return Reflect.apply(Object.getOwnPropertyDescriptor(prototype, "byteLength")!.get!, bytes, []) as number;
}

/** Identity frozen before any worker allocation. */
export interface KernelRunIdentity {
  /** Opaque bounded identifier unique to the caller's run. */
  readonly runId: string;
  /** Exact saved input revision digest. */
  readonly revisionHash: string;
  /** Exact numerical plan digest, independent of display sampling. */
  readonly planHash: string;
  /** Raw SHA-256 of the exact original WASM binary. */
  readonly buildFingerprint: string;
}

/** Owned numeric copies transferred without detaching a saved caller vector. */
export interface KernelWireInput {
  /** Original source method; no fallback or translated solver. */
  readonly mode: KuramotoRequest["mode"];
  /** Owned float64 natural frequencies. */
  readonly omega: Float64Array<ArrayBuffer>;
  /** Owned float64 initial phases. */
  readonly theta0: Float64Array<ArrayBuffer>;
  /** Owned row-major coupling coefficients, empty for mean-field. */
  readonly kNm: Float64Array<ArrayBuffer>;
  /** Original positive integration step count. */
  readonly steps: number;
  /** Original finite positive integration step in source time units. */
  readonly dt: number;
  /** Original finite global coupling coefficient. */
  readonly coupling: number;
}

/** Complete source and resource declaration sent to one owned worker. */
export interface KernelWorkerPayload {
  /** Owned copy of the exact binary whose digest binds this run. */
  readonly wasm: Uint8Array<ArrayBuffer>;
  /** Source binary digest checked again inside the worker. */
  readonly build_fingerprint: string;
  /** Original input values copied after resource admission. */
  readonly input: KernelWireInput;
  /** Actual source-declared limits, checked again against instantiated exports. */
  readonly bounds: KuramotoBounds;
  /** Explicit declared byte/work policy; not discovered free memory. */
  readonly policy: ResourcePolicy;
}

/** Frozen v1 transport envelope around the unchanged source kernel ABI. */
export interface KernelWorkerRequest {
  /** Outer protocol version. */
  readonly version: 1;
  /** Run identity retained in every event. */
  readonly run_id: string;
  /** Saved input digest retained in every outcome. */
  readonly revision_hash: string;
  /** Numerical plan digest retained in every outcome. */
  readonly plan_hash: string;
  /** Validation, execution or cancellation command. */
  readonly command: "validate" | "run" | "cancel";
  /** Execution declaration; cancellation has no numerical payload. */
  readonly payload: KernelWorkerPayload | null;
}

/** Monotonic event carrying a run-bound payload. */
export interface KernelWorkerEvent {
  /** Outer protocol version. */
  readonly version: 1;
  /** Sender run identity. */
  readonly run_id: string;
  /** Positive event sequence increasing within this run. */
  readonly sequence: number;
  /** Observable worker state, never inferred numerical convergence. */
  readonly kind: "accepted" | "progress" | "result" | "failed" | "cancelled";
  /** Original source/revision/plan identity and state-specific values. */
  readonly payload: Readonly<Record<string, unknown>>;
}

/** A terminal outcome acknowledges actual disposal, including unsuccessful disposal. */
export type OwnedKernelOutcome =
  | (KernelRunIdentity & {
      /** Original WASM completed with finite source-owned outputs. */
      readonly ok: true;
      /** Original trajectory and final phases, now owned by the caller. */
      readonly run: KuramotoRun;
      /** The worker was disposed before exposing this outcome. */
      readonly disposed: true;
    })
  | {
      /** No trajectory is accepted on a refusal or lifecycle failure. */
      readonly ok: false;
      /** Refusal category independent of a numerical success claim. */
      readonly code: "refused" | "failed" | "cancelled" | "timeout";
      /** Visible reason retained for recovery. */
      readonly reason: string;
      /** True only after the owner observed its termination operation complete. */
      readonly disposed: boolean;
    };

/** Read only ordinary own data properties; refuse accessors and unknown fields. */
export function workerData(value: unknown, fields: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("worker object required");
  const entries = dataEntries(value);
  if (entries.length !== fields.length || entries.some(([key]) => !fields.includes(key))) throw new Error("worker fields differ from v1");
  return Object.fromEntries(entries);
}

/** Require one canonical lower-case source/revision/plan SHA-256 digest. */
export function workerDigest(value: unknown): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error("canonical SHA-256 required");
  return value;
}

/** Require a bounded, nonempty run identifier without control characters. */
export function workerRunId(value: unknown): string {
  if (typeof value !== "string" || value.length < 1 || value.length > 128 || /[\u0000-\u001f\u007f]/.test(value)) throw new Error("bounded worker run identity required");
  return value;
}

/** Hash the exact binary bytes, without changing their native source codec. */
export async function kernelBinaryFingerprint(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

/** Preserve exact outer fields and identity before inspecting an execution payload. */
export function readWorkerRequest(value: unknown): KernelWorkerRequest {
  const raw = workerData(value, ["version", "run_id", "revision_hash", "plan_hash", "command", "payload"]);
  if (raw["version"] !== 1 || typeof raw["command"] !== "string" || !["validate", "run", "cancel"].includes(raw["command"])) throw new Error("unsupported worker request");
  return { version: 1, run_id: workerRunId(raw["run_id"]), revision_hash: workerDigest(raw["revision_hash"]), plan_hash: workerDigest(raw["plan_hash"]), command: raw["command"] as KernelWorkerRequest["command"], payload: raw["payload"] as KernelWorkerPayload | null };
}

/** Validate sequence and event shape before any outcome can reach a consumer. */
export function readWorkerEvent(value: unknown): KernelWorkerEvent {
  const raw = workerData(value, ["version", "run_id", "sequence", "kind", "payload"]);
  if (raw["version"] !== 1 || !Number.isSafeInteger(raw["sequence"]) || Number(raw["sequence"]) < 1 || typeof raw["kind"] !== "string" || !["accepted", "progress", "result", "failed", "cancelled"].includes(raw["kind"])) throw new Error("unsupported worker event");
  const payload = raw["payload"];
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) throw new Error("worker event payload required");
  dataEntries(payload);
  return { version: 1, run_id: workerRunId(raw["run_id"]), sequence: Number(raw["sequence"]), kind: raw["kind"] as KernelWorkerEvent["kind"], payload: payload as Readonly<Record<string, unknown>> };
}
