// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — owned source-bound kernel worker client

import { encodeKuramotoInput } from "../panel/kuramoto";
import type { KuramotoBounds, KuramotoRequest } from "../panel/kuramoto";
import { dataEntries } from "../shared/contracts/canonical";
import { admitOwnedKuramotoResources, browserResourcePolicy } from "../shared/resources/kuramotoResources";
import type { ResourcePolicy } from "../shared/resources/admission";
import { MAX_KERNEL_BINARY_BYTES, MAX_KERNEL_DEADLINE_MS, kernelBinarySize, ownedWorkerBinary, ownedWorkerVector, readWorkerEvent, workerDigest, workerRunId } from "./kernelProtocol";
import type { KernelRunIdentity, KernelWireInput, KernelWorkerEvent, KernelWorkerRequest, OwnedKernelOutcome } from "./kernelProtocol";

/** Native worker operations; an asynchronous termination must be observed before settlement. */
export interface KernelWorkerPort {
  /** Receiver installed before the first envelope is sent. */
  onmessage: ((event: { readonly data: unknown }) => void) | null;
  /** Native worker startup or runtime failure receiver. */
  onerror: ((event: { readonly message: string }) => void) | null;
  /** Send one envelope and its caller-owned transferable copies. */
  postMessage(message: unknown, transfers?: ArrayBuffer[]): void;
  /** Dispose the worker; a returned promise represents observed disposal. */
  terminate(): void | Promise<void>;
}

/** Complete declaration captured before any worker or transport buffer allocation. */
export interface OwnedKuramotoOptions extends KernelRunIdentity {
  /** Original source request; saved caller buffers are never transferred. */
  readonly request: KuramotoRequest;
  /** Original immutable kernel binary; only an owned copy is transferred. */
  readonly wasmBytes: Uint8Array;
  /** Actual kernel limits supplied by its original loader. */
  readonly bounds: KuramotoBounds;
  /** Operational disposal timeout, independent of unsupported hard wall guarantees. */
  readonly deadlineMs: number;
  /** Optional tighter source-declared byte/work ceiling. */
  readonly resourcePolicy?: ResourcePolicy;
  /** Optional observer of accepted/progress and terminal state. */
  readonly onEvent?: (event: KernelWorkerEvent) => void;
  /** Optional transport adapter; it must own a real disposable worker. */
  readonly workerFactory?: () => KernelWorkerPort;
}

/** One run owns one worker and acknowledges cancellation only after disposal. */
export interface OwnedKuramotoHandle {
  /** Settles only after observed disposal; a failed disposal remains explicit. */
  readonly result: Promise<OwnedKernelOutcome>;
  /** Cancel this run and observe its worker's termination. */
  cancel(): Promise<void>;
  /** Dispose on route/project/unmount, retaining diagnostics and prior saved data. */
  dispose(): Promise<void>;
}

function nativeWorker(): KernelWorkerPort {
  const worker = new Worker(new URL("./kernelWorker.ts", import.meta.url), { type: "module" });
  const port: KernelWorkerPort = {
    onmessage: null, onerror: null,
    postMessage(message, transfers = []) { worker.postMessage(message, transfers); },
    terminate() { return worker.terminate(); },
  };
  worker.onmessage = (event: MessageEvent<unknown>) => port.onmessage?.({ data: event.data });
  worker.onerror = event => port.onerror?.({ message: event.message });
  return port;
}

function finiteVector(value: readonly number[], expectedLength: number): Float64Array<ArrayBuffer> {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || Object.hasOwn(value, Symbol.iterator)) throw new Error("ordinary saved numeric vector required");
  if (value.length !== expectedLength) throw new Error("saved vector shape changed after admission");
  const copy = new Float64Array(expectedLength);
  for (let index = 0; index < expectedLength; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !("value" in descriptor) || typeof descriptor.value !== "number" || !Number.isFinite(descriptor.value)) throw new Error("finite numeric vector data required");
    copy[index] = descriptor.value;
  }
  return copy;
}

/** Run the original WASM through the frozen v1 transport without changing saved caller state. */
export function createOwnedKuramotoRun(options: OwnedKuramotoOptions): OwnedKuramotoHandle {
  let worker: KernelWorkerPort | null = null;
  let active = true;
  let settled = false;
  let sequence = 0;
  let accepted = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let cleanup: Promise<void> | null = null;
  let pending!: OwnedKernelOutcome;
  let identity: KernelRunIdentity | null = null;
  let snapshot: KernelWireInput | null = null;
  let terminalEvent: KernelWorkerEvent | null = null;
  let resolveOutcome!: (outcome: OwnedKernelOutcome) => void;
  const result = new Promise<OwnedKernelOutcome>(resolve => { resolveOutcome = resolve; });
  const errorText = (error: unknown) => error instanceof Error ? error.message : "owned kernel execution failed";
  const finish = async (outcome: OwnedKernelOutcome): Promise<void> => {
    if (settled) return;
    pending = outcome;
    active = false;
    if (timer !== null) { clearTimeout(timer); timer = null; }
    if (cleanup !== null) { await cleanup; return; }
    cleanup = (async () => {
      let disposed = false;
      let disposalReason = "";
      try {
        if (worker !== null) {
          worker.onmessage = null;
          worker.onerror = null;
          await worker.terminate();
          worker = null;
        }
        disposed = true;
      } catch (error: unknown) { disposalReason = errorText(error); }
      let terminal = pending;
      if (!disposed) terminal = { ok: false, code: "failed", reason: `worker disposal failed: ${disposalReason}`, disposed: false };
      else if (!terminal.ok) terminal = { ...terminal, disposed: true };
      settled = true;
      if (terminal.ok && terminalEvent !== null) {
        try { options.onEvent?.({ ...terminalEvent, payload: { ...terminalEvent.payload, disposed: true } }); }
        catch (error: unknown) { terminal = { ok: false, code: "failed", reason: errorText(error), disposed: true }; }
      }
      if (!terminal.ok && terminal.code === "cancelled" && identity !== null) {
        try {
          options.onEvent?.({ version: 1, run_id: identity.runId, sequence: ++sequence, kind: "cancelled", payload: { revision_hash: identity.revisionHash, plan_hash: identity.planHash, build_fingerprint: identity.buildFingerprint, disposed: true } });
        } catch (error: unknown) { terminal = { ok: false, code: "failed", reason: errorText(error), disposed: true }; }
      }
      resolveOutcome(terminal);
    })();
    await cleanup;
  };
  const cancel = () => finish({ ok: false, code: "cancelled", reason: "owned run cancelled after worker disposal", disposed: false });
  const handle: OwnedKuramotoHandle = { result, cancel, dispose: cancel };
  try {
    dataEntries(options);
    identity = { runId: workerRunId(options.runId), revisionHash: workerDigest(options.revisionHash), planHash: workerDigest(options.planHash), buildFingerprint: workerDigest(options.buildFingerprint) };
    if (!Number.isSafeInteger(options.deadlineMs) || options.deadlineMs < 1 || options.deadlineMs > MAX_KERNEL_DEADLINE_MS) throw new Error("bounded operational worker deadline required");
    if (!ownedWorkerBinary(options.wasmBytes)) throw new Error("bounded original WASM binary is unavailable");
    const binaryBytes = kernelBinarySize(options.wasmBytes);
    if (binaryBytes < 1 || binaryBytes > MAX_KERNEL_BINARY_BYTES) throw new Error("bounded original WASM binary is unavailable");
    dataEntries(options.request);
    dataEntries(options.bounds);
    const request = options.request;
    const bounds = { ...options.bounds };
    const n = request.omega.length;
    const policy = options.resourcePolicy ?? browserResourcePolicy(bounds);
    const admission = admitOwnedKuramotoResources({ n, steps: request.steps, mode: request.mode }, bounds, binaryBytes, policy);
    if (!admission.allowed) throw new Error(`worker resource policy refused: ${admission.blockers.join(", ")}`);
    snapshot = { mode: request.mode, omega: finiteVector(request.omega, n), theta0: finiteVector(request.theta0, n), kNm: finiteVector(request.kNm ?? [], request.mode === "networked" ? n * n : 0), steps: request.steps, dt: request.dt, coupling: request.coupling };
    if (encodeKuramotoInput({ ...snapshot, omega: Array.from(snapshot.omega), theta0: Array.from(snapshot.theta0), kNm: Array.from(snapshot.kNm) }) === null) throw new Error("original source request refused before worker allocation");
    const binary = new Uint8Array(options.wasmBytes);
    const frozen = identity;
    const input = snapshot;
    worker = (options.workerFactory ?? nativeWorker)();
    timer = setTimeout(() => { void finish({ ok: false, code: "timeout", reason: "owned kernel operational deadline reached", disposed: false }); }, options.deadlineMs);
    worker.onerror = event => { if (active) void finish({ ok: false, code: "failed", reason: event.message, disposed: false }); };
    worker.onmessage = event => {
      if (!active) return;
      try {
        const message = readWorkerEvent(event.data);
        if (message.run_id !== frozen.runId || message.sequence <= sequence) return;
        if (message.payload["revision_hash"] !== frozen.revisionHash || message.payload["plan_hash"] !== frozen.planHash) return;
        if (message.kind !== "failed" && message.payload["build_fingerprint"] !== frozen.buildFingerprint) throw new Error("worker build identity differs from the frozen run");
        sequence = message.sequence;
        if (message.kind === "failed") {
          if (typeof message.payload["reason"] !== "string") throw new Error("worker failure reason missing");
          void finish({ ok: false, code: "failed", reason: message.payload["reason"], disposed: false });
          return;
        }
        if (message.kind === "cancelled") throw new Error("worker cannot acknowledge host disposal");
        if (message.kind === "accepted") {
          if (accepted) throw new Error("worker accepted the same run twice");
          const limits = message.payload["bounds"];
          if (typeof limits !== "object" || limits === null || Array.isArray(limits)) throw new Error("worker acceptance limits missing");
          const values = Object.fromEntries(dataEntries(limits));
          if (values["maxOscillators"] !== bounds.maxOscillators || values["maxSteps"] !== bounds.maxSteps) throw new Error("worker acceptance limits differ from the frozen run");
          accepted = true;
          options.onEvent?.(message);
          if (active) worker?.postMessage({ version: 1, run_id: frozen.runId, revision_hash: frozen.revisionHash, plan_hash: frozen.planHash, command: "run", payload: null } satisfies KernelWorkerRequest);
          return;
        }
        if (!accepted) throw new Error("worker event preceded source acceptance");
        if (message.kind === "progress") { options.onEvent?.(message); return; }
        const orderParameter = message.payload["orderParameter"];
        const thetaFinal = message.payload["thetaFinal"];
        if (!ownedWorkerVector(orderParameter) || !ownedWorkerVector(thetaFinal) || orderParameter.length !== input.steps + 1 || thetaFinal.length !== n || !orderParameter.every(Number.isFinite) || !thetaFinal.every(Number.isFinite)) throw new Error("worker source output shape or values refused");
        terminalEvent = message;
        void finish({ ...frozen, ok: true, run: { orderParameter, thetaFinal }, disposed: true });
      } catch (error: unknown) { void finish({ ok: false, code: "failed", reason: errorText(error), disposed: false }); }
    };
    const payload = { wasm: binary, build_fingerprint: frozen.buildFingerprint, input, bounds, policy: admission.policy };
    worker.postMessage({ version: 1, run_id: frozen.runId, revision_hash: frozen.revisionHash, plan_hash: frozen.planHash, command: "validate", payload } satisfies KernelWorkerRequest, [binary.buffer, input.omega.buffer, input.theta0.buffer, input.kNm.buffer]);
  } catch (error: unknown) { void finish({ ok: false, code: "refused", reason: errorText(error), disposed: false }); }
  return handle;
}
