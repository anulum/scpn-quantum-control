// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original worker execution with hostile transport companions

// @vitest-environment node
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { createOwnedKuramotoRun } from "../panel/kuramoto";
import type { OwnedKuramotoHandle, OwnedKuramotoOptions } from "../panel/kuramoto";
import type { KernelWorkerPort } from "./kernelClient";
import { readWorkerEvent } from "./kernelProtocol";
import type { KernelWorkerEvent } from "./kernelProtocol";

const wasm = new Uint8Array(readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm")));
const input = { mode: "mean-field" as const, omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, steps: 40, dt: 0.01 };

function options(change: Partial<OwnedKuramotoOptions> = {}): OwnedKuramotoOptions {
  return { runId: "client-real", revisionHash: "a".repeat(64), planHash: "b".repeat(64), buildFingerprint: createHash("sha256").update(wasm).digest("hex"), request: input, wasmBytes: wasm, bounds: { maxOscillators: 128, maxSteps: 4096 }, deadlineMs: 5000, workerFactory: () => new BuiltKernelWorker(), ...change };
}

function intercept(transform: (event: KernelWorkerEvent, deliver: (event: unknown) => void) => void): KernelWorkerPort {
  const native = new BuiltKernelWorker();
  const port: KernelWorkerPort = { onmessage: null, onerror: null, postMessage: (message, transfers) => native.postMessage(message, transfers), terminate: () => native.terminate() };
  native.onmessage = event => transform(readWorkerEvent(event.data), data => port.onmessage?.({ data }));
  native.onerror = event => port.onerror?.(event);
  return port;
}

it("ignores actual duplicate and foreign envelopes and publishes finite results only after real disposal", async () => {
  const observed: KernelWorkerEvent[] = [];
  const run = createOwnedKuramotoRun(options({
    workerFactory: () => intercept((event, deliver) => {
      deliver({ ...event, run_id: "other-run" });
      deliver({ ...event, payload: { ...event.payload, revision_hash: "f".repeat(64) } });
      deliver({ ...event, payload: { ...event.payload, plan_hash: "f".repeat(64) } });
      deliver(event);
      deliver(event);
    }),
    onEvent: event => { observed.push(event); if (event.kind === "result") expect(BuiltKernelWorker.activeCount).toBe(0); },
  }));
  expect(await run.result).toMatchObject({ ok: true, disposed: true });
  expect(observed.map(event => event.kind)).toEqual(["accepted", "progress", "result"]);
  expect(observed.at(-1)?.payload["disposed"]).toBe(true);
  await run.dispose();
});

it.each(["build", "absent-limits", "array-limits", "different-limits", "early-progress", "cancel-ack", "duplicate-acceptance", "outer-version", "outer-payload"])("refuses altered real worker envelope %s before exposing a trajectory", async kind => {
  const port = () => intercept((event, deliver) => {
    if (event.kind !== "accepted") { deliver(event); return; }
    const payload = { ...event.payload };
    if (kind === "build") payload["build_fingerprint"] = "f".repeat(64);
    if (kind === "absent-limits") payload["bounds"] = null;
    if (kind === "array-limits") payload["bounds"] = [];
    if (kind === "different-limits") payload["bounds"] = { maxOscillators: 1, maxSteps: 1 };
    if (kind === "duplicate-acceptance") { deliver(event); deliver({ ...event, sequence: 2 }); return; }
    deliver({ ...event, version: kind === "outer-version" ? 2 : 1, kind: kind === "early-progress" ? "progress" : kind === "cancel-ack" ? "cancelled" : event.kind, payload: kind === "outer-payload" ? [] : payload });
  });
  expect(await createOwnedKuramotoRun(options({ workerFactory: port })).result).toMatchObject({ ok: false, code: "failed", disposed: true });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each(["float32", "data-view", "not-array", "short-order", "short-theta", "nonfinite-order", "nonfinite-theta", "shared-output"])("refuses real result corrupted as %s without emitting it to the observer", async kind => {
  const observed: string[] = [];
  const factory = () => intercept((event, deliver) => {
    if (event.kind !== "result") { deliver(event); return; }
    const payload = { ...event.payload };
    if (kind === "float32") payload["orderParameter"] = new Float32Array(41);
    if (kind === "data-view") payload["orderParameter"] = new DataView(new ArrayBuffer(328));
    if (kind === "not-array") payload["orderParameter"] = Array(41).fill(0);
    if (kind === "short-order") payload["orderParameter"] = new Float64Array(40);
    if (kind === "short-theta") payload["thetaFinal"] = new Float64Array(1);
    if (kind === "nonfinite-order") payload["orderParameter"] = new Float64Array(41).fill(Number.NaN);
    if (kind === "nonfinite-theta") payload["thetaFinal"] = new Float64Array(2).fill(Number.POSITIVE_INFINITY);
    if (kind === "shared-output") payload["orderParameter"] = new Float64Array(new SharedArrayBuffer(328));
    deliver({ ...event, payload });
  });
  expect(await createOwnedKuramotoRun(options({ workerFactory: factory, onEvent: event => observed.push(event.kind) })).result).toMatchObject({ ok: false, code: "failed", reason: "worker source output shape or values refused", disposed: true });
  expect(observed).not.toContain("result");
  expect(input.theta0).toEqual([0, 0.8]);
});

it("native binary copies cannot invoke an overridden slice or detach a retained Buffer", async () => {
  let calls = 0;
  const source = Buffer.from(wasm);
  Object.defineProperty(source, "slice", { value() { calls++; return source; } });
  const retained = Buffer.from(source);
  expect(await createOwnedKuramotoRun(options({ wasmBytes: source })).result).toMatchObject({ ok: true, disposed: true });
  expect(calls).toBe(0);
  expect(source).toEqual(retained);
  expect(source.byteLength).toBe(wasm.byteLength);
});

it("an oversized binary cannot falsify admission through a shadow byteLength getter", async () => {
  let reads = 0;
  const source = new Uint8Array(2 * 1024 * 1024 + 1);
  Object.defineProperty(source, "byteLength", { get() { reads++; return 1; } });
  const started = BuiltKernelWorker.started;
  expect(await createOwnedKuramotoRun(options({ wasmBytes: source })).result).toMatchObject({ ok: false, code: "refused", disposed: true });
  expect(reads).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("retains visible diagnostics for missing producer reasons and actual native worker bootstrap errors", async () => {
  const failed = createOwnedKuramotoRun(options({ wasmBytes: new Uint8Array([1, 2, 3]), workerFactory: () => intercept((event, deliver) => { const payload = { ...event.payload }; delete payload["reason"]; deliver({ ...event, payload }); }) }));
  expect(await failed.result).toMatchObject({ ok: false, reason: "worker failure reason missing", disposed: true });
  const nativeFailure = createOwnedKuramotoRun(options({ workerFactory: () => new BuiltKernelWorker(resolve(process.env["STUDIO_KERNEL_WORKER_ENTRY"]!, "../missing-native-entry.js")) }));
  const outcome = await nativeFailure.result;
  expect(outcome).toMatchObject({ ok: false, code: "failed", disposed: true });
  if (outcome.ok) throw new Error("native startup unexpectedly succeeded");
  expect(outcome.reason).toContain("Cannot find module");
});

it("the default browser port forwards an actual native bootstrap failure", async () => {
  const missing = resolve(process.env["STUDIO_KERNEL_WORKER_ENTRY"]!, "../missing-default-native-entry.js");
  class FailingNativeWorker extends BuiltKernelWorker {
    constructor() { super(missing); }
  }
  vi.stubGlobal("Worker", FailingNativeWorker);
  try {
    const declaration = options();
    Reflect.deleteProperty(declaration, "workerFactory");
    const outcome = await createOwnedKuramotoRun(declaration).result;
    expect(outcome).toMatchObject({ ok: false, code: "failed", disposed: true });
    if (outcome.ok) throw new Error("missing actual native entry succeeded");
    expect(outcome.reason).toContain("Cannot find module");
  } finally { vi.unstubAllGlobals(); }
});

it("callbacks already queued before cancellation cannot restore a disposed real run", async () => {
  const native = new BuiltKernelWorker();
  const port: KernelWorkerPort = { onmessage: null, onerror: null, postMessage: (message, transfers) => native.postMessage(message, transfers), terminate: () => native.terminate() };
  let receiver: KernelWorkerPort["onmessage"] = null;
  let errorReceiver: KernelWorkerPort["onerror"] = null;
  let captured: unknown;
  let owned!: OwnedKuramotoHandle;
  native.onmessage = event => {
    receiver = port.onmessage;
    errorReceiver = port.onerror;
    captured = event.data;
    receiver?.(event);
    void owned.cancel();
  };
  owned = createOwnedKuramotoRun(options({ workerFactory: () => port }));
  expect(await owned.result).toMatchObject({ ok: false, code: "cancelled", disposed: true });
  const late = receiver as KernelWorkerPort["onmessage"];
  const lateError = errorReceiver as KernelWorkerPort["onerror"];
  if (!late || !lateError) throw new Error("actual pre-disposal receivers were not captured");
  late({ data: captured });
  const failing = new BuiltKernelWorker(resolve(process.env["STUDIO_KERNEL_WORKER_ENTRY"]!, "../missing-late-native-entry.js"));
  try {
    await new Promise<void>(resolve => { failing.onerror = event => { lateError(event); resolve(); }; });
  } finally { await failing.terminate(); }
  expect(await owned.result).toMatchObject({ ok: false, code: "cancelled", disposed: true });
});

it.each(["accepted", "progress", "result", "cancelled"] as const)("observer failure in %s retains actual disposal and no accepted result", async terminal => {
  let owned!: OwnedKuramotoHandle;
  owned = createOwnedKuramotoRun(options({ onEvent: event => {
    if (event.kind === terminal) throw new Error("observer rejected actual event");
    if (terminal === "cancelled" && event.kind === "accepted") void owned.cancel();
  } }));
  expect(await owned.result).toMatchObject({ ok: false, code: "failed", reason: "observer rejected actual event", disposed: true });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("reentrant cancellation after native acceptance cannot be overwritten by a later real result", async () => {
  let owned!: OwnedKuramotoHandle;
  owned = createOwnedKuramotoRun(options({ onEvent: event => { if (event.kind === "accepted") { void owned.cancel(); void owned.dispose(); } } }));
  expect(await owned.result).toMatchObject({ ok: false, code: "cancelled", disposed: true });
  await owned.cancel();
});

it("keeps an unobserved disposal explicit and permits the actual host to reap its remaining thread", async () => {
  const native = new BuiltKernelWorker();
  const port: KernelWorkerPort = { onmessage: null, onerror: null, postMessage: (message, transfers) => native.postMessage(message, transfers), terminate() { throw new Error("host refused termination"); } };
  native.onmessage = event => port.onmessage?.(event);
  native.onerror = event => port.onerror?.(event);
  const run = createOwnedKuramotoRun(options({ workerFactory: () => port }));
  try {
    expect(await run.result).toMatchObject({ ok: false, code: "failed", reason: "worker disposal failed: host refused termination", disposed: false });
    expect(BuiltKernelWorker.activeCount).toBe(1);
  } finally { await native.terminate(); }
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses unknown source data, oversized binaries, unavailable hosts and unsafe vectors before allocation", async () => {
  const started = BuiltKernelWorker.started;
  let reads = 0;
  const accessor = [0.2, 0.2];
  Object.defineProperty(accessor, "0", { get() { reads++; return 0.2; } });
  const decorated = [0.2, 0.2];
  Object.defineProperty(decorated, Symbol.iterator, { value() { reads++; throw new Error("unsafe iterator"); } });
  const wrongPrototype = [0.2, 0.2];
  Object.setPrototypeOf(wrongPrototype, null);
  const invalid: Partial<OwnedKuramotoOptions>[] = [
    { deadlineMs: 0 }, { deadlineMs: 60001 }, { deadlineMs: Number.NaN }, { wasmBytes: new Uint8Array(2 * 1024 * 1024 + 1) },
    { wasmBytes: new Float32Array(1) as unknown as Uint8Array },
    { resourcePolicy: { source: "declared refusal", memoryBytes: 0n, overheadBytes: 0n, addressableBytes: 0xffff_ffffn, workUnits: 1000000n } },
    { request: { ...input, omega: accessor } }, { request: { ...input, omega: decorated } }, { request: { ...input, omega: wrongPrototype } },
    { request: { ...input, theta0: [] } }, { request: { ...input, omega: [Number.NaN, 0] } },
    { workerFactory() { throw new Error("real host unavailable"); } }, { workerFactory() { throw "host unavailable"; } },
  ];
  for (const change of invalid) expect(await createOwnedKuramotoRun(options(change)).result).toMatchObject({ ok: false, code: "refused", disposed: true });
  expect(reads).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
});
