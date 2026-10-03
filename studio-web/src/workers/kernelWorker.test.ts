// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — whole public owned-worker acceptance against real WASM

import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { MessageChannel } from "node:worker_threads";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as kernelFacade from "../panel/kuramoto";
import type { KuramotoRequest, OwnedKuramotoHandle, OwnedKuramotoOptions } from "../panel/kuramoto";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { installKernelWorker } from "./kernelWorker";
import { readWorkerEvent } from "./kernelProtocol";
import type { KernelWorkerEvent } from "./kernelProtocol";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });

const wasm = new Uint8Array(readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm")));
const fingerprint = createHash("sha256").update(wasm).digest("hex");
const request: KuramotoRequest = { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, steps: 40, dt: 0.01 };
const revisionHash = "a".repeat(64);
const planHash = "b".repeat(64);

/** Invoke the implemented production facade using its actual public declarations. */
function start(overrides: Partial<OwnedKuramotoOptions> = {}): OwnedKuramotoHandle {
  return kernelFacade.createOwnedKuramotoRun({ runId: "owned-positive", revisionHash, planHash, request, wasmBytes: wasm, buildFingerprint: fingerprint, bounds: { maxOscillators: 128, maxSteps: 4096 }, deadlineMs: 5000, workerFactory: () => new BuiltKernelWorker(), ...overrides });
}

describe("whole owned kernel worker public acceptance", () => {
  it("test_owned_kernel_worker_01: real two-node WASM matches the independent analytic phase difference and source reference", async () => {
    const outcome = await start().result;
    expect(outcome.ok, JSON.stringify(outcome)).toBe(true);
    expect(outcome.disposed).toBe(true);
    if (!outcome.ok) throw new Error(outcome.reason);
    const delta = 2 * Math.atan(Math.tan(0.4) * Math.exp(-1.4 * 0.4));
    expect(outcome.run.thetaFinal[1]! - outcome.run.thetaFinal[0]!).toBeCloseTo(delta, 8);
    expect(outcome.run.orderParameter[40]).toBeCloseTo(Math.cos(delta / 2), 8);
    const reference = await kernelFacade.instantiateKuramoto(wasm);
    const expected = reference.simulate(request);
    if (!expected.ok) throw new Error(expected.reason);
    expect(Array.from(outcome.run.orderParameter)).toEqual(Array.from(expected.run.orderParameter));
    expect(Array.from(outcome.run.thetaFinal)).toEqual(Array.from(expected.run.thetaFinal));
    expect(outcome.runId).toBe("owned-positive");
    expect(outcome.revisionHash).toBe(revisionHash);
    expect(outcome.planHash).toBe(planHash);
    expect(outcome.buildFingerprint).toBe(fingerprint);
  });

  it("test_owned_kernel_worker_02: cancellation and disposal acknowledge only a disposed real thread", async () => {
    for (const operation of ["cancel", "dispose"] as const) {
      const run = start({ runId: `owned-${operation}` });
      await run[operation]();
      expect(await run.result).toMatchObject({ ok: false, code: "cancelled", disposed: true });
      await run[operation]();
    }
    const run = start({ runId: "owned-deadline", deadlineMs: 1 });
    expect(await run.result).toMatchObject({ ok: false, code: "timeout", disposed: true });
  });

  it("test_owned_kernel_worker_03: absent, malformed and mismatched WASM visibly refuse without a trajectory", async () => {
    for (const bytes of [new Uint8Array(), new Uint8Array([1, 2, 3]), wasm]) {
      const outcome = await start({ runId: "owned-binary-refusal", wasmBytes: bytes, buildFingerprint: "f".repeat(64) }).result;
      expect(outcome.ok).toBe(false);
      expect(outcome.disposed).toBe(true);
      expect("run" in outcome).toBe(false);
    }
  });

  it("test_owned_kernel_worker_04: distinct run/revision/plan identities cannot exchange their real results", async () => {
    const otherRequest: KuramotoRequest = { ...request, theta0: [1, 1.2] };
    const [a, b] = await Promise.all([start({ runId: "first" }).result, start({ runId: "second", revisionHash: "c".repeat(64), planHash: "d".repeat(64), request: otherRequest }).result]);
    expect(a).toMatchObject({ ok: true, disposed: true, runId: "first", revisionHash, planHash });
    expect(b).toMatchObject({ ok: true, disposed: true, runId: "second", revisionHash: "c".repeat(64), planHash: "d".repeat(64) });
    if (!a.ok || !b.ok) throw new Error("distinct original real worker outcomes missing");
    expect(a.run.thetaFinal[0]).not.toBe(b.run.thetaFinal[0]);
  });

  it("test_owned_kernel_worker_05: transfers cannot detach or change retained source vectors or earlier results", async () => {
    const initial = [...request.theta0];
    const sourceBytes = wasm.slice();
    const first = await start({ wasmBytes: sourceBytes }).result;
    if (!first.ok) throw new Error(first.reason);
    const retained = first.run.orderParameter.slice();
    await start({ runId: "later", wasmBytes: sourceBytes, request: { ...request, coupling: 0 } }).result;
    expect(sourceBytes.byteLength).toBe(wasm.byteLength);
    expect(sourceBytes).toEqual(wasm);
    expect(request.theta0).toEqual(initial);
    expect(first.run.orderParameter).toEqual(retained);
  });

  it("refuses malformed and oversized requests before allocating an owned thread", async () => {
    let workers = 0;
    const factory = () => { workers++; return new BuiltKernelWorker(); };
    for (const bad of [{ ...request, steps: 4097 }, { ...request, dt: Number.NaN }, { ...request, theta0: [] }, { ...request, mode: "networked" as const, kNm: [1] }]) {
      expect(await start({ request: bad, workerFactory: factory }).result).toMatchObject({ ok: false, disposed: true });
    }
    expect(workers).toBe(0);
    expect(request.theta0).toEqual([0, 0.8]);
  });
});

function declaration(): Record<string, unknown> {
  return { wasm: wasm.slice(), build_fingerprint: fingerprint, input: { ...request, omega: new Float64Array(request.omega), theta0: new Float64Array(request.theta0), kNm: new Float64Array() }, bounds: { maxOscillators: 128, maxSteps: 4096 }, policy: { source: "actual source runtime admission", addressableBytes: 0xffff_ffffn, memoryBytes: 4n * 1024n * 1024n, workUnits: 1000000000n, overheadBytes: 0n } };
}

function envelope(command: "validate" | "run" | "cancel", value: unknown = null): Record<string, unknown> {
  return { version: 1, run_id: "source-runtime", revision_hash: revisionHash, plan_hash: planHash, command, payload: value };
}

function sourceRuntime(failFirstEmission = false) {
  const channel = new MessageChannel();
  const queue: KernelWorkerEvent[] = [];
  const waiting: ((event: KernelWorkerEvent) => void)[] = [];
  const timers = new Set<ReturnType<typeof setTimeout>>();
  channel.port1.on("message", (raw: unknown) => {
    const event = readWorkerEvent(raw);
    const receiver = waiting.shift();
    if (receiver) receiver(event); else queue.push(event);
  });
  installKernelWorker({
    addEventListener(_, listener) { channel.port2.on("message", (data: unknown) => listener({ data })); },
    postMessage(value, transfers = []) {
      if (failFirstEmission) { failFirstEmission = false; throw "host refused emission"; }
      channel.port2.postMessage(value, transfers);
    },
  });
  return {
    send(value: unknown) { channel.port1.postMessage(value); },
    next(): Promise<KernelWorkerEvent> {
      const event = queue.shift();
      if (event) return Promise.resolve(event);
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("real source transport did not emit an event")), 5000);
        timers.add(timeout);
        waiting.push(event => { clearTimeout(timeout); timers.delete(timeout); resolve(event); });
      });
    },
    close() { for (const timeout of timers) clearTimeout(timeout); channel.port1.close(); channel.port2.close(); },
  };
}

describe("public source entry with native MessageChannel and unchanged WASM", () => {
  it("executes the actual mean-field and networked source codecs and transfers only original owned results", async () => {
    for (const mode of ["mean-field", "networked"] as const) {
      const runtime = sourceRuntime();
      try {
        const data = declaration();
        const original = data["input"] as Record<string, unknown>;
        data["input"] = { ...original, mode, kNm: mode === "networked" ? new Float64Array([0, 0.7, 0.7, 0]) : original["kNm"] };
        runtime.send(envelope("validate", data));
        expect(await runtime.next()).toMatchObject({ kind: "accepted", sequence: 1, payload: { build_fingerprint: fingerprint } });
        runtime.send(envelope("run"));
        expect(await runtime.next()).toMatchObject({ kind: "progress", sequence: 2 });
        const result = await runtime.next();
        expect(result).toMatchObject({ kind: "result", sequence: 3, run_id: "source-runtime" });
        expect((result.payload["orderParameter"] as Float64Array).length).toBe(41);
        expect((result.payload["thetaFinal"] as Float64Array).length).toBe(2);
        runtime.send(envelope("run"));
        expect(await runtime.next()).toMatchObject({ kind: "failed", payload: { reason: "run requires accepted source validation" } });
      } finally { runtime.close(); }
    }
  });

  it.each(["missing", "malformed", "shared-binary", "huge-binary", "mode", "scalar", "vector", "shared-vector", "shape", "source-dt", "policy", "digest", "bounds"])("rejects malformed source declaration %s without a substituted trajectory", async kind => {
    const runtime = sourceRuntime();
    try {
      const data = declaration();
      const fields = data["input"] as Record<string, unknown>;
      if (kind === "missing") data["wasm"] = new Uint8Array();
      if (kind === "malformed") { data["wasm"] = new Uint8Array([1, 2, 3]); data["build_fingerprint"] = createHash("sha256").update(data["wasm"] as Uint8Array).digest("hex"); }
      if (kind === "shared-binary") data["wasm"] = new Uint8Array(new SharedArrayBuffer(8));
      if (kind === "huge-binary") data["wasm"] = new Uint8Array(2 * 1024 * 1024 + 1);
      if (kind === "mode") fields["mode"] = "unsupported";
      if (kind === "scalar") fields["dt"] = "0.01";
      if (kind === "vector") fields["omega"] = [0.2, 0.2];
      if (kind === "shared-vector") fields["omega"] = new Float64Array(new SharedArrayBuffer(16));
      if (kind === "shape") fields["theta0"] = new Float64Array();
      if (kind === "source-dt") fields["dt"] = 0;
      if (kind === "policy") data["policy"] = { ...(data["policy"] as object), memoryBytes: 0n };
      if (kind === "digest") data["build_fingerprint"] = "f".repeat(64);
      if (kind === "bounds") data["bounds"] = { maxOscillators: 127, maxSteps: 4096 };
      runtime.send(envelope("validate", data));
      expect(await runtime.next()).toMatchObject({ kind: "failed", run_id: "source-runtime" });
      expect(wasm.byteLength).toBeGreaterThan(0);
      expect(request.theta0).toEqual([0, 0.8]);
    } finally { runtime.close(); }
  });

  it("preserves the current native source on foreign, premature and replacement commands", async () => {
    const runtime = sourceRuntime();
    try {
      runtime.send(envelope("run"));
      expect(await runtime.next()).toMatchObject({ kind: "failed", payload: { reason: "worker command does not own this run revision" } });
      runtime.send(envelope("validate", declaration()));
      expect(await runtime.next()).toMatchObject({ kind: "accepted" });
      runtime.send({ ...envelope("validate", declaration()), run_id: "foreign" });
      expect(await runtime.next()).toMatchObject({ kind: "failed", sequence: 1, run_id: "foreign" });
      for (const command of [{ ...envelope("run"), run_id: "foreign" }, { ...envelope("run"), revision_hash: "f".repeat(64) }, { ...envelope("run"), plan_hash: "f".repeat(64) }, envelope("run", declaration())]) {
        runtime.send(command);
        expect(await runtime.next()).toMatchObject({ kind: "failed" });
      }
      runtime.send(envelope("run"));
      expect(await runtime.next()).toMatchObject({ kind: "progress" });
      expect(await runtime.next()).toMatchObject({ kind: "result" });
    } finally { runtime.close(); }
  });

  it("a cancellation during real source validation invalidates it and never acknowledges host disposal", async () => {
    const runtime = sourceRuntime();
    try {
      runtime.send(envelope("validate", declaration()));
      runtime.send(envelope("cancel"));
      expect(await runtime.next()).toMatchObject({ kind: "progress", payload: { stage: "cancellation requested; disposing owner must terminate" } });
      runtime.send(envelope("run"));
      expect(await runtime.next()).toMatchObject({ kind: "failed", payload: { reason: "run requires accepted source validation" } });
    } finally { runtime.close(); }
  });

  it("retains the original finite-source numerical failure and an unknown host emission diagnostic", async () => {
    for (const failedHost of [false, true]) {
      const runtime = sourceRuntime(failedHost);
      try {
        const data = declaration();
        if (!failedHost) data["input"] = { ...(data["input"] as object), coupling: Number.MAX_VALUE, dt: 1 };
        runtime.send(envelope("validate", data));
        const first = await runtime.next();
        if (failedHost) expect(first).toMatchObject({ kind: "failed", payload: { reason: "worker execution failed" } });
        else {
          expect(first.kind).toBe("accepted");
          runtime.send(envelope("run"));
          expect(await runtime.next()).toMatchObject({ kind: "progress" });
          expect(await runtime.next()).toMatchObject({ kind: "failed" });
        }
      } finally { runtime.close(); }
    }
  });
});
