// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original worker-bound experiment lifecycle

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { instantiateKuramoto } from "../../panel/kuramoto";
import { appendExperimentAttempt, createLocalExperiment, readLocalExperiment } from "./experimentArchive";
import { localExperimentCodecs } from "./kuramotoArtifacts";
import { prepareExperimentPlan, prepareSavedReplay } from "./experimentPlan";
import { appendParameterRevision, parameterSourceFromArchive } from "../parameters/parameterRevision";
import { createParameterDraft, parameterDraftReducer } from "../parameters/parameterDraft";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import { useExperimentRun } from "./useExperimentRun";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const wasm = new Uint8Array(readFileSync(process.env["STUDIO_EXPERIMENT_WASM_PATH"] ?? "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));

async function originalPlan(memoryBudget?: string) {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "independent analytic fixture", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  return prepareExperimentPlan(source, kernel, memoryBudget === undefined ? {} : { memoryBudget });
}

it("a genuine disposed native worker result remains bound to the source document and analytic model", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan); });
  await act(async () => { await view.result.current.run(); });
  const outcome = view.result.current.outcome;
  expect(view.result.current.status).toBe("succeeded");
  expect(outcome?.ok).toBe(true);
  if (!outcome?.ok) throw new Error("real native output missing");
  const delta = 2 * Math.atan(Math.tan(0.4) * Math.exp(-1.4 * 0.4));
  expect(outcome.run.thetaFinal[1]! - outcome.run.thetaFinal[0]!).toBeCloseTo(delta, 8);
  expect(outcome.revisionHash).toBe(plan.revisionHash);
  expect(view.result.current.events.at(-1)).toMatchObject({ kind: "result", payload: { disposed: true } });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("test_local_experiment_journey_02: a refused original resource plan creates no native thread", async () => {
  const plan = await originalPlan("0");
  const started = BuiltKernelWorker.started;
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan); });
  await act(async () => { await view.result.current.run(); });
  expect(view.result.current.status).toBe("refused");
  expect(view.result.current.outcome).toBeNull();
  expect(BuiltKernelWorker.started).toBe(started);
});

it("retains a genuine result after adding its archive record, but rejects a changed revision", async () => {
  const plan = await originalPlan();
  const view = renderHook(({ source }) => useExperimentRun(source, true, { workerFactory: () => new BuiltKernelWorker() }), { initialProps: { source: plan.sourceJson } });
  await act(async () => { await view.result.current.prepare(plan); });
  await act(async () => { await view.result.current.run(); });
  const attempt = view.result.current;
  if (!attempt.outcome) throw new Error("real original attempt missing");
  const archive = await appendExperimentAttempt(plan, attempt.runId, attempt.attemptId, attempt.events, attempt.outcome, localExperimentCodecs);
  view.rerender({ source: archive.json });
  await act(async () => { await view.result.current.retainSavedAttempt(archive, plan.sourceJson, localExperimentCodecs); });
  expect(view.result.current.status).toBe("succeeded");
  expect(view.result.current.outcome).toEqual(attempt.outcome);
  expect(view.result.current.plan?.revisionHash).toBe(plan.revisionHash);
  view.rerender({ source: "changed revision, not an admitted archive" });
  expect(view.result.current.outcome).toBeNull();
  expect(view.result.current.canSave).toBe(false);
});

it("test_local_experiment_journey_01: independent import replays original raw output exactly", async () => {
  const plan = await originalPlan();
  const first = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await first.result.current.prepare(plan); });
  await act(async () => { await first.result.current.run(); });
  const attempt = first.result.current;
  if (!attempt.outcome?.ok) throw new Error("genuine original result required");
  const saved = await appendExperimentAttempt(plan, attempt.runId, attempt.attemptId, attempt.events, attempt.outcome, localExperimentCodecs);
  first.unmount();
  const replay = await prepareSavedReplay(saved.json, await instantiateKuramoto(wasm), localExperimentCodecs);
  expect(replay.plan.planHash).toBe(plan.planHash);
  const fresh = renderHook(() => useExperimentRun(saved.json, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await fresh.result.current.prepare(replay.plan, replay.expected); });
  await act(async () => { await fresh.result.current.run(); });
  expect(fresh.result.current.replayVerified).toBe(true);
  expect(fresh.result.current.outcome?.ok).toBe(true);
  expect(fresh.result.current.runId).not.toBe(attempt.runId);
  expect(fresh.result.current.attemptId).not.toBe(attempt.attemptId);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("test_local_experiment_journey_03: immediate cancellation retains source and real disposal diagnostics", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan); });
  await act(async () => { const pending = view.result.current.run(); await view.result.current.cancel(); await pending; });
  expect(view.result.current.status).toBe("cancelled");
  expect(view.result.current.plan?.sourceJson).toBe(plan.sourceJson);
  expect(view.result.current.events.at(-1)).toMatchObject({ kind: "cancelled", payload: { disposed: true } });
  expect(view.result.current.canSave).toBe(true);
  const attempt = view.result.current;
  const archived = await appendExperimentAttempt(plan, attempt.runId, attempt.attemptId, attempt.events, attempt.outcome!, localExperimentCodecs);
  await expect(prepareSavedReplay(archived.json, await instantiateKuramoto(wasm), localExperimentCodecs)).rejects.toThrow("no disposed successful");
  await act(async () => { await view.result.current.cancel(); });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each(["revision", "route", "unmount"])("test_local_experiment_journey_04: a real pending worker is disposed on %s change", async boundary => {
  const plan = await originalPlan();
  const view = renderHook(({ source, active }) => useExperimentRun(source, active, { workerFactory: () => new BuiltKernelWorker() }), { initialProps: { source: plan.sourceJson, active: true } });
  await act(async () => { await view.result.current.prepare(plan); });
  let pending = Promise.resolve();
  act(() => { pending = view.result.current.run(); });
  expect(BuiltKernelWorker.activeCount).toBe(1);
  if (boundary === "unmount") view.unmount();
  else view.rerender({ source: boundary === "revision" ? "edited source" : plan.sourceJson, active: boundary !== "route" });
  await act(async () => { await pending; });
  expect(BuiltKernelWorker.activeCount).toBe(0);
  if (boundary === "revision") {
    expect(view.result.current.status).toBe("stale");
    expect(view.result.current.outcome).toBeNull();
    expect(view.result.current.canSave).toBe(false);
    await expect(view.result.current.run()).rejects.toThrow("current source-bound");
  } else if (boundary === "route") {
    expect(view.result.current.canSave).toBe(false);
    view.rerender({ source: plan.sourceJson, active: true });
    expect(view.result.current.status).toBe("cancelled");
    expect(view.result.current.events.at(-1)?.kind).toBe("cancelled");
  }
});

it("records real native import failure and timeout as failed disposed attempts", async () => {
  const original = await originalPlan();
  for (const mode of ["missing-entry", "timeout"] as const) {
    const plan = mode === "timeout" ? await prepareExperimentPlan(await readLocalExperiment(original.sourceJson, localExperimentCodecs), await instantiateKuramoto(wasm), { deadlineMs: 1 }) : original;
    const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker(mode === "missing-entry" ? process.env["STUDIO_KERNEL_WORKER_ENTRY"]! + ".missing" : undefined) }));
    await act(async () => { await view.result.current.prepare(plan); });
    await act(async () => { await view.result.current.run(); });
    expect(view.result.current.status).toBe("failed");
    expect(view.result.current.events.at(-1)).toMatchObject({ kind: "failed", payload: { disposed: true, origin: "original client outcome" } });
    const outcome = view.result.current.outcome;
    expect(outcome).toMatchObject({ ok: false, code: mode === "timeout" ? "timeout" : "failed", disposed: true });
    const attempt = view.result.current;
    const archive = await appendExperimentAttempt(plan, attempt.runId, attempt.attemptId, attempt.events, outcome!, localExperimentCodecs);
    await expect(prepareSavedReplay(archive.json, await instantiateKuramoto(wasm), localExperimentCodecs)).rejects.toThrow("no disposed successful");
    view.unmount();
  }
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("a declared replay mismatch never becomes success despite a genuine native result", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan, { orderParameter: [], thetaFinal: [] }); });
  await act(async () => { await view.result.current.run(); });
  expect(view.result.current.status).toBe("failed");
  expect(view.result.current.replayVerified).toBe(false);
  expect(view.result.current.events.slice(-2).map(event => event.kind)).toEqual(["result", "failed"]);
  const attempt = view.result.current;
  const archived = await appendExperimentAttempt(plan, attempt.runId, attempt.attemptId, attempt.events, attempt.outcome!, localExperimentCodecs);
  await expect(prepareSavedReplay(archived.json, await instantiateKuramoto(wasm), localExperimentCodecs)).rejects.toThrow("no disposed successful");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("unconfirmed native disposal blocks all subsequent execution and archiving", async () => {
  const plan = await originalPlan();
  const native = new BuiltKernelWorker();
  const port: KernelWorkerPort = { onmessage: null, onerror: null, postMessage: (message, transfers) => native.postMessage(message, transfers), terminate() { throw new Error("negative native host disposal fault"); } };
  native.onmessage = event => port.onmessage?.(event); native.onerror = event => port.onerror?.(event);
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => port }));
  try {
    await act(async () => { await view.result.current.prepare(plan); });
    await act(async () => { await view.result.current.run(); });
    expect(view.result.current.disposalBlocked).toBe(true);
    expect(view.result.current.canRun).toBe(false);
    expect(view.result.current.canSave).toBe(false);
    await expect(view.result.current.run()).rejects.toThrow("disposal is unconfirmed");
    await expect(view.result.current.prepare(plan)).rejects.toThrow("disposal unconfirmed");
  } finally { await native.terminate(); view.unmount(); }
});

it("refuses inactive or changed source controls and concurrent execution without allocating another worker", async () => {
  const plan = await originalPlan();
  const started = BuiltKernelWorker.started;
  const view = renderHook(({ source, active }) => useExperimentRun(source, active, { workerFactory: () => new BuiltKernelWorker() }), { initialProps: { source: plan.sourceJson, active: false } });
  await expect(view.result.current.run()).rejects.toThrow("current source-bound");
  await expect(view.result.current.prepare(plan)).rejects.toThrow("inactive or changed");
  const source = await readLocalExperiment(plan.sourceJson, localExperimentCodecs);
  await expect(view.result.current.retainSavedAttempt(source.archive.preview, plan.sourceJson, localExperimentCodecs)).rejects.toThrow("disposed unchanged-source attempt");
  view.rerender({ source: "unadmitted edited draft", active: true });
  await expect(view.result.current.prepare(plan)).rejects.toThrow("inactive or changed");
  expect(BuiltKernelWorker.started).toBe(started);
  view.rerender({ source: plan.sourceJson, active: true });
  await act(async () => { await view.result.current.prepare(plan); });
  let pending = Promise.resolve();
  act(() => { pending = view.result.current.run(); });
  await expect(view.result.current.run()).rejects.toThrow("prior original worker is active");
  await expect(view.result.current.retainSavedAttempt(source.archive.preview, plan.sourceJson, localExperimentCodecs)).rejects.toThrow("disposed unchanged-source attempt");
  expect(BuiltKernelWorker.started).toBe(started + 1);
  await act(async () => { await view.result.current.cancel(); await pending; });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each(["source", "route", "unmount"])("observes a %s change while admitting a plan before any allocation", async boundary => {
  const plan = await originalPlan();
  const started = BuiltKernelWorker.started;
  const view = renderHook(({ source, active }) => useExperimentRun(source, active), { initialProps: { source: plan.sourceJson, active: true } });
  const pending = view.result.current.prepare(plan);
  if (boundary === "unmount") view.unmount();
  else view.rerender({ source: boundary === "source" ? "changed draft" : plan.sourceJson, active: boundary !== "route" });
  await expect(pending).rejects.toThrow("source changed while observing");
  expect(BuiltKernelWorker.started).toBe(started);
});

it("a previously captured run control cannot allocate after its owning workbench unmounts", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan); });
  const captured = view.result.current;
  const started = BuiltKernelWorker.started;
  view.unmount();
  await expect(captured.run()).rejects.toThrow("source changed before worker allocation");
  expect(BuiltKernelWorker.started).toBe(started);
});

it("rejects retaining a real result against another admitted immutable parameter revision", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  await act(async () => { await view.result.current.prepare(plan); });
  await act(async () => { await view.result.current.run(); });
  const source = (await parameterSourceFromArchive(plan.sourceJson, localExperimentCodecs))!;
  const draft = parameterDraftReducer(createParameterDraft(source), { type: "value", key: "coupling", index: 0, text: "1.6", unit: "rad/model-time" });
  const child = await appendParameterRevision(plan.sourceJson, source, draft.snapshot, localExperimentCodecs, new Date().toISOString());
  expect((await readLocalExperiment(child.archive.json, localExperimentCodecs)).revisionHash).not.toBe(plan.revisionHash);
  await expect(view.result.current.retainSavedAttempt(child.archive, "another prior archive", localExperimentCodecs)).rejects.toThrow("disposed unchanged-source attempt");
  await expect(view.result.current.retainSavedAttempt(child.archive, plan.sourceJson, localExperimentCodecs)).rejects.toThrow("saved archive changed the immutable source");
  expect(view.result.current.plan?.sourceJson).toBe(plan.sourceJson);
  expect(view.result.current.status).toBe("succeeded");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("a replay expectation host fault preserves actual numerical diagnostics and emits a fixed failure", async () => {
  const plan = await originalPlan();
  const view = renderHook(() => useExperimentRun(plan.sourceJson, true, { workerFactory: () => new BuiltKernelWorker() }));
  const unavailable = { get orderParameter(): readonly string[] { throw new TypeError("private host comparison details"); }, thetaFinal: [] };
  await act(async () => { await view.result.current.prepare(plan, unavailable); });
  await act(async () => { await view.result.current.run(); });
  expect(view.result.current.status).toBe("failed");
  expect(view.result.current.reason).toBe("original replay comparison failed");
  expect(view.result.current.events.slice(-2).map(event => event.kind)).toEqual(["result", "failed"]);
  expect(view.result.current.events.at(-1)?.payload["reason"]).toBe("original replay comparison failed");
  expect(view.result.current.replayVerified).toBe(false);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
