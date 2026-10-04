// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual original producer source adapters

import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { createOwnedKuramotoRun, instantiateKuramoto } from "../../panel/kuramoto";
import { createLocalExperiment, readLocalExperiment, appendExperimentAttempt } from "../experiments/experimentArchive";
import { prepareExperimentPlan } from "../experiments/experimentPlan";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import type { KernelWorkerEvent } from "../../workers/kernelProtocol";
import { inspectAnalyseProducer, inspectKuramotoResult } from "./resultSources";
import { resultLimits } from "./resultModel";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });
const path = process.env["STUDIO_ANALYSE_RESULT_PATH"];
if (!path) throw new Error("Actual original CLI producer fixture path required");
const json = readFileSync(path, "utf8");
const wasmPath = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
if (!wasmPath) throw new Error("Actual original WASM path required");
const wasm = new Uint8Array(readFileSync(wasmPath));

it("reads an actually sealed original CLI export without inferring time or uncertainty", async () => {
  const result = await inspectAnalyseProducer(json);
  expect(result.sourceSha256).toBe(createHash("sha256").update(json, "utf8").digest("hex"));
  expect(result.panels[0]!.samples.map(sample => sample.x)).toEqual([0, 0.125, 2]);
  expect(result.panels[0]!.samples.map(sample => sample.value)).toEqual([1, 1, 1]);
  expect(result.panels[1]!.samples.map(sample => sample.value)).toEqual([0, 0, 0]);
  expect(result.panels.every(panel => panel.coordinateLabel === "Filtration threshold" && panel.coordinateUnit === "rad" && panel.valueDtype === "int64")).toBe(true);
  expect(result.panels.every(panel => panel.samples.every(sample => sample.interval === null))).toBe(true);
});

it.each(["", " ".repeat(resultLimits.importBytes + 1), "null", "[]", "{", '{"request":null}', '{"request":{"verb":"analyse"},"request":{}}'])("refuses malformed/ambiguous/over-budget imports %s", async text => {
  await expect(inspectAnalyseProducer(text)).rejects.toThrow();
});

it("refuses wrong verb/status/schema/claim and inexact producer integers while preserving the original export", async () => {
  const original = json;
  for (const change of [
    (record: Record<string, unknown>) => { record["request"] = { verb: "execute" }; },
    (record: Record<string, unknown>) => { record["plan"] = { verb: "simulate" }; },
    (record: Record<string, unknown>) => { record["result"] = { status: "failed" }; },
    (record: Record<string, unknown>) => { record["result"] = { status: "succeeded", outputs: { analysis_schema: "studio.sync-analysis.v2" } }; },
    (record: Record<string, unknown>) => { record["plan"] = { verb: "analyse", claim_boundary: "relabelled" }; },
  ]) {
    const record = JSON.parse(json) as Record<string, unknown>; change(record);
    await expect(inspectAnalyseProducer(JSON.stringify(record))).rejects.toThrow();
  }
  await expect(inspectAnalyseProducer(json.replace('"version": 1', '"version": 9007199254740993'))).rejects.toThrow("exactly");
  await expect(inspectAnalyseProducer(json.replace('"version": 1', '"version": -9007199254740993'))).rejects.toThrow("exactly");
  expect(json).toBe(original);
});

it("preserves real disposed WASM samples and the original saved output artifact digest", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.125, steps: 2 }, "analytically stationary source", localExperimentCodecs, "native test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const plan = await prepareExperimentPlan(source, kernel);
  const runId = crypto.randomUUID(), events: KernelWorkerEvent[] = [];
  const handle = createOwnedKuramotoRun({ runId, revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint, request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs, workerFactory: () => new BuiltKernelWorker(), onEvent: event => events.push(event) });
  const outcome = await handle.result;
  expect(outcome.ok).toBe(true);
  if (!outcome.ok) throw new Error(outcome.reason);
  const result = await inspectKuramotoResult(plan, outcome);
  const saved = await appendExperimentAttempt(plan, runId, crypto.randomUUID(), events, outcome, localExperimentCodecs);
  const restored = await readLocalExperiment(saved.json, localExperimentCodecs);
  expect(result.sourceSha256).toBe(restored.archive.members.find(member => member.schema === "studio.kuramoto-output.v1")!.sha256);
  expect(result.panels[0]!.samples.map(sample => sample.value)).toEqual([1, 1, 1]);
  expect(result.panels[0]!.samples.map(sample => sample.coordinate)).toEqual([0, 0.125, 0.25]);
  expect(result.panels[1]!.samples.map(sample => sample.value)).toEqual([0, 0]);
  expect(result.panels[1]!.samples.every(sample => sample.coordinate === 0.25)).toBe(true);
  expect(result.caption).toContain("kernel does not report measured timestamps");
  expect(BuiltKernelWorker.activeCount).toBe(0);
  const started = BuiltKernelWorker.started;
  await expect(inspectKuramotoResult(plan, { ok: false, code: "cancelled", reason: "fault-injected refusal", disposed: true })).rejects.toThrow("Disposed");
  for (const key of ["revisionHash", "planHash", "buildFingerprint"] as const) await expect(inspectKuramotoResult(plan, { ...outcome, [key]: "a".repeat(64) })).rejects.toThrow("Disposed");
  await expect(Reflect.apply(inspectKuramotoResult, undefined, [plan, { ...outcome, disposed: false }])).rejects.toThrow("Disposed");
  for (const run of [{ ...outcome.run, orderParameter: new Float64Array(1) }, { ...outcome.run, thetaFinal: new Float64Array(1) }]) await expect(inspectKuramotoResult(plan, { ...outcome, run })).rejects.toThrow("shape");
  expect(BuiltKernelWorker.started).toBe(started);
  expect(source.archive.preview.json).toBe(archive.json);
});
