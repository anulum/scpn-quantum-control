// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original numerical plan admission

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { createOwnedKuramotoRun, instantiateKuramoto } from "../../panel/kuramoto";
import { documentDigest, readJson, writeJson } from "../../shared/contracts";
import type { WorkspaceDocument } from "../../shared/contracts";
import type { WorkspaceArchiveMember, WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import type { KernelWorkerEvent } from "../../workers/kernelProtocol";
import { appendExperimentAttempt, createLocalExperiment, readLocalExperiment } from "./experimentArchive";
import { artifactContent, localExperimentCodecs, makeArtifact, readExperimentArtifact } from "./kuramotoArtifacts";
import { prepareExperimentPlan, prepareSavedReplay, verifyReplayOutcome } from "./experimentPlan";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });
const wasm = new Uint8Array(readFileSync(process.env["STUDIO_EXPERIMENT_WASM_PATH"] ?? "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));

it("binds the actual source revision, build and complete original resource plan exactly", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "independent analytic fixture", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const first = await prepareExperimentPlan(source, kernel);
  const second = await prepareExperimentPlan(source, kernel);
  expect(first.planHash).toBe(second.planHash);
  expect(first.revisionHash).toBe(source.revisionHash);
  expect(first.admission.allowed).toBe(true);
  expect(first.admission.estimate.components.some(value => value.name === "retained_and_transferred_kernel_binary")).toBe(true);
  expect(first.admission.bytesRequired).toBeGreaterThan(2n * BigInt(wasm.length));
  const refused = await prepareExperimentPlan(source, kernel, { memoryBudget: "0" });
  expect(refused.admission.allowed).toBe(false);
  expect(refused.admission.blockers).toContain("declared_storage_exceeds_budget");
  expect(refused.planHash).not.toBe(first.planHash);
  await expect(prepareExperimentPlan(source, kernel, { memoryBudget: "4194305" })).rejects.toThrow("policy ceiling");
  await expect(prepareExperimentPlan(source, kernel, { memoryBudget: "01" })).rejects.toThrow("canonical");
  await expect(prepareExperimentPlan(source, { ...kernel, sourceBytes: new Uint8Array([0]) })).rejects.toThrow("shipped kernel");
  expect(source.archive.preview.json).toBe(archive.json);
});

it("refuses a changed native environment, out-of-policy source and invalid operational deadline before allocation", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "declared admission source", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const started = BuiltKernelWorker.started;
  await expect(prepareExperimentPlan(source, { simulate: kernel.simulate, bounds: kernel.bounds })).rejects.toThrow("shipped kernel");
  await expect(prepareExperimentPlan(source, { ...kernel, sourceBytes: new Uint8Array() })).rejects.toThrow("shipped kernel");
  for (const bounds of [{ ...kernel.bounds, maxOscillators: 1 }, { ...kernel.bounds, maxSteps: 1 }]) await expect(prepareExperimentPlan(source, { ...kernel, bounds })).rejects.toThrow("native kernel limits");
  for (const policy of [
    { ...source.policy, addressableBytes: source.policy.addressableBytes + 1n },
    { ...source.policy, memoryBytes: source.policy.memoryBytes! + 1n },
    { ...source.policy, workUnits: source.policy.workUnits! + 1n },
  ]) await expect(prepareExperimentPlan({ ...source, policy }, kernel)).rejects.toThrow("source policy ceiling");
  await expect(prepareExperimentPlan({ ...source, policy: { ...source.policy, memoryBytes: null } }, kernel, { memoryBudget: "1" })).rejects.toThrow("policy ceiling");
  for (const deadlineMs of [0, 60_001, 1.5, Infinity]) await expect(prepareExperimentPlan(source, kernel, { deadlineMs })).rejects.toThrow("bounded operational deadline");
  await expect(prepareExperimentPlan({ ...source, request: { ...source.request, dt: -1 } }, kernel)).rejects.toThrow("effective original input refused");
  expect(BuiltKernelWorker.started).toBe(started);
});

/** Rebuild only a deliberately invalid saved record while retaining all genuine native source bytes. */
async function changedRecord(saved: WorkspaceArchivePreview, change: Readonly<Record<string, unknown>>, additions: readonly WorkspaceArchiveMember[] = []): Promise<string> {
  const graph = await readLocalExperiment(saved.json, localExperimentCodecs);
  const original = graph.archive.members.find(member => member.schema === "local_run_record.v1")!;
  const document = readJson(original.content) as WorkspaceDocument;
  const changed: WorkspaceDocument = { ...document, body: { ...document.body, ...change } };
  const digest = await documentDigest(changed);
  const reference = { schema: changed.schema, sha256: digest, media_type: "application/json" };
  return writeJson({ schema: saved.schema, manifest: { ...graph.archive.manifest, body: { ...graph.archive.manifest.body, artefact_refs: [reference] } },
    members: [...graph.archive.members.filter(member => member.sha256 !== original.sha256), ...additions,
      { name: `documents/${digest}.json`, kind: "document", schema: changed.schema, sha256: digest, content: writeJson(changed) }], parameter_units: graph.archive.parameterUnits });
}

/** Save an actually executed native attempt; no test-created result supplies successful evidence. */
async function savedNativeRun() {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "independent analytic source", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const plan = await prepareExperimentPlan(source, kernel, { memoryBudget: "1048576" });
  const runId = crypto.randomUUID(), events: KernelWorkerEvent[] = [];
  const owned = createOwnedKuramotoRun({ runId, revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint,
    request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs,
    workerFactory: () => new BuiltKernelWorker(), onEvent: event => events.push(event) });
  const outcome = await owned.result;
  if (!outcome.ok) throw new Error("genuine native result required");
  const saved = await appendExperimentAttempt(plan, runId, crypto.randomUUID(), events, outcome, localExperimentCodecs);
  return { kernel, plan, outcome, saved };
}

it("reconstructs an explicit original budget from a genuine saved run and rejects changed replay bits", async () => {
  const { kernel, plan, outcome, saved } = await savedNativeRun();
  const replay = await prepareSavedReplay(saved.json, kernel, localExperimentCodecs);
  expect(replay.plan.planHash).toBe(plan.planHash);
  expect(replay.plan.requestedMemoryBytes).toBe(1048576n);
  expect(() => verifyReplayOutcome(outcome, replay.expected)).not.toThrow();
  for (const expected of [
    { ...replay.expected, thetaFinal: [...replay.expected.thetaFinal.slice(1)] },
    { ...replay.expected, thetaFinal: ["0000000000000000", replay.expected.thetaFinal[1]!] },
    { ...replay.expected, orderParameter: ["0000000000000000", ...replay.expected.orderParameter.slice(1)] },
  ]) expect(() => verifyReplayOutcome(outcome, expected)).toThrow("replay differs");
  expect(() => Reflect.apply(verifyReplayOutcome, undefined, [{ ...outcome, disposed: false }, replay.expected])).toThrow("replay differs");
  expect(() => verifyReplayOutcome({ ok: false, code: "failed", reason: "deliberate negative result", disposed: true }, replay.expected)).toThrow("replay differs");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each([{ revision_hash: "a".repeat(64) }, { plan_hash: "a".repeat(64) }, { kernel_sha256: "a".repeat(64) },
  { order_parameter: ["0000000000000000"] }, { theta_final: ["0000000000000000"] }])("rejects changed archived native output %j", async change => {
  const { kernel, saved } = await savedNativeRun();
  const graph = await readLocalExperiment(saved.json, localExperimentCodecs);
  const originalOutput = graph.archive.members.find(member => member.schema === "studio.kuramoto-output.v1")!;
  const output = await readExperimentArtifact("output", artifactContent(originalOutput));
  const altered = await makeArtifact("output", { ...output, ...change });
  const json = await changedRecord(saved, { output_refs: [{ schema: altered.schema, sha256: altered.sha256, media_type: "application/json" }] }, [altered]);
  await expect(prepareSavedReplay(json, kernel, localExperimentCodecs)).rejects.toThrow("original replay output");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("rejects missing saved output, changed effective plans and a forged source-input identity", async () => {
  const { kernel, plan, saved } = await savedNativeRun();
  await expect(prepareSavedReplay(await changedRecord(saved, { output_refs: [] }), kernel, localExperimentCodecs)).rejects.toThrow("one original source output");
  const rawPlan = await readExperimentArtifact("plan", artifactContent(plan.planMember));
  const changedPlan = await makeArtifact("plan", { ...rawPlan, deadline_ms: 4999 });
  await expect(prepareSavedReplay(await changedRecord(saved, { plan_hash: changedPlan.sha256 }, [changedPlan]), kernel, localExperimentCodecs)).rejects.toThrow("original attempt lifecycle identity");
  const changedInputPlan = await makeArtifact("plan", { ...rawPlan, input_sha256: "a".repeat(64) });
  await expect(prepareSavedReplay(await changedRecord(saved, { plan_hash: changedInputPlan.sha256 }, [changedInputPlan]), kernel, localExperimentCodecs)).rejects.toThrow("original replay plan identity");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses an unsupported trusted saved-plan namespace after the real original plan byte verifier", async () => {
  const { kernel, plan, saved } = await savedNativeRun();
  const source = await readLocalExperiment(saved.json, localExperimentCodecs);
  const alternateSchema = "external.kuramoto-plan.v1";
  const verifier = localExperimentCodecs.get(plan.planMember.schema)!;
  const codecs = new Map(localExperimentCodecs);
  codecs.set(alternateSchema, async bytes => ({ ...await verifier(bytes), schema: alternateSchema }));
  const json = writeJson({ schema: saved.schema, manifest: source.archive.manifest,
    members: source.archive.members.map(member => member.sha256 === plan.planHash ? { ...member, schema: alternateSchema } : member), parameter_units: source.archive.parameterUnits });
  await expect(prepareSavedReplay(json, kernel, codecs)).rejects.toThrow("original replay plan source format is unsupported");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("finds the actual saved result when a later root artifact reference indexes original environment metadata", async () => {
  const { kernel, plan, saved } = await savedNativeRun();
  const source = await readLocalExperiment(saved.json, localExperimentCodecs);
  const environment = source.environmentMember;
  const json = writeJson({ schema: saved.schema, manifest: { ...source.archive.manifest, body: { ...source.archive.manifest.body,
    artefact_refs: [...source.archive.manifest.body["artefact_refs"] as readonly unknown[], { schema: environment.schema, sha256: environment.sha256, media_type: "application/json" }] } },
    members: source.archive.members, parameter_units: source.archive.parameterUnits });
  const replay = await prepareSavedReplay(json, kernel, localExperimentCodecs);
  expect(replay.plan.planHash).toBe(plan.planHash);
  expect(replay.expected.thetaFinal).toHaveLength(plan.request.omega.length);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
